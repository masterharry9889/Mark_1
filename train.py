#!/usr/bin/env python3
"""
LLM Pretraining Script - Ready to run.

Usage:
    # Single GPU
    python train.py --model_config configs/model_7b.yaml --data_config configs/data.yaml

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=8 train.py --model_config configs/model_7b.yaml --data_config configs/data.yaml

    # Resume from checkpoint
    python train.py --resume checkpoints/step_50000.pt
"""

import argparse
import os
import sys
import time
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import GradScaler, autocast

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.pipeline import create_pipeline
from model import MoEModel, MoEConfig
from model.tokenizer import create_tokenizer


class StreamingDataset(IterableDataset):
    """Wrapper to make MoEDataPipeline compatible with DataLoader."""

    def __init__(self, pipeline, batch_size: int, max_seq_len: int):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def __iter__(self):
        while True:
            batch = self.pipeline.sample_batch(self.batch_size)
            if batch.numel() > 0:
                # Create labels (shifted by 1 for causal LM)
                input_ids = batch[:, :-1]
                labels = batch[:, 1:]
                yield input_ids, labels


def setup_distributed():
    """Initialize distributed training if needed."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
    """Cosine decay with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, path, config, is_best=False):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "config": config,
    }
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pt")
        torch.save(state, best_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    """Load training checkpoint."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state.get("step", 0), state.get("epoch", 0)


def evaluate(model, val_loader, device, max_batches=50, use_amp=True):
    """Run evaluation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-1,
                    reduction="sum"
                )

            total_loss += loss.item()
            total_tokens += labels.numel()

            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = labels != -1
            total_correct += (preds[mask] == labels[mask]).sum().item()

    model.train()
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20))  # Cap for stability
    acc = total_correct / max(1, total_tokens)
    return avg_loss, ppl, acc


def train(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    # Load configs
    import yaml
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    # Tokenizer
    tokenizer = create_tokenizer(data_config.get("tokenizer", "gpt2"))
    vocab_size = tokenizer.vocab_size

    # Model
    moe_config = MoEConfig(
        vocab_size=vocab_size,
        **model_config
    )
    model = MoEModel(moe_config).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Data
    pipeline = create_pipeline(
        tokenizer,
        max_seq_len=data_config.get("max_seq_len", 1024),
        seed=data_config.get("seed", 42),
        buffer_size=data_config.get("buffer_size", 10000),
    )

    train_dataset = StreamingDataset(
        pipeline,
        batch_size=args.batch_size,
        max_seq_len=data_config.get("max_seq_len", 1024),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # IterableDataset handles batching
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Validation (small subset)
    val_pipeline = create_pipeline(
        tokenizer,
        max_seq_len=data_config.get("max_seq_len", 1024),
        seed=data_config.get("seed", 42) + 1,
        buffer_size=1000,
    )
    val_dataset = StreamingDataset(val_pipeline, args.batch_size, data_config.get("max_seq_len", 1024))
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8,
        fused=True,
    )

    # Scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    # AMP
    scaler = GradScaler(enabled=args.use_amp)

    # Resume
    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        if is_main:
            print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    step = start_step
    epoch = 0
    tokens_seen = 0
    log_interval = args.log_interval
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    best_val_loss = float("inf")

    if is_main:
        print(f"Starting training: max_steps={args.max_steps}, batch_size={args.batch_size}")
        print(f"Model params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
        print(f"Pipeline sources: {pipeline.get_source_stats()}")

    start_time = time.time()
    last_log_time = start_time

    for input_ids, labels in train_loader:
        if step >= args.max_steps:
            break

        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=args.use_amp):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )

        scaler.scale(loss).backward()

        # Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        tokens_seen += input_ids.numel()
        step += 1

        # Logging
        if is_main and step % log_interval == 0:
            now = time.time()
            elapsed = now - last_log_time
            tokens_per_sec = (input_ids.numel() * log_interval) / elapsed
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"{tokens_per_sec/1e6:.2f}M tok/s | {tokens_seen/1e9:.2f}B tokens")
            last_log_time = now

        # Evaluation
        if is_main and step % eval_interval == 0:
            val_loss, val_ppl, val_acc = evaluate(model, val_loader, device, args.eval_batches, args.use_amp)
            print(f"  eval: loss {val_loss:.4f} | ppl {val_ppl:.2f} | acc {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, scaler, step, epoch,
                              os.path.join(args.output_dir, f"step_{step}.pt"),
                              {**model_config, **data_config}, is_best=True)

        # Checkpoint
        if is_main and step % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, step, epoch,
                          os.path.join(args.output_dir, f"step_{step}.pt"),
                          {**model_config, **data_config})

    # Final save
    if is_main:
        save_checkpoint(model, optimizer, scheduler, scaler, step, epoch,
                      os.path.join(args.output_dir, f"step_{step}.pt"),
                      {**model_config, **data_config})
        print(f"Training complete. Final step: {step}")

    cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Pretraining")
    parser.add_argument("--model_config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--data_config", type=str, required=True, help="Data config YAML")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per GPU")
    parser.add_argument("--max_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as ratio of peak")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision")

    # Logging/eval
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Eval every N steps")
    parser.add_argument("--eval_batches", type=int, default=50, help="Batches per eval")
    parser.add_argument("--save_interval", type=int, default=5000, help="Save every N steps")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)