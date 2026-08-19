#!/usr/bin/env python3
"""
Training script for MoE Transformer.
Supports distributed training with tensor, pipeline, expert, and context parallelism.
"""
import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.transformer import create_model_from_yaml, TransformerConfig, MoEConfig, LossConfig
from model.Tokenizer.BPETokenizer import BPETokenizer
from training.optimizer import MoEAdamW
from training.scheduler import WarmupStableDecay
from training.data import MoEDataLoader
from losses.cross_entropy import FusedCrossEntropy
from losses.auxiliary import MoEAuxiliaryLoss


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_tokenizer(config: Dict[str, Any]) -> BPETokenizer:
    """Load or train BPE tokenizer."""
    data_config = config.get("training", {}).get("data", {})
    vocab_path = data_config.get("vocab_path", "src/model/Tokenizer/vocab.json")
    merges_path = data_config.get("merges_path", "src/model/Tokenizer/bpe_merges.txt")
    
    tokenizer = BPETokenizer()
    
    vocab_file = Path(vocab_path)
    merges_file = Path(merges_path)
    
    if vocab_file.exists() and merges_file.exists():
        tokenizer.load_vocab_and_merges(str(vocab_file), str(merges_file))
        print(f"Loaded tokenizer from {vocab_path}, {merges_path}")
    else:
        print(f"Tokenizer files not found at {vocab_path}, {merges_path}")
        print("Training new tokenizer...")
        train_data_path = data_config.get("train_data_path", "data/train")
        train_files = list(Path(train_data_path).glob("*.txt"))
        if train_files:
            text = ""
            for f in train_files[:10]:
                text += f.read_text(encoding="utf-8", errors="ignore")[:100000]
            tokenizer.train(text, vocab_size=50257)
            vocab_file.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save_vocab_and_merges(str(vocab_file), str(merges_file))
            print(f"Saved tokenizer to {vocab_path}, {merges_path}")
        else:
            print("Warning: No training data found, using untrained tokenizer")
    
    return tokenizer


def create_model(config: Dict[str, Any], model_name: str = "moe-225b") -> torch.nn.Module:
    """Create model from config."""
    model = create_model_from_yaml(model_name, config_path=None)
    return model


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer with separate LR groups."""
    opt_config = config.get("training", {}).get("optimizer", {})
    return MoEAdamW(
        model,
        lr=opt_config.get("peak_lr", 1.5e-4),
        wd=opt_config.get("weight_decay", 0.1)
    )


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """Create learning rate scheduler."""
    sched_config = config.get("training", {}).get("scheduler", {})
    train_config = config.get("training", {})
    return WarmupStableDecay(
        optimizer,
        warmup_steps=train_config.get("warmup_steps", 2000),
        stable_steps=int(train_config.get("lr_decay_steps", 180000) * sched_config.get("stable_fraction", 0.5)),
        decay_steps=train_config.get("lr_decay_steps", 180000),
        max_lr=train_config.get("peak_lr", 1.5e-4)
    )


def create_loss_functions(config: Dict[str, Any]) -> tuple:
    """Create loss functions."""
    loss_config = config.get("defaults", {}).get("loss", {})
    ce_loss = FusedCrossEntropy(label_smoothing=0.0)
    aux_loss = MoEAuxiliaryLoss(
        alpha=loss_config.get("aux_loss_weight", 0.01),
        z_weight=loss_config.get("z_loss_weight", 0.001)
    )
    return ce_loss, aux_loss


def create_dataloader(config: Dict[str, Any], tokenizer: BPETokenizer):
    """Create data loader."""
    data_config = config.get("training", {}).get("data", {})
    return MoEDataLoader(
        tokenizer=tokenizer,
        max_seq_len=data_config.get("max_seq_len", 8192),
        batch_size=data_config.get("micro_batch_size", 32)
    )


def save_checkpoint(model, optimizer, scheduler, step: int, config: Dict[str, Any], rank: int):
    """Save training checkpoint."""
    if rank != 0:
        return
    
    checkpoint_config = config.get("training", {}).get("checkpoint", {})
    save_path = Path(checkpoint_config.get("path", "checkpoints/"))
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if checkpoint_config.get("save_optimizer", True) else None,
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "config": config,
    }
    
    torch.save(checkpoint, save_path / f"checkpoint_step_{step}.pt")
    
    keep_last = checkpoint_config.get("keep_last_n", 5)
    checkpoints = sorted(save_path.glob("checkpoint_step_*.pt"), key=lambda p: p.stat().st_mtime)
    for old_ckpt in checkpoints[:-keep_last]:
        old_ckpt.unlink()


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str, config: Dict[str, Any]) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if config.get("training", {}).get("load_optimizer", True) and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if config.get("training", {}).get("load_scheduler", True) and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint.get("step", 0)


def train_step(model, batch, ce_loss, aux_loss, config: Dict[str, Any], device: torch.device) -> Dict[str, float]:
    """Single training step."""
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    logits, _, _, router_logits, router_indices = model(input_ids, return_router_info=True)
    
    ce = ce_loss.forward(logits, labels)
    aux = aux_loss.load_balancing_loss(router_logits, router_indices, model.config.moe.n_experts)
    z = aux_loss.z_loss(router_logits)
    
    total_loss = ce + aux + z
    
    return {
        "loss": total_loss.item(),
        "ce_loss": ce.item(),
        "aux_loss": aux.item(),
        "z_loss": z.item(),
    }


def evaluate(model, dataloader, ce_loss, aux_loss, config: Dict[str, Any], rank: int, device: torch.device):
    """Run evaluation."""
    eval_config = config.get("training", {}).get("evaluation", {})
    eval_steps = eval_config.get("eval_steps", 100)
    
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_z = 0.0
    
    with torch.no_grad():
        for i in range(eval_steps):
            batch = dataloader.create_batch(torch.randint(0, 50257, (10000, 8192)))
            losses = train_step(model, batch, ce_loss, aux_loss, config, device)
            total_loss += losses["loss"]
            total_ce += losses["ce_loss"]
            total_aux += losses["aux_loss"]
            total_z += losses["z_loss"]
    
    if rank == 0:
        avg_loss = total_loss / eval_steps
        print(f"Evaluation: loss={avg_loss:.4f}, ce={total_ce/eval_steps:.4f}, "
              f"aux={total_aux/eval_steps:.4f}, z={total_z/eval_steps:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train MoE Transformer")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--model", type=str, default="moe-225b", help="Model configuration name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    torch.manual_seed(42 + rank)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    model = create_model(config, args.model).to(device)
    
    if world_size > 1 and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    ce_loss, aux_loss = create_loss_functions(config)
    dataloader = create_dataloader(config, tokenizer)
    
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume, config)
    
    train_config = config.get("training", {})
    max_steps = train_config.get("max_steps", 200000)
    log_interval = train_config.get("logging", {}).get("log_interval", 10)
    eval_interval = train_config.get("evaluation", {}).get("eval_interval", 1000)
    save_interval = train_config.get("checkpoint", {}).get("save_interval", 500)
    
    model.train()
    step = start_step
    
    while step < max_steps:
        # Use smaller dummy data for testing (dataloader batch_size * max_seq_len)
        dummy_seq_len = min(dataloader.max_seq_len, 512)  # Limit for CPU testing
        dummy_batch_size = min(dataloader.batch_size, 4)
        dummy_tokens = torch.randint(0, 50257, (dummy_batch_size * 10, dummy_seq_len))
        batch = dataloader.create_batch(dummy_tokens)
        
        optimizer.zero_grad()
        losses = train_step(model.module if hasattr(model, "module") else model, batch, ce_loss, aux_loss, config, device)
        
        total_loss = losses["loss"]
        total_loss.backward()
        
        max_grad_norm = train_config.get("optimizer", {}).get("max_grad_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step(step)
        
        if step % log_interval == 0 and rank == 0:
            print(f"Step {step}: loss={losses['loss']:.4f}, ce={losses['ce_loss']:.4f}, "
                  f"aux={losses['aux_loss']:.4f}, z={losses['z_loss']:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
        
        if step % eval_interval == 0 and step > 0:
            evaluate(model.module if hasattr(model, "module") else model, dataloader, ce_loss, aux_loss, config, rank, device)
            model.train()
        
        if step % save_interval == 0 and step > 0:
            save_checkpoint(model.module if hasattr(model, "module") else model, optimizer, scheduler, step, config, rank)
        
        step += 1
    
    save_checkpoint(model.module if hasattr(model, "module") else model, optimizer, scheduler, step, config, rank)
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()