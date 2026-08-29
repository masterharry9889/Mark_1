#!/usr/bin/env python3
"""
Streaming training script for MoE Transformer.

Uses HF datasets streaming mode to avoid downloading entire datasets.
Implements rolling checkpoint (overwritten every step) + periodic permanent checkpoints.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.transformer import create_model_from_yaml
from model.tokenizer import create_tokenizer
from training.data import MoEDataLoader
from losses.auxiliary import MoEAuxiliaryLoss
from data.pipeline import create_pipeline


@dataclass
class TrainingState:
    """Mutable training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    total_tokens: int = 0
    best_loss: float = float('inf')


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_idx = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(gpu_idx)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(config: Dict[str, Any], model_name: str = "moe-225b") -> torch.nn.Module:
    """Create model from config."""
    model = create_model_from_yaml(model_name, config_path=None)
    return model


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer with separate LR groups for router vs rest."""
    train_config = config.get("training", {})
    opt_config = train_config.get("optimizer", {})
    
    # Separate router parameters for higher LR
    router_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "router" in name or "gate" in name:
            router_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {"params": other_params, "lr": float(train_config.get("peak_lr", 1.5e-4))},
        {"params": router_params, "lr": float(train_config.get("router_lr", 3.0e-4)), "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(
        param_groups,
        betas=tuple(float(b) for b in opt_config.get("betas", [0.9, 0.95])),
        eps=float(opt_config.get("eps", 1e-8)),
        weight_decay=float(opt_config.get("weight_decay", 0.1)),
    )


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """Create Warmup-Stable-Decay (WSD) scheduler."""
    train_config = config.get("training", {})
    sched_config = train_config.get("scheduler", {})
    
    max_steps = int(train_config.get("max_steps", 200000))
    warmup_steps = int(train_config.get("warmup_steps", 2000))
    lr_decay_steps = int(train_config.get("lr_decay_steps", 180000))
    min_lr_ratio = float(train_config.get("min_lr_ratio", 0.1))
    peak_lr = float(train_config.get("peak_lr", 1.5e-4))
    min_lr = float(train_config.get("min_lr", peak_lr * min_lr_ratio))
    stable_fraction = float(sched_config.get("stable_fraction", 0.5))
    
    stable_steps = int(max_steps * stable_fraction)
    decay_start = warmup_steps + stable_steps
    
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        elif step < decay_start:
            return 1.0
        elif step < max_steps:
            decay_progress = (step - decay_start) / max(1, max_steps - decay_start)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(decay_progress * 3.14159))).item()
        else:
            return min_lr_ratio
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_loss_functions(config: Dict[str, Any]) -> tuple:
    """Create loss functions."""
    loss_config = config.get("loss", {})
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    aux_loss = MoEAuxiliaryLoss(
        alpha=float(loss_config.get("aux_loss_weight", 0.01)),
        z_weight=float(loss_config.get("z_loss_weight", 0.001)),
    )
    return ce_loss, aux_loss


def create_streaming_dataloader(config: Dict[str, Any], tokenizer, rank: int, world_size: int, test_mode: bool = False):
    """Create streaming data loader using HF datasets with dummy fallback."""
    data_config = config.get("training", {}).get("data", {})
    max_seq_len = data_config.get("max_seq_len", 8192)
    micro_batch_size = data_config.get("micro_batch_size", 32)
    shuffle_buffer = data_config.get("shuffle_buffer", 10000)
    seed = config.get("seed", 42) + rank
    
    if test_mode:
        # Skip HF datasets entirely, use dummy data
        import torch
        vocab_size = getattr(tokenizer, 'vocab_size', 50257)
        def dummy_iterator():
            while True:
                input_ids = torch.randint(0, vocab_size, (micro_batch_size, max_seq_len - 1))
                labels = torch.randint(0, vocab_size, (micro_batch_size, max_seq_len - 1))
                yield input_ids, labels
        return dummy_iterator()
    
    pipeline = create_pipeline(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        seed=seed,
        buffer_size=shuffle_buffer,
    )
    
    def safe_iterator():
        """Iterator that falls back to dummy data on streaming failure."""
        try:
            yield from pipeline.__iter__(batch_size=micro_batch_size)
        except Exception as e:
            print(f"[DataLoader] Streaming failed: {e}. Using dummy data.")
            import torch
            vocab_size = getattr(tokenizer, 'vocab_size', 50257)
            while True:
                input_ids = torch.randint(0, vocab_size, (micro_batch_size, max_seq_len - 1))
                labels = torch.randint(0, vocab_size, (micro_batch_size, max_seq_len - 1))
                yield input_ids, labels
    
    return safe_iterator()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: TrainingState,
    config: Dict[str, Any],
    rank: int,
    is_permanent: bool = False,
):
    """Save training checkpoint.
    
    Rolling checkpoint: overwritten every step (latest.pt)
    Permanent checkpoint: saved every N steps (checkpoint_step_X.pt)
    """
    if rank != 0:
        return
    
    checkpoint_config = config.get("training", {}).get("checkpoint", {})
    save_path = Path(checkpoint_config.get("path", "checkpoints/"))
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Handle DDP wrapped model
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    checkpoint = {
        "step": state.step,
        "epoch": state.epoch,
        "total_tokens": state.total_tokens,
        "best_loss": state.best_loss,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict() if checkpoint_config.get("save_optimizer", True) else None,
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "config": config,
    }
    
    if is_permanent:
        # Permanent checkpoint with step number
        ckpt_path = save_path / f"checkpoint_step_{state.step}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"[Checkpoint] Saved permanent checkpoint: {ckpt_path}")
        
        # Clean old permanent checkpoints (keep last N)
        keep_last = checkpoint_config.get("keep_last_n", 5)
        checkpoints = sorted(save_path.glob("checkpoint_step_*.pt"), key=lambda p: p.stat().st_mtime)
        for old_ckpt in checkpoints[:-keep_last]:
            old_ckpt.unlink()
            print(f"[Checkpoint] Removed old checkpoint: {old_ckpt}")
    else:
        # Rolling checkpoint - always overwrite
        ckpt_path = save_path / "latest.pt"
        torch.save(checkpoint, ckpt_path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_path: str,
    config: Dict[str, Any],
) -> TrainingState:
    """Load training checkpoint and return training state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model_state = model.module if hasattr(model, "module") else model
    model_state.load_state_dict(checkpoint["model_state_dict"])
    
    if checkpoint.get("optimizer_state_dict") and optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if checkpoint.get("scheduler_state_dict") and scheduler and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    state = TrainingState(
        step=checkpoint.get("step", 0),
        epoch=checkpoint.get("epoch", 0),
        total_tokens=checkpoint.get("total_tokens", 0),
        best_loss=checkpoint.get("best_loss", float('inf')),
    )
    
    print(f"[Checkpoint] Resumed from step {state.step}, epoch {state.epoch}")
    return state


def train_step(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    ce_loss: torch.nn.Module,
    aux_loss: MoEAuxiliaryLoss,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """Single training step."""
    input_ids, labels = batch
    input_ids = input_ids.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    
    # Forward pass
    logits, _, _, router_logits, router_indices = model(input_ids, return_router_info=True)
    
    # Compute losses
    ce = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
    aux = aux_loss.load_balancing_loss(router_logits, router_indices, model.config.moe.n_experts)
    z = aux_loss.z_loss(router_logits)
    
    total_loss = ce + aux + z
    total_loss.backward()
    
    return {
        "loss": total_loss.item(),
        "ce_loss": ce.item(),
        "aux_loss": aux.item(),
        "z_loss": z.item(),
    }


def evaluate(
    model: torch.nn.Module,
    config: Dict[str, Any],
    tokenizer,
    ce_loss: torch.nn.Module,
    aux_loss: MoEAuxiliaryLoss,
    rank: int,
    device: torch.device,
    eval_steps: int = 100,
) -> Dict[str, float]:
    """Run evaluation on streaming data."""
    # Create fresh eval dataloader
    data_config = config.get("training", {}).get("data", {})
    max_seq_len = data_config.get("max_seq_len", 8192)
    micro_batch_size = data_config.get("micro_batch_size", 32)
    shuffle_buffer = data_config.get("shuffle_buffer", 10000)
    seed = config.get("seed", 42) + rank + 1000  # Different seed for eval
    
    pipeline = create_pipeline(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
    )
    eval_iter = iter(pipeline(batch_size=micro_batch_size))
    
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_z = 0.0
    
    with torch.no_grad():
        for i in range(eval_steps):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
            
            input_ids, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits, _, _, router_logits, router_indices = model(input_ids, return_router_info=True)
            
            ce = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
            aux = aux_loss.load_balancing_loss(router_logits, router_indices, model.config.moe.n_experts)
            z = aux_loss.z_loss(router_logits)
            
            total_loss += (ce + aux + z).item()
            total_ce += ce.item()
            total_aux += aux.item()
            total_z += z.item()
    
    model.train()
    n = max(1, eval_steps)
    
    if rank == 0:
        print(f"[Eval] loss={total_loss/n:.4f}, ce={total_ce/n:.4f}, aux={total_aux/n:.4f}, z={total_z/n:.4f}")
    
    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "aux_loss": total_aux / n,
        "z_loss": total_z / n,
    }
def main():
    parser = argparse.ArgumentParser(description="Train MoE Transformer with Streaming Data")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--model", type=str, default="moe-225b", help="Model configuration name")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--rolling-interval", type=int, default=1, help="Save rolling checkpoint every N steps")
    parser.add_argument("--permanent-interval", type=int, default=5000, help="Save permanent checkpoint every N steps")
    parser.add_argument("--test-mode", action="store_true", help="Use dummy data for quick testing")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    torch.manual_seed(42 + rank)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    if device.type == "cuda":
        gpu_idx = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_idx)
    
    # Load config
    config = load_config(args.config)
    
    # Tokenizer
    tokenizer_config = config.get("tokenizer", "gpt2")
    tokenizer = create_tokenizer(tokenizer_type=tokenizer_config)
    
    # Model
    model = create_model(config, args.model).to(device)
    
    # DDP wrapping
    if world_size > 1 and device.type == "cuda":
        gpu_idx = local_rank % torch.cuda.device_count()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu_idx],
            output_device=gpu_idx,
            find_unused_parameters=False,  # Set True if MoE causes unused params
        )
    
    # Optimizer & scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    ce_loss, aux_loss = create_loss_functions(config)
    
    # Streaming dataloader
    dataloader = create_streaming_dataloader(config, tokenizer, rank, world_size, test_mode=args.test_mode)
    data_iter = iter(dataloader)
    
    # Training state
    state = TrainingState()
    
    # Resume from checkpoint
    if args.resume:
        state = load_checkpoint(model, optimizer, scheduler, args.resume, config)
        # Fast-forward dataloader to correct position (approximate)
        for _ in range(state.step % 1000):  # Skip some batches
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
    
    # Training config
    train_config = config.get("training", {})
    max_steps = train_config.get("max_steps", 200000)
    log_interval = train_config.get("logging", {}).get("log_interval", 10)
    eval_interval = train_config.get("evaluation", {}).get("eval_interval", 1000)
    eval_steps = train_config.get("evaluation", {}).get("eval_steps", 100)
    rolling_interval = args.rolling_interval
    permanent_interval = args.permanent_interval
    max_grad_norm = train_config.get("optimizer", {}).get("max_grad_norm", 1.0)
    
    # Resume rolling checkpoint if exists
    checkpoint_config = train_config.get("checkpoint", {})
    save_path = Path(checkpoint_config.get("path", "checkpoints/"))
    rolling_ckpt = save_path / "latest.pt"
    if rolling_ckpt.exists() and not args.resume:
        print(f"[Checkpoint] Found rolling checkpoint, resuming...")
        state = load_checkpoint(model, optimizer, scheduler, str(rolling_ckpt), config)
    
    print(f"[Training] Starting from step {state.step}, max_steps={max_steps}")
    print(f"[Training] Rolling checkpoint every {rolling_interval} step(s)")
    print(f"[Training] Permanent checkpoint every {permanent_interval} step(s)")
    
    model.train()
    step = state.step
    
    try:
        while step < max_steps:
            # Get next batch from streaming dataset
            try:
                batch = next(data_iter)
            except StopIteration:
                # Dataset exhausted (shouldn't happen with streaming), reinitialize
                data_iter = iter(dataloader)
                batch = next(data_iter)
                state.epoch += 1
            
            # Training step
            optimizer.zero_grad()
            
            losses = train_step(
                model.module if hasattr(model, "module") else model,
                batch, ce_loss, aux_loss, config, device
            )
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step(step)
            
            # Update state
            step += 1
            state.step = step
            state.total_tokens += batch[0].numel() * world_size
            
            # Logging
            if step % log_interval == 0 and rank == 0:
                print(f"Step {step}: loss={losses['loss']:.4f}, ce={losses['ce_loss']:.4f}, "
                      f"aux={losses['aux_loss']:.4f}, z={losses['z_loss']:.4f}, "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}, tokens={state.total_tokens/1e9:.2f}B")
            
            # Evaluation
            if step % eval_interval == 0 and step > 0:
                eval_losses = evaluate(
                    model.module if hasattr(model, "module") else model,
                    config, tokenizer, ce_loss, aux_loss, rank, device, eval_steps
                )
                # Update best loss
                if eval_losses["loss"] < state.best_loss:
                    state.best_loss = eval_losses["loss"]
                    if rank == 0:
                        print(f"[Checkpoint] New best loss: {state.best_loss:.4f}")
            
            # Rolling checkpoint (every step by default)
            if step % rolling_interval == 0:
                save_checkpoint(model, optimizer, scheduler, state, config, rank, is_permanent=False)
            
            # Permanent checkpoint
            if step % permanent_interval == 0 and step > 0:
                save_checkpoint(model, optimizer, scheduler, state, config, rank, is_permanent=True)
            
            # Check for permanent checkpoint at end
            if step >= max_steps:
                save_checkpoint(model, optimizer, scheduler, state, config, rank, is_permanent=True)
                break
    
    except KeyboardInterrupt:
        print(f"\n[Training] Interrupted at step {step}")
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, state, config, rank, is_permanent=True)
            print("[Training] Saved interrupt checkpoint")
    
    except Exception as e:
        print(f"[Training] Error at step {step}: {e}")
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, state, config, rank, is_permanent=True)
        raise
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    
    print(f"[Training] Completed at step {step}")


if __name__ == "__main__":
    main()