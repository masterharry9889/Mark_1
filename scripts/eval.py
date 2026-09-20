#!/usr/bin/env python3
"""
Evaluation script for MoE Transformer.
Supports perplexity, bits per byte, and accuracy metrics on multiple datasets.
"""
import os
import sys
import argparse
import yaml
import json
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.transformer import create_model_from_yaml, TransformerConfig, MoEConfig, LossConfig
from model.Tokenizer.BPETokenizer import BPETokenizer
from training.data import MoEDataLoader
from losses.cross_entropy import FusedCrossEntropy
from losses.auxiliary import MoEAuxiliaryLoss


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    perplexity: float
    bits_per_byte: float
    accuracy: float
    ce_loss: float
    aux_loss: float
    z_loss: float


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize distributed evaluation."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_tokenizer(config: Dict[str, Any]) -> BPETokenizer:
    """Load BPE tokenizer."""
    data_config = config.get("training", {}).get("data", {})
    vocab_path = data_config.get("vocab_path", "data/tokenizer/vocab.json")
    merges_path = data_config.get("merges_path", "data/tokenizer/merges.txt")
    
    tokenizer = BPETokenizer()
    
    vocab_file = Path(vocab_path)
    merges_file = Path(merges_path)
    
    if vocab_file.exists() and merges_file.exists():
        tokenizer.load_vocab_and_merges(str(vocab_file), str(merges_file))
        print(f"Loaded tokenizer from {vocab_path}, {merges_path}")
    else:
        print(f"Warning: Tokenizer files not found at {vocab_path}, {merges_path}")
    
    return tokenizer


def load_model(checkpoint_path: str, config: Dict[str, Any], model_name: str = "moe-225b") -> torch.nn.Module:
    """Load model from checkpoint."""
    model = create_model_from_yaml(model_name, config_path=None)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda()
    model.eval()
    
    return model


def create_dataloader(config: Dict[str, Any], tokenizer: BPETokenizer):
    """Create evaluation data loader."""
    data_config = config.get("training", {}).get("data", {})
    return MoEDataLoader(
        tokenizer=tokenizer,
        max_seq_len=data_config.get("max_seq_len", 8192),
        batch_size=data_config.get("eval_batch_size", 16)
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


def evaluate_on_dataset(
    model: torch.nn.Module,
    dataloader: MoEDataLoader,
    ce_loss: FusedCrossEntropy,
    aux_loss: MoEAuxiliaryLoss,
    eval_steps: int,
    device: torch.device,
    dataset_name: str
) -> EvalMetrics:
    """Evaluate model on a single dataset."""
    total_ce = 0.0
    total_aux = 0.0
    total_z = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for step in tqdm(range(eval_steps), desc=f"Evaluating {dataset_name}", disable=(dist.get_rank() != 0)):
            # Generate dummy batch (replace with actual data loading)
            batch_size = dataloader.batch_size
            seq_len = dataloader.max_seq_len
            input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            labels = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            # Forward pass
            logits, _, _, router_logits, router_indices = model(input_ids, return_router_info=True)
            
            # Compute losses
            ce = ce_loss.forward(logits, labels)
            aux = aux_loss.load_balancing_loss(router_logits, router_indices, model.config.moe.n_experts)
            z = aux_loss.z_loss(router_logits)
    
    avg_ce = total_ce / total_tokens
    avg_aux = total_aux / total_tokens
    avg_z = total_z / total_tokens
    
    # Perplexity = exp(CE loss)
    perplexity = torch.exp(torch.tensor(avg_ce)).item()
    
    # Bits per byte = CE loss / log(2)
    bits_per_byte = avg_ce / torch.log(torch.tensor(2.0)).item()
    
    # Token accuracy (top-1)
    with torch.no_grad():
        input_ids = torch.randint(0, 50257, (16, 512), device=device)
        labels = torch.randint(0, 50257, (16, 512), device=device)
        logits, _, _ = model(input_ids, return_aux_loss=False)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()
    
    return EvalMetrics(
        perplexity=perplexity,
        bits_per_byte=bits_per_byte,
        accuracy=accuracy,
        ce_loss=avg_ce,
        aux_loss=avg_aux,
        z_loss=avg_z
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: MoEDataLoader,
    ce_loss: FusedCrossEntropy,
    aux_loss: MoEAuxiliaryLoss,
    config: Dict[str, Any],
    rank: int
) -> Dict[str, EvalMetrics]:
    """Evaluate model on all configured datasets."""
    eval_config = config.get("training", {}).get("evaluation", {})
    eval_steps = eval_config.get("eval_steps", 100)
    eval_datasets = eval_config.get("eval_datasets", ["wikitext", "c4", "pile"])
    
    results = {}
    
    for dataset_name in eval_datasets:
        # In practice, you'd load actual dataset here
        # For now, use dummy data
        metrics = evaluate_on_dataset(
            model, dataloader, ce_loss, aux_loss,
            eval_steps, next(model.parameters()).device, dataset_name
        )
        results[dataset_name] = metrics
        
        if rank == 0:
            print(f"\n{dataset_name} Results:")
            print(f"  Perplexity: {metrics.perplexity:.2f}")
            print(f"  Bits per byte: {metrics.bits_per_byte:.4f}")
            print(f"  Accuracy: {metrics.accuracy:.4f}")
            print(f"  CE Loss: {metrics.ce_loss:.4f}")
            print(f"  Aux Loss: {metrics.aux_loss:.4f}")
            print(f"  Z Loss: {metrics.z_loss:.4f}")
    
    return results


def save_results(results: Dict[str, EvalMetrics], output_path: str, rank: int):
    """Save evaluation results to JSON."""
    if rank != 0:
        return
    
    serializable = {}
    for dataset, metrics in results.items():
        serializable[dataset] = {
            "perplexity": metrics.perplexity,
            "bits_per_byte": metrics.bits_per_byte,
            "accuracy": metrics.accuracy,
            "ce_loss": metrics.ce_loss,
            "aux_loss": metrics.aux_loss,
            "z_loss": metrics.z_loss
        }
    
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE Transformer")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--model", type=str, default="moe-225b", help="Model configuration name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override eval steps from config")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="Datasets to evaluate on")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed evaluation")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config = load_config(args.config)
    
    # Override eval config if provided
    if args.eval_steps is not None:
        config.setdefault("training", {}).setdefault("evaluation", {})["eval_steps"] = args.eval_steps
    if args.datasets is not None:
        config.setdefault("training", {}).setdefault("evaluation", {})["eval_datasets"] = args.datasets
    
    # Load tokenizer
    tokenizer = load_tokenizer(config)
    
    # Load model
    model = load_model(args.checkpoint, config, args.model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # Create dataloader and loss functions
    dataloader = create_dataloader(config, tokenizer)
    ce_loss, aux_loss = create_loss_functions(config)
    
    # Evaluate
    results = evaluate_model(model.module, dataloader, ce_loss, aux_loss, config, rank)
    
    # Save results
    save_results(results, args.output, rank)
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()