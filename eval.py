#!/usr/bin/env python3
"""
LLM Evaluation Script - Ready to run.

Usage:
    # Evaluate on validation sets
    python eval.py --checkpoint checkpoints/best_model.pt --model_config configs/model_7b.yaml

    # Generate samples
    python eval.py --checkpoint checkpoints/best_model.pt --model_config configs/model_7b.yaml --generate

    # Run benchmarks (requires lm-eval-harness)
    python eval.py --checkpoint checkpoints/best_model.pt --model_config configs/model_7b.yaml --benchmarks
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.pipeline import create_pipeline
from model import MoEModel, MoEConfig
from model.tokenizer import create_tokenizer


class StreamingDataset:
    """Simple iterable for evaluation."""

    def __init__(self, pipeline, batch_size: int, max_seq_len: int, max_batches: int):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_batches = max_batches
        self.count = 0

    def __iter__(self):
        while self.count < self.max_batches:
            batch = self.pipeline.sample_batch(self.batch_size)
            if batch.numel() > 0:
                input_ids = batch[:, :-1]
                labels = batch[:, 1:]
                self.count += 1
                yield input_ids, labels


@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, datasets, batch_size=8, max_batches=100, use_amp=True):
    """Evaluate perplexity on multiple datasets."""
    model.eval()
    results = {}

    for name, pipeline in datasets.items():
        print(f"\nEvaluating {name}...")
        dataset = StreamingDataset(pipeline, batch_size, 1024, max_batches)
        loader = DataLoader(dataset, batch_size=None, num_workers=0)

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        batch_count = 0

        for input_ids, labels in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-1,
                    reduction="sum"
                )

            total_loss += loss.item()
            total_tokens += labels.numel()

            preds = logits.argmax(dim=-1)
            mask = labels != -1
            total_correct += (preds[mask] == labels[mask]).sum().item()
            batch_count += 1

        avg_loss = total_loss / max(1, total_tokens)
        ppl = torch.exp(torch.tensor(min(avg_loss, 20))).item()
        acc = total_correct / max(1, total_tokens)

        results[name] = {
            "loss": avg_loss,
            "perplexity": ppl,
            "accuracy": acc,
            "tokens": total_tokens,
            "batches": batch_count,
        }
        print(f"  {name}: loss={avg_loss:.4f}, ppl={ppl:.2f}, acc={acc:.4f}, tokens={total_tokens:,}")

    model.train()
    return results


@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
    """Generate text samples."""
    model.eval()
    results = []

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(generated)
                logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumprobs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        text = tokenizer.decode(generated[0].tolist())
        results.append({"prompt": prompt, "completion": text[len(prompt):], "full": text})
        print(f"\nPrompt: {prompt}")
        print(f"Completion: {text[len(prompt):]}")

    model.train()
    return results


def run_benchmarks(model, tokenizer, device, tasks=None, limit=None):
    """Run lm-eval-harness benchmarks (requires: pip install lm-eval)."""
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("lm-eval-harness not installed. Run: pip install lm-eval")
        return None

    print("Running benchmarks...")
    # Wrap model for lm-eval
    class MoELM(HFLM):
        def __init__(self, model, tokenizer, device):
            self._model = model
            self.tokenizer = tokenizer
            self._device = device
            self._batch_size = 1

        @property
        def model(self):
            return self._model

        @property
        def device(self):
            return self._device

    lm = MoELM(model, tokenizer, device)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks or ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"],
        limit=limit,
        batch_size="auto",
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model_config", type=str, required=True, help="Model config YAML")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml", help="Data config YAML")

    # Evaluation options
    parser.add_argument("--ppl", action="store_true", default=True, help="Run perplexity evaluation")
    parser.add_argument("--max_batches", type=int, default=100, help="Max batches per dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision")

    # Generation
    parser.add_argument("--generate", action="store_true", help="Generate samples")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, help="Prompts for generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")

    # Benchmarks
    parser.add_argument("--benchmarks", action="store_true", help="Run lm-eval benchmarks")
    parser.add_argument("--benchmark_tasks", type=str, nargs="+", default=None, help="Benchmark tasks")
    parser.add_argument("--benchmark_limit", type=int, default=None, help="Limit samples per task")

    # Output
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load configs
    import yaml
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    # Tokenizer
    tokenizer = create_tokenizer(data_config.get("tokenizer", "gpt2"))

    # Model
    moe_config = MoEConfig(
        vocab_size=tokenizer.vocab_size,
        **model_config
    )
    model = MoEModel(moe_config).to(device)

    # Load checkpoint
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    print(f"Loaded checkpoint from step {state.get('step', 'unknown')}")

    # Prepare validation pipelines
    val_pipelines = {}
    for src_name in ["fineweb", "redpajama_v2", "dclm", "dolma", "starcoder", "the_stack_v2", "wikipedia", "arxiv", "openassistant", "dolly"]:
        try:
            val_pipelines[src_name] = create_pipeline(
                tokenizer,
                max_seq_len=1024,
                seed=42 + hash(src_name) % 1000,
                buffer_size=1000,
            )
        except Exception as e:
            print(f"Skipping {src_name}: {e}")

    all_results = {}

    # Perplexity evaluation
    if args.ppl:
        print("\n" + "="*50)
        print("PERPLEXITY EVALUATION")
        print("="*50)
        ppl_results = evaluate_ppl(model, tokenizer, device, val_pipelines,
                                   batch_size=args.batch_size,
                                   max_batches=args.max_batches,
                                   use_amp=args.use_amp)
        all_results["perplexity"] = ppl_results

    # Generation
    if args.generate:
        print("\n" + "="*50)
        print("GENERATION")
        print("="*50)
        default_prompts = [
            "The future of artificial intelligence is",
            "In a world where machines can think,",
            "def fibonacci(n):",
            "The most important scientific discovery of the 21st century",
        ]
        prompts = args.prompts or default_prompts
        gen_results = generate_samples(model, tokenizer, device, prompts,
                                       max_new_tokens=args.max_new_tokens,
                                       temperature=args.temperature,
                                       top_k=args.top_k,
                                       top_p=args.top_p)
        all_results["generation"] = gen_results

    # Benchmarks
    if args.benchmarks:
        print("\n" + "="*50)
        print("BENCHMARKS")
        print("="*50)
        bench_results = run_benchmarks(model, tokenizer, device,
                                       tasks=args.benchmark_tasks,
                                       limit=args.benchmark_limit)
        if bench_results:
            all_results["benchmarks"] = bench_results["results"]

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    if "perplexity" in all_results:
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        for name, metrics in all_results["perplexity"].items():
            print(f"{name:20s}  ppl={metrics['perplexity']:6.2f}  loss={metrics['loss']:.4f}  acc={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()