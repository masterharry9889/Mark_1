# MoE Transformer — 225B Parameter Mixture-of-Experts Language Model

A production-ready Mixture-of-Experts (MoE) Transformer implementation with Multi-Head Latent Attention (MLA), YaRN-scaled RoPE, and support for 4D parallelism (Tensor, Pipeline, Expert, Context).

---

## Architecture Overview

### Model Configuration (MoE-225B)

| Component | Specification |
|-----------|---------------|
| **Parameters** | ~225B total (active ~28B per token) |
| **Layers** | 96 transformer blocks |
| **Hidden Dim (d_model)** | 12,288 |
| **Attention Heads** | 96 |
| **Head Dim** | 128 |
| **MLA Query Latent Dim** | 1,536 |
| **MLA KV Latent Dim** | 512 |
| **Max Sequence Length** | 8,192 (YaRN-scaled from 2,048 base) |
| **Normalization** | RMSNorm (ε=1e-6) |
| **Weight Tying** | Input/output embeddings tied |

### MoE Configuration

| Parameter | Value |
|-----------|-------|
| **Experts per Layer** | 256 |
| **Top-K Activated** | 8 |
| **Expert FFN Dim** | 16,384 (SwiGLU) |
| **Shared Expert Dim** | 16,384 |
| **Router Jitter Noise** | 0.01 |
| **Capacity Factor** | 1.25 |
| **Expert Parallelism** | Enabled (EP=8) |
| **Aux Loss Weight** | 0.01 |
| **Z-Loss Weight** | 0.001 |

### Multi-Head Latent Attention (MLA)

```
Input (B, T, D) ──► W_Q^d ──► c_Q (B, T, q_latent)
                │              │
                └──► W_KV^d ──► c_KV (B, T, kv_latent)
                    
c_Q ──► W_QK ──► Q (B, T, H, kv_latent) ──► RoPE ──► Attention
c_KV ──► W_V^U ──► V (B, T, H, head_dim)

Scores = Q @ K^T / √kv_latent
Output = Softmax(Scores) @ V ──► W_O ──► (B, T, D)
```

**Key benefit**: Compresses KV cache from `2 × L × H × D_head` to `L × kv_latent` — **93% KV cache reduction** for 225B model.

### Transformer Block

```
┌─────────────────────────────────────────────────────┐
│ x                                                   │
├─────────────────────────────────────────────────────┤
│ RMSNorm ──► MLA ──► + (residual)                   │
├─────────────────────────────────────────────────────┤
│ RMSNorm ──► MoE Layer ──► + (residual)             │
│           │                                        │
│           ├──► TopKRouter (8 experts/token)        │
│           ├──► 256 SwiGLU Experts (d_ff=16384)     │
│           └──► Shared Expert (always active)       │
└─────────────────────────────────────────────────────┘
```

### YaRN-Scaled RoPE

Extends context from 2K → 8K via:

- **Scale factor**: 4.0
- **Beta schedule**: Fast (32) / Slow (1) ramp
- **Attention factor**: 1.0
- **mscale**: 0.1 × log(scale) + 1.0

---

## Parallelism Strategy (4D)

```
┌─────────────────────────────────────────────────────────────┐
│                    225B Model (96 layers)                   │
├─────────────────────────────────────────────────────────────┤
│  Tensor Parallel (TP=8)  ──► Split QKV, FFN across GPUs    │
│  Pipeline Parallel (PP=4) ──► 24 layers per stage          │
│  Expert Parallel (EP=8)   ──► 32 experts per GPU           │
│  Context Parallel (CP=1)  ──► Sequence split for attention │
└─────────────────────────────────────────────────────────────┘
```

**Total GPUs**: 8 × 4 × 8 = **256 GPUs** (H100/A100)

### Memory per GPU (Estimated)

| Component | Memory (BF16) |
|-----------|---------------|
| Model weights (sharded) | ~45 GB |
| Optimizer states (ZeRO-1) | ~90 GB |
| Gradients | ~45 GB |
| Activations (recomputed) | ~20 GB |
| **Total** | **~200 GB** → Requires 80GB GPUs with offloading |

---

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: Flash Attention 2 (recommended)
pip install flash-attn --no-build-isolation

# Optional: Distributed training
pip install deepspeed
```

**requirements.txt**:
```
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
datasets>=2.14.0
huggingface-hub>=0.19.0
tokenizers>=0.14.0
tiktoken>=0.5.0
pyyaml>=6.0
tqdm>=4.66.0
wandb>=0.15.0
lm-eval>=0.4.0
pytest>=7.4.0
```

---

## Training

### Single-Node (8 GPUs) — MoE-7B

```bash
torchrun --nproc_per_node=8 train.py \
  --model_config configs/model_7b.yaml \
  --data_config configs/data.yaml \
  --output_dir checkpoints/moe-7b \
  --wandb_project moe-7b
```

### Multi-Node (256 GPUs) — MoE-225B

```bash
# Node 0 (master)
torchrun --nnodes=32 --nproc_per_node=8 \
  --master_addr=10.0.0.1 --master_port=29500 \
  --node_rank=0 train.py \
  --model_config configs/model_225b.yaml \
  --data_config configs/data.yaml \
  --output_dir checkpoints/moe-225b \
  --wandb_project moe-225b \
  --tensor_parallel 8 \
  --pipeline_parallel 4 \
  --expert_parallel 8

# Node 1-31
torchrun --nnodes=32 --nproc_per_node=8 \
  --master_addr=10.0.0.1 --master_port=29500 \
  --node_rank=$RANK train.py \
  --model_config configs/model_225b.yaml \
  --data_config configs/data.yaml \
  --output_dir checkpoints/moe-225b \
  --tensor_parallel 8 \
  --pipeline_parallel 4 \
  --expert_parallel 8
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_config` | Model YAML config path | Required |
| `--data_config` | Data pipeline YAML config | Required |
| `--output_dir` | Checkpoint output directory | `checkpoints/` |
| `--resume` | Resume from checkpoint path | None |
| `--wandb_project` | W&B project name | None |
| `--tensor_parallel` | TP degree | 1 |
| `--pipeline_parallel` | PP degree | 1 |
| `--expert_parallel` | EP degree | 1 |
| `--context_parallel` | CP degree | 1 |
| `--micro_batch_size` | Per-GPU micro batch size | 32 |
| `--gradient_accumulation` | Gradient accumulation steps | Auto |
| `--precision` | bf16 / fp16 / fp32 | bf16 |

### Resume Training

```bash
torchrun --nproc_per_node=8 train.py \
  --model_config configs/model_7b.yaml \
  --data_config configs/data.yaml \
  --resume checkpoints/moe-7b/step_50000.pt
```

---

## Evaluation

### Perplexity Evaluation

```bash
# Single GPU
python eval.py \
  --checkpoint checkpoints/moe-7b/best_model.pt \
  --model_config configs/model_7b.yaml \
  --data_config configs/data.yaml \
  --eval_datasets wikitext,c4,pile \
  --eval_batch_size 16 \
  --max_eval_batches 100

# Multi-GPU (DDP)
torchrun --nproc_per_node=8 eval.py \
  --checkpoint checkpoints/moe-225b/best_model.pt \
  --model_config configs/model_225b.yaml \
  --data_config configs/data.yaml \
  --eval_datasets wikitext,c4,pile \
  --eval_batch_size 16
```

### Text Generation

```bash
python eval.py \
  --checkpoint checkpoints/moe-7b/best_model.pt \
  --model_config configs/model_7b.yaml \
  --generate \
  --prompt "The future of AI is" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_p 0.95 \
  --top_k 50
```

### Benchmarks (lm-eval-harness)

```bash
# Requires: pip install lm-eval
python eval.py \
  --checkpoint checkpoints/moe-7b/best_model.pt \
  --model_config configs/model_7b.yaml \
  --benchmarks \
  --tasks hellaswag,arc_easy,arc_challenge,mmlu,winogrande \
  --limit 1000 \
  --batch_size 16
```

### Evaluation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Model checkpoint path | Required |
| `--model_config` | Model YAML config path | Required |
| `--data_config` | Data pipeline YAML config | Required |
| `--eval_datasets` | Comma-separated dataset names | `wikitext,c4,pile` |
| `--eval_batch_size` | Batch size for evaluation | 16 |
| `--max_eval_batches` | Max batches per dataset | 100 |
| `--generate` | Enable text generation | False |
| `--prompt` | Generation prompt | None |
| `--max_new_tokens` | Tokens to generate | 200 |
| `--temperature` | Sampling temperature | 0.8 |
| `--top_p` | Nucleus sampling p | 0.95 |
| `--top_k` | Top-k sampling k | 50 |
| `--benchmarks` | Run lm-eval benchmarks | False |
| `--tasks` | Benchmark tasks (comma-separated) | None |
| `--limit` | Limit samples per task | None |
| `--output` | Results output path | `eval_results.json` |

---

## Configuration Files

### Model Configs (`configs/`)

| File | Description |
|------|-------------|
| `model_config.yaml` | All model variants (small/medium/large/xlarge/225b) |
| `model_7b.yaml` | ~7B parameter MoE config |
| `model_225b.yaml` | 225B parameter MoE config |
| `data.yaml` | Data pipeline settings |
| `training.yaml` | Training hyperparameters |

### Key Training Settings (`training.yaml`)

```yaml
training:
  global_batch_size: 4194304      # 4M tokens/step
  micro_batch_size: 32
  sequence_length: 8192
  max_steps: 200000
  warmup_steps: 2000
  peak_lr: 1.5e-4
  router_lr: 3.0e-4
  min_lr_ratio: 0.1

  optimizer:
    type: "adamw"
    betas: [0.9, 0.95]
    weight_decay: 0.1
    router_weight_decay: 0.0
    max_grad_norm: 1.0

  scheduler:
    type: "wsd"              # Warmup-Stable-Decay
    warmup_type: "linear"
    decay_type: "cosine"
    stable_fraction: 0.5

  parallelism:
    tensor_parallel: 8
    pipeline_parallel: 4
    expert_parallel: 8
    zero_stage: 1

  precision: "bf16"
  use_flash_attention: true
```

---

## Project Structure

```
Mark_1/
├── train.py                 # Main training entry point
├── eval.py                  # Main evaluation entry point
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── configs/                 # YAML configurations
│   ├── model_config.yaml
│   ├── model_7b.yaml
│   ├── model_225b.yaml
│   ├── data.yaml
│   └── training.yaml
├── scripts/                 # Alternative training scripts
│   ├── train.py
│   └── eval.py
├── src/
│   ├── model/               # Model architecture
│   │   ├── transformer.py      # MoETransformer, configs
│   │   ├── transformer_block.py
│   │   ├── attention.py        # MultiHeadLatentAttention
│   │   ├── moe_layer.py        # MOELayer
│   │   ├── expert.py           # SwiGLUExpert
│   │   ├── shared_expert.py    # SharedExpert
│   │   ├── router.py           # TopKRouter
│   │   ├── norm.py             # RMSNorm, LayerNorm
│   │   ├── embeddings.py       # LlamaYaRNScaledRotaryEmbedding
│   │   ├── tokenizer.py        # TokenizerWrapper
│   │   ├── Tokenizer/
│   │   │   └── BPETokenizer.py
│   │   └── __init__.py
│   ├── training/            # Training utilities
│   │   ├── data.py
│   │   ├── optimizer.py       # MoEAdamW (router LR 10x)
│   │   └── scheduler.py       # WarmupStableDecay
│   ├── parallel/            # 4D Parallelism
│   │   ├── tensor_parallel.py
│   │   ├── expert_parallel.py
│   │   ├── pipeline_parallel.py
│   │   └── context_parallel.py
│   ├── inference/           # Inference engine
│   │   ├── engine.py
│   │   ├── expert_cache.py
│   │   └── quantize.py
│   ├── losses/              # Loss functions
│   │   ├── cross_entropy.py
│   │   └── auxiliary.py
│   └── data/
│       └── pipeline.py      # Multi-tier data pipeline
└── tests/
```

---

## Key Features

- **Multi-Head Latent Attention (MLA)** — 93% KV cache compression
- **Mixture-of-Experts** — 256 experts, top-8 routing, shared expert
- **YaRN RoPE** — 8K context from 2K base
- **4D Parallelism** — TP + PP + EP + CP
- **Router Optimization** — 10× LR, no weight decay, jitter noise
- **Fused Kernels** — Flash Attention 2, fused CE loss
- **Expert Caching** — LRU GPU/CPU offloading for inference
- **Quantization** — AWQ / GPTQ per-expert INT4/INT8
- **Multi-Tier Data** — Foundation (80%) + Quality (15%) + Alignment (5%)

---

## Citation

If you use this implementation, please cite:

```bibtex
@software{moe_transformer_225b,
  title = {MoE Transformer: 225B Parameter Mixture-of-Experts Language Model},
  author = {Mark_1 Team},
  year = {2024},
  url = {https://github.com/your-org/moe-transformer}
}
```

---

## License

MIT License — See LICENSE file for details.