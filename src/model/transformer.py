from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
from pathlib import Path

from .norm import RMSNorm
from .attention import MultiHeadLatentAttention
from .moe_layer import MOELayer
from .embeddings import LlamaYaRNScaledRotaryEmbedding


@dataclass
class MoEConfig:
    n_experts: int = 8
    top_k: int = 2
    d_ff_expert: int = 16384
    d_ff_shared: int = 16384
    router_jitter_noise: float = 0.01
    shared_expert: bool = True
    expert_parallel: bool = False
    capacity_factor: float = 1.25
    drop_tokens: bool = False

@dataclass
class LossConfig:
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 0.001


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    d_model: int = 6144
    n_layers: int = 32
    n_heads: int = 48
    q_latent_dim: int = 1536
    kv_latent_dim: int = 512
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    norm_type: str = "rmsnorm"
    rope_base: int = 10000
    rope_scale: float = 1.0
    original_max_seq_len: int = 2048
    extrapolation_factor: float = 1.0
    attn_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    moe: MoEConfig = field(default_factory=MoEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dropout: float = 0.0
    attn_dropout: float = 0.0
    residual_dropout: float = 0.0
    tie_weights: bool = True
    init_std: float = 0.02
    init_method: str = "normal"
    rope_init: str = "default"

# Config file path
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"


def load_config_from_yaml(config_path: Optional[str] = None) -> dict:
    """Load model configuration from YAML file."""
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_model_config(model_name: str, config_path: Optional[str] = None) -> dict:
    """Get a specific model configuration from YAML file."""
    config = load_config_from_yaml(config_path)
    
    # Start with defaults
    model_config = config.get('defaults', {}).copy()
    
    # Override with model-specific config
    if model_name in config.get('models', {}):
        model_specific = config['models'][model_name]
        model_config = _deep_merge(model_config, model_specific)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(config.get('models', {}).keys())}")
    
    return model_config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class MOETransformer(nn.Module):
    """
    Mixture-of-Experts Transformer with Multi-Head Latent Attention and YaRN RoPE.
    
    Architecture:
    - Token embeddings + YaRN rotary position embeddings
    - N TransformerBlocks (RMSNorm → MLA → Residual → RMSNorm → MoE → Residual)
    - Final RMSNorm → LM Head (tied to token embeddings)
    
    Args:
        config (TransformerConfig): Model configuration
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # YaRN Rotary Position Embeddings
        # MLA uses kv_latent_dim for Q/K projections
        self.rotary_emb = LlamaYaRNScaledRotaryEmbedding(
            dim=config.kv_latent_dim,
            max_position_embeddings=config.max_seq_len,
            base=config.rope_base,
            scale=config.rope_scale,
            original_max_position_embeddings=config.original_max_seq_len,
            extrapolation_factor=config.extrapolation_factor,
            attn_factor=config.attn_factor,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final norm and output head
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if config.tie_weights:
            self.output.weight = self.tok_embeddings.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Track if rotary embeddings need device sync
        self._rope_synced = False
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
        return_router_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) token indices
            targets: (batch, seq_len) target token indices for loss computation
            return_aux_loss: Whether to return accumulated aux loss from MoE layers
            return_router_info: Whether to return router logits and indices
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: CrossEntropy loss if targets provided, else None
            aux_loss: Accumulated MoE aux loss if return_aux_loss, else None
            router_logits: Concatenated router logits from all layers if return_router_info
            router_indices: Concatenated router indices from all layers if return_router_info
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Sync rotary embeddings to correct device on first forward
        if not self._rope_synced:
            self.rotary_emb.to(device)
            self._rope_synced = True
        
        # Token embeddings
        x = self.tok_embeddings(input_ids)  # (B, T, D)
        x = self.dropout(x)
        
        # Get rotary frequencies for current sequence length
        freqs_cis = self.rotary_emb(x, seq_len=seq_len)
        
        # Forward through transformer blocks
        total_aux_loss = 0.0
        all_router_logits = []
        all_router_indices = []
        for block in self.blocks:
            x, aux_loss, router_logits, router_indices = block(x, freqs_cis)
            if return_aux_loss:
                total_aux_loss += aux_loss
            if return_router_info:
                all_router_logits.append(router_logits)
                all_router_indices.append(router_indices)
        
        # Final norm and output projection
        x = self.norm(x)
        logits = self.output(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        aux_loss = total_aux_loss if return_aux_loss else None
        
        if return_router_info:
            router_logits = torch.cat(all_router_logits, dim=0)  # (n_layers * B * T, n_experts)
            router_indices = torch.cat(all_router_indices, dim=0)  # (n_layers * B * T, top_k)
            return logits, loss, aux_loss, router_logits, router_indices
        
        return logits, loss, aux_loss, None, None
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        streamer: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: (batch, seq_len) starting token sequence
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (1.0 = no scaling)
            top_k: Top-k filtering (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            eos_token_id: Stop generation when this token is generated
            streamer: Optional callback(token_id) for streaming
        
        Returns:
            Generated token sequence (batch, seq_len + generated)
        """
        self.eval()
        device = input_ids.device
        
        # Ensure rotary embeddings are on correct device
        if not self._rope_synced:
            self.rotary_emb.to(device)
            self._rope_synced = True
        
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Crop context if exceeds max_seq_len
            if generated.size(1) > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _, _ = self.forward(generated, return_aux_loss=False)
            next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stream if callback provided
            if streamer is not None:
                streamer(next_token.item())
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params
    
    def estimate_mfu(self, batch_size: int, seq_len: int, dt: float) -> float:
        """Estimate model FLOPs utilization (MFU) in theoretical FLOPs/s."""
        # Rough estimate based on transformer FLOPs formula
        flops_per_token = 6 * self.get_num_params() + 12 * self.config.n_layers * self.config.d_model * seq_len
        flops_achieved = flops_per_token * batch_size * seq_len / dt
        flops_promised = 312e12  # H100 FP16 tensor core peak
        return flops_achieved / flops_promised


# Import TransformerBlock after definition to avoid circular import
from .transformer_block import TransformerBlock


def create_model_from_config(config_dict: dict) -> MOETransformer:
    """Factory function to create model from config dictionary."""
    # Convert nested dicts to dataclasses
    moe_cfg = MoEConfig(**config_dict.pop('moe', {}))
    loss_cfg = LossConfig(**config_dict.pop('loss', {}))
    # Handle init dict
    init_cfg = config_dict.pop('init', {})
    if init_cfg:
        config_dict['init_std'] = init_cfg.get('std', 0.02)
        config_dict['init_method'] = init_cfg.get('init_method', 'normal')
        config_dict['rope_init'] = init_cfg.get('rope_init', 'default')
    config = TransformerConfig(moe=moe_cfg, loss=loss_cfg, **config_dict)
    return MOETransformer(config)


def create_model_from_yaml(model_name: str, config_path: Optional[str] = None, **overrides) -> MOETransformer:
    """Factory function to create model from YAML config file."""
    config_dict = get_model_config(model_name, config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        if key in config_dict and isinstance(config_dict[key], dict) and isinstance(value, dict):
            config_dict[key] = _deep_merge(config_dict[key], value)
        else:
            config_dict[key] = value
    
    return create_model_from_config(config_dict)




def load_model(name: str, config_path: Optional[str] = None, **overrides) -> MOETransformer:
    """Load a predefined model configuration from YAML with optional overrides."""
    # Get base config from YAML
    config_dict = get_model_config(name, config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        if key in config_dict and isinstance(config_dict[key], dict) and isinstance(value, dict):
            config_dict[key] = _deep_merge(config_dict[key], value)
        else:
            config_dict[key] = value
    
    return create_model_from_config(config_dict)