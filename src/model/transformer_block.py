import torch.nn as nn

from .norm import RMSNorm
from .attention import MultiHeadLatentAttention
from .moe_layer import MOELayer


class TransformerBlock(nn.Module):
    """One layer: Norm → Attention → Residual → Norm → MoE → Residual"""

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.attn = MultiHeadLatentAttention(config.d_model, config.n_heads, config.q_latent_dim, config.kv_latent_dim)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.moe_layer = MOELayer(config)

    def forward(self, x, freqs_cis):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, freqs_cis)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x, aux_loss, router_logits, router_indices = self.moe_layer(x)
        x = residual + x
        return x, aux_loss, router_logits, router_indices