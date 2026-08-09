"""Normalization layers for MoE Transformer."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Args:
        dim: Hidden dimension
        eps: Numerical stability epsilon
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + float(self.eps))
        return x / rms * self.weight


class LayerNorm(nn.Module):
    """Standard Layer Normalization.
    
    Args:
        dim: Hidden dimension
        eps: Numerical stability epsilon
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + float(self.eps))
        return x * self.weight + self.bias


def get_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """Factory function for normalization layers."""
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps)
    elif norm_type == "layernorm":
        return LayerNorm(dim, eps)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")