import torch
import torch.nn as nn

from .router import TopKRouter
from .expert import SwiGLUExpert
from .shared_expert import SharedExpert


class MOELayer(nn.Module):
    """Router + Expert Ensemble. Replaces the FFN in a transformer block."""

    def __init__(self, config):
        super().__init__()
        self.router = TopKRouter(
            config.d_model, config.moe.n_experts,
            config.moe.top_k, config.moe.router_jitter_noise
        )
        self.experts = nn.ModuleList([
            SwiGLUExpert(config.d_model, config.moe.d_ff_expert)
            for _ in range(config.moe.n_experts)
        ])
        self.shared_expert = (
            SharedExpert(config.d_model, config.moe.d_ff_shared)
            if config.moe.shared_expert else None
        )
        self.aux_loss_weight = config.loss.aux_loss_weight
        self.z_loss_weight = config.loss.z_loss_weight
        self.n_experts = config.moe.n_experts

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        weights, indices, logits = self.router(x_flat)
        output = self._sparse_expert_forward(x_flat, weights, indices)

        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        aux_loss = (
            self.aux_loss_weight * self.router.compute_aux_loss(logits, indices)
            + self.z_loss_weight * self.router.compute_z_loss(logits)
        )

        return output.view(B, T, D), aux_loss, logits, indices

    def _sparse_expert_forward(self, x, weights, indices):
        """Dispatch tokens to selected experts, combine results."""
        output = torch.zeros_like(x)
        for expert_idx in range(self.n_experts):
            mask = (indices == expert_idx).any(dim=-1)
            if mask.any():
                expert_out = self.experts[expert_idx](x[mask])
                slot = (indices == expert_idx).float()
                w = (weights * slot).sum(dim=-1, keepdim=True)[mask]
                output[mask] += w * expert_out
        return output