import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """
    Learnable gating network for MoE routing.
    
    Args:
        d_model (int):   Input hidden dimension (6144)
        n_experts (int): Number of experts (8)
        top_k (int):     Experts activated per token (2)
        noise_std (float): Jitter noise std for training stability (0.01)
        bias (bool):     Whether to use router bias (False for Mixtral-style)
    """

    def __init__(self, d_model, n_experts, top_k, noise_std=0.01, bias=False):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.noise_std = noise_std

        # the single learned parameter of the router
        self.gate = nn.Linear(d_model, n_experts, bias=bias)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, x):
        router_logits = self.gate(x)

        if self.training and self.noise_std > 0:
            router_logits += torch.randn_like(router_logits) * self.noise_std

        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, router_logits
    
    def compute_aux_loss(self, logits, indices):
        """Compute load balancing auxiliary loss."""
        # logits: (B*T, n_experts), indices: (B*T, top_k)
        # Encourage uniform expert utilization
        expert_counts = torch.zeros(self.n_experts, device=logits.device)
        for k in range(self.top_k):
            expert_counts.scatter_add_(0, indices[:, k].flatten(), 
                                      torch.ones_like(indices[:, k].flatten(), dtype=torch.float))
        
        # Normalize by total tokens
        expert_probs = expert_counts / indices.numel()
        
        # Uniform target
        uniform_probs = torch.ones_like(expert_probs) / self.n_experts
        
        # KL divergence
        aux_loss = (uniform_probs * (uniform_probs / (expert_probs + 1e-8)).log()).sum()
        return aux_loss
    
    def compute_z_loss(self, logits):
        """Compute z-loss to prevent logits from growing too large."""
        # logits: (B*T, n_experts)
        z_loss = (logits.logsumexp(dim=-1) ** 2).mean()
        return z_loss