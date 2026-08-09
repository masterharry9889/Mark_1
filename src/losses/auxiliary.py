# losses/auxiliary.py
class MoEAuxiliaryLoss:
    """Load balancing + router z-loss.

    L_aux = α * N * Σ(f_i * P_i)  — load balancing
    L_z = (1/BT) * Σ(logsumexp(h_i))²  — numerical stability
    """
    def __init__(self, alpha=0.01, z_weight=0.001):
        self.alpha = alpha
        self.z_weight = z_weight

    def load_balancing_loss(self, router_logits, indices, n_experts):
        # f_i: fraction of tokens routed to expert i
        # P_i: mean router probability for expert i
        f = torch.zeros(n_experts, device=router_logits.device)
        for i in range(n_experts):
            f[i] = (indices == i).any(dim=-1).float().mean()
        P = F.softmax(router_logits, dim=-1).mean(dim=0)
        return self.alpha * n_experts * (f * P).sum()

    def z_loss(self, router_logits):
        return router_logits.float().logsumexp(dim=-1).square().mean()