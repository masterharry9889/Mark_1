# parallel/expert_parallel.py
class ExpertParallel:
    """Distribute experts across GPUs.

    GPU 0: Expert 0, Expert 1
    GPU 1: Expert 2, Expert 3
    ...
    """
    def __init__(self, experts, ep_size):
        self.ep_size = ep_size
        self.local_experts = self._shard_experts(experts, ep_size)

    def _shard_experts(self, experts, ep_size):
        # Each GPU gets n_experts / ep_size experts
        return nn.ModuleList([
            experts[i] for i in range(ep_size)
            if i % ep_size == dist.get_rank()
        ])

    def all_to_all_dispatch(self, tokens, indices):
        """Send tokens to GPUs holding their selected experts."""
        # Implementation depends on distributed backend
        pass

    def all_to_all_combine(self, expert_outputs, indices):
        """Send expert results back to originating GPUs."""
        pass
