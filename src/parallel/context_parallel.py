import torch
import torch.nn.functional as F
import torch.distributed as dist
# parallel/context_parallel.py
class ContextParallel:
    """Split long sequences across GPUs for attention."""
    def __init__(self, cp_size):
        self.cp_size = cp_size
        self.rank = dist.get_rank()

    def split_sequence(self, x):
        """Split sequence dimension: (B, T, D) → (B, T/cp, D)"""
        return x.chunk(self.cp_size, dim=1)[self.rank]

    def all_gather_attention(self, q, k, v):
        """Gather KV across context-parallel ranks for full attention."""
        # All-gather K and V from other ranks
        k_gathered = self.all_gather(k, dim=1)
        v_gathered = self.all_gather(v, dim=1)
        return F.scaled_dot_product_attention(q, k_gathered, v_gathered)