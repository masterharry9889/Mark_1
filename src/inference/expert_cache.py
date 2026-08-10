import torch

# inference/expert_cache.py
class ExpertCache:
    """LRU cache for expert weights on GPU.
    Keeps most-used experts on GPU, offloads rest to CPU.
    """
    def __init__(self, all_experts, gpu_budget_gb=60):
        self.cache = {}
        self.access_order = []
        self.max_cached = int(gpu_budget_gb / 24)  # ~24 GB per expert

    def get_expert(self, expert_id):
        if expert_id in self.cache:
            self.access_order.remove(expert_id)
            self.access_order.append(expert_id)
            return self.cache[expert_id]

        if len(self.cache) >= self.max_cached:
            evict = self.access_order.pop(0)
            self.cache[evict].to('cpu')
            del self.cache[evict]

        expert = self._load(expert_id).to('cuda')
        self.cache[expert_id] = expert
        self.access_order.append(expert_id)
        return expert
