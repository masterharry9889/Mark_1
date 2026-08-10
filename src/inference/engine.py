import torch
import torch.nn.functional as F
# inference/engine.py
class MoEInferenceEngine:
    """Production inference with KV cache + expert caching."""
    def __init__(self, model, max_batch_size=32, max_seq_len=4096):
        self.model = model
        self.kv_cache = {}
        self.expert_cache = ExpertCache(model.experts, gpu_budget_gb=60)

    def generate(self, prompt_ids, max_new_tokens=512, temperature=0.8, top_p=0.95):
        input_ids = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.forward_with_cache(input_ids)
            next_token = self.sample(logits[:, -1, :], temperature, top_p)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def sample(self, logits, temperature, top_p):
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        # Top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(sorted_probs, 1)
        return sorted_indices.gather(-1, idx)
    