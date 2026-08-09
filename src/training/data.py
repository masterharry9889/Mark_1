import torch

# training/data.py
class MoEDataLoader:
    """Document packing + shuffling for MoE training."""
    def __init__(self, tokenizer, max_seq_len=1024, batch_size=1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def pack_documents(self, documents):
        packed, current = [], []
        for doc in documents:
            tokens = self.tokenizer.encode(doc)
            current.extend(tokens + [self.tokenizer.eos_token_id])
            while len(current) >= self.max_seq_len:
                packed.append(current[:self.max_seq_len])
                current = current[self.max_seq_len:]
        return torch.tensor(packed)

    def create_batch(self, token_ids):
        idx = torch.randperm(len(token_ids))[:self.batch_size]
        batch = token_ids[idx]
        return batch[:, :-1], batch[:, 1:]  # input_ids, labels