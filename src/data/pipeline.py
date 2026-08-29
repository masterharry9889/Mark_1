# src/data/pipeline.py
"""Multi-tier data pipeline for LLM pretraining.

Tier 1: Foundation (80%) - Large-scale web/crawl data
Tier 2: Quality (15%) - Curated code, knowledge, academic
Tier 3: Alignment (5%) - Instruction/chat data
"""

from dataclasses import dataclass
from typing import Iterator
import torch
from datasets import load_dataset, interleave_datasets


@dataclass
class DataSource:
    name: str
    hf_path: str
    split: str
    weight: float
    text_key: str = "text"
    streaming: bool = True
    config: str | None = None


class MoEDataPipeline:
    """Multi-source data pipeline with tiered mixing ratios."""

    # Tier 1: Foundation (80% total) - Large-scale web/crawl data
    TIER1_SOURCES = [
        DataSource("fineweb", "HuggingFaceFW/fineweb", "train", 0.31, text_key="text"),
        DataSource("c4", "allenai/c4", "train", 0.22, text_key="text", config="en"),
        DataSource("dclm", "mlfoundations/dclm-baseline-1.0-parquet", "train", 0.18, text_key="text"),
        DataSource("fineweb_edu", "HuggingFaceFW/fineweb-edu", "train", 0.09, text_key="text", config="sample-10BT"),
    ]

    # Tier 2: Quality (15% total) - Curated knowledge, academic
    TIER2_SOURCES = [
        DataSource("wikitext", "Salesforce/wikitext", "train", 0.06, text_key="text", config="wikitext-103-v1"),
        DataSource("arxiv", "ccdv/arxiv-summarization", "train", 0.09, text_key="abstract"),
    ]

    # Tier 3: Alignment (5% total) - Instruction/chat data
    # OpenAssistant: Apache-2.0, Dolly: CC-BY-SA-3.0
    TIER3_SOURCES = [
        DataSource("openassistant", "OpenAssistant/oasst1", "train", 0.03, text_key="text"),
        DataSource("dolly", "databricks/databricks-dolly-15k", "train", 0.02, text_key="instruction"),
    ]

    ALL_SOURCES = TIER1_SOURCES + TIER2_SOURCES + TIER3_SOURCES

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 1024,
        seed: int = 42,
        buffer_size: int = 10_000,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.buffer_size = buffer_size
        self._datasets = {}
        self._iterators = {}
        self._load_all()

    def _load_all(self):
        """Load all datasets with streaming."""
        for src in self.ALL_SOURCES:
            try:
                kwargs = {"split": src.split, "streaming": src.streaming}
                if src.config:
                    kwargs["name"] = src.config
                ds = load_dataset(src.hf_path, **kwargs)
                ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)
                self._datasets[src.name] = {"source": src, "dataset": ds}
                self._iterators[src.name] = iter(ds)
            except Exception as e:
                print(f"Warning: Failed to load {src.name} ({src.hf_path}): {e}")

    def _next_sample(self, name: str) -> dict | None:
        """Get next sample from dataset, reinitialize iterator if exhausted."""
        it = self._iterators.get(name)
        if it is None:
            return None
        try:
            return next(it)
        except StopIteration:
            # Re-shuffle and restart
            src = self._datasets[name]["source"]
            ds = load_dataset(src.hf_path, split=src.split, streaming=True)
            ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)
            self._iterators[name] = iter(ds)
            return next(self._iterators[name])

    def _extract_text(self, sample: dict, source: DataSource) -> str:
        """Extract text field based on source configuration."""
        key = source.text_key
        if key not in sample:
            # Fallback: try common keys
            for k in ["text", "content", "instruction", "abstract", "conversations"]:
                if k in sample:
                    key = k
                    break
        value = sample.get(key, "")
        if isinstance(value, list):
            # Handle conversations format
            return " ".join(str(v) for v in value)
        return str(value)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from the weighted mixture of all sources.
        
        Returns (input_ids, labels) where labels = input_ids shifted by 1.
        Packs multiple documents into sequences of max_seq_len.
        """
        # Collect tokens from all sources proportionally
        all_tokens = []
        for src in self.ALL_SOURCES:
            if src.name not in self._datasets:
                continue
            n_samples = max(1, int(batch_size * src.weight * 4))  # Oversample for packing
            for _ in range(n_samples):
                sample = self._next_sample(src.name)
                if sample is None:
                    continue
                text = self._extract_text(sample, src)
                tokens = self.tokenizer.encode(text)
                if tokens:
                    all_tokens.extend(tokens + [self.tokenizer.eos_token_id])

        if not all_tokens:
            return (torch.empty(0, self.max_seq_len, dtype=torch.long),
                    torch.empty(0, self.max_seq_len, dtype=torch.long))

        # Pack tokens into sequences of max_seq_len
        packed = []
        for i in range(0, len(all_tokens), self.max_seq_len):
            chunk = all_tokens[i:i + self.max_seq_len]
            if len(chunk) == self.max_seq_len:
                packed.append(chunk)
            elif len(chunk) > 1:
                # Pad last chunk if needed
                chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(chunk))
                packed.append(chunk)

        if not packed:
            return (torch.empty(0, self.max_seq_len, dtype=torch.long),
                    torch.empty(0, self.max_seq_len, dtype=torch.long))

        packed_tensor = torch.tensor(packed, dtype=torch.long)
        
        # Shuffle packed sequences
        perm = torch.randperm(len(packed_tensor))
        packed_tensor = packed_tensor[perm][:batch_size]
        
        # Create input_ids and labels (shifted by 1)
        input_ids = packed_tensor[:, :-1]
        labels = packed_tensor[:, 1:]
        
        return input_ids, labels

    def __iter__(self, batch_size: int = 32) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Infinite iterator yielding (input_ids, labels) batches."""
        while True:
            yield self.sample_batch(batch_size)

    def get_source_stats(self) -> dict:
        """Return configured weights per source."""
        return {src.name: src.weight for src in self.ALL_SOURCES}


def create_pipeline(tokenizer, max_seq_len=1024, **kwargs) -> MoEDataPipeline:
    """Factory function for easy instantiation."""
    return MoEDataPipeline(tokenizer, max_seq_len, **kwargs)