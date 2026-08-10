"""Tokenizer wrapper for consistent interface."""

import os
from pathlib import Path
from typing import Optional

# Try to import tiktoken first (GPT-2 compatible)
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Local BPE tokenizer
from .Tokenizer.BPETokenizer import BPETokenizer


class TokenizerWrapper:
    """Unified tokenizer interface matching GPT-2 tokenizer."""

    def __init__(self, tokenizer_type: str = "gpt2", vocab_path: Optional[str] = None, merges_path: Optional[str] = None):
        self.tokenizer_type = tokenizer_type

        if tokenizer_type == "gpt2":
            if not HAS_TIKTOKEN:
                raise ImportError("tiktoken required for gpt2 tokenizer. Install: pip install tiktoken")
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab
            self.eos_token_id = self.tokenizer.eot_token
            self.bos_token_id = self.tokenizer.eot_token
            self.pad_token_id = self.tokenizer.eot_token

        elif tokenizer_type == "bpe":
            self.tokenizer = BPETokenizer()
            if vocab_path and merges_path:
                self.tokenizer.load_vocab_and_merges(vocab_path, merges_path)
            elif vocab_path and os.path.exists(vocab_path.replace("vocab.json", "merges.txt")):
                self.tokenizer.load_vocab_and_merges(vocab_path, vocab_path.replace("vocab.json", "merges.txt"))
            self.vocab_size = len(self.tokenizer.vocab)
            self.eos_token_id = self.tokenizer.vocab.get("</s>", 2)
            self.bos_token_id = self.tokenizer.vocab.get("<s>", 1)
            self.pad_token_id = self.tokenizer.vocab.get("<pad>", 0)

        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.tokenizer_type == "gpt2":
            return self.tokenizer.encode(text)
        else:
            return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer_type == "gpt2":
            return self.tokenizer.decode(token_ids)
        else:
            return self.tokenizer.decode(token_ids)

    def __len__(self) -> int:
        return self.vocab_size


def create_tokenizer(tokenizer_type: str = "gpt2", **kwargs) -> TokenizerWrapper:
    """Factory function to create tokenizer."""
    return TokenizerWrapper(tokenizer_type, **kwargs)