from collections import Counter, deque
from functools import lru_cache
import json


class BPETokenizer:
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}
        # For the official OpenAI GPT-2 merges, use a rank dict:
        #  of form {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks = {}
        # Cache for special token lookups
        self._special_token_cache = {}

    def _clear_special_token_cache(self):
        """Clear the special token cache."""
        self._special_token_cache = {}

    def train(self, text, vocab_size, allowed_special=None):
        """
        Train the BPE tokenizer from scratch.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}

        import re
        # Split text by special tokens so they're not treated as regular text for BPE training
        if allowed_special:
            special_pattern = "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            parts = re.split(special_pattern, text)
            # Filter out None and keep track of special tokens
            text_parts = []
            for part in parts:
                if part in allowed_special:
                    # Special token - skip from training text
                    continue
                text_parts.append(part)
            text = "".join(text_parts)

        # Preprocess: Replace spaces with "Ġ"
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with unique characters, including "Ġ" if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted(set(processed_text))
            if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.

        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe').
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # Convert loaded vocabulary to correct format
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        # Handle newline character - add as new token if not present
        if "\n" not in self.inverse_vocab:
            newline_token_id = len(self.vocab)
            self.vocab[newline_token_id] = "\n"
            self.inverse_vocab["\n"] = newline_token_id

        # Load GPT-2 merges and store them with an assigned "rank"
        self.bpe_ranks = {}  # reset ranks
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]

            rank = 0
            for line in lines:
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    token1, token2 = pair
                    # If token1 or token2 not in vocab, skip
                    if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                        self.bpe_ranks[(token1, token2)] = rank
                        rank += 1
                        print(f"Skipping pair {pair} as one token is not in the vocabulary.")

        self._clear_special_token_cache()
    def encode(self, text, allowed_special=None):
        """
        Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens.
    
        Args:
            text (str): The input text to encode.
            allowed_special (set or None): Special tokens to allow passthrough. If None, special handling is disabled.
    
        Returns:
            List of token IDs.
        """
        import re
    
        token_ids = []
    
        # If special token handling is enabled
        if allowed_special is not None and len(allowed_special) > 0:
            # Build regex to match allowed special tokens
            special_pattern = (
                "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            )
    
            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index:match.start()]
                # Check prefix for disallowed special tokens
                self._check_disallowed_special(prefix, allowed_special)
                token_ids.extend(self._encode_ordinary(prefix))
    
                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                last_index = match.end()
    
            text = text[last_index:]  # Remaining part to process normally
    
            # Check if any disallowed special tokens are in the remainder
            self._check_disallowed_special(text, allowed_special)
    
        # If no special tokens, or remaining text after special token split:
        token_ids.extend(self._encode_ordinary(text))
        return token_ids
    
    def _check_disallowed_special(self, text, allowed_special):
        """Check if text contains disallowed special tokens."""
        disallowed = [
            tok for tok in self.inverse_vocab
            if tok.startswith("<|") and tok.endswith("|>") and tok in text and tok not in allowed_special
        ]
        if disallowed:
            raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
    
    def _encode_ordinary(self, text):
        """Encode text without special token handling."""
        # Apply same preprocessing as train: replace spaces with Ġ
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)
    
        # Greedy longest-match tokenization against vocab
        token_ids = []
        i = 0
        while i < len(processed_text):
            # Find longest prefix in vocab
            longest_match = None
            longest_len = 0
            for length in range(len(processed_text) - i, 0, -1):
                candidate = processed_text[i:i + length]
                if candidate in self.inverse_vocab:
                    longest_match = candidate
                    longest_len = length
                    break
            if longest_match is not None:
                token_ids.append(self.inverse_vocab[longest_match])
                i += longest_len
            else:
                # Fall back to character-level BPE for unknown sequence
                # Find the next known token boundary or end
                j = i + 1
                while j <= len(processed_text) and processed_text[i:j] not in self.inverse_vocab:
                    j += 1
                unknown_seq = processed_text[i:j] if j <= len(processed_text) else processed_text[i:]
                token_ids.extend(self.tokenize_with_bpe(unknown_seq))
                i = j
    
        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        # If we haven't loaded OpenAI's GPT-2 merges, use merge priority order
        if not self.bpe_ranks:
            # Apply merges in the order they were created (most frequent first)
            # self.bpe_merges preserves insertion order (Python 3.7+)
            for pair, merged_id in self.bpe_merges.items():
                # Repeatedly merge all occurrences of this pair
                can_merge = True
                while can_merge and len(token_ids) > 1:
                    can_merge = False
                    new_tokens = []
                    i = 0
                    while i < len(token_ids) - 1:
                        if (token_ids[i], token_ids[i + 1]) == pair:
                            new_tokens.append(merged_id)
                            i += 2
                            can_merge = True
                        else:
                            new_tokens.append(token_ids[i])
                            i += 1
                    if i < len(token_ids):
                        new_tokens.append(token_ids[i])
                    token_ids = new_tokens
            return token_ids

        # Otherwise, do GPT-2-style merging with the ranks:
        # 1) Convert token_ids back to string "symbols" for each ID
        symbols = [self.vocab[id_num] for id_num in token_ids]

        # Repeatedly merge all occurrences of the lowest-rank pair
        while True:
            # Collect all adjacent pairs
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            # Find the pair with the best (lowest) rank
            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            # If no valid ranked pair is present, we're done
            if bigram is None or bigram not in self.bpe_ranks:
                break

            # Merge all occurrences of that pair
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                # If we see (first, second) at position i, merge them
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)  # merged symbol
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        # Finally, convert merged symbols back to IDs
        merged_ids = []
        for sym in symbols:
            if sym in self.inverse_vocab:
                merged_ids.append(self.inverse_vocab[sym])
            else:
                # Fallback: tokenize unknown symbol character by character
                # Use no-ranks path to avoid infinite recursion
                char_ids = [self.inverse_vocab.get(c) for c in sym]
                if None in char_ids:
                    missing = [c for c, cid in zip(sym, char_ids) if cid is None]
                    raise ValueError(f"Characters not found in vocab: {missing}")
                # Apply merges without ranks
                temp_ranks = self.bpe_ranks
                self.bpe_ranks = {}
                try:
                    merged_ids.extend(self.tokenize_with_bpe(sym))
                finally:
                    self.bpe_ranks = temp_ranks
        return merged_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        decoded_string = ""
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token.startswith("Ġ"):
                decoded_string += " " + token[1:]
            else:
                decoded_string += token
        return decoded_string

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

        self._clear_special_token_cache()

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced