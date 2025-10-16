import regex
from collections import defaultdict
from typing import List, Tuple, Dict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        self.special_to_id = {}
        for token in self.special_tokens:
            token_bytes = token.encode('utf-8')
            for id_val, bytes_val in self.vocab.items():
                if bytes_val == token_bytes:
                    self.special_to_id[token] = id_val
                    break
        self.merges_priority_map = {
            pair: i for i, pair in enumerate(self.merges)}
