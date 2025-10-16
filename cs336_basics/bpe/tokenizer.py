import regex
from collections import defaultdict
from typing import Iterator, List, Tuple, Dict, Iterable

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

    def _get_bpe_merges(self, piece: bytes) -> List[bytes]:
        parts = [bytes([b]) for b in piece]
        while len(parts) > 1:
            pairs = set()
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)
            if not pairs:
                break
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (part[i], parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    def encode(self, text: str) -> List[int]:
        if not text:
            return []

        sorted_special_tokens = sorted(
            self.special_tokens, key=len, reverse=True)
        special_token_pattern = "|".join(
            map(regex.escape, sorted_special_tokens))

        if self.special_tokens:
            chunks = regex.split(f'({special_token_pattern})', text)
        else:
            chunks = [text]

        final_ids = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                if chunk in self.special_to_id:
                    final_ids.append(self.special_to_id[chunk])
                else:
                    chunk_bytes = chunk.encode('utf-8')
                    if chunk_bytes in self.bytes_to_id:
                        final_ids.append(self.bytes_to_id[chunk_bytes])
                    else:
                        if '<unk>' in self.special_to_id:
                            final_ids.append(self.special_to_id['<unk>'])
                        else:
                            final_ids.append(self.special_to_id.get(self.special_tokens[0], 0))
            else:
                for word in regex.findall(PAT, chunk):
                    if not word:
                        continue
                    merge_pieces = self._get_bpe_merges(word.encode('utf-8'))
                    for piece in merge_pieces:
                        if piece in self.bytes_to_id:
                            final_ids.append(self.bytes_to_id[piece])
                        else:
                            if '<unk>' in self.special_to_id:
                                final_ids.append(self.special_to_id['<unk>'])
                            else:
                                final_ids.append(self.special_to_id.get(self.special_tokens[0], 0))
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for id in ids:
            if id in self.vocab:
                tokens.append(self.vocab[id])
            else:
                if '<unk>' in self.special_to_id and self.special_to_id['<unk>'] in self.vocab:
                    tokens.append(self.vocab[self.special_to_id['<unk>']])
                else:
                    if self.special_tokens:
                        first_special = self.special_tokens[0].encode('utf-8')
                        tokens.append(first_special)
                    else:
                        # 最后的选择：使用空格
                        tokens.append(b' ')
        all_bytes = b''.join(tokens)
        return all_bytes.decode('utf-8', errors='replace')

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = {line.strip().split()[0]: bytes.fromhex(line.strip().split()[1]) for line in f}
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges = [tuple(line.strip().split()) for line in f]
        return cls(vocab, merges, special_tokens)
