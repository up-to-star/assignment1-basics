import regex
from collections import defaultdict
from typing import Iterator, List, Tuple, Dict, Iterable
import json

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
            best_pair = min(
                pairs, key=lambda pair: self.merges_priority_map[pair])
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
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
                            final_ids.append(self.special_to_id.get(
                                self.special_tokens[0], 0))
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
                                final_ids.append(self.special_to_id.get(
                                    self.special_tokens[0], 0))
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
    def _load_vocab(cls, path: str) -> Dict[int, bytes]:
        """加载词汇表文件"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_str = json.load(f)
        return {int(idx): token.encode('utf-8') for idx, token in vocab_str.items()}

    @classmethod
    def _load_merges(cls, path: str) -> List[Tuple[bytes, bytes]]:
        """加载合并规则文件"""
        merges = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append(
                        (parts[0].encode('utf-8'), parts[1].encode('utf-8')))
        return merges

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """从文件加载BPE模型"""
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)


if __name__ == '__main__':
    vocab = {}
    merges = []

    # 添加单字节token (0-255)
    for i in range(256):
        vocab[i] = bytes([i])

    # 添加合并规则和合并后的token
    next_id = 256

    # 添加特殊token
    special_tokens = ["<|endoftext|>", "<pad>", "<unk>"]
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        vocab[next_id] = token_bytes
        next_id += 1

    # 添加BPE合并规则
    merges.append((b"h", b"i"))   # hi -> 256
    merges.append((b"t", b"h"))   # th -> 257
    merges.append((b"e", b"r"))   # er -> 258
    merges.append((b"th", b"e"))  # the -> 259

    # 为合并后的token分配ID
    vocab[next_id] = b"hi"; next_id += 1
    vocab[next_id] = b"th"; next_id += 1
    vocab[next_id] = b"er"; next_id += 1
    vocab[next_id] = b"the"; next_id += 1

    # 创建tokenizer实例
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens
    )

    # 测试字符串
    text = "the tokenizer<|endoftext|>hi there!"

    # 编码为ID序列
    ids = tokenizer.encode(text)
    print("编码后的ID序列:", ids)

    # 还原ID序列为文本
    decoded_text = tokenizer.decode(ids)
    print("还原后的文本:", repr(decoded_text))

    # 验证还原结果
    print("还原是否正确:", decoded_text == text)
