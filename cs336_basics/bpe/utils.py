import mmap
import random
import json
from typing import Dict, List, Tuple


def bytes_to_unicode_local():
    """
    Returns a mapping from byte values to Unicode code points.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_and_sample_data(file_path: str, sample_size: int = 10 ** 9, special_token: str = "<|endoftext|>"):
    try:
        with open(file_path, 'r+', encoding='utf-8', errors='ignore') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                documents = []
                start = 0
                while start < len(mm):
                    end = mm.find(special_token.encode('utf-8'), start)
                    if end == -1:
                        doc = mm[start:].decode(
                            'utf-8', errors='replace').strip()
                        if doc:
                            documents.append(doc)
                        break
                    doc = mm[start:end].decode(
                        'utf-8', errors='replace').strip()
                    if doc:
                        documents.append(doc)
                    start = end + len(special_token)
            if len(documents) >= sample_size:
                documents = random.sample(documents, sample_size)
        return special_token.join(documents)
    except Exception as e:
        raise IOError(f"读取文件 {file_path} 时出错: {e}")


def save_vocab_and_merge(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], vocab_path: str, merge_path: str):
    vocab_str = {idx: token.decode('utf-8', errors='replace')
                 for idx, token in vocab.items()}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=2)

    with open(merge_path, 'w', encoding='utf-8') as f:
        for merge in merges:
            f.write(
                f"{merge[0].decode('utf-8', errors='replace')} {merge[1].decode('utf-8', errors='replace')}\n")
