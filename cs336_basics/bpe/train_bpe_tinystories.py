from typing import Union, List, Tuple, Dict
import os
from pathlib import Path
from utils import bytes_to_unicode_local, load_and_sample_data, save_vocab_and_merge
import re
from pre_tokenize import parallel_pre_tokenize
from bpe_index import BPEIndex
from tqdm import tqdm


def run_train_bpe(
        input_path: Union[str, os.PathLike],
        vocab_size: int, special_tokens: List[str] = ['<|endoftext|>'],
        num_processes: int = 8,
        sample_size: int = 22000,
        **kwargs) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    base_vocab_size = 256 + len(special_tokens)
    if vocab_size < base_vocab_size:
        raise ValueError(f"vocab_size 必须大于等于 {base_vocab_size}")

    bytes_to_unicode_map = bytes_to_unicode_local()
    unicode_to_bytes_map = {v: bytes([k])
                            for k, v in bytes_to_unicode_map.items()}

    # 初始化词汇表
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    existing_bytes = set(vocab.values())

    # 添加特殊token到词汇表
    for st in special_tokens:
        st_bytes = st.encode('utf-8')
        if st_bytes not in existing_bytes and len(vocab) < vocab_size:
            vocab[next_token_id] = st_bytes
            existing_bytes.add(st_bytes)
            next_token_id += 1

    # 加载并采样数据
    text = load_and_sample_data(input_path, sample_size, special_tokens[0])

    # 分割文档
    escaped_tokens = [re.escape(st) for st in special_tokens]
    split_pattern = "|".join(escaped_tokens)
    documents = [part for part in re.split(split_pattern, text) if part]

    # 预分词
    sequences = parallel_pre_tokenize(
        documents, num_processes, bytes_to_unicode_map)
    # print(sequences[:10])

    merges = []
    bpe_index = BPEIndex(sequences)
    vocab_progress = len(vocab)
    total_merges = vocab_size - vocab_progress

    progress_bar = tqdm(total=total_merges, desc="Train BPE",
                        unit="merge", mininterval=0.5)
    while vocab_progress < vocab_size:
        best_pair = bpe_index.get_most_frequent_pair()
        if best_pair is None:
            break

        new_token_str = best_pair[0] + best_pair[1]
        p1_bytes = unicode_to_bytes_map[best_pair[0]]
        p2_bytes = unicode_to_bytes_map[best_pair[1]]
        new_token_bytes = p1_bytes + p2_bytes

        merge_count = bpe_index.merge_pair(best_pair, new_token_str)
        if merge_count == 0:
            continue

        if new_token_bytes not in existing_bytes:
            vocab[next_token_id] = new_token_bytes
            existing_bytes.add(new_token_bytes)
            next_token_id += 1
            merges.append((p1_bytes, p2_bytes))
            vocab_progress += 1
            progress_bar.update(1)
        unicode_to_bytes_map[new_token_str] = new_token_bytes

    progress_bar.close()
    return vocab, merges


if __name__ == '__main__':
    config = {
        "vocab_size": 10000,
        "special_tokens": ['<|endoftext|>'],
        "num_processes": 8,
        "sample_size": 22000,
    }

    train_path = os.path.join(os.path.dirname(
        __file__), "../../data/TinyStoriesV2-GPT4-train.txt")
    valid_path = os.path.join(os.path.dirname(
        __file__), "../../data/TinyStoriesV2-GPT4-valid.txt")

    if not Path(train_path).exists():
        raise FileNotFoundError(f"训练集文件 {train_path} 不存在")
    if not Path(valid_path).exists():
        raise FileNotFoundError(f"验证集文件 {valid_path} 不存在")

    train_vocab, train_merges = run_train_bpe(train_path, **config)
    save_vocab_and_merge(train_vocab, train_merges, 'vocab.json', 'merges.txt')
    # print(train_vocab)
