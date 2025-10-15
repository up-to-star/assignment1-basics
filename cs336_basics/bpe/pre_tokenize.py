from typing import Dict, List
import regex
import multiprocessing
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(document: str, bytes_to_unicode_map: Dict[int, str]) -> List[str]:
    """
    Pre-tokenize a document into a list of sequences.
    """
    tokens = regex.findall(PAT, document, flags=regex.UNICODE)
    sequences = []
    for token in tokens:
        token_unicode = ''.join(
            bytes_to_unicode_map[byte] for byte in token.encode('utf-8'))
        sequences.append(list(token_unicode))
    return sequences


global_worker_byte_map = None


def init_worker(bytes_to_unicode_map: Dict[int, str]):
    """
    Initialize the worker process with the bytes_to_unicode_map.
    """
    global global_worker_byte_map
    global_worker_byte_map = bytes_to_unicode_map


def pre_tokenize_worker(document: str) -> List[List[str]]:
    """
    Worker function for pre-tokenizing a document.
    """
    return pre_tokenize(document, global_worker_byte_map)


def parallel_pre_tokenize(documents: List[str], num_processes: int, bytes_to_unicode_map: Dict[int, str]) -> List[List[str]]:
    """
    Pre-tokenize documents in parallel.
    """
    if num_processes <= 1:
        return [seq for doc in documents for seq in pre_tokenize(doc, bytes_to_unicode_map)]

    with multiprocessing.Pool(num_processes, initializer=init_worker, initargs=(bytes_to_unicode_map,)) as pool:
        results = list(tqdm(pool.imap(pre_tokenize_worker, documents, chunksize=50), total=len(
            documents), desc="Pre-tokenizing", mininterval=1.0))
    return [seq for doc in results for seq in doc]
