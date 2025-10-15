
from typing import Union, List
import os


def run_train_bpe(input_path: Union[str, os.PathLike], vocab_size: int, special_tokens: List[str] = ['<|endoftext|>'], num_processes: int = 8, sample_size: int = 22000, **kwargs):
    pass


if __name__ == '__main__':
    print("train_bpe_tinystories")
