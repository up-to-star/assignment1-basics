import numpy as np
import torch
from torch import nn
import numpy.typing as npt
import os
from typing import IO, Any, BinaryIO


def get_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str):
    seq_len = len(x)
    max_valid_start = seq_len - context_length - 1
    start_indices = np.random.randint(0, max_valid_start + 1, size=batch_size)
    input_batch = torch.stack([torch.tensor(x[start:start+context_length], dtype=torch.long)
                              for start in start_indices])
    target_batch = torch.stack(
        [torch.tensor(x[start+1:start+context_length+1], dtype=torch.long)
         for start in start_indices])
    input_tensor = input_batch.to(device)
    target_tensor = target_batch.to(device)
    return input_tensor, target_tensor


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration
