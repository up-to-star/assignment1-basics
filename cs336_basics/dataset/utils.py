import numpy as np
import torch
import numpy.typing as npt


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
