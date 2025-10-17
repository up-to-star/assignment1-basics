import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(
            d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()\

        x = x / rms * self.weight
        return x.to(in_dtype)
