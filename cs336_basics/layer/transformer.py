from tkinter import NO
from .attention import CasualMultiHeadAttention
from .norm import RMSNorm
from .swiglu import SwiGLU
import torch
from torch import nn


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, d_ff: int, use_rope: bool = True, rope_theta: float = 10000.0, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = CasualMultiHeadAttention(
            d_model, num_heads, max_seq_len, rope_theta=rope_theta, use_rope=use_rope, device=device, dtype=dtype)

        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positins: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), token_positins)
        x = x + self.ffn(self.norm2(x))
        return x
