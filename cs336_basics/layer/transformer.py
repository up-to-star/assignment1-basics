from tkinter import NO
from .attention import CasualMultiHeadAttention
from .norm import RMSNorm
from .swiglu import SwiGLU
from .embedding import Embedding
from .linear import Linear
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


class TransfromerLM(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float = 10000.0, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, d_ff,
                             use_rope=True, rope_theta=rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.proj = Linear(d_model, vocab_size, device=device, dtype=dtype)

        self.context_length = context_length

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        assert seq_len <= self.context_length, f"Sequence length {seq_len} exceeds context length {self.context_length}"

        x = self.embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(
            0).expand(batch_size, seq_len)

        for block in self.blocks:
            x = block(x, positions)

        x = self.norm(x)
        logits = self.proj(x)
        return logits
