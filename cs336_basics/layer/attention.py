from turtle import forward
import torch
from torch import nn
from .rope import RoPE
from .linear import Linear


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the softmax of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int, optional): Dimension along which to compute the softmax. Defaults to -1.

    Returns:
        torch.Tensor: Softmax tensor.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = (x - x_max).exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k: int):
        super().__init__()
        self.scale = 1 / (d_k ** 0.5)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Scaled Dot-Product Attention layer.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, n_heads, seq_len, d_k).
            key (torch.Tensor): Key tensor of shape (batch_size, n_heads, seq_len, d_k).
            value (torch.Tensor): Value tensor of shape (batch_size, n_heads, seq_len, d_k).
            mask (torch.Tensor, optional): Optional mask tensor of shape (batch_size, 1, seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_heads, seq_len, d_k).
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = stable_softmax(scores, dim=-1)
        return torch.matmul(attn, value)


class CasualMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1024, rope_theta: float = 10000.0, use_rope: bool = True, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.use_rope = use_rope
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.attn = ScaledDotProductAttention(self.d_k)

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            self.rope = RoPE(rope_theta, self.d_k, max_seq_len,
                             device=device)
        mask = torch.tril(torch.ones(
            (max_seq_len, max_seq_len), device=device, dtype=torch.bool))
        self.register_buffer('casual_mask', mask.unsqueeze(
            0).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads,
                   self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads,
                   self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads,
                   self.d_k).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        out = self.attn(q, k, v, self.casual_mask[:seq_len, :seq_len])
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        return self.out_proj(out)
