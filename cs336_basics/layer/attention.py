from turtle import forward
import torch
from torch import nn


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
