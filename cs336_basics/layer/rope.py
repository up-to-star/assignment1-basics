import torch
from torch import nn


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.d_k = d_k
        freqs = 1.0 / \
            (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        # 得到每个token 每一组对应的角度
        angle_rads = positions.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer('cos', torch.cos(angle_rads), persistent=False)
        self.register_buffer('sin', torch.sin(angle_rads), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.d_k:
            raise ValueError(
                f"x.size(-1) must be {self.d_k}, but got {x.size(-1)}")
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # flatten last two dimensions, 将最后两个维度展平成一个维度
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out


if __name__ == '__main__':
    rope = RoPE(100.0, 4, 8)
    x = torch.randn(1, 8, 4)
    token_positions = torch.arange(8)
    out = rope(x, token_positions)
