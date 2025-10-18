import torch
from torch import nn
from .linear import Linear

def silu(x: torch.Tensor): return x * torch.sigmoid(x)

class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = Linear(d_model, d_ff, **factory_kwargs)
        self.linear2 = Linear(d_ff, d_model, **factory_kwargs)
        self.linear3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.linear1(x)
        w3_x = self.linear3(x)
        silu_x = silu(w1_x)
        glu_x = silu_x * w3_x
        return self.linear2(glu_x)
