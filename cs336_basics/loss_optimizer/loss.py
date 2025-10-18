import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math


def cross_entropy_loss(logits, targets):
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits_shift = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shift), dim=-1))
    target_probs = torch.gather(
        logits_shift, dim=-1, index=targets.unsqueeze(-1))
    loss = log_sum_exp - target_probs.squeeze(-1)
    return loss.mean()


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas,
                    "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 1)
                grad = p.grad.data
                m = state.get('m', torch.zeros_like(grad))
                v = state.get('v', torch.zeros_like(grad))
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
        return loss

if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
