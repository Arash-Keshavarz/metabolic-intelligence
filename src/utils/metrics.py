from __future__ import annotations

import torch


@torch.no_grad()
def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))

