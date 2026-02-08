"""Learnable Swish activation — per-channel beta parameter."""

import torch
import torch.nn as nn


class LearnableSwish(nn.Module):
    """Swish with a learnable per-channel beta: x * sigmoid(beta * x).

    When beta=1 this is standard SiLU. The learnable beta allows the network
    to smoothly interpolate between linear (beta->0) and ReLU-like (beta->inf).

    Args:
        dim: Number of channels. Beta is a (dim,) parameter initialized to ones.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)
