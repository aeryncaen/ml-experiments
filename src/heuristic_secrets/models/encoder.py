import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ByteEmbedding(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1, rope_base: int = 10000):
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(256, width)
        self.norm = RMSNorm(width)
        self.dropout = nn.Dropout(dropout)

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, width, 2).float() / width))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        emb = self.embed(x)

        pos = torch.arange(L, device=x.device, dtype=emb.dtype)
        inv_freq = self.inv_freq.to(emb.dtype)  # type: ignore[union-attr]
        angles = torch.outer(pos, inv_freq)
        cos, sin = angles.cos(), angles.sin()

        x1, x2 = emb[..., 0::2], emb[..., 1::2]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos

        rotated = torch.stack([rotated_1, rotated_2], dim=-1).flatten(-2)
        return self.dropout(self.norm(F.silu(rotated)))
