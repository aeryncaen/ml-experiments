from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ByteEmbedding
from .backbone import ConvBackbone, AttentionBackbone


class FeatureMaker(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...


class ConvBranch(nn.Module):
    def __init__(
        self,
        width: int,
        kernel_size: int,
        groups: int = 2,
        offset_scale: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        from .backbone import DeformConv1d
        self.conv = DeformConv1d(width, kernel_size, groups, offset_scale)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.dropout(self.norm(self.conv(x)))
        if mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled = h.sum(dim=1) / lengths
        else:
            pooled = h.mean(dim=1)
        return self.proj(pooled).squeeze(-1)


class ConvFeature(FeatureMaker):
    def __init__(
        self,
        width: int,
        kernel_sizes: list[int] = (3, 5, 7, 9),
        groups: int = 2,
        offset_scale: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = ByteEmbedding(width, dropout)
        self.branches = nn.ModuleList([
            ConvBranch(width, ks, groups, offset_scale, dropout)
            for ks in kernel_sizes
        ])
        self._output_dim = len(kernel_sizes)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.embedding(x)
        features = [branch(h, mask) for branch in self.branches]
        return torch.stack(features, dim=-1)


class AttentionFeature(FeatureMaker):
    def __init__(
        self,
        width: int,
        depth: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        pool: str = "attn",
    ):
        super().__init__()
        self.embedding = ByteEmbedding(width, dropout)
        self.backbone = AttentionBackbone(width, depth, num_heads, ffn_mult, dropout)
        self.pool_type = pool
        self._output_dim = 1

        if pool == "attn":
            self.query = nn.Parameter(torch.randn(width))
            self.scale = width**-0.5
        self.proj = nn.Linear(width, 1)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.embedding(x)
        h = self.backbone(h, mask)

        if self.pool_type == "attn":
            B = h.shape[0]
            q = self.query.unsqueeze(0).expand(B, -1)
            attn = torch.bmm(q.unsqueeze(1), h.transpose(1, 2)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
            attn = F.softmax(attn, dim=-1)
            pooled = torch.bmm(attn, h).squeeze(1)
        elif mask is not None:
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled = h.sum(dim=1) / lengths
        else:
            pooled = h.mean(dim=1)

        return self.proj(pooled)


class HeuristicFeature(FeatureMaker):
    def __init__(self, fn: Callable[[bytes], float], name: str = "heuristic"):
        super().__init__()
        self.fn = fn
        self.name = name
        self._output_dim = 1

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        results = []

        for i in range(B):
            if mask is not None:
                length = (~mask[i]).sum().item()
                byte_seq = x[i, :length].cpu().numpy().tobytes()
            else:
                byte_seq = x[i].cpu().numpy().tobytes()
            results.append(self.fn(byte_seq))

        return torch.tensor(results, device=device, dtype=torch.float32).unsqueeze(-1)


class PrecomputedFeature(FeatureMaker):
    def __init__(self, index: int, name: str = "precomputed"):
        super().__init__()
        self.index = index
        self.name = name
        self._output_dim = 1

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x[:, self.index : self.index + 1]
