from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


@dataclass
class ByteMaskerConfig:
    vocab_size: int = 256
    width: int = 32
    depth: int = 3
    kernel_size: int = 9
    groups: int = 2
    offset_scale: float = 2.0
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "width": self.width,
            "depth": self.depth,
            "kernel_size": self.kernel_size,
            "groups": self.groups,
            "offset_scale": self.offset_scale,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ByteMaskerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def tiny(cls) -> "ByteMaskerConfig":
        return cls(width=32, depth=2, kernel_size=7, groups=2, dropout=0.1)

    @classmethod
    def small(cls) -> "ByteMaskerConfig":
        return cls(width=32, depth=3, kernel_size=9, groups=2, dropout=0.1)

    @classmethod
    def medium(cls) -> "ByteMaskerConfig":
        return cls(width=48, depth=4, kernel_size=9, groups=2, dropout=0.1)


class DeformConv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        groups: int = 4,
        offset_scale: float = 2.0,
    ):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")

        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.group_channels = channels // groups
        self.offset_scale = offset_scale

        self.dw_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )

        self.offset_net = nn.Linear(channels, groups * kernel_size)
        self.mask_net = nn.Linear(channels, groups * kernel_size)
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset_net.weight.data, 0.)
        constant_(self.offset_net.bias.data, 0.)
        constant_(self.mask_net.weight.data, 0.)
        constant_(self.mask_net.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, C = x.shape
        K = self.kernel_size
        G = self.groups
        gc = self.group_channels
        
        x_proj = self.input_proj(x)

        x_dw = x.transpose(1, 2)
        x_dw = self.dw_conv[0](x_dw)
        x_dw = x_dw.transpose(1, 2)
        x_dw = self.dw_conv[1](x_dw)
        x_dw = self.dw_conv[2](x_dw)

        offsets = self.offset_net(x_dw).view(N, L, G, K) * self.offset_scale
        mask = F.softmax(self.mask_net(x_dw).view(N, L, G, K), dim=-1)

        ref_offsets = torch.linspace(-(K // 2), K // 2, K, device=x.device)
        pos_indices = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L, 1)

        x_grouped = x_proj.view(N, L, G, gc)
        output = torch.zeros(N, L, G, gc, device=x.device, dtype=x.dtype)
        
        batch_idx = torch.arange(N, device=x.device).view(N, 1, 1)
        group_idx = torch.arange(G, device=x.device).view(1, 1, G)

        for k in range(K):
            abs_pos = pos_indices + ref_offsets[k] + offsets[:, :, :, k]
            abs_pos_clamped = abs_pos.clamp(0, L - 1)
            
            p_floor = abs_pos_clamped.long().clamp(0, L - 1)
            p_ceil = (p_floor + 1).clamp(0, L - 1)
            w_ceil = abs_pos_clamped - p_floor.float()
            w_floor = 1.0 - w_ceil
            
            oob = (abs_pos < 0) | (abs_pos > L - 1)
            w_floor = w_floor * (~oob).float()
            w_ceil = w_ceil * (~oob).float()

            v_floor = x_grouped[batch_idx, p_floor, group_idx, :]
            v_ceil = x_grouped[batch_idx, p_ceil, group_idx, :]
            
            sampled = v_floor * w_floor.unsqueeze(-1) + v_ceil * w_ceil.unsqueeze(-1)
            output = output + sampled * mask[:, :, :, k:k+1]

        return self.output_proj(output.reshape(N, L, C))


class DeformConv1dBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        groups: int = 4,
        offset_scale: float = 2.0,
    ):
        super().__init__()
        self.deform_conv = DeformConv1d(
            channels=channels,
            kernel_size=kernel_size,
            groups=groups,
            offset_scale=offset_scale,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.deform_conv(x))


class ByteMaskerModel(nn.Module):
    def __init__(self, config: ByteMaskerConfig, embedding: nn.Embedding | None = None):
        super().__init__()
        self.config = config

        self.embedding = embedding if embedding is not None else nn.Embedding(config.vocab_size, config.width)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DeformConv1dBlock(
                channels=config.width,
                kernel_size=config.kernel_size,
                groups=config.groups,
                offset_scale=config.offset_scale,
            )
            for _ in range(config.depth)
        ])
        self.layer_dropout = nn.Dropout(config.dropout)

        self.output_head = nn.Sequential(
            nn.Linear(config.width, config.width // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.embedding(x)
        h = self.embed_dropout(h)

        for layer in self.layers:
            h = layer(h)
            h = self.layer_dropout(h)

        logits = self.output_head(h).squeeze(-1)
        
        if return_hidden:
            return logits, h
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x, return_hidden=False)
            assert isinstance(logits, torch.Tensor)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).long()
