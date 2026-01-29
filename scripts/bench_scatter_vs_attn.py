#!/usr/bin/env python3

import argparse
import math
import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import torchaudio

from heuristic_secrets.models.scatter_attention import (
    FlatRippleClassifierND,
    HierarchicalLocalAttention,
    HierarchicalLocalAttentionND,
    LocalAttentionND,
    RMSNorm,
    RippleClassifierND,
    SGSBBlockND,
    apply_rope,
    sinusoidal_pos_embed_nd,
)
from heuristic_secrets.models.ripple_attention import RippleAttention, RippleClassifier, RippleChannelClassifier
from heuristic_secrets.models.backbone import SSMMixer3
from heuristic_secrets.models.backbone2d import SSMBlock3_2d
from heuristic_secrets.models.telephone_attention import TelephoneAttentionND
from heuristic_secrets.models.ponder import PonderWrapper, PonderTrainer, PonderTrainConfig
from heuristic_secrets.data.synthetic import load_task, TASKS


SPEECHCOMMANDS_LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
]


class PackRGB:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = (x * 255).to(torch.int32).unbind(0)
        packed = (r << 16) | (g << 8) | b
        return (packed.float() / 16777215.0) * 2 - 1


class SpeechCommandsDataset(Dataset):
    def __init__(self, root: str, subset: str, seq_len: int = 16000):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root, download=True, subset=subset)
        self.seq_len = seq_len
        self.label_to_idx = {label: i for i, label in enumerate(SPEECHCOMMANDS_LABELS)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        waveform, sample_rate, label, *_ = self.dataset[idx]
        waveform = waveform.squeeze(0)

        if waveform.shape[0] < self.seq_len:
            waveform = F.pad(waveform, (0, self.seq_len - waveform.shape[0]))
        else:
            waveform = waveform[:self.seq_len]

        return waveform, self.label_to_idx[label]


class SwiGLU(nn.Module):
    def __init__(self, width: int, mult: int = 4):
        super().__init__()
        hidden = int(width * mult * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MLDecoderHead(nn.Module):
    def __init__(self, width: int, n_classes: int, num_groups: int | None = None, num_heads: int = 8):
        super().__init__()
        self.width = width
        self.n_classes = n_classes
        
        for h in [num_heads, 4, 2, 1]:
            if width % h == 0:
                num_heads = h
                break
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim ** -0.5
        
        if num_groups is None:
            num_groups = min(n_classes, 100)
        self.num_groups = num_groups
        self.group_size = (n_classes + num_groups - 1) // num_groups
        
        self.queries = nn.Parameter(torch.randn(num_groups, width) * 0.02)
        
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        
        self.group_fc = nn.Parameter(torch.randn(num_groups, width, self.group_size) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        q = self.q_proj(self.queries).view(1, self.num_groups, H, D).expand(B, -1, -1, -1)
        k = self.k_proj(x).view(B, L, H, D)
        v = self.v_proj(x).view(B, L, H, D)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, self.num_groups, C)
        
        logits = torch.einsum('bgc,gco->bgo', attn, self.group_fc)
        logits = logits.reshape(B, -1)[:, :self.n_classes]
        
        return logits


class SDPAttention(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, num_kv_heads: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = width // num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(width, self.kv_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(width, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q)
        k = apply_rope(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out(out)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, num_kv_heads: int | None = None, use_ssm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention(width, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.norm_ssm = RMSNorm(width)
            self.ssm = SSMMixer3(width, n_heads=num_heads, use_conv=False, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.norm_ssm(x))
            x = x + ssm_out
        return x


class HierarchicalBlock(nn.Module):
    def __init__(self, width: int, window_size: int = 17, num_channels: int = 4, dropout: float = 0.1, use_ssm: bool = False, conv_position: str = 'both', attn_residual: bool = True, merge_mode: str = 'lowrank', lowrank_hier: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.hier_attn = HierarchicalLocalAttention(width, window_size, num_channels, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
        self.attn_norm = RMSNorm(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm_norm = RMSNorm(width)
            self.ssm = SSMMixer3(width, n_heads=num_channels, use_conv=False, dropout=dropout)
            self.ssm_out_norm = RMSNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.hier_attn(self.norm1(x)))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.ssm_norm(x))
            x = x + self.ssm_out_norm(ssm_out)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, width: int, reduction: int = 4):
        super().__init__()
        hidden = max(width // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        scale = self.pool(x.transpose(1, 2)).squeeze(-1)
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(scale))))
        return x * scale.unsqueeze(1)


class ConvBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int = 17):
        super().__init__()
        self.dwconv = nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.norm = RMSNorm(width)
        hidden = int(width * 4 * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.pwconv1 = nn.Linear(width, hidden, bias=False)
        self.pwconv2 = nn.Linear(hidden, width, bias=False)
        self.se = SqueezeExcite(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(F.silu(self.pwconv1(x)))
        x = self.se(x)
        return residual + x


class TelephoneAttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.telephone_attn = TelephoneAttentionND(
            channels=width, ndim=1,
            num_heads=num_heads, use_triton=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.telephone_attn(self.norm1(x))
        x = x + h
        return x


class RippleAttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int = 8, order: str = "tele,conv,lowrank"):
        super().__init__()
        self.ripple_attn = RippleAttention(
            channels=width, num_heads=num_heads,
            use_triton=True, order=order
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.ripple_attn(x)
        x = x + h
        return x


class SDPAttention2D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, num_kv_heads: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = width // num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(width, self.kv_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(width, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        L = H * W
        
        x_flat = x.reshape(B, L, C)
        q = self.q_proj(x_flat).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_flat).reshape(B, L, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q)
        k = apply_rope(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, H, W, C)
        return self.out(out)


class AttentionBlock2D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, num_kv_heads: int | None = None, use_ssm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention2D(width, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.norm_ssm = RMSNorm(width)
            self.ssm = SSMBlock3_2d(width, n_heads=num_heads, use_conv=False, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.norm_ssm(x))
            x = x + ssm_out
        return x


class LocalBlock2D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 7, num_channels: int = 4, dropout: float = 0.1, use_ssm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.local_attn = LocalAttentionND(width, kernel_size, ndim=2, num_channels=num_channels)
        self.attn_norm = RMSNorm(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm_norm = RMSNorm(width)
            self.ssm = SSMBlock3_2d(width, n_heads=num_channels, use_conv=False, dropout=dropout)
            self.ssm_out_norm = RMSNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.local_attn(self.norm1(x)))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.ssm_norm(x))
            x = x + self.ssm_out_norm(ssm_out)
        return x


class HierarchicalBlock2D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 7, num_channels: int = 4, dropout: float = 0.1, use_ssm: bool = False, conv_position: str = 'both', attn_residual: bool = True, merge_mode: str = 'lowrank', lowrank_hier: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.hier_attn = HierarchicalLocalAttentionND(width, kernel_size, ndim=2, num_channels=num_channels, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
        self.attn_norm = RMSNorm(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm_norm = RMSNorm(width)
            self.ssm = SSMBlock3_2d(width, n_heads=num_channels, use_conv=False, dropout=dropout)
            self.ssm_out_norm = RMSNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.hier_attn(self.norm1(x)))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.ssm_norm(x))
            x = x + self.ssm_out_norm(ssm_out)
        return x


class SqueezeExcite2D(nn.Module):
    def __init__(self, width: int, reduction: int = 4):
        super().__init__()
        hidden = max(width // reduction, 8)
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        scale = x.mean(dim=(1, 2))
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(scale))))
        return x * scale.unsqueeze(1).unsqueeze(2)


class ConvBlock2D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv2d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.norm = RMSNorm(width)
        hidden = int(width * 4 * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.pwconv1 = nn.Linear(width, hidden, bias=False)
        self.pwconv2 = nn.Linear(hidden, width, bias=False)
        self.se = SqueezeExcite2D(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        residual = x
        x = self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(F.silu(self.pwconv1(x)))
        x = self.se(x)
        return residual + x


class SequenceClassifier(nn.Module):
    def __init__(self, block: nn.Module, width: int, n_layers: int, n_classes: int, seq_len: int, ml_decoder: bool = False):
        super().__init__()
        self.embed = nn.Linear(1, width)
        self.embed_norm = RMSNorm(width)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
        self.pos_norm = RMSNorm(width)
        self.layers = nn.ModuleList([block for _ in range(n_layers)])
        self.norm = RMSNorm(width)
        self.ml_decoder = ml_decoder
        if ml_decoder:
            self.head = MLDecoderHead(width, n_classes)
        else:
            self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_norm(F.silu(self.embed(x.unsqueeze(-1)))) + self.pos_norm(F.silu(self.pos_embed))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.ml_decoder:
            return self.head(x)
        else:
            return self.head(x.mean(dim=1))


class ImageClassifier(nn.Module):
    def __init__(self, block: nn.Module, width: int, n_layers: int, n_classes: int, img_size: tuple[int, int]):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = nn.Linear(1, width)
        self.embed_norm = RMSNorm(width)
        self.pos_embed = nn.Parameter(torch.randn(1, *img_size, width) * 0.02)
        self.pos_norm = RMSNorm(width)
        self.layers = nn.ModuleList([block for _ in range(n_layers)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, *self.img_size, 1)
        x = self.embed_norm(F.silu(self.patch_embed(x))) + self.pos_norm(F.silu(self.pos_embed))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=(1, 2))
        return self.head(x)


class VolumeClassifier(nn.Module):
    def __init__(self, block: nn.Module, width: int, n_layers: int, n_classes: int, vol_size: tuple[int, int, int]):
        super().__init__()
        self.vol_size = vol_size
        self.patch_embed = nn.Linear(1, width)
        self.embed_norm = RMSNorm(width)
        self.pos_embed = nn.Parameter(torch.randn(1, *vol_size, width) * 0.02)
        self.pos_norm = RMSNorm(width)
        self.layers = nn.ModuleList([block for _ in range(n_layers)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1).unsqueeze(-1)
        x = self.embed_norm(F.silu(self.patch_embed(x))) + self.pos_norm(F.silu(self.pos_embed))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=(1, 2, 3))
        return self.head(x)


class SequenceLM(nn.Module):
    def __init__(self, block: nn.Module, width: int, n_layers: int, vocab_size: int, seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.embed_norm = RMSNorm(width)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
        self.pos_norm = RMSNorm(width)
        self.layers = nn.ModuleList([block for _ in range(n_layers)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_norm(self.embed(x)) + self.pos_norm(F.silu(self.pos_embed))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class AudioClassifier(nn.Module):
    def __init__(self, block: nn.Module, width: int, n_layers: int, n_classes: int, seq_len: int):
        super().__init__()
        self.embed = nn.Linear(1, width)
        self.embed_norm = RMSNorm(width)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
        self.pos_norm = RMSNorm(width)
        self.layers = nn.ModuleList([block for _ in range(n_layers)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_norm(F.silu(self.embed(x.unsqueeze(-1)))) + self.pos_norm(F.silu(self.pos_embed))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


class SDPAttention3D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, num_kv_heads: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = width // num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(width, self.kv_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(width, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        L = H * W * D
        
        x_flat = x.reshape(B, L, C)
        q = self.q_proj(x_flat).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_flat).reshape(B, L, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q)
        k = apply_rope(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, H, W, D, C)
        return self.out(out)


class AttentionBlock3D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, num_kv_heads: int | None = None):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention3D(width, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LocalBlock3D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 5, num_channels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.local_attn = LocalAttentionND(width, kernel_size, ndim=3, num_channels=num_channels)
        self.attn_norm = RMSNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.local_attn(self.norm1(x)))
        return x


class HierarchicalBlock3D(nn.Module):
    def __init__(self, width: int, window_size: int = 5, num_channels: int = 4, dropout: float = 0.1, conv_position: str = 'both', attn_residual: bool = True, merge_mode: str = 'lowrank', lowrank_hier: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.hier_attn = HierarchicalLocalAttentionND(width, window_size, ndim=3, num_channels=num_channels, poolable_dims=(0, 1), conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
        self.attn_norm = RMSNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.hier_attn(self.norm1(x)))
        return x


class SqueezeExcite3D(nn.Module):
    def __init__(self, width: int, reduction: int = 4):
        super().__init__()
        hidden = max(width // reduction, 8)
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        scale = x.mean(dim=(1, 2, 3))
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(scale))))
        return x * scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)


class ConvBlock3D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 5):
        super().__init__()
        self.dwconv = nn.Conv3d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.norm = RMSNorm(width)
        hidden = int(width * 4 * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.pwconv1 = nn.Linear(width, hidden, bias=False)
        self.pwconv2 = nn.Linear(hidden, width, bias=False)
        self.se = SqueezeExcite3D(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        residual = x
        x = self.dwconv(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pwconv2(F.silu(self.pwconv1(x)))
        x = self.se(x)
        return residual + x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class DuoModel(nn.Module):
    """Two models trained with inverse curriculum, merged at inference."""
    
    MERGE_STRATEGIES = ['mean', 'max', 'logsum', 'harmonic']
    
    def __init__(self, model_hard: nn.Module, model_easy: nn.Module, merge: str = 'mean'):
        super().__init__()
        self.model_hard = model_hard  # Trains on hardest X%
        self.model_easy = model_easy  # Trains on easiest (100-X)%
        self.merge = merge
        assert merge in self.MERGE_STRATEGIES, f"Unknown merge: {merge}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns merged logits."""
        logits_hard = self.model_hard(x)
        logits_easy = self.model_easy(x)
        return self.merge_logits(logits_hard, logits_easy)
    
    def forward_both(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (merged, hard, easy) logits for training."""
        logits_hard = self.model_hard(x)
        logits_easy = self.model_easy(x)
        merged = self.merge_logits(logits_hard, logits_easy)
        return merged, logits_hard, logits_easy
    
    def merge_logits(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        if self.merge == 'mean':
            return (l1 + l2) / 2
        elif self.merge == 'max':
            return torch.max(l1, l2)
        elif self.merge == 'logsum':
            # Product of probabilities in log space
            return F.log_softmax(l1, dim=-1) + F.log_softmax(l2, dim=-1)
        elif self.merge == 'harmonic':
            # Harmonic mean: 2*a*b / (a+b)
            eps = 1e-8
            return 2 * l1 * l2 / (l1 + l2 + eps)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge}")


def train_epoch_duo(duo_model, loader, optimizer, device, scheduler=None, flatten=True, task_type='classification', hard_pct=0.5, label_smoothing=0.1):
    """Train duo model with inverse curriculum: hard model on top hard_pct%, easy model on bottom (1-hard_pct)%."""
    duo_model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    desc = f"Train [DUO h{int(hard_pct*100)}%/e{int((1-hard_pct)*100)}%]"
    pbar = tqdm(loader, desc=desc, leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if flatten and task_type == 'classification':
            inputs = inputs.view(inputs.size(0), -1)
        elif not flatten and task_type == 'classification' and inputs.dim() == 4:
            # For ND models: squeeze channel dim (B, 1, H, W) -> (B, H, W)
            inputs = inputs.squeeze(1)
        
        optimizer.zero_grad()
        
        # Forward both models
        merged, logits_hard, logits_easy = duo_model.forward_both(inputs)
        
        if task_type == 'lm':
            # Language modeling: per-token losses
            merged_flat = merged.view(-1, merged.size(-1))
            hard_flat = logits_hard.view(-1, logits_hard.size(-1))
            easy_flat = logits_easy.view(-1, logits_easy.size(-1))
            labels_flat = labels.view(-1)
            
            # Compute per-token loss from merged output for difficulty ranking
            per_token_loss = F.cross_entropy(merged_flat, labels_flat, ignore_index=-100, reduction='none', label_smoothing=label_smoothing)
            valid_mask = labels_flat != -100
            valid_losses = per_token_loss[valid_mask]
            valid_indices = torch.where(valid_mask)[0]
            
            # Sort by difficulty (descending)
            k_hard = max(1, int(hard_pct * valid_losses.size(0)))
            _, sorted_idx = valid_losses.sort(descending=True)
            hard_token_idx = valid_indices[sorted_idx[:k_hard]]
            easy_token_idx = valid_indices[sorted_idx[k_hard:]]
            
            # Compute losses for each model on their respective tokens
            loss_hard = F.cross_entropy(hard_flat[hard_token_idx], labels_flat[hard_token_idx], label_smoothing=label_smoothing) if len(hard_token_idx) > 0 else merged.new_zeros(())
            loss_easy = F.cross_entropy(easy_flat[easy_token_idx], labels_flat[easy_token_idx], label_smoothing=label_smoothing) if len(easy_token_idx) > 0 else merged.new_zeros(())
            loss = loss_hard + loss_easy
            
            # Accuracy on merged output
            preds = merged_flat.argmax(dim=-1)
            correct += (preds[valid_mask] == labels_flat[valid_mask]).sum().item()
            total += valid_mask.sum().item()
        else:
            # Classification: per-sample losses
            per_sample_loss = F.cross_entropy(merged, labels, reduction='none', label_smoothing=label_smoothing)
            
            # Sort by difficulty (descending = hardest first)
            k_hard = max(1, int(hard_pct * per_sample_loss.size(0)))
            _, sorted_idx = per_sample_loss.sort(descending=True)
            hard_idx = sorted_idx[:k_hard]
            easy_idx = sorted_idx[k_hard:]
            
            # Compute losses for each model on their respective samples
            loss_hard = F.cross_entropy(logits_hard[hard_idx], labels[hard_idx], label_smoothing=label_smoothing) if len(hard_idx) > 0 else merged.new_zeros(())
            loss_easy = F.cross_entropy(logits_easy[easy_idx], labels[easy_idx], label_smoothing=label_smoothing) if len(easy_idx) > 0 else merged.new_zeros(())
            loss = loss_hard + loss_easy
            
            # Accuracy on merged output
            correct += (merged.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(duo_model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}", acc=f"{correct/max(total,1):.4f}")
    
    return total_loss / len(loader), correct / max(total, 1)


def slerp_weights(sd_a, sd_b, t):
    merged = {}
    for key in sd_a:
        a, b = sd_a[key].float(), sd_b[key].float()
        if a.dim() == 0 or a.numel() == 1:
            merged[key] = ((1 - t) * a + t * b).to(sd_a[key].dtype)
            continue
        a_flat, b_flat = a.flatten(), b.flatten()
        na, nb = torch.linalg.norm(a_flat), torch.linalg.norm(b_flat)
        if na < 1e-8 or nb < 1e-8:
            merged[key] = ((1 - t) * a + t * b).to(sd_a[key].dtype)
            continue
        a_norm, b_norm = a_flat / na, b_flat / nb
        dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
        omega = torch.acos(dot)
        if omega.abs() < 1e-6:
            merged[key] = ((1 - t) * a + t * b).to(sd_a[key].dtype)
            continue
        sa, sb = torch.sin((1 - t) * omega) / torch.sin(omega), torch.sin(t * omega) / torch.sin(omega)
        mag = (1 - t) * na + t * nb
        merged[key] = ((sa * a_norm + sb * b_norm) * mag).reshape(a.shape).to(sd_a[key].dtype)
    return merged


def lerp_weights(sd_a, sd_b, t):
    return {k: ((1 - t) * sd_a[k].float() + t * sd_b[k].float()).to(sd_a[k].dtype) for k in sd_a}


def merge_teacher_student(teacher_sd, student_sd, alpha, method='ema'):
    if method == 'ema':
        return lerp_weights(teacher_sd, student_sd, alpha)
    elif method == 'lerp':
        return lerp_weights(teacher_sd, student_sd, alpha)
    elif method == 'slerp':
        return slerp_weights(teacher_sd, student_sd, alpha)
    else:
        raise ValueError(f'Unknown merge method: {method}')


@torch.no_grad()
def generate_teacher_logits(model, loader, device, flatten=True, task_type='classification'):
    model.eval()
    teacher_map = {}
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Distill", leave=False):
            inputs = inputs.to(device)
            if flatten and task_type == 'classification':
                inputs = inputs.view(inputs.size(0), -1)
            elif not flatten and task_type == 'classification' and inputs.dim() == 4:
                inputs = inputs.squeeze(1)
            logits = model(inputs).cpu()
            for i in range(inputs.size(0)):
                teacher_map[inputs[i].cpu().numpy().tobytes()] = logits[i]
    return teacher_map


@torch.no_grad()
def generate_teacher_logits_indexed(model, loader, device, flatten=True, task_type='classification'):
    model.eval()
    all_logits = []
    bs = loader.batch_size
    for i in range(0, len(loader.x), bs):
        x_batch = loader.x[i:i+bs].to(device)
        if flatten and task_type == 'classification':
            x_batch = x_batch.view(x_batch.size(0), -1)
        elif not flatten and task_type == 'classification' and x_batch.dim() == 4:
            x_batch = x_batch.squeeze(1)
        all_logits.append(model(x_batch).cpu())
    return torch.cat(all_logits, dim=0)


def ensemble_teacher_logits(logit_history, loss_history, top_k=3):
    stacked_logits = torch.stack(logit_history)
    stacked_losses = torch.stack(loss_history)
    E, N = stacked_losses.shape
    k = min(top_k, E)
    _, best_idx = stacked_losses.topk(k, dim=0, largest=False)
    selected_logits = torch.gather(stacked_logits, 0, best_idx.unsqueeze(-1).expand(-1, -1, stacked_logits.shape[-1]))
    selected_losses = torch.gather(stacked_losses, 0, best_idx)
    weights = 1.0 / (selected_losses + 1e-8)
    weights = weights / weights.sum(dim=0, keepdim=True)
    return (selected_logits * weights.unsqueeze(-1)).sum(dim=0)


def indexed_to_hashmap(logits_tensor, loader, device, flatten=True, task_type='classification'):
    teacher_map = {}
    bs = loader.batch_size
    idx = 0
    for i in range(0, len(loader.x), bs):
        x_batch = loader.x[i:i+bs].to(device)
        if flatten and task_type == 'classification':
            x_batch = x_batch.view(x_batch.size(0), -1)
        elif not flatten and task_type == 'classification' and x_batch.dim() == 4:
            x_batch = x_batch.squeeze(1)
        for j in range(x_batch.size(0)):
            teacher_map[x_batch[j].cpu().numpy().tobytes()] = logits_tensor[idx]
            idx += 1
    return teacher_map


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)


def train_epoch(model, loader, optimizer, device, scheduler=None, flatten=True, task_type='classification', wtf_mode=False, hard_pct=None, scaler=None, label_smoothing=0.1, teacher_logits=None, distill_alpha=0.5, distill_temp=2.0):
    model.train()
    total_loss_t = torch.tensor(0.0, device=device)
    correct_t = torch.tensor(0, device=device)
    total_t = torch.tensor(0, device=device)

    desc = "Train"
    if wtf_mode:
        desc += " [WTF]"
    if hard_pct is not None:
        desc += f" [H{int(hard_pct*100)}%]"
    if teacher_logits is not None:
        desc += " [SD]"
    pbar = tqdm(loader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if flatten and task_type == 'classification':
            inputs = inputs.view(inputs.size(0), -1)
        elif not flatten and task_type == 'classification' and inputs.dim() == 4:
            # For ND models: squeeze channel dim (B, 1, H, W) -> (B, H, W)
            inputs = inputs.squeeze(1)

        optimizer.zero_grad()
        
        amp_ctx = torch.autocast(device.type, dtype=torch.float16) if scaler else nullcontext()
        with amp_ctx:
            logits = model(inputs)

            if task_type == 'lm':
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                per_token_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction='none', label_smoothing=label_smoothing)
                if wtf_mode:
                    mask = labels_flat != -100
                    valid_losses = per_token_loss[mask]
                    median = valid_losses.median()
                    min_l, max_l = valid_losses.min(), valid_losses.max()
                    weights = torch.where(
                        valid_losses < median,
                        (valid_losses - median) / (median - min_l + 1e-8),
                        10 * (valid_losses - median) / (max_l - median + 1e-8)
                    )
                    loss = (valid_losses * weights).sum() / mask.sum()
                else:
                    if hard_pct is not None:
                        valid_mask = labels_flat != -100
                        valid_losses = per_token_loss[valid_mask]
                        k = max(1, int(hard_pct * valid_losses.size(0)))
                        topk_losses, _ = valid_losses.topk(k)
                        loss = topk_losses.mean()
                    else:
                        loss = per_token_loss.mean()
                mask = labels_flat != -100
                preds = logits_flat.argmax(dim=-1)
                correct_t += (preds[mask] == labels_flat[mask]).sum()
                total_t += mask.sum()
            else:
                per_sample_loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=label_smoothing)
                if wtf_mode:
                    median = per_sample_loss.median()
                    min_l, max_l = per_sample_loss.min(), per_sample_loss.max()
                    weights = torch.where(
                        per_sample_loss < median,
                        (per_sample_loss - median) / (median - min_l + 1e-8),
                        10 * (per_sample_loss - median) / (max_l - median + 1e-8)
                    )
                    loss = (per_sample_loss * weights).mean()
                else:
                    if hard_pct is not None:
                        k = max(1, int(hard_pct * per_sample_loss.size(0)))
                        topk_losses, _ = per_sample_loss.topk(k)
                        loss = topk_losses.mean()
                    else:
                        loss = per_sample_loss.mean()
                correct_t += (logits.argmax(dim=-1) == labels).sum()
                total_t += labels.size(0)

            if teacher_logits is not None:
                t_list = [teacher_logits[inputs[i].cpu().numpy().tobytes()] for i in range(inputs.size(0))]
                t_logits = torch.stack(t_list).to(device)
                kd_loss = distillation_loss(logits, t_logits, distill_temp)
                loss = (1 - distill_alpha) * loss + distill_alpha * kd_loss

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss_t += loss.detach()
        
        pbar.set_postfix(
            loss=f"{total_loss_t.item()/(pbar.n+1):.4f}",
            acc=f"{correct_t.item()/max(total_t.item(),1):.4f}"
        )

    total_loss = total_loss_t.item()
    correct = correct_t.item()
    total = max(total_t.item(), 1)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def compute_sample_losses(model, loader, device, flatten=True, task_type='classification'):
    model.eval()
    all_losses = []
    
    for inputs, labels in tqdm(loader, desc="Computing sample losses", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if flatten and task_type == 'classification':
            inputs = inputs.view(inputs.size(0), -1)
        
        logits = model(inputs)
        
        if task_type == 'lm':
            per_seq_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction='none')
            per_seq_loss = per_seq_loss.view(logits.size(0), -1).mean(dim=-1)
            all_losses.append(per_seq_loss.cpu())
        else:
            per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
            all_losses.append(per_sample_loss.cpu())
    
    return torch.cat(all_losses)


@torch.no_grad()
def evaluate(model, loader, device, desc="Eval", flatten=True, task_type='classification'):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if flatten and task_type == 'classification':
            inputs = inputs.view(inputs.size(0), -1)
        elif not flatten and task_type == 'classification' and inputs.dim() == 4:
            # For ND models: squeeze channel dim (B, 1, H, W) -> (B, H, W)
            inputs = inputs.squeeze(1)

        logits = model(inputs)

        if task_type == 'lm':
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
            mask = labels_flat != -100
            preds = logits_flat.argmax(dim=-1)
            correct += (preds[mask] == labels_flat[mask]).sum().item()
            total += mask.sum().item()
        else:
            loss = F.cross_entropy(logits, labels)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}", acc=f"{correct/max(total,1):.4f}")

    return total_loss / len(loader), correct / max(total, 1)


def train_model(model, train_loader, test_loader, device, epochs, lr, warmup_epochs=2, cosine_start=0.1, swa=False, swa_start=0.8, swa_lr=1e-5, hard_mining=False, hard_start=0.5, hard_end=0.05, first_epoch_pct=None, wtf_mode=False, checkpoint_dir=None, model_name='model', verbose=True, flatten=True, task_type='classification', use_amp=False, label_smoothing=0.1, self_distill=False, distill_alpha=0.5, distill_temp=2.0, distill_merge=None, distill_merge_alpha=0.5, distill_from_merged=False, distill_ensemble=False, distill_ensemble_k=3):
    import os
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    post_warmup = total_steps - warmup_steps
    static_steps = int(post_warmup * cosine_start)
    decay_steps = post_warmup - static_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        step_after_warmup = step - warmup_steps
        if step_after_warmup < static_steps:
            return 1.0
        decay_progress = (step_after_warmup - static_steps) / decay_steps
        return 0.5 * (1 + math.cos(math.pi * decay_progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    swa_model = AveragedModel(model) if swa else None
    swa_epoch = int(epochs * swa_start) if swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr) if swa else None
    in_swa_phase = False
    
    import copy
    merged_model = copy.deepcopy(model) if (self_distill and distill_merge) else None
    prev_sd = None
    ensemble_logits_history = []
    ensemble_loss_history = []
    
    current_loader = train_loader
    batch_size = train_loader.batch_size

    # Compute epoch-level cosine schedule for hard mining %
    post_warmup_epochs = epochs - warmup_epochs
    static_epochs = int(post_warmup_epochs * cosine_start)
    decay_epochs = post_warmup_epochs - static_epochs
    
    def get_hard_pct(ep):
        """Cosine-annealed hard mining: hard_start at warmup end → hard_end at training end"""
        if ep < warmup_epochs:
            return hard_start
        # Cosine decay from hard_start to hard_end over post-warmup phase
        progress = (ep - warmup_epochs) / max(epochs - warmup_epochs, 1)
        cosine_mult = 0.5 * (1 + math.cos(math.pi * progress))
        return hard_end + (hard_start - hard_end) * cosine_mult

    for epoch in range(epochs):
        use_swa_sched = swa and epoch >= swa_epoch
        if use_swa_sched and not in_swa_phase:
            in_swa_phase = True
        
        active_scheduler = swa_scheduler if in_swa_phase else scheduler
        is_duo = isinstance(model, DuoModel)
        if epoch == 0 and first_epoch_pct is not None:
            hard_pct = first_epoch_pct
        else:
            hard_pct = get_hard_pct(epoch)
        
        teacher = None
        if self_distill and epoch > 0:
            if distill_ensemble:
                epoch_logits = generate_teacher_logits_indexed(model, current_loader, device, flatten=flatten, task_type=task_type)
                epoch_losses = F.cross_entropy(epoch_logits, current_loader.y[:len(epoch_logits)], reduction='none')
                ensemble_logits_history.append(epoch_logits)
                ensemble_loss_history.append(epoch_losses)
                combined = ensemble_teacher_logits(ensemble_logits_history, ensemble_loss_history, top_k=distill_ensemble_k)
                teacher = indexed_to_hashmap(combined, current_loader, device, flatten=flatten, task_type=task_type)
            else:
                teacher_model = merged_model if (distill_from_merged and merged_model is not None and prev_sd is not None) else model
                teacher = generate_teacher_logits(teacher_model, current_loader, device, flatten=flatten, task_type=task_type)
        
        if is_duo:
            train_loss, train_acc = train_epoch_duo(
                model, current_loader, optimizer, device, active_scheduler,
                flatten=flatten, task_type=task_type, hard_pct=hard_pct, label_smoothing=label_smoothing
            )
        else:
            train_loss, train_acc = train_epoch(
                model, current_loader, optimizer, device, active_scheduler, 
                flatten=flatten, task_type=task_type, wtf_mode=wtf_mode, 
                hard_pct=hard_pct if hard_mining else None, scaler=scaler, label_smoothing=label_smoothing,
                teacher_logits=teacher, distill_alpha=distill_alpha, distill_temp=distill_temp,
            )
        
        if use_swa_sched:
            swa_model.update_parameters(model)
        
        test_loss, test_acc = evaluate(model, test_loader, device, flatten=flatten, task_type=task_type)
        
        merge_acc = None
        if merged_model is not None and teacher is not None:
            student_sd = model.state_dict()
            teacher_sd = {k: v.to(student_sd[k].device) for k, v in prev_sd.items()} if prev_sd is not None else {k: v.clone() for k, v in student_sd.items()}
            new_sd = merge_teacher_student(teacher_sd, student_sd, distill_merge_alpha, method=distill_merge)
            merged_model.load_state_dict(new_sd)
            _, merge_acc = evaluate(merged_model, test_loader, device, desc="Eval [M]", flatten=flatten, task_type=task_type)
        prev_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        
        swa_acc = None
        if use_swa_sched:
            update_bn(train_loader, swa_model, device=device)
            _, swa_acc = evaluate(swa_model, test_loader, device, flatten=flatten, task_type=task_type)
        
        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            swa_str = f' swa_acc={swa_acc:.4f}' if swa_acc is not None else ''
            merge_str = f' merge_acc={merge_acc:.4f} [{distill_merge}]' if merge_acc is not None else ''
            phase_str = ' [SWA]' if in_swa_phase else ''
            mine_str = ' [HEM]' if hard_mining else ''
            duo_str = f' [DUO h{int(hard_pct*100)}%/e{int((1-hard_pct)*100)}%]' if is_duo else ''
            sd_str = ' [SD]' if teacher is not None else ''
            if teacher is not None and distill_ensemble:
                sd_str = f' [SD-E{len(ensemble_logits_history)}]'
            print(f'Epoch {epoch+1:2d}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}{merge_str}{swa_str} lr={current_lr:.2e}{phase_str}{mine_str}{duo_str}{sd_str}')
        
        if checkpoint_dir:
            ckpt = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'test_acc': test_acc,
            }
            if swa and epoch >= swa_epoch:
                ckpt['swa_model'] = swa_model.state_dict()
                ckpt['swa_acc'] = swa_acc
            torch.save(ckpt, os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch+1:03d}.pt'))

    if swa:
        update_bn(train_loader, swa_model, device=device)
        _, final_acc = evaluate(swa_model, test_loader, device, flatten=flatten, task_type=task_type)
    else:
        _, final_acc = evaluate(model, test_loader, device, flatten=flatten, task_type=task_type)
    return final_acc


def find_config_for_params(
    block_factory_fn,
    classifier_factory_fn, 
    target_params: int,
    head_options: list[int] = [2, 4, 8],
    min_w: int = 16,
    max_w: int = 512,
) -> tuple[int, int]:
    best_config = (min_w, head_options[0])
    best_diff = float('inf')
    
    for num_heads in head_options:
        align = 2 * num_heads
        lo, hi = max(min_w, align), max_w
        
        while lo <= hi:
            mid = (lo + hi) // 2
            mid = max(mid // align * align, align)
            
            try:
                block_factory = block_factory_fn(num_heads)
                model = classifier_factory_fn(block_factory, mid)
                params = sum(p.numel() for p in model.parameters())
            except:
                hi = mid - align
                continue
            
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = (mid, num_heads)
            
            if params < target_params:
                lo = mid + align
            elif params > target_params:
                hi = mid - align
            else:
                return best_config
    
    return best_config


def build_model(model_type, layers, n_classes, seq_len, device, num_channels=4, use_ssm=False, no_mlp=False, conv_position='both', attn_residual=True, merge_mode='lowrank', lowrank_hier=True, kernel_size=17, attn_order='tele,conv,lowrank', target_params=400_000, ml_decoder=False, cross_layer=False, router_top_k=0):
    
    if model_type == 'ripple':
        def block_factory_fn(h):
            return lambda w: None
        def classifier_factory_fn(block_factory, w):
            return RippleClassifier(
                width=w, n_layers=layers, n_classes=n_classes, seq_len=seq_len,
                num_heads=num_channels, order=attn_order, cross_layer=cross_layer, vocab_size=256
            )
        width, _ = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
        return RippleClassifier(
            width=width, n_layers=layers, n_classes=n_classes, seq_len=seq_len,
            num_heads=num_channels, order=attn_order, cross_layer=cross_layer, vocab_size=256
        ).to(device)
    
    if model_type == 'ripple-channel':
        def block_factory_fn(h):
            return lambda w: None
        def classifier_factory_fn(block_factory, w):
            return RippleChannelClassifier(
                width=w, n_channels=layers, n_classes=n_classes, seq_len=seq_len,
                topology=attn_order, num_heads=num_channels, vocab_size=256, router_top_k=router_top_k
            )
        width, _ = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
        return RippleChannelClassifier(
            width=width, n_channels=layers, n_classes=n_classes, seq_len=seq_len,
            topology=attn_order, num_heads=num_channels, vocab_size=256, router_top_k=router_top_k
        ).to(device)
    
    if model_type == 'attention':
        def block_factory_fn(h):
            return lambda w: AttentionBlock(w, num_heads=h, use_ssm=use_ssm)
    elif model_type == 'hier':
        def block_factory_fn(h):
            return lambda w: HierarchicalBlock(w, window_size=kernel_size, num_channels=h, use_ssm=use_ssm, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
    elif model_type == 'sgsb':
        def block_factory_fn(h):
            return lambda w: SGSBBlockND(w, kernel_size=kernel_size, ndim=1, num_channels=h)
    elif model_type == 'conv':
        def block_factory_fn(h):
            return lambda w: ConvBlock(w, kernel_size=kernel_size)
    elif model_type == 'gather':
        def block_factory_fn(h):
            return lambda w: TelephoneAttentionBlock(w, num_heads=h)
    elif model_type == 'flat':
        def block_factory_fn(h):
            return lambda w: None
        def classifier_factory_fn(block_factory, w):
            return FlatRippleClassifierND(embed_dim=w, n_classes=n_classes, iterations=layers, kernel_size=kernel_size, ndim=1, num_channels=num_channels)
        width, _ = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
        return FlatRippleClassifierND(embed_dim=width, n_classes=n_classes, iterations=layers, kernel_size=kernel_size, ndim=1, num_channels=num_channels).to(device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    def classifier_factory_fn(block_factory, w):
        block_fn = lambda: block_factory(w)
        model = SequenceClassifier(block_fn(), w, layers, n_classes, seq_len, ml_decoder=ml_decoder)
        model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
        return model
    
    width, num_heads = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
    block_factory = block_factory_fn(num_heads)
    return classifier_factory_fn(block_factory, width).to(device)


def build_model_2d(model_type, layers, n_classes, img_size, device, num_channels=4, use_ssm=False, conv_position='both', attn_residual=True, merge_mode='lowrank', lowrank_hier=True, kernel_size=7, cross_layer=False, attn_order='tele,conv,lowrank', target_params=400_000, router_top_k=0):
    WIDTH_ATTN = 64
    WIDTH_LOCAL = 64
    WIDTH_HIER = 64
    WIDTH_SGSB = 64
    WIDTH_CONV = 70
    
    if model_type == 'attention':
        block_fn = lambda: AttentionBlock2D(WIDTH_ATTN, num_heads=num_channels, use_ssm=use_ssm)
        width = WIDTH_ATTN
    elif model_type == 'local':
        block_fn = lambda: LocalBlock2D(WIDTH_LOCAL, kernel_size=kernel_size, num_channels=num_channels, use_ssm=use_ssm)
        width = WIDTH_LOCAL
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock2D(WIDTH_HIER, kernel_size=kernel_size, num_channels=num_channels, use_ssm=use_ssm, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
        width = WIDTH_HIER
    elif model_type == 'sgsb':
        block_fn = lambda: SGSBBlockND(WIDTH_SGSB, kernel_size=kernel_size, ndim=2, num_channels=num_channels)
        width = WIDTH_SGSB
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock2D(WIDTH_CONV, kernel_size=kernel_size)
        width = WIDTH_CONV
    elif model_type == 'ripple':
        h, w = img_size
        seq_len = h * w
        def block_factory_fn(heads):
            return lambda width: None
        def classifier_factory_fn(block_factory, width):
            return RippleClassifier(
                width=width, n_layers=layers, n_classes=n_classes, seq_len=seq_len,
                num_heads=num_channels, order=attn_order, cross_layer=cross_layer, embed_2d=(h, w)
            )
        width, _ = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
        return RippleClassifier(
            width=width, n_layers=layers, n_classes=n_classes, seq_len=seq_len,
            num_heads=num_channels, order=attn_order, cross_layer=cross_layer, embed_2d=(h, w)
        ).to(device)
    elif model_type == 'ripple-channel':
        h, w = img_size
        seq_len = h * w
        def block_factory_fn(heads):
            return lambda width: None
        def classifier_factory_fn(block_factory, width):
            return RippleChannelClassifier(
                width=width, n_channels=layers, n_classes=n_classes, seq_len=seq_len,
                topology=attn_order, num_heads=num_channels, embed_2d=(h, w), router_top_k=router_top_k
            )
        width, _ = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
        return RippleChannelClassifier(
            width=width, n_channels=layers, n_classes=n_classes, seq_len=seq_len,
            topology=attn_order, num_heads=num_channels, embed_2d=(h, w), router_top_k=router_top_k
        ).to(device)
    elif model_type == 'flat':
        def block_factory_fn(h):
            return lambda w: None
        def classifier_factory_fn(block_factory, w):
            return FlatRippleClassifierND(
                embed_dim=w, n_classes=n_classes, iterations=layers,
                kernel_size=kernel_size, ndim=2, num_channels=num_channels,
            )
        width, _ = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
        return FlatRippleClassifierND(
            embed_dim=width, n_classes=n_classes, iterations=layers,
            kernel_size=kernel_size, ndim=2, num_channels=num_channels,
        ).to(device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    model = ImageClassifier(block_fn(), width, layers, n_classes, img_size)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def build_model_3d(model_type, layers, n_classes, vol_size, device, num_channels=4, conv_position='both', attn_residual=True, merge_mode='lowrank', lowrank_hier=True, kernel_size=5):
    WIDTH_ATTN = 48
    WIDTH_LOCAL = 48
    WIDTH_HIER = 48
    WIDTH_SGSB = 48
    WIDTH_CONV = 52
    
    if model_type == 'attention':
        block_fn = lambda: AttentionBlock3D(WIDTH_ATTN, num_heads=num_channels)
        width = WIDTH_ATTN
    elif model_type == 'local':
        block_fn = lambda: LocalBlock3D(WIDTH_LOCAL, kernel_size=kernel_size, num_channels=num_channels)
        width = WIDTH_LOCAL
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock3D(WIDTH_HIER, window_size=kernel_size, num_channels=num_channels, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
        width = WIDTH_HIER
    elif model_type == 'sgsb':
        block_fn = lambda: SGSBBlockND(WIDTH_SGSB, kernel_size=kernel_size, ndim=3, num_channels=num_channels)
        width = WIDTH_SGSB
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock3D(WIDTH_CONV, kernel_size=kernel_size)
        width = WIDTH_CONV
    elif model_type == 'ripple':
        return RippleClassifierND(
            embed_dim=WIDTH_SGSB, n_classes=n_classes, n_layers=layers,
            kernel_size=kernel_size, ndim=3, num_channels=num_channels,
        ).to(device)
    elif model_type == 'flat':
        return FlatRippleClassifierND(
            embed_dim=WIDTH_SGSB, n_classes=n_classes, iterations=layers,
            kernel_size=kernel_size, ndim=3, num_channels=num_channels,
        ).to(device)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    model = VolumeClassifier(block_fn(), width, layers, n_classes, vol_size)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def build_model_lm(model_type, layers, vocab_size, seq_len, device, num_channels=4, use_ssm=False, conv_position='both', attn_residual=True, merge_mode='lowrank', lowrank_hier=True, kernel_size=17, cross_layer=False, attn_order='tele,conv,lowrank', target_params=400_000):
    WIDTH_ATTN = 128
    WIDTH_HIER = 128
    WIDTH_SGSB = 128
    WIDTH_CONV = 140
    WIDTH_GATHER = 128

    if model_type == 'ripple':
        def block_factory_fn(h):
            return lambda w: None
        def classifier_factory_fn(block_factory, w):
            return RippleClassifier(
                width=w, n_layers=layers, n_classes=vocab_size, seq_len=seq_len,
                num_heads=num_channels, order=attn_order, cross_layer=cross_layer, vocab_size=vocab_size
            )
        width, _ = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
        return RippleClassifier(
            width=width, n_layers=layers, n_classes=vocab_size, seq_len=seq_len,
            num_heads=num_channels, order=attn_order, cross_layer=cross_layer, vocab_size=vocab_size
        ).to(device)

    if model_type == 'attention':
        block_fn = lambda: AttentionBlock(WIDTH_ATTN, num_heads=num_channels, use_ssm=use_ssm)
        width = WIDTH_ATTN
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock(WIDTH_HIER, window_size=kernel_size, num_channels=num_channels, use_ssm=use_ssm, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
        width = WIDTH_HIER
    elif model_type == 'sgsb':
        block_fn = lambda: SGSBBlockND(WIDTH_SGSB, kernel_size=kernel_size, ndim=1, num_channels=num_channels)
        width = WIDTH_SGSB
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock(WIDTH_CONV, kernel_size=kernel_size)
        width = WIDTH_CONV
    elif model_type == 'gather':
        block_fn = lambda: TelephoneAttentionBlock(WIDTH_GATHER, num_heads=num_channels)
        width = WIDTH_GATHER
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    model = SequenceLM(block_fn(), width, layers, vocab_size, seq_len)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def build_model_audio(model_type, layers, n_classes, seq_len, device, num_channels=4, use_ssm=False, conv_position='both', attn_residual=True, merge_mode='lowrank', lowrank_hier=True, kernel_size=17, attn_order='tele,conv,lowrank', target_params=400_000):
    
    if model_type == 'attention':
        def block_factory_fn(h):
            return lambda w: AttentionBlock(w, num_heads=h, use_ssm=use_ssm)
    elif model_type == 'hier':
        def block_factory_fn(h):
            return lambda w: HierarchicalBlock(w, window_size=kernel_size, num_channels=h, use_ssm=use_ssm, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
    elif model_type == 'sgsb':
        def block_factory_fn(h):
            return lambda w: SGSBBlockND(w, kernel_size=kernel_size, ndim=1, num_channels=h)
    elif model_type == 'conv':
        def block_factory_fn(h):
            return lambda w: ConvBlock(w, kernel_size=kernel_size)
    elif model_type == 'gather':
        def block_factory_fn(h):
            return lambda w: TelephoneAttentionBlock(w, num_heads=h)
    elif model_type == 'ripple':
        def block_factory_fn(h):
            return lambda w: RippleAttentionBlock(w, num_heads=h, order=attn_order)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    def classifier_factory_fn(block_factory, w):
        block_fn = lambda: block_factory(w)
        model = AudioClassifier(block_fn(), w, layers, n_classes, seq_len)
        model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
        return model
    
    width, num_heads = find_config_for_params(block_factory_fn, classifier_factory_fn, target_params)
    block_factory = block_factory_fn(num_heads)
    return classifier_factory_fn(block_factory, width).to(device)


class PreloadedDataset:
    def __init__(self, loader, device, augment=None):
        print('Preloading dataset to CPU...')
        all_x, all_y = [], []
        for x, y in loader:
            all_x.append(x)
            all_y.append(y)
        self.x = torch.cat(all_x, dim=0)
        self.y = torch.cat(all_y, dim=0)
        if augment:
            print('  Applying augmentation...')
            self.x = augment(self.x)
        self.batch_size = loader.batch_size
        self.device = device
        print(f'  Loaded {len(self.x)} samples ({self.x.element_size() * self.x.numel() / 1e9:.2f} GB)')
    
    def __iter__(self):
        indices = torch.randperm(len(self.x))
        for i in range(0, len(self.x), self.batch_size):
            idx = indices[i:i+self.batch_size]
            yield self.x[idx].to(self.device), self.y[idx].to(self.device)
    
    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size


def make_affine_augment(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), pack_rgb=False):
    pack = PackRGB() if pack_rgb else None
    def augment(x):
        B = x.shape[0]
        angle = (torch.rand(B) * 2 - 1) * degrees * (3.14159 / 180)
        tx = (torch.rand(B) * 2 - 1) * translate[0]
        ty = (torch.rand(B) * 2 - 1) * translate[1]
        s = torch.rand(B) * (scale[1] - scale[0]) + scale[0]
        
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros(B, 2, 3)
        theta[:, 0, 0] = s * cos_a
        theta[:, 0, 1] = -s * sin_a
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = s * sin_a
        theta[:, 1, 1] = s * cos_a
        theta[:, 1, 2] = ty
        
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
        if pack:
            x = torch.stack([pack(xi) for xi in x])
        return x
    return augment


def make_crop_flip_augment(padding=4, pack_rgb=False):
    pack = PackRGB() if pack_rgb else None
    def augment(x):
        B, C, H, W = x.shape
        x = F.pad(x, [padding]*4, mode='reflect')
        top = torch.randint(0, 2*padding + 1, (B,))
        left = torch.randint(0, 2*padding + 1, (B,))
        crops = []
        for i in range(B):
            crops.append(x[i:i+1, :, top[i]:top[i]+H, left[i]:left[i]+W])
        x = torch.cat(crops, dim=0)
        flip_mask = torch.rand(B) > 0.5
        x[flip_mask] = x[flip_mask].flip(-1)
        if pack:
            x = torch.stack([pack(xi) for xi in x])
        return x
    return augment


def load_dataset(name, batch_size, mode_3d=False, seq_len_override=None, num_workers=0, no_augment=False):
    task_type = 'classification'
    vocab_size = None

    if name in TASKS:
        task_type = 'lm'
        sl = seq_len_override or 256
        train_loader, test_loader, config = load_task(
            name, seq_len=sl, num_train=100000, num_test=10000, batch_size=batch_size
        )
        vocab_size = config.get('vocab_size', 2)
        return train_loader, test_loader, vocab_size, sl, None, task_type

    if name == 'speech':
        sl = seq_len_override or 16000
        train_data = SpeechCommandsDataset('data', 'training', seq_len=sl)
        test_data = SpeechCommandsDataset('data', 'testing', seq_len=sl)
        n_classes = len(SPEECHCOMMANDS_LABELS)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader, n_classes, sl, None, 'audio'

    if name == 'mnist':
        if no_augment:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('data', train=True, download=True, transform=train_transform)
        test_data = datasets.MNIST('data', train=False, download=True, transform=test_transform)
        n_classes, seq_len, img_size = 10, 784, (28, 28)

    elif name == 'fashion':
        if no_augment:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_data = datasets.FashionMNIST('data', train=True, download=True, transform=train_transform)
        test_data = datasets.FashionMNIST('data', train=False, download=True, transform=test_transform)
        n_classes, seq_len, img_size = 10, 784, (28, 28)

    elif name == 'cifar10':
        if no_augment:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                PackRGB(),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            PackRGB(),
        ])
        if mode_3d:
            n_classes, seq_len, img_size = 10, 3072, (32, 32, 3)
        else:
            n_classes, seq_len, img_size = 10, 1024, (32, 32)
        train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10('data', train=False, download=True, transform=test_transform)

    elif name == 'cifar100':
        if no_augment:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                PackRGB(),
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            PackRGB(),
        ])
        if mode_3d:
            n_classes, seq_len, img_size = 100, 3072, (32, 32, 3)
        else:
            n_classes, seq_len, img_size = 100, 1024, (32, 32)
        train_data = datasets.CIFAR100('data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR100('data', train=False, download=True, transform=test_transform)

    else:
        raise ValueError(f'Unknown dataset: {name}. Available: {list(TASKS.keys()) + ["speech", "mnist", "fashion", "cifar10", "cifar100"]}')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, n_classes, seq_len, img_size, task_type


def main():
    all_datasets = ['mnist', 'fashion', 'cifar10', 'cifar100', 'speech'] + list(TASKS.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='mnist', help=f'Available: {all_datasets}')
    parser.add_argument('--device', type=str, choices=['cpu', 'mps', 'cuda', 'auto'], default='auto')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers (0 for reproducibility)')
    parser.add_argument('--seq-len', type=int, default=None, help='Override sequence length')
    parser.add_argument('--2d', dest='mode_2d', action='store_true', help='Use 2D models (image-native)')
    parser.add_argument('--3d', dest='mode_3d', action='store_true', help='Use 3D models (RGB as depth)')
    parser.add_argument('--cosine-start', type=float, default=0.1, help='Fraction of post-warmup before cosine decay (default: 0.1)')
    parser.add_argument('--swa', action='store_true', help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa-start', type=float, default=0.8, help='Fraction of training before SWA kicks in (default: 0.8)')
    parser.add_argument('--swa-lr', type=float, default=1e-5, help='Learning rate for SWA phase (default: 1e-5)')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory to save checkpoints (default: no checkpoints)')
    parser.add_argument('--channels', type=int, default=4, help='Number of attention channels/heads (default: 4)')
    parser.add_argument('--kernel-size', type=int, default=None, help='Kernel/window size for attention (default: 17 for 1D, 7 for 2D, 5 for 3D)')
    parser.add_argument('--ssm', action='store_true', help='Add SSM block after attention')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision (fp16)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for kernel fusion')
    parser.add_argument('--profile', action='store_true', help='Profile 1 batch after 10 warmup, then exit')
    parser.add_argument('--no-preload', dest='preload', action='store_false', help='Disable preloading dataset to memory')
    parser.set_defaults(preload=True)
    parser.add_argument('--hard-mining', action='store_true', help='Enable hard example mining (reweight samples by previous epoch loss)')
    parser.add_argument('--hard-start', type=float, default=0.5, help='Hard mining start %% (default: 0.5 = 50%%)')
    parser.add_argument('--hard-end', type=float, default=0.05, help='Hard mining end %% (default: 0.05 = 5%%)')
    parser.add_argument('--first-epoch-pct', type=float, default=None, help='First epoch: only backprop hardest N%% of samples (e.g. 0.3 for 30%%)')
    parser.add_argument('--wtf-mode', action='store_true', help='WTF mode: gradient ascent on easy samples, descent on hard (per-batch median split)')
    parser.add_argument('--duo', action='store_true', help='Duo mode: train two models with inverse curriculum (uses --hard-start/--hard-end), merge at inference')
    parser.add_argument('--duo-merge', type=str, default='mean', choices=DuoModel.MERGE_STRATEGIES, help='Duo merge strategy (default: mean)')
    parser.add_argument('--conv-position', type=str, default='both', choices=['pre', 'post', 'both'], help='Where to run conv relative to attention (default: both)')
    parser.add_argument('--merge-mode', type=str, default='lowrank', choices=['gate', 'learned', 'lowrank'], help='Merge mode for residuals (default: lowrank)')
    parser.add_argument('--lowrank-hier', action='store_true', default=True, help='Use low-rank full attention instead of windowed attention at each hierarchy level (default: True)')
    parser.add_argument('--no-attn-residual', action='store_true', help='Disable attention residual connection')
    parser.add_argument('--attn-order', type=str, default='tele,conv,lowrank', help='Order of attention layers for ripple model (default: tele,conv,lowrank)')
    parser.add_argument('--target-params', type=int, default=400_000, help='Target total model params (default: 400000)')
    parser.add_argument('--ml-decoder', action='store_true', help='Use ML-Decoder classification head instead of GAP+Linear')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor (default: 0.1, 0 to disable)')
    parser.add_argument('--cross-layer', action='store_true', help='Enable cross-layer attention for ripple model (accumulates layer history)')
    parser.add_argument('--ponder', action='store_true', help='Wrap model with learned internal loss (ML3-style)')
    parser.add_argument('--ponder-meta-lr', type=float, default=3e-4, help='Meta learning rate for L_internal')
    parser.add_argument('--ponder-reward-scale', type=float, default=100.0, help='Reward multiplier for REINFORCE')
    parser.add_argument('--ponder-min-supervised', type=float, default=0.0, help='Min CE supervision ratio for L_internal (0=full RL, 0.2=always 20%% CE)')

    parser.add_argument('--ponder-meta-warmup', type=int, default=1, help='Epochs of pure CE supervision for L_internal (default: 1)')
    parser.add_argument('--ponder-meta-wean', type=int, default=1, help='Epochs to wean L_internal from CE to RL (default: 1)')
    parser.add_argument('--with-router', type=int, default=0, help='Top-K channel routing for ripple-channel (0=disabled)')
    parser.add_argument('--self-distill', action='store_true', help='Enable self-distillation: each epoch distills from previous epoch model')
    parser.add_argument('--distill-alpha', type=float, default=0.5, help='Distillation loss weight (default: 0.5)')
    parser.add_argument('--distill-temp', type=float, default=2.0, help='Distillation temperature (default: 2.0)')
    parser.add_argument('--distill-merge', type=str, default=None, choices=['ema', 'slerp', 'lerp'], help='Merge teacher/student weights each epoch (default: disabled)')
    parser.add_argument('--distill-merge-alpha', type=float, default=0.5, help='Merge ratio: 0=all teacher, 1=all student (default: 0.5)')
    parser.add_argument('--distill-from-merged', action='store_true', help='Use merged model as teacher for next epoch distillation')
    parser.add_argument('--distill-ensemble', action='store_true', help='Ensemble distillation: accumulate logits across epochs, weight by inverse loss')
    parser.add_argument('--distill-ensemble-k', type=int, default=3, help='Top-K best epochs per sample for ensemble (default: 3)')

    args = parser.parse_args()

    def seed_everything(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # For full determinism (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    seed_everything(args.seed)

    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}')

    train_loader, test_loader, n_classes_or_vocab, seq_len, img_size, task_type = load_dataset(
        args.dataset, args.batch_size, args.mode_3d, args.seq_len, num_workers=args.workers,
        no_augment=args.preload
    )

    if args.preload:
        if args.dataset in ('mnist', 'fashion'):
            train_aug = make_affine_augment(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
        elif args.dataset in ('cifar10', 'cifar100') and not args.mode_3d:
            train_aug = make_affine_augment(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), pack_rgb=True)
        elif args.dataset in ('cifar10', 'cifar100') and args.mode_3d:
            train_aug = make_crop_flip_augment(padding=4)
        else:
            train_aug = None
        train_loader = PreloadedDataset(train_loader, device, augment=train_aug)
        test_loader = PreloadedDataset(test_loader, device, augment=None)

    attn_residual = not args.no_attn_residual
    
    if task_type == 'lm':
        vocab_size = n_classes_or_vocab
        kernel_size = args.kernel_size or 17
        all_model_types = ['attention', 'sgsb', 'conv', 'gather', 'ripple']
        builder = lambda mt: build_model_lm(mt, args.layers, vocab_size, seq_len, device, args.channels, args.ssm, args.conv_position, attn_residual, args.merge_mode, args.lowrank_hier, kernel_size, args.cross_layer, args.attn_order, args.target_params)
        shape_str = f'seq_len={seq_len}, vocab={vocab_size}'
        flatten = False
        print(f'Task type: Language Modeling (token prediction)')
    elif task_type == 'audio':
        n_classes = n_classes_or_vocab
        kernel_size = args.kernel_size or 17
        all_model_types = ['attention', 'sgsb', 'ripple', 'flat', 'conv', 'gather']
        builder = lambda mt: build_model_audio(mt, args.layers, n_classes, seq_len, device, args.channels, args.ssm, args.conv_position, attn_residual, args.merge_mode, args.lowrank_hier, kernel_size, args.attn_order, args.target_params)
        shape_str = f'seq_len={seq_len}, n_classes={n_classes}'
        flatten = False
        print(f'Task type: Audio Classification')
    elif args.mode_3d:
        n_classes = n_classes_or_vocab
        kernel_size = args.kernel_size or 5
        all_model_types = ['attention', 'local', 'sgsb', 'ripple', 'flat', 'conv']
        builder = lambda mt: build_model_3d(mt, args.layers, n_classes, img_size, device, args.channels, args.conv_position, attn_residual, args.merge_mode, args.lowrank_hier, kernel_size)
        shape_str = f'vol_size={img_size}'
        flatten = False
        print(f'Task type: 3D Classification (SSM not supported)')
    elif args.mode_2d:
        n_classes = n_classes_or_vocab
        kernel_size = args.kernel_size or 7
        all_model_types = ['attention', 'local', 'sgsb', 'ripple', 'flat', 'conv']
        builder = lambda mt: build_model_2d(mt, args.layers, n_classes, img_size, device, args.channels, args.ssm, args.conv_position, attn_residual, args.merge_mode, args.lowrank_hier, kernel_size, args.cross_layer, args.attn_order, args.target_params, args.with_router)
        shape_str = f'img_size={img_size}'
        flatten = args.model not in ('ripple', 'ripple-channel')
        print(f'Task type: 2D Classification')
    else:
        n_classes = n_classes_or_vocab
        kernel_size = args.kernel_size or 17
        all_model_types = ['attention', 'sgsb', 'ripple', 'ripple-channel', 'flat', 'conv', 'gather']
        builder = lambda mt: build_model(mt, args.layers, n_classes, seq_len, device, args.channels, args.ssm, False, args.conv_position, attn_residual, args.merge_mode, args.lowrank_hier, kernel_size, args.attn_order, args.target_params, args.ml_decoder, args.cross_layer, args.with_router)
        shape_str = f'seq_len={seq_len}'
        flatten = True
        print(f'Task type: 1D Classification')

    model_types = all_model_types if args.model == 'all' else [args.model]

    print(f'\nModel parameters ({args.layers} layers, {shape_str}):')
    for mt in model_types:
        model = builder(mt)
        params = count_params(model)
        class_name = type(model).__name__
        cross_layer_str = ' [cross-layer]' if (mt == 'ripple' and args.cross_layer) else ''
        if args.duo:
            print(f'  {mt:12s}: {params:,} params x2 (duo) = {params*2:,} total ({class_name}){cross_layer_str}')
        else:
            print(f'  {mt:12s}: {params:,} params ({class_name}){cross_layer_str}')

    results = {mt: [] for mt in model_types}

    for run in range(args.runs):
        seed = args.seed + run * 42
        seed_everything(seed)

        if args.runs > 1:
            print(f'\n{"="*60}')
            print(f'Run {run+1}/{args.runs} (seed={seed})')
            print('='*60)

        for mt in model_types:
            seed_everything(seed)
            
            if args.duo:
                # Create two independent models with inverse curriculum
                model_hard = builder(mt)
                seed_everything(seed + 1)  # Different init for model_easy
                model_easy = builder(mt)
                model = DuoModel(model_hard, model_easy, merge=args.duo_merge)
                print(f'\nTraining {mt} [DUO: curriculum {int(args.hard_start*100)}%→{int(args.hard_end*100)}%, merge={args.duo_merge}]...')
            else:
                model = builder(mt)
                print(f'\nTraining {mt}...')
            
            # RippleClassifierND/FlatRippleClassifierND in 2D/3D mode expects spatial input, not flattened
            # In 1D mode, ripple/flat needs flattened 1D sequence input
            model_flatten = flatten and not (mt in ('ripple', 'ripple-channel', 'flat') and (args.mode_2d or args.mode_3d))
            
            if args.ponder and not args.duo:
                model = PonderWrapper(model)
                ponder_params = sum(p.numel() for p in model.l_internal.parameters())
                print(f'  Ponder: +{ponder_params:,} params (L_internal)')

            if args.compile:
                model = torch.compile(model)
            
            if args.profile:
                print(f'\nProfiling {mt}...')
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
                
                # Get a batch
                inputs, labels = next(iter(train_loader))
                inputs = inputs.to(device)
                labels = labels.to(device)
                if model_flatten:
                    inputs = inputs.view(inputs.size(0), -1)
                elif not model_flatten and inputs.dim() == 4:
                    inputs = inputs.squeeze(1)
                
                # Warmup
                import time
                print('Warming up 100 batches...')
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(100):
                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                print(f'Warmup: {100/(t1-t0):.2f} batches/sec ({(t1-t0)/100*1000:.1f} ms/batch)')
                
                # Profile
                print('Profiling 1 batch...')
                from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True,
                ) as prof:
                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
                print(f'\nMemory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
                
                # Export for chrome://tracing
                prof.export_chrome_trace(f'profile_{mt}.json')
                print(f'Trace exported to profile_{mt}.json')
                print('Open chrome://tracing and load the JSON file for detailed view')
                
                continue  # Skip training for this model
            
            if args.ponder and not args.duo:
                ponder_config = PonderTrainConfig(
                    meta_lr=args.ponder_meta_lr,
                    model_lr=args.lr,
                    epochs=args.epochs,
                    reward_scale=args.ponder_reward_scale,
                    meta_min_supervised=args.ponder_min_supervised,

                    meta_warmup_epochs=args.ponder_meta_warmup,
                    meta_wean_epochs=args.ponder_meta_wean,
                )
                trainer = PonderTrainer(
                    model, train_loader, test_loader, device, ponder_config,
                    flatten_input=model_flatten,
                    squeeze_channel=not model_flatten,
                )
                acc = trainer.train()
            else:
                acc = train_model(
                    model, train_loader, test_loader, device, args.epochs, args.lr,
                    cosine_start=args.cosine_start, swa=args.swa, swa_start=args.swa_start,
                    swa_lr=args.swa_lr, hard_mining=args.hard_mining, hard_start=args.hard_start,
                    hard_end=args.hard_end, first_epoch_pct=args.first_epoch_pct,
                    wtf_mode=args.wtf_mode, checkpoint_dir=args.checkpoint_dir, model_name=f'{mt}_run{run}',
                    verbose=(args.runs == 1), flatten=model_flatten, task_type=task_type, use_amp=args.amp,
                    label_smoothing=args.label_smoothing,
                    self_distill=args.self_distill, distill_alpha=args.distill_alpha, distill_temp=args.distill_temp,
                    distill_merge=args.distill_merge, distill_merge_alpha=args.distill_merge_alpha,
                    distill_from_merged=args.distill_from_merged,
                    distill_ensemble=args.distill_ensemble, distill_ensemble_k=args.distill_ensemble_k
                )
            results[mt].append(acc)
            print(f'{mt}: {acc:.4f}')

    print(f'\n{"="*60}')
    print('Final Results')
    print('='*60)

    for mt in model_types:
        accs = results[mt]
        mean_acc = sum(accs) / len(accs)
        if len(accs) > 1:
            std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
            print(f'{mt:12s}: {mean_acc:.4f} ± {std_acc:.4f} (n={len(accs)})')
        else:
            print(f'{mt:12s}: {mean_acc:.4f}')


if __name__ == '__main__':
    main()
