#!/usr/bin/env python3

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import torchaudio

from heuristic_secrets.models.scatter_attention import (
    HierarchicalLocalAttention,
    HierarchicalLocalAttentionND,
    LocalAttentionND,
    RMSNorm,
    apply_rope,
    sinusoidal_pos_embed_nd,
)
from heuristic_secrets.models.backbone import SSMMixer3
from heuristic_secrets.models.backbone2d import SSMBlock3_2d
from heuristic_secrets.data.synthetic import load_task, TASKS


SPEECHCOMMANDS_LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
]


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
    def __init__(self, width: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = width * mult
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class SDPAttention(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(width, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
        q = apply_rope(q.transpose(1, 2)).transpose(1, 2)
        k = apply_rope(k.transpose(1, 2)).transpose(1, 2)
        
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out(out)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.1, use_ssm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention(width, num_heads, dropout)
        self.attn_norm = RMSNorm(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm_norm = RMSNorm(width)
            self.ssm = SSMMixer3(width, n_heads=num_heads, use_conv=False, dropout=dropout)
            self.ssm_out_norm = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.attn(self.norm1(x)))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.ssm_norm(x))
            x = x + self.ssm_out_norm(ssm_out)
        x = x + self.mlp(self.norm2(x))
        return x


class HierarchicalBlock(nn.Module):
    def __init__(self, width: int, window_size: int = 17, num_channels: int = 4, mlp_mult: int = 4, dropout: float = 0.1, use_ssm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.hier_attn = HierarchicalLocalAttention(width, window_size, num_channels)
        self.attn_norm = RMSNorm(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm_norm = RMSNorm(width)
            self.ssm = SSMMixer3(width, n_heads=num_channels, use_conv=False, dropout=dropout)
            self.ssm_out_norm = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.hier_attn(self.norm1(x)))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.ssm_norm(x))
            x = x + self.ssm_out_norm(ssm_out)
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int = 17, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x).transpose(1, 2)
        h = self.pointwise(self.depthwise(h)).transpose(1, 2)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SDPAttention2D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(width, width, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        L = H * W
        
        x_flat = x.reshape(B, L, C)
        qkv = self.qkv(x_flat).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
        q = apply_rope(q.transpose(1, 2)).transpose(1, 2)
        k = apply_rope(k.transpose(1, 2)).transpose(1, 2)
        
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        out = out.transpose(1, 2).reshape(B, H, W, C)
        return self.out(out)


class AttentionBlock2D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.1, use_ssm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention2D(width, num_heads, dropout)
        self.attn_norm = RMSNorm(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm_norm = RMSNorm(width)
            self.ssm = SSMBlock3_2d(width, n_heads=num_heads, use_conv=False, dropout=dropout)
            self.ssm_out_norm = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.attn(self.norm1(x)))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.ssm_norm(x))
            x = x + self.ssm_out_norm(ssm_out)
        x = x + self.mlp(self.norm2(x))
        return x


class LocalBlock2D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 7, num_channels: int = 4, mlp_mult: int = 4, dropout: float = 0.1, use_ssm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.local_attn = LocalAttentionND(width, kernel_size, ndim=2, num_channels=num_channels)
        self.attn_norm = RMSNorm(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm_norm = RMSNorm(width)
            self.ssm = SSMBlock3_2d(width, n_heads=num_channels, use_conv=False, dropout=dropout)
            self.ssm_out_norm = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.local_attn(self.norm1(x)))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.ssm_norm(x))
            x = x + self.ssm_out_norm(ssm_out)
        x = x + self.mlp(self.norm2(x))
        return x


class HierarchicalBlock2D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 7, num_channels: int = 4, mlp_mult: int = 4, dropout: float = 0.1, use_ssm: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.hier_attn = HierarchicalLocalAttentionND(width, kernel_size, ndim=2, num_channels=num_channels)
        self.attn_norm = RMSNorm(width)
        self.use_ssm = use_ssm
        if use_ssm:
            self.ssm_norm = RMSNorm(width)
            self.ssm = SSMBlock3_2d(width, n_heads=num_channels, use_conv=False, dropout=dropout)
            self.ssm_out_norm = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.hier_attn(self.norm1(x)))
        if self.use_ssm:
            ssm_out, _ = self.ssm(self.ssm_norm(x))
            x = x + self.ssm_out_norm(ssm_out)
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock2D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 7, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.depthwise = nn.Conv2d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        h = self.norm1(x).permute(0, 3, 1, 2)
        h = self.pointwise(self.depthwise(h)).permute(0, 2, 3, 1)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SequenceClassifier(nn.Module):
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


class AttentionBlock3D(nn.Module):
    def __init__(self, width: int, num_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = SDPAttention2D(width, num_heads, dropout)
        self.attn_norm = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        x_flat = x.reshape(B, H * W * D, C)
        attn_out = self.attn.qkv(self.norm1(x_flat).reshape(B, H*W*D, C))
        qkv = attn_out.reshape(B, H*W*D, 3, self.attn.num_heads, self.attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.attn.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.attn.k_norm(k.transpose(1, 2)).transpose(1, 2)
        q = apply_rope(q.transpose(1, 2)).transpose(1, 2)
        k = apply_rope(k.transpose(1, 2)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = self.attn.out(out.transpose(1, 2).reshape(B, H, W, D, C))
        x = x + self.attn_norm(out)
        x = x + self.mlp(self.norm2(x))
        return x


class LocalBlock3D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 5, num_channels: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.local_attn = LocalAttentionND(width, kernel_size, ndim=3, num_channels=num_channels)
        self.attn_norm = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.local_attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class HierarchicalBlock3D(nn.Module):
    def __init__(self, width: int, window_size: int = 5, num_channels: int = 4, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.hier_attn = HierarchicalLocalAttentionND(width, window_size, ndim=3, num_channels=num_channels, poolable_dims=(0, 1))
        self.attn_norm = RMSNorm(width)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_norm(self.hier_attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock3D(nn.Module):
    def __init__(self, width: int, kernel_size: int = 5, mlp_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.depthwise = nn.Conv3d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.pointwise = nn.Conv3d(width, width, 1)
        self.norm2 = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, C = x.shape
        h = self.norm1(x).permute(0, 4, 1, 2, 3)
        h = self.pointwise(self.depthwise(h)).permute(0, 2, 3, 4, 1)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, optimizer, device, scheduler=None, flatten=True, task_type='classification', wtf_mode=False, hard_pct=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    desc = "Train"
    if wtf_mode:
        desc += " [WTF]"
    if hard_pct is not None:
        desc += f" [H{int(hard_pct*100)}%]"
    pbar = tqdm(loader, desc=desc, leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if flatten and task_type == 'classification':
            inputs = inputs.view(inputs.size(0), -1)

        optimizer.zero_grad()
        logits = model(inputs)

        if task_type == 'lm':
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            per_token_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction='none')
            if wtf_mode:
                mask = labels_flat != -100
                valid_losses = per_token_loss[mask]
                # Normalize weights to [-1, +10] centered on median
                # Below median: [min, median] -> [-1, 0] (detrain easy)
                # Above median: [median, max] -> [0, +10] (push hard)
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
            correct += (preds[mask] == labels_flat[mask]).sum().item()
            total += mask.sum().item()
        else:
            per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
            if wtf_mode:
                # Normalize weights to [-1, +10] centered on median
                # Below median: [min, median] -> [-1, 0] (detrain easy)
                # Above median: [median, max] -> [0, +10] (push hard)
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
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}", acc=f"{correct/max(total,1):.4f}")

    return total_loss / len(loader), correct / max(total, 1)


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


def train_model(model, train_loader, test_loader, device, epochs, lr, warmup_epochs=2, cosine_start=0.1, swa=False, swa_start=0.8, swa_lr=1e-5, hard_mining=False, hard_start=0.5, hard_end=0.05, first_epoch_pct=None, wtf_mode=False, checkpoint_dir=None, model_name='model', verbose=True, flatten=True, task_type='classification'):
    import os
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    
    current_loader = train_loader
    train_dataset = train_loader.dataset
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
        # First epoch uses first_epoch_pct if set, otherwise use cosine hard mining
        if epoch == 0 and first_epoch_pct is not None:
            hard_pct = first_epoch_pct
        else:
            hard_pct = get_hard_pct(epoch) if hard_mining else None
        train_loss, train_acc = train_epoch(
            model, current_loader, optimizer, device, active_scheduler, 
            flatten=flatten, task_type=task_type, wtf_mode=wtf_mode, hard_pct=hard_pct
        )
        
        if use_swa_sched:
            swa_model.update_parameters(model)
        
        test_loss, test_acc = evaluate(model, test_loader, device, flatten=flatten, task_type=task_type)
        
        swa_acc = None
        if use_swa_sched:
            update_bn(train_loader, swa_model, device=device)
            _, swa_acc = evaluate(swa_model, test_loader, device, flatten=flatten, task_type=task_type)
        
        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            swa_str = f' swa_acc={swa_acc:.4f}' if swa_acc is not None else ''
            phase_str = ' [SWA]' if in_swa_phase else ''
            mine_str = ' [HEM]' if hard_mining else ''
            print(f'Epoch {epoch+1:2d}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}{swa_str} lr={current_lr:.2e}{phase_str}{mine_str}')
        
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


def build_model(model_type, layers, n_classes, seq_len, device, num_channels=4, use_ssm=False):
    WIDTH_ATTN = 64
    WIDTH_HIER = 64
    WIDTH_CONV = 70
    
    if model_type == 'attention':
        block_fn = lambda: AttentionBlock(WIDTH_ATTN, num_heads=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_ATTN
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock(WIDTH_HIER, window_size=17, num_channels=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_HIER
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock(WIDTH_CONV, kernel_size=17, mlp_mult=4)
        width = WIDTH_CONV
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    model = SequenceClassifier(block_fn(), width, layers, n_classes, seq_len)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def build_model_2d(model_type, layers, n_classes, img_size, device, num_channels=4, use_ssm=False):
    WIDTH_ATTN = 64
    WIDTH_LOCAL = 64
    WIDTH_HIER = 64
    WIDTH_CONV = 70
    
    if model_type == 'attention':
        block_fn = lambda: AttentionBlock2D(WIDTH_ATTN, num_heads=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_ATTN
    elif model_type == 'local':
        block_fn = lambda: LocalBlock2D(WIDTH_LOCAL, kernel_size=7, num_channels=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_LOCAL
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock2D(WIDTH_HIER, kernel_size=7, num_channels=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_HIER
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock2D(WIDTH_CONV, kernel_size=7, mlp_mult=4)
        width = WIDTH_CONV
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    model = ImageClassifier(block_fn(), width, layers, n_classes, img_size)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def build_model_3d(model_type, layers, n_classes, vol_size, device, num_channels=4):
    WIDTH_ATTN = 48
    WIDTH_LOCAL = 48
    WIDTH_HIER = 48
    WIDTH_CONV = 52
    
    if model_type == 'attention':
        block_fn = lambda: AttentionBlock3D(WIDTH_ATTN, num_heads=num_channels, mlp_mult=4)
        width = WIDTH_ATTN
    elif model_type == 'local':
        block_fn = lambda: LocalBlock3D(WIDTH_LOCAL, kernel_size=5, num_channels=num_channels, mlp_mult=4)
        width = WIDTH_LOCAL
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock3D(WIDTH_HIER, window_size=5, num_channels=num_channels, mlp_mult=4)
        width = WIDTH_HIER
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock3D(WIDTH_CONV, kernel_size=5, mlp_mult=4)
        width = WIDTH_CONV
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    model = VolumeClassifier(block_fn(), width, layers, n_classes, vol_size)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def build_model_lm(model_type, layers, vocab_size, seq_len, device, num_channels=4, use_ssm=False):
    WIDTH_ATTN = 128
    WIDTH_HIER = 128
    WIDTH_CONV = 140

    if model_type == 'attention':
        block_fn = lambda: AttentionBlock(WIDTH_ATTN, num_heads=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_ATTN
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock(WIDTH_HIER, window_size=17, num_channels=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_HIER
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock(WIDTH_CONV, kernel_size=17, mlp_mult=4)
        width = WIDTH_CONV
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    model = SequenceLM(block_fn(), width, layers, vocab_size, seq_len)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def build_model_audio(model_type, layers, n_classes, seq_len, device, num_channels=4, use_ssm=False):
    WIDTH_ATTN = 64
    WIDTH_HIER = 64
    WIDTH_CONV = 70

    if model_type == 'attention':
        block_fn = lambda: AttentionBlock(WIDTH_ATTN, num_heads=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_ATTN
    elif model_type == 'hier':
        block_fn = lambda: HierarchicalBlock(WIDTH_HIER, window_size=17, num_channels=num_channels, mlp_mult=4, use_ssm=use_ssm)
        width = WIDTH_HIER
    elif model_type == 'conv':
        block_fn = lambda: ConvBlock(WIDTH_CONV, kernel_size=17, mlp_mult=4)
        width = WIDTH_CONV
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    model = AudioClassifier(block_fn(), width, layers, n_classes, seq_len)
    model.layers = nn.ModuleList([block_fn() for _ in range(layers)])
    return model.to(device)


def load_dataset(name, batch_size, mode_3d=False, seq_len_override=None, num_workers=0):
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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('data', train=False, download=True, transform=transform)
        n_classes, seq_len, img_size = 10, 784, (28, 28)

    elif name == 'fashion':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_data = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
        n_classes, seq_len, img_size = 10, 784, (28, 28)

    elif name == 'cifar10':
        if mode_3d:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            n_classes, seq_len, img_size = 10, 3072, (32, 32, 3)
        else:
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.4734,), (0.2516,))
            ])
            n_classes, seq_len, img_size = 10, 1024, (32, 32)
        train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

    elif name == 'cifar100':
        if mode_3d:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            n_classes, seq_len, img_size = 100, 3072, (32, 32, 3)
        else:
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.4734,), (0.2516,))
            ])
            n_classes, seq_len, img_size = 100, 1024, (32, 32)
        train_data = datasets.CIFAR100('data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100('data', train=False, download=True, transform=transform)

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
    parser.add_argument('--ssm', action='store_true', help='Add SSM block between attention and MLP')
    parser.add_argument('--hard-mining', action='store_true', help='Enable hard example mining (reweight samples by previous epoch loss)')
    parser.add_argument('--hard-start', type=float, default=0.5, help='Hard mining start %% (default: 0.5 = 50%%)')
    parser.add_argument('--hard-end', type=float, default=0.05, help='Hard mining end %% (default: 0.05 = 5%%)')
    parser.add_argument('--first-epoch-pct', type=float, default=None, help='First epoch: only backprop hardest N%% of samples (e.g. 0.3 for 30%%)')
    parser.add_argument('--wtf-mode', action='store_true', help='WTF mode: gradient ascent on easy samples, descent on hard (per-batch median split)')
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
    
    seed_everything(args.seed)

    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}')

    train_loader, test_loader, n_classes_or_vocab, seq_len, img_size, task_type = load_dataset(
        args.dataset, args.batch_size, args.mode_3d, args.seq_len, num_workers=args.workers
    )

    if task_type == 'lm':
        vocab_size = n_classes_or_vocab
        all_model_types = ['attention', 'hier', 'conv']
        builder = lambda mt: build_model_lm(mt, args.layers, vocab_size, seq_len, device, args.channels, args.ssm)
        shape_str = f'seq_len={seq_len}, vocab={vocab_size}'
        flatten = False
        print(f'Task type: Language Modeling (token prediction)')
    elif task_type == 'audio':
        n_classes = n_classes_or_vocab
        all_model_types = ['attention', 'hier', 'conv']
        builder = lambda mt: build_model_audio(mt, args.layers, n_classes, seq_len, device, args.channels, args.ssm)
        shape_str = f'seq_len={seq_len}, n_classes={n_classes}'
        flatten = False
        print(f'Task type: Audio Classification')
    elif args.mode_3d:
        n_classes = n_classes_or_vocab
        all_model_types = ['attention', 'local', 'hier', 'conv']
        builder = lambda mt: build_model_3d(mt, args.layers, n_classes, img_size, device, args.channels)
        shape_str = f'vol_size={img_size}'
        flatten = False
        print(f'Task type: 3D Classification (SSM not supported)')
    elif args.mode_2d:
        n_classes = n_classes_or_vocab
        all_model_types = ['attention', 'local', 'hier', 'conv']
        builder = lambda mt: build_model_2d(mt, args.layers, n_classes, img_size, device, args.channels, args.ssm)
        shape_str = f'img_size={img_size}'
        flatten = True
        print(f'Task type: 2D Classification')
    else:
        n_classes = n_classes_or_vocab
        all_model_types = ['attention', 'hier', 'conv']
        builder = lambda mt: build_model(mt, args.layers, n_classes, seq_len, device, args.channels, args.ssm)
        shape_str = f'seq_len={seq_len}'
        flatten = True
        print(f'Task type: 1D Classification')

    model_types = all_model_types if args.model == 'all' else [args.model]

    print(f'\nModel parameters ({args.layers} layers, {shape_str}):')
    for mt in model_types:
        model = builder(mt)
        print(f'  {mt:12s}: {count_params(model):,} params')

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
            model = builder(mt)

            print(f'\nTraining {mt}...')
            acc = train_model(
                model, train_loader, test_loader, device, args.epochs, args.lr,
                cosine_start=args.cosine_start, swa=args.swa, swa_start=args.swa_start,
                swa_lr=args.swa_lr, hard_mining=args.hard_mining, hard_start=args.hard_start,
                hard_end=args.hard_end, first_epoch_pct=args.first_epoch_pct,
                wtf_mode=args.wtf_mode, checkpoint_dir=args.checkpoint_dir, model_name=f'{mt}_run{run}',
                verbose=(args.runs == 1), flatten=flatten, task_type=task_type
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
