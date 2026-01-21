"""
Scatter Attention: learned position warp + power law envelope + bilinear splat.
Dual of gather-attention: scatter weights to positions, average collisions, apply.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .backbone import AdaptiveDeformConv1d


class KernelNetND(nn.Module):
    def __init__(self, ndim: int, out_channels: int, hidden: int = 32, layers: int = 3, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        net = [nn.Linear(ndim, hidden)]
        for _ in range(layers - 1):
            net.extend([nn.SiLU(), nn.Linear(hidden, hidden)])
        net.extend([nn.SiLU(), nn.Linear(hidden, out_channels)])
        self.net = nn.Sequential(*net)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.net(positions * self.omega_0)


class AdaptiveDeformConvND(nn.Module):
    def __init__(
        self,
        channels: int,
        ndim: int = 1,
        kernel_size: int = 7,
        groups: int = 1,
        init_sigma: float = 0.3,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
        offset_scale: float = 2.0,
        depthwise: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.groups = channels if depthwise else groups
        self.group_channels = 1 if depthwise else channels // groups
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.offset_scale = offset_scale
        self.depthwise = depthwise
        
        self.kernel_numel = kernel_size ** ndim
        
        init_raw = math.log(math.exp(init_sigma) - 1.0) if init_sigma > 0 else 0.0
        self.raw_sigma = nn.Parameter(torch.full((ndim,), init_raw))
        
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        
        conv_cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndim - 1]
        self.dw_conv = nn.Sequential(
            conv_cls(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            conv_cls(channels, channels, 1),
        )
        
        self.offset_net = nn.Linear(channels, self.groups * self.kernel_numel * ndim)
        self.mask_net = nn.Linear(channels, self.groups * self.kernel_numel)
        
        kernel_out = self.groups * self.group_channels
        self.kernel_net = KernelNetND(ndim, kernel_out, hidden=32, layers=3)
        
        grid_1d = torch.linspace(-0.5, 0.5, kernel_size)
        grids = torch.meshgrid(*[grid_1d] * ndim, indexing='ij')
        grid = torch.stack([g.flatten() for g in grids], dim=-1)
        self.register_buffer('grid', grid)
        
        ref_1d = torch.arange(kernel_size).float() - kernel_size // 2
        refs = torch.meshgrid(*[ref_1d] * ndim, indexing='ij')
        ref_offsets = torch.stack([r.flatten() for r in refs], dim=-1)
        self.register_buffer('ref_offsets', ref_offsets)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.zeros_(self.offset_net.weight)
        nn.init.zeros_(self.offset_net.bias)
        nn.init.zeros_(self.mask_net.weight)
        nn.init.zeros_(self.mask_net.bias)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def _compute_envelope(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.view(1, self.ndim)
        dist_sq = (self.grid / sigma.clamp(min=1e-6)) ** 2
        envelope = torch.exp(-0.5 * dist_sq.sum(dim=-1))
        return envelope / envelope.sum().clamp(min=1e-8)
    
    def _sample_nd(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        spatial = x.shape[1:-1]
        C = x.shape[-1]
        G = self.groups
        gc = self.group_channels
        K = self.kernel_numel
        L = reduce(mul, spatial, 1)
        
        x_grouped = x.reshape(B, L, G, gc)
        
        positions = positions.clamp(min=0)
        for d in range(self.ndim):
            positions[..., d] = positions[..., d].clamp(max=spatial[d] - 1)
        
        p_floor = positions.long()
        p_ceil = (p_floor + 1).clamp(max=torch.tensor([s - 1 for s in spatial], device=x.device))
        w_ceil = positions - p_floor.float()
        w_floor = 1.0 - w_ceil
        
        result = torch.zeros(B, L, G, K, gc, device=x.device, dtype=x.dtype)
        
        for corner in range(2 ** self.ndim):
            weight = torch.ones(B, L, G, K, device=x.device, dtype=x.dtype)
            idx = torch.zeros(B, L, G, K, dtype=torch.long, device=x.device)
            stride = 1
            
            for d in range(self.ndim - 1, -1, -1):
                use_ceil = (corner >> d) & 1
                p = p_ceil[..., d] if use_ceil else p_floor[..., d]
                w = w_ceil[..., d] if use_ceil else w_floor[..., d]
                
                p = p.clamp(0, spatial[d] - 1)
                idx = idx + p * stride
                weight = weight * w
                stride *= spatial[d]
            
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, -1, gc)
            x_flat = x_grouped.reshape(B, -1, gc).unsqueeze(2).expand(-1, -1, G, -1)
            
            gathered = torch.gather(x_flat, 1, idx_exp.reshape(B, -1, G, gc)).reshape(B, L, G, K, gc)
            result = result + gathered * weight.unsqueeze(-1)
        
        return result
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        spatial = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        G = self.groups
        gc = self.group_channels
        K = self.kernel_numel
        L = reduce(mul, spatial, 1)
        
        sigma = F.softplus(self.raw_sigma).clamp(self.min_sigma, self.max_sigma)
        
        x_proj = self.input_proj(x)
        
        x_conv = x.reshape(B, L, C).mT.reshape(B, C, *spatial)
        x_conv = self.dw_conv(x_conv)
        x_dw = x_conv.reshape(B, C, L).mT
        
        envelope = self._compute_envelope(sigma)
        kernel_weights = self.kernel_net(self.grid)
        
        offsets = torch.tanh(self.offset_net(x_dw)).reshape(B, L, G, K, self.ndim) * self.offset_scale
        
        pos_indices = torch.stack(torch.meshgrid(
            *[torch.arange(s, device=x.device, dtype=torch.float32) for s in spatial],
            indexing='ij'
        ), dim=-1).reshape(1, L, 1, 1, self.ndim)
        
        abs_pos = pos_indices + self.ref_offsets.reshape(1, 1, 1, K, self.ndim) + offsets
        
        sampled = self._sample_nd(x_proj, abs_pos)
        
        kw = kernel_weights.reshape(G, gc, K)
        sampled = sampled * kw.permute(0, 2, 1).unsqueeze(0).unsqueeze(0)
        
        raw_mask = self.mask_net(x_dw).reshape(B, L, G, K)
        raw_mask = raw_mask * envelope.reshape(1, 1, 1, K)
        attn_mask = F.softmax(raw_mask, dim=-1)
        
        output = (sampled * attn_mask.unsqueeze(-1)).sum(dim=3)
        output = output.reshape(B, L, C)
        output = output.reshape(B, *spatial, C)
        
        offset_reg = (offsets ** 2).mean()
        entropy = -(attn_mask * (attn_mask + 1e-8).log()).sum(dim=-1).mean()
        
        return self.output_proj(output), {"offset_reg": offset_reg, "entropy_reg": -entropy}


class ScatterAttention1d(nn.Module):
    
    k_indices: torch.Tensor
    
    def __init__(
        self,
        num_samples: int = 16,
        max_offset: float = 32.0,
        init_stride: float = 1.0,
        init_beta: float = 1.0,
        init_alpha: float = 1.0,
        init_strength: float = 1.0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_offset = max_offset
        
        self.base_deformation = nn.Parameter(torch.tensor(0.0))
        self.base_stride = nn.Parameter(torch.tensor(init_stride))
        self.base_beta_fwd = nn.Parameter(torch.tensor(init_beta))
        self.base_beta_bwd = nn.Parameter(torch.tensor(init_beta))
        self.base_strength = nn.Parameter(torch.tensor(init_strength))
        self.base_alpha_fwd = nn.Parameter(torch.tensor(init_alpha))
        self.base_alpha_bwd = nn.Parameter(torch.tensor(init_alpha))
        self.sample_bias = nn.Parameter(torch.zeros(num_samples))
        
        k_indices = torch.arange(num_samples, dtype=torch.float32) - num_samples // 2
        self.register_buffer('k_indices', k_indices)
    
    def forward(
        self,
        x: torch.Tensor,
        film_biases: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, L = x.shape
        device = x.device
        
        if film_biases is None:
            film_biases = {}
        
        deformation = self.base_deformation + film_biases.get('deformation', 0)
        stride = F.softplus(self.base_stride + film_biases.get('stride', 0))
        beta_fwd = F.softplus(self.base_beta_fwd + film_biases.get('beta_fwd', 0))
        beta_bwd = F.softplus(self.base_beta_bwd + film_biases.get('beta_bwd', 0))
        strength = F.softplus(self.base_strength + film_biases.get('strength', 0))
        alpha_fwd = F.softplus(self.base_alpha_fwd + film_biases.get('alpha_fwd', 0))
        alpha_bwd = F.softplus(self.base_alpha_bwd + film_biases.get('alpha_bwd', 0))
        sample_bias = self.sample_bias + film_biases.get('sample_bias', 0)
        
        deformation = deformation.clamp(-self.max_offset, self.max_offset)
        
        k = self.k_indices
        k_abs = k.abs()
        
        warped_fwd = (k_abs ** beta_fwd) * stride
        warped_bwd = (k_abs ** beta_bwd) * stride
        warped = torch.where(k >= 0, warped_fwd, -warped_bwd)
        
        l_indices = torch.arange(L, device=device, dtype=torch.float32)
        centers = l_indices + deformation
        
        positions = centers.unsqueeze(-1) + warped.unsqueeze(0)
        positions = positions.clamp(0, L - 1)
        positions = positions.unsqueeze(0).expand(B, -1, -1)
        
        envelope_fwd = strength / (1 + k_abs) ** alpha_fwd
        envelope_bwd = strength / (1 + k_abs) ** alpha_bwd
        envelope = torch.where(k >= 0, envelope_fwd, envelope_bwd)
        
        raw_weights = envelope * (1 + sample_bias.tanh())
        raw_weights = raw_weights.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        
        weight_map, hit_count = self._splat(positions, raw_weights, L)
        
        avg_weights = weight_map / hit_count.clamp(min=1e-6)
        norm_weights = avg_weights / avg_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        
        return x * norm_weights
    
    def _splat(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        L: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = positions.shape[0]
        device = positions.device
        
        pos_floor = positions.floor().long().clamp(0, L - 1)
        pos_ceil = (pos_floor + 1).clamp(0, L - 1)
        
        frac = positions - pos_floor.float()
        w_floor = (1 - frac) * weights
        w_ceil = frac * weights
        h_floor = 1 - frac
        h_ceil = frac
        
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(pos_floor)
        
        idx_floor = (batch_idx * L + pos_floor).reshape(-1)
        idx_ceil = (batch_idx * L + pos_ceil).reshape(-1)
        
        weight_map = torch.zeros(B * L, device=device, dtype=weights.dtype)
        hit_count = torch.zeros(B * L, device=device, dtype=weights.dtype)
        
        weight_map.scatter_add_(0, idx_floor, w_floor.reshape(-1))
        weight_map.scatter_add_(0, idx_ceil, w_ceil.reshape(-1))
        hit_count.scatter_add_(0, idx_floor, h_floor.reshape(-1))
        hit_count.scatter_add_(0, idx_ceil, h_ceil.reshape(-1))
        
        return weight_map.view(B, L), hit_count.view(B, L)


class SEBlock1d(nn.Module):
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
    
    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = x.mean(dim=1)
        s = F.silu(self.fc1(pooled))
        s = self.fc2(s)
        if bias is not None:
            s = s + bias
        s = torch.sigmoid(s)
        return x * s.unsqueeze(1)


class SEBlock2d(nn.Module):
    
    def __init__(self, scatter_channels: int, num_samples: int, reduction: int = 4):
        super().__init__()
        total = scatter_channels * num_samples
        mid = max(1, total // reduction)
        self.scatter_channels = scatter_channels
        self.num_samples = num_samples
        
        self.fc1_global = nn.Linear(total, mid)
        self.fc2_global = nn.Linear(mid, total)
        
        self.fc1_local = nn.Linear(total, mid)
        self.fc2_local = nn.Linear(mid, total)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, SC, K = x.shape
        total = SC * K
        
        pooled = x.mean(dim=1).reshape(B, total)
        s_global = F.silu(self.fc1_global(pooled))
        s_global = torch.sigmoid(self.fc2_global(s_global)).reshape(B, 1, SC, K)
        x = x * s_global
        
        x_flat = x.reshape(B * L, total)
        s_local = F.silu(self.fc1_local(x_flat))
        s_local = torch.sigmoid(self.fc2_local(s_local)).reshape(B, L, SC, K)
        x = x * s_local
        
        return x


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def apply_rope(x: torch.Tensor, positions: torch.Tensor | None = None, base: float = 10000.0) -> torch.Tensor:
    """Apply rotary position embedding.
    
    Args:
        x: (..., C) tensor
        positions: (...) integer positions. If None, uses [0, 1, ..., L-1] for dim -2.
        base: RoPE base frequency
    """
    C = x.shape[-1]
    half_c = C // 2
    device = x.device
    dtype = x.dtype
    
    if positions is None:
        L = x.shape[-2]
        positions = torch.arange(L, device=device)
        for _ in range(x.ndim - 2):
            positions = positions.unsqueeze(0)
    
    dim_idx = torch.arange(half_c, device=device, dtype=dtype)
    freqs = 1.0 / (base ** (dim_idx / half_c))
    
    angles = positions.unsqueeze(-1).float() * freqs
    cos = angles.cos()
    sin = angles.sin()
    
    x1, x2 = x[..., :half_c], x[..., half_c:half_c * 2]
    
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    
    if C % 2 == 1:
        x_rotated = torch.cat([x_rotated, x[..., -1:]], dim=-1)
    
    return x_rotated


def sinusoidal_pos_embed(L: int, C: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0) -> torch.Tensor:
    pos = torch.arange(L, device=device, dtype=dtype)
    dim_idx = torch.arange(C // 2, device=device, dtype=dtype)
    freqs = 1.0 / (base ** (2 * dim_idx / C))
    
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    
    embed = torch.zeros(L, C, device=device, dtype=dtype)
    embed[:, 0::2] = angles.sin()
    embed[:, 1::2] = angles.cos()
    
    return embed


def sinusoidal_pos_embed_nd(
    shape: tuple[int, ...],
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> torch.Tensor:
    ndim = len(shape)
    dim_per_axis = embed_dim // ndim
    remainder = embed_dim % ndim
    
    embeds = []
    for axis, size in enumerate(shape):
        axis_dim = dim_per_axis + (1 if axis < remainder else 0)
        half_dim = axis_dim // 2
        
        pos = torch.arange(size, device=device, dtype=dtype)
        dim_idx = torch.arange(half_dim, device=device, dtype=dtype)
        freqs = 1.0 / (base ** (2 * dim_idx / max(axis_dim, 1)))
        
        angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
        
        axis_embed = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if axis_dim % 2 == 1:
            axis_embed = torch.cat([axis_embed, torch.zeros(size, 1, device=device, dtype=dtype)], dim=-1)
        
        view_shape = [1] * axis + [size] + [1] * (ndim - axis - 1) + [axis_dim]
        expand_shape = list(shape) + [axis_dim]
        axis_embed = axis_embed.view(*view_shape).expand(*expand_shape)
        embeds.append(axis_embed)
    
    return torch.cat(embeds, dim=-1)


class LocalAttentionND(nn.Module):
    
    rel_dist: torch.Tensor
    
    def __init__(self, embed_dim: int, kernel_size: int | tuple[int, ...] = 7, ndim: int = 1, num_channels: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.num_channels = num_channels
        self.channel_dim = embed_dim // num_channels
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim
        self.kernel_size = kernel_size
        self.half_k = tuple(k // 2 for k in kernel_size)
        self.window_size = 1
        for k in kernel_size:
            self.window_size *= k
        
        self.scale = self.channel_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.q_norm = RMSNorm(self.channel_dim)
        self.k_norm = RMSNorm(self.channel_dim)
        # Projects to (width, sharpness) per channel
        self.window_proj = nn.Linear(embed_dim, 2 * num_channels)
        self.out_norm = RMSNorm(embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Init: width→mid, sharpness→5.0
        nn.init.zeros_(self.window_proj.weight)
        nn.init.zeros_(self.window_proj.bias)
        
        rel_coords = [torch.arange(k).float() - k // 2 for k in kernel_size]
        grids = torch.meshgrid(*rel_coords, indexing='ij')
        rel_dist = torch.sqrt(torch.stack([g**2 for g in grids]).sum(dim=0))
        self.register_buffer('rel_dist', rel_dist.flatten())
        self.max_dist = rel_dist.max().item()
    
    def _unfold_nd(self, x: torch.Tensor) -> torch.Tensor:
        pad_dims = [0, 0]
        for half in reversed(self.half_k):
            pad_dims.extend([half, half])
        x = F.pad(x, pad_dims)
        
        for dim in range(self.ndim):
            x = x.unfold(dim + 1, self.kernel_size[dim], 1)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        L = 1
        for s in spatial_shape:
            L *= s
        
        H = self.num_channels
        D = self.channel_dim
        
        qkv = F.silu(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=-1)
        
        q_flat = self.q_norm(q.reshape(B, L, H, D))
        k_flat = self.k_norm(k.reshape(B, L, H, D))
        q_flat = apply_rope(q_flat)
        k_flat = apply_rope(k_flat)
        k = k_flat.reshape(*k.shape)
        
        window_params = F.silu(self.window_proj(x))
        width_raw, sharpness_raw = window_params.chunk(2, dim=-1)
        width = width_raw.sigmoid() * self.max_dist + 0.5
        sharpness = sharpness_raw.sigmoid() * 9.5 + 0.5
        
        k_win = self._unfold_nd(k)
        v_win = self._unfold_nd(v)
        
        perm = [0] + list(range(1, self.ndim + 1)) + list(range(self.ndim + 2, self.ndim * 2 + 2)) + [self.ndim + 1]
        k_win = k_win.permute(*perm).reshape(B, L, self.window_size, H, D).permute(0, 1, 3, 4, 2)
        v_win = v_win.permute(*perm).reshape(B, L, self.window_size, H, D).permute(0, 1, 3, 4, 2)
        
        width_flat = width.reshape(B, L, H, 1)
        sharpness_flat = sharpness.reshape(B, L, H, 1)
        
        scores = torch.einsum('blhd,blhdw->blhw', q_flat, k_win) * self.scale
        
        soft_mask = torch.sigmoid((width_flat - self.rel_dist) * sharpness_flat)
        scores = scores - (1 - soft_mask) * 1e4
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('blhw,blhdw->blhd', attn, v_win).reshape(B, L, C)
        
        out = self.out_norm(out.reshape(*x.shape[:-1], C))
        return F.silu(self.out(out + v))


class LocalAttention(LocalAttentionND):
    def __init__(self, embed_dim: int, kernel_size: int = 17, num_channels: int = 1):
        super().__init__(embed_dim, kernel_size, ndim=1, num_channels=num_channels)


class LocalBlock(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int = 17, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = LocalAttention(embed_dim, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class HierarchicalLocalAttentionND(nn.Module):
    
    def __init__(self, embed_dim: int, window_size: int | tuple[int, ...] = 17, ndim: int = 1, num_channels: int = 1, poolable_dims: tuple[int, ...] | None = None, min_size: int = 4, adapt_stem: bool = False, adapt_reduce: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.num_channels = num_channels
        self.channel_dim = embed_dim // num_channels
        self.poolable_dims = poolable_dims if poolable_dims is not None else tuple(range(ndim))
        self.min_size = min_size
        self.adapt_stem = adapt_stem
        self.adapt_reduce = adapt_reduce
        
        if isinstance(window_size, int):
            window_size = (window_size,) * ndim
        self.window_size = window_size
        
        conv_cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndim - 1]
        pool_cls = [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d][ndim - 1]
        
        if self.adapt_stem:
            self.stem_conv = AdaptiveDeformConvND(embed_dim, ndim=ndim, kernel_size=7, depthwise=True)
        else:
            self.stem_conv = conv_cls(embed_dim, embed_dim, kernel_size=2, padding=1)
        self.conv_norm = RMSNorm(embed_dim)
        
        if self.adapt_reduce:
            self.reduce_conv = AdaptiveDeformConvND(embed_dim, ndim=ndim, kernel_size=7, depthwise=True)
            self.reduce_pool = pool_cls(kernel_size=2, stride=2)
        elif len(self.poolable_dims) == ndim:
            self.reduce_conv = conv_cls(embed_dim, embed_dim, kernel_size=2, stride=2)
        else:
            reduce_stride = tuple(2 if d in self.poolable_dims else 1 for d in range(ndim))
            reduce_kernel = tuple(2 if d in self.poolable_dims else 1 for d in range(ndim))
            self.reduce_conv = conv_cls(embed_dim, embed_dim, kernel_size=reduce_kernel, stride=reduce_stride)
        
        self.attn = LocalAttentionND(embed_dim, window_size, ndim, num_channels)
        
        self.level_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.level_kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.level_scale = embed_dim ** -0.5
        self.ctx_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.hidden_kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        nn.init.zeros_(self.out_proj.weight)
    
    def _compute_n_levels(self, spatial_shape: tuple[int, ...]) -> int:
        if not self.poolable_dims:
            return 1
        min_poolable = min(spatial_shape[d] for d in self.poolable_dims)
        return max(1, (min_poolable // self.min_size).bit_length())
    
    def _reduce(self, h: torch.Tensor) -> torch.Tensor:
        if self.adapt_reduce:
            spatial = h.shape[2:]
            B, C = h.shape[:2]
            h_cl = h.reshape(B, C, -1).mT.reshape(B, *spatial, C)
            out, _ = self.reduce_conv(h_cl)
            out = out.reshape(B, -1, C).mT.reshape(B, C, *spatial)
            out = self.reduce_pool(out)
            return out
        return self.reduce_conv(h)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        H = self.num_channels
        D = self.channel_dim
        L = 1
        for s in spatial_shape:
            L *= s
        
        n_levels = self._compute_n_levels(spatial_shape)
        
        if self.adapt_stem:
            conv_out, _ = self.stem_conv(x)
            conv_out = self.conv_norm(conv_out.reshape(B, L, C)).reshape(B, *spatial_shape, C)
            h = x + conv_out
        else:
            h = x.reshape(B, L, C).mT.reshape(B, C, *spatial_shape)
            conv_out = self.stem_conv(h)
            for dim, size in enumerate(spatial_shape):
                conv_out = conv_out.narrow(dim + 2, 0, size)
            conv_out = self.conv_norm(conv_out.reshape(B, C, L).mT).mT.reshape(B, C, *spatial_shape)
            h = (h + conv_out).reshape(B, C, L).mT.reshape(B, *spatial_shape, C)
        
        levels = []
        current_shape = list(spatial_shape)
        
        for i in range(n_levels):
            h = self.attn(h)
            levels.append(h)
            if i < n_levels - 1:
                h_flat = h.reshape(B, -1, C).mT.reshape(B, C, *h.shape[1:-1])
                h_reduced = self._reduce(h_flat)
                current_shape = list(h_reduced.shape[2:])
                h = h_reduced.reshape(B, C, -1).mT.reshape(B, *current_shape, C)
        
        level0_flat = levels[0].reshape(B, L, C)
        
        # 1. Learned query attends to level means -> global context
        level_means = torch.stack([lvl.reshape(B, -1, C).mean(dim=1) for lvl in levels], dim=1)  # (B, n_levels, C)
        level_kv = self.level_kv(level_means)  # (B, n_levels, 2C)
        level_k, level_v = level_kv.chunk(2, dim=-1)  # each (B, n_levels, C)
        
        # level_query: (1, 1, C) -> broadcast to (B, 1, C)
        ctx_scores = torch.einsum('bqc,bkc->bqk', self.level_query.expand(B, -1, -1), level_k) * self.level_scale
        ctx_weights = F.softmax(ctx_scores, dim=-1)
        ctx = torch.einsum('bqk,bkc->bqc', ctx_weights, level_v)  # (B, 1, C)
        
        # 2. Context cross-attends to level0 hidden states
        q = self.ctx_proj(ctx)  # (B, 1, C)
        hidden_kv = self.hidden_kv(level0_flat)  # (B, L, 2C)
        hidden_k, hidden_v = hidden_kv.chunk(2, dim=-1)  # each (B, L, C)
        
        hidden_scores = torch.einsum('bqc,bkc->bqk', q, hidden_k) * self.level_scale
        hidden_weights = F.softmax(hidden_scores, dim=-1)
        update = torch.einsum('bqk,bkc->bqc', hidden_weights, hidden_v)  # (B, 1, C)
        
        # 3. Project and add residual (broadcasts (B,1,C) to (B,L,C))
        out = level0_flat + self.out_proj(update)
        return out.reshape(B, *spatial_shape, C)


class HierarchicalLocalAttention(HierarchicalLocalAttentionND):
    def __init__(self, embed_dim: int, window_size: int = 17, num_channels: int = 1, poolable_dims: tuple[int, ...] | None = None, adapt_stem: bool = False, adapt_reduce: bool = False):
        super().__init__(embed_dim, window_size, ndim=1, num_channels=num_channels, poolable_dims=poolable_dims, adapt_stem=adapt_stem, adapt_reduce=adapt_reduce)


class HierarchicalLocalBlockND(nn.Module):
    
    def __init__(self, embed_dim: int, window_size: int | tuple[int, ...] = 17, ndim: int = 1, num_channels: int = 1, poolable_dims: tuple[int, ...] | None = None, eps: float = 1e-6, adapt_stem: bool = False, adapt_reduce: bool = False):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = HierarchicalLocalAttentionND(embed_dim, window_size, ndim, num_channels, poolable_dims, adapt_stem=adapt_stem, adapt_reduce=adapt_reduce)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class HierarchicalLocalBlock(HierarchicalLocalBlockND):
    def __init__(self, embed_dim: int, window_size: int = 17, num_channels: int = 1, poolable_dims: tuple[int, ...] | None = None, eps: float = 1e-6, adapt_stem: bool = False, adapt_reduce: bool = False):
        super().__init__(embed_dim, window_size, ndim=1, num_channels=num_channels, poolable_dims=poolable_dims, eps=eps, adapt_stem=adapt_stem, adapt_reduce=adapt_reduce)


LocalKernelAttention = LocalAttention
LocalKernelBlock = LocalBlock
