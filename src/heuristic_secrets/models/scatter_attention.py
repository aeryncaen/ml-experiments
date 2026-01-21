"""
Scatter Attention: learned position warp + power law envelope + bilinear splat.
Dual of gather-attention: scatter weights to positions, average collisions, apply.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    B, L, C = x.shape
    half_c = C // 2
    device = x.device
    dtype = x.dtype
    
    pos = torch.arange(L, device=device, dtype=dtype)
    dim_idx = torch.arange(half_c, device=device, dtype=dtype)
    freqs = 1.0 / (base ** (dim_idx / half_c))
    
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()
    
    x1, x2 = x[..., :half_c], x[..., half_c:]
    
    x_rope = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    
    return x_rope


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
    
    def __init__(self, embed_dim: int, kernel_size: int | tuple[int, ...] = 7, ndim: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim
        self.kernel_size = kernel_size
        self.half_k = tuple(k // 2 for k in kernel_size)
        self.window_size = 1
        for k in kernel_size:
            self.window_size *= k
        
        self.scale = embed_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.width_proj = nn.Linear(embed_dim, 1)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        
        nn.init.zeros_(self.width_proj.weight)
        nn.init.zeros_(self.width_proj.bias)
        
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
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        width = self.width_proj(x).sigmoid() * self.max_dist + 0.5
        
        k_win = self._unfold_nd(k)
        v_win = self._unfold_nd(v)
        
        perm = [0] + list(range(1, self.ndim + 1)) + list(range(self.ndim + 2, self.ndim * 2 + 2)) + [self.ndim + 1]
        k_win = k_win.permute(*perm).reshape(B, L, self.window_size, C).transpose(-1, -2)
        v_win = v_win.permute(*perm).reshape(B, L, self.window_size, C).transpose(-1, -2)
        
        q_flat = q.reshape(B, L, C)
        width_flat = width.reshape(B, L, 1)
        
        scores = torch.einsum('blc,blcw->blw', q_flat, k_win) * self.scale
        
        soft_mask = torch.sigmoid((width_flat - self.rel_dist) * 5.0)
        scores = scores - (1 - soft_mask) * 1e4
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('blw,blcw->blc', attn, v_win)
        
        out = out.reshape(*x.shape[:-1], C)
        return self.out(out + v)


class LocalAttention(LocalAttentionND):
    def __init__(self, embed_dim: int, kernel_size: int = 17):
        super().__init__(embed_dim, kernel_size, ndim=1)


class LocalBlock(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int = 17, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = LocalAttention(embed_dim, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class HierarchicalLocalAttentionND(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int | tuple[int, ...] = 17, n_levels: int = 4, ndim: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        self.ndim = ndim
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim
        self.kernel_size = kernel_size
        
        conv_cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndim - 1]
        self.stem_conv = conv_cls(embed_dim, embed_dim, kernel_size=2, padding=1)
        
        self.attn = LocalAttentionND(embed_dim, kernel_size, ndim)
        self.query = nn.Parameter(torch.randn(embed_dim * n_levels) * 0.02)
        self.film = nn.Linear(embed_dim * n_levels, embed_dim * 2)
        
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        L = 1
        for s in spatial_shape:
            L *= s
        
        h = x.reshape(B, L, C).mT.reshape(B, C, *spatial_shape)
        h = self.stem_conv(h)
        for dim, size in enumerate(spatial_shape):
            h = h.narrow(dim + 2, 0, size)
        h = h.reshape(B, C, L).mT.reshape(B, *spatial_shape, C)
        
        levels = []
        
        for i in range(self.n_levels):
            h = self.attn(h)
            levels.append(h)
            if i < self.n_levels - 1:
                h_flat = h.reshape(B, -1, C).mT.reshape(B, C, *h.shape[1:-1])
                pool_fn = [F.avg_pool1d, F.avg_pool2d, F.avg_pool3d][self.ndim - 1]
                h_pooled = pool_fn(h_flat, 2)
                new_shape = h_pooled.shape[2:]
                h = h_pooled.reshape(B, C, -1).mT.reshape(B, *new_shape, C)
        
        level0_flat = levels[0].reshape(B, L, C)
        upsampled = [level0_flat]
        for lvl in levels[1:]:
            lvl_flat = lvl.reshape(B, -1, C)
            lvl_up = F.interpolate(lvl_flat.mT, size=L, mode='nearest').mT
            upsampled.append(lvl_up)
        concat = torch.cat(upsampled, dim=-1)
        
        scores = concat @ self.query
        attn = F.softmax(scores, dim=-1)
        global_ctx = torch.einsum('bl,blc->bc', attn, concat)
        
        film_params = self.film(global_ctx)
        scale, bias = film_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        bias = bias.unsqueeze(1)
        
        out = level0_flat * (1 + scale) + bias
        return out.reshape(B, *spatial_shape, C)


class HierarchicalLocalAttention(HierarchicalLocalAttentionND):
    def __init__(self, embed_dim: int, kernel_size: int = 17, n_levels: int = 4):
        super().__init__(embed_dim, kernel_size, n_levels, ndim=1)


class HierarchicalLocalBlockND(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int | tuple[int, ...] = 17, n_levels: int = 4, ndim: int = 1, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = HierarchicalLocalAttentionND(embed_dim, kernel_size, n_levels, ndim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class HierarchicalLocalBlock(HierarchicalLocalBlockND):
    def __init__(self, embed_dim: int, kernel_size: int = 17, n_levels: int = 4, eps: float = 1e-6):
        super().__init__(embed_dim, kernel_size, n_levels, ndim=1, eps=eps)


LocalKernelAttention = LocalAttention
LocalKernelBlock = LocalBlock
