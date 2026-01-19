import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

DEBUG_NAN = True


def check_nan(t: torch.Tensor, name: str) -> None:
    if DEBUG_NAN and torch.isnan(t).any():
        print(f"NaN detected in {name}, shape={t.shape}, device={t.device}")
        print(f"  nan count: {torch.isnan(t).sum().item()}")
        print(f"  inf count: {torch.isinf(t).sum().item()}")
        raise RuntimeError(f"NaN in {name}")


class RotaryEmbedding(nn.Module):
    """Precomputes and caches rotary position embeddings (RoPE)."""

    inv_freq: torch.Tensor
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(self, dim: int, base: int = 10000, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        pos = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(pos, self.inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cache[:seq_len].to(device),
            self.sin_cache[:seq_len].to(device),
        )


def apply_rotary_pos_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. q,k: [B, H, L, D], cos,sin: [L, D/2]."""

    def rotate(x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return rotate(q, cos, sin), rotate(k, cos, sin)


def chunked_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    pad_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Memory-efficient sliding window attention. O(L * window) instead of O(L²).

    q, k, v: [B, H, L, D]
    pad_mask: [B, L] True = padded position
    Returns: [B, H, L, D]
    """
    B, H, L, D = q.shape
    left_pad = window_size // 2
    right_pad = window_size - left_pad - 1

    k_padded = F.pad(k, (0, 0, left_pad, right_pad), value=0)
    v_padded = F.pad(v, (0, 0, left_pad, right_pad), value=0)

    if pad_mask is not None:
        mask_padded = F.pad(pad_mask, (left_pad, right_pad), value=True)
    else:
        mask_padded = None

    k_windows = k_padded.unfold(2, window_size, 1)  # [B, H, L, D, W]
    v_windows = v_padded.unfold(2, window_size, 1)  # [B, H, L, D, W]
    k_windows = k_windows.permute(0, 1, 2, 4, 3)  # [B, H, L, W, D]
    v_windows = v_windows.permute(0, 1, 2, 4, 3)  # [B, H, L, W, D]

    scale = D**-0.5
    scores = torch.einsum("bhld,bhlwd->bhlw", q, k_windows) * scale  # [B, H, L, W]

    if mask_padded is not None:
        mask_windows = mask_padded.unfold(1, window_size, 1)  # [B, L, W]
        scores = scores.masked_fill(mask_windows.unsqueeze(1), float("-inf"))

    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    if dropout_p > 0 and training:
        attn = F.dropout(attn, p=dropout_p, training=True)

    out = torch.einsum("bhlw,bhlwd->bhld", attn, v_windows)
    return out


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
            raise ValueError(
                f"channels ({channels}) must be divisible by groups ({groups})"
            )

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

    def _reset_parameters(self) -> None:
        constant_(self.offset_net.weight.data, 0.0)
        constant_(self.offset_net.bias.data, 0.0)
        constant_(self.mask_net.weight.data, 0.0)
        constant_(self.mask_net.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

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
        mask = torch.nan_to_num(mask, nan=0.0)

        ref_offsets = torch.linspace(-(K // 2), K // 2, K, device=x.device)
        pos_indices = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L, 1, 1)

        x_grouped = x_proj.view(N, L, G, gc)

        abs_pos = pos_indices + ref_offsets.view(1, 1, 1, K) + offsets
        abs_pos_clamped = abs_pos.clamp(0, L - 1)

        p_floor = abs_pos_clamped.long().clamp(0, L - 1)
        p_ceil = (p_floor + 1).clamp(0, L - 1)
        w_ceil = abs_pos_clamped - p_floor.float()
        w_floor = 1.0 - w_ceil

        oob = (abs_pos < 0) | (abs_pos > L - 1)
        w_floor = w_floor * (~oob).float()
        w_ceil = w_ceil * (~oob).float()

        p_floor_flat = p_floor.view(N, -1)
        p_ceil_flat = p_ceil.view(N, -1)

        x_flat = x_grouped.view(N, L, -1)
        v_floor = x_flat.gather(1, p_floor_flat.unsqueeze(-1).expand(-1, -1, gc * G))
        v_ceil = x_flat.gather(1, p_ceil_flat.unsqueeze(-1).expand(-1, -1, gc * G))

        v_floor = v_floor.view(N, L, G, K, gc)
        v_ceil = v_ceil.view(N, L, G, K, gc)

        sampled = v_floor * w_floor.unsqueeze(-1) + v_ceil * w_ceil.unsqueeze(-1)
        output = (sampled * mask.unsqueeze(-1)).sum(dim=3)

        return self.output_proj(output.reshape(N, L, C))


class AdaptiveConvBiases:
    """Container for all adaptive convolution biases from attention."""

    __slots__ = ("sigma", "offset_scale", "omega")

    def __init__(
        self,
        sigma: torch.Tensor | None = None,
        offset_scale: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
    ):
        self.sigma = sigma
        self.offset_scale = offset_scale
        self.omega = omega


class AdaptiveDeformConv1d(nn.Module):
    """
    Deformable convolution with KernelNet-generated weights and adaptive Gaussian envelope.

    Combines:
    - KernelNet (SIREN): generates continuous kernel weights for any position
    - Deformable offsets: content-dependent WHERE to sample
    - Attention masks: content-dependent weighting per position
    - Gaussian envelope: adaptive receptive field size
    - SE block: channel recalibration

    All adaptive parameters can be biased by attention outputs.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        groups: int = 4,
        offset_scale: float = 2.0,
        init_sigma: float = 0.3,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
        depthwise: bool = False,
        se_reduction: int = 4,
        kernel_hidden: int = 32,
        kernel_layers: int = 3,
        omega_0: float = 30.0,
        min_kernel_size: int = 3,
        max_kernel_size: int = 31,
        dynamic_kernel: bool = True,
    ):
        super().__init__()
        if not depthwise and channels % groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by groups ({groups})"
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = channels if depthwise else groups
        self.group_channels = 1 if depthwise else channels // groups
        self.base_offset_scale = offset_scale
        self.depthwise = depthwise
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.base_omega_0 = omega_0
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.dynamic_kernel = dynamic_kernel

        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))

        grid = torch.linspace(-0.5, 0.5, kernel_size)
        self.register_buffer("grid", grid)

        kernel_out = self.groups * self.group_channels
        self.kernel_net = KernelNet1d(
            out_channels=kernel_out,
            hidden_channels=kernel_hidden,
            num_layers=kernel_layers,
            omega_0=omega_0,
        )

        self.dw_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )

        self.offset_net = nn.Linear(channels, self.groups * max_kernel_size)
        self.mask_net = nn.Linear(channels, self.groups * max_kernel_size)

        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

        self.se = SEBlock(channels, se_reduction)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        constant_(self.offset_net.weight.data, 0.0)
        constant_(self.offset_net.bias.data, 0.0)
        constant_(self.mask_net.weight.data, 0.0)
        constant_(self.mask_net.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def _get_effective_kernel_size(self, sigma: torch.Tensor) -> int:
        sigma_val = sigma.mean().item() if sigma.dim() > 0 else sigma.item()
        if not math.isfinite(sigma_val):
            return self.kernel_size
        raw_k = int(6 * sigma_val / self.max_sigma * self.max_kernel_size)
        raw_k = max(self.min_kernel_size, min(self.max_kernel_size, raw_k))
        return raw_k // 2 * 2 + 1

    def _compute_envelope(
        self, grid: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """exp(-g² / (2σ²)), normalized."""
        if sigma.dim() > 0:
            envelope = torch.exp(
                -(grid.unsqueeze(0) ** 2) / (2 * sigma.unsqueeze(-1) ** 2)
            )
        else:
            envelope = torch.exp(-(grid**2) / (2 * sigma**2))
        return envelope / envelope.sum(dim=-1, keepdim=True)

    def _get_kernel_weights(
        self, grid: torch.Tensor, omega_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        positions = (grid * 2).view(1, 1, -1)

        if omega_bias is not None:
            omega_scale = 1.0 + 0.1 * omega_bias.mean()
            positions = positions * omega_scale

        weights = self.kernel_net(positions)
        return weights.squeeze(0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        biases: AdaptiveConvBiases | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x: (N, L, C) -> (output, aux_losses). aux_losses has offset_reg and entropy_reg."""
        N, L, C = x.shape
        G = self.groups
        gc = self.group_channels
        max_K = self.max_kernel_size

        sigma_bias = biases.sigma if biases else None
        offset_scale_bias = biases.offset_scale if biases else None
        omega_bias = biases.omega if biases else None

        sigma = self.log_sigma.exp()
        if sigma_bias is not None:
            sigma = sigma + torch.nan_to_num(sigma_bias, nan=0.0)
        sigma = sigma.clamp(self.min_sigma, self.max_sigma)

        K = max_K

        grid = torch.linspace(-0.5, 0.5, K, device=x.device)

        x_proj = self.input_proj(x)

        x_dw = x.transpose(1, 2)
        x_dw = self.dw_conv[0](x_dw)
        x_dw = x_dw.transpose(1, 2)
        x_dw = self.dw_conv[1](x_dw)
        x_dw = self.dw_conv[2](x_dw)

        envelope = self._compute_envelope(grid, sigma)
        kernel_weights = self._get_kernel_weights(grid, omega_bias)

        offset_scale = self.base_offset_scale
        if offset_scale_bias is not None:
            offset_scale = offset_scale * (1.0 + 0.2 * offset_scale_bias.mean())

        offsets = self.offset_net(x_dw).view(N, L, G, K) * offset_scale
        raw_mask = self.mask_net(x_dw).view(N, L, G, K)

        if envelope.dim() == 1:
            raw_mask = raw_mask * envelope.view(1, 1, 1, K)
        else:
            raw_mask = raw_mask * envelope.view(N, 1, 1, K)

        attn_mask = F.softmax(raw_mask, dim=-1)
        attn_mask = torch.nan_to_num(attn_mask, nan=0.0)

        ref_offsets = torch.linspace(-(K // 2), K // 2, K, device=x.device)
        pos_indices = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L, 1)

        x_grouped = x_proj.view(N, L, G, gc)
        output = torch.zeros(N, L, G, gc, device=x.device, dtype=x.dtype)

        batch_idx = torch.arange(N, device=x.device).view(N, 1, 1)
        group_idx = torch.arange(G, device=x.device).view(1, 1, G)

        kw = kernel_weights.view(G, gc, K)

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

            kernel_weight_k = kw[:, :, k].unsqueeze(0).unsqueeze(0)
            sampled = sampled * kernel_weight_k

            output = output + sampled * attn_mask[:, :, :, k : k + 1]

        output = output.reshape(N, L, C)
        output = self.se(output, mask)

        offset_reg = (offsets**2).mean()
        entropy = -(attn_mask * (attn_mask + 1e-8).log()).sum(dim=-1).mean()
        aux_losses = {"offset_reg": offset_reg, "entropy_reg": -entropy}

        return self.output_proj(output), aux_losses


class AdaptiveConvBranch(nn.Module):
    def __init__(
        self,
        width: int,
        init_sigma: float = 0.3,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
        groups: int = 4,
        kernel_size: int = 15,
        max_kernel_size: int = 31,
    ):
        super().__init__()
        self.norm = RMSNorm(width)
        self.conv = AdaptiveDeformConv1d(
            width,
            kernel_size=kernel_size,
            groups=groups,
            init_sigma=init_sigma,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            max_kernel_size=max_kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        biases: AdaptiveConvBiases | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out, aux = self.conv(x, mask, biases)
        return self.norm(F.silu(out)), aux


class Sine(nn.Module):
    """Sine activation for SIREN networks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SIRENLayer(nn.Module):
    """
    Linear layer with omega_0 scaling for SIREN: y = sin(omega_0 * (Wx + b))
    Uses Conv1d with kernel_size=1 for efficient batch processing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Conv1d(in_features, out_features, kernel_size=1, bias=bias)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class KernelNet1d(nn.Module):
    """
    Neural network that generates 1D convolution kernel weights for any continuous position.

    Based on SIREN (Implicit Neural Representations with Periodic Activation Functions).
    Input: relative positions in [-1, 1], shape [1, 1, K]
    Output: kernel weights, shape [1, out_channels, K]

    The key insight: this network has FIXED parameters but can generate kernels of ANY size
    by simply changing the number of input positions.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 3,
        omega_0: float = 30.0,
        bias: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels

        layers: list[nn.Module] = []
        layers.append(
            SIRENLayer(1, hidden_channels, omega_0=omega_0, is_first=True, bias=bias)
        )
        for _ in range(num_layers - 2):
            layers.append(
                SIRENLayer(
                    hidden_channels,
                    hidden_channels,
                    omega_0=omega_0,
                    is_first=False,
                    bias=bias,
                )
            )

        self.output_linear = nn.Conv1d(
            hidden_channels, out_channels, kernel_size=1, bias=bias
        )
        self._init_output_weights()
        self.net = nn.Sequential(*layers)

    def _init_output_weights(self) -> None:
        with torch.no_grad():
            self.output_linear.weight.uniform_(-0.01, 0.01)
            if self.output_linear.bias is not None:
                self.output_linear.bias.zero_()

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: [1, 1, K] -> [1, out_channels, K]"""
        return self.output_linear(self.net(positions))


class ConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        groups: int = 4,
        offset_scale: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = DeformConv1d(channels, kernel_size, groups, offset_scale)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.norm(self.conv(x)))


class ConvBackbone(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        kernel_size: int = 9,
        groups: int = 2,
        offset_scale: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                ConvBlock(width, kernel_size, groups, offset_scale, dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, width: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = width * ffn_mult
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class AttentionBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        window_size: int = 0,
        use_rope: bool = False,
    ):
        super().__init__()
        assert width % num_heads == 0
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout_p = dropout
        self.window_size = window_size
        self.use_rope = use_rope

        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

        self.norm2 = RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_mult, dropout)

        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L, _ = x.shape

        if self.window_size > 0 and L > self.window_size:
            attn_len = self.window_size
            x_attn = x[:, :attn_len]
            x_rest = x[:, attn_len:]
            mask_attn = mask[:, :attn_len] if mask is not None else None
        else:
            attn_len = L
            x_attn = x
            x_rest = None
            mask_attn = mask

        qkv = self.qkv(x_attn).reshape(B, attn_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rotary(attn_len, x.device)
            q, k = apply_rotary_pos_emb_qk(q, k, cos, sin)

        attn_mask = None
        if mask_attn is not None:
            attn_mask = (
                mask_attn.unsqueeze(1)
                .unsqueeze(2)
                .expand(B, self.num_heads, attn_len, attn_len)
            )
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, attn_len, self.width)
        x_attn = self.norm1(x_attn + self.dropout(self.out(out)))
        x_attn = self.norm2(x_attn + self.ffn(x_attn))

        if x_rest is not None:
            x_rest = self.norm2(x_rest + self.ffn(x_rest))
            x = torch.cat([x_attn, x_rest], dim=1)
        else:
            x = x_attn
        return x


class ContextualAttentionBlock(nn.Module):
    """
    AttentionBlock extended with outputs for adaptive conv modulation.

    Outputs per-branch biases for:
    - sigma: Gaussian envelope width
    - offset_scale: deformable offset magnitude
    - omega: KernelNet SIREN frequency
    - context: vector for downstream attention pooler
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        n_branches: int = 0,
        context_dim: int = 0,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        window_size: int = 0,
        use_rope: bool = False,
    ):
        super().__init__()
        assert width % num_heads == 0
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.dropout_p = dropout
        self.n_branches = n_branches
        self.context_dim = context_dim
        self.window_size = window_size
        self.use_rope = use_rope

        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

        self.norm2 = RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_mult, dropout)

        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim)

        if n_branches > 0 or context_dim > 0:
            self.pool_query = nn.Parameter(torch.randn(1, width))
            self.pool_scale = width**-0.5

        if n_branches > 0:
            self.sigma_proj = nn.Linear(width, n_branches)
            self.offset_scale_proj = nn.Linear(width, n_branches)
            self.omega_proj = nn.Linear(width, n_branches)

            for proj in [self.sigma_proj, self.offset_scale_proj, self.omega_proj]:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

        if context_dim > 0:
            self.context_proj = nn.Linear(width, context_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, AdaptiveConvBiases | None, torch.Tensor | None]
    ):
        B, L, _ = x.shape

        if self.window_size > 0 and L > self.window_size:
            attn_len = self.window_size
            x_attn = x[:, :attn_len]
            x_rest = x[:, attn_len:]
            mask_attn = mask[:, :attn_len] if mask is not None else None
        else:
            attn_len = L
            x_attn = x
            x_rest = None
            mask_attn = mask

        check_nan(x_attn, "ctx_attn_x_input")
        qkv = self.qkv(x_attn).reshape(B, attn_len, 3, self.num_heads, self.head_dim)
        check_nan(qkv, "ctx_attn_qkv")
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rotary(attn_len, x.device)
            q, k = apply_rotary_pos_emb_qk(q, k, cos, sin)
            check_nan(q, "ctx_attn_q_rope")
            check_nan(k, "ctx_attn_k_rope")

        attn_mask = None
        if mask_attn is not None:
            attn_mask = (
                mask_attn.unsqueeze(1)
                .unsqueeze(2)
                .expand(B, self.num_heads, attn_len, attn_len)
            )
        if DEBUG_NAN:
            print(
                f"SDPA inputs: q={q.min().item():.4f}/{q.max().item():.4f}, k={k.min().item():.4f}/{k.max().item():.4f}, v={v.min().item():.4f}/{v.max().item():.4f}"
            )
            if attn_mask is not None:
                print(f"  attn_mask all True (fully masked): {attn_mask.all().item()}")
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        check_nan(out, "ctx_attn_sdpa_out")

        out = out.transpose(1, 2).reshape(B, attn_len, self.width)
        x_attn = self.norm1(x_attn + self.dropout(self.out(out)))
        check_nan(x_attn, "ctx_attn_after_norm1")
        x_attn = self.norm2(x_attn + self.ffn(x_attn))
        check_nan(x_attn, "ctx_attn_after_norm2")

        if x_rest is not None:
            x_rest = self.norm2(x_rest + self.ffn(x_rest))
            x = torch.cat([x_attn, x_rest], dim=1)
        else:
            x = x_attn

        if self.n_branches == 0 and self.context_dim == 0:
            return x

        q_pool = self.pool_query.unsqueeze(0).expand(B, -1, -1)
        pool_target = x_attn
        pool_mask = mask_attn
        attn_scores = torch.bmm(q_pool, pool_target.transpose(1, 2)) * self.pool_scale
        if pool_mask is not None:
            attn_scores = attn_scores.masked_fill(pool_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        pooled = torch.bmm(attn_weights, pool_target).squeeze(1)

        if self.n_branches > 0:
            biases = AdaptiveConvBiases(
                sigma=self.sigma_proj(pooled),
                offset_scale=self.offset_scale_proj(pooled),
                omega=self.omega_proj(pooled),
            )
        else:
            biases = None

        context = self.context_proj(pooled) if self.context_dim > 0 else None

        return x, biases, context


class AttentionBackbone(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.layers = nn.ModuleList(
            [AttentionBlock(width, num_heads, ffn_mult, dropout) for _ in range(depth)]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, L, C]
        if mask is not None:
            x_masked = x.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            w = x_masked.sum(dim=1) / lengths
        else:
            w = x.mean(dim=1)
        w = self.fc(w)
        return x * w.unsqueeze(1)


class SSMMixer(nn.Module):
    """
    Selective State Space Model mixer with deformable convolution.

    Replaces Mamba's simple Conv1d with DeformConv1d + SE for richer local context.
    Uses pure PyTorch implementation (no CUDA kernels) for device compatibility.
    """

    def __init__(
        self,
        width: int,
        state_size: int = 16,
        conv_kernel: int = 7,
        conv_groups: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.width = width
        self.state_size = state_size
        self.expand = expand
        self.intermediate_size = width * expand
        self.dt_rank = dt_rank or (width + 15) // 16  # ceil(width / 16)

        # Input projection: width -> 2 * intermediate (for gating)
        self.in_proj = nn.Linear(width, self.intermediate_size * 2, bias=False)

        # Deformable conv + SE instead of Mamba's simple Conv1d
        self.conv = DeformConv1d(self.intermediate_size, conv_kernel, conv_groups)
        self.conv_norm = nn.LayerNorm(self.intermediate_size)
        self.se = SEBlock(self.intermediate_size)
        self.conv_act = nn.SiLU()

        # Selective projection: intermediate -> dt_rank + 2*state_size
        # Produces input-dependent dt, B, C
        self.x_proj = nn.Linear(
            self.intermediate_size, self.dt_rank + self.state_size * 2, bias=False
        )

        # Time step projection (discretization)
        self.dt_proj = nn.Linear(self.dt_rank, self.intermediate_size, bias=True)

        # S4D real initialization for A
        A = torch.arange(1, self.state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        # D is the skip connection coefficient
        self.D = nn.Parameter(torch.ones(self.intermediate_size))

        # Output projection
        self.out_proj = nn.Linear(self.intermediate_size, width, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_dt_proj()

    def _init_dt_proj(self):
        """Initialize dt_proj bias for stable discretization."""
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize bias so dt starts in reasonable range [0.001, 0.1]
        dt = torch.exp(
            torch.rand(self.intermediate_size) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        # Inverse of softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, width]
            mask: [B, L] padding mask (True = masked)
        Returns:
            [B, L, width]
        """
        B, L, _ = x.shape

        # 1. Input projection with gating
        xz = self.in_proj(x)  # [B, L, 2*intermediate]
        x_branch, z = xz.chunk(2, dim=-1)  # each [B, L, intermediate]

        # Apply mask to x_branch before conv
        if mask is not None:
            x_branch = x_branch.masked_fill(mask.unsqueeze(-1), 0.0)

        # 2. Deformable conv + SE for local context
        x_conv = self.conv(x_branch)  # [B, L, intermediate]
        x_conv = self.conv_norm(x_conv)
        x_conv = self.se(x_conv, mask)
        x_conv = self.conv_act(x_conv)

        if mask is not None:
            x_conv = x_conv.masked_fill(mask.unsqueeze(-1), 0.0)

        # 3. Selective SSM parameters
        ssm_params = self.x_proj(x_conv)  # [B, L, dt_rank + 2*state_size]
        dt, B_param, C_param = ssm_params.split(
            [self.dt_rank, self.state_size, self.state_size], dim=-1
        )

        # 4. Discretization
        dt = self.dt_proj(dt)  # [B, L, intermediate]
        dt = F.softplus(dt)  # [B, L, intermediate]

        # 5. SSM scan
        A = -torch.exp(self.A_log.float())  # [intermediate, state_size]

        # Discretize A and B
        # dA = exp(A * dt)
        # dB = dt * B
        dA = torch.exp(
            A[None, None, :, :] * dt[:, :, :, None]
        )  # [B, L, intermediate, state_size]
        dB = (
            dt[:, :, :, None] * B_param[:, None, :, :].float()
        )  # [B, L, intermediate, state_size]

        # Input-weighted B
        dB_x = dB * x_conv[:, :, :, None].float()  # [B, L, intermediate, state_size]

        # Sequential scan (pure PyTorch, works on all devices)
        ssm_state = torch.zeros(
            B,
            self.intermediate_size,
            self.state_size,
            device=x.device,
            dtype=torch.float32,
        )
        outputs = []

        for i in range(L):
            ssm_state = (
                dA[:, i] * ssm_state + dB_x[:, i]
            )  # [B, intermediate, state_size]
            y_i = torch.einsum(
                "bis,bs->bi", ssm_state, C_param[:, i].float()
            )  # [B, intermediate]
            outputs.append(y_i)

        y = torch.stack(outputs, dim=1)  # [B, L, intermediate]

        # 6. Skip connection and gating
        y = y + x_conv * self.D[None, None, :]
        y = y * F.silu(z)  # Gate with z branch

        # 7. Output projection
        out = self.out_proj(y.to(x.dtype))
        return self.dropout(out)


class SSMBlock(nn.Module):
    """SSM block with pre-norm and residual connection."""

    def __init__(
        self,
        width: int,
        state_size: int = 16,
        conv_kernel: int = 7,
        conv_groups: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = RMSNorm(width)
        self.mixer = SSMMixer(
            width, state_size, conv_kernel, conv_groups, expand, dropout=dropout
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return x + self.mixer(self.norm(x), mask)


class SSMBackbone(nn.Module):
    """Stack of SSM blocks."""

    def __init__(
        self,
        width: int,
        depth: int,
        state_size: int = 16,
        conv_kernel: int = 7,
        conv_groups: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                SSMBlock(width, state_size, conv_kernel, conv_groups, expand, dropout)
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def apply_rotary_emb(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to x using cumulative angles.
    x: [..., N] where N is even
    angles: [..., N/2]
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class SSMMixer3(nn.Module):
    """
    Mamba-3 style SSM mixer with proper multi-head structure.

    Core features:
    - Trapezoidal discretization (lookback term)
    - Data-dependent RoPE for complex state dynamics
    - BC biases with QK-norm (per-head)
    - Multi-head structure matching Mamba-2/3

    Optional: Branched deformable conv + SE enhancement
    Optional: Adaptive conv with Gaussian envelope
    """

    def __init__(
        self,
        width: int,
        state_size: int = 64,
        n_heads: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_conv: bool = False,
        conv_kernel: int = 15,
        conv_groups: int = 4,
        adaptive_conv: bool = False,
        depthwise_conv: bool = False,
        init_sigma: float = 0.3,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
    ):
        super().__init__()
        self.width = width
        self.state_size = state_size
        self.n_heads = n_heads
        self.expand = expand
        self.intermediate_size = width * expand
        self.head_dim = self.intermediate_size // n_heads
        self.use_conv = use_conv
        self.adaptive_conv = adaptive_conv and use_conv

        assert self.intermediate_size % n_heads == 0, (
            "intermediate_size must be divisible by n_heads"
        )
        assert state_size % 2 == 0, "state_size must be even for RoPE"

        self.in_proj = nn.Linear(width, self.intermediate_size * 2, bias=False)

        if use_conv:
            if adaptive_conv:
                self.conv = AdaptiveDeformConv1d(
                    channels=self.intermediate_size,
                    kernel_size=conv_kernel,
                    groups=conv_groups,
                    init_sigma=init_sigma,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    depthwise=depthwise_conv,
                )
            else:
                self.conv = DeformConv1d(
                    self.intermediate_size, conv_kernel, conv_groups
                )
                self.conv_norm = nn.LayerNorm(self.intermediate_size)
                self.se = SEBlock(self.intermediate_size)

        # B, C projections -> (n_heads * state_size)
        self.B_proj = nn.Linear(
            self.intermediate_size, n_heads * state_size, bias=False
        )
        self.C_proj = nn.Linear(
            self.intermediate_size, n_heads * state_size, bias=False
        )

        # BC biases - per head, initialized to ones
        self.b_bias = nn.Parameter(torch.ones(n_heads, state_size))
        self.c_bias = nn.Parameter(torch.ones(n_heads, state_size))

        # QK-style normalization
        self.norm_b = RMSNorm(state_size)
        self.norm_c = RMSNorm(state_size)

        # dt projection - per head
        self.dt_proj = nn.Linear(self.intermediate_size, n_heads, bias=True)
        self.A_log = nn.Parameter(torch.zeros(n_heads))

        # Theta for data-dependent RoPE - per head
        self.theta_proj = nn.Linear(
            self.intermediate_size, n_heads * (state_size // 2), bias=False
        )

        # Lambda for trapezoidal - scalar per head (not per channel!)
        self.lambda_proj = nn.Linear(self.intermediate_size, n_heads, bias=True)

        # Skip connection - per head
        self.D = nn.Parameter(torch.ones(n_heads, self.head_dim))

        self.out_proj = nn.Linear(self.intermediate_size, width, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_params()

    def _init_params(self):
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.lambda_proj.weight)
        nn.init.constant_(self.lambda_proj.bias, 2.0)  # sigmoid(2) ≈ 0.88, near Euler

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        biases: AdaptiveConvBiases | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, L, _ = x.shape
        H = self.n_heads
        N = self.state_size
        P = self.head_dim

        xz = self.in_proj(x)
        check_nan(xz, "ssm3_in_proj")
        x_branch, z = xz.chunk(2, dim=-1)

        if mask is not None:
            x_branch = x_branch.masked_fill(mask.unsqueeze(-1), 0.0)

        aux_losses = {}
        if self.use_conv:
            if self.adaptive_conv:
                x_ssm, aux_losses = self.conv(x_branch, mask, biases)
                check_nan(x_ssm, "ssm3_adaptive_conv")
                x_ssm = F.silu(x_ssm)
            else:
                x_ssm = self.conv(x_branch)
                x_ssm = self.conv_norm(x_ssm)
                x_ssm = self.se(x_ssm, mask)
                x_ssm = F.silu(x_ssm)
            check_nan(x_ssm, "ssm3_conv")
            if mask is not None:
                x_ssm = x_ssm.masked_fill(mask.unsqueeze(-1), 0.0)
        else:
            x_ssm = x_branch

        B_param = self.B_proj(x_ssm).view(B, L, H, N)
        C_param = self.C_proj(x_ssm).view(B, L, H, N)

        B_param = self.norm_b(B_param) + self.b_bias
        C_param = self.norm_c(C_param) + self.c_bias
        check_nan(B_param, "ssm3_B_param")
        check_nan(C_param, "ssm3_C_param")

        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log.float())
        check_nan(dt, "ssm3_dt")
        check_nan(A, "ssm3_A")

        theta = self.theta_proj(x_ssm).view(B, L, H, N // 2)
        delta_theta = dt.unsqueeze(-1) * theta
        cum_theta = torch.cumsum(delta_theta, dim=1)
        check_nan(cum_theta, "ssm3_cum_theta")

        B_rot = apply_rotary_emb(B_param, cum_theta)
        C_rot = apply_rotary_emb(C_param, cum_theta)
        check_nan(B_rot, "ssm3_B_rot")
        check_nan(C_rot, "ssm3_C_rot")

        lam = torch.sigmoid(self.lambda_proj(x_ssm))
        alpha = torch.exp(dt * A)
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt
        check_nan(alpha, "ssm3_alpha")
        check_nan(beta, "ssm3_beta")
        check_nan(gamma, "ssm3_gamma")

        x_heads = x_ssm.view(B, L, H, P)

        y = self._ssm_scan(x_heads, B_rot, C_rot, alpha, beta, gamma, mask)
        check_nan(y, "ssm3_scan_output")

        # Skip connection and gating
        y = y.view(B, L, H, P)
        y = y + x_heads * self.D
        y = y.view(B, L, self.intermediate_size)
        y = y * F.silu(z)

        return self.dropout(self.out_proj(y)), aux_losses

    def _ssm_scan(
        self,
        x: torch.Tensor,
        B_rot: torch.Tensor,
        C_rot: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        mask: torch.Tensor | None,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        batch, L, H, P = x.shape
        N = self.state_size
        device = x.device
        dtype = x.dtype

        B_rot = B_rot.float()
        C_rot = C_rot.float()
        x = x.float()
        alpha = alpha.float()
        beta = beta.float()
        gamma = gamma.float()

        Bx = torch.einsum("blhn,blhp->blhnp", B_rot, x)
        Bx_prev = F.pad(Bx[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        u = gamma[:, :, :, None, None] * Bx + beta[:, :, :, None, None] * Bx_prev
        check_nan(u, "scan_u")

        outputs = torch.empty(batch, L, H, P, device=device, dtype=torch.float32)
        state = torch.zeros(batch, H, N, P, device=device, dtype=torch.float32)

        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            K = chunk_end - chunk_start

            alpha_chunk = alpha[:, chunk_start:chunk_end]
            u_chunk = u[:, chunk_start:chunk_end]
            C_chunk = C_rot[:, chunk_start:chunk_end]

            decay_matrix = self._build_decay_matrix(alpha_chunk, K, device)
            check_nan(decay_matrix, f"scan_decay_matrix_chunk{chunk_start}")

            h_from_input = torch.einsum("bijh,bjhnp->bihnp", decay_matrix, u_chunk)
            check_nan(h_from_input, f"scan_h_from_input_chunk{chunk_start}")

            carry = torch.cumprod(alpha_chunk, dim=1)
            check_nan(carry, f"scan_carry_chunk{chunk_start}")
            h_from_state = torch.einsum("bih,bhnp->bihnp", carry, state)

            h_chunk = h_from_input + h_from_state
            check_nan(h_chunk, f"scan_h_chunk{chunk_start}")

            y_chunk = torch.einsum("blhnp,blhn->blhp", h_chunk, C_chunk)
            check_nan(y_chunk, f"scan_y_chunk{chunk_start}")
            outputs[:, chunk_start:chunk_end] = y_chunk

            state = h_chunk[:, -1]

        if mask is not None:
            outputs = outputs.masked_fill(mask[:, :, None, None], 0.0)

        return outputs.to(dtype)

    def _build_decay_matrix(
        self, alpha: torch.Tensor, K: int, device: torch.device
    ) -> torch.Tensor:
        batch, _, H = alpha.shape

        check_nan(alpha, "decay_alpha_input")
        log_alpha = torch.log(alpha.clamp(min=1e-8))
        check_nan(log_alpha, "decay_log_alpha")
        cumsum = torch.cumsum(log_alpha, dim=1)
        check_nan(cumsum, "decay_cumsum")

        cumsum_i = cumsum.unsqueeze(2)
        cumsum_j = cumsum.unsqueeze(1)

        log_decay = cumsum_i - cumsum_j
        check_nan(log_decay, "decay_log_decay_pre_mask")

        causal_mask = torch.tril(torch.ones(K, K, device=device, dtype=torch.bool))
        log_decay = log_decay.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(-1), float("-inf")
        )

        result = torch.exp(log_decay)
        check_nan(result, "decay_matrix_result")
        return result


class SSMBlock3(nn.Module):
    def __init__(
        self,
        width: int,
        state_size: int = 64,
        n_heads: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_conv: bool = False,
        conv_kernel: int = 15,
        conv_groups: int = 4,
        adaptive_conv: bool = False,
        depthwise_conv: bool = False,
        init_sigma: float = 0.3,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
    ):
        super().__init__()
        self.norm = RMSNorm(width)
        self.mixer = SSMMixer3(
            width,
            state_size,
            n_heads,
            expand,
            dropout,
            use_conv,
            conv_kernel,
            conv_groups,
            adaptive_conv,
            depthwise_conv,
            init_sigma,
            min_sigma,
            max_sigma,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        biases: AdaptiveConvBiases | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out, aux_losses = self.mixer(x, mask, biases)
        return self.norm(x + out), aux_losses


class SSMBackbone3(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        state_size: int = 64,
        n_heads: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_conv: bool = False,
        conv_kernel: int = 7,
        conv_groups: int = 4,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                SSMBlock3(
                    width,
                    state_size,
                    n_heads,
                    expand,
                    dropout,
                    use_conv,
                    conv_kernel,
                    conv_groups,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MultiKernelSSMBlock(nn.Module):
    def __init__(
        self,
        width: int,
        kernel_sizes: tuple[int, ...] = (3, 5, 7, 9),
        state_size: int = 64,
        n_heads: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        conv_groups: int = 4,
        num_features: int = 4,
        adaptive_conv: bool = False,
        depthwise_conv: bool = False,
        context_dim: int = 0,
        adaptive_kernel_size: int = 15,
        init_sigmas: tuple[float, ...] | None = None,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
        attn_window_size: int = 512,
        attn_use_rope: bool = True,
    ):
        super().__init__()
        self.width = width
        self.adaptive_conv = adaptive_conv
        self.context_dim = context_dim

        if adaptive_conv and init_sigmas is not None:
            self.n_branches = len(init_sigmas)
        else:
            self.n_branches = len(kernel_sizes)

        self.kernel_sizes = kernel_sizes

        scale = self.n_branches**0.5
        branch_heads = max(1, round(n_heads / scale))
        self.branch_width = round(width / scale / branch_heads) * branch_heads
        self.total_width = self.branch_width * self.n_branches

        branch_state = max(8, round(state_size / scale))
        if branch_state % 2 != 0:
            branch_state += 1

        intermediate_size = self.branch_width * expand
        branch_groups = max(1, min(conv_groups, self.branch_width // 4))
        while self.branch_width % branch_groups != 0 and branch_groups > 1:
            branch_groups -= 1

        self.in_proj = nn.Linear(width, self.total_width)

        if adaptive_conv and init_sigmas is not None:
            self.branches = nn.ModuleList(
                [
                    AdaptiveConvBranch(
                        self.branch_width,
                        init_sigma=sigma,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        groups=branch_groups,
                        kernel_size=adaptive_kernel_size,
                    )
                    for sigma in init_sigmas
                ]
            )
        else:
            self.branches = nn.ModuleList(
                [
                    AdaptiveConvBranch(
                        self.branch_width,
                        init_sigma=0.3,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        groups=branch_groups,
                        kernel_size=max(ks, 3) if ks != 0 else 7,
                    )
                    for ks in kernel_sizes
                ]
            )

        self.proj_down = nn.Linear(self.total_width, width)
        self.norm_merge = RMSNorm(width)

        self.merge_attn = AttentionBlock(
            width,
            n_heads,
            ffn_mult=4,
            dropout=dropout,
            window_size=attn_window_size,
            use_rope=attn_use_rope,
        )

        self.ssm = SSMMixer3(
            width,
            state_size,
            n_heads,
            expand,
            dropout,
            use_conv=False,
            adaptive_conv=False,
        )
        self.norm_ssm = RMSNorm(width)

        self.pool_queries = nn.Parameter(torch.randn(1, width))
        self.pool_scale = width**-0.5

        if context_dim > 0:
            self.context_gate = nn.Linear(context_dim, width)

        self.feature_proj = nn.Sequential(
            nn.Linear(width, num_features),
            nn.SiLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        mask: torch.Tensor | None = None,
        biases: AdaptiveConvBiases | None = None,
        pooler_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, L, _ = x.shape
        h = self.in_proj(x)

        chunks = h.chunk(self.n_branches, dim=-1)

        all_aux_losses: list[dict[str, torch.Tensor]] = []
        branch_outputs = []
        for i, (branch, chunk) in enumerate(zip(self.branches, chunks)):
            if biases is not None:
                branch_bias = AdaptiveConvBiases(
                    sigma=biases.sigma[:, i : i + 1].squeeze(-1)
                    if biases.sigma is not None
                    else None,
                    offset_scale=biases.offset_scale[:, i : i + 1].squeeze(-1)
                    if biases.offset_scale is not None
                    else None,
                    omega=biases.omega[:, i : i + 1].squeeze(-1)
                    if biases.omega is not None
                    else None,
                )
                out, aux = branch(chunk, mask, branch_bias)
            else:
                out, aux = branch(chunk, mask)
            branch_outputs.append(out)
            all_aux_losses.append(aux)

        combined = torch.cat(branch_outputs, dim=-1)

        aux_losses: dict[str, torch.Tensor] = {}
        if all_aux_losses:
            for key in all_aux_losses[0]:
                total = torch.stack([d[key] for d in all_aux_losses]).mean()
                aux_losses[key] = total

        h = self.proj_down(combined)
        h = self.norm_merge(h + residual)

        h = self.merge_attn(h, mask)

        h_res = h
        h, ssm_aux = self.ssm(h, mask)
        for k, v in ssm_aux.items():
            aux_losses[f"ssm_{k}"] = v
        h = self.norm_ssm(h + h_res)

        q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
        if pooler_context is not None and self.context_dim > 0:
            context_bias = self.context_gate(pooler_context).unsqueeze(1)
            q = q + context_bias

        attn = torch.bmm(q, h.transpose(1, 2)) * self.pool_scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled = torch.bmm(attn, h).squeeze(1)
        features = self.feature_proj(pooled)

        return self.dropout(h), features, aux_losses
