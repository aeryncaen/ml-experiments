import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

DEBUG_NAN = True


def check_nan(t: torch.Tensor, name: str) -> None:
    if DEBUG_NAN and t.device.type == "cuda" and torch.isnan(t).any():
        print(f"NaN detected in {name}, shape={t.shape}, device={t.device}")
        print(f"  nan count: {torch.isnan(t).sum().item()}")
        print(f"  inf count: {torch.isinf(t).sum().item()}")
        raise RuntimeError(f"NaN in {name}")


@dataclass
class AdaptiveConvBiases2d:
    sigma: torch.Tensor | None = None
    offset_scale: torch.Tensor | None = None
    omega: torch.Tensor | None = None
    se_bias: torch.Tensor | None = None


class RotaryEmbedding2d(nn.Module):
    inv_freq_h: torch.Tensor
    inv_freq_w: torch.Tensor
    cos_cache_h: torch.Tensor
    sin_cache_h: torch.Tensor
    cos_cache_w: torch.Tensor
    sin_cache_w: torch.Tensor

    def __init__(self, dim: int, base: int = 10000, max_size: int = 256):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq_h", inv_freq)
        self.register_buffer("inv_freq_w", inv_freq.clone())
        self._build_cache(max_size)

    def _build_cache(self, size: int) -> None:
        pos = torch.arange(size, dtype=torch.float32, device=self.inv_freq_h.device)
        freqs_h = torch.outer(pos, self.inv_freq_h)
        freqs_w = torch.outer(pos, self.inv_freq_w)
        self.register_buffer("cos_cache_h", freqs_h.cos(), persistent=False)
        self.register_buffer("sin_cache_h", freqs_h.sin(), persistent=False)
        self.register_buffer("cos_cache_w", freqs_w.cos(), persistent=False)
        self.register_buffer("sin_cache_w", freqs_w.sin(), persistent=False)

    def forward(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if height > self.cos_cache_h.shape[0] or width > self.cos_cache_w.shape[0]:
            self._build_cache(max(height, width))
        cos_h = self.cos_cache_h[:height].to(device)
        sin_h = self.sin_cache_h[:height].to(device)
        cos_w = self.cos_cache_w[:width].to(device)
        sin_w = self.sin_cache_w[:width].to(device)
        return cos_h, sin_h, cos_w, sin_w


def apply_rotary_pos_emb_2d(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> torch.Tensor:
    B, num_heads, H, W, head_dim = x.shape
    half_dim = head_dim // 2

    x_h = x[..., :half_dim]
    x_w = x[..., half_dim:]

    cos_h = cos_h.view(1, 1, H, 1, -1)
    sin_h = sin_h.view(1, 1, H, 1, -1)
    cos_w = cos_w.view(1, 1, 1, W, -1)
    sin_w = sin_w.view(1, 1, 1, W, -1)

    x_h1, x_h2 = x_h[..., 0::2], x_h[..., 1::2]
    rot_h1 = x_h1 * cos_h - x_h2 * sin_h
    rot_h2 = x_h1 * sin_h + x_h2 * cos_h
    x_h = torch.stack([rot_h1, rot_h2], dim=-1).flatten(-2)

    x_w1, x_w2 = x_w[..., 0::2], x_w[..., 1::2]
    rot_w1 = x_w1 * cos_w - x_w2 * sin_w
    rot_w2 = x_w1 * sin_w + x_w2 * cos_w
    x_w = torch.stack([rot_w1, rot_w2], dim=-1).flatten(-2)

    return torch.cat([x_h, x_w], dim=-1)


def apply_rotary_pos_emb_qk_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rot = apply_rotary_pos_emb_2d(q, cos_h, sin_h, cos_w, sin_w)
    k_rot = apply_rotary_pos_emb_2d(k, cos_h, sin_h, cos_w, sin_w)
    return q_rot, k_rot


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(width * mult * 2 / 3)
        self.w1 = nn.Linear(width, hidden)
        self.w2 = nn.Linear(width, hidden)
        self.w3 = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class SEBlock2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, H, W, C = x.shape
        if mask is not None:
            mask_2d = mask.view(B, H, W, 1).float()
            x_masked = x * (1 - mask_2d)
            valid_count = (1 - mask_2d).sum(dim=(1, 2), keepdim=True).clamp(min=1)
            pooled = x_masked.sum(dim=(1, 2), keepdim=True) / valid_count
        else:
            pooled = x.mean(dim=(1, 2), keepdim=True)

        w = self.fc2(F.silu(self.fc1(pooled)))
        if bias is not None:
            w = w + bias.view(B, 1, 1, C)
        scale = torch.sigmoid(w)
        return x * scale


class DeformConv2d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        groups: int = 1,
        offset_scale: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.group_channels = channels // groups
        self.offset_scale = offset_scale
        K = kernel_size * kernel_size

        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

        self.offset_net = nn.Linear(channels, groups * K * 2)
        self.mask_net = nn.Linear(channels, groups * K)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        constant_(self.offset_net.weight.data, 0.0)
        constant_(self.offset_net.bias.data, 0.0)
        constant_(self.mask_net.weight.data, 0.0)
        constant_(self.mask_net.bias.data, 0.0)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, H, W, C = x.shape
        G = self.groups
        gc = self.group_channels
        K = self.kernel_size
        K2 = K * K

        x_proj = self.input_proj(x)

        x_dw = x.permute(0, 3, 1, 2).contiguous()
        x_dw = self.dw_conv(x_dw)
        x_dw = x_dw.permute(0, 2, 3, 1).contiguous()

        offsets = self.offset_net(x_dw).view(B, H, W, G, K2, 2) * self.offset_scale
        attn_mask = F.softmax(self.mask_net(x_dw).view(B, H, W, G, K2), dim=-1)
        attn_mask = torch.nan_to_num(attn_mask, nan=0.0)

        ref_h = torch.arange(-(K // 2), K // 2 + 1, device=x.device).float()
        ref_w = torch.arange(-(K // 2), K // 2 + 1, device=x.device).float()
        ref_grid = torch.stack(torch.meshgrid(ref_h, ref_w, indexing="ij"), dim=-1)
        ref_grid = ref_grid.view(1, 1, 1, 1, K2, 2)

        pos_h = torch.arange(H, device=x.device).float().view(1, H, 1, 1, 1, 1)
        pos_w = torch.arange(W, device=x.device).float().view(1, 1, W, 1, 1, 1)

        abs_pos = ref_grid + offsets
        abs_pos[..., 0] = abs_pos[..., 0] + pos_h.squeeze(-1)
        abs_pos[..., 1] = abs_pos[..., 1] + pos_w.squeeze(-1)

        abs_pos[..., 0] = abs_pos[..., 0].clamp(0, H - 1)
        abs_pos[..., 1] = abs_pos[..., 1].clamp(0, W - 1)

        x_grouped = x_proj.view(B, H, W, G, gc)
        output = torch.zeros(B, H, W, G, gc, device=x.device, dtype=x.dtype)

        for k in range(K2):
            pos_k = abs_pos[:, :, :, :, k]

            h_floor = pos_k[..., 0].long().clamp(0, H - 1)
            w_floor = pos_k[..., 1].long().clamp(0, W - 1)
            h_ceil = (h_floor + 1).clamp(0, H - 1)
            w_ceil = (w_floor + 1).clamp(0, W - 1)

            h_weight = pos_k[..., 0] - h_floor.float()
            w_weight = pos_k[..., 1] - w_floor.float()

            batch_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
            group_idx = torch.arange(G, device=x.device).view(1, 1, 1, G)

            v_ff = x_grouped[batch_idx, h_floor, w_floor, group_idx]
            v_fc = x_grouped[batch_idx, h_floor, w_ceil, group_idx]
            v_cf = x_grouped[batch_idx, h_ceil, w_floor, group_idx]
            v_cc = x_grouped[batch_idx, h_ceil, w_ceil, group_idx]

            w_ff = (1 - h_weight) * (1 - w_weight)
            w_fc = (1 - h_weight) * w_weight
            w_cf = h_weight * (1 - w_weight)
            w_cc = h_weight * w_weight

            sampled = (
                v_ff * w_ff.unsqueeze(-1)
                + v_fc * w_fc.unsqueeze(-1)
                + v_cf * w_cf.unsqueeze(-1)
                + v_cc * w_cc.unsqueeze(-1)
            )

            output = output + sampled * attn_mask[:, :, :, :, k : k + 1]

        output = output.reshape(B, H, W, C)
        return self.output_proj(output)


class AdaptiveDeformConv2d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        groups: int = 1,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
        init_sigma: float = 0.3,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.group_channels = channels // groups
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.omega_max = 2.0

        init_raw = math.log(math.exp(init_sigma) - 1.0) if init_sigma > 0 else 0.0
        self.raw_sigma = nn.Parameter(torch.tensor(init_raw))
        self.base_offset_scale = nn.Parameter(torch.tensor(0.1))

        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

        K2 = kernel_size * kernel_size
        self.offset_net = nn.Linear(channels, groups * K2 * 2)
        self.mask_net = nn.Linear(channels, groups * K2)

        self.kernel_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, groups * self.group_channels),
        )

        self.se = SEBlock2d(channels)

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

    def _compute_envelope(
        self, grid_h: torch.Tensor, grid_w: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        gh = grid_h.view(1, -1, 1)
        gw = grid_w.view(1, 1, -1)

        if sigma.dim() > 0:
            sigma = sigma.view(-1, 1, 1)

        envelope = torch.exp(-(gh**2 + gw**2) / (2 * sigma**2))
        envelope = envelope / envelope.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        return envelope

    def _get_kernel_weights(
        self,
        grid_h: torch.Tensor,
        grid_w: torch.Tensor,
        omega_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        K = grid_h.shape[0]
        gh = grid_h.view(-1, 1)
        gw = grid_w.view(1, -1).expand(K, -1)
        positions = torch.stack([gh.expand(K, K).flatten(), gw.flatten()], dim=-1) * 2

        if omega_bias is not None:
            omega_scale = 1.0 + torch.tanh(omega_bias.mean()) * self.omega_max
            positions = positions * omega_scale

        weights = self.kernel_net(positions)
        return weights.view(K, K, self.groups, self.group_channels)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        biases: AdaptiveConvBiases2d | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, H, W, C = x.shape
        G = self.groups
        gc = self.group_channels
        K = self.kernel_size
        K2 = K * K

        sigma_bias = biases.sigma if biases else None
        offset_scale_bias = biases.offset_scale if biases else None
        omega_bias = biases.omega if biases else None

        raw = self.raw_sigma
        if sigma_bias is not None:
            raw = raw + torch.nan_to_num(sigma_bias, nan=0.0)
        sigma = F.softplus(raw).clamp(min=1e-3, max=self.max_sigma)

        grid = torch.linspace(-0.5, 0.5, K, device=x.device)

        x_proj = self.input_proj(x)

        x_dw = x.permute(0, 3, 1, 2).contiguous()
        x_dw = self.dw_conv(x_dw)
        x_dw = x_dw.permute(0, 2, 3, 1).contiguous()

        envelope = self._compute_envelope(grid, grid, sigma)
        kernel_weights = self._get_kernel_weights(grid, grid, omega_bias)

        offset_scale = self.base_offset_scale
        if offset_scale_bias is not None:
            offset_scale = offset_scale * (1.0 + 0.2 * offset_scale_bias.mean())

        offsets = self.offset_net(x_dw).view(B, H, W, G, K2, 2) * offset_scale
        raw_mask = self.mask_net(x_dw).view(B, H, W, G, K2)

        if envelope.shape[0] == 1:
            envelope_flat = envelope.view(1, 1, 1, 1, K2)
        else:
            envelope_flat = envelope.view(B, 1, 1, 1, K2)
        raw_mask = raw_mask * envelope_flat

        attn_mask = F.softmax(raw_mask, dim=-1)
        attn_mask = torch.nan_to_num(attn_mask, nan=0.0)

        ref_h = torch.linspace(-(K // 2), K // 2, K, device=x.device)
        ref_w = torch.linspace(-(K // 2), K // 2, K, device=x.device)
        ref_grid = torch.stack(
            torch.meshgrid(ref_h, ref_w, indexing="ij"), dim=-1
        ).view(K2, 2)

        pos_h = torch.arange(H, device=x.device).float().view(1, H, 1, 1, 1)
        pos_w = torch.arange(W, device=x.device).float().view(1, 1, W, 1, 1)

        x_grouped = x_proj.view(B, H, W, G, gc)

        abs_h = pos_h + ref_grid[:, 0].view(1, 1, 1, 1, K2) + offsets[..., 0]
        abs_w = pos_w + ref_grid[:, 1].view(1, 1, 1, 1, K2) + offsets[..., 1]

        abs_h_clamped = abs_h.clamp(0, H - 1)
        abs_w_clamped = abs_w.clamp(0, W - 1)

        h_floor = abs_h_clamped.long().clamp(0, H - 1)
        w_floor = abs_w_clamped.long().clamp(0, W - 1)
        h_ceil = (h_floor + 1).clamp(0, H - 1)
        w_ceil = (w_floor + 1).clamp(0, W - 1)

        h_weight = abs_h_clamped - h_floor.float()
        w_weight = abs_w_clamped - w_floor.float()

        oob = (abs_h < 0) | (abs_h > H - 1) | (abs_w < 0) | (abs_w > W - 1)
        valid = (~oob).float()

        b_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1, 1)
        g_idx = torch.arange(G, device=x.device).view(1, 1, 1, G, 1)

        v_ff = x_grouped[b_idx, h_floor, w_floor, g_idx]
        v_fc = x_grouped[b_idx, h_floor, w_ceil, g_idx]
        v_cf = x_grouped[b_idx, h_ceil, w_floor, g_idx]
        v_cc = x_grouped[b_idx, h_ceil, w_ceil, g_idx]

        w_ff = ((1 - h_weight) * (1 - w_weight) * valid).unsqueeze(-1)
        w_fc = ((1 - h_weight) * w_weight * valid).unsqueeze(-1)
        w_cf = (h_weight * (1 - w_weight) * valid).unsqueeze(-1)
        w_cc = (h_weight * w_weight * valid).unsqueeze(-1)

        sampled = v_ff * w_ff + v_fc * w_fc + v_cf * w_cf + v_cc * w_cc

        kw = kernel_weights.view(1, 1, 1, G, K2, gc)
        sampled = sampled * kw

        output = (sampled * attn_mask.unsqueeze(-1)).sum(dim=4)
        output = output.reshape(B, H, W, C)
        output = self.se(output, mask)

        offset_reg = (offsets**2).mean()
        entropy = -(attn_mask * (attn_mask + 1e-8).log()).sum(dim=-1).mean()
        aux_losses = {"offset_reg": offset_reg, "entropy_reg": -entropy}

        return self.output_proj(output), aux_losses


class AdaptiveConvBranch2d(nn.Module):
    def __init__(
        self,
        width: int,
        init_sigma: float = 0.3,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
        groups: int = 4,
        kernel_size: int = 15,
        se_reduction: int = 4,
        se_bias_dim: int | None = None,
    ):
        super().__init__()
        self.width = width
        self.conv = AdaptiveDeformConv2d(
            width,
            kernel_size=kernel_size,
            groups=groups,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            init_sigma=init_sigma,
        )
        self.norm = RMSNorm(width)
        self.se = SEBlock2d(width, reduction=se_reduction)
        if se_bias_dim is not None and se_bias_dim != width:
            self.se_bias_proj = nn.Linear(se_bias_dim, width)
            nn.init.zeros_(self.se_bias_proj.weight)
            nn.init.zeros_(self.se_bias_proj.bias)
        else:
            self.se_bias_proj = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        biases: AdaptiveConvBiases2d | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out, aux = self.conv(x, mask, biases)
        out = self.norm(F.silu(out))
        se_bias = None
        if biases is not None and biases.se_bias is not None:
            se_bias = biases.se_bias
            if self.se_bias_proj is not None:
                se_bias = self.se_bias_proj(se_bias)
        out = self.se(out, mask, se_bias)
        return out, aux


class AttentionBlock2d(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        window_size: int = 0,
        use_rope: bool = True,
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
            self.rotary = RotaryEmbedding2d(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, H, W, C = x.shape

        if self.window_size > 0 and (H > self.window_size or W > self.window_size):
            win_h = min(self.window_size, H)
            win_w = min(self.window_size, W)
            x_attn = x[:, :win_h, :win_w]
            mask_attn = mask[:, :win_h, :win_w] if mask is not None else None
        else:
            win_h, win_w = H, W
            x_attn = x
            mask_attn = mask

        L = win_h * win_w
        x_flat = x_attn.reshape(B, L, C)

        qkv = self.qkv(x_flat).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.view(B, win_h, win_w, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )
        k = k.view(B, win_h, win_w, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )
        v = v.view(B, win_h, win_w, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )

        if self.use_rope:
            cos_h, sin_h, cos_w, sin_w = self.rotary(win_h, win_w, x.device)
            q, k = apply_rotary_pos_emb_qk_2d(q, k, cos_h, sin_h, cos_w, sin_w)

        q_flat = q.reshape(B, self.num_heads, L, self.head_dim)
        k_flat = k.reshape(B, self.num_heads, L, self.head_dim)
        v_flat = v.reshape(B, self.num_heads, L, self.head_dim)

        attn_mask = None
        if mask_attn is not None:
            mask_flat = mask_attn.reshape(B, L)
            attn_mask = (
                mask_flat.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, L, L)
            )

        scale = self.head_dim**-0.5
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        if self.training and self.dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)
        out = torch.matmul(attn_weights, v_flat)

        out = out.transpose(1, 2).reshape(B, L, self.width)
        out = out.view(B, win_h, win_w, C)

        if self.window_size > 0 and (H > self.window_size or W > self.window_size):
            top_left = self.norm1(
                x_attn
                + self.dropout(self.out(out.reshape(B, L, C)).view(B, win_h, win_w, C))
            )
            top_left = self.norm2(top_left + self.ffn(top_left))

            if W > win_w:
                top_right = x[:, :win_h, win_w:]
                top_right = self.norm2(top_right + self.ffn(top_right))
                top_row = torch.cat([top_left, top_right], dim=2)
            else:
                top_row = top_left

            if H > win_h:
                bottom_row = x[:, win_h:]
                bottom_row = self.norm2(bottom_row + self.ffn(bottom_row))
                x_out = torch.cat([top_row, bottom_row], dim=1)
            else:
                x_out = top_row
            return x_out
        else:
            x = self.norm1(
                x + self.dropout(self.out(out.reshape(B, L, C)).view(B, H, W, C))
            )
            x = self.norm2(x + self.ffn(x))
            return x


class ContextualAttentionBlock2d(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        n_branches: int = 0,
        context_dim: int = 0,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        window_size: int = 0,
        use_rope: bool = True,
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
            self.rotary = RotaryEmbedding2d(self.head_dim)

        if n_branches > 0 or context_dim > 0:
            self.pool_query = nn.Parameter(torch.randn(1, width))
            self.pool_scale = width**-0.5

        if n_branches > 0:
            self.sigma_proj = nn.Linear(width, n_branches)
            self.offset_scale_proj = nn.Linear(width, n_branches)
            self.omega_proj = nn.Linear(width, n_branches)
            self.se_bias_proj = nn.Linear(width, n_branches * width)

            for proj in [self.sigma_proj, self.offset_scale_proj, self.omega_proj]:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
            nn.init.zeros_(self.se_bias_proj.weight)
            nn.init.zeros_(self.se_bias_proj.bias)

        if context_dim > 0:
            self.context_proj = nn.Linear(width, context_dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, AdaptiveConvBiases2d | None, torch.Tensor | None]
    ):
        B, H, W, C = x.shape

        if self.window_size > 0 and (H > self.window_size or W > self.window_size):
            win_h = min(self.window_size, H)
            win_w = min(self.window_size, W)
            x_attn = x[:, :win_h, :win_w]
            mask_attn = mask[:, :win_h, :win_w] if mask is not None else None
        else:
            win_h, win_w = H, W
            x_attn = x
            mask_attn = mask

        L = win_h * win_w
        x_flat = x_attn.reshape(B, L, C)

        check_nan(x_flat, "ctx_attn2d_x_input")
        qkv = self.qkv(x_flat).reshape(B, L, 3, self.num_heads, self.head_dim)
        check_nan(qkv, "ctx_attn2d_qkv")
        q, k, v = qkv.unbind(dim=2)

        q = q.view(B, win_h, win_w, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )
        k = k.view(B, win_h, win_w, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )
        v = v.view(B, win_h, win_w, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )

        if self.use_rope:
            cos_h, sin_h, cos_w, sin_w = self.rotary(win_h, win_w, x.device)
            q, k = apply_rotary_pos_emb_qk_2d(q, k, cos_h, sin_h, cos_w, sin_w)
            check_nan(q, "ctx_attn2d_q_rope")
            check_nan(k, "ctx_attn2d_k_rope")

        q_flat = q.reshape(B, self.num_heads, L, self.head_dim)
        k_flat = k.reshape(B, self.num_heads, L, self.head_dim)
        v_flat = v.reshape(B, self.num_heads, L, self.head_dim)

        attn_mask = None
        if mask_attn is not None:
            mask_flat = mask_attn.reshape(B, L)
            attn_mask = (
                mask_flat.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, L, L)
            )

        scale = self.head_dim**-0.5
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) * scale
        check_nan(scores, "ctx_attn2d_scores")
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
            check_nan(scores, "ctx_attn2d_scores_masked")
        attn_weights = F.softmax(scores, dim=-1)
        check_nan(attn_weights, "ctx_attn2d_weights")
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        if self.training and self.dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)
        out = torch.matmul(attn_weights, v_flat)
        check_nan(out, "ctx_attn2d_out")

        out = out.transpose(1, 2).reshape(B, L, self.width)
        out = out.view(B, win_h, win_w, C)

        x_attn = self.norm1(
            x_attn
            + self.dropout(self.out(out.reshape(B, L, C)).view(B, win_h, win_w, C))
        )
        check_nan(x_attn, "ctx_attn2d_after_norm1")
        x_attn = self.norm2(x_attn + self.ffn(x_attn))
        check_nan(x_attn, "ctx_attn2d_after_norm2")

        if self.window_size > 0 and (H > self.window_size or W > self.window_size):
            top_left = x_attn
            if W > win_w:
                top_right = x[:, :win_h, win_w:]
                top_right = self.norm2(top_right + self.ffn(top_right))
                top_row = torch.cat([top_left, top_right], dim=2)
            else:
                top_row = top_left

            if H > win_h:
                bottom_row = x[:, win_h:]
                bottom_row = self.norm2(bottom_row + self.ffn(bottom_row))
                x = torch.cat([top_row, bottom_row], dim=1)
            else:
                x = top_row
        else:
            x = x_attn

        if self.n_branches == 0 and self.context_dim == 0:
            return x

        q_pool = self.pool_query.unsqueeze(0).expand(B, -1, -1)
        pool_target = x_attn.reshape(B, win_h * win_w, C)
        pool_mask = (
            mask_attn.reshape(B, win_h * win_w) if mask_attn is not None else None
        )

        attn_scores = torch.bmm(q_pool, pool_target.transpose(1, 2)) * self.pool_scale
        if pool_mask is not None:
            attn_scores = attn_scores.masked_fill(pool_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        pooled = torch.bmm(attn_weights, pool_target).squeeze(1)

        if self.n_branches > 0:
            se_bias = self.se_bias_proj(pooled).view(B, self.n_branches, self.width)
            biases = AdaptiveConvBiases2d(
                sigma=self.sigma_proj(pooled),
                offset_scale=self.offset_scale_proj(pooled),
                omega=self.omega_proj(pooled),
                se_bias=se_bias,
            )
        else:
            biases = None

        context = self.context_proj(pooled) if self.context_dim > 0 else None

        return x, biases, context


class SSMBlock3_2d(nn.Module):
    def __init__(
        self,
        width: int,
        state_size: int = 64,
        n_heads: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_conv: bool = True,
        adaptive_conv: bool = False,
        conv_groups: int = 4,
        kernel_size: int = 15,
        init_sigma: float = 0.3,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
    ):
        super().__init__()
        self.width = width
        self.state_size = state_size
        self.n_heads = n_heads
        self.intermediate_size = width * expand
        self.head_dim = self.intermediate_size // n_heads
        self.use_conv = use_conv
        self.adaptive_conv = adaptive_conv

        assert self.intermediate_size % n_heads == 0
        assert state_size % 2 == 0

        self.in_proj = nn.Linear(width, 2 * self.intermediate_size)
        self.out_proj = nn.Linear(self.intermediate_size, width)

        if use_conv:
            if adaptive_conv:
                self.conv = AdaptiveDeformConv2d(
                    self.intermediate_size,
                    kernel_size=kernel_size,
                    groups=conv_groups,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    init_sigma=init_sigma,
                )
            else:
                self.conv = nn.Conv2d(
                    self.intermediate_size,
                    self.intermediate_size,
                    kernel_size=3,
                    padding=1,
                    groups=self.intermediate_size,
                )
                self.conv_norm = RMSNorm(self.intermediate_size)
                self.se = SEBlock2d(self.intermediate_size)

        H = n_heads
        N = state_size
        P = self.head_dim

        self.B_proj = nn.Linear(self.intermediate_size, H * N)
        self.C_proj = nn.Linear(self.intermediate_size, H * N)
        self.dt_proj = nn.Linear(self.intermediate_size, H)
        self.theta_proj = nn.Linear(self.intermediate_size, H * (N // 2))
        self.lambda_proj = nn.Linear(self.intermediate_size, H)

        self.norm_b = RMSNorm(N)
        self.norm_c = RMSNorm(N)
        self.b_bias = nn.Parameter(torch.zeros(H, N))
        self.c_bias = nn.Parameter(torch.zeros(H, N))

        self.A_log = nn.Parameter(torch.zeros(H))
        self.D = nn.Parameter(torch.ones(H, P))

        self.dropout = nn.Dropout(dropout)

        self._init_dt_proj()

        nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.lambda_proj.weight)
        nn.init.constant_(self.lambda_proj.bias, 2.0)

    def _init_dt_proj(self) -> None:
        import math

        dt_init_std = (self.intermediate_size // self.n_heads) ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        biases: AdaptiveConvBiases2d | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, H_img, W_img, _ = x.shape
        Heads = self.n_heads
        N = self.state_size
        P = self.head_dim

        xz = self.in_proj(x)
        check_nan(xz, "ssm3_2d_in_proj")
        x_branch, z = xz.chunk(2, dim=-1)

        if mask is not None:
            x_branch = x_branch.masked_fill(mask.unsqueeze(-1), 0.0)

        aux_losses = {}
        if self.use_conv:
            if self.adaptive_conv:
                x_ssm, aux_losses = self.conv(x_branch, mask, biases)
                check_nan(x_ssm, "ssm3_2d_adaptive_conv")
                x_ssm = F.silu(x_ssm)
            else:
                x_conv = x_branch.permute(0, 3, 1, 2)
                x_conv = self.conv(x_conv)
                x_ssm = x_conv.permute(0, 2, 3, 1)
                x_ssm = self.conv_norm(x_ssm)
                x_ssm = self.se(x_ssm, mask)
                x_ssm = F.silu(x_ssm)
            check_nan(x_ssm, "ssm3_2d_conv")
            if mask is not None:
                x_ssm = x_ssm.masked_fill(mask.unsqueeze(-1), 0.0)
        else:
            x_ssm = x_branch

        B_param = self.B_proj(x_ssm).view(B, H_img, W_img, Heads, N)
        C_param = self.C_proj(x_ssm).view(B, H_img, W_img, Heads, N)

        B_param = self.norm_b(B_param) + self.b_bias
        C_param = self.norm_c(C_param) + self.c_bias
        check_nan(B_param, "ssm3_2d_B_param")
        check_nan(C_param, "ssm3_2d_C_param")

        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(self.A_log.float())
        check_nan(dt, "ssm3_2d_dt")
        check_nan(A, "ssm3_2d_A")

        x_heads = x_ssm.view(B, H_img, W_img, Heads, P)

        y = self._ssm_scan_4way(x_heads, B_param, C_param, dt, A, mask)
        check_nan(y, "ssm3_2d_scan_output")

        y = y.view(B, H_img, W_img, Heads, P)
        y = y + x_heads * self.D
        y = y.view(B, H_img, W_img, self.intermediate_size)
        y = y * F.silu(z)

        return self.dropout(self.out_proj(y)), aux_losses

    def _ssm_scan_4way(
        self,
        x: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B_batch, H_img, W_img, Heads, P = x.shape
        N = self.state_size
        L = H_img * W_img

        # MPS workaround: when Heads==1, squeeze dt before permute/reshape/flip
        # to avoid MPS Linear backward bug with 4D tensors having trailing dim=1
        dt_squeezed = Heads == 1
        if dt_squeezed:
            dt = dt.squeeze(-1)  # [B, H, W]

        outputs = []

        for direction in range(4):
            if direction == 0:
                x_dir = x.reshape(B_batch, L, Heads, P)
                B_dir = B_param.reshape(B_batch, L, Heads, N)
                C_dir = C_param.reshape(B_batch, L, Heads, N)
                dt_dir = (
                    dt.reshape(B_batch, L)
                    if dt_squeezed
                    else dt.reshape(B_batch, L, Heads)
                )
            elif direction == 1:
                x_dir = x.reshape(B_batch, L, Heads, P).flip(1)
                B_dir = B_param.reshape(B_batch, L, Heads, N).flip(1)
                C_dir = C_param.reshape(B_batch, L, Heads, N).flip(1)
                dt_dir = (
                    dt.reshape(B_batch, L).flip(1)
                    if dt_squeezed
                    else dt.reshape(B_batch, L, Heads).flip(1)
                )
            elif direction == 2:
                x_dir = x.permute(0, 2, 1, 3, 4).reshape(B_batch, L, Heads, P)
                B_dir = B_param.permute(0, 2, 1, 3, 4).reshape(B_batch, L, Heads, N)
                C_dir = C_param.permute(0, 2, 1, 3, 4).reshape(B_batch, L, Heads, N)
                dt_dir = (
                    dt.permute(0, 2, 1).reshape(B_batch, L)
                    if dt_squeezed
                    else dt.permute(0, 2, 1, 3).reshape(B_batch, L, Heads)
                )
            else:
                x_dir = x.permute(0, 2, 1, 3, 4).reshape(B_batch, L, Heads, P).flip(1)
                B_dir = (
                    B_param.permute(0, 2, 1, 3, 4).reshape(B_batch, L, Heads, N).flip(1)
                )
                C_dir = (
                    C_param.permute(0, 2, 1, 3, 4).reshape(B_batch, L, Heads, N).flip(1)
                )
                dt_dir = (
                    dt.permute(0, 2, 1).reshape(B_batch, L).flip(1)
                    if dt_squeezed
                    else dt.permute(0, 2, 1, 3).reshape(B_batch, L, Heads).flip(1)
                )

            # Restore Heads dim for dt if squeezed
            if dt_squeezed:
                dt_dir = dt_dir.unsqueeze(-1)

            y_dir = self._ssm_scan_1d(x_dir, B_dir, C_dir, dt_dir, A)

            if direction == 1:
                y_dir = y_dir.flip(1)
            elif direction == 2:
                y_dir = y_dir.reshape(B_batch, W_img, H_img, Heads, P).permute(
                    0, 2, 1, 3, 4
                )
                y_dir = y_dir.reshape(B_batch, H_img * W_img, Heads, P)
            elif direction == 3:
                y_dir = y_dir.flip(1)
                y_dir = y_dir.reshape(B_batch, W_img, H_img, Heads, P).permute(
                    0, 2, 1, 3, 4
                )
                y_dir = y_dir.reshape(B_batch, H_img * W_img, Heads, P)

            outputs.append(y_dir)

        y = torch.stack(outputs, dim=0).mean(dim=0)
        y = y.reshape(B_batch, H_img, W_img, Heads, P)

        if mask is not None:
            y = y.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        return y

    def _ssm_scan_1d(
        self,
        x: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        chunk_size: int = 784,
    ) -> torch.Tensor:
        batch, L, Heads, P = x.shape
        N = self.state_size
        device = x.device
        dtype = x.dtype

        alpha = torch.exp(dt.unsqueeze(-1).unsqueeze(-1) * A.view(1, 1, Heads, 1, 1))
        check_nan(alpha, "ssm2d_scan_alpha")

        B_param = B_param.float()
        C_param = C_param.float()
        x = x.float()
        alpha = alpha.float()

        Bx = torch.einsum("blhn,blhp->blhnp", B_param, x)
        check_nan(Bx, "ssm2d_scan_Bx")

        outputs = torch.empty(batch, L, Heads, P, device=device, dtype=torch.float32)
        state = torch.zeros(batch, Heads, N, P, device=device, dtype=torch.float32)

        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            K = chunk_end - chunk_start

            alpha_chunk = alpha[:, chunk_start:chunk_end]
            Bx_chunk = Bx[:, chunk_start:chunk_end]
            C_chunk = C_param[:, chunk_start:chunk_end]

            chunk_out, state = self._scan_chunk(
                alpha_chunk, Bx_chunk, C_chunk, state, K
            )
            check_nan(chunk_out, f"ssm2d_scan_chunk{chunk_start}_out")
            outputs[:, chunk_start:chunk_end] = chunk_out

        return outputs.to(dtype)

    def _scan_chunk(
        self,
        alpha: torch.Tensor,
        Bx: torch.Tensor,
        C: torch.Tensor,
        h_prev: torch.Tensor,
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, Heads, N, P = Bx.shape

        alpha_cumprod = alpha.cumprod(dim=1)
        check_nan(alpha_cumprod, "scan_chunk_alpha_cumprod")

        inv_alpha = 1.0 / alpha.clamp(min=1e-8)
        inv_alpha_cumprod = inv_alpha.cumprod(dim=1)
        check_nan(inv_alpha_cumprod, "scan_chunk_inv_alpha_cumprod")

        Bx_scaled = Bx * inv_alpha_cumprod
        Bx_scaled_cumsum = Bx_scaled.cumsum(dim=1)
        check_nan(Bx_scaled_cumsum, "scan_chunk_Bx_cumsum")

        h_all = alpha_cumprod * (h_prev.unsqueeze(1) + Bx_scaled_cumsum)
        check_nan(h_all, "scan_chunk_h_all")

        y = torch.einsum("bkhnp,bkhn->bkhp", h_all, C)
        check_nan(y, "scan_chunk_y")
        h_final = h_all[:, -1]

        return y, h_final


class MultiKernelSSMBlock2d(nn.Module):
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
        context_dim: int = 0,
        adaptive_kernel_size: int = 15,
        init_sigmas: tuple[float, ...] | None = None,
        min_sigma: float = 0.05,
        max_sigma: float = 0.5,
        attn_window_size: int = 64,
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

        branch_groups = max(1, min(conv_groups, self.branch_width // 4))
        while self.branch_width % branch_groups != 0 and branch_groups > 1:
            branch_groups -= 1

        self.in_proj = nn.Linear(width, self.total_width)

        if adaptive_conv and init_sigmas is not None:
            self.branches = nn.ModuleList(
                [
                    AdaptiveConvBranch2d(
                        self.branch_width,
                        init_sigma=sigma,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        groups=branch_groups,
                        kernel_size=adaptive_kernel_size,
                        se_bias_dim=width,
                    )
                    for sigma in init_sigmas
                ]
            )
        else:
            self.branches = nn.ModuleList(
                [
                    AdaptiveConvBranch2d(
                        self.branch_width,
                        init_sigma=0.3,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        groups=branch_groups,
                        kernel_size=max(ks, 3) if ks != 0 else 7,
                        se_bias_dim=width,
                    )
                    for ks in kernel_sizes
                ]
            )

        self.proj_down = nn.Linear(self.total_width, width)
        self.norm_merge = RMSNorm(width)

        self.merge_attn = AttentionBlock2d(
            width,
            num_heads=max(1, width // 8),
            ffn_mult=4,
            dropout=dropout,
            window_size=attn_window_size,
            use_rope=attn_use_rope,
        )

        self.ssm = SSMBlock3_2d(
            width,
            state_size=branch_state,
            n_heads=branch_heads,
            expand=expand,
            dropout=dropout,
            use_conv=False,
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
        biases: AdaptiveConvBiases2d | None = None,
        pooler_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, H, W, _ = x.shape
        check_nan(x, "mkssm2d_input")
        h = self.in_proj(x)
        check_nan(h, "mkssm2d_in_proj")

        chunks = h.chunk(self.n_branches, dim=-1)

        all_aux_losses: list[dict[str, torch.Tensor]] = []
        branch_outputs = []
        for i, (branch, chunk) in enumerate(zip(self.branches, chunks)):
            if biases is not None:
                branch_bias = AdaptiveConvBiases2d(
                    sigma=biases.sigma[:, i : i + 1].squeeze(-1)
                    if biases.sigma is not None
                    else None,
                    offset_scale=biases.offset_scale[:, i : i + 1].squeeze(-1)
                    if biases.offset_scale is not None
                    else None,
                    omega=biases.omega[:, i : i + 1].squeeze(-1)
                    if biases.omega is not None
                    else None,
                    se_bias=biases.se_bias[:, i, :]
                    if biases.se_bias is not None
                    else None,
                )
                out, aux = branch(chunk, mask, branch_bias)
            else:
                out, aux = branch(chunk, mask)
            check_nan(out, f"mkssm2d_branch{i}_out")
            branch_outputs.append(out)
            all_aux_losses.append(aux)

        combined = torch.cat(branch_outputs, dim=-1)
        check_nan(combined, "mkssm2d_combined_branches")

        aux_losses: dict[str, torch.Tensor] = {}
        if all_aux_losses:
            for key in all_aux_losses[0]:
                total = torch.stack([d[key] for d in all_aux_losses]).mean()
                aux_losses[key] = total

        h = self.proj_down(combined)
        check_nan(h, "mkssm2d_proj_down")
        h = self.norm_merge(h + residual)
        check_nan(h, "mkssm2d_norm_merge")

        h = self.merge_attn(h, mask)
        check_nan(h, "mkssm2d_merge_attn")

        h_res = h
        h, ssm_aux = self.ssm(h, mask)
        for k, v in ssm_aux.items():
            aux_losses[f"ssm_{k}"] = v
        h = self.norm_ssm(h + h_res)

        h_flat = h.reshape(B, H * W, -1)
        q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
        if pooler_context is not None and self.context_dim > 0:
            context_bias = self.context_gate(pooler_context).unsqueeze(1)
            q = q + context_bias

        attn = torch.bmm(q, h_flat.transpose(1, 2)) * self.pool_scale
        if mask is not None:
            mask_flat = mask.reshape(B, H * W)
            attn = attn.masked_fill(mask_flat.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled = torch.bmm(attn, h_flat).squeeze(1)
        features = self.feature_proj(pooled)

        return self.dropout(h), features, aux_losses
