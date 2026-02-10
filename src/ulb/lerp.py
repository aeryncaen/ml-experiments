"""Temporal mixing operations for K preprocessing.

These operations blend K tokens with their temporal neighbors before RoPE,
providing SSM-equivalent state mixing capabilities:

- CausalLerp: blends last 1/4 of head_dim with t-1 neighbor (lerp).
- CausalAdd: adds gated t-1 signal (additive, no attenuation).
- KAcausalLerp: lerp with both t-1 and t+1 neighbors.
- KAcausalAdd: additive with both t-1 and t+1 neighbors.
- KTemporalConv: depthwise causal conv on last 1/4 of head_dim.

Gate initialization: weight=0, bias=-2.0 (sigmoid ~ 0.12), so the block
starts near-identity and learns to mix over training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLerp(nn.Module):
    """Causal temporal lerp on the last quarter of channels.

    Blends k[t] with k[t-1] via a content-dependent gate:
        k_mixed = (1 - gate) * k[t] + gate * k[t-1]

    Only applies to the last quarter_dim channels; the rest pass through.

    Args:
        d_model:     Input dimension (for the gate projection from x).
        n_heads:     Number of attention heads.
        quarter_dim: Number of channels to lerp (head_dim // 4).
        init_bias:   Initial gate bias (default -2.0, sigmoid ~ 0.12).
    """

    def __init__(self, d_model: int, n_heads: int, quarter_dim: int, init_bias: float = -2.0):
        super().__init__()
        self.quarter_dim = quarter_dim
        self.n_heads = n_heads
        self.gate_proj = nn.Linear(d_model, n_heads * quarter_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

    def forward(self, k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply causal lerp to K.

        Args:
            k: (B, T, H, head_dim) — key tensor.
            x: (B, T, D) — original input (for gate computation).

        Returns:
            K with last quarter_dim channels blended with t-1.
        """
        b, t, _, _ = k.shape
        if t < 2:
            # KV-cache decode commonly runs one token per step.
            # With no previous token in the local window, causal lerp should
            # not attenuate channels; return identity instead.
            return k
        qd = self.quarter_dim
        gate = torch.sigmoid(self.gate_proj(x)).view(b, t, self.n_heads, qd)

        k_static = k[:, :, :, :-qd]
        k_cur = k[:, :, :, -qd:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_mixed = (1 - gate) * k_cur + gate * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)


class CausalAdd(nn.Module):
    """Causal additive temporal mixing on the last quarter of channels.

    Adds gated neighbor signal without attenuating current token:
        k_mixed = k[t] + gate * k[t-1]

    Same interface as CausalLerp but purely additive — current value passes
    through unchanged, neighbor signal accumulates on top.

    Args:
        d_model:     Input dimension (for the gate projection from x).
        n_heads:     Number of attention heads.
        quarter_dim: Number of channels to mix (head_dim // 4).
        init_bias:   Initial gate bias (default -2.0, sigmoid ~ 0.12).
    """

    def __init__(self, d_model: int, n_heads: int, quarter_dim: int, init_bias: float = -2.0):
        super().__init__()
        self.quarter_dim = quarter_dim
        self.n_heads = n_heads
        self.gate_proj = nn.Linear(d_model, n_heads * quarter_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

    def forward(self, k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply causal additive mixing to K.

        Args:
            k: (B, T, H, head_dim) — key tensor.
            x: (B, T, D) — original input (for gate computation).

        Returns:
            K with last quarter_dim channels += gated t-1.
        """
        b, t, _, _ = k.shape
        if t < 2:
            return k
        qd = self.quarter_dim
        gate = torch.sigmoid(self.gate_proj(x)).view(b, t, self.n_heads, qd)

        k_static = k[:, :, :, :-qd]
        k_cur = k[:, :, :, -qd:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_mixed = k_cur + gate * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)


class KAcausalLerp(nn.Module):
    """Acausal temporal lerp on the last quarter of K channels.

    Like CausalLerp but also peeks at t+1:
        k_mixed = (1 - g_fwd - g_bwd) * k[t] + g_fwd * k[t+1] + g_bwd * k[t-1]

    Args:
        d_model:     Input dimension (for gate projections from x).
        n_heads:     Number of attention heads.
        quarter_dim: Number of channels to lerp (head_dim // 4).
        init_bias:   Initial gate bias (default -2.0, sigmoid ~ 0.12).
    """

    def __init__(self, d_model: int, n_heads: int, quarter_dim: int, init_bias: float = -2.0):
        super().__init__()
        self.quarter_dim = quarter_dim
        self.n_heads = n_heads
        self.gate_fwd_proj = nn.Linear(d_model, n_heads * quarter_dim, bias=True)
        self.gate_bwd_proj = nn.Linear(d_model, n_heads * quarter_dim, bias=True)
        nn.init.zeros_(self.gate_fwd_proj.weight)
        nn.init.constant_(self.gate_fwd_proj.bias, init_bias)
        nn.init.zeros_(self.gate_bwd_proj.weight)
        nn.init.constant_(self.gate_bwd_proj.bias, init_bias)

    def forward(self, k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = k.shape
        if t < 2:
            return k
        qd = self.quarter_dim
        g_fwd = torch.sigmoid(self.gate_fwd_proj(x)).view(b, t, self.n_heads, qd)
        g_bwd = torch.sigmoid(self.gate_bwd_proj(x)).view(b, t, self.n_heads, qd)

        k_static = k[:, :, :, :-qd]
        k_cur = k[:, :, :, -qd:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_next = F.pad(k_cur[:, 1:],  (0, 0, 0, 0, 0, 1))
        k_mixed = (1 - g_fwd - g_bwd) * k_cur + g_fwd * k_next + g_bwd * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)


class KAcausalAdd(nn.Module):
    """Acausal additive temporal mixing on the last quarter of K channels.

    Like CausalAdd but also peeks at t+1:
        k_mixed = k[t] + g_fwd * k[t+1] + g_bwd * k[t-1]

    Args:
        d_model:     Input dimension (for gate projections from x).
        n_heads:     Number of attention heads.
        quarter_dim: Number of channels to mix (head_dim // 4).
        init_bias:   Initial gate bias (default -2.0, sigmoid ~ 0.12).
    """

    def __init__(self, d_model: int, n_heads: int, quarter_dim: int, init_bias: float = -2.0):
        super().__init__()
        self.quarter_dim = quarter_dim
        self.n_heads = n_heads
        self.gate_fwd_proj = nn.Linear(d_model, n_heads * quarter_dim, bias=True)
        self.gate_bwd_proj = nn.Linear(d_model, n_heads * quarter_dim, bias=True)
        nn.init.zeros_(self.gate_fwd_proj.weight)
        nn.init.constant_(self.gate_fwd_proj.bias, init_bias)
        nn.init.zeros_(self.gate_bwd_proj.weight)
        nn.init.constant_(self.gate_bwd_proj.bias, init_bias)

    def forward(self, k: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, t, _, _ = k.shape
        if t < 2:
            return k
        qd = self.quarter_dim
        g_fwd = torch.sigmoid(self.gate_fwd_proj(x)).view(b, t, self.n_heads, qd)
        g_bwd = torch.sigmoid(self.gate_bwd_proj(x)).view(b, t, self.n_heads, qd)

        k_static = k[:, :, :, :-qd]
        k_cur = k[:, :, :, -qd:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_next = F.pad(k_cur[:, 1:],  (0, 0, 0, 0, 0, 1))
        k_mixed = k_cur + g_fwd * k_next + g_bwd * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)


class EmbeddingLerp(nn.Module):
    """Acausal temporal lerp on the last quarter of embedding dims.

    Applied once to token embeddings before any blocks. Centered mixing:
        x_mixed = (1 - g_fwd - g_bwd) * x[t] + g_fwd * x[t+1] + g_bwd * x[t-1]

    Operates on (B, T, D) directly — no head dimension.
    Only touches the last quarter of D; first 3/4 pass through unchanged.

    Args:
        d_model:   Embedding dimension.
        init_bias: Initial gate bias (default -2.0, sigmoid ~ 0.12).
    """

    def __init__(self, d_model: int, init_bias: float = -2.0):
        super().__init__()
        self.quarter_dim = d_model // 4
        self.gate_fwd_proj = nn.Linear(d_model, self.quarter_dim, bias=True)
        self.gate_bwd_proj = nn.Linear(d_model, self.quarter_dim, bias=True)
        nn.init.zeros_(self.gate_fwd_proj.weight)
        nn.init.constant_(self.gate_fwd_proj.bias, init_bias)
        nn.init.zeros_(self.gate_bwd_proj.weight)
        nn.init.constant_(self.gate_bwd_proj.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply acausal lerp to embeddings.

        Args:
            x: (B, T, D) — token embeddings.

        Returns:
            (B, T, D) with last quarter blended with t-1 and t+1.
        """
        b, t, _ = x.shape
        if t < 2:
            return x
        qd = self.quarter_dim
        g_fwd = torch.sigmoid(self.gate_fwd_proj(x))  # (B, T, qd)
        g_bwd = torch.sigmoid(self.gate_bwd_proj(x))

        x_static = x[:, :, :-qd]
        x_cur = x[:, :, -qd:]
        x_prev = F.pad(x_cur[:, :-1], (0, 0, 1, 0))
        x_next = F.pad(x_cur[:, 1:],  (0, 0, 0, 1))
        x_mixed = (1 - g_fwd - g_bwd) * x_cur + g_fwd * x_next + g_bwd * x_prev
        return torch.cat([x_static, x_mixed], dim=-1)


class KTemporalConv(nn.Module):
    """Causal depthwise temporal conv on the last quarter of K channels.

    Like QTemporalConv but causally padded (left-pad, no future leakage).
    Position t can only see t and t-1 (kernel_size=2) or t, t-1, t-2 (kernel_size=3).

    Operates on last quarter_dim channels (matching CausalLerp's convention).

    Args:
        n_heads: Number of attention heads.
        quarter_dim: Number of channels to mix (head_dim // 4).
        kernel_size: Temporal kernel size (2 or 3).
    """

    def __init__(self, n_heads: int, quarter_dim: int, kernel_size: int):
        super().__init__()
        assert kernel_size in (2, 3), f"kernel_size must be 2 or 3, got {kernel_size}"
        self.n_heads = n_heads
        self.quarter_dim = quarter_dim
        self.kernel_size = kernel_size
        channels = n_heads * quarter_dim
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, groups=channels, bias=True)

        # Identity-like init: mostly current token, small backward influence
        with torch.no_grad():
            self.conv.weight.zero_()
            assert self.conv.bias is not None
            self.conv.bias.zero_()
            if kernel_size == 2:
                # weight layout: [channels, 1, kernel_size], conv sees [t-1, t]
                self.conv.weight[:, 0, 0] = 0.12  # t-1
                self.conv.weight[:, 0, 1] = 0.88  # t
            else:  # kernel_size == 3
                # conv sees [t-2, t-1, t]
                self.conv.weight[:, 0, 0] = 0.12  # t-2
                self.conv.weight[:, 0, 1] = 0.12  # t-1
                self.conv.weight[:, 0, 2] = 0.76  # t

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """Apply causal temporal conv to K.

        Args:
            k: (B, T, H, head_dim)

        Returns:
            K with last quarter_dim channels convolved over time (causal).
        """
        b, t, _, _ = k.shape
        if t < 2:
            return k

        qd = self.quarter_dim
        k_static = k[:, :, :, :-qd]
        k_cur = k[:, :, :, -qd:]  # (B, T, H, qd)

        # (B, T, H, qd) -> (B, H*qd, T)
        x = k_cur.permute(0, 2, 3, 1).reshape(b, self.n_heads * qd, t)
        # Left-pad for causal: position t sees [t-(ks-1), ..., t]
        x = F.pad(x, (self.kernel_size - 1, 0))
        y = self.conv(x)[..., :t]
        # (B, H*qd, T) -> (B, T, H, qd)
        y = y.view(b, self.n_heads, qd, t).permute(0, 3, 1, 2)

        return torch.cat([k_static, y], dim=-1)
