"""Temporal lerp operations for K and Q preprocessing.

These operations blend tokens with their temporal neighbors before RoPE,
providing SSM-equivalent state mixing capabilities:

- CausalLerp (K): blends last 1/4 of head_dim with t-1 neighbor.
  Equivalent to Mamba-3's trapezoidal discretization.

- AcausalLerp (Q): blends last 1/2 of head_dim with t-1 AND t+1 neighbors
  (separate learned gates). The t+1 (forward) gate enables backward
  information flow — a key advantage over purely causal SSMs.

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


class AcausalLerp(nn.Module):
    """Acausal temporal lerp on the 3rd quarter of channels.

    Blends q[t] with both q[t-1] (backward) and q[t+1] (forward) via
    separate content-dependent gates:
        q_mixed = (1 - g_fwd - g_bwd) * q[t] + g_fwd * q[t+1] + g_bwd * q[t-1]

    The forward gate (t+1) enables backward information flow through the
    sequence, which is how FusedGate breaks past the ~50% causal ceiling
    on induction tasks.

    Only applies to the 3rd quarter of head_dim (channels [-half_dim:-quarter_dim]),
    leaving the last quarter for K-mix and the first half static.
    Last position gets zeros for forward neighbor (causal fallback).

    Args:
        d_model:      Input dimension (for gate projections from x).
        n_heads:      Number of attention heads.
        quarter_dim:  Number of channels to lerp (head_dim // 4).
        init_bias:    Initial gate bias (default -2.0, sigmoid ~ 0.12).
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

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply acausal lerp to Q.

        Args:
            q: (B, T, H, head_dim) — query tensor.
            x: (B, T, D) — original input (for gate computation).

        Returns:
            Q with 3rd quarter channels blended with t-1 and t+1.
        """
        b, t, _, _ = q.shape
        if t < 2:
            # During incremental decode, future token q[t+1] is unavailable.
            # Returning identity here preserves stable single-step behavior and
            # avoids artificial shrinking from zero-padding neighbors.
            return q
        qd = self.quarter_dim
        g_fwd = torch.sigmoid(self.gate_fwd_proj(x)).view(b, t, self.n_heads, qd)
        g_bwd = torch.sigmoid(self.gate_bwd_proj(x)).view(b, t, self.n_heads, qd)

        # 3rd quarter: channels [-2*qd:-qd]
        q_pre = q[:, :, :, :-2*qd]        # first half (static)
        q_cur = q[:, :, :, -2*qd:-qd]     # 3rd quarter (Q-mix target)
        q_post = q[:, :, :, -qd:]          # last quarter (K-mix territory, static)
        q_prev = F.pad(q_cur[:, :-1], (0, 0, 0, 0, 1, 0))   # causal: t-1
        q_next = F.pad(q_cur[:, 1:],  (0, 0, 0, 0, 0, 1))   # acausal: t+1, last pos gets zeros
        q_mixed = (1 - g_fwd - g_bwd) * q_cur + g_fwd * q_next + g_bwd * q_prev
        return torch.cat([q_pre, q_mixed, q_post], dim=-1)


class QTemporalConv(nn.Module):
    """Depthwise temporal conv on the 3rd quarter of Q channels.

    This is a simple conv baseline for Q-peeking comparisons.
    It applies a depthwise 1D convolution over time on each head-channel
    independently, using right-padding so the output at position t can depend
    on future positions up to t + (kernel_size - 1).

    Operates on the 3rd quarter of head_dim (channels [-2*qd:-qd]),
    leaving the last quarter for K-mix and the first half static.

    - kernel_size=2 -> lookahead 1
    - kernel_size=3 -> lookahead 2

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

        # Identity-like init: y_t ~= x_t at start
        with torch.no_grad():
            self.conv.weight.zero_()
            assert self.conv.bias is not None
            self.conv.bias.zero_()
            if kernel_size == 2:
                # Match q-lerp's small initial forward influence (~0.12)
                self.conv.weight[:, 0, 0] = 0.88
                self.conv.weight[:, 0, 1] = 0.12
            else:  # kernel_size == 3
                self.conv.weight[:, 0, 0] = 0.76
                self.conv.weight[:, 0, 1] = 0.12
                self.conv.weight[:, 0, 2] = 0.12

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Apply temporal conv to Q.

        Args:
            q: (B, T, H, head_dim)

        Returns:
            Q with 3rd quarter channels convolved over time.
        """
        b, t, _, _ = q.shape
        if t < 2:
            return q

        qd = self.quarter_dim
        q_pre = q[:, :, :, :-2*qd]        # first half (static)
        q_cur = q[:, :, :, -2*qd:-qd]     # 3rd quarter (Q-mix target)
        q_post = q[:, :, :, -qd:]          # last quarter (K-mix territory, static)

        # (B, T, H, qd) -> (B, H*qd, T)
        x = q_cur.permute(0, 2, 3, 1).reshape(b, self.n_heads * qd, t)
        x = F.pad(x, (0, self.kernel_size - 1))
        y = self.conv(x)[..., :t]
        # (B, H*qd, T) -> (B, T, H, qd)
        y = y.view(b, self.n_heads, qd, t).permute(0, 3, 1, 2)

        return torch.cat([q_pre, y, q_post], dim=-1)


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
