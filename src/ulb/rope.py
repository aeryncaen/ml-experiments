"""Hybrid RoPE: fixed positional rotations + data-dependent rotations via cumsum.

The head dimension is split into rotation pairs:
  - First half: standard fixed-frequency RoPE
  - Second half: data-dependent angles from cumsum(proj(x))

The data-dependent rotation is the key SSM-equivalent mechanism:
cumulative rotation angles act as complex-valued state transitions,
giving the model state-tracking capability without sequential scans.

Layout within head_dim:
  [fixed_x1 (fp) | fixed_x2 (fp) | dd_x1 (dp) | dd_x2 (dp)]
where fp = head_dim // 4 fixed pairs, dp = head_dim // 4 data-dependent pairs.

This aligns DD rotations with the temporal mixing channels:
  - 3rd quarter (Q-mix) gets DD rotation
  - 4th quarter (K-mix) gets DD rotation
  - 1st and 2nd quarters (static) get fixed RoPE
"""

import torch
import torch.nn as nn


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to paired dimensions.

    Args:
        x:   (..., 2*n_pairs) — the dimensions to rotate
        cos: (..., n_pairs) — cosine of rotation angles
        sin: (..., n_pairs) — sine of rotation angles

    Returns:
        Rotated tensor, same shape as x.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


class HybridRoPE(nn.Module):
    """Hybrid RoPE: fixed + data-dependent rotations.

    Fixed rotations use standard inverse-frequency schedule.
    Data-dependent rotations use cumsum of a learned projection from the input,
    providing SSM-equivalent state tracking through cumulative phase shifts.

    Args:
        d_model:  Input dimension (for the data-dependent projection).
        n_heads:  Number of attention heads.
        head_dim: Dimension per head. Must be divisible by 4.
        rope_base: Base for fixed RoPE inverse frequencies (default 10000).
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope_base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.fixed_pairs = head_dim // 4   # first half of rotation pairs
        self.dd_pairs = head_dim // 4      # second half: data-dependent

        # Fixed RoPE inverse frequencies
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.fixed_pairs * 2, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Data-dependent projection: x -> per-head rotation deltas, then cumsum
        self.dd_proj = nn.Linear(d_model, n_heads * self.dd_pairs, bias=True)
        nn.init.zeros_(self.dd_proj.weight)
        nn.init.zeros_(self.dd_proj.bias)

    def compute_dd_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cumulative data-dependent rotation angles.

        Args:
            x: (B, T, D) input tensor.

        Returns:
            (B, T, H, dd_pairs) cumulative rotation angles.
        """
        b, t, _ = x.shape
        deltas = self.dd_proj(x)  # (B, T, H * dd_pairs)
        deltas = deltas.view(b, t, self.n_heads, self.dd_pairs)
        return deltas.cumsum(dim=1)

    def forward(self, qk: torch.Tensor, dd_angles: torch.Tensor) -> torch.Tensor:
        """Apply hybrid RoPE to Q or K tensor.

        Args:
            qk:        (B, T, H, head_dim)
            dd_angles: (B, T, H, dd_pairs) — from compute_dd_angles()

        Returns:
            Rotated tensor, same shape as qk.
        """
        fp = self.fixed_pairs
        dp = self.dd_pairs
        T = qk.shape[1]

        # Split into fixed and data-dependent regions
        # Layout: [fixed_x1 (fp) | fixed_x2 (fp) | dd_x1 (dp) | dd_x2 (dp)]
        qk_fixed = torch.cat([qk[..., :fp], qk[..., fp:2*fp]], dim=-1)        # (B,T,H,2*fp)
        qk_dd = torch.cat([qk[..., 2*fp:2*fp+dp], qk[..., 2*fp+dp:]], dim=-1)  # (B,T,H,2*dp)

        # Fixed RoPE
        t = torch.arange(T, device=qk.device, dtype=self.inv_freq.dtype)
        freqs_fixed = torch.outer(t, self.inv_freq)  # (T, fp)
        cos_f = freqs_fixed.cos()[None, :, None, :]  # (1, T, 1, fp)
        sin_f = freqs_fixed.sin()[None, :, None, :]
        qk_fixed = apply_rotary(qk_fixed, cos_f, sin_f)

        # Data-dependent RoPE
        cos_d = dd_angles.cos()  # (B, T, H, dp)
        sin_d = dd_angles.sin()
        qk_dd = apply_rotary(qk_dd, cos_d, sin_d)

        # Reassemble: [fixed_x1 | fixed_x2 | dd_x1 | dd_x2]
        return torch.cat([qk_fixed[..., :fp], qk_fixed[..., fp:],
                          qk_dd[..., :dp], qk_dd[..., dp:]], dim=-1)


class PairedRoPE(nn.Module):
    """Paired-head RoPE: interleaves adjacent heads at 2x sequence length.

    Even-indexed heads use even position indices (2t), odd-indexed heads use
    odd position indices (2t+1). This doubles the effective sequence length
    that attention operates over.

    Uses the same hybrid (fixed + data-dependent) rotation scheme as HybridRoPE,
    but with separate even/odd frequency schedules.

    Args:
        n_heads:   Total number of heads (must be even; paired into n_heads//2 groups).
        head_dim:  Dimension per head. Must be divisible by 4.
        rope_base: Base for fixed RoPE inverse frequencies.
    """

    def __init__(self, n_heads: int, head_dim: int, rope_base: float = 10000.0):
        super().__init__()
        assert n_heads % 2 == 0, f"n_heads must be even for pairing, got {n_heads}"
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.fixed_pairs = head_dim // 4
        self.dd_pairs = head_dim // 4

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.fixed_pairs * 2, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, q_or_k: torch.Tensor, dd_angles: torch.Tensor) -> torch.Tensor:
        """Apply paired RoPE with even/odd position encodings.

        Args:
            q_or_k:    (B, T, n_heads//2, head_dim*2) — paired heads concatenated.
            dd_angles: (B, T, n_heads, dd_pairs) — will be reshaped for paired heads.

        Returns:
            Rotated tensor, same shape as q_or_k.
        """
        b, T, n2, hd2 = q_or_k.shape
        fp = self.fixed_pairs
        dp = self.dd_pairs
        hd = self.head_dim

        # Even/odd position indices
        t = torch.arange(T, device=q_or_k.device, dtype=self.inv_freq.dtype)
        t_even, t_odd = 2 * t, 2 * t + 1
        freqs_even = torch.outer(t_even, self.inv_freq)  # (T, fp)
        freqs_odd = torch.outer(t_odd, self.inv_freq)

        # Reshape dd_angles: (B,T,H,dp) -> (B,T,n2,2*dp) for paired heads
        dd = dd_angles.view(b, T, n2, 2 * dp)
        dd0 = dd[..., :dp]
        dd1 = dd[..., dp:]

        # Split into two head_dim chunks
        h0 = q_or_k[..., :hd]
        h1 = q_or_k[..., hd:]

        # h0: even positions
        # Layout: [fixed_x1 (fp) | fixed_x2 (fp) | dd_x1 (dp) | dd_x2 (dp)]
        h0_fixed = torch.cat([h0[..., :fp], h0[..., fp:2*fp]], dim=-1)
        h0_dd = torch.cat([h0[..., 2*fp:2*fp+dp], h0[..., 2*fp+dp:]], dim=-1)
        cos_f0 = freqs_even.cos()[None, :, None, :]
        sin_f0 = freqs_even.sin()[None, :, None, :]
        h0_fixed = apply_rotary(h0_fixed, cos_f0, sin_f0)
        h0_dd = apply_rotary(h0_dd, dd0.cos(), dd0.sin())
        h0 = torch.cat([h0_fixed[..., :fp], h0_fixed[..., fp:],
                         h0_dd[..., :dp], h0_dd[..., dp:]], dim=-1)

        # h1: odd positions
        h1_fixed = torch.cat([h1[..., :fp], h1[..., fp:2*fp]], dim=-1)
        h1_dd = torch.cat([h1[..., 2*fp:2*fp+dp], h1[..., 2*fp+dp:]], dim=-1)
        cos_f1 = freqs_odd.cos()[None, :, None, :]
        sin_f1 = freqs_odd.sin()[None, :, None, :]
        h1_fixed = apply_rotary(h1_fixed, cos_f1, sin_f1)
        h1_dd = apply_rotary(h1_dd, dd1.cos(), dd1.sin())
        h1 = torch.cat([h1_fixed[..., :fp], h1_fixed[..., fp:],
                         h1_dd[..., :dp], h1_dd[..., dp:]], dim=-1)

        return torch.cat([h0, h1], dim=-1)
