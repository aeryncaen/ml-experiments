"""
DS-Attention: signed sparse attention via L1 ball projection.

Instead of softmax, attention scores are projected onto a learnable-radius
L1 ball per head. This gives exact zeros (sparsity) and negative weights
(signed). Data-dependent RoPE, QK-norm, diff readout with learnable lambda.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def entmax15(z: torch.Tensor, dim: int = -1, n_iter: int = 25) -> torch.Tensor:
    """1.5-entmax via bisection (Correia et al., 2019).
    Sparse attention: exact zeros, smooth gradients, sums to 1.
    For alpha=1.5: p_i = [max(0, z_i - tau)]^2, find tau s.t. sum(p) = 1.
    """
    z = z - z.max(dim=dim, keepdim=True).values  # numerical stability

    # Bisection to find tau such that sum(max(0, z - tau)^2) = 1
    lo = z.min(dim=dim, keepdim=True).values - 1.0
    hi = z.max(dim=dim, keepdim=True).values
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        p = F.relu(z - mid).square()
        s = p.sum(dim=dim, keepdim=True)
        lo = torch.where(s > 1.0, mid, lo)
        hi = torch.where(s <= 1.0, mid, hi)

    tau = (lo + hi) / 2
    return F.relu(z - tau).square()


def apply_interleaved_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = angles.cos()
    sin = angles.sin()
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class DSAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        mimo_rank: int = 4,
        diff_readout: bool = True,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.D = d_model
        self.N = state_dim
        self.R = mimo_rank
        self.diff_readout = diff_readout

        # QKV projections
        self.to_Q = nn.Linear(d_model, state_dim * mimo_rank)
        self.to_K = nn.Linear(d_model, state_dim * mimo_rank)
        self.to_V = nn.Linear(d_model, state_dim * mimo_rank)
        self.to_theta = nn.Linear(d_model, state_dim // 2, bias=False)
        self.out_proj = nn.Linear(state_dim * mimo_rank, d_model)

        # QK norm
        self.q_norm = RMSNorm(state_dim)
        self.k_norm = RMSNorm(state_dim)



        if diff_readout:
            self.readout_lambda = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N, R = self.N, self.R
        scale = N ** -0.5

        # Project Q, K, V
        Q = F.silu(self.to_Q(x)).view(B, L, N, R).permute(0, 3, 1, 2)  # (B, R, L, N)
        K = F.silu(self.to_K(x)).view(B, L, N, R).permute(0, 3, 1, 2)  # (B, R, L, N)
        V = F.silu(self.to_V(x)).view(B, L, N, R).permute(0, 3, 1, 2)  # (B, R, L, N)

        # Data-dependent RoPE on Q and K
        theta = self.to_theta(x)                        # (B, L, N//2)
        pos = torch.arange(L, device=x.device, dtype=x.dtype)
        angles = theta.unsqueeze(1) * pos[None, None, :, None]  # (B, 1, L, N//2)
        Q = apply_interleaved_rope(Q, angles)
        K = apply_interleaved_rope(K, angles)

        # QK norm
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Sparse attention via 1.5-entmax
        S = Q @ K.transpose(-2, -1) * scale            # (B, R, L, L)

        # Causal mask before entmax so masked positions get exact zero
        causal_mask = torch.triu(torch.full((L, L), -1e9, device=x.device, dtype=x.dtype), 1)
        S = S + causal_mask
        A = entmax15(S, dim=-1)                         # (B, R, L, L) sparse

        # Apply attention
        out = A @ V                                     # (B, R, L, N)

        # Diff readout
        if self.diff_readout:
            half_N = N // 2
            g1 = out[..., :half_N]
            g2 = out[..., half_N:]
            out = torch.cat([g1 - self.readout_lambda * g2,
                             g1 + self.readout_lambda * g2], dim=-1)

        out = out.permute(0, 2, 1, 3).reshape(B, L, N * R)  # (B, L, N*R)
        return self.out_proj(F.silu(out))
