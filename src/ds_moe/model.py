"""
DS-MoE: Diffusive-State Relay Mixture of Experts.
Single-file core: norms, RoPE, SE, ops (DiffAttn, DS1, DiffMLP),
Block, embeds, heads, model shells, weight init.
"""

from __future__ import annotations

import math
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


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def apply_interleaved_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE: even/odd index rotation.
    x: (..., N)  where N is even
    angles: (..., N//2)  broadcastable to x's shape minus last dim halved
    """
    x1 = x[..., ::2]   # even indices -> (... , N//2)
    x2 = x[..., 1::2]  # odd indices  -> (... , N//2)
    cos = angles.cos()
    sin = angles.sin()
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, relu2: bool = False):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.act = relu_squared if relu2 else F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., spatial..., C) — pool over all dims except batch and last
        spatial_dims = tuple(range(1, x.ndim - 1))
        scale = x.mean(dim=spatial_dims)                    # (B, C)
        scale = torch.sigmoid(self.fc2(self.act(self.fc1(scale))))
        for _ in spatial_dims:
            scale = scale.unsqueeze(1)
        return x * scale


class DS1(nn.Module):
    """Diffusive State 1: iterative-convolution SSM.

    Bank pattern: large projection weights (to_B, to_C, to_X, to_decay,
    to_theta, to_lambda, out_proj) are NOT stored here. They live in the
    model shell's ssm_bank and are passed as `ssm_w` to forward().

    This module stores only small per-op params: B/C biases, conv weights,
    optional SE, optional BC-norm, optional diff lambdas.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int = 64,
        mimo_rank: int = 4,
        n_iters: int = 2,
        diffuse_se: bool = False,
        diff_inject: bool = False,
        diff_readout: bool = False,
        bc_norm: bool = False,
        relu2: bool = False,
    ):
        super().__init__()
        self.D = dim
        self.N = state_dim
        self.R = mimo_rank
        self.n_iters = n_iters
        self.diff_inject = diff_inject
        self.diff_readout = diff_readout
        self.act = relu_squared if relu2 else F.silu

        self.B_bias = nn.Parameter(torch.ones(state_dim * mimo_rank))
        self.C_bias = nn.Parameter(torch.ones(state_dim * mimo_rank))

        if diff_inject:
            self.inject_lambda = nn.Parameter(torch.tensor(0.5))
        if diff_readout:
            self.readout_lambda = nn.Parameter(torch.tensor(0.5))

        self.bc_norm = bc_norm
        if bc_norm:
            self.b_norm = RMSNorm(state_dim)
            self.c_norm = RMSNorm(state_dim)

        self.diffuse = nn.Conv1d(state_dim, state_dim, kernel_size=3, padding=1, groups=state_dim)

        self.has_se = diffuse_se
        if diffuse_se:
            self.diffuse_se = SqueezeExcite(state_dim, relu2=relu2)

    @staticmethod
    def bank_size(dim: int, state_dim: int = 64, mimo_rank: int = 4) -> int:
        """Total number of elements per layer in the SSM bank.
        Bank layout (transposed storage, each is (in_features, out_features)):
          to_B:      D x (N*R)
          to_C:      D x (N*R)
          to_X:      D x R
          to_decay:  D x N
          to_theta:  D x (N//2)
          to_lambda: D x 1
          out_proj:  (N*R) x D
        All stored as a single flat vector per layer.
        """
        N, R, D = state_dim, mimo_rank, dim
        return (D * N * R      # to_B
                + D * N * R    # to_C
                + D * R        # to_X
                + D * N        # to_decay
                + D * (N // 2) # to_theta
                + D * 1        # to_lambda
                + N * R * D)   # out_proj

    def _unpack_weights(self, ssm_w: torch.Tensor) -> tuple:
        """Unpack flat weight vector into individual projection matrices.
        ssm_w: (bank_size,) — flat weight vector for this layer.
        Returns transposed-storage weight matrices for each projection.
        """
        D, N, R = self.D, self.N, self.R
        idx = 0

        def take(rows: int, cols: int) -> torch.Tensor:
            nonlocal idx
            size = rows * cols
            w = ssm_w[idx:idx + size].view(rows, cols)
            idx += size
            return w

        w_B = take(D, N * R)          # (D, N*R)
        w_C = take(D, N * R)          # (D, N*R)
        w_X = take(D, R)              # (D, R)
        w_decay = take(D, N)          # (D, N)
        w_theta = take(D, N // 2)     # (D, N//2)
        w_lambda = take(D, 1)         # (D, 1)
        w_out = take(N * R, D)        # (N*R, D)

        assert idx == ssm_w.numel(), f"Bank unpack mismatch: used {idx}, have {ssm_w.numel()}"
        return w_B, w_C, w_X, w_decay, w_theta, w_lambda, w_out

    def forward(self, x: torch.Tensor, ssm_w: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        ssm_w: flat weight vector from ssm_bank for this layer
        Returns: (B, L, D)
        """
        B_batch, L, D = x.shape
        N, R = self.N, self.R
        act = self.act

        w_B, w_C, w_X, w_decay, w_theta, w_lambda, w_out = self._unpack_weights(ssm_w)

        B_proj = act(x @ w_B + self.B_bias)         # (B, L, N*R)
        C_proj = act(x @ w_C + self.C_bias)         # (B, L, N*R)
        X_r = act(x @ w_X)                          # (B, L, R)
        decay = torch.sigmoid(x @ w_decay)           # (B, L, N)
        theta = x @ w_theta                          # (B, L, N//2)
        lam = torch.sigmoid(x @ w_lambda)            # (B, L, 1)

        B_base = B_proj.view(B_batch, L, N, R).permute(0, 3, 1, 2).contiguous()
        C_base = C_proj.view(B_batch, L, N, R).permute(0, 3, 1, 2).contiguous()

        if self.bc_norm:
            B_base = self.b_norm(B_base)
            C_base = self.c_norm(C_base)

        # X_r: (B, L, R) -> (B, R, L, 1)
        X_r = X_r.permute(0, 2, 1).unsqueeze(-1)

        # decay: (B, L, N) -> (B, 1, L, N)
        decay = decay.unsqueeze(1)
        # lam: (B, L, 1) -> (B, 1, L, 1)
        lam = lam.unsqueeze(1)

        pos = torch.arange(L, device=x.device, dtype=x.dtype)  # (L,)

        H = torch.zeros(B_batch, R, L, N, device=x.device, dtype=x.dtype)
        C_rot = C_base
        prev_inject = None

        for i in range(self.n_iters):
            # angles: (B, 1, L, N//2) = content_theta * position * iteration
            angles = theta.unsqueeze(1) * pos[None, None, :, None] * (i + 1)
            B_rot = apply_interleaved_rope(B_base, angles)
            C_rot = apply_interleaved_rope(C_base, angles)

            if self.diff_inject:
                half_N = N // 2
                B1, B2 = B_rot[..., :half_N], B_rot[..., half_N:]
                inj1 = B1 * X_r
                inj2 = B2 * X_r
                inject = torch.cat([inj1 - self.inject_lambda * inj2,
                                    inj1 + self.inject_lambda * inj2], dim=-1)
            else:
                inject = B_rot * X_r

            H = H.permute(0, 1, 3, 2).reshape(B_batch * R, N, L)
            H = self.diffuse(H)
            H = H.reshape(B_batch, R, N, L).permute(0, 1, 3, 2)

            if self.has_se:
                H_flat = H.reshape(B_batch * R, L, N)
                H_flat = self.diffuse_se(H_flat)
                H = H_flat.reshape(B_batch, R, L, N)

            alpha = decay    # (B, 1, L, N)
            gamma = lam     # (B, 1, L, 1)
            if prev_inject is not None:
                beta = (1.0 - gamma) * alpha
                H = alpha * H + beta * prev_inject + gamma * inject
            else:
                H = alpha * H + inject
            prev_inject = inject

        if self.diff_readout:
            half_N = N // 2
            C1, C2 = C_rot[..., :half_N], C_rot[..., half_N:]
            H1, H2 = H[..., :half_N], H[..., half_N:]
            g1 = C1 * H1
            g2 = C2 * H2
            gated = torch.cat([g1 - self.readout_lambda * g2,
                               g1 + self.readout_lambda * g2], dim=-1)
        else:
            gated = C_rot * H

        out = gated.permute(0, 2, 1, 3).reshape(B_batch, L, N * R)
        y = act(out @ w_out)
        return y
