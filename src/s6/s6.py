"""S6: S4D base + S5 MIMO + Mamba selectivity + trapezoidal discretization + data-dependent RoPE."""

import math
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pathlib import Path

from .scan import parallel_scan

# Import LearnableFeatureMap from zoology (sibling repo)
_zoology_path = str(Path(__file__).resolve().parents[2] / 'zoology')
import sys as _sys
if _zoology_path not in _sys.path:
    _sys.path.insert(0, _zoology_path)
from zoology.mixers.luna import LearnableFeatureMap


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_DPLR_HiPPO(N):
    hippo = make_HiPPO(N)
    P = np.sqrt(np.arange(N) + 0.5)
    B = np.sqrt(2 * np.arange(N) + 1.0)
    S = hippo + P[:, np.newaxis] * P[np.newaxis, :]
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    Lambda_imag, V = np.linalg.eigh(S * -1j)
    return Lambda_real + 1j * Lambda_imag


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def apply_rotary_emb(x, angles):
    # x: (..., P) even, angles: (..., P//2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = torch.cos(angles), torch.sin(angles)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class S6Kernel(nn.Module):
    """S6 SSM: input-dependent discretization + parallel scan, MIMO via shared diagonal state."""

    def __init__(self, d_model, N=64, M=4, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        P = N
        assert P % 2 == 0, "d_state must be even for RoPE"
        assert P % M == 0, f"d_state ({P}) must be divisible by M ({M})"
        self.H = H
        self.P = P

        # A: HiPPO-LegS eigenvalues (P,) complex — shared across channels
        Lambda = make_DPLR_HiPPO(P)
        log_A_real = torch.log(-torch.tensor(Lambda.real, dtype=torch.float))  # (P,)
        A_imag = torch.tensor(Lambda.imag, dtype=torch.float)  # (P,)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        # B: input projection (H → P) with SiLU activation
        self.phi_B = nn.Sequential(
            nn.Linear(H, P),
            nn.SiLU(),
        )

        # Fused input projection for dt, lam, theta (B is now separate via feature bank)
        # Layout: [dt(P), lam(P), theta(P//2)]
        self.x_proj = nn.Linear(H, P + P + P // 2)
        self._split_sizes = [P, P, P // 2]

        # B norm + bias (Mamba-3 pattern)
        self.b_norm = RMSNorm(P)
        self.b_bias = nn.Parameter(torch.ones(P))

        # dt bias (init in log-uniform [dt_min, dt_max])
        log_dt_bias = torch.rand(P) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt_bias = nn.Parameter(log_dt_bias)

        # lambda_proj bias init: sigmoid(2) ≈ 0.88 biased toward current input
        with torch.no_grad():
            bias = self.x_proj.bias
            # lam region starts at P, ends at 2P
            bias[P:2*P] = 2.0
            # theta region: zero weights and bias for init (starts from zero rotation)
            self.x_proj.weight[2*P:] = 0.0
            bias[2*P:] = 0.0

    def forward(self, u):
        """
        Args:
            u: (B, L, H) input sequence
        Returns:
            h: (B, L, P) all hidden states after scan
            cum_theta: (B, L, P//2) cumulative rotation angles for C readout
        """
        B, L, H = u.shape
        P = self.P

        # Materialize complex A
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (P,) complex

        # B: input projection with SiLU
        Bu_raw = self.phi_B(u)  # (B, L, P)

        # dt, lam, theta from fused linear projection
        x_proj = self.x_proj(u)  # (B, L, P+P+P//2)
        dt_raw, lam_raw, theta = x_proj.split(self._split_sizes, dim=-1)

        dt = F.softplus(F.silu(dt_raw) + self.log_dt_bias)  # (B, L, P)
        lam = torch.sigmoid(lam_raw)  # (B, L, P)

        Bu = self.b_norm(Bu_raw) + self.b_bias  # (B, L, P)

        # Data-dependent RoPE on B
        dt_half = dt.view(B, L, P // 2, 2).mean(-1)  # (B, L, P//2)
        cum_theta = torch.cumsum(dt_half * theta, dim=1)  # (B, L, P//2)
        Bu = apply_rotary_emb(Bu, cum_theta)
        # Bu_prev: shift AFTER rotation so position t-1 keeps its own rotation angle
        Bu_prev = F.pad(Bu[:, :-1], (0, 0, 1, 0))

        # Trapezoidal discretization (complex)
        alpha = torch.exp(dt.to(torch.cfloat) * A)  # (B, L, P) complex decay
        Bu_c = Bu.to(torch.cfloat)
        Bu_prev_c = Bu_prev.to(torch.cfloat)
        dt_c = dt.to(torch.cfloat)
        inject = lam * dt_c * Bu_c + (1 - lam) * dt_c * alpha * Bu_prev_c  # (B, L, P) complex

        # Parallel scan: h[t] = alpha[t] * h[t-1] + inject[t]
        h = parallel_scan(alpha, inject)  # (B, L, P) complex

        return h, cum_theta

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S6(nn.Module):
    def __init__(self, d_model, d_state=64, M=4, layer_idx=None, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h

        # D skip connection
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel (parallel scan, not FFT conv)
        self.kernel = S6Kernel(self.h, N=self.n, M=M, **kernel_args)

        # C: static MIMO readout (H, P) complex — maps state back to output channels
        C = torch.randn(self.h, self.n, dtype=torch.cfloat) / math.sqrt(self.n)
        self.C = nn.Parameter(torch.view_as_real(C))  # (H, P, 2) stored as real
        self.c_proj = nn.Linear(self.h, self.n)  # input-dependent gating of C
        self.c_norm = RMSNorm(self.n)
        self.c_bias = nn.Parameter(torch.ones(self.n))

    def forward(self, u, **kwargs):
        """ Input and output shape (B, L, H) """
        B, L, H = u.shape

        # Run SSM scan
        h, cum_theta = self.kernel(u)  # h: (B, L, P), cum_theta: (B, L, P//2)

        # Input-dependent C gating + static C readout
        c_gate = self.c_norm(F.silu(self.c_proj(u))) + self.c_bias  # (B, L, P)
        c_gate = apply_rotary_emb(c_gate, cum_theta)  # rotate to match state
        h_gated = h * c_gate  # (B, L, P) — input-dependent selection of state dims

        # MIMO readout: (H, P) complex @ (B, L, P) complex -> (B, L, H), take real
        C = torch.view_as_complex(self.C)  # (H, P)
        y = torch.einsum('hp,blp->blh', C, h_gated.to(C.dtype)).real

        # Skip connection + activation
        y = y + u * self.D
        y = F.silu(y)
        return y
