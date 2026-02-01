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


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x):
        # x: (B, L, C)
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(x.mean(dim=1)))))
        return x * scale.unsqueeze(1)


class MultiScaleDepthwiseConv(nn.Module):
    """Split channels into 4 groups: 3 with different kernel sizes, 1 passthrough. Then SE."""
    def __init__(self, channels, kernel_sizes=(3, 6, 12)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        n_groups = len(kernel_sizes) + 1  # +1 for passthrough
        assert channels % n_groups == 0, f"channels {channels} not divisible by {n_groups}"
        self.group_size = channels // n_groups
        self.convs = nn.ModuleList([
            nn.Conv1d(self.group_size, self.group_size, k, padding=k // 2, groups=self.group_size)
            for k in kernel_sizes
        ])
        self.se = SqueezeExcite1D(channels)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        gs = self.group_size
        n_conv = len(self.kernel_sizes)
        conv_chunks = [x[..., i * gs:(i + 1) * gs] for i in range(n_conv)]
        passthrough = x[..., n_conv * gs:]
        out = []
        for chunk, conv in zip(conv_chunks, self.convs):
            y = F.silu(conv(chunk.transpose(1, 2)))[..., :L].transpose(1, 2)
            out.append(y)
        out.append(passthrough)
        out = torch.cat(out, dim=-1)  # (B, L, C)
        return self.se(out)


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

        # B: LUNA feature bank (H → P) — learned nonlinear input selectivity
        L_fb = P // M
        self.phi_B = LearnableFeatureMap(
            d=H, M=M, L=L_fb, hidden=64,
            nonneg=False, act="silu", ch_rms=True,
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

    def forward(self, u, Bu_raw=None):
        """
        Args:
            u: (B, L, H) input sequence
            Bu_raw: (B, L, P) optional pre-computed B projection (skips phi_B if provided)
        Returns:
            h: (B, L, P) all hidden states after scan
            cum_theta: (B, L, P//2) cumulative rotation angles for C readout
        """
        B, L, H = u.shape
        P = self.P

        # Materialize complex A
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (P,) complex

        # B: use provided Bu_raw or compute from phi_B
        if Bu_raw is None:
            Bu_raw = self.phi_B(u.unsqueeze(1)).squeeze(1)  # (B, L, P)

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


class S6Attention(nn.Module):
    """Causal attention with 80/20 weight sharing.

    Q: 80% of rows shared with c_proj, 20% dedicated.
    K: 80% of rows shared with phi_B.W, 20% dedicated.
    V: standalone.
    Separate forward ops — just shared Parameter storage.
    """
    def __init__(self, d_model, d_state, M=4, num_heads=1,
                 phi_B=None, c_proj=None):
        super().__init__()
        H = d_model
        P = d_state
        self.H = H
        self.P = P
        self.M = M
        self.num_heads = num_heads
        self.head_dim = P // num_heads
        assert P % num_heads == 0

        # Q: 80% rows from c_proj, 20% dedicated
        self.shared_q_rows = int(0.8 * P)
        self.ded_q_rows = P - self.shared_q_rows
        assert c_proj is not None
        self._c_proj = c_proj
        self.q_ded_weight = nn.Parameter(torch.randn(self.ded_q_rows, H) / math.sqrt(H))
        self.q_ded_bias = nn.Parameter(torch.zeros(self.ded_q_rows))

        # K: 80% W rows from phi_B, 20% dedicated
        self.shared_k_dirs = int(0.8 * M)
        self.ded_k_dirs = M - self.shared_k_dirs
        assert phi_B is not None
        self._phi_B = phi_B
        self.k_ded_W = nn.Parameter(torch.randn(self.ded_k_dirs, H) / math.sqrt(H))
        self.k_ded_b = nn.Parameter(torch.zeros(self.ded_k_dirs))

        self.v_proj = nn.Linear(H, P, bias=False, dtype=torch.bfloat16)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.q_norm.weight.data = self.q_norm.weight.data.bfloat16()
        self.k_norm.weight.data = self.k_norm.weight.data.bfloat16()

        self.out_proj = nn.Linear(P, H, bias=False, dtype=torch.bfloat16)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, u, y_ssm):
        """
        u: (B, L, H) — input for Q/K/V projections
        y_ssm: (B, L, H) — SSM output (residual target)
        """
        B_batch, L, H = u.shape
        P = self.P
        M = self.M
        nh = self.num_heads
        hd = self.head_dim

        # Q: assemble shared c_proj rows + dedicated rows
        q_w = torch.cat([self._c_proj.weight[:self.shared_q_rows],
                         self.q_ded_weight], dim=0)
        q_b = torch.cat([self._c_proj.bias[:self.shared_q_rows],
                         self.q_ded_bias], dim=0)
        Q = F.linear(u, q_w, q_b).bfloat16()  # (B, L, P)

        # K: assemble shared phi_B rows + dedicated rows, run feature map
        k_W = torch.cat([self._phi_B.W[:self.shared_k_dirs],
                         self.k_ded_W], dim=0)  # (M, H)
        k_b = torch.cat([self._phi_B.b[:self.shared_k_dirs],
                         self.k_ded_b], dim=0)  # (M,)
        k_proj = torch.einsum('mh,blh->blm', k_W, u) + k_b  # (B, L, M)
        k_flat = k_proj.reshape(-1, 1).float()
        k_feat = self._phi_B.channel_mlp(k_flat)
        k_feat = k_feat.view(B_batch, L, M, self._phi_B.L)
        K = (k_feat * self._phi_B.scale).reshape(B_batch, L, P).bfloat16()

        V = self.v_proj(u.bfloat16())  # (B, L, P)

        Q = self.q_norm(Q.view(B_batch, L, nh, hd)).transpose(1, 2)
        K = self.k_norm(K.view(B_batch, L, nh, hd)).transpose(1, 2)
        V = V.view(B_batch, L, nh, hd).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B_batch, L, P)

        return y_ssm + self.gate * self.out_proj(attn_out).float()


class S6(nn.Module):
    def __init__(self, d_model, d_state=64, M=4, layer_idx=None, num_heads=1, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h

        # Multi-scale depthwise conv (before projections)
        assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4 for msconv"
        self.msconv = MultiScaleDepthwiseConv(d_model, kernel_sizes=(3, 6, 12))

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

        # Post-readout norm (before attention residual)
        self.readout_norm = RMSNorm(self.h)

        # Attention (80/20 weight sharing with phi_B and c_proj)
        self.attn = S6Attention(d_model, d_state, M=M, num_heads=num_heads,
                                phi_B=self.kernel.phi_B, c_proj=self.c_proj)

    def forward(self, u, **kwargs):
        """ Input and output shape (B, L, H) """
        B, L, H = u.shape

        # msconv first
        x = self.msconv(u)

        # Run SSM scan (phi_B computed internally by kernel)
        h, cum_theta = self.kernel(x)

        # Input-dependent C gating + static C readout
        c_proj_out = self.c_proj(x)  # (B, L, P)
        c_gate = self.c_norm(F.silu(c_proj_out)) + self.c_bias  # (B, L, P)
        c_gate = apply_rotary_emb(c_gate, cum_theta)  # rotate to match state
        h_gated = h * c_gate  # (B, L, P) — input-dependent selection of state dims

        # MIMO readout: (H, P) complex @ (B, L, P) complex -> (B, L, H), take real
        C = torch.view_as_complex(self.C)  # (H, P)
        y = torch.einsum('hp,blp->blh', C, h_gated.to(C.dtype)).real

        # Skip connection + activation
        y = y + x * self.D
        y = F.silu(y)

        # Post-readout norm + residual + attention
        y_normed = self.readout_norm(y) + x
        y = self.attn(x, y_normed)

        return y
