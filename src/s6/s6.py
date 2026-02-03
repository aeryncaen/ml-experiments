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
    """Split channels into 4 groups: passthrough, 5 causal, 5 retro, 5 center. Then SE."""
    def __init__(self, channels):
        super().__init__()
        n_groups = 4
        assert channels % n_groups == 0, f"channels {channels} not divisible by {n_groups}"
        self.group_size = channels // n_groups
        self.kernel_specs = [
            ("pass", None),
            ("causal", 5),
            ("retro", 5),
            ("center", 5),
        ]
        # Convs for the 3 non-passthrough groups, in the same order as kernel_specs[1:]
        self.convs = nn.ModuleList([
            nn.Conv1d(
                self.group_size,
                self.group_size,
                k,
                padding=(k - 1 if mode in ("causal", "retro") else k // 2),
                groups=self.group_size,
            )
            for mode, k in self.kernel_specs[1:]
        ])
        self.se = SqueezeExcite1D(channels)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        gs = self.group_size
        n_groups = len(self.kernel_specs)
        conv_chunks = [x[..., i * gs:(i + 1) * gs] for i in range(n_groups)]
        out = []
        conv_idx = 0
        for chunk, (mode, _k) in zip(conv_chunks, self.kernel_specs):
            if mode == "pass":
                out.append(chunk)
                continue
            conv = self.convs[conv_idx]
            conv_idx += 1
            x_in = chunk.transpose(1, 2)
            if mode == "retro":
                x_in = torch.flip(x_in, dims=(-1,))
            y = conv(x_in)
            if mode in ("causal", "retro"):
                y = y[..., :L]
            if mode == "retro":
                y = torch.flip(y, dims=(-1,))
            y = F.silu(y).transpose(1, 2)
            out.append(y)
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

        # Fused input projection for dt, lam, theta, alpha2 (B is now separate via feature bank)
        # Layout: [dt(P), lam(P), theta(P//2), alpha2_real(P//2), alpha2_imag(P//2)]
        # alpha2 is a separate learned decay for t-2 position in AB2 (only needed for g1, g2 = P//2 total)
        # We project P//2 for each of real/imag since only half the state dims use AB2
        self.x_proj = nn.Linear(H, P + P + P // 2 + P // 2 + P // 2)
        self._split_sizes = [P, P, P // 2, P // 2, P // 2]

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
            theta_start = 2*P
            theta_end = 2*P + P//2
            self.x_proj.weight[theta_start:theta_end] = 0.0
            bias[theta_start:theta_end] = 0.0
            # alpha2 region: init to produce decay ~0.9 (sigmoid(2.2) ≈ 0.9)
            # alpha2_real: sigmoid activation, init bias for moderate decay
            alpha2_real_start = theta_end
            alpha2_real_end = alpha2_real_start + P//2
            bias[alpha2_real_start:alpha2_real_end] = 2.2  # sigmoid(2.2) ≈ 0.9
            # alpha2_imag: tanh activation scaled, init to zero (no rotation initially)
            alpha2_imag_start = alpha2_real_end
            alpha2_imag_end = alpha2_imag_start + P//2
            self.x_proj.weight[alpha2_imag_start:alpha2_imag_end] = 0.0
            bias[alpha2_imag_start:alpha2_imag_end] = 0.0

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
        assert P % 4 == 0, "d_state must be divisible by 4 for scan grouping"

        # Materialize complex A
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (P,) complex

        # B: use provided Bu_raw or compute from phi_B
        if Bu_raw is None:
            Bu_raw = self.phi_B(u.unsqueeze(1)).squeeze(1)  # (B, L, P)

        # dt, lam, theta, alpha2 from fused linear projection
        x_proj = self.x_proj(u)  # (B, L, P+P+P//2+P//2+P//2)
        dt_raw, lam_raw, theta, alpha2_real_raw, alpha2_imag_raw = x_proj.split(self._split_sizes, dim=-1)

        dt = F.softplus(F.silu(dt_raw) + self.log_dt_bias)  # (B, L, P)
        lam = torch.sigmoid(lam_raw)  # (B, L, P)
        
        # alpha2: learned complex decay for t-2 position in AB2
        # Magnitude from sigmoid (0 to 1), phase from tanh scaled to [-pi, pi]
        alpha2_mag = torch.sigmoid(alpha2_real_raw)  # (B, L, P//2)
        alpha2_phase = torch.tanh(alpha2_imag_raw) * math.pi  # (B, L, P//2)
        alpha2_half = alpha2_mag * torch.exp(1j * alpha2_phase)  # (B, L, P//2) complex
        # alpha2_half covers both g1 and g2 (each P//4 elements, total P//2)
        # First P//4 elements for g1 (causal), second P//4 for g2 (retro)

        Bu = self.b_norm(Bu_raw) + self.b_bias  # (B, L, P)

        # Data-dependent RoPE on B
        dt_half = dt.view(B, L, P // 2, 2).mean(-1)  # (B, L, P//2)
        cum_theta = torch.cumsum(dt_half * theta, dim=1)  # (B, L, P//2)
        Bu = apply_rotary_emb(Bu, cum_theta)
        # Bu_prev: shift AFTER rotation so position t-1 keeps its own rotation angle
        Bu_prev1 = F.pad(Bu[:, :-1], (0, 0, 1, 0))
        Bu_prev2 = F.pad(Bu[:, :-2], (0, 0, 2, 0))
        Bu_next1 = F.pad(Bu[:, 1:], (0, 0, 0, 1))
        Bu_next2 = F.pad(Bu[:, 2:], (0, 0, 0, 2))

        # Trapezoidal discretization (complex)
        alpha = torch.exp(dt.to(torch.cfloat) * A)  # (B, L, P) complex decay

        Bu_c = Bu.to(torch.cfloat)
        Bu_prev1_c = Bu_prev1.to(torch.cfloat)
        Bu_prev2_c = Bu_prev2.to(torch.cfloat)
        Bu_next1_c = Bu_next1.to(torch.cfloat)
        Bu_next2_c = Bu_next2.to(torch.cfloat)
        dt_c = dt.to(torch.cfloat)

        g = P // 4
        g0 = slice(0, g)
        g1 = slice(g, 2 * g)
        g2 = slice(2 * g, 3 * g)
        g3 = slice(3 * g, 4 * g)
        
        # Slice alpha2_half for g1 and g2 groups
        alpha2_g1 = alpha2_half[:, :, :g]   # (B, L, P//4) for causal AB2
        alpha2_g2 = alpha2_half[:, :, g:]   # (B, L, P//4) for retro AB2

        inject = torch.zeros_like(Bu_c)
        # pass-through: current only
        inject[:, :, g0] = lam[:, :, g0] * dt_c[:, :, g0] * Bu_c[:, :, g0]
        # causal AB2: look behind 2 with learned decay for t-2
        inject[:, :, g1] = (
            lam[:, :, g1] * dt_c[:, :, g1] * Bu_c[:, :, g1]
            + (1 - lam[:, :, g1]) * dt_c[:, :, g1] * (
                alpha[:, :, g1] * Bu_prev1_c[:, :, g1]
                + (alpha[:, :, g1] * alpha2_g1) * Bu_prev2_c[:, :, g1]
            )
        )
        # retro AB2: look ahead 2 with learned decay for t+2
        inject[:, :, g2] = (
            lam[:, :, g2] * dt_c[:, :, g2] * Bu_c[:, :, g2]
            + (1 - lam[:, :, g2]) * dt_c[:, :, g2] * (
                alpha[:, :, g2] * Bu_next1_c[:, :, g2]
                + (alpha[:, :, g2] * alpha2_g2) * Bu_next2_c[:, :, g2]
            )
        )
        # center: one behind + one ahead, single decay both directions
        inject[:, :, g3] = (
            lam[:, :, g3] * dt_c[:, :, g3] * Bu_c[:, :, g3]
            + (1 - lam[:, :, g3]) * dt_c[:, :, g3] * (
                alpha[:, :, g3] * (Bu_prev1_c[:, :, g3] + Bu_next1_c[:, :, g3])
            )
        )

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
    """Cross-attention over shared SSM hidden state.
    
    Q: projected from input u
    K, V: projected from scanned hidden state h (shared with SSM path)
    """
    def __init__(self, d_model, d_state, num_heads=1):
        super().__init__()
        H = d_model
        P = d_state
        self.H = H
        self.P = P
        self.num_heads = num_heads
        self.head_dim = P // num_heads
        assert P % num_heads == 0

        # Q from input, K/V from hidden state
        self.q_proj = nn.Linear(H, P, bias=False)
        self.k_proj = nn.Linear(P, P, bias=False)  # P -> P since h is (B, L, P)
        self.v_proj = nn.Linear(P, P, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.out_proj = nn.Linear(P, H, bias=False)

    def forward(self, u, h):
        """
        u: (B, L, H) — input for Q projection
        h: (B, L, P) — scanned hidden state (real part) for K/V
        Returns: (B, L, H) attention output
        """
        B_batch, L, H = u.shape
        P = self.P
        nh = self.num_heads
        hd = self.head_dim

        Q = self.q_proj(u)  # (B, L, P)
        K = self.k_proj(h)  # (B, L, P)
        V = self.v_proj(h)  # (B, L, P)

        Q = self.q_norm(Q.view(B_batch, L, nh, hd)).transpose(1, 2)
        K = self.k_norm(K.view(B_batch, L, nh, hd)).transpose(1, 2)
        V = V.view(B_batch, L, nh, hd).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B_batch, L, P)

        return self.out_proj(attn_out)


class S6(nn.Module):
    """S6: Shared-state SSM + Attention with learned mixing.
    
    Architecture:
    1. Run SSM scan on input → hidden state h
    2. SSM path: C readout from h → y_ssm  
    3. Attention path: Q from input, K/V from h → y_attn
    4. Learned gate mixes: gate * y_ssm + (1 - gate) * y_attn
    """
    def __init__(self, d_model, d_state=64, M=4, layer_idx=None, num_heads=1, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h

        # D skip connection
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel (parallel scan)
        self.kernel = S6Kernel(self.h, N=self.n, M=M, **kernel_args)

        # C: static MIMO readout (H, P) complex — maps state back to output channels
        C = torch.randn(self.h, self.n, dtype=torch.cfloat) / math.sqrt(self.n)
        self.C = nn.Parameter(torch.view_as_real(C))  # (H, P, 2) stored as real
        self.c_proj = nn.Linear(self.h, self.n)  # input-dependent gating of C
        self.c_norm = RMSNorm(self.n)
        self.c_bias = nn.Parameter(torch.ones(self.n))

        # Attention over shared hidden state
        self.attn = S6Attention(d_model, d_state, num_heads=num_heads)

        # Learned gate to mix SSM and attention outputs
        # Initialized to 0.5 for equal mixing, projects from input for data-dependent gating
        self.mix_gate = nn.Linear(d_model, d_model, bias=True)
        with torch.no_grad():
            self.mix_gate.weight.zero_()
            self.mix_gate.bias.fill_(0.0)  # sigmoid(0) = 0.5

        # Output norm
        self.out_norm = RMSNorm(d_model)

    def forward(self, u, **kwargs):
        """ Input and output shape (B, L, H) """
        B, L, H = u.shape

        # Run SSM scan directly on input (no conv preprocessing)
        h, cum_theta = self.kernel(u)  # h: (B, L, P) complex

        # === SSM Path ===
        # Input-dependent C gating + static C readout
        c_proj_out = self.c_proj(u)  # (B, L, P)
        c_gate = self.c_norm(F.silu(c_proj_out)) + self.c_bias  # (B, L, P)
        c_gate = apply_rotary_emb(c_gate, cum_theta)  # rotate to match state
        h_gated = h * c_gate  # (B, L, P) — input-dependent selection of state dims

        # MIMO readout: (H, P) complex @ (B, L, P) complex -> (B, L, H), take real
        C = torch.view_as_complex(self.C)  # (H, P)
        y_ssm = torch.einsum('hp,blp->blh', C, h_gated.to(C.dtype)).real

        # === Attention Path ===
        # Use real part of h for attention (Q from u, K/V from h)
        h_real = h.real.float()  # (B, L, P)
        y_attn = self.attn(u, h_real)  # (B, L, H)

        # === Mix SSM and Attention with learned gate ===
        gate = torch.sigmoid(self.mix_gate(u))  # (B, L, H) data-dependent gate
        y = gate * y_ssm + (1 - gate) * y_attn

        # Skip connection + activation + norm
        y = y + u * self.D
        y = self.out_norm(F.silu(y))

        return y
