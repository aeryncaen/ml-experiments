"""S6: S4D base + S5 MIMO + Mamba selectivity + trapezoidal discretization + data-dependent RoPE."""

import math
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .scan import parallel_scan


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
            ("center", 5),
            ("center", 5),
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
            y = conv(x_in)
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
    """S6 SSM: input-dependent discretization + parallel scan, direct H-dim (no bottleneck), real-valued."""

    def __init__(self, d_model, N=None, M=4, dt_min=0.001, dt_max=0.1, lr=None):
        # N (d_state) is ignored - we work directly in H dimensions
        super().__init__()
        H = d_model
        P = H  # No bottleneck: scan in model dimension
        assert P % 4 == 0, f"d_model ({P}) must be divisible by 4 for grouping"
        self.H = H
        self.P = P

        # A: learnable decay (H,) - spread of decay rates, all negative for stability
        log_A = torch.log(torch.linspace(1, H, H).float())
        self.register("log_A", log_A, lr)

        # Fused input projection for dt, lam, alpha2
        # Layout: [dt(H), lam(H), alpha2(H)]
        # alpha2 is separate learned decay for t-1 position in AB-2
        self.x_proj = nn.Linear(H, H + H + H)
        self._split_sizes = [H, H, H]

        # Input norm + bias
        self.in_norm = RMSNorm(H)
        self.in_bias = nn.Parameter(torch.ones(H))

        # dt bias (init in log-uniform [dt_min, dt_max])
        log_dt_bias = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt_bias = nn.Parameter(log_dt_bias)

        # Bias init
        with torch.no_grad():
            bias = self.x_proj.bias
            # lam: sigmoid(2) ≈ 0.88 biased toward current input
            bias[H:2*H] = 2.0
            # alpha2: sigmoid(2) ≈ 0.88
            bias[2*H:] = 2.0

    def forward(self, u):
        """
        Args:
            u: (B, L, H) input sequence
        Returns:
            h: (B, L, H) scanned output
        """
        B, L, H = u.shape

        # Materialize A (negative for stable decay)
        log_A: torch.Tensor = self.log_A  # type: ignore
        A = -torch.exp(log_A)  # (H,) negative real

        # Norm + bias on input
        x = self.in_norm(u) + self.in_bias  # (B, L, H)

        # dt, lam, alpha2 from fused projection
        proj = self.x_proj(u)
        dt_raw, lam_raw, alpha2_raw = proj.split(self._split_sizes, dim=-1)

        dt = F.softplus(F.silu(dt_raw) + self.log_dt_bias)  # (B, L, H)
        lam = torch.sigmoid(lam_raw)  # (B, L, H)
        alpha2 = torch.sigmoid(alpha2_raw)  # (B, L, H) in (0, 1)

        # Shifted inputs for AB-2
        x_prev1 = F.pad(x[:, :-1], (0, 0, 1, 0))
        x_next1 = F.pad(x[:, 1:], (0, 0, 0, 1))

        # Primary decay: alpha = exp(dt * A), in (0, 1) since A < 0
        alpha = torch.exp(dt * A)  # (B, L, H)

        g = H // 4
        g0 = slice(0, g)
        g1 = slice(g, 2 * g)
        g2 = slice(2 * g, 3 * g)
        g3 = slice(3 * g, 4 * g)

        inject = torch.zeros_like(x)
        # pass-through: current only
        inject[:, :, g0] = lam[:, :, g0] * dt[:, :, g0] * x[:, :, g0]
        # AB-2 groups: alpha2 for t-1, alpha for t+1
        for gs in (g1, g2, g3):
            inject[:, :, gs] = (
                lam[:, :, gs] * dt[:, :, gs] * x[:, :, gs]
                + (1 - lam[:, :, gs]) * dt[:, :, gs] * (
                    alpha2[:, :, gs] * x_prev1[:, :, gs] + alpha[:, :, gs] * x_next1[:, :, gs]
                )
            )

        # Parallel scan: h[t] = alpha[t] * h[t-1] + inject[t]
        h = parallel_scan(alpha, inject)  # (B, L, H)

        return h

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
    """Simple causal attention on scanned state h."""

    def __init__(self, d_model, num_heads=1):
        super().__init__()
        H = d_model
        self.H = H
        self.num_heads = num_heads
        self.head_dim = H // num_heads
        assert H % num_heads == 0

        self.q_proj = nn.Linear(H, H)
        self.k_proj = nn.Linear(H, H)
        self.v_proj = nn.Linear(H, H)
        self.out_proj = nn.Linear(H, H)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, h):
        """
        h: (B, L, H) — scanned state
        Returns: (B, L, H) — attention output
        """
        B, L, H = h.shape
        nh = self.num_heads
        hd = self.head_dim

        Q = self.q_proj(h).view(B, L, nh, hd)
        K = self.k_proj(h).view(B, L, nh, hd)
        V = self.v_proj(h).view(B, L, nh, hd)

        Q = self.q_norm(Q).transpose(1, 2)  # (B, nh, L, hd)
        K = self.k_norm(K).transpose(1, 2)
        V = V.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, H)

        return self.out_proj(attn_out)


class S6(nn.Module):
    def __init__(self, d_model, d_state=None, M=4, layer_idx=None, num_heads=1, **kernel_args):
        # d_state is ignored - we work in H dimensions (no bottleneck)
        super().__init__()

        self.h = d_model
        self.d_output = self.h

        # D skip connection
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel (parallel scan in H dimensions, no bottleneck)
        self.kernel = S6Kernel(self.h, M=M, **kernel_args)

        # Attention on scanned state
        self.attn = S6Attention(d_model, num_heads=num_heads)

        # Mix gate: learnable scalar to blend SSM and attention
        self.mix_gate = nn.Parameter(torch.tensor(0.5))

        # Output norm
        self.out_norm = RMSNorm(self.h)

    def forward(self, u, **kwargs):
        """Input and output shape (B, L, H)"""
        B, L, H = u.shape

        # Scan directly on input (no msconv, no C readout)
        h = self.kernel(u)  # (B, L, H)

        # SSM output: scanned state + skip
        y_ssm = h + u * self.D
        y_ssm = F.silu(y_ssm)

        # Attention on scanned state (shared h)
        y_attn = self.attn(h)

        # Mix SSM and attention outputs
        gate = torch.sigmoid(self.mix_gate)
        y = gate * y_ssm + (1 - gate) * y_attn

        return self.out_norm(y) + u  # residual
