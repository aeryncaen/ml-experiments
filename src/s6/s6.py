"""S6: Simplified SSM with direct H-dimensional scan, no bottleneck."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .scan import parallel_scan


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class S6Kernel(nn.Module):
    """Simplified S6: scan directly in H dimensions, no bottleneck."""

    def __init__(self, d_model, d_state=None, M=None, dt_min=0.001, dt_max=0.1, lr=None, **kwargs):
        # d_state and M are ignored - kept for API compatibility
        super().__init__()
        H = d_model
        self.H = H

        # A: learnable complex decay per hidden dim (H,)
        # Initialize with negative real part (stable) and small imaginary part
        log_A_real = torch.log(torch.linspace(1, H, H).float())  # spread of decay rates
        A_imag = torch.zeros(H)  # start with real-only
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        # Input-dependent dt: project to get timestep scaling
        self.dt_proj = nn.Linear(H, H)
        
        # dt bias init in log-uniform [dt_min, dt_max]
        log_dt_bias = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt_bias = nn.Parameter(log_dt_bias)

    def forward(self, u):
        """
        Args:
            u: (B, L, H) input sequence
        Returns:
            h: (B, L, H) scanned output
        """
        B, L, H = u.shape

        # Complex A: negative real part for stability
        log_A_real: torch.Tensor = self.log_A_real  # type: ignore
        A_imag: torch.Tensor = self.A_imag  # type: ignore
        A = -torch.exp(log_A_real) + 1j * A_imag  # (H,) complex

        # Input-dependent dt
        dt = F.softplus(self.dt_proj(u) + self.log_dt_bias)  # (B, L, H)

        # Complex discretization: alpha = exp(dt * A)
        alpha = torch.exp(dt.to(torch.cfloat) * A)  # (B, L, H) complex

        # Injection: scaled input
        inject = dt.to(torch.cfloat) * u.to(torch.cfloat)  # (B, L, H) complex

        # Parallel scan: h[t] = alpha[t] * h[t-1] + inject[t]
        h = parallel_scan(alpha, inject)  # (B, L, H) complex

        # Return real part
        return h.real

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S6Attention(nn.Module):
    """Attention operating on scanned state h."""

    def __init__(self, d_model, num_heads=1):
        super().__init__()
        H = d_model
        self.H = H
        self.num_heads = num_heads
        self.head_dim = H // num_heads
        assert H % num_heads == 0

        # Q, K, V projections from scanned state h
        self.q_proj = nn.Linear(H, H)
        self.k_proj = nn.Linear(H, H)
        self.v_proj = nn.Linear(H, H)
        self.out_proj = nn.Linear(H, H)

        # QK norm
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, h):
        """
        h: (B, L, H) - scanned state
        Returns: (B, L, H) - attention output
        """
        B, L, H = h.shape
        nh = self.num_heads
        hd = self.head_dim

        Q = self.q_proj(h).view(B, L, nh, hd)
        K = self.k_proj(h).view(B, L, nh, hd)
        V = self.v_proj(h).view(B, L, nh, hd)

        # QK norm
        Q = self.q_norm(Q).transpose(1, 2)  # (B, nh, L, hd)
        K = self.k_norm(K).transpose(1, 2)
        V = V.transpose(1, 2)

        # Causal attention
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, nh, L, hd)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, H)

        return self.out_proj(attn_out)


class S6(nn.Module):
    def __init__(self, d_model, d_state=None, M=None, num_heads=1, layer_idx=None, **kernel_args):
        # d_state and M are ignored - kept for API compatibility
        super().__init__()
        self.h = d_model
        self.d_output = self.h

        # D skip connection
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel (scan in H dimensions)
        self.kernel = S6Kernel(self.h, **kernel_args)

        # Attention on scanned state
        self.attn = S6Attention(d_model, num_heads=num_heads)

        # Mix gate: learnable scalar to blend SSM and attention
        self.mix_gate = nn.Parameter(torch.tensor(0.5))

        # Output norm
        self.out_norm = RMSNorm(self.h)

    def forward(self, u, **kwargs):
        """Input and output shape (B, L, H)"""
        B, L, H = u.shape

        # Scan directly on input (no msconv, no projection)
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
