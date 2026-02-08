"""Benchmark DS1 vs S4D vs Mamba vs USB on SSM litmus tests.

Tasks: delay-1, selective copy, induction head
Metrics: final loss, param count, peak memory, wall time

Requires: einops, mamba_ssm (for Mamba CUDA ops on CUDA box), tqdm
"""

import sys, os, time, math, random, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
try:
    from torch.profiler import profile, ProfilerActivity
    HAS_PROFILER = True
except Exception:
    HAS_PROFILER = False
from einops import rearrange, repeat
from tqdm import tqdm

DETERMINISTIC = os.getenv("BENCH_DETERMINISTIC", "0") == "1"
if DETERMINISTIC:
    # Must be set before any CUDA context is created
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        except Exception:
            pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mamba'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 's5-pytorch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zoology'))

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# S4D (standalone, from s4/models/s4/s4d.py — no DropoutNd dependency)
# ---------------------------------------------------------------------------
class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.log_dt = nn.Parameter(log_dt)
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        return K


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, **kernel_args):
        super().__init__()
        self.h = d_model
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=d_state, **kernel_args)
        self.activation = nn.GELU()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u):
        # u: (B, L, H) -> internal (B, H, L)
        u = u.transpose(-1, -2)
        L = u.size(-1)
        k = self.kernel(L=L)
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        y = self.activation(y)
        y = self.output_linear(y)
        return y.transpose(-1, -2)


# ---------------------------------------------------------------------------
# S5 (PyTorch port from s5-pytorch checkout)
# ---------------------------------------------------------------------------
try:
    from s5.s5_model import S5 as S5Module
    HAS_S5 = True
except ImportError:
    HAS_S5 = False
    S5Module = None


class S5Wrapper(nn.Module):
    """Wraps S5 to accept (B, L, H) and return (B, L, H)."""
    def __init__(self, width, state_width=256):
        super().__init__()
        if not HAS_S5:
            raise ImportError("S5 not available. Install s5-pytorch.")
        assert S5Module is not None
        self.s5 = S5Module(width=width, state_width=state_width)

    def forward(self, x):
        return self.s5(x)


# ---------------------------------------------------------------------------
# Mamba (real implementation from mamba checkout)
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None


class MambaWrapper(nn.Module):
    """Wraps Mamba to accept (B, L, D) and return (B, L, D)."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("Mamba not available. Install mamba_ssm.")
        assert Mamba is not None
        self.mamba = Mamba(d_model=d_model, use_fast_path=True, **kwargs)

    def forward(self, x):
        return self.mamba(x)


# ---------------------------------------------------------------------------
# DS1 wrapper
# ---------------------------------------------------------------------------
from ds_moe.model import DS1, RMSNorm

# ---------------------------------------------------------------------------
# S6 / USB (Unified Sequence Block)
# ---------------------------------------------------------------------------
from s6.usb_block import USBBlock, USBConfig
from ideal.stack import IdealWrapper
from ulb import ULBBlock, ULBConfig


class USBWrapper(nn.Module):
    """Wraps USB to accept (B, L, H) and return (B, L, H)."""
    def __init__(self, d_model, headdim=64, expansion_factor=2, layer_idx=0, 
                 scan_state_modes=('elementwise', 'elementwise', 'elementwise'), **kwargs):
        super().__init__()
        config = USBConfig(
            d_model=d_model,
            headdim=headdim,
            expansion_factor=expansion_factor,
            layer_idx=layer_idx,
            scan_state_modes=scan_state_modes,
        )
        self.usb = USBBlock(config)

    def forward(self, x):
        return self.usb(x)



class MHABlock(nn.Module):
    """Causal conv3 → causal MHA → SwiGLU MLP."""
    def __init__(self, d_model, n_heads=4, mlp_inner=0):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0, groups=d_model, bias=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.has_mlp = mlp_inner > 0
        if self.has_mlp:
            self.mlp_norm = RMSNorm(d_model)
            self.gate_proj = nn.Linear(d_model, mlp_inner, bias=False)
            self.up_proj = nn.Linear(d_model, mlp_inner, bias=False)
            self.down_proj = nn.Linear(mlp_inner, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = F.pad(x.transpose(1, 2), (2, 0))
        h = self.conv(h).transpose(1, 2)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        if self.has_mlp:
            r = self.mlp_norm(h)
            h = h + self.down_proj(F.silu(self.gate_proj(r)) * self.up_proj(r))
        return h


class FusedGateBlock(nn.Module):
    """Fused attention+MLP: up_proj → swish → QKV → attn → skip-multiply → swish → down_proj.

    No internal residual — caller adds residual. No internal pre-norm — caller pre-norms.
    Matches the fused_gate architecture from train_fineweb_vanilla.py.
    """

    def __init__(self, d_model, n_heads=4, paired=False, attn_mode='softmax'):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_heads
        self.paired = paired
        self.attn_mode = attn_mode
        self.inner_dim = d_model  # no expansion
        assert self.inner_dim % n_heads == 0
        self.head_dim = self.inner_dim // n_heads
        self.half_dim = self.head_dim // 2

        # up/down projections
        self.up_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.down_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        # QKV (KV have bias)
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        # Learnable Swish beta (per-channel)
        self.swish_beta_up = nn.Parameter(torch.ones(self.inner_dim))
        self.swish_beta_down = nn.Parameter(torch.ones(self.inner_dim))

        # Post-attention norm (before skip-multiply)
        self.attn_norm = nn.RMSNorm(self.inner_dim)

        # QK norm + post-norm bias (Mamba-3 style BC bias, init ones)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.q_bias = nn.Parameter(torch.ones(self.head_dim))
        self.k_bias = nn.Parameter(torch.ones(self.head_dim))

        # RoPE: first half of rotation pairs = fixed, second half = data-dependent
        # head_dim operates in pairs for rotation, so head_dim//2 pairs total
        self.rope_fixed_pairs = self.head_dim // 4   # first half of pairs: fixed RoPE
        self.rope_dd_pairs = self.head_dim // 4      # second half of pairs: data-dependent
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rope_fixed_pairs * 2, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)  # (rope_fixed_pairs,)
        # Data-dependent RoPE: project x → per-head rotation deltas, then cumsum
        self.rope_dd_proj = nn.Linear(d_model, n_heads * self.rope_dd_pairs, bias=True)
        nn.init.zeros_(self.rope_dd_proj.weight)
        nn.init.zeros_(self.rope_dd_proj.bias)
        if paired:
            assert n_heads % 2 == 0
            inv_freq_p = 1.0 / (10000 ** (torch.arange(0, self.rope_fixed_pairs * 2, 2).float() / self.head_dim))
            self.register_buffer('inv_freq_paired', inv_freq_p, persistent=False)

        # K causal lerp: 1/4 of head_dim (last quarter)
        self.quarter_dim = self.head_dim // 4
        self.k_gate_proj = nn.Linear(d_model, n_heads * self.quarter_dim, bias=True)
        nn.init.zeros_(self.k_gate_proj.weight)
        nn.init.constant_(self.k_gate_proj.bias, -2.0)

        # Q acausal lerp: 1/2 of head_dim (last half), separate fwd/bwd gates
        self.q_gate_fwd_proj = nn.Linear(d_model, n_heads * self.half_dim, bias=True)
        self.q_gate_bwd_proj = nn.Linear(d_model, n_heads * self.half_dim, bias=True)
        nn.init.zeros_(self.q_gate_fwd_proj.weight)
        nn.init.constant_(self.q_gate_fwd_proj.bias, -2.0)
        nn.init.zeros_(self.q_gate_bwd_proj.weight)
        nn.init.constant_(self.q_gate_bwd_proj.bias, -2.0)

        # Blend: gate + RMSNorm on silu² output to match softmax output scale
        if attn_mode == 'blend':
            self.blend_gate_proj = nn.Linear(d_model, n_heads, bias=True)
            nn.init.zeros_(self.blend_gate_proj.weight)
            nn.init.constant_(self.blend_gate_proj.bias, -1.1)
            self.silu2_norm = nn.RMSNorm(self.head_dim)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        """Apply rotary embedding. x: (..., 2*n_pairs), cos/sin: (..., n_pairs)."""
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)

    def _dd_rope_angles(self, x):
        """Compute cumulative data-dependent rotation angles.
        x: (B, T, D) → returns (B, T, H, rope_dd_pairs) cumulative angles.
        """
        b, t, _ = x.shape
        # Project to per-head rotation deltas
        deltas = self.rope_dd_proj(x)  # (B, T, H * rope_dd_pairs)
        deltas = deltas.view(b, t, self.n_head, self.rope_dd_pairs)
        # Cumulative sum across time → cumulative rotation angle
        return deltas.cumsum(dim=1)

    def _hybrid_rope(self, qk, dd_angles):
        """Apply hybrid RoPE: fixed on first half of pairs, data-dependent on second half.
        qk:        (B, T, H, head_dim)
        dd_angles: (B, T, H, rope_dd_pairs) — cumulative angles for dynamic dims
        """
        fp = self.rope_fixed_pairs
        dp = self.rope_dd_pairs
        T = qk.shape[1]

        # Split into fixed and dynamic dim regions
        # Layout: [fixed_x1 (fp) | dd_x1 (dp) | fixed_x2 (fp) | dd_x2 (dp)]
        qk_fixed = torch.cat([qk[..., :fp], qk[..., fp+dp:fp+dp+fp]], dim=-1)  # (B,T,H,2*fp)
        qk_dd = torch.cat([qk[..., fp:fp+dp], qk[..., fp+dp+fp:]], dim=-1)     # (B,T,H,2*dp)

        # Fixed RoPE
        t = torch.arange(T, device=qk.device, dtype=self.inv_freq.dtype)
        freqs_fixed = torch.outer(t, self.inv_freq)  # (T, fp)
        cos_f = freqs_fixed.cos()[None, :, None, :]   # (1, T, 1, fp)
        sin_f = freqs_fixed.sin()[None, :, None, :]
        qk_fixed = self._apply_rotary(qk_fixed, cos_f, sin_f)

        # Data-dependent RoPE
        cos_d = dd_angles.cos()  # (B, T, H, dp)
        sin_d = dd_angles.sin()
        qk_dd = self._apply_rotary(qk_dd, cos_d, sin_d)

        # Reassemble: [fixed_x1 | dd_x1 | fixed_x2 | dd_x2]
        return torch.cat([qk_fixed[..., :fp], qk_dd[..., :dp],
                          qk_fixed[..., fp:], qk_dd[..., dp:]], dim=-1)

    def _paired_rope(self, q_or_k, dd_angles):
        """Even/odd position encodings for paired head dim (2*head_dim width).
        q_or_k:    (B, T, n_head//2, head_dim*2)
        dd_angles: (B, T, n_head, rope_dd_pairs) — will be reshaped for paired heads
        """
        b, T, n2, hd2 = q_or_k.shape
        fp = self.rope_fixed_pairs
        dp = self.rope_dd_pairs

        # Fixed RoPE with even/odd position indices
        t = torch.arange(T, device=q_or_k.device, dtype=self.inv_freq_paired.dtype)
        t_even, t_odd = 2 * t, 2 * t + 1
        freqs_even = torch.outer(t_even, self.inv_freq_paired)  # (T, fp)
        freqs_odd = torch.outer(t_odd, self.inv_freq_paired)
        # Paired: concat even+odd freqs → (T, 2*fp)
        cos_f = torch.cat([freqs_even.cos(), freqs_odd.cos()], dim=-1)[None, :, None, :]
        sin_f = torch.cat([freqs_even.sin(), freqs_odd.sin()], dim=-1)[None, :, None, :]

        # Data-dependent: reshape angles from (B,T,H,dp) → (B,T,n2,2*dp) for paired heads
        dd = dd_angles.view(b, T, n2, 2 * dp)
        cos_d = dd.cos()
        sin_d = dd.sin()

        # Split paired head_dim*2 into fixed and dd regions
        # head_dim*2 layout: [fp | dp | fp | dp | fp | dp | fp | dp]
        # Actually for paired, each "half" is head_dim, so:
        # [head0: fp|dp|fp|dp] [head1: fp|dp|fp|dp]
        # Simpler: treat as two head_dim chunks, apply hybrid to each, concat
        hd = self.head_dim
        h0 = q_or_k[..., :hd]
        h1 = q_or_k[..., hd:]
        dd0 = dd[..., :dp]
        dd1 = dd[..., dp:]

        # Fixed part of h0
        h0_fixed = torch.cat([h0[..., :fp], h0[..., fp+dp:fp+dp+fp]], dim=-1)
        h0_dd = torch.cat([h0[..., fp:fp+dp], h0[..., fp+dp+fp:]], dim=-1)
        # Apply fixed rope with even freqs to h0
        cos_f0 = freqs_even.cos()[None, :, None, :]
        sin_f0 = freqs_even.sin()[None, :, None, :]
        h0_fixed = self._apply_rotary(h0_fixed, cos_f0, sin_f0)
        h0_dd = self._apply_rotary(h0_dd, dd0.cos(), dd0.sin())
        h0 = torch.cat([h0_fixed[..., :fp], h0_dd[..., :dp],
                         h0_fixed[..., fp:], h0_dd[..., dp:]], dim=-1)

        # Fixed part of h1 (odd positions)
        h1_fixed = torch.cat([h1[..., :fp], h1[..., fp+dp:fp+dp+fp]], dim=-1)
        h1_dd = torch.cat([h1[..., fp:fp+dp], h1[..., fp+dp+fp:]], dim=-1)
        cos_f1 = freqs_odd.cos()[None, :, None, :]
        sin_f1 = freqs_odd.sin()[None, :, None, :]
        h1_fixed = self._apply_rotary(h1_fixed, cos_f1, sin_f1)
        h1_dd = self._apply_rotary(h1_dd, dd1.cos(), dd1.sin())
        h1 = torch.cat([h1_fixed[..., :fp], h1_dd[..., :dp],
                         h1_fixed[..., fp:], h1_dd[..., dp:]], dim=-1)

        return torch.cat([h0, h1], dim=-1)

    def _silu2_attention(self, q, k, v):
        """SiLU² attention: silu(logits)², unnormalized.
        q, k, v: (B, H, T, D) — standard SDPA layout.
        """
        scale = 1.0 / math.sqrt(q.shape[-1])
        logits = (q @ k.transpose(-2, -1)) * scale
        T = logits.shape[-1]
        causal_mask = torch.tril(torch.ones(T, T, device=logits.device, dtype=logits.dtype))
        weights = F.silu(logits) ** 2 * causal_mask
        return weights @ v

    def _blend_attention(self, q, k, v, gate):
        """Output-level lerp between softmax and RMSNorm'd SiLU² attention.
        gate: (B, H, T, 1) — per-position, per-head blend ratio (sigmoid).
        y = (1 - gate) * softmax_attn(q,k,v) + gate * rmsnorm(silu2_attn(q,k,v))
        """
        y_softmax = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y_silu2 = self.silu2_norm(self._silu2_attention(q, k, v))
        return (1 - gate) * y_softmax + gate * y_silu2

    def _attend(self, q, k, v, blend_gate=None):
        """Dispatch to the configured attention mode."""
        if self.attn_mode == 'silu2':
            return self._silu2_attention(q, k, v)
        elif self.attn_mode == 'blend':
            return self._blend_attention(q, k, v, blend_gate)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def _k_causal_lerp(self, k, gate):
        """1-step causal lerp on last quarter of K channels.
        k:    (B, T, H, head_dim)
        gate: (B, T, H, quarter_dim)
        """
        qd = self.quarter_dim
        k_static = k[:, :, :, :-qd]
        k_cur = k[:, :, :, -qd:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_mixed = (1 - gate) * k_cur + gate * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)

    def _q_acausal_lerp(self, q, g_fwd, g_bwd):
        """Acausal lerp on last half of Q channels. Separate fwd/bwd gates.
        Last token gets causal-only (no forward neighbor).
        q:     (B, T, H, head_dim)
        g_fwd: (B, T, H, half_dim)
        g_bwd: (B, T, H, half_dim)
        """
        hd = self.half_dim
        q_static = q[:, :, :, :-hd]
        q_cur = q[:, :, :, -hd:]
        q_prev = F.pad(q_cur[:, :-1], (0, 0, 0, 0, 1, 0))   # causal: t-1
        q_next = F.pad(q_cur[:, 1:],  (0, 0, 0, 0, 0, 1))   # acausal: t+1, last pos gets zeros
        q_mixed = (1 - g_fwd - g_bwd) * q_cur + g_fwd * q_next + g_bwd * q_prev
        return torch.cat([q_static, q_mixed], dim=-1)

    def forward(self, x):
        b, t, d = x.shape

        # Expand and activate
        h_up = self.up_proj(x)
        h_up = h_up * torch.sigmoid(self.swish_beta_up * h_up)

        # QKV
        q = self.q_proj(h_up).view(b, t, self.n_head, self.head_dim)
        k = self.k_proj(h_up).view(b, t, self.n_head, self.head_dim)
        v = self.v_proj(h_up).view(b, t, self.n_head, self.head_dim)

        # QK norm + post-norm bias (Mamba-3 style BC bias)
        q = self.q_norm(q) * self.q_bias
        k = self.k_norm(k) * self.k_bias

        # K causal lerp (1/4 dims) — before RoPE
        k_gate = torch.sigmoid(self.k_gate_proj(x)).view(b, t, self.n_head, self.quarter_dim)
        k = self._k_causal_lerp(k, k_gate)

        # Q acausal lerp (1/2 dims) — before RoPE
        q_gf = torch.sigmoid(self.q_gate_fwd_proj(x)).view(b, t, self.n_head, self.half_dim)
        q_gb = torch.sigmoid(self.q_gate_bwd_proj(x)).view(b, t, self.n_head, self.half_dim)
        q = self._q_acausal_lerp(q, q_gf, q_gb)

        # Data-dependent rotation angles (shared for Q and K)
        dd_angles = self._dd_rope_angles(x)

        # Blend gate: per-position, per-head (only computed for blend mode)
        blend_gate = None
        if self.attn_mode == 'blend':
            blend_gate = torch.sigmoid(self.blend_gate_proj(x))  # (B, T, H)
            blend_gate = blend_gate.transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)

        if self.paired:
            n2 = self.n_head // 2
            q = q.view(b, t, n2, self.head_dim * 2)
            k = k.view(b, t, n2, self.head_dim * 2)
            q = self._paired_rope(q, dd_angles)
            k = self._paired_rope(k, dd_angles)
            q = q.view(b, t * 2, n2, self.head_dim)
            k = k.view(b, t * 2, n2, self.head_dim)
            v = v.reshape(b, t * 2, n2, self.head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if blend_gate is not None:
                # Paired doubles T: interleave gate for even/odd positions
                # (B, H, T, 1) → (B, n2, 2T, 1) by repeating each position for both sub-heads
                bg = blend_gate.view(b, n2, 2, t, 1)  # (B, n2, 2_heads, T, 1)
                blend_gate = bg.permute(0, 1, 3, 2, 4).reshape(b, n2, t * 2, 1)  # interleave
            y = self._attend(q, k, v, blend_gate)
            y = y.transpose(1, 2).contiguous().view(b, t, self.n_head, self.head_dim)
        else:
            q = self._hybrid_rope(q, dd_angles)
            k = self._hybrid_rope(k, dd_angles)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = self._attend(q, k, v, blend_gate)
            y = y.transpose(1, 2)

        y = y.contiguous().view(b, t, self.inner_dim)

        # Skip-multiply
        y = self.attn_norm(y) * h_up

        # Down-project with Swish
        y = self.down_proj(y * torch.sigmoid(self.swish_beta_down * y))

        return y


class DS1Wrapper(nn.Module):
    def __init__(self, dim, state_dim=64, mimo_rank=4, n_iters=2, **kwargs):
        super().__init__()
        self.ds1 = DS1(dim=dim, state_dim=state_dim, mimo_rank=mimo_rank,
                        n_iters=n_iters, **kwargs)
        bank_size = DS1.bank_size(dim, state_dim, mimo_rank,
                                   out_gate=kwargs.get('out_gate', False),
                                   diff_attn=kwargs.get('diff_attn', False))
        self.bank = nn.Parameter(torch.randn(bank_size) * 0.02)

    def forward(self, x):
        return self.ds1(x, self.bank)


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------
# Unified vocab: 64 tokens (0-31 normal, 32-63 marked for selective_copy)
# All generators return (input, target) both (B, L) with ignore_index=-100 where needed.
VOCAB_SIZE = 64
ALL_TASKS = ['delay', 'selective_copy', 'induction', 'parity', 'mod_arith']


def gen_delay(B, L, device='cpu'):
    """Token input 0-31, target is same sequence shifted by 1 position."""
    x = torch.randint(0, 32, (B, L), device=device)
    target = torch.full_like(x, -100)
    target[:, 1:] = x[:, :L - 1]
    return x, target


def gen_selective_copy(B, L, n_markers=4, device='cpu'):
    """Tokens 0-31 unmarked, 32-63 marked. Target: last n_markers positions predict marked tokens."""
    tokens = torch.randint(0, 32, (B, L), device=device)
    inp = tokens.clone()
    target = torch.full((B, L), -100, dtype=torch.long, device=device)
    for b in range(B):
        idxs = torch.randperm(L, device=device)[:n_markers].sort().values
        inp[b, idxs] += 32  # mark in vocab
        target[b, -n_markers:] = tokens[b, idxs]
    return inp, target


def gen_parity(B, L, device='cpu'):
    """Binary input {0,1}, target is running XOR (cumulative parity)."""
    x = torch.randint(0, 2, (B, L), device=device)
    target = x.cumsum(dim=1) % 2
    return x, target


def gen_mod_arith(B, L, mod_base=5, device='cpu'):
    """Digit input 0..mod_base-1, target is running sum mod base."""
    x = torch.randint(0, mod_base, (B, L), device=device)
    target = x.cumsum(dim=1) % mod_base
    return x, target


def gen_induction(B, L, device='cpu'):
    """Two-copy repeated pattern with variable length, predict next token.

    Each sample: random pattern of length plen (L//3 to 2L//3), copied
    twice. Window always starts at position 0 of the double-pattern.
    Variable pattern length means the repeat boundary is at a different
    position each sample, preventing positional shortcuts.

    ALL positions scored. First copy is unpredictable. Causal ceiling
    is ~50%. Breaking past 50% requires acausal information flow.
    """
    inp = torch.zeros(B, L, dtype=torch.long, device=device)
    tgt = torch.zeros(B, L, dtype=torch.long, device=device)
    lo = L // 3          # shortest pattern
    hi = 2 * L // 3      # longest pattern
    lo = max(lo, (L + 2) // 2)  # must have 2*plen >= L+1
    if hi < lo:
        hi = lo
    for b in range(B):
        plen = torch.randint(lo, hi + 1, (1,)).item()
        pattern = torch.randint(0, 32, (plen,), device=device)
        seq = torch.cat([pattern, pattern])  # 2*plen >= L+1
        inp[b] = seq[:L]
        tgt[b] = seq[1:L + 1]
    return inp, tgt


def gen_mixed(B, L, device='cpu'):
    """Mixed batch: each sample drawn from a random task.
    Returns (input, target, task_ids) where task_ids is (B,) index into ALL_TASKS."""
    inp = torch.zeros(B, L, dtype=torch.long, device=device)
    tgt = torch.full((B, L), -100, dtype=torch.long, device=device)
    task_ids = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        tid = random.randint(0, len(ALL_TASKS) - 1)
        task = ALL_TASKS[tid]
        task_ids[b] = tid
        _g = {
            'delay': gen_delay, 'selective_copy': gen_selective_copy,
            'induction': gen_induction, 'parity': gen_parity, 'mod_arith': gen_mod_arith,
        }
        x, t = _g[task](1, L, device=device)
        inp[b] = x[0]
        tgt[b] = t[0]

    return inp, tgt, task_ids


# ---------------------------------------------------------------------------
# Data pregeneration, disk caching, and GPU preloading
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / '.bench_cache'


def _cache_key(task_name, n_steps, B, L, seed):
    raw = f"{task_name}_{n_steps}_{B}_{L}_{seed}"
    return hashlib.md5(raw.encode()).hexdigest()


def pregen_task_data(task_name, n_train_batches, n_val_batches, B, L, seed, device='cpu'):
    """Generate train and val batches for a task, cache to disk, return dict with train/val lists."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(task_name, n_train_batches + n_val_batches, B, L, seed)
    cache_path = CACHE_DIR / f"{task_name}_{key}.pt"

    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        train_batches = cached['train']
        val_batches = cached['val']
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        _gen = {
            'delay': lambda: gen_delay(B, L, device='cpu'),
            'selective_copy': lambda: gen_selective_copy(B, L, device='cpu'),
            'parity': lambda: gen_parity(B, L, device='cpu'),
            'mod_arith': lambda: gen_mod_arith(B, L, device='cpu'),
            'induction': lambda: gen_induction(B, L, device='cpu'),
            'mixed': lambda: gen_mixed(B, L, device='cpu'),
        }
        def gen_batch(task_name):
            return _gen[task_name]()
        
        train_batches = []
        for _ in tqdm(range(n_train_batches), desc=f"Generating {task_name} train", leave=False):
            train_batches.append(gen_batch(task_name))
        
        val_batches = []
        for _ in tqdm(range(n_val_batches), desc=f"Generating {task_name} val", leave=False):
            val_batches.append(gen_batch(task_name))
        
        torch.save({'train': train_batches, 'val': val_batches}, cache_path)

    # Preload everything to target device
    def _to(t):
        return t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t
    
    return {
        'train': [tuple(_to(t) for t in batch) for batch in train_batches],
        'val': [tuple(_to(t) for t in batch) for batch in val_batches],
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _set_usb_debug(model, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, USBBlock):
            m.debug = enabled


def _set_usb_debug_active(model, active: bool) -> None:
    for m in model.modules():
        if isinstance(m, USBBlock):
            m.debug_active = active


def _collect_usb_debug(model):
    logs = []
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            usb = getattr(layer, 'usb', None)
            dbg = getattr(usb, 'last_debug', None) if usb is not None else None
            if dbg is not None:
                logs.append((i, dbg))
    else:
        for i, m in enumerate([m for m in model.modules() if isinstance(m, USBBlock)]):
            if m.last_debug is not None:
                logs.append((i, m.last_debug))
    return logs


def _format_usb_debug(step: int, layer_idx: int, dbg: dict) -> list[str]:
    def fmt_stats(s: dict) -> str:
        return f"{s['mean']:.2e}/{s['min']:.2e}/{s['max']:.2e}"

    def fmt_gate(s: dict) -> str:
        return f"{s['mean']:.2e} lo={s['low']:.2e} hi={s['high']:.2e}"

    def fmt_seq(s: dict) -> str:
        return f"t0={s['t0']:.2e} tm={s['tmid']:.2e} te={s['tend']:.2e} min={s['min']:.2e} max={s['max']:.2e}"

    lines = []
    for g in ['g1', 'g2', 'g3']:
        pg = dbg[g]
        if 'delta' in pg and 'epsilon' in pg and 'zeta' in pg:
            lines.append(
                f"[usb] step={step} layer={layer_idx} {g} "
                f"alpha={fmt_stats(pg['alpha'])} delta={fmt_stats(pg['delta'])} "
                f"epsilon={fmt_stats(pg['epsilon'])} zeta={fmt_stats(pg['zeta'])} gate={fmt_gate(pg['gate'])}"
            )
        elif 'gamma' in pg:
            lines.append(
                f"[usb] step={step} layer={layer_idx} {g} "
                f"alpha={fmt_stats(pg['alpha'])} beta={fmt_stats(pg['beta'])} gamma={fmt_stats(pg['gamma'])} "
                f"delta={fmt_stats(pg['delta'])} lambda={fmt_stats(pg['lambda'])} gate={fmt_gate(pg['gate'])}"
            )
        else:
            lines.append(
                f"[usb] step={step} layer={layer_idx} {g} "
                f"alpha={fmt_stats(pg['alpha'])} beta={fmt_stats(pg['beta'])} gate={fmt_gate(pg['gate'])}"
            )

    rms = dbg['rms']
    lines.append(
        f"[usb] step={step} layer={layer_idx} rms "
        f"k1={rms['k_g1']:.2e} v1={rms['v_g1']:.2e} s1={rms['state_g1']:.2e} "
        f"sr1={rms['state_read_g1']:.2e} o1={rms['out_g1']:.2e}"
    )
    lines.append(
        f"[usb] step={step} layer={layer_idx} rms "
        f"k2={rms['k_g2']:.2e} v2={rms['v_g2']:.2e} s2={rms['state_g2']:.2e} "
        f"sr2={rms['state_read_g2']:.2e} o2={rms['out_g2']:.2e}"
    )
    lines.append(
        f"[usb] step={step} layer={layer_idx} rms "
        f"v3={rms['v_g3']:.2e} c3={rms['conv_g3']:.2e} o3={rms['out_g3']:.2e} attn={rms['attn_out']:.2e}"
    )

    seq = dbg['seq_rms']
    lines.append(f"[usb] step={step} layer={layer_idx} seq g1 {fmt_seq(seq['g1'])}")
    lines.append(f"[usb] step={step} layer={layer_idx} seq g2 {fmt_seq(seq['g2'])}")

    router = dbg.get('router')
    if router is not None:
        if router.get('enabled', True):
            overlap = router['topk_overlap']
            overlap_str = f"{overlap:.2e}" if overlap is not None else "n/a"
            lines.append(
                f"[usb] step={step} layer={layer_idx} router "
                f"temp={router['temp']:.2e} smooth={router['smooth']:.0f} "
                f"K={router['centers']:.0f} W={router['window']:.0f} "
                f"entropy={router['topk_entropy']:.2e} overlap={overlap_str} "
                f"w={router['topk_weight_min']:.2e}/{router['topk_weight_mean']:.2e}/{router['topk_weight_max']:.2e} "
                f"score={router['scores_min']:.2e}/{router['scores_mean']:.2e}/{router['scores_max']:.2e} "
                f"raw={router['scores_raw_min']:.2e}/{router['scores_raw_mean']:.2e}/{router['scores_raw_max']:.2e} "
                f"lg={router['local_gate']:.2e} kvb={router['kv_blend']:.2e}"
            )
        else:
            lines.append(
                f"[usb] step={step} layer={layer_idx} router "
                f"enabled=0 temp={router['temp']:.2e} smooth={router['smooth']:.0f} "
                f"lg={router['local_gate']:.2e} kvb={router['kv_blend']:.2e} lrg={router['lowrank_gate']:.2e}"
            )
    return lines
def _grad_stats(named_params):
    total_sq = 0.0
    total_abs = 0.0
    total_elems = 0
    max_abs = 0.0
    max_name = None
    has_nan = False
    has_inf = False

    for name, param in named_params:
        grad = param.grad
        if grad is None:
            continue
        if grad.is_sparse:
            grad = grad.coalesce().values()
        if grad.numel() == 0:
            continue
        grad = grad.detach()

        has_nan = has_nan or torch.isnan(grad).any().item()
        has_inf = has_inf or torch.isinf(grad).any().item()

        abs_grad = grad.abs()
        max_val = abs_grad.max().item()
        if max_val > max_abs:
            max_abs = max_val
            max_name = name

        total_sq += grad.float().pow(2).sum().item()
        total_abs += abs_grad.float().sum().item()
        total_elems += grad.numel()

    global_norm = math.sqrt(total_sq)
    mean_abs = total_abs / total_elems if total_elems else 0.0
    return {
        'global_norm': global_norm,
        'mean_abs': mean_abs,
        'max_abs': max_abs,
        'max_name': max_name,
        'has_nan': has_nan,
        'has_inf': has_inf,
    }


def _tensor_stats(tensor: torch.Tensor) -> dict:
    t = tensor.detach()
    if t.is_sparse:
        t = t.coalesce().values()
    if t.numel() == 0:
        return {
            'mean_abs': 0.0,
            'max_abs': 0.0,
            'has_nan': False,
            'has_inf': False,
        }
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    abs_t = t.abs()
    return {
        'mean_abs': abs_t.float().mean().item(),
        'max_abs': abs_t.max().item(),
        'has_nan': has_nan,
        'has_inf': has_inf,
    }


def _make_activation_hook(name, store, active_flag):
    def hook(_module, _inputs, output):
        if not active_flag['on']:
            return
        out = output
        if isinstance(out, (tuple, list)) and len(out) > 0:
            out = out[0]
        if not torch.is_tensor(out):
            return
        store[name] = _tensor_stats(out)

    return hook


def train_task(model, task_name, dim, max_epochs=100, lr=1e-4, B=32, L=32, device='cpu',
               preloaded_data=None, early_stop_acc=0.99, grad_log_every=50,
               grad_explode=1e3, grad_vanish=1e-6, act_log_every=50,
               act_explode=1e3, usb_debug_every=0):
    """Train with epochs and early stopping when val accuracy exceeds threshold."""
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    tracked_named_params = list(model.named_parameters())

    if usb_debug_every:
        _set_usb_debug(model, True)

    act_stats_step = {}
    act_active = {'on': False}
    act_hooks = []

    vocab_size = VOCAB_SIZE  # 64 for all tasks
    embed = nn.Embedding(vocab_size, dim).to(device)
    head = nn.Linear(dim, vocab_size).to(device)
    opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
    tracked_named_params += [(f"embed.{n}", p) for n, p in embed.named_parameters()]
    tracked_named_params += [(f"head.{n}", p) for n, p in head.named_parameters()]

    if act_log_every:
        if isinstance(model, StackedModel):
            for i, layer in enumerate(model.layers):
                act_hooks.append(layer.register_forward_hook(
                    _make_activation_hook(f"layer{i}", act_stats_step, act_active)
                ))
        else:
            act_hooks.append(model.register_forward_hook(
                _make_activation_hook("model", act_stats_step, act_active)
            ))

    assert preloaded_data is not None, "preloaded_data required — use pregen_task_data()"
    train_data = preloaded_data['train']
    val_data = preloaded_data['val']

    def _forward(batch, hard_mine=False):
        if task_name == 'mixed':
            inp, tgt, task_ids = batch
        else:
            inp, tgt = batch
        B_cur = inp.shape[0]
        y = head(model(embed(inp)))  # (B, L, V)

        if hard_mine and B_cur > 1:
            # Hard mining (vectorized): per-sample loss, weight hard samples more
            L_cur = tgt.shape[1]
            # Per-position loss: (B, L)
            per_pos_loss = F.cross_entropy(
                y.reshape(-1, vocab_size), tgt.reshape(-1), ignore_index=-100, reduction='none'
            ).reshape(B_cur, L_cur)
            valid_mask = tgt != -100  # (B, L)
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)  # (B,)
            per_sample_loss = per_pos_loss.sum(dim=1) / valid_count  # (B,)
            # Weights: rank by loss, hard=2x, easy=0.5x
            with torch.no_grad():
                lo_v, hi_v = per_sample_loss.min(), per_sample_loss.max()
                if hi_v > lo_v:
                    ranks = (per_sample_loss.detach() - lo_v) / (hi_v - lo_v)
                    weights = 0.5 + 1.5 * ranks
                else:
                    weights = torch.ones(B_cur, device=inp.device)
                weights = weights / weights.mean()
                preds = y.argmax(dim=-1)  # (B, L)
                correct = ((preds == tgt) & valid_mask).float().sum(dim=1)
                acc = (correct / valid_count).mean().item()
            loss = (per_sample_loss * weights).mean()
        else:
            logits_flat = y.reshape(-1, vocab_size)
            tgt_flat = tgt.reshape(-1)
            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=-100)
            with torch.no_grad():
                mask = tgt_flat != -100
                if mask.any():
                    acc = (logits_flat[mask].argmax(-1) == tgt_flat[mask]).float().mean().item()
                else:
                    acc = 0.0
        return loss, acc

    def _eval_val():
        model.eval()
        val_losses = []
        total_correct = 0
        total_valid = 0
        per_task_correct = {t: 0 for t in ALL_TASKS}
        per_task_total = {t: 0 for t in ALL_TASKS}
        is_mixed = task_name == 'mixed'
        with torch.no_grad():
            for batch in val_data:
                if is_mixed:
                    inp, tgt, task_ids = batch
                else:
                    inp, tgt = batch
                    task_ids = None
                y = head(model(embed(inp)))
                logits_flat = y.reshape(-1, vocab_size)
                tgt_flat = tgt.reshape(-1)
                loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=-100)
                val_losses.append(loss.item())
                valid = tgt_flat != -100
                if valid.any():
                    correct = (logits_flat[valid].argmax(-1) == tgt_flat[valid])
                    total_correct += correct.sum().item()
                    total_valid += valid.sum().item()
                # Per-task breakdown for mixed
                if is_mixed and task_ids is not None:
                    for tid_idx, subtask in enumerate(ALL_TASKS):
                        mask_b = task_ids == tid_idx
                        if not mask_b.any():
                            continue
                        y_sub = y[mask_b]
                        tgt_sub = tgt[mask_b]
                        lf = y_sub.reshape(-1, vocab_size)
                        tf = tgt_sub.reshape(-1)
                        v = tf != -100
                        if v.any():
                            per_task_correct[subtask] += (lf[v].argmax(-1) == tf[v]).sum().item()
                            per_task_total[subtask] += v.sum().item()
        model.train()
        avg_loss = sum(val_losses) / len(val_losses)
        avg_acc = total_correct / total_valid if total_valid > 0 else 0.0
        subtask_accs = None
        if is_mixed:
            subtask_accs = {}
            for t in ALL_TASKS:
                subtask_accs[t] = per_task_correct[t] / per_task_total[t] if per_task_total[t] > 0 else 0.0
        return avg_loss, avg_acc, subtask_accs

    mem_baseline = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_baseline = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    
    n_train = len(train_data)

    # Cosine annealing with linear warmup: 1 epoch warmup from lr/10, then cosine decay to lr/20
    warmup_steps = n_train
    total_training_steps = n_train * max_epochs
    warmup_ratio = 0.1    # start at lr * 0.1
    min_ratio = 0.05      # decay to lr * 0.05
    def lr_schedule(step):
        if step < warmup_steps:
            return warmup_ratio + (1.0 - warmup_ratio) * step / warmup_steps
        # Cosine decay from 1.0 to min_ratio over remaining steps
        progress = (step - warmup_steps) / max(1, total_training_steps - warmup_steps)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_schedule)
    total_steps = 0
    initial_loss = None

    grad_min_norm = float('inf')
    grad_max_norm = 0.0
    grad_explode_steps = 0
    grad_vanish_steps = 0
    grad_nan_steps = 0
    grad_inf_steps = 0

    act_max_abs = 0.0
    act_explode_steps = 0
    act_nan_steps = 0
    act_inf_steps = 0
    
    # Track best result
    best_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_subtask_accs = None
    final_epoch = 0
    stop_reason = "MAX_EPOCH"
    
    hard_mine_epoch = int(max_epochs * 2 / 3)  # last 1/3 of training
    
    epoch_pbar = tqdm(range(max_epochs), desc="Epochs", leave=False)
    for epoch in epoch_pbar:
        use_hard_mine = task_name == 'mixed' and epoch >= hard_mine_epoch
        if epoch == hard_mine_epoch and task_name == 'mixed':
            tqdm.write(f"[hard mining] activated at epoch {epoch + 1}/{max_epochs}")
        
        # Shuffle train indices each epoch
        train_indices = torch.randperm(n_train).tolist()
        
        epoch_losses = []
        epoch_accs = []
        
        step_pbar = tqdm(train_indices, desc=f"Epoch {epoch+1}", leave=False)
        for idx in step_pbar:
            batch = train_data[idx]
            track_usb = usb_debug_every and (total_steps % usb_debug_every == 0)
            if track_usb:
                _set_usb_debug_active(model, True)
            track_act = act_log_every and (total_steps % act_log_every == 0)
            if track_act:
                act_stats_step.clear()
                act_active['on'] = True
            loss, acc = _forward(batch, hard_mine=use_hard_mine)
            act_active['on'] = False
            if track_usb:
                _set_usb_debug_active(model, False)
                for layer_idx, dbg in _collect_usb_debug(model):
                    for line in _format_usb_debug(total_steps, layer_idx, dbg):
                        tqdm.write(line)

            if track_act:
                step_max_abs = 0.0
                step_max_name = None
                step_has_nan = False
                step_has_inf = False
                for name, stats in act_stats_step.items():
                    if stats['max_abs'] > step_max_abs:
                        step_max_abs = stats['max_abs']
                        step_max_name = name
                    step_has_nan = step_has_nan or stats['has_nan']
                    step_has_inf = step_has_inf or stats['has_inf']

                act_max_abs = max(act_max_abs, step_max_abs)
                if step_has_nan:
                    act_nan_steps += 1
                if step_has_inf:
                    act_inf_steps += 1
                if step_max_abs > act_explode:
                    act_explode_steps += 1

                if step_has_nan or step_has_inf or step_max_abs > act_explode:
                    max_name = step_max_name if step_max_name else "n/a"
                    tqdm.write(
                        f"[act] step={total_steps} max_abs={step_max_abs:.3e} "
                        f"max_module={max_name} nan={step_has_nan} inf={step_has_inf}"
                    )
            
            opt.zero_grad(set_to_none=True)
            loss.backward()

            if grad_log_every and (total_steps % grad_log_every == 0):
                stats = _grad_stats(tracked_named_params)
                grad_min_norm = min(grad_min_norm, stats['global_norm'])
                grad_max_norm = max(grad_max_norm, stats['global_norm'])

                if stats['has_nan']:
                    grad_nan_steps += 1
                if stats['has_inf']:
                    grad_inf_steps += 1
                if stats['global_norm'] > grad_explode:
                    grad_explode_steps += 1
                if stats['global_norm'] < grad_vanish:
                    grad_vanish_steps += 1

                if (stats['has_nan'] or stats['has_inf'] or
                        stats['global_norm'] > grad_explode or stats['global_norm'] < grad_vanish):
                    max_name = stats['max_name'] if stats['max_name'] else "n/a"
                    tqdm.write(
                        f"[grad] step={total_steps} norm={stats['global_norm']:.3e} "
                        f"mean_abs={stats['mean_abs']:.3e} max_abs={stats['max_abs']:.3e} "
                        f"max_param={max_name} nan={stats['has_nan']} inf={stats['has_inf']}"
                    )
            opt.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            epoch_accs.append(acc)
            total_steps += 1
            
            step_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.1%}")
        
        # Record initial loss from first epoch
        if initial_loss is None:
            initial_loss = sum(epoch_losses[:10]) / min(10, len(epoch_losses))
        
        # Check for train accuracy plateau (memorization)
        train_acc = sum(epoch_accs) / len(epoch_accs)
        
        # Evaluate on validation set
        val_loss, val_acc, subtask_accs = _eval_val()
        final_epoch = epoch + 1
        
        # Track best
        if val_acc > best_val_acc:
            best_epoch = final_epoch
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_subtask_accs = subtask_accs
        
        if subtask_accs is not None:
            _short = {'delay': 'D', 'selective_copy': 'S', 'induction': 'I', 'parity': 'P', 'mod_arith': 'M'}
            parts = " ".join(f"{_short[t]}={a:.0%}" for t, a in subtask_accs.items())
            epoch_pbar.set_postfix_str(f"L={val_loss:.3f} A={val_acc:.0%} {parts}")
        else:
            epoch_pbar.set_postfix(val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.1%}")
        
        # Early stopping: converged
        # Mixed task: all subtasks must exceed threshold
        if subtask_accs is not None:
            converged = all(a >= early_stop_acc for a in subtask_accs.values())
        else:
            converged = val_acc >= early_stop_acc
        if converged:
            stop_reason = "CONVERGED"
            break
        
        # Early stopping: train plateau (100% train acc but val not improving)
        if train_acc >= 0.9999 and val_acc < early_stop_acc:
            stop_reason = "PLATEAU"
            break

    if device == 'cuda':
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    peak_mem = 0
    if device == 'cuda':
        peak_mem = (torch.cuda.max_memory_allocated() - mem_baseline) / 1024 / 1024  # MB

    if grad_log_every:
        if grad_min_norm == float('inf'):
            grad_min_norm = 0.0
        tqdm.write(
            f"[grad] summary min_norm={grad_min_norm:.3e} max_norm={grad_max_norm:.3e} "
            f"explode={grad_explode_steps} vanish={grad_vanish_steps} "
            f"nan={grad_nan_steps} inf={grad_inf_steps}"
        )

    if act_log_every:
        tqdm.write(
            f"[act] summary max_abs={act_max_abs:.3e} explode={act_explode_steps} "
            f"nan={act_nan_steps} inf={act_inf_steps}"
        )

    for hook in act_hooks:
        hook.remove()

    return {
        'initial': initial_loss if initial_loss else 0.0,
        'final': best_val_loss,
        'acc': best_val_acc,
        'best_epoch': best_epoch,
        'epochs': final_epoch,
        'steps': total_steps,
        'wall_s': wall,
        'peak_mem_mb': peak_mem,
        'converged': (all(a >= early_stop_acc for a in best_subtask_accs.values())
                      if best_subtask_accs is not None
                      else best_val_acc >= early_stop_acc),
        'stop_reason': stop_reason,
        'subtask_accs': best_subtask_accs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


class StackedModel(nn.Module):
    """Stack N identical layers with pre-norm residual connections."""
    def __init__(self, make_layer, n_layers, dim):
        super().__init__()
        self.layers = nn.ModuleList([make_layer() for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return self.final_norm(x)


class MoEStackedModel(nn.Module):
    """MoE grid: n_experts wide × n_layers tall.
    
    v1 (original): each layer routes itself — router picks experts, weighted sum of outputs.
    v2 (sender-routed): previous layer's output decides next layer's routing.
       Merge weights computed AFTER expert outputs are available.
    """
    def __init__(self, make_layer, n_layers, dim, n_experts=4, top_k=2, version=1):
        super().__init__()
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.version = version
        # Grid of experts: [layer][expert]
        self.experts = nn.ModuleList([
            nn.ModuleList([make_layer() for _ in range(n_experts)])
            for _ in range(n_layers)
        ])
        # Per-layer pre-norm
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)

        if version == 1:
            # v1: per-layer router decides own experts, blind weighting
            self.routers = nn.ModuleList([nn.Linear(dim, n_experts, bias=False) for _ in range(n_layers)])
            self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))
        elif version == 2:
            # v2: sender-routed — layer l's output routes to layer l+1
            # Layer 0 routing comes from input content
            # n_layers routers: router[l] uses layer l's output to pick layer l+1 experts
            # (router[0] uses input to pick layer 0 experts)
            self.routers = nn.ModuleList([nn.Linear(dim, n_experts, bias=False) for _ in range(n_layers)])
            # Per-layer merge scorer: looks at expert outputs to decide weights
            self.merge_scorers = nn.ModuleList([nn.Linear(dim, 1, bias=False) for _ in range(n_layers)])
            # Learned layer weighting for final output
            self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))

    def _v1_forward(self, x):
        B, T, D = x.shape
        layer_outputs = [x]

        for norm, router, experts in zip(self.norms, self.routers, self.experts):
            h = norm(x)
            h_pool = h.mean(dim=1)
            logits = router(h_pool)
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)

            expert_outs = torch.stack([e(h) for e in experts], dim=2)
            idx_expanded = topk_idx[:, None, :, None].expand(-1, T, -1, D)
            selected = expert_outs.gather(2, idx_expanded)
            out = (selected * topk_weights[:, None, :, None]).sum(dim=2)
            x = x + out
            layer_outputs.append(x)

        w = F.softmax(self.layer_weights, dim=0)
        x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))
        return self.final_norm(x)

    def _v2_forward(self, x):
        B, T, D = x.shape
        layer_outputs = [x]

        # Initial routing decision from input content
        route_signal = x.mean(dim=1)  # (B, D)

        for l, (norm, experts) in enumerate(zip(self.norms, self.experts)):
            h = norm(x)

            # Routing: decided by previous layer's output (or input for layer 0)
            logits = self.routers[l](route_signal)  # (B, n_experts)
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (B, top_k)

            # Run all experts, gather top-k per sample
            expert_outs = torch.stack([e(h) for e in experts], dim=2)  # (B, T, n_experts, D)
            idx_expanded = topk_idx[:, None, :, None].expand(-1, T, -1, D)  # (B, T, top_k, D)
            selected = expert_outs.gather(2, idx_expanded)  # (B, T, top_k, D)

            # Merge: score each expert output AFTER computation
            # Pool each expert's output, score it, add router logits for gradient flow
            selected_pooled = selected.mean(dim=1)  # (B, top_k, D)
            output_scores = self.merge_scorers[l](selected_pooled).squeeze(-1)  # (B, top_k)
            # Gather the router logits for selected experts — gives router gradient signal
            router_scores = topk_vals  # (B, top_k) — already the top-k logits
            scores = output_scores + router_scores
            merge_weights = F.softmax(scores, dim=-1)  # (B, top_k)

            out = (selected * merge_weights[:, None, :, None]).sum(dim=2)  # (B, T, D)
            x = x + out
            layer_outputs.append(x)

            # This layer's output becomes routing signal for next layer
            route_signal = x.mean(dim=1)  # (B, D)

        w = F.softmax(self.layer_weights, dim=0)
        x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))
        return self.final_norm(x)

    def forward(self, x):
        if self.version == 2:
            return self._v2_forward(x)
        return self._v1_forward(x)


def _stack(make_layer, n_layers, dim):
    """Always wrap with pre-norm residual stacking."""
    return StackedModel(make_layer, n_layers, dim)


def _stack_moe(make_layer, n_layers, dim, n_experts=4, top_k=2, version=1):
    """Wrap with MoE stacking."""
    return MoEStackedModel(make_layer, n_layers, dim, n_experts=n_experts, top_k=top_k, version=version)


def _count_stacked_params(make_fn, n_layers, dim):
    """Count params of a StackedModel without keeping it around."""
    m = _stack(make_fn, n_layers, dim)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def _count_layer_params(make_fn):
    """Count params of a single layer (fast, no stacking)."""
    m = make_fn()
    n = sum(p.numel() for p in m.parameters())
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n


def _find_knob(make_fn_factory, n_layers, dim, target_params, lo=1, hi=4096):
    """Binary search for the integer knob value that gets closest to target_params.
    make_fn_factory(knob) should return a make_fn (callable that creates one layer).
    Uses single-layer param counting + overhead formula for speed."""
    # Compute stacking overhead once: norms per layer + final norm
    norm_params_per_layer = dim  # RMSNorm has dim params
    stacking_overhead = (n_layers + 1) * norm_params_per_layer  # n_layers norms + final_norm
    layer_target = (target_params - stacking_overhead) // n_layers

    best_knob, best_diff = lo, float('inf')
    iters = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        iters += 1
        print(f"    _find_knob: iter={iters} lo={lo} hi={hi} mid={mid}", flush=True)
        try:
            layer_n = _count_layer_params(make_fn_factory(mid))
        except Exception as e:
            print(f"    _find_knob: mid={mid} failed: {e}", flush=True)
            hi = mid - 1
            continue
        total_n = layer_n * n_layers + stacking_overhead
        diff = abs(total_n - target_params)
        print(f"    _find_knob: mid={mid} -> {total_n:,} params (diff={diff:,})", flush=True)
        if diff < best_diff:
            best_diff = diff
            best_knob = mid
        if total_n < target_params:
            lo = mid + 1
        elif total_n > target_params:
            hi = mid - 1
        else:
            break
    print(f"    _find_knob: chose knob={best_knob} ({iters} iters)", flush=True)
    return best_knob


def make_models(dim, n_layers=1, requested_models=None, match_params=True, n_experts=4, top_k=2, moe=False):
    """Build models with configurable depth.
    
    If match_params=True, auto-size SSM/MHA internal dimensions to match FusedGateBlendP param count.
    If requested_models is provided, only build those models (skips unavailable ones with warning).
    """
    models = {}
    
    # Compute reference param count from FusedGateBlendP
    target_params = None
    if match_params:
        try:
            target_params = _count_stacked_params(
                lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=True, attn_mode='blend'),
                n_layers, dim
            )
            print(f"  Auto-sizing to match FusedGateBlendP: {target_params:,} params")
        except Exception:
            pass  # FusedGateBlock not available, skip matching
    
    def _wanted(name):
        return requested_models is None or name in requested_models

    def try_add(name, make_fn, cfg_desc):
        if not _wanted(name):
            return
        try:
            if moe:
                model = _stack_moe(make_fn, n_layers, dim, n_experts=n_experts, top_k=top_k)
            else:
                model = _stack(make_fn, n_layers, dim)
            setattr(model, '_bench_config', cfg_desc)
            models[name] = model
        except ImportError as e:
            print(f"  Warning: {name} not available ({e})")
        except Exception as e:
            print(f"  Warning: {name} failed to initialize ({e})")

    # DS1++: DS1 + signed sparse attention
    try_add('DS1++', lambda: DS1Wrapper(dim=dim, state_dim=48, mimo_rank=4, n_iters=2, diff_attn=True),
            f"DS1Wrapper(dim={dim}, state_dim=48, mimo_rank=4, n_iters=2, diff_attn=True)")

    # S4D: auto-size d_state to match target
    if target_params and _wanted('S4D'):
        s4d_dstate = _find_knob(lambda k: lambda: S4D(d_model=dim, d_state=k), n_layers, dim, target_params)
    else:
        s4d_dstate = 144
    try_add('S4D', lambda: S4D(d_model=dim, d_state=s4d_dstate),
            f"S4D(d_model={dim}, d_state={s4d_dstate})")

    # S5: auto-size state_width to match target
    if target_params and _wanted('S5'):
        s5_sw = _find_knob(lambda k: lambda: S5Wrapper(width=dim, state_width=k), n_layers, dim, target_params)
    else:
        s5_sw = 140
    try_add('S5', lambda: S5Wrapper(width=dim, state_width=s5_sw),
            f"S5Wrapper(width={dim}, state_width={s5_sw})")

    # Mamba: auto-size d_state to match target (expand=2 default)
    if target_params and _wanted('Mamba'):
        mamba_dstate = _find_knob(lambda k: lambda: MambaWrapper(d_model=dim, d_state=k, d_conv=4, expand=2), n_layers, dim, target_params)
    else:
        mamba_dstate = 72
    try_add('Mamba', lambda: MambaWrapper(d_model=dim, d_state=mamba_dstate, d_conv=4, expand=2),
            f"MambaWrapper(d_model={dim}, d_state={mamba_dstate}, d_conv=4, expand=2)")

    # USB: Full MHA with directional scans (elementwise state - good for state tracking)
    try_add('USB', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2,
                                       scan_state_modes=('elementwise', 'elementwise', 'elementwise')),
            f"USBWrapper(d_model={dim}, headdim=32, expansion_factor=2, scan_state_modes=('elementwise','elementwise','elementwise'))")
    
    # USB_outer: USB with outer product state (all groups - good for retrieval)
    try_add('USB_outer', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2,
                                             scan_state_modes=('outer', 'outer', 'outer')),
            f"USBWrapper(d_model={dim}, headdim=32, expansion_factor=2, scan_state_modes=('outer','outer','outer'))")
    
    # USB_hybrid: G1/G2 outer (retrieval), G3 elementwise (state tracking)
    try_add('USB_hybrid', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2,
                                              scan_state_modes=('outer', 'outer', 'elementwise')),
            f"USBWrapper(d_model={dim}, headdim=32, expansion_factor=2, scan_state_modes=('outer','outer','elementwise'))")

    # MHA: conv + attention + SwiGLU MLP, auto-size mlp_inner to match target
    if target_params:
        mha_mlp = _find_knob(lambda k: lambda: MHABlock(d_model=dim, n_heads=4, mlp_inner=k), n_layers, dim, target_params)
    else:
        mha_mlp = 0
    try_add('MHA', lambda: MHABlock(d_model=dim, n_heads=4, mlp_inner=mha_mlp),
            f"MHABlock(d_model={dim}, n_heads=4, mlp_inner={mha_mlp})")

    # Ideal: orthogonal projection attention (no softmax, Cayley-parameterized heads)
    try_add('Ideal', lambda: IdealWrapper(d_model=dim, n_heads=4, ffn_mult=4.0),
            f"IdealWrapper(d_model={dim}, n_heads=4, ffn_mult=4.0)")

    # FusedGate: fused attention+MLP block (up → swish → attn → skip-mul → swish → down)
    try_add('FusedGate', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=False),
            f"FusedGateBlock(d_model={dim}, n_heads=4, paired=False, attn_mode='softmax')")

    # FusedGatePaired: same but with paired head attention
    try_add('FusedGatePaired', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=True),
            f"FusedGateBlock(d_model={dim}, n_heads=4, paired=True, attn_mode='softmax')")

    # SiLU² attention variants
    try_add('FusedGateSilu2', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=False, attn_mode='silu2'),
            f"FusedGateBlock(d_model={dim}, n_heads=4, paired=False, attn_mode='silu2')")
    try_add('FusedGateSilu2P', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=True, attn_mode='silu2'),
            f"FusedGateBlock(d_model={dim}, n_heads=4, paired=True, attn_mode='silu2')")

    # Blend: output-level lerp between softmax and silu², content-dependent gate
    try_add('FusedGateBlend', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=False, attn_mode='blend'),
            f"FusedGateBlock(d_model={dim}, n_heads=4, paired=False, attn_mode='blend')")
    try_add('FusedGateBlendP', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=True, attn_mode='blend'),
            f"FusedGateBlock(d_model={dim}, n_heads=4, paired=True, attn_mode='blend')")

    # ULB (Universal Learning Block) — generalized FusedGateBlock from src/ulb/
    try_add('ULBBlendP', lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='blend')),
            f"ULBBlock(ULBConfig(d_model={dim}, n_heads=4, paired=True, attn_mode='blend'))")

    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark SSM models on litmus tests')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile on models')
    parser.add_argument('--compile-mode', type=str, default='default', 
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--dim', type=int, default=64, help='Model dimension')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--train-batches', type=int, default=100, help='Number of training batches per epoch')
    parser.add_argument('--val-batches', type=int, default=20, help='Number of validation batches')
    parser.add_argument('--early-stop-acc', type=float, default=0.99, help='Early stop when val acc exceeds this')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence length')
    parser.add_argument('--grad-log-every', type=int, default=50,
                        help='Log gradient stats every N steps (0 to disable)')
    parser.add_argument('--grad-explode', type=float, default=1e3,
                        help='Gradient norm threshold for explosion warning')
    parser.add_argument('--grad-vanish', type=float, default=1e-6,
                        help='Gradient norm threshold for vanishing warning')
    parser.add_argument('--act-log-every', type=int, default=50,
                        help='Log activation stats every N steps (0 to disable)')
    parser.add_argument('--act-explode', type=float, default=1e3,
                        help='Activation max-abs threshold for explosion warning')
    parser.add_argument('--usb-debug-every', type=int, default=0,
                        help='Log USB internal stats every N steps (0 to disable)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to test (default: all)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Specific tasks to test (default: all)')
    parser.add_argument('--no-match-params', action='store_true',
                        help='Disable auto-sizing SSMs to match FusedGateBlendP param count')
    parser.add_argument('--moe', action='store_true', help='Use MoE stacking for all models')
    parser.add_argument('--n-experts', type=int, default=4, help='Number of MoE experts per layer')
    parser.add_argument('--top-k', type=int, default=2, help='Top-k expert selection per sample')
    parser.add_argument('--csv', type=str, default=None,
                        help='Output CSV file path (optional)')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if args.compile:
        print(f"torch.compile enabled (mode={args.compile_mode})")
    
    dim = args.dim
    n_layers = args.layers
    all_tasks = ['delay', 'selective_copy', 'induction', 'parity', 'mod_arith']
    tasks = args.tasks if args.tasks else all_tasks
    max_epochs = args.max_epochs
    n_train = args.train_batches
    n_val = args.val_batches
    B, L = args.batch_size, args.seq_len

    requested = args.models  # None means all available
    match_params = not args.no_match_params
    models_info = make_models(dim, n_layers=n_layers, requested_models=requested, match_params=match_params,
                              n_experts=args.n_experts, top_k=args.top_k, moe=args.moe)
    all_names = list(models_info.keys())
    
    if not all_names:
        print("No models available to test!")
        sys.exit(1)
    
    if requested:
        missing = set(requested) - set(all_names)
        if missing:
            print(f"Warning: Requested models not available: {missing}")
    
    print(f"\n{'Model':<12} {'Params':>10} {'Layers':>8}  Config")
    print('-' * 140)
    for name in all_names:
        m = models_info[name]
        cfg = getattr(m, '_bench_config', '<unknown>')
        print(f"{name:<12} {count_params(m):>10,} {n_layers:>8}  {cfg}")

    print(f"\nMax {max_epochs} epochs, {n_train} train batches, {n_val} val batches, B={B}, L={L}, dim={dim}, layers={n_layers}")
    print(f"Early stop at >{args.early_stop_acc:.0%} val accuracy")
    print("\nTraining config:")
    print("- Optimizer: Adam")
    print("- Loss: cross_entropy(ignore_index=-100)")
    print("- Vocab: unified VOCAB_SIZE=64 (shared embedding/head across tasks)")
    print("- LR: mixed=1e-3, standalone tasks=1e-4")
    print("- LR schedule: 1 epoch linear warmup (0.1x -> 1.0x), then cosine decay to 0.05x")
    hard_mine_epoch = int(max_epochs * 2 / 3)
    print(f"- Hard mining (mixed only): OFF until epoch {hard_mine_epoch + 1}, ON from epoch {hard_mine_epoch + 1}..{max_epochs}")
    print("  weighting: easiest~0.5x, hardest~2.0x, normalized to mean=1.0 per batch")
    print("- Mixed mode data scale: train/val batch counts multiplied by 5 (one per subtask)")
    print("- Mixed convergence criterion: all subtasks must exceed early-stop threshold")
    print('=' * 100)

    header = f"{'Model':<10} {'Task':<16} {'Init':>8} {'Best':>8} {'Val Acc':>8} {'Best@':>6} {'Epochs':>7} {'Wall(s)':>8} {'Status':>10}"
    print(header)
    print('-' * 100)

    # Collect all results for CSV and summary
    all_results = []

    task_pbar = tqdm(tasks, desc="Tasks", position=0)
    for task in task_pbar:
        task_pbar.set_description(f"Task: {task}")
        # Mixed task: 5x more batches so each subtask sees ~same samples/epoch as individual tasks
        task_n_train = n_train * len(ALL_TASKS) if task == 'mixed' else n_train
        task_n_val = n_val * len(ALL_TASKS) if task == 'mixed' else n_val
        print(f"\nPregenerating {task} data...")
        task_data = pregen_task_data(task, task_n_train, task_n_val, B, L, SEED, device=DEVICE)
        print(f"  {len(task_data['train'])} train + {len(task_data['val'])} val batches ready on {DEVICE}")
        
        model_pbar = tqdm(all_names, desc="Models", position=1, leave=False)
        for name in model_pbar:
            model_pbar.set_description(f"Model: {name}")
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            model = make_models(dim, n_layers=n_layers, requested_models=[name], match_params=match_params,
                                n_experts=args.n_experts, top_k=args.top_k, moe=args.moe)[name]
            param_count = count_params(model)
            
            # Optionally compile the model
            if args.compile:
                model = torch.compile(model, mode=args.compile_mode)
            
            task_lr = 1e-3 if task == 'mixed' else 1e-4
            r = train_task(model, task, dim, max_epochs=max_epochs, lr=task_lr, B=B, L=L, device=DEVICE,
                           preloaded_data=task_data, early_stop_acc=args.early_stop_acc,
                           grad_log_every=args.grad_log_every,
                           grad_explode=args.grad_explode,
                           grad_vanish=args.grad_vanish,
                           act_log_every=args.act_log_every,
                           act_explode=args.act_explode,
                           usb_debug_every=args.usb_debug_every)
            tqdm.write(f"{name:<10} {task:<16} {r['initial']:>8.4f} {r['final']:>8.4f} {r['acc']:>7.1%} {r['best_epoch']:>6} {r['epochs']:>7} {r['wall_s']:>8.1f} {r['stop_reason']:>10}")
            if r.get('subtask_accs'):
                parts = "  ".join(f"{t}: {a:.1%}" for t, a in r['subtask_accs'].items())
                tqdm.write(f"{'':>10} subtasks: {parts}")
            
            # Store result
            result_entry = {
                'model': name,
                'task': task,
                'params': param_count,
                'initial_loss': r['initial'],
                'final_loss': r['final'],
                'val_acc': r['acc'],
                'best_epoch': r['best_epoch'],
                'epochs': r['epochs'],
                'wall_s': r['wall_s'],
                'peak_mem_mb': r['peak_mem_mb'],
                'converged': r['converged'],
                'stop_reason': r['stop_reason'],
            }
            if r.get('subtask_accs'):
                for t, a in r['subtask_accs'].items():
                    result_entry[f'acc_{t}'] = a
            all_results.append(result_entry)
        del task_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write CSV if requested
    if args.csv:
        import csv
        fieldnames = [
            'model', 'task', 'params', 'initial_loss', 'final_loss', 
            'val_acc', 'best_epoch', 'epochs', 'wall_s', 'peak_mem_mb', 'converged', 'stop_reason'
        ] + [f'acc_{t}' for t in ALL_TASKS]
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {args.csv}")

    # Print markdown summary table
    print("\n" + "=" * 100)
    print("## Benchmark Results\n")
    
    # Group by task for clearer presentation
    for task in tasks:
        print(f"### {task}\n")
        if task == 'mixed':
            subtask_hdrs = " | ".join(f"{t}" for t in ALL_TASKS)
            print(f"| Model | Params | Val Acc | {subtask_hdrs} | Best @ | Epochs | Stop | Time |")
            sep_cols = " | ".join("------:" for _ in ALL_TASKS)
            print(f"|:------|-------:|--------:| {sep_cols} |-------:|-------:|:-----|-----:|")
        else:
            print("| Model | Params | Val Acc | Best @ | Epochs | Stop Reason | Time |")
            print("|:------|-------:|--------:|-------:|-------:|:------------|-----:|")
        
        task_results = [r for r in all_results if r['task'] == task]
        # Sort by val_acc descending
        task_results.sort(key=lambda x: x['val_acc'], reverse=True)
        
        for r in task_results:
            acc_str = f"{r['val_acc']:.1%}"
            if r['converged']:
                acc_str = f"**{acc_str}**"
            
            best_epoch = r.get('best_epoch', r['epochs'])
            stop = r.get('stop_reason', 'MAX_EPOCH')
            wall_s = r.get('wall_s', 0)
            
            if task == 'mixed':
                sub_cols = " | ".join(f"{r.get(f'acc_{t}', 0):.1%}" for t in ALL_TASKS)
                print(f"| {r['model']} | {r['params']:,} | {acc_str} | {sub_cols} | {best_epoch} | {r['epochs']} | {stop} | {wall_s:.1f}s |")
            else:
                print(f"| {r['model']} | {r['params']:,} | {acc_str} | {best_epoch} | {r['epochs']} | {stop} | {wall_s:.1f}s |")
        
        print()
    
    # Legend
    print("**Legend:**")
    print("- **Bold** = converged (>{:.0%} val acc)".format(args.early_stop_acc))
    print("- Best @ = epoch with best val accuracy")
    print("- Stop: CONVERGED (hit target) | PLATEAU (train stuck at 100%) | MAX_EPOCH")
