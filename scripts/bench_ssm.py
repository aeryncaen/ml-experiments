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
    """Causal conv3 → causal MHA."""
    def __init__(self, d_model, n_heads=4, se_reduction=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0, groups=d_model, bias=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        B, T, D = x.shape
        h = F.pad(x.transpose(1, 2), (2, 0))
        h = self.conv(h).transpose(1, 2)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
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
        """SiLU² attention: replace softmax with silu(logits)².
        q, k, v: (B, H, T, D) — standard SDPA layout.
        Returns: (B, H, T, D)
        """
        scale = 1.0 / math.sqrt(q.shape[-1])
        logits = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
        # Causal mask: zero out future positions
        T = logits.shape[-1]
        causal_mask = torch.tril(torch.ones(T, T, device=logits.device, dtype=logits.dtype))
        weights = F.silu(logits) ** 2 * causal_mask
        return weights @ v

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
            if self.attn_mode == 'silu2':
                y = self._silu2_attention(q, k, v)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(b, t, self.n_head, self.head_dim)
        else:
            q = self._hybrid_rope(q, dd_angles)
            k = self._hybrid_rope(k, dd_angles)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if self.attn_mode == 'silu2':
                y = self._silu2_attention(q, k, v)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
def gen_delay(B, L, vocab_size=32, delay=1, device='cpu'):
    """Token input, target is same sequence shifted by `delay` positions."""
    x = torch.randint(0, vocab_size, (B, L), device=device)
    target = torch.zeros_like(x)
    target[:, delay:] = x[:, :L - delay]
    return x, target


def gen_selective_copy(B, L, vocab_size=32, n_markers=4, device='cpu'):
    """Token input with marker channel. Target is the n_markers marked tokens.
    Input: (B, L) tokens + (B, L) marker flags → fed as 2-token tuples via embedding.
    Output: (B, n_markers) token predictions at end of sequence."""
    tokens = torch.randint(0, vocab_size, (B, L), device=device)
    markers = torch.zeros(B, L, dtype=torch.long, device=device)
    target = torch.zeros(B, n_markers, dtype=torch.long, device=device)
    for b in range(B):
        idxs = torch.randperm(L, device=device)[:n_markers].sort().values
        markers[b, idxs] = 1
        target[b] = tokens[b, idxs]
    return tokens, markers, target


def gen_parity(B, L, device='cpu'):
    """Binary input {0,1}, target is running XOR (cumulative parity)."""
    x = torch.randint(0, 2, (B, L), device=device)
    target = x.cumsum(dim=1) % 2  # running parity
    return x, target


def gen_mod_arith(B, L, mod_base=5, device='cpu'):
    """Digit input 0..mod_base-1, target is running sum mod base."""
    x = torch.randint(0, mod_base, (B, L), device=device)
    target = x.cumsum(dim=1) % mod_base
    return x, target


def gen_induction(B, L, vocab_size=32, device='cpu'):
    half = L // 2
    pattern = torch.randint(0, vocab_size, (B, half), device=device)
    seq = torch.cat([pattern, pattern], dim=1)  # (B, L)
    input_seq = seq[:, :-1]
    target_seq = seq[:, 1:]
    return input_seq, target_seq


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
        
        def gen_batch(task_name):
            if task_name == 'delay':
                return gen_delay(B, L, vocab_size=32, device='cpu')
            elif task_name == 'selective_copy':
                return gen_selective_copy(B, L, vocab_size=32, device='cpu')
            elif task_name == 'parity':
                return gen_parity(B, L, device='cpu')
            elif task_name == 'mod_arith':
                return gen_mod_arith(B, L, device='cpu')
            elif task_name == 'induction':
                return gen_induction(B, L, vocab_size=32, device='cpu')
        
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


def train_task(model, task_name, dim, max_epochs=100, lr=1e-3, B=32, L=32, device='cpu',
               preloaded_data=None, early_stop_acc=0.99, grad_log_every=50,
               grad_explode=1e3, grad_vanish=1e-6, act_log_every=50,
               act_explode=1e3, usb_debug_every=0):
    """Train with epochs and early stopping when val accuracy exceeds threshold."""
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    tracked_named_params = list(model.named_parameters())
    embed = None
    marker_embed = None
    head = None
    vocab_size = None

    if usb_debug_every:
        _set_usb_debug(model, True)

    act_stats_step = {}
    act_active = {'on': False}
    act_hooks = []

    if task_name == 'delay':
        vocab_size = 32
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
        tracked_named_params += [(f"embed.{n}", p) for n, p in embed.named_parameters()]
        tracked_named_params += [(f"head.{n}", p) for n, p in head.named_parameters()]
    elif task_name == 'selective_copy':
        vocab_size = 32
        embed = nn.Embedding(vocab_size, dim).to(device)
        marker_embed = nn.Embedding(2, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters())
                         + list(marker_embed.parameters()) + list(head.parameters()), lr=lr)
        tracked_named_params += [(f"embed.{n}", p) for n, p in embed.named_parameters()]
        tracked_named_params += [(f"marker_embed.{n}", p) for n, p in marker_embed.named_parameters()]
        tracked_named_params += [(f"head.{n}", p) for n, p in head.named_parameters()]
    elif task_name == 'parity':
        vocab_size = 2
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
        tracked_named_params += [(f"embed.{n}", p) for n, p in embed.named_parameters()]
        tracked_named_params += [(f"head.{n}", p) for n, p in head.named_parameters()]
    elif task_name == 'mod_arith':
        vocab_size = 5
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
        tracked_named_params += [(f"embed.{n}", p) for n, p in embed.named_parameters()]
        tracked_named_params += [(f"head.{n}", p) for n, p in head.named_parameters()]
    elif task_name == 'induction':
        vocab_size = 32
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

    def _forward(batch):
        if task_name == 'delay':
            assert embed is not None and head is not None
            assert vocab_size is not None
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'selective_copy':
            assert embed is not None and marker_embed is not None and head is not None
            assert vocab_size is not None
            tokens, markers, tgt = batch
            x = embed(tokens) + marker_embed(markers)
            y = head(model(x))
            n_markers = tgt.shape[1]
            y_last = y[:, -n_markers:]
            loss = F.cross_entropy(y_last.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y_last.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'parity':
            assert embed is not None and head is not None
            assert vocab_size is not None
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'mod_arith':
            assert embed is not None and head is not None
            assert vocab_size is not None
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        else:  # induction
            assert embed is not None and head is not None
            assert vocab_size is not None
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc

    def _eval_val():
        model.eval()
        val_accs = []
        val_losses = []
        with torch.no_grad():
            for batch in val_data:
                loss, acc = _forward(batch)
                val_losses.append(loss.item())
                val_accs.append(acc)
        model.train()
        return sum(val_losses) / len(val_losses), sum(val_accs) / len(val_accs)

    mem_baseline = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_baseline = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    
    n_train = len(train_data)
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
    final_epoch = 0
    stop_reason = "MAX_EPOCH"
    
    epoch_pbar = tqdm(range(max_epochs), desc="Epochs", leave=False, ncols=100)
    for epoch in epoch_pbar:
        # Shuffle train indices each epoch
        train_indices = torch.randperm(n_train).tolist()
        
        epoch_losses = []
        epoch_accs = []
        
        step_pbar = tqdm(train_indices, desc=f"Epoch {epoch+1}", leave=False, ncols=80)
        for idx in step_pbar:
            batch = train_data[idx]
            track_usb = usb_debug_every and (total_steps % usb_debug_every == 0)
            if track_usb:
                _set_usb_debug_active(model, True)
            track_act = act_log_every and (total_steps % act_log_every == 0)
            if track_act:
                act_stats_step.clear()
                act_active['on'] = True
            loss, acc = _forward(batch)
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
        val_loss, val_acc = _eval_val()
        final_epoch = epoch + 1
        
        # Track best
        if val_acc > best_val_acc:
            best_epoch = final_epoch
            best_val_acc = val_acc
            best_val_loss = val_loss
        
        epoch_pbar.set_postfix(val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.1%}")
        
        # Early stopping: converged
        if val_acc >= early_stop_acc:
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
        'converged': best_val_acc >= early_stop_acc,
        'stop_reason': stop_reason,
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


def _stack(make_layer, n_layers, dim):
    """Always wrap with pre-norm residual stacking."""
    return StackedModel(make_layer, n_layers, dim)


def make_models(dim, n_layers=1, requested_models=None):
    """Build models with configurable depth. Param counts are per-layer (~50-57K).
    
    If requested_models is provided, only build those models (skips unavailable ones with warning).
    """
    models = {}
    
    def try_add(name, make_fn):
        if requested_models is not None and name not in requested_models:
            return
        try:
            models[name] = _stack(make_fn, n_layers, dim)
        except ImportError as e:
            print(f"  Warning: {name} not available ({e})")
        except Exception as e:
            print(f"  Warning: {name} failed to initialize ({e})")

    # DS1++: DS1 + signed sparse attention
    try_add('DS1++', lambda: DS1Wrapper(dim=dim, state_dim=48, mimo_rank=4, n_iters=2, diff_attn=True))

    # S4D: ~27K params/layer (d_state=144 → 27,008 params)
    try_add('S4D', lambda: S4D(d_model=dim, d_state=144))

    # S5: ~27K params/layer (state_width=140 → 27,352 params)
    try_add('S5', lambda: S5Wrapper(width=dim, state_width=140))

    # Mamba: ~27K params/layer (expand=1, d_state=72 → 27,200 params)
    try_add('Mamba', lambda: MambaWrapper(d_model=dim, d_state=72, d_conv=4, expand=1))

    # USB: Full MHA with directional scans (elementwise state - good for state tracking)
    try_add('USB', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2,
                                       scan_state_modes=('elementwise', 'elementwise', 'elementwise')))
    
    # USB_outer: USB with outer product state (all groups - good for retrieval)
    try_add('USB_outer', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2,
                                             scan_state_modes=('outer', 'outer', 'outer')))
    
    # USB_hybrid: G1/G2 outer (retrieval), G3 elementwise (state tracking)
    try_add('USB_hybrid', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2,
                                              scan_state_modes=('outer', 'outer', 'elementwise')))

    # MHA: ~19K params/layer (QuadConv + MHA only, no MLP/Mamba to scale)
    try_add('MHA', lambda: MHABlock(d_model=dim, n_heads=4))

    # Ideal: orthogonal projection attention (no softmax, Cayley-parameterized heads)
    try_add('Ideal', lambda: IdealWrapper(d_model=dim, n_heads=4, ffn_mult=4.0))

    # FusedGate: fused attention+MLP block (up → swish → attn → skip-mul → swish → down)
    try_add('FusedGate', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=False))

    # FusedGatePaired: same but with paired head attention
    try_add('FusedGatePaired', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=True))

    # SiLU² attention variants
    try_add('FusedGateSilu2', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=False, attn_mode='silu2'))
    try_add('FusedGateSilu2P', lambda: FusedGateBlock(d_model=dim, n_heads=4, paired=True, attn_mode='silu2'))

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
    models_info = make_models(dim, n_layers=n_layers, requested_models=requested)
    all_names = list(models_info.keys())
    
    if not all_names:
        print("No models available to test!")
        sys.exit(1)
    
    if requested:
        missing = set(requested) - set(all_names)
        if missing:
            print(f"Warning: Requested models not available: {missing}")
    
    print(f"\n{'Model':<10} {'Params':>10} {'Layers':>8}")
    print('-' * 30)
    for name in all_names:
        m = models_info[name]
        print(f"{name:<10} {count_params(m):>10,} {n_layers:>8}")

    print(f"\nMax {max_epochs} epochs, {n_train} train batches, {n_val} val batches, B={B}, L={L}, dim={dim}, layers={n_layers}")
    print(f"Early stop at >{args.early_stop_acc:.0%} val accuracy")
    print('=' * 100)

    header = f"{'Model':<10} {'Task':<16} {'Init':>8} {'Best':>8} {'Val Acc':>8} {'Best@':>6} {'Epochs':>7} {'Wall(s)':>8} {'Status':>10}"
    print(header)
    print('-' * 100)

    # Collect all results for CSV and summary
    all_results = []

    task_pbar = tqdm(tasks, desc="Tasks", position=0)
    for task in task_pbar:
        task_pbar.set_description(f"Task: {task}")
        print(f"\nPregenerating {task} data...")
        task_data = pregen_task_data(task, n_train, n_val, B, L, SEED, device=DEVICE)
        print(f"  {len(task_data['train'])} train + {len(task_data['val'])} val batches ready on {DEVICE}")
        
        model_pbar = tqdm(all_names, desc="Models", position=1, leave=False)
        for name in model_pbar:
            model_pbar.set_description(f"Model: {name}")
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            model = make_models(dim, n_layers=n_layers, requested_models=[name])[name]
            param_count = count_params(model)
            
            # Optionally compile the model
            if args.compile:
                model = torch.compile(model, mode=args.compile_mode)
            
            r = train_task(model, task, dim, max_epochs=max_epochs, lr=1e-3, B=B, L=L, device=DEVICE,
                           preloaded_data=task_data, early_stop_acc=args.early_stop_acc,
                           grad_log_every=args.grad_log_every,
                           grad_explode=args.grad_explode,
                           grad_vanish=args.grad_vanish,
                           act_log_every=args.act_log_every,
                           act_explode=args.act_explode,
                           usb_debug_every=args.usb_debug_every)
            tqdm.write(f"{name:<10} {task:<16} {r['initial']:>8.4f} {r['final']:>8.4f} {r['acc']:>7.1%} {r['best_epoch']:>6} {r['epochs']:>7} {r['wall_s']:>8.1f} {r['stop_reason']:>10}")
            
            # Store result
            all_results.append({
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
            })
        del task_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write CSV if requested
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'model', 'task', 'params', 'initial_loss', 'final_loss', 
                'val_acc', 'best_epoch', 'epochs', 'wall_s', 'peak_mem_mb', 'converged', 'stop_reason'
            ])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {args.csv}")

    # Print markdown summary table
    print("\n" + "=" * 100)
    print("## Benchmark Results\n")
    
    # Group by task for clearer presentation
    for task in tasks:
        print(f"### {task}\n")
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
            
            print(f"| {r['model']} | {r['params']:,} | {acc_str} | {best_epoch} | {r['epochs']} | {stop} | {wall_s:.1f}s |")
        
        print()
    
    # Legend
    print("**Legend:**")
    print("- **Bold** = converged (>{:.0%} val acc)".format(args.early_stop_acc))
    print("- Best @ = epoch with best val accuracy")
    print("- Stop: CONVERGED (hit target) | PLATEAU (train stuck at 100%) | MAX_EPOCH")
