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
from ulb import ULBBlock, ULBConfig, StackedULB, MoEStackedULB, PoolOfExperts
try:
    from lloom import LLooM, LLooMConfig
    HAS_LLOOM = True
except ImportError:
    HAS_LLOOM = False


class LLooMBenchWrapper(nn.Module):
    """Wraps LLooM for the benchmark harness.

    LLooM is self-contained (own stems, norms, routing loops) so it cannot
    use _stack() / _stack_moe() / _stack_poe().  This wrapper adapts it to
    the harness interface: (B, L, D) -> (B, L, D).

    Exposes:
        router_noise_scale: Mutable float — the harness anneals this linearly
            to 0 over training.  Passed into LLooM.forward() each call.
        last_mean_hops: Set after each forward — harness reads this for eval
            stats display.
        last_info: Full routing info dict from the last forward call.
    """

    def __init__(self, **kwargs):
        super().__init__()
        if not HAS_LLOOM:
            raise ImportError("LLooM not available")
        self.lloom = LLooM(LLooMConfig(**kwargs))
        self.aux_loss = 0.0
        self.router_noise_scale = self.lloom.config.router_noise
        self.last_mean_hops: float | None = None
        self.last_info: dict | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, info = self.lloom(x, noise_scale=self.router_noise_scale)
        self.aux_loss = 0.0  # placeholder for future routing loss
        # Materialize tensor values to Python floats — info dict comes from
        # LLooM.forward() which stores raw tensors (no .item()) so that the
        # compiled graph has no graph-breaking scalar extractions.
        self.last_info = {
            k: (v.item() if isinstance(v, torch.Tensor) and v.ndim == 0 else v)
            for k, v in info.items()
        }
        self.last_mean_hops = self.last_info.get('mean_global_hops')
        return out


class DualMHABenchWrapper(nn.Module):
    """Two independent MHA stacks whose outputs are subtracted.

    bench_ssm version: operates on (B, L, D) hidden states (no embed/head).
    Each half is a full pre-norm residual stack with its own blocks and norms.
    Output = stack_a(x) - stack_b(x).
    """

    def __init__(self, d_model, n_heads=4, n_layers=1, mlp_inner=0):
        super().__init__()
        self.stack_a = StackedModel(
            lambda: MHABlock(d_model=d_model, n_heads=n_heads, mlp_inner=mlp_inner),
            n_layers, d_model)
        self.stack_b = StackedModel(
            lambda: MHABlock(d_model=d_model, n_heads=n_heads, mlp_inner=mlp_inner),
            n_layers, d_model)

    def forward(self, x):
        return self.stack_a(x) - self.stack_b(x)


class DiffMHABlock(nn.Module):
    """Two independent MHA blocks whose outputs are subtracted.

    Single-layer version for bench_ssm stacking. _stack() wraps with
    pre-norm + residual, so effective update is:
        x = x + (block_a(norm(x)) - block_b(norm(x)))
    """

    def __init__(self, d_model, n_heads=4, mlp_inner=0):
        super().__init__()
        self.block_a = MHABlock(d_model=d_model, n_heads=n_heads, mlp_inner=mlp_inner)
        self.block_b = MHABlock(d_model=d_model, n_heads=n_heads, mlp_inner=mlp_inner)

    def forward(self, x):
        return self.block_a(x) - self.block_b(x)


class MulMHABlock(nn.Module):
    """Two independent MHA blocks whose outputs are multiplied (element-wise).

    Single-layer version for bench_ssm stacking. _stack() wraps with
    pre-norm + residual, so effective update is:
        x = x + (block_a(norm(x)) * block_b(norm(x)))
    """

    def __init__(self, d_model, n_heads=4, mlp_inner=0):
        super().__init__()
        self.block_a = MHABlock(d_model=d_model, n_heads=n_heads, mlp_inner=mlp_inner)
        self.block_b = MHABlock(d_model=d_model, n_heads=n_heads, mlp_inner=mlp_inner)

    def forward(self, x):
        return self.block_a(x) * self.block_b(x)


class OuterMHABlock(nn.Module):
    """Two independent MHA blocks combined via outer product.

    Computes a(x) * (b(x) @ proj) — one stream gates the other through
    a learned linear transform.  proj initialized to identity.
    """

    def __init__(self, d_model, n_heads=4, mlp_inner=0):
        super().__init__()
        self.block_a = MHABlock(d_model=d_model, n_heads=n_heads, mlp_inner=mlp_inner)
        self.block_b = MHABlock(d_model=d_model, n_heads=n_heads, mlp_inner=mlp_inner)
        self.proj = nn.Parameter(torch.eye(d_model))

    def forward(self, x):
        a = self.block_a(x)
        b = self.block_b(x)
        return a * (b @ self.proj)


class StupidAttnBlock(nn.Module):
    """Pairwise gated value merge with no Q/K/V projections.

    For each pair (i, j), compute element-wise product h[i] * h[j],
    project that product through G to get a sigmoid gate, and use that
    gate to merge answering values h[j] into querying positions i.

    Features are grouped into heads. Each head has its own projection G
    from head_dim -> 1, producing one gate per pair per head.
    """

    def __init__(self, d_model, n_heads=64, causal=True):
        super().__init__()
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
                              groups=d_model, bias=True)
        # Per-head projection G: head_dim -> 1 (for pairwise gate logits)
        self.g_proj = nn.Parameter(torch.randn(n_heads, self.head_dim) * (self.head_dim ** -0.5))
        self.g_bias = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim

        # Causal conv
        h = F.pad(x.transpose(1, 2), (2, 0))
        h = self.conv(h).transpose(1, 2)  # (B, T, D)

        h_heads = h.view(B, T, H, hd)  # (B, T, H, hd)

        # Pairwise element-wise products: (i,j) -> h[i] * h[j]
        pair = h_heads.unsqueeze(2) * h_heads.unsqueeze(1)  # (B, T_i, T_j, H, hd)

        # Project pairwise product through G, then sigmoid gate
        logits = (pair * self.g_proj.view(1, 1, 1, H, hd)).sum(dim=-1)
        logits = logits + self.g_bias.view(1, 1, 1, H)  # (B, T_i, T_j, H)
        weights = torch.sigmoid(logits)

        # Causal mask + normalize over answering positions j
        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
            weights = weights * mask.unsqueeze(0).unsqueeze(-1)

        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)

        # Merge answering values h[j] into querying position i
        out = (weights.unsqueeze(-1) * h_heads.unsqueeze(1)).sum(dim=2)  # (B, T, H, hd)
        out = out.reshape(B, T, D)

        return out

    
class FeatAttnBlock(nn.Module):
    """Causal MHA + feature-attention (no SwiGLU MLP).

    Feature-attention replaces the MLP sublayer: expands each token into
    feature groups, runs non-causal self-attention over them, skip-multiplies,
    and projects back down.  This is dynamic content-dependent feature mixing.
    """

    def __init__(self, d_model, n_heads=4, feat_expansion=4,
                 feat_n_heads=1, feat_first=False, transpose_groups=False,
                 up_factor=1):
        super().__init__()
        self.feat_first = feat_first
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
                              groups=d_model, bias=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.feat_norm = RMSNorm(d_model)
        from mha import FeatureAttn
        self.feat_attn = FeatureAttn(d_model, feat_expansion=feat_expansion,
                                     n_heads=feat_n_heads,
                                     transpose_groups=transpose_groups,
                                     up_factor=up_factor)

    def forward(self, x):
        B, T, D = x.shape
        # Causal conv
        h = F.pad(x.transpose(1, 2), (2, 0))
        h = self.conv(h).transpose(1, 2)

        if self.feat_first:
            # Feature attention first
            h = h + self.feat_attn(self.feat_norm(h))
            # Then sequence attention
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            attn_out, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
            h = attn_out
        else:
            # Sequence attention first
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
            h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
            # Then feature attention
            h = h + self.feat_attn(self.feat_norm(h))
        return h


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


class MHAFeatMLPBlock(nn.Module):
    """Causal MHA + feat-attn + SwiGLU MLP, with switchable feat-attn position.

    Tests whether feat-attn can make MLP useful for algorithmic tasks.
    feat_position='before_mlp': MHA → feat-attn → MLP
    feat_position='after_mlp':  MHA → MLP → feat-attn
    """

    def __init__(self, d_model, n_heads=4, mlp_inner=0, feat_position='before_mlp'):
        super().__init__()
        assert feat_position in ('before_mlp', 'after_mlp')
        self.feat_position = feat_position
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
                              groups=d_model, bias=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Feat-attn (2D tensor projections, same as ULB2DBlock's feat-attn)
        C = int(d_model ** 0.5)
        assert C * C == d_model, f"d_model ({d_model}) must be a perfect square"
        self.C = C
        init_scale = (C * C) ** -0.5
        self.feat_wq = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wk = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wv = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wo = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_norm = RMSNorm(d_model)

        # SwiGLU MLP
        self.has_mlp = mlp_inner > 0
        if self.has_mlp:
            self.mlp_norm = RMSNorm(d_model)
            self.gate_proj = nn.Linear(d_model, mlp_inner, bias=False)
            self.up_proj = nn.Linear(d_model, mlp_inner, bias=False)
            self.down_proj = nn.Linear(mlp_inner, d_model, bias=False)

    def _feat_attn(self, h):
        B, T, D = h.shape
        C = self.C
        x2d = self.feat_norm(h).view(B, T, C, C)
        fQ = torch.einsum('...cd,cdef->...ef', x2d, self.feat_wq)
        fK = torch.einsum('...cd,cdef->...ef', x2d, self.feat_wk)
        fV = torch.einsum('...cd,cdef->...ef', x2d, self.feat_wv)
        fQ = fQ.reshape(B * T, 1, C, C)
        fK = fK.reshape(B * T, 1, C, C)
        fV = fV.reshape(B * T, 1, C, C)
        y = F.scaled_dot_product_attention(fQ, fK, fV, is_causal=False)
        y = y.view(B, T, C, C)
        y = torch.einsum('...cd,cdef->...ef', y, self.feat_wo)
        return y.reshape(B, T, D)

    def _mlp(self, h):
        r = self.mlp_norm(h)
        return self.down_proj(F.silu(self.gate_proj(r)) * self.up_proj(r))

    def forward(self, x):
        B, T, D = x.shape
        h = F.pad(x.transpose(1, 2), (2, 0))
        h = self.conv(h).transpose(1, 2)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)

        if self.feat_position == 'before_mlp':
            h = h + self._feat_attn(h)
            if self.has_mlp:
                h = h + self._mlp(h)
        else:
            if self.has_mlp:
                h = h + self._mlp(h)
            h = h + self._feat_attn(h)
        return h


class MHA2DBlock(nn.Module):
    """Causal MHA with 2D token representations.

    Each token is a (C, D') matrix instead of a D-dim vector.
    The block receives (B, T, D) where D = C * D', reshapes to (B, T, C, D').

    QKV projections are 2D: W_q, W_k, W_v are each (C, D', C, D') tensors
    that map (C, D') -> (C, D') matrices.

    Attention score between token i and j: sum over elements of Q_i * K_j
    (Frobenius inner product), scaled. Output is weighted sum of V matrices.

    Multi-head: split D' into heads. Each head gets (C, head_dim).
    Score per head: Frobenius inner product of (C, head_dim) matrices.
    """

    def __init__(self, d_model, n_channels=4, n_heads=4, mlp_inner=0):
        super().__init__()
        self.d_model = d_model
        self.C = n_channels
        self.D_prime = d_model // n_channels
        self.n_heads = n_heads
        assert d_model % n_channels == 0
        assert self.D_prime % n_heads == 0
        self.head_dim = self.D_prime // n_heads

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
                              groups=d_model, bias=True)

        # 2D QKV projections: (C, D', C, D') — map (C, D') matrix to (C, D') matrix
        # Implemented as (C*D', C*D') linear but conceptually 2D-to-2D
        self.w_q = nn.Parameter(torch.randn(self.C, self.D_prime, self.C, self.D_prime)
                                * (self.C * self.D_prime) ** -0.5)
        self.w_k = nn.Parameter(torch.randn(self.C, self.D_prime, self.C, self.D_prime)
                                * (self.C * self.D_prime) ** -0.5)
        self.w_v = nn.Parameter(torch.randn(self.C, self.D_prime, self.C, self.D_prime)
                                * (self.C * self.D_prime) ** -0.5)

        # Output projection: also 2D
        self.w_o = nn.Parameter(torch.randn(self.C, self.D_prime, self.C, self.D_prime)
                                * (self.C * self.D_prime) ** -0.5)

        self.scale = (self.C * self.head_dim) ** -0.5

        self.has_mlp = mlp_inner > 0
        if self.has_mlp:
            self.mlp_norm = RMSNorm(d_model)
            self.gate_proj = nn.Linear(d_model, mlp_inner, bias=False)
            self.up_proj = nn.Linear(d_model, mlp_inner, bias=False)
            self.down_proj = nn.Linear(mlp_inner, d_model, bias=False)

    def _proj_2d(self, x, w):
        """Apply 2D projection: x is (B, T, C, D'), w is (C, D', C, D')."""
        return torch.einsum('btcd,cdef->btef', x, w)

    def forward(self, x):
        B, T, D = x.shape
        # Causal conv (on flat D)
        h = F.pad(x.transpose(1, 2), (2, 0))
        h = self.conv(h).transpose(1, 2)                               # (B, T, D)

        # Reshape to 2D: (B, T, C, D')
        h2d = h.view(B, T, self.C, self.D_prime)

        # 2D QKV projections
        Q = self._proj_2d(h2d, self.w_q)                               # (B, T, C, D')
        K = self._proj_2d(h2d, self.w_k)
        V = self._proj_2d(h2d, self.w_v)

        # Multi-head: split D' into heads -> (B, T, C, n_heads, head_dim)
        Q = Q.view(B, T, self.C, self.n_heads, self.head_dim)
        K = K.view(B, T, self.C, self.n_heads, self.head_dim)
        V = V.view(B, T, self.C, self.n_heads, self.head_dim)

        # Attention scores: Frobenius inner product per head over (C, head_dim)
        # Q: (B, T, C, H, hd) -> (B, H, T, C*hd) for score computation
        Q_flat = Q.permute(0, 3, 1, 2, 4).reshape(B, self.n_heads, T, self.C * self.head_dim)
        K_flat = K.permute(0, 3, 1, 2, 4).reshape(B, self.n_heads, T, self.C * self.head_dim)
        V_flat = V.permute(0, 3, 1, 2, 4).reshape(B, self.n_heads, T, self.C * self.head_dim)

        # Standard scaled dot-product attention (causal)
        attn_out = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat, is_causal=True)
        # (B, H, T, C*hd)

        # Reshape back to 2D: (B, T, C, D')
        attn_out = attn_out.reshape(B, self.n_heads, T, self.C, self.head_dim)
        attn_out = attn_out.permute(0, 2, 3, 1, 4).reshape(B, T, self.C, self.D_prime)

        # 2D output projection
        attn_out = self._proj_2d(attn_out, self.w_o)                   # (B, T, C, D')

        # Flatten back to (B, T, D)
        h = attn_out.reshape(B, T, D)

        if self.has_mlp:
            r = self.mlp_norm(h)
            h = h + self.down_proj(F.silu(self.gate_proj(r)) * self.up_proj(r))
        return h


class MHA2DSimpleBlock(nn.Module):
    """Seq-attn per channel + feat-attn per position on sqrt(D) x sqrt(D) tokens.

    Reshape (B, T, D) to (B, T, C, C) where C = sqrt(D).
    1. Seq-attn: C independent causal attention streams over T positions.
       QKV are full 2D projections (C,C,C,C) mapping (C,C) -> (C,C).
    2. Feat-attn: at each position, non-causal attention over C channels.
       QKV are full 2D projections (C,C,C,C).
    No expansions, no bottlenecks, no MLP.
    """

    def __init__(self, d_model, n_channels=None):
        super().__init__()
        self.d_model = d_model
        self.C = n_channels or int(d_model ** 0.5)
        assert self.C * self.C == d_model, f"d_model ({d_model}) must be a perfect square"
        C = self.C

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0,
                              groups=d_model, bias=True)

        # Seq-attn: full 2D QKV projections (C,C) -> (C,C)
        init_scale = (C * C) ** -0.5
        self.seq_wq = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.seq_wk = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.seq_wv = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.seq_wo = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.seq_norm = RMSNorm(d_model)

        # Feat-attn: full 2D QKV projections (C,C) -> (C,C)
        self.feat_wq = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wk = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wv = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wo = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_norm = RMSNorm(d_model)

    def _proj_2d(self, x, w):
        """x: (..., C, C), w: (C, C, C, C) -> (..., C, C)"""
        return torch.einsum('...cd,cdef->...ef', x, w)

    def forward(self, x):
        B, T, D = x.shape
        C = self.C

        # Causal conv
        h = F.pad(x.transpose(1, 2), (2, 0))
        h = self.conv(h).transpose(1, 2)                               # (B, T, D)

        # --- Seq-attn: per-channel attention over T positions ---
        h2d = self.seq_norm(h).view(B, T, C, C)                       # (B, T, C, C)
        Q = self._proj_2d(h2d, self.seq_wq)                           # (B, T, C, C)
        K = self._proj_2d(h2d, self.seq_wk)
        V = self._proj_2d(h2d, self.seq_wv)
        # Each channel is a head: reshape to (B, C, T, C) for SDPA
        Q = Q.permute(0, 2, 1, 3)                                     # (B, C, T, C)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)    # (B, C, T, C)
        y = y.permute(0, 2, 1, 3).contiguous()                        # (B, T, C, C)
        y = self._proj_2d(y, self.seq_wo)
        h = h + y.reshape(B, T, D)                                    # residual

        # --- Feat-attn: per-position attention over C channels ---
        h2d = self.feat_norm(h).view(B, T, C, C)                      # (B, T, C, C)
        Q = self._proj_2d(h2d, self.feat_wq)
        K = self._proj_2d(h2d, self.feat_wk)
        V = self._proj_2d(h2d, self.feat_wv)
        # Each position independently: (B*T, 1, C, C) for single-head SDPA over C
        Q = Q.reshape(B * T, 1, C, C)
        K = K.reshape(B * T, 1, C, C)
        V = V.reshape(B * T, 1, C, C)
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=False)   # (B*T, 1, C, C)
        y = y.view(B, T, C, C)
        y = self._proj_2d(y, self.feat_wo)
        h = h + y.reshape(B, T, D)                                    # residual

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
               preloaded_data=None, early_stop_acc=0.98, grad_log_every=50,
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
        # Collect aux_loss from ReLU routing (0.0 if topk mode)
        aux_loss = getattr(model, 'aux_loss', 0.0)

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
            loss = (per_sample_loss * weights).mean() + aux_loss
        else:
            logits_flat = y.reshape(-1, vocab_size)
            tgt_flat = tgt.reshape(-1)
            loss = F.cross_entropy(logits_flat, tgt_flat, ignore_index=-100) + aux_loss
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
        val_hops = []
        val_routing_stats = []  # list of dicts from model.last_info (LLooM etc.)
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
                _h = getattr(model, 'last_mean_hops', None)
                if _h is not None:
                    val_hops.append(_h.item() if hasattr(_h, 'item') else _h)
                _info = getattr(model, 'last_info', None)
                if _info is not None:
                    val_routing_stats.append(_info)
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
        subtask_accs = None
        if is_mixed:
            subtask_accs = {}
            for t in ALL_TASKS:
                subtask_accs[t] = per_task_correct[t] / per_task_total[t] if per_task_total[t] > 0 else 0.0
            avg_acc = sum(subtask_accs.values()) / len(subtask_accs)
        else:
            avg_acc = total_correct / total_valid if total_valid > 0 else 0.0
        mean_hops = sum(val_hops) / len(val_hops) if val_hops else None
        # Average routing stats across val batches
        routing_summary = None
        if val_routing_stats:
            routing_summary = {}
            _stat_keys = ('mean_seq_hops', 'mean_tok_hops', 'mean_global_hops', 'mean_bridges',
                          'stem_go_seq', 'stem_go_tok', 'stem_go_exit')
            for k in _stat_keys:
                vals = [d[k] for d in val_routing_stats if k in d]
                if vals:
                    routing_summary[k] = sum(vals) / len(vals)
        return avg_loss, avg_acc, subtask_accs, mean_hops, routing_summary

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
    
    use_hard_mine = False  # activated when majority of subtasks converge
    
    # PoolOfExperts annealing
    _initial_router_noise = getattr(model, 'router_noise_scale', None)


    epoch_pbar = tqdm(range(max_epochs), desc="Epochs", leave=False)
    for epoch in epoch_pbar:
        
        # Anneal router noise
        if _initial_router_noise is not None:
            frac = epoch / max_epochs
            model.router_noise_scale = _initial_router_noise * (1.0 - frac)

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

            step_loss = loss.item()
            epoch_losses.append(step_loss)
            epoch_accs.append(acc)
            total_steps += 1
            
            step_pbar.set_postfix(loss=f"{step_loss:.4f}", acc=f"{acc:.1%}")
        
        # Record initial loss from first epoch
        if initial_loss is None:
            initial_loss = sum(epoch_losses[:10]) / min(10, len(epoch_losses))
        
        # Check for train accuracy plateau (memorization)
        train_acc = sum(epoch_accs) / len(epoch_accs)
        
        # Evaluate on validation set
        val_loss, val_acc, subtask_accs, val_mean_hops, routing_summary = _eval_val()
        final_epoch = epoch + 1
        
        # Track best
        if val_acc >= best_val_acc:
            best_epoch = final_epoch
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_subtask_accs = subtask_accs
        
        _hops_str = ""
        if routing_summary is not None:
            # Detailed routing stats (LLooM): seq/tok hops + bridges
            sh = routing_summary.get('mean_seq_hops', 0)
            th = routing_summary.get('mean_tok_hops', 0)
            gh = routing_summary.get('mean_global_hops', 0)
            br = routing_summary.get('mean_bridges', 0)
            _hops_str = f" hops={gh:.1f}(s{sh:.1f}+t{th:.1f}) br={br:.2f}"
        elif val_mean_hops is not None:
            _hops_str = f" hops={val_mean_hops:.1f}"
        if subtask_accs is not None:
            _short = {'delay': 'D', 'selective_copy': 'S', 'induction': 'I', 'parity': 'P', 'mod_arith': 'M'}
            parts = " ".join(f"{_short[t]}={a:.1%}" for t, a in subtask_accs.items())
            epoch_pbar.set_postfix_str(f"L={val_loss:.3f} A={val_acc:.1%} {parts}{_hops_str}")
        else:
            epoch_pbar.set_postfix_str(f"L={val_loss:.4f} A={val_acc:.1%}{_hops_str}")
        
        # Hard mining: on when majority of subtasks have individually converged, off otherwise
        prev_hard_mine = use_hard_mine
        if subtask_accs is not None:
            n_converged = sum(1 for a in subtask_accs.values() if round(a, 3) >= early_stop_acc)
            use_hard_mine = n_converged > len(subtask_accs) / 2
            if use_hard_mine != prev_hard_mine:
                state = "ON" if use_hard_mine else "OFF"
                tqdm.write(f"[hard mining] {state} at epoch {epoch + 1}/{max_epochs} "
                           f"({n_converged}/{len(subtask_accs)} subtasks converged)")

        # Early stopping: converged
        # Mixed task: all subtasks must exceed threshold
        if subtask_accs is not None:
            converged = all(round(a, 3) >= early_stop_acc for a in subtask_accs.values())
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
        'converged': (all(round(a, 3) >= early_stop_acc for a in best_subtask_accs.values())
                      if best_subtask_accs is not None
                      else best_val_acc >= early_stop_acc),
        'stop_reason': stop_reason,
        'subtask_accs': best_subtask_accs,
        'embed_state_dict': embed.state_dict(),
        'head_state_dict': head.state_dict(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


StackedModel = StackedULB  # backward compat alias
MoEStackedModel = MoEStackedULB  # backward compat alias


def _stack(make_layer, n_layers, dim):
    """Always wrap with pre-norm residual stacking."""
    return StackedModel(make_layer, n_layers, dim)


def _stack_moe(make_layer, n_layers, dim, n_experts=4, top_k=2, version=1):
    """Wrap with MoE stacking."""
    return MoEStackedModel(make_layer, n_layers, dim, n_experts=n_experts, top_k=top_k, version=version)


def _stack_poe(make_layer, dim, pool_size=8, top_k=2, max_hops=None,
               block_shared_fraction=0.0, router_shared_fraction=0.0,
               hop_shared_fraction=0.0, router_noise=1.0):
    """Wrap with PoolOfExperts (dynamic-depth expert pool)."""
    return PoolOfExperts(make_layer, pool_size=pool_size, dim=dim, top_k=top_k, max_hops=max_hops,
                         router_noise=router_noise,
                         block_shared_fraction=block_shared_fraction,
                         router_shared_fraction=router_shared_fraction,
                         hop_shared_fraction=hop_shared_fraction)


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


def make_models(dim, n_layers=1, requested_models=None, match_params=True, n_experts=4, top_k=2, moe=False,
                poe=False, poe_max_hops=None,
                poe_block_share_fraction=0.0, poe_router_share_fraction=0.0,
                poe_hop_share_fraction=0.0, router_noise=1.0,
                feat_transpose=False, feat_up_factor=2):
    """Build models with configurable depth.
    
    If match_params=True, auto-size SSM/MHA internal dimensions to match ULBBlendP param count.
    If requested_models is provided, only build those models (skips unavailable ones with warning).
    """
    models = {}
    
    # Compute reference param count from ULBBlendP
    target_params = None
    if match_params:
        try:
            target_params = _count_stacked_params(
                lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='blend')),
                n_layers, dim
            )
            print(f"  Auto-sizing to match ULBBlendP: {target_params:,} params")
        except Exception:
            pass  # ULBBlock not available, skip matching
    
    def _wanted(name):
        return requested_models is None or name in requested_models

    def try_add(name, make_fn, cfg_desc):
        if not _wanted(name):
            return
        try:
            if poe:
                model = _stack_poe(make_fn, dim, pool_size=n_experts * n_layers, top_k=top_k, max_hops=poe_max_hops,
                                   block_shared_fraction=poe_block_share_fraction,
                                   router_shared_fraction=poe_router_share_fraction,
                                   hop_shared_fraction=poe_hop_share_fraction,
                                   router_noise=router_noise)
            elif moe:
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

    # StupidAttn: pairwise element-wise multiply, gate back in
    try_add('StupidAttn', lambda: StupidAttnBlock(d_model=dim),
            f"StupidAttnBlock(d_model={dim})")

    # MHA + feat-attn + MLP: test whether feat-attn makes MLP useful for algorithmic tasks
    try_add('MHA+FA+MLP', lambda: MHAFeatMLPBlock(d_model=dim, n_heads=4, mlp_inner=mha_mlp, feat_position='before_mlp'),
            f"MHAFeatMLPBlock(d_model={dim}, mlp_inner={mha_mlp}, feat_position='before_mlp')")
    try_add('MHA+MLP+FA', lambda: MHAFeatMLPBlock(d_model=dim, n_heads=4, mlp_inner=mha_mlp, feat_position='after_mlp'),
            f"MHAFeatMLPBlock(d_model={dim}, mlp_inner={mha_mlp}, feat_position='after_mlp')")

    # MHA2D: 2D token representations with 2D QKV projections
    _sqrt_dim = int(dim ** 0.5)
    assert _sqrt_dim * _sqrt_dim == dim, f"dim ({dim}) must be a perfect square for MHA2D"
    try_add('MHA2D', lambda: MHA2DBlock(d_model=dim, n_channels=_sqrt_dim, n_heads=_sqrt_dim, mlp_inner=mha_mlp),
            f"MHA2DBlock(d_model={dim}, n_channels={_sqrt_dim}, n_heads={_sqrt_dim}, mlp_inner={mha_mlp})")

    # MHA2DS: simple 2D — seq-attn per channel + feat-attn per position, no MLP
    try_add('MHA2DS', lambda: MHA2DSimpleBlock(d_model=dim),
            f"MHA2DSimpleBlock(d_model={dim}, C={_sqrt_dim})")

    # FeatAttn: MHA + feature-attention (no MLP)
    _ft = feat_transpose
    _fu = feat_up_factor
    try_add('FeatAttn', lambda: FeatAttnBlock(d_model=dim, n_heads=4, up_factor=_fu, transpose_groups=_ft),
            f"FeatAttnBlock(d_model={dim}, n_heads=4, feat_expansion=4, up_factor={_fu}, transpose={_ft})")

    # FeatAttnFirst: feature-attention before sequence-attention
    try_add('FeatAttnFirst', lambda: FeatAttnBlock(d_model=dim, n_heads=4, feat_first=True, up_factor=_fu, transpose_groups=_ft),
            f"FeatAttnBlock(d_model={dim}, n_heads=4, feat_first=True, up_factor={_fu}, transpose={_ft})")

    # DualMHA: two independent MHA stacks, subtract outputs
    if _wanted('DualMHA'):
        _dual_mlp = mha_mlp  # reuse same MLP inner dim as MHA
        model = DualMHABenchWrapper(d_model=dim, n_heads=4, n_layers=n_layers, mlp_inner=_dual_mlp)
        model._bench_config = f"DualMHA(d_model={dim}, n_heads=4, n_layers={n_layers}, mlp_inner={_dual_mlp})"
        models['DualMHA'] = model

    # DiffMHA: paired MHA blocks, subtract deltas at each depth level
    try_add('DiffMHA', lambda: DiffMHABlock(d_model=dim, n_heads=4, mlp_inner=mha_mlp),
            f"DiffMHABlock(d_model={dim}, n_heads=4, mlp_inner={mha_mlp})")

    # MulMHA: paired MHA blocks, multiply deltas at each depth level
    try_add('MulMHA', lambda: MulMHABlock(d_model=dim, n_heads=4, mlp_inner=mha_mlp),
            f"MulMHABlock(d_model={dim}, n_heads=4, mlp_inner={mha_mlp})")

    # OuterMHA: paired MHA blocks, outer product (a * linear(b)) at each depth level
    try_add('OuterMHA', lambda: OuterMHABlock(d_model=dim, n_heads=4, mlp_inner=mha_mlp),
            f"OuterMHABlock(d_model={dim}, n_heads=4, mlp_inner={mha_mlp})")

    # ExpandKV: K/V expanded 4x along sequence dimension
    if _wanted('ExpandKV'):
        try:
            from mha import ExpandKVMHALayer
            try_add('ExpandKV', lambda: ExpandKVMHALayer(dim=dim, n_heads=4, kv_expand=4),
                    f"ExpandKVMHALayer(dim={dim}, n_heads=4, kv_expand=4)")
        except ImportError:
            pass

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

    # ULB (Universal Learning Block) ablation family
    try_add('ULBBlendP', lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='blend', swish_mode='learnable')),
            f"ULBBlock(ULBConfig(d_model={dim}, n_heads=4, paired=True, attn_mode='blend', swish_mode='learnable'))")

    # K-mix ablation
    try_add('ULBBlendPNoK', lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='blend', k_mix='none', swish_mode='learnable')),
            f"ULBBlock(ULBConfig(d_model={dim}, n_heads=4, paired=True, attn_mode='blend', k_mix='none', swish_mode='learnable'))")

    # Attention-path ablations
    try_add('ULBSoftmaxP', lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='softmax', swish_mode='learnable')),
            f"ULBBlock(ULBConfig(d_model={dim}, n_heads=4, paired=True, attn_mode='softmax', swish_mode='learnable'))")
    try_add('ULBSilu2P', lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='silu2', swish_mode='learnable')),
            f"ULBBlock(ULBConfig(d_model={dim}, n_heads=4, paired=True, attn_mode='silu2', swish_mode='learnable'))")

    # Activation ablation
    try_add('ULBBlendPSiLU', lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='blend', swish_mode='silu')),
            f"ULBBlock(ULBConfig(d_model={dim}, n_heads=4, paired=True, attn_mode='blend', swish_mode='silu'))")

    # Paired-head ablation
    try_add('ULBBlend', lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=False, attn_mode='blend', swish_mode='learnable')),
            f"ULBBlock(ULBConfig(d_model={dim}, n_heads=4, paired=False, attn_mode='blend', swish_mode='learnable'))")

    # ULB + feature-attention sublayer (sigmoid gate + feat-attn with tensor projections, NOT true 2D)
    _sqrt_dim_ulb = int(dim ** 0.5)
    try_add('ULBBlendPFA', lambda: ULBBlock(ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='blend', swish_mode='learnable', feat_attn=True, feat_c_h=_sqrt_dim_ulb, feat_c_w=_sqrt_dim_ulb)),
            f"ULBBlock(ULBConfig(d_model={dim}, feat_attn=True, feat_c_h={_sqrt_dim_ulb}, feat_c_w={_sqrt_dim_ulb}))")

    # True end-to-end 2D ULB — all projections are (C,C,C,C) tensors, tokens as (B,T,C,C) throughout
    # Wrap to accept/return flat (B,T,D) for bench_ssm compatibility
    from ulb.block import ULB2DBlock, ULB2DConfig
    class _ULB2DFlat(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.block = ULB2DBlock(cfg)
            self.c_h, self.c_w = cfg.c_h, cfg.c_w
            self.aux_loss = 0.0
        def forward(self, x):
            B, T, D = x.shape
            y = self.block(x.view(B, T, self.c_h, self.c_w))
            self.aux_loss = self.block.aux_loss
            return y.reshape(B, T, D)
    try_add('ULBBlendP2D', lambda: _ULB2DFlat(ULB2DConfig(c_h=_sqrt_dim_ulb, c_w=_sqrt_dim_ulb)),
            f"ULB2DBlock(ULB2DConfig(c_h={_sqrt_dim_ulb}, c_w={_sqrt_dim_ulb}))")
    try_add('ULBBlendP2D-noFA', lambda: _ULB2DFlat(ULB2DConfig(c_h=_sqrt_dim_ulb, c_w=_sqrt_dim_ulb, use_feat_attn=False)),
            f"ULB2DBlock(ULB2DConfig(c_h={_sqrt_dim_ulb}, c_w={_sqrt_dim_ulb}, use_feat_attn=False))")

    # LLooM: dual-paradigm adaptive routing (self-contained, bypasses stacking)
    if _wanted('LLooM'):
        try:
            pool_size = n_experts  # same pool size for both sides
            lloom_top_k = min(top_k, pool_size)
            # Token side gets 2x the hop budget (MLP hops are cheap)
            seq_hops = poe_max_hops if poe_max_hops is not None else 2 * pool_size
            tok_hops = seq_hops * 2
            # Map PoE sharing args to LLooM config.
            # PoE has 3 sharing fractions; LLooM has a single shared_fraction
            # default plus 4 optional per-category overrides. We set the
            # global default from block share, and per-category router share
            # from the max of router + hop share.
            expert_share = poe_block_share_fraction
            router_share = max(poe_router_share_fraction, poe_hop_share_fraction)
            lloom_cfg = dict(dim=dim, seq_pool_size=pool_size, tok_pool_size=pool_size,
                             seq_top_k=lloom_top_k, tok_top_k=lloom_top_k,
                             seq_max_hops=seq_hops, tok_max_hops=tok_hops,
                             max_bridge_crossings=2,
                             shared_fraction=expert_share,
                             seq_router_shared_fraction=router_share,
                             tok_router_shared_fraction=router_share,
                             router_noise=router_noise,
                             exit_ramp_scale=args.exit_ramp_scale)
            model = LLooMBenchWrapper(**lloom_cfg)
            _es = expert_share
            _rs = router_share
            model._bench_config = (f"LLooM(dim={dim}, pool={pool_size}, top_k={lloom_top_k}, "
                                   f"seq_hops={seq_hops}, tok_hops={tok_hops}, "
                                   f"expert_share={_es}, router_share={_rs}, noise={router_noise})")
            models['LLooM'] = model
        except ImportError as e:
            print(f"  Warning: LLooM not available ({e})")
        except Exception as e:
            print(f"  Warning: LLooM failed to initialize ({e})")

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
    parser.add_argument('--early-stop-acc', type=float, default=0.98, help='Early stop when val acc exceeds this')
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
                        help='Disable auto-sizing SSMs to match ULBBlendP param count')
    parser.add_argument('--moe', action='store_true', help='Use MoE stacking for all models')
    parser.add_argument('--poe', action='store_true', help='Use PoolOfExperts (dynamic-depth expert pool)')
    parser.add_argument('--n-experts', type=int, default=4, help='Number of MoE experts per layer')
    parser.add_argument('--poe-max-hops', type=int, default=None, help='Max routing hops for PoolOfExperts (default: 2x pool_size)')
    parser.add_argument('--poe-block-share-fraction', type=float, default=0.0,
                        help='Fraction of expert block output dims shared across pool (0.0=independent)')
    parser.add_argument('--poe-router-share-fraction', type=float, default=0.0,
                        help='Fraction of expert router output dims shared across pool (0.0=independent)')
    parser.add_argument('--poe-hop-share-fraction', type=float, default=0.0,
                        help='Fraction of hop embed/gate dims shared across experts (0.0=independent)')
    parser.add_argument('--router-noise', type=float, default=1.0,
                        help='Starting router noise scale (linearly annealed to 0 over training)')
    parser.add_argument('--exit-ramp-scale', type=float, default=2.0,
                        help='Exit bias ramp scale for LLooM/PoE (exit_bias = exit_ramp_scale * hops_used/max_hops)')
    parser.add_argument('--top-k', type=int, default=2, help='Top-k expert selection per sample')
    parser.add_argument('--feat-transpose', action='store_true',
                        help='Transpose FeatureAttn groups: attend over D/G features instead of G groups')
    parser.add_argument('--feat-up-factor', type=int, default=2,
                        help='FeatureAttn up/down projection factor (default: 2)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Output CSV file path (optional)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save trained models (optional)')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for data pregeneration and model init')
    args = parser.parse_args()

    if args.moe and args.poe:
        parser.error("--moe and --poe are mutually exclusive")

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
    run_seed = args.seed

    requested = args.models  # None means all available
    match_params = not args.no_match_params
    models_info = make_models(dim, n_layers=n_layers, requested_models=requested, match_params=match_params,
                              n_experts=args.n_experts, top_k=args.top_k, moe=args.moe,
                              poe=args.poe, poe_max_hops=args.poe_max_hops,
                              poe_block_share_fraction=args.poe_block_share_fraction,
                              poe_router_share_fraction=args.poe_router_share_fraction,
                              poe_hop_share_fraction=args.poe_hop_share_fraction,
                              router_noise=args.router_noise,
                              feat_transpose=args.feat_transpose, feat_up_factor=args.feat_up_factor)
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

    print(f"\nMax {max_epochs} epochs, {n_train} train batches, {n_val} val batches, B={B}, L={L}, dim={dim}, layers={n_layers}, seed={run_seed}")
    print(f"Early stop at >={args.early_stop_acc:.0%} val accuracy")
    print("\nTraining config:")
    print("- Optimizer: Adam")
    print("- Loss: cross_entropy(ignore_index=-100)")
    print("- Vocab: unified VOCAB_SIZE=64 (shared embedding/head across tasks)")
    print("- LR: mixed=1e-3, standalone tasks=1e-4")
    print("- LR schedule: 1 epoch linear warmup (0.1x -> 1.0x), then cosine decay to 0.05x")
    print("- Hard mining (mixed only): activates when majority of subtasks individually converge")
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
        task_data = pregen_task_data(task, task_n_train, task_n_val, B, L, run_seed, device=DEVICE)
        print(f"  {len(task_data['train'])} train + {len(task_data['val'])} val batches ready on {DEVICE}")
        
        model_pbar = tqdm(all_names, desc="Models", position=1, leave=False)
        for name in model_pbar:
            model_pbar.set_description(f"Model: {name}")
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
            model = make_models(dim, n_layers=n_layers, requested_models=[name], match_params=match_params,
                                n_experts=args.n_experts, top_k=args.top_k, moe=args.moe,
                                poe=args.poe, poe_max_hops=args.poe_max_hops,
                                poe_block_share_fraction=args.poe_block_share_fraction,
                                poe_router_share_fraction=args.poe_router_share_fraction,
                                poe_hop_share_fraction=args.poe_hop_share_fraction,
                                router_noise=args.router_noise,
                                feat_transpose=args.feat_transpose, feat_up_factor=args.feat_up_factor)[name]
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
            
            # Save model if requested
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(args.save_dir, f"{name}_{task}.pt")
                _model = getattr(model, '_orig_mod', model)  # unwrap torch.compile
                save_data = {
                    'state_dict': _model.state_dict(),
                    'embed_state_dict': r['embed_state_dict'],
                    'head_state_dict': r['head_state_dict'],
                    'model_name': name,
                    'task': task,
                    'dim': dim,
                    'n_layers': n_layers,
                    'params': param_count,
                    'result': r,
                    'args': vars(args),
                }
                torch.save(save_data, save_path)
                tqdm.write(f"  saved → {save_path}")

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
    print("\n" + "=" * 120)
    print("## Benchmark Results\n")
    print(f"Global setup: dim={dim}, layers={n_layers}, B={B}, L={L}, max_epochs={max_epochs}, train_batches={n_train}, val_batches={n_val}")
    print()

    # Group by task for clearer presentation
    for task in tasks:
        print(f"### {task}\n")
        if task == 'mixed':
            subtask_hdrs = " | ".join(f"{t}" for t in ALL_TASKS)
            print(f"| Rank | Model | Params | Init Loss | Best Loss | Val Acc | Min Subtask | {subtask_hdrs} | Best @ | Epochs | Stop | Time |")
            sep_cols = " | ".join("------:" for _ in ALL_TASKS)
            print(f"|----:|:------|-------:|----------:|----------:|--------:|------------:| {sep_cols} |-------:|-------:|:-----|-----:|")
        else:
            print("| Rank | Model | Params | Init Loss | Best Loss | Val Acc | Best @ | Epochs | Stop | Time |")
            print("|----:|:------|-------:|----------:|----------:|--------:|-------:|-------:|:-----|-----:|")

        task_results = [r for r in all_results if r['task'] == task]
        task_results.sort(key=lambda x: x['val_acc'], reverse=True)

        for rank, r in enumerate(task_results, start=1):
            acc_str = f"{r['val_acc']:.1%}"
            if r['converged']:
                acc_str = f"**{acc_str}**"

            best_epoch = r.get('best_epoch', r['epochs'])
            stop = r.get('stop_reason', 'MAX_EPOCH')
            wall_s = r.get('wall_s', 0)
            init_loss = r.get('initial_loss', float('nan'))
            best_loss = r.get('final_loss', float('nan'))

            if task == 'mixed':
                sub_vals = [r.get(f'acc_{t}', 0.0) for t in ALL_TASKS]
                min_sub = min(sub_vals) if sub_vals else 0.0
                sub_cols = " | ".join(f"{a:.1%}" for a in sub_vals)
                print(f"| {rank} | {r['model']} | {r['params']:,} | {init_loss:.4f} | {best_loss:.4f} | {acc_str} | {min_sub:.1%} | {sub_cols} | {best_epoch} | {r['epochs']} | {stop} | {wall_s:.1f}s |")
            else:
                print(f"| {rank} | {r['model']} | {r['params']:,} | {init_loss:.4f} | {best_loss:.4f} | {acc_str} | {best_epoch} | {r['epochs']} | {stop} | {wall_s:.1f}s |")

        print()
    
    # Legend
    print("**Legend:**")
    print("- **Bold** = converged (>={:.0%} val acc)".format(args.early_stop_acc))
    print("- Best @ = epoch with best val accuracy")
    print("- Stop: CONVERGED (hit target) | PLATEAU (train stuck at 100%) | MAX_EPOCH")
