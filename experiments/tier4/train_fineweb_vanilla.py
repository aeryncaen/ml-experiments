import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.profiler
from einops import rearrange
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from s6 import USBBlock, USBConfig


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


@dataclass
class HParams:
    data_path: str = os.environ.get("DATA_PATH", str(Path(__file__).resolve().parent))
    train_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_train_*.bin")
    val_files: str = os.path.join(data_path, "data/fineweb10B/fineweb_val_*.bin")

    model_type: str = os.environ.get("MODEL_TYPE", "transformer")  # transformer | feat_attn | fused_seq_feat | fused_qkv | three_stage | three_stage_fsa | transformer_shift | transformer_gate | fused_gate | transformer_s4d | s6 | ulb | ulb_fa | ulb_2d
    vocab_size: int = 50304
    n_layer: int = _env_int("N_LAYER", 12)
    n_head: int = _env_int("N_HEAD", 12)
    d_model: int = _env_int("D_MODEL", 768)
    seq_len: int = _env_int("SEQ_LEN", 2048)

    # 2D factorization (ulb_2d): 0 = auto-factor from d_model
    n_features: int = _env_int("N_FEATURES", 0)
    desc_dim: int = _env_int("DESC_DIM", 0)

    # Feature-attention MLP knobs (feat_attn model type)
    fa_n_features: int = _env_int("FA_N_FEATURES", 64)       # features to attend over (must divide hidden=2048)
    fa_activation: str = os.environ.get("FA_ACTIVATION", "softmax")  # softmax | silu | silu2
    fa_post_act: str = os.environ.get("FA_POST_ACT", "none")  # none | silu | gelu — element-wise activation after feature attention
    fa_pre_act: str = os.environ.get("FA_PRE_ACT", "none")   # none | silu | gelu — element-wise activation on gate before Q=K scores
    fa_qk_norm: bool = _env_bool("FA_QK_NORM", False)        # RMSNorm on Q=K before scoring

    # Fused gate knobs (0 = use d_model, no expansion)
    fused_inner_dim: int = _env_int("FUSED_INNER_DIM", 0)

    # S6 knobs
    s6_headdim: int = _env_int("S6_HEADDIM", 64)
    s6_expansion_factor: int = _env_int("S6_EXPANSION_FACTOR", 2)
    s6_post_scan_attention: bool = _env_bool("S6_POST_SCAN_ATTENTION", True)
    s6_scan_state_modes: str = os.environ.get("S6_SCAN_STATE_MODES", "elementwise,elementwise,elementwise")

    train_steps: int = _env_int("TRAIN_STEPS", 2000)
    batch_size: int = _env_int("BATCH_SIZE", 8)
    val_steps: int = _env_int("VAL_STEPS", 32)
    val_every: int = _env_int("VAL_EVERY", 100)
    lr: float = _env_float("LR", 3e-4)
    warmup_steps: int = _env_int("WARMUP_STEPS", 200)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.1)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)
    lr_schedule: str = os.environ.get("LR_SCHEDULE", "cosine")  # cosine | wsd
    wsd_decay_frac: float = _env_float("WSD_DECAY_FRAC", 0.33)  # fraction of steps for decay phase
    compile: bool = _env_bool("TORCH_COMPILE", True)
    torch_profile: bool = _env_bool("TORCH_PROFILE", False)
    torch_profile_steps: int = _env_int("TORCH_PROFILE_STEPS", 50)


HP = HParams()


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return rank, world_size, torch.device("cuda", local_rank)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print0(rank: int, s: str):
    if rank == 0:
        print(s, flush=True)


def _load_data_shard(file: Path) -> torch.Tensor:
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert int(header[0]) == 20240520, "magic number mismatch in .bin"
    assert int(header[1]) == 1, "unsupported .bin version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "token count mismatch"
    return tokens


class ShardStream:
    def __init__(self, pattern: str, rank: int, world_size: int, seq_len: int, batch_size: int):
        self.files = [Path(f) for f in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")
        self.rank = rank
        self.world_size = world_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokens_per_rank = seq_len * batch_size
        self.tokens_per_global_step = self.tokens_per_rank * world_size
        self.file_idx = 0
        self.pos = 0
        self.tokens = _load_data_shard(self.files[self.file_idx])

    def _advance_shard(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def next_batch(self, device: torch.device):
        needed = self.tokens_per_global_step + 1
        if self.pos + needed >= self.tokens.numel():
            self._advance_shard()
        start = self.pos + self.rank * self.tokens_per_rank
        end = start + self.tokens_per_rank + 1
        buf = self.tokens[start:end]
        self.pos += self.tokens_per_global_step
        x = buf[:-1].to(dtype=torch.int64).view(self.batch_size, self.seq_len)
        y = buf[1:].to(dtype=torch.int64).view(self.batch_size, self.seq_len)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class Mamba3Mixer(nn.Module):
    """Mamba-3-style mixer: chunked SSD with trapezoidal discretization,
    data-dependent RoPE on B/C, and BC bias.

    Quadratic form: Y = (decay_mask ⊙ C B^T) X  per chunk.
    C B^T is (chunk, chunk) via N-contraction — never materializes (P, N) per position.
    Inter-chunk state passing via sequential scan over chunk boundaries.
    """

    def __init__(self, d_model: int, n_head: int, d_state: int = 64):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head  # p
        self.d_state = d_state             # n

        # Input projections: x -> (X, B, C, dt, lambda, theta)
        Nr = d_state // 2
        proj_dim = (
            d_model             # X  (H * p)
            + n_head * d_state  # B  (H * n)
            + n_head * d_state  # C  (H * n)
            + n_head            # log_dt (H)
            + n_head            # lambda_logit (H)
            + n_head * Nr       # theta (H * n/2)
        )
        self.in_proj = nn.Linear(d_model, proj_dim, bias=False)

        # BC bias: learnable, head-specific, channel-wise, init=1
        self.b_bias = nn.Parameter(torch.ones(n_head, d_state))
        self.c_bias = nn.Parameter(torch.ones(n_head, d_state))
        self.b_bias._no_weight_decay = True   # type: ignore[attr-defined]
        self.c_bias._no_weight_decay = True   # type: ignore[attr-defined]

        # QK-norm on B,C
        self.b_norm = nn.RMSNorm(d_state)
        self.c_norm = nn.RMSNorm(d_state)

        # Scalar decay per head
        self.log_A = nn.Parameter(torch.log(0.5 * torch.ones(n_head)))
        self.log_A._no_weight_decay = True    # type: ignore[attr-defined]

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        B, T, D = x.shape
        H, P, N = self.n_head, self.head_dim, self.d_state
        Nr = N // 2
        L = chunk_size
        assert T % L == 0
        K = T // L  # number of chunks

        # ---- project ----
        proj = self.in_proj(x)                                    # (B, T, proj_dim)
        i = 0
        X   = proj[..., i:i+H*P];  i += H*P
        Bv  = proj[..., i:i+H*N];  i += H*N
        Cv  = proj[..., i:i+H*N];  i += H*N
        ldt = proj[..., i:i+H];    i += H
        llam= proj[..., i:i+H];    i += H
        th  = proj[..., i:i+H*Nr]; i += H*Nr

        X   = rearrange(X,   'b t (h p) -> b t h p', h=H)
        Bv  = rearrange(Bv,  'b t (h n) -> b t h n', h=H)
        Cv  = rearrange(Cv,  'b t (h n) -> b t h n', h=H)
        ldt = rearrange(ldt, 'b t h -> b t h')
        llam= rearrange(llam,'b t h -> b t h')
        th  = rearrange(th,  'b t (h r) -> b t h r', h=H)

        # ---- discretization ----
        dt    = F.softplus(ldt).clamp(max=2.0)                    # (B, T, H)
        A_neg = self.log_A.exp().neg()                            # (H,) guaranteed negative
        alpha = torch.exp(dt * A_neg).clamp(max=0.999)            # (B, T, H) strict <1
        lam   = torch.sigmoid(llam)                               # (B, T, H)
        gamma = lam * dt                                          # (B, T, H)
        beta  = (1.0 - lam) * dt * alpha                         # (B, T, H)

        # ---- QK-norm + BC bias ----
        Bv = self.b_norm(Bv) * self.b_bias
        Cv = self.c_norm(Cv) * self.c_bias

        # ---- data-dependent RoPE on B, C ----
        cum_th = th.cumsum(dim=1)
        cos_th, sin_th = cum_th.cos(), cum_th.sin()
        def dd_rope(v: torch.Tensor) -> torch.Tensor:
            v1, v2 = v[..., :Nr], v[..., Nr:]
            return torch.cat([v1*cos_th - v2*sin_th, v1*sin_th + v2*cos_th], dim=-1)
        Bv = dd_rope(Bv)
        Cv = dd_rope(Cv)

        # ---- trapezoidal: apply size-2 conv to B before chunking ----
        # B_trap_t = γ_t * B_t + β_t * B_{t-1}   (and same scaling to X)
        # But we need to keep B and X separate for the SSD form.
        # The trap conv applies to the "KV" = B * x product in the recurrence,
        # but in SSD quadratic form Y = (mask ⊙ C B^T) X, the trap modifies B.
        # We scale: B_trap_t = γ_t * B_t,  and add β_t * B_{t-1} contribution
        # For the quadratic form we need to apply trap to both B and X consistently.
        # Simplest correct way: fold γ into B, β into a shifted-B term.
        Bv_g = Bv * gamma.unsqueeze(-1)                           # γ_t * B_t
        Bv_b = Bv * beta.unsqueeze(-1)                            # β_t * B_t (will shift)
        # Shift Bv_b by 1 position: Bv_b_shifted[t] = β_t * B_{t-1}
        # Bv_b is (B, T, H, N) — pad dim=1 (time): F.pad pads from last dim, so (0,0, 0,0, 1,0)
        Bv_b_shifted = F.pad(Bv_b[:, :-1], (0, 0, 0, 0, 1, 0))  # zero at t=0
        Bv_trap = Bv_g + Bv_b_shifted                             # (B, T, H, N)
        # Similarly for X: (B, T, H, P)
        X_g = X * gamma.unsqueeze(-1)
        X_b = X * beta.unsqueeze(-1)
        X_b_shifted = F.pad(X_b[:, :-1], (0, 0, 0, 0, 1, 0))
        X_trap = X_g + X_b_shifted                                # (B, T, H, P)

        # ---- chunk ----
        Bv_c  = rearrange(Bv_trap, 'b (k l) h n -> b k h l n', l=L)   # (B, K, H, L, N)
        Cv_c  = rearrange(Cv,      'b (k l) h n -> b k h l n', l=L)
        X_c   = rearrange(X_trap,  'b (k l) h p -> b k h l p', l=L)
        al_c  = rearrange(alpha,   'b (k l) h   -> b k h l',   l=L)

        # ---- intra-chunk decay mask (L, L) ----
        log_al = al_c.clamp(min=1e-6).log()                      # (B, K, H, L)
        log_cum = log_al.cumsum(dim=-1)                           # (B, K, H, L)
        # decay[i,j] = exp(cum[i] - cum[j]) for i>=j
        decay = (log_cum.unsqueeze(-1) - log_cum.unsqueeze(-2)).exp()
        decay = decay * torch.tril(torch.ones(L, L, device=x.device))  # (B, K, H, L, L)

        # ---- intra-chunk SSD: Y_intra = (decay ⊙ C B^T) X ----
        # C B^T: (B, K, H, L, L) via N-contraction  — this is the key: no (P,N) per position
        CB = torch.einsum('bkhin,bkhjn->bkhij', Cv_c, Bv_c)      # (B, K, H, L, L)
        attn = decay * CB                                         # (B, K, H, L, L)
        Y_intra = torch.einsum('bkhij,bkhjp->bkhip', attn, X_c)  # (B, K, H, L, P)

        # ---- inter-chunk: accumulate (N, P) states across chunks ----
        # Each chunk's contribution to the state: sum_l decay_to_end[l] * B[l] ⊗ X[l]
        # = B_c^T @ diag(decay_to_end) @ X_c   — (N, L) @ (L, L) @ (L, P) = (N, P)
        # decay_to_end[l] = exp(log_cum[L-1] - log_cum[l])
        decay_to_end = (log_cum[..., -1:] - log_cum).exp()        # (B, K, H, L)
        # chunk_state[k] = B_c^T @ diag(d2e) @ X_c = einsum('bkhln,bkhl,bkhlp->bkhnp')
        chunk_state = torch.einsum(
            'bkhln,bkhl,bkhlp->bkhnp', Bv_c, decay_to_end, X_c
        )                                                         # (B, K, H, N, P)
        chunk_decay = log_al.sum(dim=-1).exp()                    # (B, K, H) total decay per chunk

        # Sequential scan across K chunks
        prev_states = torch.zeros(B, K, H, N, P, device=x.device, dtype=x.dtype)
        h = torch.zeros(B, H, N, P, device=x.device, dtype=x.dtype)
        for k in range(K):
            prev_states[:, k] = h
            h = chunk_decay[:, k, :, None, None] * h + chunk_state[:, k]

        # ---- inter-chunk output contribution ----
        # For position l in chunk k: Y_inter = C[k,l] @ (decay_from_start[l] * prev_state[k]) @ ... hmm
        # Actually: Y_inter[k,l] = C[k,l]^T @ (decay_from_start[l] * h_prev[k]) ... but h is (N,P)
        # So Y_inter[k,l] in R^P = h_prev[k]^T @ (decay[l] * C[k,l])   with h (N,P), C (N,)
        # = (decay[l] * C[k,l])^T @ h_prev[k]  -> einsum over N -> (P,)
        decay_from_start = log_cum.exp()                          # (B, K, H, L)
        # C_scaled = decay_from_start * C
        C_scaled = Cv_c * decay_from_start.unsqueeze(-1)          # (B, K, H, L, N)
        Y_inter = torch.einsum(
            'bkhln,bkhnp->bkhlp', C_scaled, prev_states
        )                                                         # (B, K, H, L, P)

        # ---- combine and project out ----
        Y = Y_intra + Y_inter                                     # (B, K, H, L, P)
        Y = rearrange(Y, 'b k h l p -> b (k l) (h p)')           # (B, T, D)
        return self.out_proj(Y)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor):
        t = x.shape[1]
        if self.cos_cached is None or t != self.seq_len_cached:
            self.seq_len_cached = t
            inv_freq = self.inv_freq.to(device=x.device)
            tt = torch.arange(t, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(tt, inv_freq)
            self.cos_cached = freqs.cos()[None, :, None, :]
            self.sin_cached = freqs.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached


class PairedRotary(nn.Module):
    """Rotary for paired heads: even/odd position encodings for 2*dim width.

    When applied to a tensor of last dim = 2*head_dim, the first head_dim
    channels get even-position encodings and the second head_dim channels
    get odd-position encodings. After reshape to 2T length, this gives
    adjacent heads interleaved positional identity.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim  # head_dim (half of the paired width)
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor):
        t = x.shape[1]
        if self.cos_cached is None or t != self.seq_len_cached:
            self.seq_len_cached = t
            inv_freq = self.inv_freq.to(device=x.device)
            tt = torch.arange(t, device=x.device, dtype=inv_freq.dtype)
            t_even = 2 * tt
            t_odd = 2 * tt + 1
            freqs_even = torch.outer(t_even, inv_freq)
            freqs_odd = torch.outer(t_odd, inv_freq)
            # Concat along last dim: [even_cos | odd_cos], [even_sin | odd_sin]
            self.cos_cached = torch.cat([freqs_even.cos(), freqs_odd.cos()], dim=-1)[None, :, None, :]
            self.sin_cached = torch.cat([freqs_even.sin(), freqs_odd.sin()], dim=-1)[None, :, None, :]
        return self.cos_cached, self.sin_cached


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim)
        k = self.k(x).view(b, t, self.n_head, self.head_dim)
        v = self.v(x).view(b, t, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if HAS_FLASH_ATTN:
            y = flash_attn_func(q, k, v, causal=True)         # (b, t, h, d)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2)
        y = y.contiguous().view(b, t, c)
        return self.proj(y)


class ShiftAttention(nn.Module):
    """Attention with temporal channel-group shifts on K and V.

    Within each head, d_head is split into groups. Each group's channels
    at position t come from a different past position (t, t-1, t-2, t-4).
    Each KV cache entry becomes a chimera encoding a temporal neighborhood.
    No new parameters. FlashAttention unchanged. Just a reindex.
    """

    def __init__(self, d_model: int, n_head: int, shifts: tuple[int, ...] = (0, 1, 2, 4)):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert self.head_dim % len(shifts) == 0, \
            f"head_dim {self.head_dim} must be divisible by {len(shifts)} shift groups"
        self.cpg = self.head_dim // len(shifts)  # channels per group
        self.shifts_list = shifts
        self.max_shift = max(shifts)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    def _shift_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Shift channel groups within each head along the time axis.
        x: (B, T, H, D).  Group i gets shifted by shifts[i] positions causally.
        """
        if self.max_shift == 0:
            return x
        B, T, H, D = x.shape
        out = torch.zeros_like(x)
        for i, s in enumerate(self.shifts_list):
            sl = slice(i * self.cpg, (i + 1) * self.cpg)
            if s == 0:
                out[:, :, :, sl] = x[:, :, :, sl]
            else:
                out[:, s:, :, sl] = x[:, :T-s, :, sl]
                # out[:, :s, :, sl] stays zero — no history yet
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim)
        k = self.k(x).view(b, t, self.n_head, self.head_dim)
        v = self.v(x).view(b, t, self.n_head, self.head_dim)
        k = self._shift_kv(k)
        v = self._shift_kv(v)
        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if HAS_FLASH_ATTN:
            y = flash_attn_func(q, k, v, causal=True)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2)
        y = y.contiguous().view(b, t, c)
        return self.proj(y)


class GatedNeighborAttention(nn.Module):
    """Attention with gated neighbor mixing on K.

    Half the channels in each head's K get a single lerp with t-1:
        k_out[t] = (1 - g[t]) * k[t] + g[t] * k[t-1]
    Gate is per-channel per-head, content-dependent.
    First half of channels stays untouched.
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.half_dim = self.head_dim // 2

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.gate_proj = nn.Linear(d_model, n_head * self.half_dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def _gated_neighbor(self, k: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        half = self.half_dim
        k_static = k[:, :, :, :half]
        k_cur = k[:, :, :, half:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_mixed = (1 - gate) * k_cur + gate * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim)
        k = self.k(x).view(b, t, self.n_head, self.head_dim)
        v = self.v(x).view(b, t, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        gate = torch.sigmoid(self.gate_proj(x)).view(b, t, self.n_head, self.half_dim)
        k = self._gated_neighbor(k, gate)
        if HAS_FLASH_ATTN:
            y = flash_attn_func(q, k, v, causal=True)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2)
        y = y.contiguous().view(b, t, c)
        return self.proj(y)


class GatedNeighborBlock(nn.Module):
    """Separate attn + MLP block using GatedNeighborAttention."""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = GatedNeighborAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class FusedGatedNeighborBlock(nn.Module):
    """Fused attention+MLP block. No separate MLP — one expand/contract cycle.

    x -> norm -> up_proj(d_model -> inner) -> SiLU -> h_up
             -> Q(inner -> inner)                      |
             -> K(inner -> inner)                      |
             -> V(inner -> inner)                      |
             -> gated neighbor on K                    |
             -> RoPE on Q, K                           |
             -> attention(Q, K, V) -> attn_out         |
             -> norm(attn_out) + h_up  <---------------+
             -> down_proj(inner -> d_model) -> residual

    SiLU after up_proj feeds rich features into QKV. Skip-add from h_up
    to normed attention output preserves pre-attn features for down_proj.
    """

    def __init__(self, d_model: int, n_head: int, inner_dim: int | None = None, expand: int = 1, paired: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.paired = paired
        self.inner_dim = inner_dim if inner_dim is not None else d_model * expand
        assert self.inner_dim % n_head == 0
        self.head_dim = self.inner_dim // n_head
        self.half_dim = self.head_dim // 2

        self.norm = RMSNorm(d_model)

        # Shared expansion: d_model -> inner_dim
        self.up_proj = nn.Linear(d_model, self.inner_dim, bias=False)

        # QKV projections from expanded space
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        # Post-attention norm (before skip add)
        self.attn_norm = RMSNorm(self.inner_dim)

        # Down-project: inner_dim -> d_model
        self.down_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        # Learnable Swish beta (per-channel)
        self.swish_beta_up = nn.Parameter(torch.ones(self.inner_dim))
        self.swish_beta_down = nn.Parameter(torch.ones(self.inner_dim))

        # QK norm (per-head RMSNorm before RoPE)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # RoPE — paired mode uses PairedRotary with 2*head_dim width
        if paired:
            assert n_head % 2 == 0, "paired heads requires even n_head"
            self.rotary = PairedRotary(self.head_dim)
        else:
            self.rotary = Rotary(self.head_dim)

        # Neighbor gate: single lerp on K, per-channel per-head
        self.neighbor_gate_proj = nn.Linear(d_model, n_head * self.half_dim, bias=True)
        nn.init.zeros_(self.neighbor_gate_proj.weight)
        nn.init.constant_(self.neighbor_gate_proj.bias, -2.0)

    def _gated_neighbor(self, k: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        half = self.half_dim
        k_static = k[:, :, :, :half]
        k_cur = k[:, :, :, half:]
        k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
        k_mixed = (1 - gate) * k_cur + gate * k_prev
        return torch.cat([k_static, k_mixed], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.norm(x)

        # Expand and activate (Swish with learnable beta)
        h_up = self.up_proj(h)
        h_up = h_up * torch.sigmoid(self.swish_beta_up * h_up)

        # QKV from activated expanded space
        q = self.q_proj(h_up).view(b, t, self.n_head, self.head_dim)
        k = self.k_proj(h_up).view(b, t, self.n_head, self.head_dim)
        v = self.v_proj(h_up).view(b, t, self.n_head, self.head_dim)

        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Gated neighbor mixing on K (before pairing)
        gate = torch.sigmoid(self.neighbor_gate_proj(h)).view(b, t, self.n_head, self.half_dim)
        k = self._gated_neighbor(k, gate)

        if self.paired:
            # Merge adjacent head pairs: (b, t, n_head, hd) -> (b, t, n_head//2, hd*2)
            n2 = self.n_head // 2
            q = q.view(b, t, n2, self.head_dim * 2)
            k = k.view(b, t, n2, self.head_dim * 2)

            # Paired RoPE (even/odd positions for each half)
            cos, sin = self.rotary(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

            # Unmerge to doubled sequence: (b, t, n2, hd*2) -> (b, t*2, n2, hd)
            q = q.view(b, t * 2, n2, self.head_dim)
            k = k.view(b, t * 2, n2, self.head_dim)
            v = v.reshape(b, t * 2, n2, self.head_dim)

            # Attention on interleaved 2T sequence
            if HAS_FLASH_ATTN:
                y = flash_attn_func(q, k, v, causal=True)
            else:
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                y = y.transpose(1, 2)

            # Reshape back: (b, t*2, n2, hd) -> (b, t, n_head, hd)
            y = y.contiguous().view(b, t, self.n_head, self.head_dim)
        else:
            # Standard RoPE
            cos, sin = self.rotary(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

            # Attention
            if HAS_FLASH_ATTN:
                y = flash_attn_func(q, k, v, causal=True)
            else:
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                y = y.transpose(1, 2)

        y = y.contiguous().view(b, t, self.inner_dim)

        # Skip-multiply
        y = self.attn_norm(y) * h_up

        # Down-project (with Swish)
        y = self.down_proj(y * torch.sigmoid(self.swish_beta_down * y))

        return y


class TransformerShiftBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = ShiftAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("ts/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("ts/block_mlp"):
            x = x + self.mlp(self.ln2(x))
        return x


class MLP(nn.Module):
    """SwiGLU MLP: gate_proj + up_proj → SiLU gating → down_proj."""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        # Snap to multiple of 256 for hardware efficiency
        hidden = ((hidden + 255) // 256) * 256
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("tf/mlp"):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FeatureAttentionMLP(nn.Module):
    """SwiGLU MLP with feature attention replacing element-wise gating.

    SwiGLU:         out = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    Feature Attn:   out = down_proj(feat_attn(gate_proj(x), up_proj(x)))

    gate_proj provides shared Q=K (Reformer-style), up_proj provides V.
    Attention is over n_features (non-causal, within each token).
    Same three projections, same param count as SwiGLU.

    Args:
        d_model:    Input/output dimension.
        n_features: Number of features to attend over.
        activation: Attention activation — 'softmax', 'silu', or 'silu2'.
    """

    def __init__(self, d_model: int, n_features: int, activation: str = 'softmax',
                 post_act: str = 'none', pre_act: str = 'none', qk_norm: bool = False):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        assert hidden % n_features == 0, (
            f"MLP hidden dim ({hidden}) must be divisible by n_features ({n_features})")
        self.n_features = n_features
        self.desc_dim = hidden // n_features
        self.hidden = hidden
        self.activation = activation
        self.post_act = post_act
        self.pre_act = pre_act
        self.qk_norm = qk_norm
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)  # → shared Q=K
        self.up_proj = nn.Linear(d_model, hidden, bias=False)    # → V
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        if qk_norm:
            self.qk_rms_norm = RMSNorm(self.desc_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("fa/mlp"):
            B, T, D = x.shape
            NF = self.n_features
            DD = self.desc_dim

            # gate_proj → Q=K, up_proj → V
            qk = self.gate_proj(x).view(B * T, NF, DD)  # shared Q and K
            v = self.up_proj(x).view(B * T, NF, DD)

            # Pre-attention activation on gate (restores SiLU from SwiGLU gate path)
            if self.pre_act == 'silu':
                qk = F.silu(qk)
            elif self.pre_act == 'gelu':
                qk = F.gelu(qk)
            elif self.pre_act != 'none':
                raise ValueError(f"Unknown FA_PRE_ACT: {self.pre_act}")

            # QK normalization (cosine similarity instead of raw dot product)
            if self.qk_norm:
                qk = self.qk_rms_norm(qk)

            # Attention scores from gate (Q=K, Reformer-style)
            scale = DD ** -0.5
            scores = torch.bmm(qk, qk.transpose(-2, -1)) * scale  # (B*T, NF, NF)

            # Activation
            if self.activation == 'softmax':
                weights = torch.softmax(scores, dim=-1)
            elif self.activation == 'silu':
                weights = F.silu(scores)
            elif self.activation == 'silu2':
                weights = F.silu(scores).square()
            else:
                raise ValueError(f"Unknown FA_ACTIVATION: {self.activation}")

            # Attend over V (from up_proj)
            out = torch.bmm(weights, v)  # (B*T, NF, DD)
            out = out.view(B, T, self.hidden)

            # Post-attention element-wise activation
            if self.post_act == 'silu':
                out = F.silu(out)
            elif self.post_act == 'gelu':
                out = F.gelu(out)
            elif self.post_act != 'none':
                raise ValueError(f"Unknown FA_POST_ACT: {self.post_act}")

            return self.down_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("tf/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("tf/block_mlp"):
            x = x + self.mlp(self.ln2(x))
        return x


class TransformerFeatureAttnBlock(nn.Module):
    """Transformer block with feature attention MLP (replacing GELU with feature self-attn)."""

    def __init__(self, d_model: int, n_head: int, n_features: int, activation: str,
                 post_act: str = 'none', pre_act: str = 'none', qk_norm: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = FeatureAttentionMLP(d_model, n_features, activation, post_act=post_act, pre_act=pre_act, qk_norm=qk_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("fa/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("fa/block_mlp"):
            x = x + self.mlp(self.ln2(x))
        return x


class ThreeStageBlock(nn.Module):
    """Three-stage block: seq attention → feature attention → MLP.

    Stage 1 — Sequence attention (tokens talk to each other):
      q, k, v = Q/K/V_proj(x)       # standard projections
      seq_out = attn(q, k, v) + x    # causal, RoPE, residual

    Stage 2 — Feature attention (token talks to itself):
      q_feat = q_pre_rope.view(B*T, N_f, D_f)   # reuse Q before RoPE
      k_feat = k_pre_rope.view(B*T, N_f, D_f)   # reuse K before RoPE
      v_feat = seq_out.view(B*T, N_f, D_f)       # attend over seq_out
      feat_out = feat_attn(q_feat, k_feat, v_feat)
      feat_out = SiLU(feat_out) + seq_out         # activate + residual

    Stage 3 — MLP (enrichment):
      out = SwiGLU(feat_out) + feat_out           # standard MLP, residual

    Same params as standard transformer — feature attention adds zero projections.
    """

    def __init__(self, d_model: int, n_head: int, n_features: int,
                 feat_activation: str = 'softmax'):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.n_features = n_features
        self.desc_dim = d_model // n_features
        self.feat_activation = feat_activation

        assert d_model % n_features == 0, (
            f"d_model ({d_model}) must be divisible by n_features ({n_features})")

        # Stage 1: sequence attention (standard)
        self.ln1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

        # Stage 2: feature attention (no new projections)
        self.ln2 = RMSNorm(d_model)

        # Stage 3: MLP (standard SwiGLU)
        self.ln3 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_head
        HD = self.head_dim
        NF = self.n_features
        DD = self.desc_dim

        # ── Stage 1: Sequence attention ──
        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, NH, HD)
        k = self.k_proj(h).view(B, T, NH, HD)
        v = self.v_proj(h).view(B, T, NH, HD)

        q_pre_rope = q  # save for feature attention
        k_pre_rope = k  # save for feature attention

        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if HAS_FLASH_ATTN:
            seq_out = flash_attn_func(q, k, v, causal=True)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            seq_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            seq_out = seq_out.transpose(1, 2)

        seq_out = self.out_proj(seq_out.contiguous().view(B, T, D))
        x = x + seq_out  # residual

        # ── Stage 2: Feature attention ──
        h2 = self.ln2(x)
        q_feat = q_pre_rope.contiguous().view(B * T, NF, DD)
        k_feat = k_pre_rope.contiguous().view(B * T, NF, DD)
        v_feat = h2.view(B * T, NF, DD)

        scale = DD ** -0.5
        scores = torch.bmm(q_feat, k_feat.transpose(-2, -1)) * scale

        if self.feat_activation == 'softmax':
            weights = torch.softmax(scores, dim=-1)
        elif self.feat_activation == 'silu':
            weights = F.silu(scores)
        elif self.feat_activation == 'silu2':
            weights = F.silu(scores).square()
        else:
            raise ValueError(f"Unknown feat_activation: {self.feat_activation}")

        feat_out = torch.bmm(weights, v_feat)
        feat_out = F.silu(feat_out.view(B, T, D))
        x = x + feat_out  # residual

        # ── Stage 3: MLP ──
        x = x + self.mlp(self.ln3(x))

        return x


class ThreeStageFSABlock(nn.Module):
    """Three-stage block: feature attention → seq attention → MLP.

    Compute Q/K/V projections once from ln1(x), then:

    Stage 1 — Feature attention (reorganize own features):
      q_feat = Q.view(B*T, N_f, D_f)
      k_feat = K.view(B*T, N_f, D_f)
      v_feat = ln1(x).view(B*T, N_f, D_f)
      feat_out = SiLU(feat_attn(q_feat, k_feat, v_feat)) + x

    Stage 2 — Sequence attention (tokens talk with reorganized repr):
      Apply RoPE to Q, K from the SAME projections
      seq_out = attn(q, k, v) + feat_out

    Stage 3 — MLP (enrichment):
      out = SwiGLU(seq_out) + seq_out

    Zero extra params vs baseline — same as ThreeStageBlock, just stages 1 and 2 swapped.
    """

    def __init__(self, d_model: int, n_head: int, n_features: int,
                 feat_activation: str = 'softmax'):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.n_features = n_features
        self.desc_dim = d_model // n_features
        self.feat_activation = feat_activation

        assert d_model % n_features == 0, (
            f"d_model ({d_model}) must be divisible by n_features ({n_features})")

        # Projections shared by both feature and sequence attention
        self.ln1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

        # MLP
        self.ln3 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_head
        HD = self.head_dim
        NF = self.n_features
        DD = self.desc_dim

        # Project Q/K/V once, use for both attentions
        h = self.ln1(x)
        q = self.q_proj(h)  # (B, T, D)
        k = self.k_proj(h)  # (B, T, D)
        v = self.v_proj(h)  # (B, T, D)

        # ── Stage 1: Feature attention — reshape as (N_f, D_f) ──
        q_feat = q.view(B * T, NF, DD)
        k_feat = k.view(B * T, NF, DD)
        v_feat = v.view(B * T, NF, DD)

        scale = DD ** -0.5
        scores = torch.bmm(q_feat, k_feat.transpose(-2, -1)) * scale

        if self.feat_activation == 'softmax':
            weights = torch.softmax(scores, dim=-1)
        elif self.feat_activation == 'silu':
            weights = F.silu(scores)
        elif self.feat_activation == 'silu2':
            weights = F.silu(scores).square()
        else:
            raise ValueError(f"Unknown feat_activation: {self.feat_activation}")

        feat_out = torch.bmm(weights, v_feat)
        feat_out = F.silu(feat_out.view(B, T, D))
        x = x + feat_out  # residual

        # ── Stage 2: Sequence attention — reshape as (N_heads, H_dim), apply RoPE ──
        q_seq = q.view(B, T, NH, HD)
        k_seq = k.view(B, T, NH, HD)
        v_seq = v.view(B, T, NH, HD)

        cos, sin = self.rotary(q_seq)
        q_seq = apply_rotary(q_seq, cos, sin)
        k_seq = apply_rotary(k_seq, cos, sin)

        if HAS_FLASH_ATTN:
            seq_out = flash_attn_func(q_seq, k_seq, v_seq, causal=True)
        else:
            q_seq, k_seq, v_seq = q_seq.transpose(1, 2), k_seq.transpose(1, 2), v_seq.transpose(1, 2)
            seq_out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True)
            seq_out = seq_out.transpose(1, 2)

        seq_out = self.out_proj(seq_out.contiguous().view(B, T, D))
        x = x + seq_out  # residual

        # ── Stage 3: MLP ──
        x = x + self.mlp(self.ln3(x))

        return x


class FusedSeqFeatureAttnBlock(nn.Module):
    """Fused block: shared QK/V projections → seq attention → feature attention → down.

    Single set of projections serves both sequence-level and feature-level attention:
      qk = SiLU(QK_proj(x))     # (B, T, hidden) — shared Q=K for both attentions
      v  = V_proj(x)             # (B, T, hidden)

      # 1. Sequence attention over T (causal, with RoPE)
      qk_seq = qk.view(B, T, n_heads, head_dim)
      v_seq  = v.view(B, T, n_heads, head_dim)
      out = seq_attn(qk_seq, qk_seq, v_seq)  # Q=K, attend over T

      # 2. Feature attention over N_f (non-causal, within each token)
      out_feat = out.view(B*T, N_f, D_f)
      qk_feat  = qk.view(B*T, N_f, D_f)
      out = feat_attn(qk_feat, qk_feat, out_feat)  # Q=K, attend over N_f

      out = SiLU(out)
      out = down_proj(out)

    Params: QK_proj(d→H) + V_proj(d→H) + down(H→d) = 3*d*H
    vs standard block: Q+K+V+O(4*d²) + gate+up+down(3*d*H) ≈ 7.1M → 4.7M at d=768,H=2048
    """

    def __init__(self, d_model: int, hidden: int, n_seq_heads: int,
                 n_features: int, feat_activation: str = 'softmax'):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_seq_heads = n_seq_heads
        self.seq_head_dim = hidden // n_seq_heads
        self.n_features = n_features
        self.desc_dim = hidden // n_features
        self.feat_activation = feat_activation

        assert hidden % n_seq_heads == 0, f"hidden ({hidden}) must be divisible by n_seq_heads ({n_seq_heads})"
        assert hidden % n_features == 0, f"hidden ({hidden}) must be divisible by n_features ({n_features})"

        self.ln = RMSNorm(d_model)
        self.qk_proj = nn.Linear(d_model, hidden, bias=False)
        self.v_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.rotary = Rotary(self.seq_head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_seq_heads
        HD = self.seq_head_dim
        NF = self.n_features
        DD = self.desc_dim
        H = self.hidden

        h = self.ln(x)

        # Shared projections
        qk = F.silu(self.qk_proj(h))  # (B, T, H) — pre-activated gate
        v = self.v_proj(h)              # (B, T, H)

        # ── Sequence attention (causal, RoPE, Q=K) ──
        qk_seq = qk.view(B, T, NH, HD)
        v_seq = v.view(B, T, NH, HD)
        cos, sin = self.rotary(qk_seq)
        q_seq = apply_rotary(qk_seq, cos, sin)
        k_seq = apply_rotary(qk_seq, cos, sin)  # Q=K, same RoPE

        if HAS_FLASH_ATTN:
            out = flash_attn_func(q_seq, k_seq, v_seq, causal=True)
        else:
            q_seq = q_seq.transpose(1, 2)
            k_seq = k_seq.transpose(1, 2)
            v_seq = v_seq.transpose(1, 2)
            out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True)
            out = out.transpose(1, 2)

        out = out.contiguous().view(B, T, H)

        # ── Feature attention (non-causal, within each token) ──
        qk_feat = qk.view(B * T, NF, DD)
        out_feat = out.view(B * T, NF, DD)

        scale = DD ** -0.5
        scores = torch.bmm(qk_feat, qk_feat.transpose(-2, -1)) * scale

        if self.feat_activation == 'softmax':
            weights = torch.softmax(scores, dim=-1)
        elif self.feat_activation == 'silu':
            weights = F.silu(scores)
        elif self.feat_activation == 'silu2':
            weights = F.silu(scores).square()
        else:
            raise ValueError(f"Unknown feat_activation: {self.feat_activation}")

        out = torch.bmm(weights, out_feat)  # (B*T, NF, DD)
        out = F.silu(out.view(B, T, H))     # post-activation
        out = self.down_proj(out)

        return x + out


class FusedQKVBlock(nn.Module):
    """Fused block with full Q/K/V projections shared across seq + feature attention.

    Flow:
      q = Q_proj(x)                          # (B, T, H)
      k = K_proj(x)                          # (B, T, H)
      v = V_proj(x)                          # (B, T, H)

      seq_out = seq_attn(q, k, v)            # causal, RoPE, over T
      seq_out = seq_out + v                   # skip from V around seq attn

      feat_out = feat_attn(q, q, seq_out)    # Q=K from q, over N_f
      out = SiLU(feat_out)
      out = down_proj(out)

    Params: Q(d,H) + K(d,H) + V(d,H) + down(H,d) = 4*d*H
    At d=768, H=2304: 124M params with 12 layers.
    """

    def __init__(self, d_model: int, hidden: int, n_seq_heads: int,
                 n_features: int, feat_activation: str = 'softmax'):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.n_seq_heads = n_seq_heads
        self.seq_head_dim = hidden // n_seq_heads
        self.n_features = n_features
        self.desc_dim = hidden // n_features
        self.feat_activation = feat_activation

        assert hidden % n_seq_heads == 0, f"hidden ({hidden}) must be divisible by n_seq_heads ({n_seq_heads})"
        assert hidden % n_features == 0, f"hidden ({hidden}) must be divisible by n_features ({n_features})"

        self.ln = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, hidden, bias=False)
        self.k_proj = nn.Linear(d_model, hidden, bias=False)
        self.v_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.rotary = Rotary(self.seq_head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        NH = self.n_seq_heads
        HD = self.seq_head_dim
        NF = self.n_features
        DD = self.desc_dim
        H = self.hidden

        h = self.ln(x)

        # Full Q, K, V projections
        q = self.q_proj(h)   # (B, T, H)
        k = self.k_proj(h)   # (B, T, H)
        v = self.v_proj(h)   # (B, T, H)

        # ── Sequence attention (causal, RoPE, separate Q/K) ──
        q_seq = q.view(B, T, NH, HD)
        k_seq = k.view(B, T, NH, HD)
        v_seq = v.view(B, T, NH, HD)
        cos, sin = self.rotary(q_seq)
        q_seq = apply_rotary(q_seq, cos, sin)
        k_seq = apply_rotary(k_seq, cos, sin)

        if HAS_FLASH_ATTN:
            seq_out = flash_attn_func(q_seq, k_seq, v_seq, causal=True)
        else:
            q_seq = q_seq.transpose(1, 2)
            k_seq = k_seq.transpose(1, 2)
            v_seq = v_seq.transpose(1, 2)
            seq_out = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True)
            seq_out = seq_out.transpose(1, 2)

        seq_out = seq_out.contiguous().view(B, T, H)

        # Skip connection: V around seq attention
        seq_out = seq_out + v

        # ── Feature attention (non-causal, Q=K from q) ──
        q_feat = q.view(B * T, NF, DD)
        seq_feat = seq_out.view(B * T, NF, DD)

        scale = DD ** -0.5
        scores = torch.bmm(q_feat, q_feat.transpose(-2, -1)) * scale

        if self.feat_activation == 'softmax':
            weights = torch.softmax(scores, dim=-1)
        elif self.feat_activation == 'silu':
            weights = F.silu(scores)
        elif self.feat_activation == 'silu2':
            weights = F.silu(scores).square()
        else:
            raise ValueError(f"Unknown feat_activation: {self.feat_activation}")

        out = torch.bmm(weights, seq_feat)  # (B*T, NF, DD)
        out = F.silu(out.view(B, T, H))     # post-activation
        out = self.down_proj(out)

        return x + out


class TransformerMamba3Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln0 = RMSNorm(d_model)
        self.mamba3 = Mamba3Mixer(d_model, n_head)
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("tm3/block_mamba3"):
            x = x + self.mamba3(self.ln0(x))
        with torch.autograd.profiler.record_function("tm3/block_attn"):
            x = x + self.attn(self.ln1(x))
        with torch.autograd.profiler.record_function("tm3/block_mlp"):
            x = x + self.mlp(self.ln2(x))
        return x


def chunked_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute cross-entropy without materializing full (B*T, vocab) logits.

    Processes chunk_size tokens at a time through the LM head and loss,
    avoiding the ~6 GB logits tensor at batch 32 / seqlen 2048 / vocab 50k.
    """
    B, T, D = hidden.shape
    hidden_flat = hidden.reshape(-1, D)       # (B*T, D)
    targets_flat = targets.reshape(-1)        # (B*T,)
    total_tokens = B * T
    loss_sum = hidden.new_zeros(())
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        logits_chunk = F.linear(hidden_flat[start:end], weight)  # (chunk, vocab)
        loss_sum = loss_sum + F.cross_entropy(logits_chunk, targets_flat[start:end], reduction="sum")
    return loss_sum / total_tokens


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


class GPTTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTTransformerConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([TransformerMamba3Block(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTTransformerShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([TransformerShiftBlock(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTGatedNeighbor(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([GatedNeighborBlock(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)
        # Re-apply gate init after _init_weights (which overwrites it)
        for block in self.blocks:
            nn.init.zeros_(block.attn.gate_proj.weight)
            nn.init.constant_(block.attn.gate_proj.bias, -2.0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTFusedGatedNeighbor(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        inner = HP.fused_inner_dim if HP.fused_inner_dim > 0 else None
        paired_str = os.environ.get("PAIRED_HEAD_LAYERS", "0,2,5,9")
        paired_layers = set(int(x) for x in paired_str.split(",") if x.strip()) if paired_str.strip() else set()
        self.blocks = nn.ModuleList([
            FusedGatedNeighborBlock(HP.d_model, HP.n_head, inner_dim=inner, paired=(i in paired_layers))
            for i in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)
        # Re-apply gate init after _init_weights (which overwrites it)
        for block in self.blocks:
            nn.init.zeros_(block.neighbor_gate_proj.weight)
            nn.init.constant_(block.neighbor_gate_proj.bias, -2.0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = x + block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTS6(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        modes = tuple(x.strip() for x in HP.s6_scan_state_modes.split(",") if x.strip())
        if len(modes) != 3:
            raise ValueError("S6_SCAN_STATE_MODES must have 3 comma-separated values")
        cfg = USBConfig(
            d_model=HP.d_model,
            headdim=HP.s6_headdim,
            expansion_factor=HP.s6_expansion_factor,
            post_scan_attention=HP.s6_post_scan_attention,
            scan_state_modes=modes,
        )
        self.blocks = nn.ModuleList([USBBlock(cfg) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        x = self.ln_f(x)
        if targets is not None:
            loss = chunked_cross_entropy(x, self.lm_head.weight, targets)
            return None, loss
        logits = self.lm_head(x)
        return logits, None


class GPTULB(nn.Module):
    """ULBBlendP wrapper for fineweb training (sigmoid gate, no feat-attn)."""
    def __init__(self, feat_attn: bool = False):
        super().__init__()
        from ulb.transformer import CausalULB
        self.inner = CausalULB(
            vocab_size=HP.vocab_size,
            dim=HP.d_model,
            n_heads=HP.n_head,
            n_layers=HP.n_layer,
            max_seq_len=HP.seq_len,
            paired=True,
            attn_mode='blend',
            feat_attn=feat_attn,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.inner(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTULB1D(nn.Module):
    """Clean 1D ULB with silu² feat-attn MLP (for ablating 2D vs 1D)."""
    def __init__(self):
        super().__init__()
        from ulb.transformer import CausalULB1D
        self.inner = CausalULB1D(
            vocab_size=HP.vocab_size,
            d_model=HP.d_model,
            n_heads=HP.n_head,
            n_layers=HP.n_layer,
            max_seq_len=HP.seq_len,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.inner(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTULB2D(nn.Module):
    """True 2D ULB wrapper for fineweb training."""
    def __init__(self):
        super().__init__()
        from ulb.transformer import CausalULB2D
        nf, dd = HP.n_features, HP.desc_dim
        if nf == 0 or dd == 0:
            # Auto-factor from d_model, targeting 1:3 ratio (n_features:desc_dim)
            dim = HP.d_model
            target_nf = int((dim / 3) ** 0.5)
            best = 1
            for s in range(1, int(dim ** 0.5) + 1):
                if dim % s == 0 and (dim // s) % 4 == 0:
                    if abs(s - target_nf) <= abs(best - target_nf):
                        best = s
            nf, dd = best, dim // best
        self.inner = CausalULB2D(
            vocab_size=HP.vocab_size,
            n_features=nf,
            desc_dim=dd,
            n_layers=HP.n_layer,
            max_seq_len=HP.seq_len,
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.inner(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTFeatureAttn(nn.Module):
    """GPT with feature attention MLP — SwiGLU gating replaced by feature self-attention.

    Same three projections (gate/up/down), same param count as GPTTransformer.
    Configured via FA_N_FEATURES and FA_ACTIVATION.
    """

    def __init__(self):
        super().__init__()
        nf = HP.fa_n_features
        act = HP.fa_activation
        hidden = int(HP.d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        assert hidden % nf == 0, (
            f"MLP hidden dim ({hidden}) not divisible by FA_N_FEATURES ({nf}). "
            f"Valid: {[f for f in [8, 16, 32, 64, 128, 256] if hidden % f == 0]}")
        post_act = HP.fa_post_act
        pre_act = HP.fa_pre_act
        qk_norm = HP.fa_qk_norm
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([
            TransformerFeatureAttnBlock(HP.d_model, HP.n_head, nf, act, post_act=post_act, pre_act=pre_act, qk_norm=qk_norm)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTFusedSeqFeature(nn.Module):
    """GPT with fused seq+feature attention blocks — shared QK/V projections."""

    def __init__(self):
        super().__init__()
        if HP.fused_inner_dim > 0:
            hidden = HP.fused_inner_dim
        else:
            hidden = int(HP.d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        nf = HP.fa_n_features
        act = HP.fa_activation
        # n_seq_heads: use head_dim=64 by default
        n_seq_heads = hidden // 64
        assert hidden % nf == 0, (
            f"hidden ({hidden}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([
            FusedSeqFeatureAttnBlock(HP.d_model, hidden, n_seq_heads, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTThreeStage(nn.Module):
    """GPT with three-stage blocks: seq attn → feat attn → MLP. Same params as baseline."""

    def __init__(self):
        super().__init__()
        nf = HP.fa_n_features
        act = HP.fa_activation
        assert HP.d_model % nf == 0, (
            f"d_model ({HP.d_model}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([
            ThreeStageBlock(HP.d_model, HP.n_head, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTThreeStageFSA(nn.Module):
    """GPT with three-stage blocks: feat attn → seq attn → MLP.

    Reversed order from GPTThreeStage to test whether reorganizing features
    before inter-token communication helps. Zero extra params — reuses Q
    projection for both feature attention (pre-RoPE Q as feat Q=K) and
    sequence attention (re-projected from post-feat-attn state + RoPE).
    """

    def __init__(self):
        super().__init__()
        nf = HP.fa_n_features
        act = HP.fa_activation
        assert HP.d_model % nf == 0, (
            f"d_model ({HP.d_model}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([
            ThreeStageFSABlock(HP.d_model, HP.n_head, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class GPTFusedQKV(nn.Module):
    """GPT with fused QKV blocks — full Q/K/V shared across seq + feature attention."""

    def __init__(self):
        super().__init__()
        if HP.fused_inner_dim > 0:
            hidden = HP.fused_inner_dim
        else:
            hidden = int(HP.d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256
        nf = HP.fa_n_features
        act = HP.fa_activation
        n_seq_heads = hidden // 64
        assert hidden % nf == 0, (
            f"hidden ({hidden}) not divisible by FA_N_FEATURES ({nf})")
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([
            FusedQKVBlock(HP.d_model, hidden, n_seq_heads, nf, act)
            for _ in range(HP.n_layer)
        ])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(_init_weights)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def build_model() -> nn.Module:
    if HP.model_type == "transformer":
        return GPTTransformer()
    if HP.model_type == "feat_attn":
        return GPTFeatureAttn()
    if HP.model_type == "fused_seq_feat":
        return GPTFusedSeqFeature()
    if HP.model_type == "fused_qkv":
        return GPTFusedQKV()
    if HP.model_type == "three_stage":
        return GPTThreeStage()
    if HP.model_type == "three_stage_fsa":
        return GPTThreeStageFSA()
    if HP.model_type == "transformer_shift":
        return GPTTransformerShift()
    if HP.model_type == "transformer_s4d":
        return GPTTransformerConv()
    if HP.model_type == "transformer_gate":
        return GPTGatedNeighbor()
    if HP.model_type == "fused_gate":
        return GPTFusedGatedNeighbor()
    if HP.model_type == "s6":
        return GPTS6()
    if HP.model_type == "ulb":
        return GPTULB(feat_attn=False)
    if HP.model_type == "ulb_fa":
        return GPTULB1D()
    if HP.model_type == "ulb_2d":
        return GPTULB2D()
    raise ValueError(f"Unknown MODEL_TYPE={HP.model_type}")


def lr_for_step(step: int) -> float:
    # Warmup phase (same for all schedules)
    if step < HP.warmup_steps:
        return HP.lr * (step + 1) / max(1, HP.warmup_steps)

    if HP.lr_schedule == "wsd":
        # Warmup-Stable-Decay: hold peak LR, then cosine decay in final fraction
        decay_steps = int(HP.train_steps * HP.wsd_decay_frac)
        stable_end = HP.train_steps - decay_steps
        if step <= stable_end:
            return HP.lr
        # Cosine decay from peak to ~0 over the decay phase
        t = (step - stable_end) / max(1, decay_steps)
        return HP.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    else:
        # Default: cosine decay from warmup end to train_steps
        t = (step - HP.warmup_steps) / max(1, HP.train_steps - HP.warmup_steps)
        return HP.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))


@torch.no_grad()
def evaluate(model: nn.Module, val_stream: ShardStream, device: torch.device, world_size: int) -> float:
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    for _ in range(HP.val_steps):
        x, y = val_stream.next_batch(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss_sum += loss
    loss_sum /= HP.val_steps
    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
    model.train()
    return float(loss_sum.item())


def main():
    rank, world_size, device = setup_dist()
    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print0(rank, f"rank={rank} world_size={world_size} device={device}")
    _extra = f" n_features={HP.n_features} desc_dim={HP.desc_dim}" if HP.n_features > 0 and HP.desc_dim > 0 else ""
    if HP.model_type == "feat_attn":
        hidden = ((int(HP.d_model * 8 / 3) + 255) // 256) * 256
        _extra += f" fa_n_features={HP.fa_n_features} fa_desc_dim={hidden // HP.fa_n_features} fa_activation={HP.fa_activation} fa_pre_act={HP.fa_pre_act} fa_post_act={HP.fa_post_act} fa_qk_norm={HP.fa_qk_norm}"
    _sched = f"lr_schedule={HP.lr_schedule}"
    if HP.lr_schedule == "wsd":
        _sched += f" decay_frac={HP.wsd_decay_frac}"
    print0(rank, f"model_type={HP.model_type} layers={HP.n_layer} heads={HP.n_head} d_model={HP.d_model} seq_len={HP.seq_len}{_extra} {_sched}")

    train_stream = ShardStream(HP.train_files, rank, world_size, HP.seq_len, HP.batch_size)
    val_stream = ShardStream(HP.val_files, rank, world_size, HP.seq_len, HP.batch_size)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print0(rank, f"parameters={n_params:,}")

    # Build optimizer param groups before compile/DDP wrap
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or getattr(p, "_no_weight_decay", False):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    print0(rank, f"optimizer: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")
    param_groups = [
        {"params": decay_params, "weight_decay": HP.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if HP.compile:
        model = torch.compile(model, dynamic=False)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    optimizer = torch.optim.AdamW(param_groups, lr=HP.lr, betas=(0.9, 0.95))

    profiler = None
    if HP.torch_profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        profiler.start()
        print0(rank, f"torch profiler enabled for {HP.torch_profile_steps} train steps")

    t0 = time.time()
    for step in range(HP.train_steps + 1):
        if step % HP.val_every == 0 or step == HP.train_steps:
            val_loss = evaluate(model, val_stream, device, world_size)
            print0(rank, f"step {step:5d} | val_loss {val_loss:.5f}")
            if step == HP.train_steps:
                break

        lr = lr_for_step(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = train_stream.next_batch(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if HP.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(decay_params + no_decay_params, HP.grad_clip)
        optimizer.step()

        if step % 20 == 0:
            loss_t = loss.detach()
            if world_size > 1:
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            dt = (time.time() - t0) / max(1, step + 1)
            print0(rank, f"step {step:5d} | train_loss {loss_t.item():.5f} | lr {lr:.3e} | sec/step {dt:.3f}")

        if profiler is not None:
            profiler.step()
            if step + 1 >= HP.torch_profile_steps:
                profiler.stop()
                if rank == 0:
                    print("\n=== Torch Profile: CUDA Time ===", flush=True)
                    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=40), flush=True)
                    print("\n=== Torch Profile: CUDA Memory ===", flush=True)
                    print(profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=40), flush=True)
                profiler = None

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
