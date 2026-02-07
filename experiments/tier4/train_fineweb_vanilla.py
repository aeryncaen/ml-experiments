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

    model_type: str = os.environ.get("MODEL_TYPE", "transformer")  # transformer | transformer_shift | transformer_gate | fused_gate | transformer_s4d | s6
    vocab_size: int = 50304
    n_layer: int = _env_int("N_LAYER", 12)
    n_head: int = _env_int("N_HEAD", 12)
    d_model: int = _env_int("D_MODEL", 768)
    seq_len: int = _env_int("SEQ_LEN", 2048)

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

    def __init__(self, d_model: int, n_head: int, inner_dim: int | None = None, expand: int = 1, layer_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.inner_dim = inner_dim if inner_dim is not None else d_model * expand
        assert self.inner_dim % n_head == 0
        self.head_dim = self.inner_dim // n_head
        self.sub_dim = self.head_dim // 2  # Q/K split dim for diff attention
        self.half_dim = self.sub_dim // 2  # neighbor gate split within sub_dim

        self.norm = RMSNorm(d_model)

        # Shared expansion: d_model -> inner_dim
        self.up_proj = nn.Linear(d_model, self.inner_dim, bias=False)

        # QKV projections from expanded space
        self.q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)

        # Differential attention lambda (per-layer)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.randn(self.sub_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.sub_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.sub_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.sub_dim) * 0.1)

        # Per-head RMSNorm after diff attention
        self.head_norm = RMSNorm(self.head_dim)

        # Post-attention norm (before skip add)
        self.attn_norm = RMSNorm(self.inner_dim)

        # Down-project: inner_dim -> d_model
        self.down_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        # Learnable Swish beta (per-channel)
        self.swish_beta_up = nn.Parameter(torch.ones(self.inner_dim))
        self.swish_beta_down = nn.Parameter(torch.ones(self.inner_dim))

        # QK norm (per-head RMSNorm before RoPE)
        self.q_norm = RMSNorm(self.sub_dim)
        self.k_norm = RMSNorm(self.sub_dim)

        # RoPE on sub_dim (applied to both Q1/Q2, K1/K2)
        self.rotary = Rotary(self.sub_dim)

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

        # QKV from activated expanded space — split Q,K into two sub-heads
        q = self.q_proj(h_up).view(b, t, self.n_head, 2, self.sub_dim)
        k = self.k_proj(h_up).view(b, t, self.n_head, 2, self.sub_dim)
        v = self.v_proj(h_up).view(b, t, self.n_head, self.head_dim)

        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]  # (b, t, n_head, sub_dim)
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]

        # QK norm then RoPE on each sub-head
        q1, q2 = self.q_norm(q1), self.q_norm(q2)
        k1, k2 = self.k_norm(k1), self.k_norm(k2)
        cos, sin = self.rotary(q1)
        q1 = apply_rotary(q1, cos, sin)
        q2 = apply_rotary(q2, cos, sin)
        k1 = apply_rotary(k1, cos, sin)
        k2 = apply_rotary(k2, cos, sin)

        # Gated neighbor mixing on K (both sub-heads share the same gate)
        gate = torch.sigmoid(self.neighbor_gate_proj(h)).view(b, t, self.n_head, self.half_dim)
        k1 = self._gated_neighbor(k1, gate)
        k2 = self._gated_neighbor(k2, gate)

        # Compute lambda
        lam = (torch.exp(self.lambda_q1 * self.lambda_k1).sum()
             - torch.exp(self.lambda_q2 * self.lambda_k2).sum()
             + self.lambda_init)

        # Differential attention: attn1 - lambda * attn2
        if HAS_FLASH_ATTN:
            a1 = flash_attn_func(q1, k1, v, causal=True)
            a2 = flash_attn_func(q2, k2, v, causal=True)
        else:
            q1t, k1t, q2t, k2t = q1.transpose(1, 2), k1.transpose(1, 2), q2.transpose(1, 2), k2.transpose(1, 2)
            vt = v.transpose(1, 2)
            a1 = F.scaled_dot_product_attention(q1t, k1t, vt, is_causal=True).transpose(1, 2)
            a2 = F.scaled_dot_product_attention(q2t, k2t, vt, is_causal=True).transpose(1, 2)

        # Diff + per-head norm + scale
        y = (1 - self.lambda_init) * self.head_norm(a1 - lam * a2)

        y = y.contiguous().view(b, t, self.inner_dim)

        # Skip-multiply
        y = self.attn_norm(y) * h_up

        # Down-project (with Swish)
        y = self.down_proj(y * torch.sigmoid(self.swish_beta_down * y))

        return x + y


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
    def __init__(self, d_model: int):
        super().__init__()
        hidden = int(4 * d_model)
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.profiler.record_function("tf/mlp"):
            return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


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
        self.blocks = nn.ModuleList([FusedGatedNeighborBlock(HP.d_model, HP.n_head, inner_dim=inner, layer_idx=i) for i in range(HP.n_layer)])
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


def build_model() -> nn.Module:
    if HP.model_type == "transformer":
        return GPTTransformer()
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
    raise ValueError(f"Unknown MODEL_TYPE={HP.model_type}")


def lr_for_step(step: int) -> float:
    if step < HP.warmup_steps:
        return HP.lr * (step + 1) / max(1, HP.warmup_steps)
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
    print0(rank, f"model_type={HP.model_type} layers={HP.n_layer} heads={HP.n_head} d_model={HP.d_model} seq_len={HP.seq_len}")

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
