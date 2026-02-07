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
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

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

    model_type: str = os.environ.get("MODEL_TYPE", "transformer")  # transformer | transformer_s4d | s6
    vocab_size: int = 50304
    n_layer: int = _env_int("N_LAYER", 12)
    n_head: int = _env_int("N_HEAD", 12)
    d_model: int = _env_int("D_MODEL", 768)
    seq_len: int = _env_int("SEQ_LEN", 2048)

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
    """Mamba-3-style mixer: trapezoidal SSD + data-dependent RoPE on B,C + BC bias.

    Implements the structured mask Y = (L_decay @ L_trap) ⊙ C B^T ) X
    where L_trap is the bidiagonal (β_t, γ_t) trapezoidal conv mask,
    and B,C get cumulative data-dependent rotary embeddings.

    Uses the SSD (State Space Duality) quadratic form for training —
    no explicit recurrence, just a masked matmul.
    """

    def __init__(self, d_model: int, n_head: int, d_state: int = 64):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head  # p
        self.d_state = d_state             # n

        # Input projections: x -> (X, B, C, dt, lambda, theta)
        # X: value, B: input proj, C: output proj, dt: timestep, lambda: trap weight, theta: rotation
        proj_dim = (
            d_model          # X  (H * p)
            + n_head * d_state  # B  (H * n)
            + n_head * d_state  # C  (H * n)
            + n_head            # log_dt (H)
            + n_head            # lambda_logit (H)
            + n_head * (d_state // 2)  # theta (H * n/2) for data-dependent RoPE
        )
        self.in_proj = nn.Linear(d_model, proj_dim, bias=False)

        # BC bias (Mamba-3 §3.4): learnable, head-specific, channel-wise, init=1
        self.b_bias = nn.Parameter(torch.ones(n_head, d_state))
        self.c_bias = nn.Parameter(torch.ones(n_head, d_state))
        self.b_bias._no_weight_decay = True   # type: ignore[attr-defined]
        self.c_bias._no_weight_decay = True   # type: ignore[attr-defined]

        # QK-norm on B,C (Mamba-3 §3.4)
        self.b_norm = nn.RMSNorm(d_state)
        self.c_norm = nn.RMSNorm(d_state)

        # log A (scalar decay per head, like Mamba-2)
        self.log_A = nn.Parameter(torch.log(0.5 * torch.ones(n_head)))
        self.log_A._no_weight_decay = True    # type: ignore[attr-defined]

        # Output
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        B, T, D = x.shape
        H, P, N = self.n_head, self.head_dim, self.d_state
        Nr = N // 2  # half state dim for RoPE pairs
        C = chunk_size
        assert T % C == 0, f"seq_len {T} must be divisible by chunk_size {C}"
        n_chunks = T // C

        # ---- Project input ----
        proj = self.in_proj(x)  # (B, T, proj_dim)
        idx = 0
        X = proj[..., idx:idx + H * P].view(B, T, H, P);          idx += H * P
        Bv = proj[..., idx:idx + H * N].view(B, T, H, N);         idx += H * N
        Cv = proj[..., idx:idx + H * N].view(B, T, H, N);         idx += H * N
        log_dt = proj[..., idx:idx + H].view(B, T, H);            idx += H
        lam_logit = proj[..., idx:idx + H].view(B, T, H);         idx += H
        theta = proj[..., idx:idx + H * Nr].view(B, T, H, Nr);    idx += H * Nr

        # ---- Discretization params ----
        dt = F.softplus(log_dt)                                    # (B, T, H)
        alpha = torch.exp(dt * self.log_A.exp().neg())             # (B, T, H) decay ∈ (0,1)
        lam = torch.sigmoid(lam_logit)                             # (B, T, H) trap weight

        # ---- QK-norm + BC bias ----
        Bv = self.b_norm(Bv) * self.b_bias                        # (B, T, H, N)
        Cv = self.c_norm(Cv) * self.c_bias                        # (B, T, H, N)

        # ---- Data-dependent RoPE on B,C ----
        cum_theta = theta.cumsum(dim=1)                            # (B, T, H, Nr)
        cos_th = cum_theta.cos()
        sin_th = cum_theta.sin()
        def apply_dd_rope(v: torch.Tensor) -> torch.Tensor:
            v1, v2 = v[..., :Nr], v[..., Nr:]
            return torch.cat([v1 * cos_th - v2 * sin_th, v1 * sin_th + v2 * cos_th], dim=-1)
        Bv = apply_dd_rope(Bv)
        Cv = apply_dd_rope(Cv)

        # ---- Trapezoidal coefficients ----
        gamma = lam * dt                                           # (B, T, H)
        beta = (1.0 - lam) * dt * alpha                            # (B, T, H)

        # ---- Reshape into chunks: (B, n_chunks, C, ...) ----
        def chunkify(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, n_chunks, C, *t.shape[2:])

        X_c = chunkify(X)           # (B, nc, C, H, P)
        Bv_c = chunkify(Bv)         # (B, nc, C, H, N)
        Cv_c = chunkify(Cv)         # (B, nc, C, H, N)
        alpha_c = chunkify(alpha)    # (B, nc, C, H)
        gamma_c = chunkify(gamma)    # (B, nc, C, H)
        beta_c = chunkify(beta)      # (B, nc, C, H)

        # ---- Intra-chunk: trapezoidal BX ----
        # BX[b,nc,c,h,p,n] = B[b,nc,c,h,n] * X[b,nc,c,h,p]  (outer product)
        BX = torch.einsum('bschn,bschp->bschpn', Bv_c, X_c)      # (B, nc, C, H, P, N)

        # Trapezoidal: BX_trap[c] = γ[c]*BX[c] + β[c]*BX[c-1]
        # For c=0 in each chunk, BX[c-1] comes from previous chunk's last position
        # We handle cross-chunk BX below; first do the within-chunk part
        # Permute to (B, nc, H, C, P, N) for easier indexing
        BX = BX.permute(0, 1, 3, 2, 4, 5)                         # (B, nc, H, C, P, N)
        g = gamma_c.permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)  # (B, nc, H, C, 1, 1)
        b = beta_c.permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)

        # Shifted BX: within chunk, BX_shifted[:,c] = BX[:,c-1] for c>0
        # For c=0, need last position of previous chunk
        BX_last = BX[:, :, :, -1:, :, :]                          # (B, nc, H, 1, P, N)
        # Shift: prev chunk's last -> current chunk's position 0
        BX_prev = F.pad(BX_last[:, :-1], (0, 0, 0, 0, 0, 0, 0, 0, 1, 0))  # (B, nc, H, 1, P, N)
        BX_shifted = torch.cat([BX_prev, BX[:, :, :, :-1, :, :]], dim=3)    # (B, nc, H, C, P, N)

        BX_trap = g * BX + b * BX_shifted                          # (B, nc, H, C, P, N)

        # ---- Intra-chunk decay mask: (C, C) per chunk ----
        log_alpha_c = alpha_c.clamp(min=1e-6).log()                # (B, nc, C, H)
        log_alpha_c = log_alpha_c.permute(0, 1, 3, 2)              # (B, nc, H, C)
        log_cum = log_alpha_c.cumsum(dim=-1)                       # (B, nc, H, C)
        # decay[i,j] = exp(log_cum[i] - log_cum[j]) for i >= j, within chunk
        decay_intra = (log_cum.unsqueeze(-1) - log_cum.unsqueeze(-2)).exp()  # (B, nc, H, C, C)
        causal = torch.tril(torch.ones(C, C, device=x.device))
        decay_intra = decay_intra * causal                         # (B, nc, H, C, C)

        # ---- Intra-chunk SSD matmul ----
        # states_intra = decay_intra @ BX_trap  -> (B, nc, H, C, P, N)
        states_intra = torch.matmul(
            decay_intra,                                           # (B, nc, H, C, C)
            BX_trap.reshape(B, n_chunks, H, C, P * N)              # (B, nc, H, C, P*N)
        ).reshape(B, n_chunks, H, C, P, N)

        # ---- Inter-chunk state passing ----
        # Each chunk produces a final state: h_chunk = sum over chunk of decayed BX_trap
        # chunk_total_decay = product of all alphas in the chunk = exp(sum log_alpha)
        chunk_total_decay = log_alpha_c.sum(dim=-1).exp()           # (B, nc, H)
        # Final state of each chunk = last row of states_intra
        # But we need the *recurrent* state that accumulates across chunks
        # h_{chunk} = chunk_total_decay * h_{chunk-1} + states_intra[:, :, :, -1]
        # states_intra[:,:,:,-1] is the state at the last position from intra-chunk only
        chunk_state_local = states_intra[:, :, :, -1, :, :]        # (B, nc, H, P, N)

        # Sequential scan across chunks (nc is small, e.g. 32 for T=2048,C=64)
        states_list = []
        h = torch.zeros(B, H, P, N, device=x.device, dtype=x.dtype)
        for i in range(n_chunks):
            # h is the carry state entering this chunk
            states_list.append(h)
            # Update carry: decay all positions in this chunk, then add local contribution
            h = chunk_total_decay[:, i, :].unsqueeze(-1).unsqueeze(-1) * h + chunk_state_local[:, i]
        # states_list[i] = carry state entering chunk i, shape (B, H, P, N)
        prev_states = torch.stack(states_list, dim=1)               # (B, nc, H, P, N)

        # ---- Add inter-chunk contribution ----
        # For each position (chunk i, pos j), the inter-chunk contribution is:
        #   decay_from_chunk_start_to_pos_j @ prev_states[i]
        # decay from chunk start to pos j = exp(log_cum[j])  (cumsum within chunk)
        decay_from_start = log_cum.exp().unsqueeze(-1).unsqueeze(-1)  # (B, nc, H, C, 1, 1)
        inter_contrib = decay_from_start * prev_states.unsqueeze(3)   # (B, nc, H, C, P, N)
        states_total = states_intra + inter_contrib                   # (B, nc, H, C, P, N)

        # ---- Contract with C ----
        Cv_c_h = Cv_c.permute(0, 1, 3, 2, 4)                      # (B, nc, H, C, N)
        Y = torch.einsum('bnhcpk,bnhck->bnhcp', states_total, Cv_c_h)  # (B, nc, H, C, P)

        # ---- Reshape to (B, T, D) ----
        Y = Y.permute(0, 1, 3, 2, 4).reshape(B, T, D)
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
        with torch.autograd.profiler.record_function("tf/attn_qkv_rope"):
            q = self.q(x).view(b, t, self.n_head, self.head_dim)
            k = self.k(x).view(b, t, self.n_head, self.head_dim)
            v = self.v(x).view(b, t, self.n_head, self.head_dim)
            cos, sin = self.rotary(q)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        with torch.autograd.profiler.record_function("tf/attn_sdpa"):
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


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
    if HP.model_type == "transformer_s4d":
        return GPTTransformerConv()
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
