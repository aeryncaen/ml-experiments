import glob
import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


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

    vocab_size: int = 50304
    n_layer: int = _env_int("N_LAYER", 12)
    n_head: int = _env_int("N_HEAD", 12)
    d_model: int = _env_int("D_MODEL", 768)
    seq_len: int = _env_int("SEQ_LEN", 2048)

    train_steps: int = _env_int("TRAIN_STEPS", 2000)
    batch_size: int = _env_int("BATCH_SIZE", 8)
    val_steps: int = _env_int("VAL_STEPS", 32)
    val_every: int = _env_int("VAL_EVERY", 100)
    lr: float = _env_float("LR", 3e-4)
    warmup_steps: int = _env_int("WARMUP_STEPS", 200)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.1)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)
    compile: bool = _env_bool("TORCH_COMPILE", True)

    geo_prebias_enable: bool = _env_bool("GEO_PREBIAS_ENABLE", False)
    geo_prebias_method: str = os.environ.get("GEO_PREBIAS_METHOD", "kl_bucket")
    geo_prebias_mtp_weights: str = os.environ.get("GEO_PREBIAS_MTP_WEIGHTS", "1.0,0.5,0.25")
    geo_prebias_blend: float = _env_float("GEO_PREBIAS_BLEND", 0.75)
    geo_prebias_max_tokens: int = _env_int("GEO_PREBIAS_MAX_TOKENS", 50_000_000)
    geo_prebias_chunk_tokens: int = _env_int("GEO_PREBIAS_CHUNK_TOKENS", 5_000_000)
    geo_prebias_cache_dir: str = os.environ.get("GEO_PREBIAS_CACHE_DIR", os.path.join(data_path, "cache", "geo_prebias_vanilla"))
    geo_prebias_force_recompute: bool = _env_bool("GEO_PREBIAS_FORCE_RECOMPUTE", False)


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


def _parse_mtp_weights(spec: str) -> list[float]:
    vals = [float(p.strip()) for p in spec.split(",") if p.strip()]
    return vals if vals else [1.0]


def _compute_kl_bucket_basis(tokens: np.ndarray, vocab_size: int, width: int, chunk_tokens: int = 5_000_000, eps: float = 1e-8) -> np.ndarray:
    if tokens.size < 3:
        return np.zeros((vocab_size, width), dtype=np.float32)
    counts = np.zeros((vocab_size, width), dtype=np.float64)
    cur = tokens[2:]
    prev1 = tokens[1:-1]
    prev2 = tokens[:-2]
    n = cur.size
    n_chunks = (n + chunk_tokens - 1) // chunk_tokens
    for ci in range(n_chunks):
        s = ci * chunk_tokens
        e = min((ci + 1) * chunk_tokens, n)
        cur_s = cur[s:e]
        prev1_s = prev1[s:e]
        prev2_s = prev2[s:e]
        context_id = prev1_s.astype(np.int64) + vocab_size * prev2_s.astype(np.int64)
        buckets = context_id % width
        np.add.at(counts, (cur_s, buckets), 1.0)
    col_sum = counts.sum(axis=0, keepdims=True)
    p_t_given_b = counts / np.maximum(col_sum, 1.0)
    token_sum = counts.sum(axis=1, keepdims=True)
    p_t = token_sum / np.maximum(np.sum(counts), 1.0)
    basis = np.log(p_t_given_b + eps) - np.log(p_t + eps)
    basis = basis - basis.mean(axis=0, keepdims=True)
    col_norm = np.linalg.norm(basis, axis=0, keepdims=True)
    basis = basis / np.maximum(col_norm, 1e-12)
    return basis.astype(np.float32)


def _compute_kl_bucket_mtp_basis(tokens: np.ndarray, vocab_size: int, width: int, mtp_weights: list[float], chunk_tokens: int = 5_000_000, eps: float = 1e-8) -> np.ndarray:
    if tokens.size < 3:
        return np.zeros((vocab_size, width), dtype=np.float32)
    counts = np.zeros((vocab_size, width), dtype=np.float64)
    for h, w in enumerate(mtp_weights, start=1):
        if w == 0.0 or tokens.size - h - 1 <= 0:
            continue
        cur = tokens[h + 1 :]
        prev1 = tokens[1:-h]
        prev2 = tokens[: -(h + 1)]
        n = cur.size
        n_chunks = (n + chunk_tokens - 1) // chunk_tokens
        for ci in range(n_chunks):
            s = ci * chunk_tokens
            e = min((ci + 1) * chunk_tokens, n)
            cur_s = cur[s:e]
            prev1_s = prev1[s:e]
            prev2_s = prev2[s:e]
            context_id = prev1_s.astype(np.int64) + vocab_size * prev2_s.astype(np.int64)
            buckets = context_id % width
            np.add.at(counts, (cur_s, buckets), float(w))
    col_sum = counts.sum(axis=0, keepdims=True)
    p_t_given_b = counts / np.maximum(col_sum, 1.0)
    token_sum = counts.sum(axis=1, keepdims=True)
    p_t = token_sum / np.maximum(np.sum(counts), 1.0)
    basis = np.log(p_t_given_b + eps) - np.log(p_t + eps)
    basis = basis - basis.mean(axis=0, keepdims=True)
    col_norm = np.linalg.norm(basis, axis=0, keepdims=True)
    basis = basis / np.maximum(col_norm, 1e-12)
    return basis.astype(np.float32)


def _collect_train_tokens(pattern: str, max_tokens: int) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    chunks = []
    total = 0
    for f in files:
        t = _load_data_shard(Path(f)).cpu().numpy().astype(np.int64)
        remain = max_tokens - total
        if remain <= 0:
            break
        if t.size > remain:
            t = t[:remain]
        chunks.append(t)
        total += t.size
        if total >= max_tokens:
            break
    if not chunks:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(chunks, axis=0)


def _basis_signature(files: list[str], vocab_size: int, width: int, method: str, max_tokens: int, mtp_weights: str) -> str:
    h = hashlib.sha1()
    h.update(str(vocab_size).encode())
    h.update(str(width).encode())
    h.update(method.encode())
    h.update(str(max_tokens).encode())
    h.update(mtp_weights.encode())
    for f in files:
        p = Path(f)
        st = p.stat()
        h.update(str(p).encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


def load_or_compute_geo_basis(vocab_size: int, width: int) -> np.ndarray:
    train_files = sorted(glob.glob(HP.train_files))
    if not train_files:
        raise RuntimeError(f"No train files matched: {HP.train_files}")
    sig = _basis_signature(train_files, vocab_size, width, HP.geo_prebias_method, HP.geo_prebias_max_tokens, HP.geo_prebias_mtp_weights)
    cache_dir = Path(HP.geo_prebias_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"basis_{sig}.npy"
    if cache_file.exists() and not HP.geo_prebias_force_recompute:
        return np.load(cache_file)
    tokens = _collect_train_tokens(HP.train_files, HP.geo_prebias_max_tokens)
    if HP.geo_prebias_method == "kl_bucket":
        basis = _compute_kl_bucket_basis(tokens, vocab_size, width, chunk_tokens=HP.geo_prebias_chunk_tokens)
    elif HP.geo_prebias_method == "kl_bucket_mtp":
        basis = _compute_kl_bucket_mtp_basis(tokens, vocab_size, width, _parse_mtp_weights(HP.geo_prebias_mtp_weights), chunk_tokens=HP.geo_prebias_chunk_tokens)
    else:
        raise ValueError(f"Unknown GEO_PREBIAS_METHOD: {HP.geo_prebias_method}")
    np.save(cache_file, basis)
    return basis


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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
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
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(HP.vocab_size, HP.d_model)
        self.blocks = nn.ModuleList([Block(HP.d_model, HP.n_head) for _ in range(HP.n_layer)])
        self.ln_f = RMSNorm(HP.d_model)
        self.lm_head = nn.Linear(HP.d_model, HP.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

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

    print0(rank, f"rank={rank} world_size={world_size} device={device}")
    print0(rank, f"model: layers={HP.n_layer} heads={HP.n_head} d_model={HP.d_model} seq_len={HP.seq_len}")

    train_stream = ShardStream(HP.train_files, rank, world_size, HP.seq_len, HP.batch_size)
    val_stream = ShardStream(HP.val_files, rank, world_size, HP.seq_len, HP.batch_size)

    model = GPT().to(device)

    if HP.geo_prebias_enable:
        if rank == 0:
            print0(rank, f"geo_prebias on: method={HP.geo_prebias_method} blend={HP.geo_prebias_blend} mtp={HP.geo_prebias_mtp_weights}")
            basis_np = load_or_compute_geo_basis(HP.vocab_size, HP.d_model)
            basis_t = torch.from_numpy(basis_np).to(device=device, dtype=model.wte.weight.dtype)
        else:
            basis_t = torch.zeros((HP.vocab_size, HP.d_model), device=device, dtype=model.wte.weight.dtype)
        if world_size > 1:
            dist.broadcast(basis_t, src=0)
        alpha = float(HP.geo_prebias_blend)
        with torch.no_grad():
            model.wte.weight.mul_(1.0 - alpha).add_(basis_t, alpha=alpha)
        print0(rank, "applied geo prebias to embedding/lm_head")

    if HP.compile:
        model = torch.compile(model, dynamic=False)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    optimizer = torch.optim.AdamW(model.parameters(), lr=HP.lr, betas=(0.9, 0.95), weight_decay=HP.weight_decay)

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
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if HP.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP.grad_clip)
        optimizer.step()

        if step % 20 == 0:
            loss_t = loss.detach()
            if world_size > 1:
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            dt = (time.time() - t0) / max(1, step + 1)
            print0(rank, f"step {step:5d} | train_loss {loss_t.item():.5f} | lr {lr:.3e} | sec/step {dt:.3f}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
