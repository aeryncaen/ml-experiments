import glob
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
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

    model_type: str = os.environ.get("MODEL_TYPE", "transformer")  # transformer | s6
    vocab_size: int = 50304
    n_layer: int = _env_int("N_LAYER", 12)
    n_head: int = _env_int("N_HEAD", 12)
    d_model: int = _env_int("D_MODEL", 768)
    seq_len: int = _env_int("SEQ_LEN", 2048)

    # S6 knobs
    s6_headdim: int = _env_int("S6_HEADDIM", 64)
    s6_expansion_factor: int = _env_int("S6_EXPANSION_FACTOR", 2)
    s6_post_scan_attention: bool = _env_bool("S6_POST_SCAN_ATTENTION", True)
    s6_scan_state_modes: str = os.environ.get("S6_SCAN_STATE_MODES", "outer,outer,elementwise")

    train_steps: int = _env_int("TRAIN_STEPS", 2000)
    batch_size: int = _env_int("BATCH_SIZE", 8)
    val_steps: int = _env_int("VAL_STEPS", 32)
    val_every: int = _env_int("VAL_EVERY", 100)
    lr: float = _env_float("LR", 3e-4)
    warmup_steps: int = _env_int("WARMUP_STEPS", 200)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.1)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)
    compile: bool = _env_bool("TORCH_COMPILE", True)


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
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class TransformerBlock(nn.Module):
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
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
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
