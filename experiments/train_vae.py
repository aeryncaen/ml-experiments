"""Standalone ByteChunkVAE trainer on fineweb byte shards.

Usage:
    # 1. First, convert BPE shards to byte shards (one-time):
    python -m vae.data \
        --bpe-pattern "experiments/tier4/data/fineweb10B/fineweb_train_*.bin" \
        --output-dir "data/fineweb_bytes"

    # 2. Train the VAE:
    python experiments/train_vae.py

    # With env overrides:
    CHUNK_SIZE=16 D_LATENT=64 BETA=0.01 TRAIN_STEPS=5000 python experiments/train_vae.py

    # Multi-GPU:
    torchrun --nproc_per_node=2 experiments/train_vae.py
"""

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vae.model import ByteChunkVAE, VAEConfig
from vae.data import ByteShardStream, detokenize_shards


# ---------------------------------------------------------------------------
# Config (env-var overridable)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")

def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class HParams:
    # Data
    bpe_dir: str = _env_str("BPE_DIR", str(Path(__file__).resolve().parent / "tier4/data/fineweb10B"))
    byte_dir: str = _env_str("BYTE_DIR", str(Path(__file__).resolve().parent.parent / "data/fineweb_bytes"))

    # Model
    chunk_size: int = _env_int("CHUNK_SIZE", 16)
    d_model: int = _env_int("VAE_D_MODEL", 256)
    d_latent: int = _env_int("D_LATENT", 64)
    n_heads: int = _env_int("VAE_N_HEADS", 4)
    enc_layers: int = _env_int("ENC_LAYERS", 4)
    dec_layers: int = _env_int("DEC_LAYERS", 4)
    beta: float = _env_float("BETA", 0.01)
    dropout: float = _env_float("VAE_DROPOUT", 0.0)

    # Training
    train_steps: int = _env_int("TRAIN_STEPS", 10000)
    batch_size: int = _env_int("BATCH_SIZE", 256)
    lr: float = _env_float("LR", 3e-4)
    warmup_steps: int = _env_int("WARMUP_STEPS", 500)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.01)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)
    val_every: int = _env_int("VAL_EVERY", 200)
    val_steps: int = _env_int("VAL_STEPS", 20)
    save_every: int = _env_int("SAVE_EVERY", 2000)
    compile: bool = _env_bool("TORCH_COMPILE", True)
    log_every: int = _env_int("LOG_EVERY", 20)
    save_dir: str = _env_str("SAVE_DIR", str(Path(__file__).resolve().parent.parent / "out/vae"))
    max_convert_shards: int | None = None  # set for quick testing


HP = HParams()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def lr_for_step(step: int) -> float:
    if step < HP.warmup_steps:
        return HP.lr * (step + 1) / max(1, HP.warmup_steps)
    t = (step - HP.warmup_steps) / max(1, HP.train_steps - HP.warmup_steps)
    return HP.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_stream, device, world_size, rank):
    model.eval()
    total_recon = 0.0
    total_kl = 0.0
    total_acc = 0.0
    for _ in range(HP.val_steps):
        x = val_stream.next_batch(device)
        loss, recon, kl, acc = model(x)
        total_recon += recon.item()
        total_kl += kl.item()
        total_acc += acc.item()
    n = HP.val_steps
    model.train()
    return total_recon / n, total_kl / n, total_acc / n


# ---------------------------------------------------------------------------
# Sample reconstruction display
# ---------------------------------------------------------------------------

def show_reconstruction(model, val_stream, device):
    """Show a few reconstructed chunks for qualitative inspection."""
    model.eval()
    from vae.model import ByteChunkVAE, BYTE_OFFSET, BOS, EOS, PAD
    x = val_stream.next_batch(device)[:4]  # just 4 samples
    mu, log_var = model.encoder(x) if not hasattr(model, 'module') else model.module.encoder(x)
    z = mu  # deterministic for display
    logits = model.decoder(z) if not hasattr(model, 'module') else model.module.decoder(z)
    preds = logits.argmax(dim=-1)

    for i in range(min(4, x.shape[0])):
        orig_bytes = []
        pred_bytes = []
        for j in range(x.shape[1]):
            tok = x[i, j].item()
            if tok >= BYTE_OFFSET:
                orig_bytes.append(tok - BYTE_OFFSET)
            ptok = preds[i, j].item()
            if ptok >= BYTE_OFFSET:
                pred_bytes.append(ptok - BYTE_OFFSET)
        orig_str = bytes(orig_bytes).decode("utf-8", errors="replace")
        pred_str = bytes(pred_bytes).decode("utf-8", errors="replace")
        print(f"  [{i}] orig: {repr(orig_str)}")
        print(f"  [{i}] pred: {repr(pred_str)}")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rank, world_size, device = setup_dist()
    torch.manual_seed(42 + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print0(rank, f"ByteChunkVAE trainer | rank={rank} world_size={world_size} device={device}")
    print0(rank, f"  chunk_size={HP.chunk_size} d_model={HP.d_model} d_latent={HP.d_latent}")
    print0(rank, f"  enc_layers={HP.enc_layers} dec_layers={HP.dec_layers} beta={HP.beta}")
    print0(rank, f"  batch_size={HP.batch_size} lr={HP.lr} steps={HP.train_steps}")

    # --- Ensure byte shards exist ---
    byte_dir = Path(HP.byte_dir)
    train_pattern = str(byte_dir / "fineweb_train_*_bytes.bin")
    val_pattern = str(byte_dir / "fineweb_val_*_bytes.bin")

    import glob
    if not glob.glob(train_pattern):
        print0(rank, "No byte shards found. Converting BPE shards...")
        if rank == 0:
            bpe_train = os.path.join(HP.bpe_dir, "fineweb_train_*.bin")
            bpe_val = os.path.join(HP.bpe_dir, "fineweb_val_*.bin")
            detokenize_shards(bpe_train, str(byte_dir), HP.max_convert_shards)
            detokenize_shards(bpe_val, str(byte_dir), 1)
        if world_size > 1:
            dist.barrier()
        print0(rank, "Byte shard conversion done.")

    # --- Data ---
    train_stream = ByteShardStream(
        train_pattern, HP.chunk_size, HP.batch_size, rank, world_size,
    )
    val_stream = ByteShardStream(
        val_pattern, HP.chunk_size, HP.batch_size, rank, world_size,
    )

    # --- Model ---
    cfg = VAEConfig(
        chunk_size=HP.chunk_size,
        d_model=HP.d_model,
        d_latent=HP.d_latent,
        n_heads=HP.n_heads,
        enc_layers=HP.enc_layers,
        dec_layers=HP.dec_layers,
        beta=HP.beta,
        dropout=HP.dropout,
    )
    model = ByteChunkVAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print0(rank, f"  parameters: {n_params:,}")

    # --- Optimizer ---
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": HP.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=HP.lr, betas=(0.9, 0.95))

    # --- Compile / DDP ---
    if HP.compile and device.type == "cuda":
        model = torch.compile(model, dynamic=False)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    # --- Train ---
    os.makedirs(HP.save_dir, exist_ok=True)
    t0 = time.time()

    for step in range(HP.train_steps + 1):
        # Validate
        if step % HP.val_every == 0 or step == HP.train_steps:
            val_recon, val_kl, val_acc = evaluate(model, val_stream, device, world_size, rank)
            print0(rank, f"step {step:5d} | val_recon {val_recon:.4f} val_kl {val_kl:.4f} val_acc {val_acc:.4f}")
            if step > 0 and rank == 0:
                show_reconstruction(model, val_stream, device)
            if step == HP.train_steps:
                break

        # LR schedule
        lr = lr_for_step(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        x = train_stream.next_batch(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, recon_loss, kl_loss, accuracy = model(x)
        else:
            loss, recon_loss, kl_loss, accuracy = model(x)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if HP.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                decay_params + no_decay_params, HP.grad_clip,
            )
        optimizer.step()

        # Log
        if step % HP.log_every == 0:
            loss_t = loss.detach()
            if world_size > 1:
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            dt = (time.time() - t0) / max(1, step + 1)
            print0(
                rank,
                f"step {step:5d} | loss {loss_t.item():.4f} "
                f"recon {recon_loss.item():.4f} kl {kl_loss.item():.4f} "
                f"acc {accuracy.item():.4f} | lr {lr:.3e} sec/step {dt:.3f}",
            )

        # Save
        if step > 0 and step % HP.save_every == 0 and rank == 0:
            raw_model = model.module if hasattr(model, "module") else model
            # If compiled, unwrap _orig_mod
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod
            ckpt = {
                "model": raw_model.state_dict(),
                "config": cfg,
                "step": step,
                "hparams": HP.__dict__,
            }
            path = os.path.join(HP.save_dir, f"vae_step{step:06d}.pt")
            torch.save(ckpt, path)
            print0(rank, f"  saved {path}")

    # Final save
    if rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        ckpt = {
            "model": raw_model.state_dict(),
            "config": cfg,
            "step": HP.train_steps,
            "hparams": HP.__dict__,
        }
        path = os.path.join(HP.save_dir, "vae_final.pt")
        torch.save(ckpt, path)
        print0(rank, f"  saved {path}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
