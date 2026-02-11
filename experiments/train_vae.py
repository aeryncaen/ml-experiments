"""Standalone ByteChunkVAE trainer on fineweb byte shards.

Usage:
    # 1. First, convert BPE shards to byte shards (one-time):
    PYTHONPATH=src python -m vae.data \
        --bpe-pattern "experiments/tier4/data/fineweb10B/fineweb_train_*.bin" \
        --output-dir "data/fineweb_bytes"

    # 2. Train the VAE:
    PYTHONPATH=src python experiments/train_vae.py --byte-dir data/fineweb_bytes

    # With overrides:
    PYTHONPATH=src python experiments/train_vae.py --chunk-size 16 --d-latent 64 --beta 0.01

    # Multi-GPU:
    PYTHONPATH=src torchrun --nproc_per_node=2 experiments/train_vae.py
"""

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vae.model import ByteChunkVAE, VAEConfig, BYTE_OFFSET, VOCAB_SIZE, PAD
from vae.data import ByteShardStream, detokenize_shards


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ByteChunkVAE")

    # Data
    p.add_argument("--bpe-dir", type=str,
                    default=str(Path(__file__).resolve().parent / "tier4/data/fineweb10B"),
                    help="Directory with fineweb BPE .bin shards (for auto-conversion)")
    p.add_argument("--byte-dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "data/fineweb_bytes"),
                    help="Directory with fineweb byte .bin shards")

    # Model
    p.add_argument("--chunk-size", type=int, default=16, help="Bytes per chunk (including BOS/EOS)")
    p.add_argument("--d-model", type=int, default=256, help="Encoder/decoder hidden dim")
    p.add_argument("--d-latent", type=int, default=64, help="Latent bottleneck dim")
    p.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    p.add_argument("--enc-layers", type=int, default=4, help="Encoder transformer depth")
    p.add_argument("--dec-layers", type=int, default=4, help="Decoder transformer depth")
    p.add_argument("--beta", type=float, default=0.01, help="KL weight (beta-VAE)")
    p.add_argument("--dropout", type=float, default=0.0)

    # Training
    p.add_argument("--train-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-steps", type=int, default=20)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "out/vae"))
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile")

    # Hard example mining
    p.add_argument("--hard-mining-threshold", type=float, default=0.95,
                    help="Val acc threshold to activate hard mining")
    p.add_argument("--hard-mining-ratio", type=int, default=4,
                    help="Oversample ratio: fetch N*batch, keep hardest batch")

    return p.parse_args()


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


def lr_for_step(step: int, args) -> float:
    """Cosine LR: warmup from lr*0.1 to lr, then cosine decay to lr*0.05."""
    warmup_ratio = 0.1
    min_ratio = 0.05
    if step < args.warmup_steps:
        return args.lr * (warmup_ratio + (1.0 - warmup_ratio) * step / max(1, args.warmup_steps))
    progress = (step - args.warmup_steps) / max(1, args.train_steps - args.warmup_steps)
    return args.lr * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)))


def unwrap_model(model):
    """Unwrap DDP / torch.compile wrappers to get the raw module."""
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def save_checkpoint(model, cfg, step, path):
    raw = unwrap_model(model)
    ckpt = {
        "model": raw.state_dict(),
        "config": cfg,
        "step": step,
    }
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_stream, device, args):
    model.eval()
    total_recon = 0.0
    total_kl = 0.0
    total_acc = 0.0
    for _ in range(args.val_steps):
        x = val_stream.next_batch(device)
        loss, recon, kl, acc = model(x)
        total_recon += recon.item()
        total_kl += kl.item()
        total_acc += acc.item()
    n = args.val_steps
    model.train()
    return total_recon / n, total_kl / n, total_acc / n


# ---------------------------------------------------------------------------
# Sample reconstruction display
# ---------------------------------------------------------------------------

def show_reconstruction(model, val_stream, device):
    """Show a few reconstructed chunks for qualitative inspection."""
    model.eval()
    x = val_stream.next_batch(device)[:4]
    raw = unwrap_model(model)
    mu, _ = raw.encoder(x)
    logits = raw.decoder(mu)  # deterministic
    preds = logits.argmax(dim=-1)

    for i in range(min(4, x.shape[0])):
        orig_bytes = [t.item() - BYTE_OFFSET for t in x[i] if t.item() >= BYTE_OFFSET]
        pred_bytes = [t.item() - BYTE_OFFSET for t in preds[i] if t.item() >= BYTE_OFFSET]
        orig_str = bytes(orig_bytes).decode("utf-8", errors="replace")
        pred_str = bytes(pred_bytes).decode("utf-8", errors="replace")
        print(f"  [{i}] orig: {repr(orig_str)}")
        print(f"  [{i}] pred: {repr(pred_str)}")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world_size, device = setup_dist()
    torch.manual_seed(42 + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print0(rank, f"ByteChunkVAE trainer | rank={rank} world_size={world_size} device={device}")
    print0(rank, f"  chunk_size={args.chunk_size} d_model={args.d_model} d_latent={args.d_latent}")
    print0(rank, f"  enc_layers={args.enc_layers} dec_layers={args.dec_layers} beta={args.beta}")
    print0(rank, f"  batch_size={args.batch_size} lr={args.lr} steps={args.train_steps}")

    # --- Ensure byte shards exist ---
    byte_dir = Path(args.byte_dir)
    train_pattern = str(byte_dir / "fineweb_train_*_bytes.bin")
    val_pattern = str(byte_dir / "fineweb_val_*_bytes.bin")

    if not glob.glob(train_pattern):
        print0(rank, "No byte shards found. Converting BPE shards...")
        if rank == 0:
            bpe_train = os.path.join(args.bpe_dir, "fineweb_train_*.bin")
            bpe_val = os.path.join(args.bpe_dir, "fineweb_val_*.bin")
            detokenize_shards(bpe_train, str(byte_dir))
            detokenize_shards(bpe_val, str(byte_dir))
        if world_size > 1:
            dist.barrier()
        print0(rank, "Byte shard conversion done.")

    # --- Data ---
    train_stream = ByteShardStream(
        train_pattern, args.chunk_size, args.batch_size, rank, world_size,
    )
    val_stream = ByteShardStream(
        val_pattern, args.chunk_size, args.batch_size, rank, world_size,
    )

    # --- Model ---
    cfg = VAEConfig(
        chunk_size=args.chunk_size,
        d_model=args.d_model,
        d_latent=args.d_latent,
        n_heads=args.n_heads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        beta=args.beta,
        dropout=args.dropout,
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
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # --- Compile / DDP ---
    if not args.no_compile and device.type == "cuda":
        model = torch.compile(model, dynamic=False)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    # --- Train ---
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(args.save_dir, "vae_best.pt")
    hard_mining_active = False

    pbar = tqdm(range(args.train_steps + 1), desc="train", disable=(rank != 0))
    for step in pbar:
        # Validate
        if step % args.val_every == 0 or step == args.train_steps:
            val_recon, val_kl, val_acc = evaluate(model, val_stream, device, args)
            print0(rank, f"\nstep {step:5d} | val_recon {val_recon:.4f} val_kl {val_kl:.4f} val_acc {val_acc:.4f}")
            if step > 0 and rank == 0:
                show_reconstruction(model, val_stream, device)
            # Best checkpoint
            if val_acc > best_val_acc and rank == 0:
                best_val_acc = val_acc
                save_checkpoint(model, cfg, step, best_path)
                print0(rank, f"  new best val_acc {val_acc:.4f} -> {best_path}")
            # Activate hard mining once threshold is crossed
            if val_acc >= args.hard_mining_threshold and not hard_mining_active:
                hard_mining_active = True
                print0(rank, f"  hard mining activated (val_acc {val_acc:.4f} >= {args.hard_mining_threshold})")
            if step == args.train_steps:
                break

        # LR schedule
        lr = lr_for_step(step, args)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward (with optional hard example mining)
        if hard_mining_active:
            # Fetch oversized batch, score, keep hardest
            raw_model = unwrap_model(model)
            candidates = []
            for _ in range(args.hard_mining_ratio):
                candidates.append(train_stream.next_batch(device))
            x_big = torch.cat(candidates, dim=0)  # (ratio*B, K)
            per_loss = raw_model.per_sample_loss(x_big)  # (ratio*B,)
            _, hard_idx = per_loss.topk(args.batch_size)
            x = x_big[hard_idx]
        else:
            x = train_stream.next_batch(device)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, recon_loss, kl_loss, accuracy = model(x)
        else:
            loss, recon_loss, kl_loss, accuracy = model(x)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                decay_params + no_decay_params, args.grad_clip,
            )
        optimizer.step()

        # tqdm update
        mining = "H" if hard_mining_active else ""
        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            recon=f"{recon_loss.item():.3f}",
            kl=f"{kl_loss.item():.3f}",
            acc=f"{accuracy.item():.3f}",
            lr=f"{lr:.1e}",
            m=mining,
        )

        # Periodic save
        if step > 0 and step % args.save_every == 0 and rank == 0:
            path = os.path.join(args.save_dir, f"vae_step{step:06d}.pt")
            save_checkpoint(model, cfg, step, path)
            print0(rank, f"  saved {path}")

    # Final save
    if rank == 0:
        path = os.path.join(args.save_dir, "vae_final.pt")
        save_checkpoint(model, cfg, args.train_steps, path)
        print0(rank, f"  saved {path}")
        print0(rank, f"  best val_acc: {best_val_acc:.4f} ({best_path})")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
