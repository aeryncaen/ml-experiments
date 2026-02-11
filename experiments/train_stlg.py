"""Train STLG — Straight-Through Latent Generator.

Causal transformer that predicts the next latent vector from a frozen
ByteChunkVAE. Loss is CE against target byte chunks, computed by decoding
predicted latents through the frozen VAE decoder.

Usage:
    PYTHONPATH=src python experiments/train_stlg.py \
        --vae-checkpoint out/vae/vae_best.pt \
        --byte-dir data/fineweb_bytes
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vae.model import ByteChunkVAE, VAEConfig, VOCAB_SIZE, PAD, BYTE_OFFSET
from vae.data import PieceStream
from stlg.model import STLG, STLGConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train STLG")

    # Data
    p.add_argument("--byte-dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "data/fineweb_bytes"),
                    help="Directory with fineweb byte .bin shards")

    # VAE
    p.add_argument("--vae-checkpoint", type=str, required=True,
                    help="Path to frozen VAE checkpoint (vae_best.pt)")

    # STLG model
    p.add_argument("--d-model", type=int, default=128, help="STLG transformer hidden dim")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    # Training
    p.add_argument("--train-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64,
                    help="Pieces per batch (each piece = 17 chunks)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-steps", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--save-dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "out/stlg"))
    p.add_argument("--no-compile", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_dist():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        rank, world_size = 0, 1
        device = torch.device("cuda")
    else:
        rank, world_size = 0, 1
        device = torch.device("cpu")
    return rank, world_size, device


def print0(rank: int, s: str):
    if rank == 0:
        print(s, flush=True)


def lr_for_step(step: int, args) -> float:
    warmup_ratio = 0.1
    min_ratio = 0.05
    if step < args.warmup_steps:
        return args.lr * (warmup_ratio + (1.0 - warmup_ratio) * step / max(1, args.warmup_steps))
    progress = (step - args.warmup_steps) / max(1, args.train_steps - args.warmup_steps)
    return args.lr * (min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress)))


def unwrap_model(model):
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def load_frozen_vae(path: str, device: torch.device) -> ByteChunkVAE:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vae = ByteChunkVAE(cfg).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def save_checkpoint(model, stlg_cfg, vae_cfg, step, path):
    raw = unwrap_model(model)
    # Only save STLG params, not the frozen VAE
    stlg_state = {k: v for k, v in raw.state_dict().items() if not k.startswith("vae.")}
    torch.save({
        "model": stlg_state,
        "config": stlg_cfg,
        "vae_config": vae_cfg,
        "step": step,
    }, path)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_stream, device, args):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    for _ in range(args.val_steps):
        pieces = val_stream.next_batch(device)
        loss, acc = model(pieces)
        total_loss += loss.item()
        total_acc += acc.item()
    n = args.val_steps
    model.train()
    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Show samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def show_samples(model, val_stream, device):
    model.eval()
    raw = unwrap_model(model)
    pieces = val_stream.next_batch(device)[:2]
    B, S, K = pieces.shape

    flat_chunks = pieces.reshape(B * S, K)
    flat_mu, _ = raw.vae.encoder(flat_chunks)
    latents = flat_mu.reshape(B, S, -1)

    pred_latents = raw.predict_latents(latents[:, :-1])
    pred_flat = pred_latents.reshape(B * (S - 1), -1)
    logits = raw.vae.decoder(pred_flat)
    pred_tokens = logits.argmax(dim=-1).reshape(B, S - 1, K)

    for i in range(B):
        target = pieces[i, 1:]
        predicted = pred_tokens[i]

        orig_bytes = [t.item() - BYTE_OFFSET for row in target for t in row if t.item() >= BYTE_OFFSET]
        pred_bytes = [t.item() - BYTE_OFFSET for row in predicted for t in row if t.item() >= BYTE_OFFSET]

        orig_str = bytes(orig_bytes).decode("utf-8", errors="replace")[:80]
        pred_str = bytes(pred_bytes).decode("utf-8", errors="replace")[:80]
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
        torch.cuda.manual_seed(42 + rank)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Load frozen VAE
    print0(rank, f"Loading frozen VAE from {args.vae_checkpoint}")
    vae = load_frozen_vae(args.vae_checkpoint, device)
    vae_cfg = vae.cfg
    print0(rank, f"  VAE: chunk_size={vae_cfg.chunk_size} d_latent={vae_cfg.d_latent}")

    # Data
    byte_dir = Path(args.byte_dir)
    train_pattern = str(byte_dir / "fineweb_train_*_bytes.bin")
    val_pattern = str(byte_dir / "fineweb_val_*_bytes.bin")

    train_stream = PieceStream(
        train_pattern, vae_cfg.chunk_size, args.batch_size, rank, world_size,
    )
    val_stream = PieceStream(
        val_pattern, vae_cfg.chunk_size, args.batch_size, rank, world_size,
        shuffle=True,
    )

    # Compute sequence length from VAE config
    piece_bytes = vae_cfg.chunk_size * 16
    seq_len = (piece_bytes + 2 + vae_cfg.chunk_size - 1) // vae_cfg.chunk_size

    # STLG model (includes frozen VAE)
    stlg_cfg = STLGConfig(
        d_latent=vae_cfg.d_latent,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=seq_len,
        dropout=args.dropout,
    )
    model = STLG(stlg_cfg, vae).to(device)

    # Count only trainable params (not frozen VAE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_vae_params = sum(p.numel() for p in vae.parameters())

    print0(rank, f"STLG trainer | rank={rank} world_size={world_size} device={device}")
    print0(rank, f"  d_model={args.d_model} n_heads={args.n_heads} n_layers={args.n_layers}")
    print0(rank, f"  seq_len={seq_len} d_latent={vae_cfg.d_latent}")
    print0(rank, f"  batch_size={args.batch_size} lr={args.lr} steps={args.train_steps}")
    print0(rank, f"  STLG parameters: {n_params:,} (VAE frozen: {n_vae_params:,})")

    # Optimizer (only trainable params)
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

    # Compile / DDP
    if not args.no_compile and device.type == "cuda":
        model = torch.compile(model, dynamic=False)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    # Train
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(args.save_dir, "stlg_best.pt")

    pbar = tqdm(range(args.train_steps + 1), desc="train", disable=(rank != 0))
    for step in pbar:
        # Validate
        if step % args.val_every == 0 or step == args.train_steps:
            val_loss, val_acc = evaluate(model, val_stream, device, args)
            print0(rank, f"\nstep {step:5d} | val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
            if step > 0 and rank == 0:
                show_samples(model, val_stream, device)
            if val_acc > best_val_acc and rank == 0:
                best_val_acc = val_acc
                save_checkpoint(model, stlg_cfg, vae_cfg, step, best_path)
                print0(rank, f"  new best val_acc {val_acc:.4f} -> {best_path}")
            if step == args.train_steps:
                break

        # LR schedule
        lr = lr_for_step(step, args)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        pieces = train_stream.next_batch(device)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, accuracy = model(pieces)
        else:
            loss, accuracy = model(pieces)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                decay_params + no_decay_params, args.grad_clip,
            )
        optimizer.step()

        # tqdm
        if step % 100 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{accuracy.item():.3f}",
                lr=f"{lr:.1e}",
            )

        # Periodic save
        if step > 0 and step % args.save_every == 0 and rank == 0:
            path = os.path.join(args.save_dir, f"stlg_step{step:06d}.pt")
            save_checkpoint(model, stlg_cfg, vae_cfg, step, path)
            print0(rank, f"  saved {path}")

    # Final save
    if rank == 0:
        path = os.path.join(args.save_dir, "stlg_final.pt")
        save_checkpoint(model, stlg_cfg, vae_cfg, args.train_steps, path)
        print0(rank, f"  saved {path}")
        print0(rank, f"  best val_acc: {best_val_acc:.4f} ({best_path})")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
