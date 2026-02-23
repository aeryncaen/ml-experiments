#!/usr/bin/env python3
"""
Training script for YAMIT pipeline validation.

Trains a vanilla transformer on tokenized binary shards with:
- WSD learning rate scheduler (warmup → stable → linear decay)
- AdamW optimizer with param group separation (no WD on embeddings/norms)
- BF16 autocast
- Periodic validation evaluation
- Save-best checkpoint logic (by validation loss)
- Logging to stdout + optional wandb

Usage:
    python train.py \
        --data-dir ./tokenized \
        --output-dir ./checkpoints \
        --total-tokens 3_000_000_000 \
        --micro-batch-size 8 \
        --seq-len 4096 \
        --lr 6e-4

Shard format (produced by Go tokenizer):
    .bin: flat little-endian uint32 token IDs
    .idx: little-endian uint64 byte offsets, N+1 entries for N documents
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from model import CONFIG_130M, Transformer, TransformerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────


class ShardedTokenDataset(IterableDataset):
    """Streams sequences from tokenized binary shards.

    Each shard is a pair of files:
        {name}.bin — flat uint32 token array
        {name}.idx — uint64 document boundary offsets

    Documents are concatenated with an EOS token separator, then chunked
    into fixed-length sequences for training.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        eos_token_id: int = 151_643,  # Qwen3 EOS
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.eos_token_id = eos_token_id
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        # Find all .bin files recursively.
        self.bin_files = sorted(glob.glob(os.path.join(data_dir, "**/*.bin"), recursive=True))
        if not self.bin_files:
            raise FileNotFoundError(f"No .bin files found in {data_dir}")
        log.info(f"Found {len(self.bin_files)} shards in {data_dir}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        bin_files = list(self.bin_files)

        if self.shuffle_shards:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(bin_files)

        # Split shards across workers if using multi-worker DataLoader.
        if worker_info is not None:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id
            bin_files = bin_files[worker_id::n_workers]

        buffer = np.empty(0, dtype=np.uint32)
        chunk_size = self.seq_len + 1  # input + target

        for bin_path in bin_files:
            # Memory-map the shard for zero-copy reading.
            tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")

            # Append EOS between documents if we have an index file.
            idx_path = bin_path.replace(".bin", ".idx")
            if os.path.exists(idx_path):
                offsets = np.fromfile(idx_path, dtype=np.uint64)
                # offsets are byte offsets; convert to token offsets.
                token_offsets = offsets // 4

                docs = []
                for i in range(len(token_offsets) - 1):
                    start = int(token_offsets[i])
                    end = int(token_offsets[i + 1])
                    if end > start:
                        docs.append(tokens[start:end])
                        docs.append(np.array([self.eos_token_id], dtype=np.uint32))

                if docs:
                    shard_tokens = np.concatenate(docs)
                else:
                    shard_tokens = tokens
            else:
                shard_tokens = np.array(tokens)

            buffer = np.concatenate([buffer, shard_tokens])

            # Yield complete chunks from the buffer.
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                yield {
                    "input_ids": torch.from_numpy(chunk[:-1].copy()).long(),
                    "labels": torch.from_numpy(chunk[1:].copy()).long(),
                }


# ── WSD Scheduler ─────────────────────────────────────────────────────────


class WSDScheduler:
    """Warmup → Stable → Decay learning rate scheduler.

    - Phase 1 (warmup): linear ramp from 0 to peak_lr over warmup_steps.
    - Phase 2 (stable): constant peak_lr.
    - Phase 3 (decay): linear decay from peak_lr to 0 over decay_steps.

    The decay phase starts at (total_steps - decay_steps).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        decay_fraction: float = 0.1,  # final 10% of training
    ):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = int(total_steps * decay_fraction)
        self.decay_start = total_steps - self.decay_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup.
            return self.peak_lr * (step / max(1, self.warmup_steps))
        elif step < self.decay_start:
            # Stable phase.
            return self.peak_lr
        else:
            # Linear decay to 0.
            progress = (step - self.decay_start) / max(1, self.decay_steps)
            return self.peak_lr * (1.0 - progress)

    def step(self, step: int):
        lr = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        return lr


# ── Checkpointing ─────────────────────────────────────────────────────────


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    config: dict,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )
    log.info(f"Checkpoint saved to {path} (step {step})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    log.info(f"Checkpoint loaded from {path} (step {ckpt.get('step', '?')})")
    return ckpt


# ── Evaluation ────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    max_batches: int = 50,
    device: torch.device = torch.device("cuda"),
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, labels=labels)

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


# ── Training loop ─────────────────────────────────────────────────────────


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Model.
    cfg = CONFIG_130M
    model = Transformer(cfg).to(device)
    log.info(f"Model params: {model.param_count():,}")

    # Compile for speed.
    if hasattr(torch, "compile") and device.type == "cuda":
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer with param group separation.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay on embeddings, norms, and biases.
        if "tok_emb" in name or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == "cuda",
    )

    # Compute total steps from token budget.
    tokens_per_step = args.micro_batch_size * args.seq_len
    total_steps = args.total_tokens // tokens_per_step
    log.info(
        f"Total steps: {total_steps:,} "
        f"({args.total_tokens:,} tokens / {tokens_per_step:,} tokens per step)"
    )

    # Scheduler.
    scheduler = WSDScheduler(
        optimizer=optimizer,
        peak_lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        decay_fraction=args.decay_fraction,
    )

    # Data.
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_dataset = ShardedTokenDataset(
        train_dir, seq_len=args.seq_len, shuffle_shards=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = ShardedTokenDataset(
        val_dir, seq_len=args.seq_len, shuffle_shards=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    # Resume from checkpoint if available.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_val_loss = float("inf")
    last_ckpt = output_dir / "last.pt"
    if last_ckpt.exists() and args.resume:
        ckpt = load_checkpoint(str(last_ckpt), model, optimizer)
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log.info(f"Resuming from step {start_step}, best val loss {best_val_loss:.4f}")

    # GradScaler not needed for BF16 (no loss scaling required).
    grad_clip = args.grad_clip

    # Training.
    model.train()
    step = start_step
    tokens_seen = step * tokens_per_step
    log_interval = args.log_interval
    eval_interval = args.eval_interval
    save_interval = args.save_interval

    running_loss = 0.0
    running_count = 0
    t0 = time.time()

    log.info(f"Starting training from step {step}")

    train_iter = iter(train_loader)

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Forward + backward.
        lr = scheduler.step(step)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, labels=labels)

        loss.backward()

        # Gradient clipping.
        if grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            ).item()
        else:
            grad_norm = 0.0

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Stats.
        running_loss += loss.item()
        running_count += 1
        tokens_seen += tokens_per_step
        step += 1

        # Logging.
        if step % log_interval == 0:
            avg_loss = running_loss / running_count
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / max(1, elapsed)
            log.info(
                f"step={step:>7d} | "
                f"loss={avg_loss:.4f} | "
                f"lr={lr:.2e} | "
                f"grad_norm={grad_norm:.2f} | "
                f"tokens={tokens_seen:,.0f} | "
                f"tok/s={tok_per_sec:,.0f}"
            )
            running_loss = 0.0
            running_count = 0

        # Evaluation.
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, max_batches=50, device=device)
            log.info(f"step={step:>7d} | val_loss={val_loss:.4f} | best={best_val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    str(output_dir / "best.pt"),
                    model, optimizer, step, best_val_loss,
                    config=vars(args),
                )
                log.info(f"New best model saved (val_loss={val_loss:.4f})")

        # Periodic save.
        if step % save_interval == 0:
            save_checkpoint(
                str(output_dir / "last.pt"),
                model, optimizer, step, best_val_loss,
                config=vars(args),
            )

    # Final save.
    save_checkpoint(
        str(output_dir / "last.pt"),
        model, optimizer, step, best_val_loss,
        config=vars(args),
    )

    # Final eval.
    val_loss = evaluate(model, val_loader, max_batches=100, device=device)
    log.info(f"Training complete. Final val_loss={val_loss:.4f}, best={best_val_loss:.4f}")

    # Write training summary.
    elapsed = time.time() - t0
    summary = {
        "total_steps": step,
        "total_tokens": tokens_seen,
        "best_val_loss": best_val_loss,
        "final_val_loss": val_loss,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_seen / max(1, elapsed),
        "model_params": model.param_count() if hasattr(model, "param_count") else -1,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Training summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="YAMIT pipeline training")
    parser.add_argument("--data-dir", type=str, required=True, help="Tokenized data directory (with train/ and val/ subdirs)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Checkpoint output directory")
    parser.add_argument("--total-tokens", type=int, default=3_000_000_000, help="Total training token budget")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Micro batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip max norm")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="LR warmup steps")
    parser.add_argument("--decay-fraction", type=float, default=0.1, help="Fraction of training for LR decay (final 10%)")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-interval", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
