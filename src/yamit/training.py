"""Reusable training utilities for YAMIT: data loading, scheduler, checkpointing."""

from __future__ import annotations

import glob
import logging
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset

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
        eos_token_id: int,
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
    """Warmup -> Stable -> Decay learning rate scheduler.

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
