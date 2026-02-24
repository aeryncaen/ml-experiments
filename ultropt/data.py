"""
Data loading for ultropt experiments.

Reads tokenized yamit shards (.bin uint32 + .idx doc boundaries).
Tokenizer metadata (vocab_size, eos) loaded from artifact_meta.json.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Tokenizer metadata
# ---------------------------------------------------------------------------

def load_tokenizer_meta(tokenizer_dir: str) -> dict:
    """Load artifact_meta.json from a yamit tokenizer artifacts directory.

    Returns dict with at least: vocab_size (int), eos_token_id (int).
    """
    meta_path = os.path.join(tokenizer_dir, "artifact_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "vocab_size": meta["final_vocab_size"],
        "eos_token_id": meta["special_tokens"]["eos"],
        "raw": meta,
    }


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------

def _load_data_shard(file: Path, eos_token_id: int) -> torch.Tensor:
    """Load a yamit .bin shard.  Insert EOS between documents if .idx exists."""
    tokens_np = np.fromfile(str(file), dtype=np.uint32)
    idx_path = file.with_suffix(".idx")
    if idx_path.exists():
        offsets = np.fromfile(str(idx_path), dtype=np.uint64)
        if offsets.size >= 2:
            tok_off = offsets // 4  # byte offsets -> token offsets
            docs = []
            eos = np.array([eos_token_id], dtype=np.uint32)
            for i in range(len(tok_off) - 1):
                s, e = int(tok_off[i]), int(tok_off[i + 1])
                if e > s:
                    docs.append(tokens_np[s:e])
                    docs.append(eos)
            if docs:
                tokens_np = np.concatenate(docs)
    return torch.from_numpy(tokens_np.astype(np.int64)).pin_memory()


# ---------------------------------------------------------------------------
# ShardStream — simple memmap-based batch iterator
# ---------------------------------------------------------------------------

class ShardStream:
    """Streams (x, y) batches from tokenized binary shards.

    Single-process, no DataLoader overhead.  Cycles through shards.
    Designed for single-GPU (rank=0, world_size=1) but supports DDP.
    """

    def __init__(
        self,
        pattern: str,
        seq_len: int,
        batch_size: int,
        eos_token_id: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.files = [Path(f) for f in sorted(glob.glob(pattern, recursive=True))]
        if not self.files:
            raise FileNotFoundError(f"No .bin files matched: {pattern}")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eos_token_id = eos_token_id
        self.rank = rank
        self.world_size = world_size
        self.tokens_per_rank = seq_len * batch_size
        self.tokens_per_global_step = self.tokens_per_rank * world_size
        self.file_idx = 0
        self.pos = 0
        self.tokens = _load_data_shard(self.files[0], eos_token_id)

    def _advance_shard(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.file_idx], self.eos_token_id)
        self.pos = 0

    def reset(self):
        """Reset to start of first shard (for deterministic val eval)."""
        self.file_idx = 0
        self.pos = 0
        self.tokens = _load_data_shard(self.files[0], self.eos_token_id)

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        needed = self.tokens_per_global_step + 1
        if self.pos + needed >= self.tokens.numel():
            self._advance_shard()
        start = self.pos + self.rank * self.tokens_per_rank
        end = start + self.tokens_per_rank + 1
        buf = self.tokens[start:end]
        self.pos += self.tokens_per_global_step
        x = buf[:-1].view(self.batch_size, self.seq_len)
        y = buf[1:].view(self.batch_size, self.seq_len)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
