"""Data pipeline for ByteChunkVAE.

Two-stage:
1. detokenize_shards(): reads fineweb .bin files (GPT-2 BPE uint16),
   decodes to raw bytes, writes .bin byte shards (uint8, same header format).
2. ByteShardStream: loads byte shards and serves random fixed-size chunks
   with BOS/EOS/PAD framing for VAE training.
"""

import glob
import os
from pathlib import Path

import numpy as np
import torch

from .model import BOS, EOS, PAD, BYTE_OFFSET


# ---------------------------------------------------------------------------
# Stage 1: BPE .bin -> raw byte .bin conversion
# ---------------------------------------------------------------------------

def _load_bpe_shard(path: Path) -> np.ndarray:
    """Load a fineweb BPE shard (.bin with 256-int32 header, then uint16 tokens)."""
    header = np.fromfile(str(path), dtype=np.int32, count=256)
    assert int(header[0]) == 20240520, f"bad magic in {path}"
    assert int(header[1]) == 1, f"unsupported version in {path}"
    num_tokens = int(header[2])
    tokens = np.fromfile(str(path), dtype=np.uint16, offset=256 * 4, count=num_tokens)
    return tokens


def _write_byte_shard(path: Path, data: np.ndarray):
    """Write a raw byte shard: 256-int32 header + uint8 bytes."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240521  # different magic to distinguish from BPE shards
    header[1] = 1
    header[2] = len(data)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(data.astype(np.uint8).tobytes())


def load_byte_shard(path: Path) -> np.ndarray:
    """Load a raw byte shard."""
    header = np.fromfile(str(path), dtype=np.int32, count=256)
    assert int(header[0]) == 20240521, f"bad magic in {path} (expected byte shard)"
    num_bytes = int(header[2])
    data = np.fromfile(str(path), dtype=np.uint8, offset=256 * 4, count=num_bytes)
    return data


def detokenize_shards(
    bpe_pattern: str,
    output_dir: str,
    max_shards: int | None = None,
):
    """Convert fineweb BPE .bin shards to raw byte .bin shards.

    Args:
        bpe_pattern: glob for input BPE shards, e.g. "data/fineweb10B/fineweb_train_*.bin"
        output_dir: where to write byte shards
        max_shards: if set, only convert this many shards (for testing)
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    files = sorted(glob.glob(bpe_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {bpe_pattern}")
    if max_shards:
        files = files[:max_shards]

    os.makedirs(output_dir, exist_ok=True)

    for i, fpath in enumerate(files):
        fpath = Path(fpath)
        out_name = fpath.stem + "_bytes.bin"
        out_path = Path(output_dir) / out_name

        if out_path.exists():
            print(f"  [{i+1}/{len(files)}] {out_path.name} exists, skipping")
            continue

        print(f"  [{i+1}/{len(files)}] {fpath.name} -> {out_path.name} ...", end=" ", flush=True)
        bpe_tokens = _load_bpe_shard(fpath)

        # Decode BPE tokens to raw bytes
        raw_bytes = enc.decode_bytes(bpe_tokens.tolist())
        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)

        _write_byte_shard(out_path, byte_array)
        print(f"{len(byte_array):,} bytes")

    print(f"Done. {len(files)} shards written to {output_dir}/")


# ---------------------------------------------------------------------------
# Stage 2: Byte shard loader for VAE training
# ---------------------------------------------------------------------------

class ByteShardStream:
    """Streams random fixed-size byte chunks from pre-converted byte shards.

    Each chunk is framed as: [BOS, b0+3, b1+3, ..., EOS, PAD...]
    where content length = chunk_size - 2.

    Loads one shard at a time into memory, serves random chunks from it,
    advances to next shard when exhausted.
    """

    def __init__(
        self,
        shard_pattern: str,
        chunk_size: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.files = sorted(glob.glob(shard_pattern))
        if not self.files:
            raise FileNotFoundError(f"No byte shards matched: {shard_pattern}")
        self.chunk_size = chunk_size
        self.content_len = chunk_size - 2  # space for BOS + EOS
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        # Shard state
        self.file_idx = rank % len(self.files)  # start at different shards per rank
        self.data: np.ndarray = np.array([], dtype=np.uint8)
        self.n_chunks = 0
        self.pos = 0
        self._load_shard()

    def _load_shard(self):
        path = Path(self.files[self.file_idx])
        self.data = load_byte_shard(path)
        self.n_chunks = len(self.data) // self.content_len
        self.pos = 0

    def _advance_shard(self):
        self.file_idx = (self.file_idx + self.world_size) % len(self.files)
        self._load_shard()

    def next_batch(self, device: torch.device) -> torch.Tensor:
        """Returns (batch_size, chunk_size) long tensor of byte token IDs."""
        if self.pos + self.batch_size > self.n_chunks:
            self._advance_shard()

        B, K, C = self.batch_size, self.chunk_size, self.content_len

        # Grab contiguous byte block and reshape
        start = self.pos * C
        raw = self.data[start : start + B * C].reshape(B, C)

        # Build chunks: [BOS, bytes+OFFSET, EOS, PAD...]
        # All chunks are full-length (no partial last chunk in the middle of a shard)
        chunks = torch.full((B, K), PAD, dtype=torch.long)
        chunks[:, 0] = BOS
        chunks[:, 1 : C + 1] = torch.from_numpy(raw.astype(np.int64)) + BYTE_OFFSET
        chunks[:, C + 1] = EOS

        self.pos += B
        return chunks.to(device)


# ---------------------------------------------------------------------------
# CLI: python -m vae.data --bpe-pattern "path/to/*.bin" --output-dir "path/to/bytes"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert fineweb BPE shards to raw byte shards")
    parser.add_argument("--bpe-pattern", type=str, required=True,
                        help="Glob pattern for BPE .bin files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write byte shards")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Max shards to convert (for testing)")
    args = parser.parse_args()

    detokenize_shards(args.bpe_pattern, args.output_dir, args.max_shards)
