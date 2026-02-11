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
    """Streams fixed-size byte chunks from pre-converted byte shards.

    Takes contiguous pieces of chunk_size*16 bytes from the shard, frames each
    as [BOS, bytes+3, bytes+3, ..., EOS], then chunks that sequence into
    chunk_size-width pieces. The first chunk starts with BOS, the last chunk
    has EOS followed by PAD, and all middle chunks are pure byte content.

    Each next_batch() call returns batch_size chunks drawn from these pieces.
    """

    def __init__(
        self,
        shard_pattern: str,
        chunk_size: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
    ):
        self.files = sorted(glob.glob(shard_pattern))
        if not self.files:
            raise FileNotFoundError(f"No byte shards matched: {shard_pattern}")
        self.chunk_size = chunk_size
        self.piece_bytes = chunk_size * 16  # raw bytes per piece
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle

        # Shard state
        self.file_idx = rank % len(self.files)
        self.data: np.ndarray = np.array([], dtype=np.uint8)
        self.chunks: torch.Tensor = torch.empty(0)  # pre-chunked buffer
        self.pos = 0
        self._load_shard()

    def _load_shard(self):
        path = Path(self.files[self.file_idx])
        self.data = load_byte_shard(path)
        self.chunks = self._make_chunks()
        if self.shuffle:
            perm = torch.randperm(len(self.chunks))
            self.chunks = self.chunks[perm]
        self.pos = 0

    def _make_chunks(self) -> torch.Tensor:
        """Convert raw byte shard into chunked training data."""
        K = self.chunk_size
        pb = self.piece_bytes
        n_pieces = len(self.data) // pb

        if n_pieces == 0:
            return torch.empty(0, K, dtype=torch.long)

        # Trim to whole pieces
        raw = self.data[: n_pieces * pb].reshape(n_pieces, pb)

        # Frame each piece: [BOS, byte+3, byte+3, ..., EOS]
        # Total token length = pb + 2, chunked into ceil((pb+2)/K) chunks
        seq_len = pb + 2
        n_chunks_per_piece = (seq_len + K - 1) // K
        padded_len = n_chunks_per_piece * K  # pad to multiple of chunk_size

        # Build full token sequences (n_pieces, padded_len)
        seqs = np.full((n_pieces, padded_len), PAD, dtype=np.int64)
        seqs[:, 0] = BOS
        seqs[:, 1 : pb + 1] = raw.astype(np.int64) + BYTE_OFFSET
        seqs[:, pb + 1] = EOS

        # Reshape into chunks: (n_pieces, n_chunks_per_piece, K)
        seqs = seqs.reshape(n_pieces, n_chunks_per_piece, K)

        # Flatten to (n_pieces * n_chunks_per_piece, K)
        return torch.from_numpy(seqs.reshape(-1, K))

    def _advance_shard(self):
        self.file_idx = (self.file_idx + self.world_size) % len(self.files)
        self._load_shard()

    def next_batch(self, device: torch.device) -> torch.Tensor:
        """Returns (batch_size, chunk_size) long tensor of byte token IDs.

        10% of the batch is replaced with augmented chunks:
        - Half get random right-PAD (content truncated, EOS placed, rest PAD)
        - Half get BOS prepended (content shifted right, first byte becomes BOS)
        """
        if self.pos + self.batch_size > len(self.chunks):
            self._advance_shard()

        batch = self.chunks[self.pos : self.pos + self.batch_size].clone()
        self.pos += self.batch_size

        K = self.chunk_size
        n_aug = max(1, self.batch_size // 10)

        # Pick random indices to augment
        aug_idx = torch.randperm(self.batch_size)[:n_aug]
        n_pad = n_aug // 2
        n_bos = n_aug - n_pad

        # --- Right-PAD augmentation: truncate at random position, place EOS, PAD rest ---
        for i in aug_idx[:n_pad]:
            # Find how many content bytes (non-special) are in this chunk
            row = batch[i]
            # Pick a random truncation point: keep 1..K-2 tokens, leave room for EOS
            cut = torch.randint(1, K, (1,)).item()
            batch[i, cut] = EOS
            batch[i, cut + 1:] = PAD

        # --- BOS augmentation: prepend BOS, shift content right, drop last byte ---
        for i in aug_idx[n_pad:]:
            row = batch[i]
            batch[i, 1:] = row[:-1].clone()
            batch[i, 0] = BOS

        return batch.to(device)


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
