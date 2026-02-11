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


def build_byte_token_map(unique_bytes: np.ndarray | None = None) -> np.ndarray:
    """Build a byte-value -> token-id lookup table (size 256, dtype int64).

    If unique_bytes is None, uses the standard byte mapping: token = byte + BYTE_OFFSET.
    If unique_bytes is provided, maps those bytes to dense IDs starting at BYTE_OFFSET.
    Unmapped bytes get token 0 (PAD) — should never appear if data is clean.
    """
    table = np.zeros(256, dtype=np.int64)  # unmapped -> 0 (PAD)
    if unique_bytes is None:
        # Standard: byte b -> token b + 3
        for b in range(256):
            table[b] = b + BYTE_OFFSET
    else:
        for i, b in enumerate(sorted(unique_bytes)):
            table[b] = i + BYTE_OFFSET
    return table


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
# Piece-level loader for STLG training (serves full pieces, not individual chunks)
# ---------------------------------------------------------------------------

class PieceStream:
    """Streams full pieces (B, n_chunks, chunk_size) for STLG training.

    Each piece is a sequence of chunks that belong together — the STLG
    needs to see the full ordered sequence to do next-latent prediction.
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
        self.piece_bytes = chunk_size * 16
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle

        self.file_idx = rank % len(self.files)
        self.pieces: torch.Tensor = torch.empty(0)
        self.pos = 0
        self._load_shard()

    def _load_shard(self):
        path = Path(self.files[self.file_idx])
        data = load_byte_shard(path)
        K = self.chunk_size
        pb = self.piece_bytes
        n_pieces = len(data) // pb

        if n_pieces == 0:
            self.pieces = torch.empty(0, 1, K, dtype=torch.long)
            self.pos = 0
            return

        raw = data[: n_pieces * pb].reshape(n_pieces, pb)

        seq_len = pb + 2
        n_chunks = (seq_len + K - 1) // K
        padded_len = n_chunks * K

        seqs = np.full((n_pieces, padded_len), PAD, dtype=np.int64)
        seqs[:, 0] = BOS
        seqs[:, 1 : pb + 1] = raw.astype(np.int64) + BYTE_OFFSET
        seqs[:, pb + 1] = EOS

        # (n_pieces, n_chunks, K)
        self.pieces = torch.from_numpy(seqs.reshape(n_pieces, n_chunks, K))
        self.n_chunks = n_chunks

        if self.shuffle:
            perm = torch.randperm(len(self.pieces))
            self.pieces = self.pieces[perm]
        self.pos = 0

    def _advance_shard(self):
        self.file_idx = (self.file_idx + self.world_size) % len(self.files)
        self._load_shard()

    def next_batch(self, device: torch.device) -> torch.Tensor:
        """Returns (batch_size, n_chunks, chunk_size) long tensor."""
        if self.pos + self.batch_size > len(self.pieces):
            self._advance_shard()

        batch = self.pieces[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
        return batch.to(device)


# ---------------------------------------------------------------------------
# Shakespeare data loader
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_shakespeare_bytes(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray]:
    """Download tinyshakespeare and return (uint8 data, byte_map).

    byte_map is a size-256 int64 lookup table mapping byte values to dense
    token IDs starting at BYTE_OFFSET (3). Shakespeare has 65 unique chars,
    so token IDs are 3-67, vocab_size=68.
    """
    data_path = Path(data_dir) / "shakespeare.txt"
    if not data_path.exists():
        print(f"Downloading Shakespeare -> {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(SHAKESPEARE_URL, data_path)
    text = data_path.read_text()
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    unique_bytes = np.array(sorted(set(data.tolist())), dtype=np.uint8)
    byte_map = build_byte_token_map(unique_bytes)
    return data, byte_map


def _make_chunks_from_bytes(data: np.ndarray, chunk_size: int,
                            byte_map: np.ndarray | None = None) -> torch.Tensor:
    """Convert raw bytes into chunked training data with BOS/EOS/PAD framing.

    Same logic as ByteShardStream._make_chunks: takes pieces of chunk_size*16
    bytes, frames as [BOS, bytes+3, ..., EOS], chunks into chunk_size-width.

    If byte_map is provided (size-256 lookup table), uses it to convert byte
    values to token IDs. Otherwise uses the default byte + BYTE_OFFSET.
    """
    K = chunk_size
    pb = K * 16  # piece bytes
    n_pieces = len(data) // pb

    if n_pieces == 0:
        return torch.empty(0, K, dtype=torch.long)

    raw = data[: n_pieces * pb].reshape(n_pieces, pb)
    seq_len = pb + 2
    n_chunks_per_piece = (seq_len + K - 1) // K
    padded_len = n_chunks_per_piece * K

    seqs = np.full((n_pieces, padded_len), PAD, dtype=np.int64)
    seqs[:, 0] = BOS
    if byte_map is not None:
        seqs[:, 1 : pb + 1] = byte_map[raw]
    else:
        seqs[:, 1 : pb + 1] = raw.astype(np.int64) + BYTE_OFFSET
    seqs[:, pb + 1] = EOS

    return torch.from_numpy(seqs.reshape(-1, K))


def _make_pieces_from_bytes(data: np.ndarray, chunk_size: int,
                            byte_map: np.ndarray | None = None) -> torch.Tensor:
    """Convert raw bytes into pieces (n_pieces, n_chunks_per_piece, chunk_size)."""
    K = chunk_size
    pb = K * 16
    n_pieces = len(data) // pb

    if n_pieces == 0:
        return torch.empty(0, 1, K, dtype=torch.long)

    raw = data[: n_pieces * pb].reshape(n_pieces, pb)
    seq_len = pb + 2
    n_chunks = (seq_len + K - 1) // K
    padded_len = n_chunks * K

    seqs = np.full((n_pieces, padded_len), PAD, dtype=np.int64)
    seqs[:, 0] = BOS
    if byte_map is not None:
        seqs[:, 1 : pb + 1] = byte_map[raw]
    else:
        seqs[:, 1 : pb + 1] = raw.astype(np.int64) + BYTE_OFFSET
    seqs[:, pb + 1] = EOS

    return torch.from_numpy(seqs.reshape(n_pieces, n_chunks, K))


class ShakespeareStream:
    """Streams shuffled byte chunks from tinyshakespeare for VAE training.

    Same chunk format as ByteShardStream. Splits data 90/10 train/val.
    """

    def __init__(self, chunk_size: int, batch_size: int, split: str = "train",
                 data_dir: str = "data"):
        data, byte_map = load_shakespeare_bytes(data_dir)
        n = len(data)
        split_idx = int(n * 0.9)

        if split == "train":
            data = data[:split_idx]
        else:
            data = data[split_idx:]

        self.chunks = _make_chunks_from_bytes(data, chunk_size, byte_map)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.pos = 0
        self._shuffle()

    def _shuffle(self):
        perm = torch.randperm(len(self.chunks))
        self.chunks = self.chunks[perm]
        self.pos = 0

    def next_batch(self, device: torch.device) -> torch.Tensor:
        if self.pos + self.batch_size > len(self.chunks):
            self._shuffle()

        batch = self.chunks[self.pos : self.pos + self.batch_size].clone()
        self.pos += self.batch_size

        # 10% augmentation (same as ByteShardStream)
        K = self.chunk_size
        n_aug = max(1, self.batch_size // 10)
        aug_idx = torch.randperm(self.batch_size)[:n_aug]
        n_pad = n_aug // 2

        for i in aug_idx[:n_pad]:
            cut = torch.randint(1, K, (1,)).item()
            batch[i, cut] = EOS
            batch[i, cut + 1:] = PAD

        for i in aug_idx[n_pad:]:
            row = batch[i]
            batch[i, 1:] = row[:-1].clone()
            batch[i, 0] = BOS

        return batch.to(device)


class ShakespearePieceStream:
    """Streams full pieces from tinyshakespeare for STLG training."""

    def __init__(self, chunk_size: int, batch_size: int, split: str = "train",
                 data_dir: str = "data"):
        data, byte_map = load_shakespeare_bytes(data_dir)
        n = len(data)
        split_idx = int(n * 0.9)

        if split == "train":
            data = data[:split_idx]
        else:
            data = data[split_idx:]

        self.pieces = _make_pieces_from_bytes(data, chunk_size, byte_map)
        self.batch_size = batch_size
        self.pos = 0
        self._shuffle()

    def _shuffle(self):
        perm = torch.randperm(len(self.pieces))
        self.pieces = self.pieces[perm]
        self.pos = 0

    def next_batch(self, device: torch.device) -> torch.Tensor:
        if self.pos + self.batch_size > len(self.pieces):
            self._shuffle()

        batch = self.pieces[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size
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
