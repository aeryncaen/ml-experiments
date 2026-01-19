"""SpanFinder dataset with chunking matching inference behavior."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from heuristic_secrets.data.types import SpanFinderSample
from heuristic_secrets.pipeline.chunker import Chunker


@dataclass
class SpanFinderDatasetStats:
    total_samples: int
    total_chunks: int
    total_spans: int
    avg_chunks_per_sample: float
    chunk_size: int
    chunk_overlap: int


def load_jsonl(path: Path) -> list[SpanFinderSample]:
    """Load SpanFinderSamples from a JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(SpanFinderSample.from_dict(data))
    return samples


@dataclass
class ChunkedSample:
    """A single chunk from a document with span labels."""
    byte_ids: torch.Tensor  # (chunk_size,) - byte values 0-255
    targets: torch.Tensor   # (chunk_size, 2) - [start_label, end_label] per position
    length: int             # actual length (before padding)
    is_first: bool
    is_last: bool


class SpanFinderDataset(Dataset):
    """Dataset for SpanFinder training using chunked samples.
    
    Matches inference behavior: splits documents into overlapping chunks
    and trains on each chunk independently. Spans that cross chunk boundaries
    result in partial labels (start-only or end-only within the chunk).
    """

    def __init__(
        self,
        samples: list[SpanFinderSample],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.samples = samples
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = Chunker(chunk_size, chunk_overlap)

        # Precompute all chunks
        self.chunks: list[ChunkedSample] = []
        self._precompute()

    def _precompute(self) -> None:
        for sample in self.samples:
            data = sample.text.encode("utf-8")
            
            # Convert span positions to sets for O(1) lookup
            # Note: ends in SpanFinderSample are exclusive, so end-1 is the last byte
            start_positions = set(sample.starts)
            end_positions = set(e - 1 for e in sample.ends)  # Convert to inclusive

            for chunk in self.chunker.chunk(data):
                chunk_len = len(chunk.data)
                
                # Create byte tensor
                byte_ids = torch.tensor(list(chunk.data), dtype=torch.long)
                
                # Create target labels for this chunk
                # Position i in chunk corresponds to position (chunk.start + i) in document
                targets = torch.zeros(chunk_len, 2, dtype=torch.float32)
                
                for i in range(chunk_len):
                    doc_pos = chunk.start + i
                    if doc_pos in start_positions:
                        targets[i, 0] = 1.0  # Start marker
                    if doc_pos in end_positions:
                        targets[i, 1] = 1.0  # End marker

                self.chunks.append(ChunkedSample(
                    byte_ids=byte_ids,
                    targets=targets,
                    length=chunk_len,
                    is_first=chunk.is_first,
                    is_last=chunk.is_last,
                ))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        return chunk.byte_ids, chunk.targets


def collate_chunks(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that pads chunks to same length.
    
    Returns:
        byte_ids: (batch, max_chunk_len) - padded byte sequences
        targets: (batch, max_chunk_len, 2) - padded target labels
        lengths: (batch,) - original chunk lengths
    """
    byte_seqs, target_seqs = zip(*batch)
    lengths = torch.tensor([len(b) for b in byte_seqs], dtype=torch.long)
    max_len = int(lengths.max().item())

    # Pad byte sequences
    padded_bytes = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(byte_seqs):
        padded_bytes[i, :len(seq)] = seq

    # Pad targets
    padded_targets = torch.zeros(len(batch), max_len, 2, dtype=torch.float32)
    for i, target in enumerate(target_seqs):
        padded_targets[i, :len(target)] = target

    return padded_bytes, padded_targets, lengths


def create_batches(
    dataset: SpanFinderDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create batches from chunked dataset.
    
    Since all chunks are ~chunk_size (except possibly the last chunk of each doc),
    we don't need complex bucketing - just batch them directly.
    
    Returns list of (byte_ids, targets, lengths) tuples.
    """
    import random

    if seed is not None:
        random.seed(seed)

    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = [dataset[j] for j in batch_indices]
        byte_ids, targets, lengths = collate_chunks(batch_data)
        batches.append((byte_ids, targets, lengths))

    return batches


def load_spanfinder_dataset(
    data_dir: Path,
    split: str = "train",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> tuple[SpanFinderDataset, SpanFinderDatasetStats]:
    """Load a SpanFinder dataset with chunking.

    Args:
        data_dir: Root data directory (contains splits/documents/)
        split: Which split to load ("train", "val", "test")
        chunk_size: Size of each chunk in bytes (default 512)
        chunk_overlap: Overlap between chunks (default 64)

    Returns:
        Tuple of (dataset, stats)
    """
    splits_dir = Path(data_dir) / "splits" / "documents"
    samples = load_jsonl(splits_dir / f"{split}.jsonl")

    dataset = SpanFinderDataset(
        samples=samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    total_spans = sum(len(s.starts) for s in samples)

    stats = SpanFinderDatasetStats(
        total_samples=len(samples),
        total_chunks=len(dataset),
        total_spans=total_spans,
        avg_chunks_per_sample=len(dataset) / len(samples) if samples else 0,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return dataset, stats
