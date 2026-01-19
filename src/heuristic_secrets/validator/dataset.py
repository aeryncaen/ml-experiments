import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from heuristic_secrets.validator.features import (
    FrequencyTable,
    extract_features,
)
from heuristic_secrets.data.types import ValidatorSample


# Cache directory for precomputed datasets
CACHE_DIR = Path(".cache/validator")


@dataclass
class ValidatorDatasetStats:
    total_samples: int
    positive_samples: int
    negative_samples: int
    text_freq: FrequencyTable
    innocuous_freq: FrequencyTable
    secret_freq: FrequencyTable


def load_jsonl(path: Path) -> list[ValidatorSample]:
    samples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(ValidatorSample.from_dict(data))
    return samples


def build_text_freq_from_documents(data_dir: Path) -> FrequencyTable:
    from heuristic_secrets.data.types import SpanFinderSample
    
    docs_path = data_dir / "splits" / "documents" / "train.jsonl"
    if not docs_path.exists():
        return FrequencyTable({})
    
    counter: Counter[int] = Counter()
    total = 0
    
    with open(docs_path) as f:
        for line in f:
            data = json.loads(line)
            sample = SpanFinderSample.from_dict(data)
            text_bytes = sample.text.encode("utf-8")
            
            # Create mask of secret positions
            secret_positions = set()
            for start, end in zip(sample.starts, sample.ends):
                for pos in range(start, end):
                    if pos < len(text_bytes):
                        secret_positions.add(pos)
            
            # Count only non-secret bytes
            for pos, byte in enumerate(text_bytes):
                if pos not in secret_positions:
                    counter[byte] += 1
                    total += 1
    
    if total == 0:
        return FrequencyTable({})
    
    return FrequencyTable({byte: count / total for byte, count in counter.items()})


def build_frequency_table(samples: list[ValidatorSample]) -> FrequencyTable:
    counter: Counter[int] = Counter()
    total = 0
    for sample in samples:
        data = sample.text.encode("utf-8")
        counter.update(data)
        total += len(data)

    if total == 0:
        return FrequencyTable({})

    return FrequencyTable({byte: count / total for byte, count in counter.items()})





VALIDATOR_MAX_BYTE_LEN = 4096


class ValidatorDataset(Dataset):
    def __init__(
        self,
        samples: list[ValidatorSample],
        text_freq: FrequencyTable,
        innocuous_freq: FrequencyTable,
        secret_freq: FrequencyTable,
        feature_indices: list[int] | None = None,
        max_byte_len: int = VALIDATOR_MAX_BYTE_LEN,
    ):
        self.samples = samples
        self.text_freq = text_freq
        self.innocuous_freq = innocuous_freq
        self.secret_freq = secret_freq
        self.feature_indices = feature_indices
        self.max_byte_len = max_byte_len

        self.features: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.byte_ids: list[torch.Tensor] = []
        self.lengths: list[int] = []

        self._precompute_features()

    def _precompute_features(self) -> None:
        for sample in self.samples:
            data = sample.text.encode("utf-8")
            feat = extract_features(
                data,
                self.text_freq,
                self.innocuous_freq,
                self.secret_freq,
            )
            feat_tensor = torch.tensor(feat.to_list(), dtype=torch.float32)
            if self.feature_indices is not None:
                feat_tensor = feat_tensor[self.feature_indices]
            self.features.append(feat_tensor)
            self.labels.append(torch.tensor([sample.label], dtype=torch.float32))
            truncated = data[:self.max_byte_len]
            self.byte_ids.append(torch.tensor(list(truncated), dtype=torch.long))
            self.lengths.append(len(truncated))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.byte_ids[idx]

    def get_tensors(self, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return all features, labels, and byte_ids as stacked tensors.
        
        Args:
            max_len: If provided, pad byte_ids to this length. Otherwise use max in dataset.
        """
        X = torch.stack(self.features)
        y = torch.stack(self.labels)
        
        if max_len is None:
            max_len = max(self.lengths)
        
        padded_bytes = []
        for byte_tensor in self.byte_ids:
            if len(byte_tensor) < max_len:
                padding = torch.zeros(max_len - len(byte_tensor), dtype=torch.long)
                padded = torch.cat([byte_tensor, padding])
            else:
                padded = byte_tensor[:max_len]
            padded_bytes.append(padded)
        
        byte_ids = torch.stack(padded_bytes)
        return X, y, byte_ids


LENGTH_BINS = [64, 128, 256, 512, 1024, 2048, 4096]


def create_bucketed_batches(
    dataset: ValidatorDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create batches bucketed by sequence length to minimize padding."""
    import random
    
    if seed is not None:
        random.seed(seed)
    
    # Use -1 as the overflow bin key (sequences longer than largest bin)
    OVERFLOW_BIN = -1
    bins: dict[int, list[int]] = {b: [] for b in LENGTH_BINS}
    bins[OVERFLOW_BIN] = []
    
    for idx, length in enumerate(dataset.lengths):
        for bin_max in LENGTH_BINS:
            if length <= bin_max:
                bins[bin_max].append(idx)
                break
        else:
            bins[OVERFLOW_BIN].append(idx)
    
    batches = []
    for bin_max, indices in bins.items():
        if not indices:
            continue
        
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            batch_features = [dataset.features[j] for j in batch_indices]
            batch_labels = [dataset.labels[j] for j in batch_indices]
            batch_bytes = [dataset.byte_ids[j] for j in batch_indices]
            
            max_len = max(len(b) for b in batch_bytes)
            padded_bytes = []
            for b in batch_bytes:
                if len(b) < max_len:
                    padding = torch.zeros(max_len - len(b), dtype=torch.long)
                    padded_bytes.append(torch.cat([b, padding]))
                else:
                    padded_bytes.append(b)
            
            batches.append((
                torch.stack(batch_features),
                torch.stack(batch_labels),
                torch.stack(padded_bytes),
            ))
    
    if shuffle:
        random.shuffle(batches)
    
    return batches


def _compute_cache_key(data_dir: Path, split: str, feature_indices: list[int] | None, max_byte_len: int) -> str:
    splits_dir = Path(data_dir) / "splits" / "validator"
    docs_dir = Path(data_dir) / "splits" / "documents"
    
    files_to_check = [
        splits_dir / "train.jsonl",
        splits_dir / f"{split}.jsonl" if split != "train" else None,
        docs_dir / "train.jsonl",
    ]
    
    hash_parts = [str(data_dir.resolve()), split, str(feature_indices), str(max_byte_len)]
    for f in files_to_check:
        if f is not None and f.exists():
            hash_parts.append(f"{f}:{f.stat().st_mtime}")
    
    hash_str = "|".join(hash_parts)
    return hashlib.md5(hash_str.encode()).hexdigest()[:12]


def _get_cache_path(data_dir: Path, split: str, feature_indices: list[int] | None, max_byte_len: int) -> Path:
    cache_key = _compute_cache_key(data_dir, split, feature_indices, max_byte_len)
    feat_str = "all" if feature_indices is None else "-".join(map(str, feature_indices))
    return CACHE_DIR / f"validator_{split}_f{feat_str}_b{max_byte_len}_{cache_key}.pt"


def load_validator_dataset(
    data_dir: Path,
    split: str = "train",
    feature_indices: list[int] | None = None,
    max_byte_len: int = VALIDATOR_MAX_BYTE_LEN,
    use_cache: bool = True,
) -> tuple[ValidatorDataset, ValidatorDatasetStats]:
    data_dir = Path(data_dir)
    cache_path = _get_cache_path(data_dir, split, feature_indices, max_byte_len)
    
    if use_cache and cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        
        dataset = ValidatorDataset.__new__(ValidatorDataset)
        dataset.samples = cached["samples"]
        dataset.text_freq = cached["text_freq"]
        dataset.innocuous_freq = cached["innocuous_freq"]
        dataset.secret_freq = cached["secret_freq"]
        dataset.feature_indices = cached["feature_indices"]
        dataset.max_byte_len = cached.get("max_byte_len", VALIDATOR_MAX_BYTE_LEN)
        dataset.features = cached["features"]
        dataset.labels = cached["labels"]
        dataset.byte_ids = cached["byte_ids"]
        dataset.lengths = cached["lengths"]
        
        stats = ValidatorDatasetStats(
            total_samples=len(dataset.samples),
            positive_samples=sum(1 for s in dataset.samples if s.label == 1),
            negative_samples=sum(1 for s in dataset.samples if s.label == 0),
            text_freq=dataset.text_freq,
            innocuous_freq=dataset.innocuous_freq,
            secret_freq=dataset.secret_freq,
        )
        return dataset, stats
    
    print(f"Building dataset (no cache found at {cache_path})")
    splits_dir = data_dir / "splits" / "validator"

    train_samples = load_jsonl(splits_dir / "train.jsonl")

    secrets = [s for s in train_samples if s.label == 1]
    innocuous = [s for s in train_samples if s.label == 0]

    text_freq = build_text_freq_from_documents(data_dir)
    # innocuous_freq from false positive candidates (UUIDs, hashes, etc.)
    innocuous_freq = build_frequency_table(innocuous)
    # secret_freq from actual secrets
    secret_freq = build_frequency_table(secrets)

    if split == "train":
        samples = train_samples
    else:
        samples = load_jsonl(splits_dir / f"{split}.jsonl")

    dataset = ValidatorDataset(
        samples=samples,
        text_freq=text_freq,
        innocuous_freq=innocuous_freq,
        secret_freq=secret_freq,
        feature_indices=feature_indices,
        max_byte_len=max_byte_len,
    )

    stats = ValidatorDatasetStats(
        total_samples=len(samples),
        positive_samples=sum(1 for s in samples if s.label == 1),
        negative_samples=sum(1 for s in samples if s.label == 0),
        text_freq=text_freq,
        innocuous_freq=innocuous_freq,
        secret_freq=secret_freq,
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "samples": dataset.samples,
            "text_freq": dataset.text_freq,
            "innocuous_freq": dataset.innocuous_freq,
            "secret_freq": dataset.secret_freq,
            "feature_indices": dataset.feature_indices,
            "max_byte_len": dataset.max_byte_len,
            "features": dataset.features,
            "labels": dataset.labels,
            "byte_ids": dataset.byte_ids,
            "lengths": dataset.lengths,
        }, cache_path)
        print(f"Cached dataset to {cache_path}")

    return dataset, stats
