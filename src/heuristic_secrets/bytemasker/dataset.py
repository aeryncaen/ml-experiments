import base64
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from heuristic_secrets.data.types import SpanFinderSample


CACHE_DIR = Path(".cache/bytemasker")
DEFAULT_MAX_LEN = 512
DEFAULT_PREFIX_LEN = 64
DEFAULT_SUFFIX_LEN = 64
DEFAULT_MIDDLE_LEN = 384


@dataclass
class LineSample:
    bytes: bytes
    mask: list[int]
    has_secret: bool
    source: str
    categories: list[str] | None = None
    secret_ids: list[str] | None = None  # Unique IDs for each secret span in this sample


def extract_lines_with_masks(sample: SpanFinderSample) -> list[LineSample]:
    text_bytes = sample.text.encode("utf-8")
    
    pos_to_secret_ids: dict[int, set[str]] = {}
    for span_idx, (start, end) in enumerate(zip(sample.starts, sample.ends)):
        secret_id = f"{sample.source}:{span_idx}"
        for pos in range(start, min(end, len(text_bytes))):
            if pos not in pos_to_secret_ids:
                pos_to_secret_ids[pos] = set()
            pos_to_secret_ids[pos].add(secret_id)
    
    lines = []
    current_line_start = 0
    
    for i, byte in enumerate(text_bytes):
        if byte == ord('\n'):
            line_bytes = text_bytes[current_line_start:i]
            if len(line_bytes) > 0:
                mask = []
                line_secret_ids: set[str] = set()
                for j in range(len(line_bytes)):
                    pos = current_line_start + j
                    if pos in pos_to_secret_ids:
                        mask.append(1)
                        line_secret_ids.update(pos_to_secret_ids[pos])
                    else:
                        mask.append(0)
                lines.append(LineSample(
                    bytes=line_bytes,
                    mask=mask,
                    has_secret=len(line_secret_ids) > 0,
                    source=sample.source,
                    secret_ids=list(line_secret_ids) if line_secret_ids else None,
                ))
            current_line_start = i + 1
    
    if current_line_start < len(text_bytes):
        line_bytes = text_bytes[current_line_start:]
        if len(line_bytes) > 0:
            mask = []
            line_secret_ids = set()
            for j in range(len(line_bytes)):
                pos = current_line_start + j
                if pos in pos_to_secret_ids:
                    mask.append(1)
                    line_secret_ids.update(pos_to_secret_ids[pos])
                else:
                    mask.append(0)
            lines.append(LineSample(
                bytes=line_bytes,
                mask=mask,
                has_secret=len(line_secret_ids) > 0,
                source=sample.source,
                secret_ids=list(line_secret_ids) if line_secret_ids else None,
            ))
    
    return lines


def window_line(
    line: LineSample,
    max_len: int = DEFAULT_MAX_LEN,
    prefix_len: int = DEFAULT_PREFIX_LEN,
    suffix_len: int = DEFAULT_SUFFIX_LEN,
    middle_len: int = DEFAULT_MIDDLE_LEN,
) -> list[LineSample]:
    if len(line.bytes) <= max_len:
        return [line]
    
    prefix_bytes = line.bytes[:prefix_len]
    prefix_mask = line.mask[:prefix_len]
    suffix_bytes = line.bytes[-suffix_len:]
    suffix_mask = line.mask[-suffix_len:]
    
    middle_region_bytes = line.bytes[prefix_len:-suffix_len]
    middle_region_mask = line.mask[prefix_len:-suffix_len]
    
    windows = []
    for start in range(0, len(middle_region_bytes), middle_len):
        end = min(start + middle_len, len(middle_region_bytes))
        
        window_bytes = prefix_bytes + middle_region_bytes[start:end] + suffix_bytes
        window_mask = prefix_mask + middle_region_mask[start:end] + suffix_mask
        
        windows.append(LineSample(
            bytes=window_bytes,
            mask=window_mask,
            has_secret=any(m == 1 for m in window_mask),
            source=line.source,
            categories=line.categories,
            secret_ids=line.secret_ids if any(m == 1 for m in window_mask) else None,
        ))
    
    return windows


DEFAULT_MULTILINE_MAX_LINES = 9
DEFAULT_MULTILINE_OVERLAP = 2
DEFAULT_MULTILINE_MIN_LINES = 4
DEFAULT_MULTILINE_MAX_BYTES = 512


def extract_multiline_windows(
    sample: SpanFinderSample,
    max_lines: int = DEFAULT_MULTILINE_MAX_LINES,
    overlap: int = DEFAULT_MULTILINE_OVERLAP,
    min_lines: int = DEFAULT_MULTILINE_MIN_LINES,
    max_bytes: int = DEFAULT_MULTILINE_MAX_BYTES,
) -> list[LineSample]:
    lines = extract_lines_with_masks(sample)
    lines = [ln for ln in lines if ln.bytes.strip()]
    
    if len(lines) < min_lines:
        return []
    
    windows = []
    step = max_lines - overlap
    
    start = 0
    while start < len(lines):
        window_lines = []
        window_bytes_total = 0
        
        for i in range(start, min(start + max_lines, len(lines))):
            line_len = len(lines[i].bytes) + 1
            if window_bytes_total + line_len > max_bytes and len(window_lines) >= min_lines:
                break
            window_lines.append(lines[i])
            window_bytes_total += line_len
        
        if len(window_lines) < min_lines:
            break
        
        combined_bytes = b"\n".join(ln.bytes for ln in window_lines)
        combined_mask: list[int] = []
        combined_secret_ids: set[str] = set()
        
        for i, ln in enumerate(window_lines):
            combined_mask.extend(ln.mask)
            if i < len(window_lines) - 1:
                combined_mask.append(0)
            if ln.secret_ids:
                combined_secret_ids.update(ln.secret_ids)
        
        windows.append(LineSample(
            bytes=combined_bytes,
            mask=combined_mask,
            has_secret=len(combined_secret_ids) > 0,
            source=sample.source,
            secret_ids=list(combined_secret_ids) if combined_secret_ids else None,
        ))
        
        start += step
    
    return windows


def load_spanfinder_samples(path: Path) -> list[SpanFinderSample]:
    with open(path) as f:
        lines = f.readlines()
    samples = []
    for line in tqdm(lines, desc="Loading samples"):
        data = json.loads(line)
        samples.append(SpanFinderSample.from_dict(data))
    return samples


class LineMaskDataset(Dataset):
    def __init__(
        self,
        lines: list[LineSample],
        max_len: int = DEFAULT_MAX_LEN,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        suffix_len: int = DEFAULT_SUFFIX_LEN,
        middle_len: int = DEFAULT_MIDDLE_LEN,
        show_progress: bool = True,
    ):
        self.max_len = max_len
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.middle_len = middle_len
        
        windowed_lines = []
        for line in tqdm(lines, desc="Windowing lines", disable=not show_progress):
            windowed_lines.extend(window_line(line, max_len, prefix_len, suffix_len, middle_len))
        
        self.lines = windowed_lines
        self.secret_lines = [i for i, ln in enumerate(windowed_lines) if ln.has_secret]
        self.clean_lines = [i for i, ln in enumerate(windowed_lines) if not ln.has_secret]
        
        self.byte_tensors: list[torch.Tensor] = []
        self.mask_tensors: list[torch.Tensor] = []
        self.lengths: list[int] = []
        
        for line in tqdm(windowed_lines, desc="Tensorizing", disable=not show_progress):
            self.byte_tensors.append(torch.tensor(list(line.bytes), dtype=torch.long))
            self.mask_tensors.append(torch.tensor(line.mask, dtype=torch.float32))
            self.lengths.append(len(line.bytes))

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (bytes, mask) tensors."""
        return self.byte_tensors[idx], self.mask_tensors[idx]

    @classmethod
    def from_spanfinder_file(
        cls,
        path: Path,
        max_len: int = DEFAULT_MAX_LEN,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        suffix_len: int = DEFAULT_SUFFIX_LEN,
        middle_len: int = DEFAULT_MIDDLE_LEN,
    ) -> "LineMaskDataset":
        samples = load_spanfinder_samples(path)
        all_lines = []
        for sample in samples:
            all_lines.extend(extract_lines_with_masks(sample))
        return cls(all_lines, max_len=max_len, prefix_len=prefix_len, suffix_len=suffix_len, middle_len=middle_len)


def compute_length_bins(lengths: list[int], n_bins: int = 8) -> list[int]:
    if not lengths:
        return [512]
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    bins = []
    for i in range(1, n_bins):
        idx = int(n * i / n_bins)
        bins.append(sorted_lens[idx])
    bins.append(sorted_lens[-1])
    return sorted(set(bins))


def create_bucketed_batches(
    dataset: LineMaskDataset,
    batch_size: int,
    indices: list[int] | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    show_progress: bool = True,
    length_bins: list[int] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if seed is not None:
        random.seed(seed)
    
    if indices is None:
        indices = list(range(len(dataset)))
    
    if length_bins is None:
        length_bins = compute_length_bins([dataset.lengths[i] for i in indices])
    
    bins: dict[int, list[int]] = {b: [] for b in length_bins}
    bins[-1] = []
    
    for idx in indices:
        length = dataset.lengths[idx]
        for bin_max in length_bins:
            if length <= bin_max:
                bins[bin_max].append(idx)
                break
        else:
            bins[-1].append(idx)
    
    total_batches = sum(
        (len(bin_indices) + batch_size - 1) // batch_size
        for bin_indices in bins.values() if bin_indices
    )
    
    batches = []
    pbar = tqdm(total=total_batches, desc="Creating batches", disable=not show_progress)
    
    for bin_indices in bins.values():
        if not bin_indices:
            continue
        
        if shuffle:
            random.shuffle(bin_indices)
        
        for i in range(0, len(bin_indices), batch_size):
            batch_indices = bin_indices[i:i + batch_size]
            
            batch_bytes = [dataset.byte_tensors[j] for j in batch_indices]
            batch_masks = [dataset.mask_tensors[j] for j in batch_indices]
            batch_lengths = torch.tensor([dataset.lengths[j] for j in batch_indices])
            
            padded_bytes = pad_sequence(batch_bytes, batch_first=True, padding_value=0)
            padded_masks = pad_sequence(batch_masks, batch_first=True, padding_value=0.0)
            
            batches.append((padded_bytes, padded_masks, batch_lengths))
            pbar.update(1)
    
    pbar.close()
    
    if shuffle:
        random.shuffle(batches)
    
    return batches


@dataclass
class DatasetStats:
    """Statistics about a LineMaskDataset."""
    total_lines: int
    lines_with_secrets: int
    lines_without_secrets: int
    total_bytes: int
    secret_bytes: int
    avg_line_length: float
    max_line_length: int
    
    def __str__(self) -> str:
        return (
            f"Lines: {self.total_lines:,} "
            f"({self.lines_with_secrets:,} with secrets, "
            f"{self.lines_without_secrets:,} clean)\n"
            f"Bytes: {self.total_bytes:,} total, {self.secret_bytes:,} secret "
            f"({100*self.secret_bytes/self.total_bytes:.2f}%)\n"
            f"Line length: avg={self.avg_line_length:.1f}, max={self.max_line_length}"
        )


def compute_stats(dataset: LineMaskDataset) -> DatasetStats:
    """Compute statistics about a dataset."""
    total_bytes = sum(dataset.lengths)
    secret_bytes = sum(m.sum().item() for m in dataset.mask_tensors)
    
    return DatasetStats(
        total_lines=len(dataset),
        lines_with_secrets=len(dataset.secret_lines),
        lines_without_secrets=len(dataset.clean_lines),
        total_bytes=total_bytes,
        secret_bytes=int(secret_bytes),
        avg_line_length=total_bytes / len(dataset) if dataset else 0,
        max_line_length=max(dataset.lengths) if dataset.lengths else 0,
    )


def _compute_cache_key(
    data_dir: Path, split: str, max_len: int, prefix_len: int, suffix_len: int, middle_len: int
) -> str:
    splits_dir = Path(data_dir) / "splits" / "documents"
    path = splits_dir / f"{split}.jsonl"
    
    hash_parts = [str(data_dir.resolve()), split, str(max_len), str(prefix_len), str(suffix_len), str(middle_len)]
    if path.exists():
        hash_parts.append(f"{path}:{path.stat().st_mtime}")
    
    hash_str = "|".join(hash_parts)
    return hashlib.md5(hash_str.encode()).hexdigest()[:12]


def _get_cache_path(
    data_dir: Path, split: str, max_len: int, prefix_len: int, suffix_len: int, middle_len: int, secrets_only: bool = False
) -> Path:
    cache_key = _compute_cache_key(data_dir, split, max_len, prefix_len, suffix_len, middle_len)
    suffix = "_secrets" if secrets_only else ""
    return CACHE_DIR / f"bytemasker_{split}_max{max_len}_p{prefix_len}s{suffix_len}m{middle_len}{suffix}_{cache_key}.jsonl"


def _save_cache_jsonl(cache_path: Path, dataset: "LineMaskDataset", stats: DatasetStats) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        header = {
            "version": 2,
            "max_len": dataset.max_len,
            "prefix_len": dataset.prefix_len,
            "suffix_len": dataset.suffix_len,
            "middle_len": dataset.middle_len,
            "stats": stats.__dict__,
        }
        f.write(json.dumps(header) + '\n')
        for i in tqdm(range(len(dataset)), desc="Writing cache"):
            line = dataset.lines[i]
            row = {
                "b": base64.b64encode(line.bytes).decode('ascii'),
                "m": line.mask,
                "s": line.has_secret,
                "ids": line.secret_ids,
            }
            f.write(json.dumps(row) + '\n')
    print(f"Cached to {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")


def _load_cache_jsonl(cache_path: Path, secrets_only: bool = False) -> tuple["LineMaskDataset", DatasetStats] | None:
    with open(cache_path) as f:
        lines_raw = f.readlines()
    
    header = json.loads(lines_raw[0])
    if header.get("version", 1) < 2:
        print(f"Cache {cache_path} is outdated (no secret_ids), regenerating...")
        return None
    max_len = header["max_len"]
    prefix_len = header.get("prefix_len", DEFAULT_PREFIX_LEN)
    suffix_len = header.get("suffix_len", DEFAULT_SUFFIX_LEN)
    middle_len = header.get("middle_len", DEFAULT_MIDDLE_LEN)
    stats = DatasetStats(**header["stats"])
    
    dataset = LineMaskDataset.__new__(LineMaskDataset)
    dataset.lines = []
    dataset.max_len = max_len
    dataset.prefix_len = prefix_len
    dataset.suffix_len = suffix_len
    dataset.middle_len = middle_len
    dataset.byte_tensors = []
    dataset.mask_tensors = []
    dataset.lengths = []
    dataset.secret_lines = []
    dataset.clean_lines = []
    
    idx = 0
    for line in tqdm(lines_raw[1:], desc="Loading cache"):
        row = json.loads(line)
        has_secret = row["s"]
        
        if secrets_only and not has_secret:
            continue
        
        raw_bytes = base64.b64decode(row["b"])
        mask = row["m"]
        secret_ids = row.get("ids")
        
        dataset.lines.append(LineSample(
            bytes=raw_bytes,
            mask=mask,
            has_secret=has_secret,
            source="",
            secret_ids=secret_ids,
        ))
        dataset.byte_tensors.append(torch.tensor(list(raw_bytes), dtype=torch.long))
        dataset.mask_tensors.append(torch.tensor(mask, dtype=torch.float32))
        dataset.lengths.append(len(raw_bytes))
        
        if has_secret:
            dataset.secret_lines.append(idx)
        else:
            dataset.clean_lines.append(idx)
        idx += 1
    
    if secrets_only:
        stats = compute_stats(dataset)
    
    return dataset, stats


def load_bytemasker_dataset(
    data_dir: Path,
    split: str = "train",
    max_len: int = DEFAULT_MAX_LEN,
    prefix_len: int = DEFAULT_PREFIX_LEN,
    suffix_len: int = DEFAULT_SUFFIX_LEN,
    middle_len: int = DEFAULT_MIDDLE_LEN,
    use_cache: bool = True,
    secrets_only: bool = False,
) -> tuple[LineMaskDataset, DatasetStats]:
    data_dir = Path(data_dir)
    cache_path = _get_cache_path(data_dir, split, max_len, prefix_len, suffix_len, middle_len, secrets_only=secrets_only)
    
    if use_cache and cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        result = _load_cache_jsonl(cache_path, secrets_only=False)
        if result is not None:
            return result
    
    full_cache_path = _get_cache_path(data_dir, split, max_len, prefix_len, suffix_len, middle_len, secrets_only=False)
    if use_cache and secrets_only and full_cache_path.exists():
        print(f"Loading from full cache, filtering to secrets only...")
        result = _load_cache_jsonl(full_cache_path, secrets_only=True)
        if result is not None:
            dataset, stats = result
            _save_cache_jsonl(cache_path, dataset, stats)
            return dataset, stats
    
    print(f"Building dataset (no cache at {cache_path})")
    splits_dir = data_dir / "splits" / "documents"
    path = splits_dir / f"{split}.jsonl"
    
    samples = load_spanfinder_samples(path)
    all_lines = []
    for sample in tqdm(samples, desc="Extracting lines"):
        lines = extract_lines_with_masks(sample)
        if secrets_only:
            lines = [ln for ln in lines if ln.has_secret]
        all_lines.extend(lines)
    
    dataset = LineMaskDataset(
        all_lines, max_len=max_len, prefix_len=prefix_len, suffix_len=suffix_len, middle_len=middle_len
    )
    stats = compute_stats(dataset)
    
    if use_cache:
        _save_cache_jsonl(cache_path, dataset, stats)
    
    return dataset, stats


def load_mixed_windows(
    data_dir: Path,
    split: str = "train",
    max_len: int = DEFAULT_MAX_LEN,
    prefix_len: int = DEFAULT_PREFIX_LEN,
    suffix_len: int = DEFAULT_SUFFIX_LEN,
    middle_len: int = DEFAULT_MIDDLE_LEN,
    multiline_max_lines: int = DEFAULT_MULTILINE_MAX_LINES,
    multiline_overlap: int = DEFAULT_MULTILINE_OVERLAP,
    multiline_min_lines: int = DEFAULT_MULTILINE_MIN_LINES,
    multiline_max_bytes: int = DEFAULT_MULTILINE_MAX_BYTES,
    show_progress: bool = True,
) -> tuple[list[LineSample], set[str]]:
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits" / "documents"
    path = splits_dir / f"{split}.jsonl"
    
    samples = load_spanfinder_samples(path)
    
    all_windows: list[LineSample] = []
    all_secret_ids: set[str] = set()
    
    for sample in tqdm(samples, desc="Extracting windows", disable=not show_progress):
        single_lines = extract_lines_with_masks(sample)
        for ln in single_lines:
            windowed = window_line(ln, max_len, prefix_len, suffix_len, middle_len)
            all_windows.extend(windowed)
            if ln.secret_ids:
                all_secret_ids.update(ln.secret_ids)
        
        multi_windows = extract_multiline_windows(
            sample,
            max_lines=multiline_max_lines,
            overlap=multiline_overlap,
            min_lines=multiline_min_lines,
            max_bytes=multiline_max_bytes,
        )
        all_windows.extend(multi_windows)
        for mw in multi_windows:
            if mw.secret_ids:
                all_secret_ids.update(mw.secret_ids)
    
    return all_windows, all_secret_ids
