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

from heuristic_secrets.data.types import SpanFinderSample, SecretCategory, CATEGORY_LABELS


CACHE_DIR = Path(".cache/bytemasker")
DEFAULT_MAX_LEN = 512
DEFAULT_PREFIX_LEN = 64
DEFAULT_SUFFIX_LEN = 64
DEFAULT_MIDDLE_LEN = 384


def _content_hash_secret_id(secret_bytes: bytes) -> str:
    """Generate a content-based secret ID from the secret bytes.
    
    This ensures the same secret text always gets the same ID,
    regardless of which file/source it appears in. Critical for
    preventing data leakage during train/val splitting.
    """
    return hashlib.sha256(secret_bytes).hexdigest()[:16]


MIN_WINDOW_LENGTH = 7


@dataclass
class LineSample:
    bytes: bytes
    mask: list[int]
    has_secret: bool
    source: str
    categories: list[str] | None = None
    secret_ids: list[str] | None = None


def extract_lines_with_masks(sample: SpanFinderSample) -> list[LineSample]:
    text_bytes = sample.text.encode("utf-8")
    
    pos_to_secret_ids: dict[int, set[str]] = {}
    pos_to_categories: dict[int, set[str]] = {}
    
    for span_idx, (start, end) in enumerate(zip(sample.starts, sample.ends)):
        category = "unknown"
        is_true_secret = True
        if sample.categories and span_idx < len(sample.categories):
            category = sample.categories[span_idx]
            try:
                is_true_secret = CATEGORY_LABELS.get(SecretCategory(category), 0) == 1
            except ValueError:
                is_true_secret = not category.startswith("fp_")
        
        if not is_true_secret:
            continue
        
        secret_bytes = text_bytes[start:min(end, len(text_bytes))]
        secret_id = _content_hash_secret_id(secret_bytes)
        for pos in range(start, min(end, len(text_bytes))):
            if pos not in pos_to_secret_ids:
                pos_to_secret_ids[pos] = set()
            pos_to_secret_ids[pos].add(secret_id)
            if pos not in pos_to_categories:
                pos_to_categories[pos] = set()
            pos_to_categories[pos].add(category)
    
    lines = []
    current_line_start = 0
    
    for i, byte in enumerate(text_bytes):
        if byte == ord('\n'):
            line_bytes = text_bytes[current_line_start:i]
            if len(line_bytes) > 0 and line_bytes.strip():
                mask = []
                line_secret_ids: set[str] = set()
                line_categories: set[str] = set()
                for j in range(len(line_bytes)):
                    pos = current_line_start + j
                    if pos in pos_to_secret_ids:
                        mask.append(1)
                        line_secret_ids.update(pos_to_secret_ids[pos])
                        line_categories.update(pos_to_categories[pos])
                    else:
                        mask.append(0)
                lines.append(LineSample(
                    bytes=line_bytes,
                    mask=mask,
                    has_secret=len(line_secret_ids) > 0,
                    source=sample.source,
                    categories=list(line_categories) if line_categories else None,
                    secret_ids=list(line_secret_ids) if line_secret_ids else None,
                ))
            current_line_start = i + 1
    
    if current_line_start < len(text_bytes):
        line_bytes = text_bytes[current_line_start:]
        if len(line_bytes) > 0 and line_bytes.strip():
            mask = []
            line_secret_ids: set[str] = set()
            line_categories: set[str] = set()
            for j in range(len(line_bytes)):
                pos = current_line_start + j
                if pos in pos_to_secret_ids:
                    mask.append(1)
                    line_secret_ids.update(pos_to_secret_ids[pos])
                    line_categories.update(pos_to_categories[pos])
                else:
                    mask.append(0)
            lines.append(LineSample(
                bytes=line_bytes,
                mask=mask,
                has_secret=len(line_secret_ids) > 0,
                source=sample.source,
                categories=list(line_categories) if line_categories else None,
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
    lines = [ln for ln in lines if ln.bytes.strip() and len(ln.bytes) <= max_bytes]
    
    if len(lines) < 2:
        return []
    
    windows = []
    start = 0
    while start < len(lines):
        window_lines = []
        window_bytes_total = 0
        
        for i in range(start, min(start + max_lines, len(lines))):
            line_len = len(lines[i].bytes) + 1
            if window_bytes_total + line_len > max_bytes:
                break
            window_lines.append(lines[i])
            window_bytes_total += line_len
        
        if len(window_lines) >= 2:
            combined_bytes = b"\n".join(ln.bytes for ln in window_lines)
            combined_mask: list[int] = []
            combined_secret_ids: set[str] = set()
            combined_categories: set[str] = set()
            
            for j, ln in enumerate(window_lines):
                combined_mask.extend(ln.mask)
                if j < len(window_lines) - 1:
                    combined_mask.append(0)
                if ln.secret_ids:
                    combined_secret_ids.update(ln.secret_ids)
                if ln.categories:
                    combined_categories.update(ln.categories)
            
            windows.append(LineSample(
                bytes=combined_bytes,
                mask=combined_mask,
                has_secret=len(combined_secret_ids) > 0,
                source=sample.source,
                categories=list(combined_categories) if combined_categories else None,
                secret_ids=list(combined_secret_ids) if combined_secret_ids else None,
            ))
        
        step = max(1, len(window_lines) - overlap)
        start += step
    
    return windows


def extract_secret_only_samples(
    sample: SpanFinderSample,
    max_bytes: int = DEFAULT_MAX_LEN,
) -> list[LineSample]:
    if not sample.starts:
        return []
    
    text_bytes = sample.text.encode("utf-8")
    samples = []
    
    for span_idx, (start, end) in enumerate(zip(sample.starts, sample.ends)):
        original_secret_bytes = text_bytes[start:end]
        if len(original_secret_bytes) == 0:
            continue
        
        secret_id = _content_hash_secret_id(original_secret_bytes)
        
        secret_bytes = original_secret_bytes
        if len(secret_bytes) > max_bytes:
            secret_bytes = secret_bytes[:max_bytes]
        
        category = None
        is_true_secret = True
        if sample.categories and span_idx < len(sample.categories):
            category = sample.categories[span_idx]
            try:
                is_true_secret = CATEGORY_LABELS.get(SecretCategory(category), 0) == 1
            except ValueError:
                is_true_secret = not category.startswith("fp_")
        mask = [1 if is_true_secret else 0] * len(secret_bytes)
        
        samples.append(LineSample(
            bytes=secret_bytes,
            mask=mask,
            has_secret=is_true_secret,
            source=sample.source,
            categories=[category] if category else None,
            secret_ids=[secret_id] if is_true_secret else None,
        ))
    
    return samples


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
            for w in window_line(line, max_len, prefix_len, suffix_len, middle_len):
                if len(w.bytes) >= MIN_WINDOW_LENGTH:
                    windowed_lines.append(w)
        
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
        if len(raw_bytes) < MIN_WINDOW_LENGTH:
            continue
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
            for w in windowed:
                if len(w.bytes) >= MIN_WINDOW_LENGTH:
                    all_windows.append(w)
                    if w.secret_ids:
                        all_secret_ids.update(w.secret_ids)
        
        multi_windows = extract_multiline_windows(
            sample,
            max_lines=multiline_max_lines,
            overlap=multiline_overlap,
            min_lines=multiline_min_lines,
            max_bytes=multiline_max_bytes,
        )
        for mw in multi_windows:
            if len(mw.bytes) >= MIN_WINDOW_LENGTH:
                all_windows.append(mw)
                if mw.secret_ids:
                    all_secret_ids.update(mw.secret_ids)
        
        secret_only = extract_secret_only_samples(sample, max_bytes=max_len)
        for so in secret_only:
            if len(so.bytes) >= MIN_WINDOW_LENGTH:
                all_windows.append(so)
                if so.secret_ids:
                    all_secret_ids.update(so.secret_ids)
    
    return all_windows, all_secret_ids


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}
    
    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def _build_sample_components(
    samples: list[LineSample],
) -> tuple[list[list[int]], dict[int, set[str]]]:
    """Group samples into components where samples sharing any secret_id are together.
    
    Returns:
        components: List of sample index lists (each component)
        component_categories: Map from component root to set of categories
    """
    uf = _UnionFind()
    secret_to_samples: dict[str, list[int]] = {}
    
    for idx, sample in enumerate(samples):
        uf.find(idx)
        if sample.secret_ids:
            for sid in sample.secret_ids:
                if sid not in secret_to_samples:
                    secret_to_samples[sid] = []
                secret_to_samples[sid].append(idx)
    
    for sid, sample_indices in secret_to_samples.items():
        for i in range(1, len(sample_indices)):
            uf.union(sample_indices[0], sample_indices[i])
    
    root_to_members: dict[int, list[int]] = {}
    for idx in range(len(samples)):
        root = uf.find(idx)
        if root not in root_to_members:
            root_to_members[root] = []
        root_to_members[root].append(idx)
    
    component_categories: dict[int, set[str]] = {}
    for root, members in root_to_members.items():
        cats: set[str] = set()
        for idx in members:
            sample_cats = samples[idx].categories
            if sample_cats is not None:
                cats.update(sample_cats)
            elif samples[idx].has_secret:
                cats.add("unknown_secret")
            else:
                cats.add("clean")
        component_categories[root] = cats
    
    components = list(root_to_members.values())
    comp_roots = list(root_to_members.keys())
    return components, {i: component_categories[comp_roots[i]] for i in range(len(components))}


def stratified_split_samples(
    samples: list[LineSample],
    train_ratio: float = 0.88,
    val_ratio: float = 0.12,
    seed: int | None = None,
) -> tuple[list[LineSample], list[LineSample]]:
    """Split samples with multi-label stratification, ensuring no secret leakage.
    
    Groups samples by shared secret_ids (same secret text = same group).
    Splits at component level with stratification by category distribution.
    """
    assert abs(train_ratio + val_ratio - 1.0) < 0.01
    
    if seed is not None:
        random.seed(seed)
    
    components, comp_categories = _build_sample_components(samples)
    
    all_categories = set()
    for cats in comp_categories.values():
        all_categories.update(cats)
    
    cat_counts: dict[str, int] = {cat: 0 for cat in all_categories}
    for comp_idx, cats in comp_categories.items():
        for cat in cats:
            cat_counts[cat] += 1
    
    cat_to_target: dict[str, dict[str, float]] = {}
    for cat in all_categories:
        total = cat_counts[cat]
        cat_to_target[cat] = {
            "train": total * train_ratio,
            "val": total * val_ratio,
        }
    
    split_counts: dict[str, dict[str, int]] = {
        cat: {"train": 0, "val": 0} for cat in all_categories
    }
    
    comp_indices = list(range(len(components)))
    random.shuffle(comp_indices)
    comp_indices.sort(key=lambda i: min(cat_counts[c] for c in comp_categories[i]))
    
    assignments: dict[int, str] = {}
    
    for comp_idx in comp_indices:
        cats = comp_categories[comp_idx]
        
        best_split = "train"
        best_score = float("-inf")
        
        for split in ["train", "val"]:
            score = 0.0
            for cat in cats:
                target = cat_to_target[cat][split]
                current = split_counts[cat][split]
                if target > 0:
                    score += (target - current) / target
            
            if score > best_score:
                best_score = score
                best_split = split
        
        assignments[comp_idx] = best_split
        for cat in cats:
            split_counts[cat][best_split] += 1
    
    train_samples: list[LineSample] = []
    val_samples: list[LineSample] = []
    
    for comp_idx, component in enumerate(components):
        split = assignments[comp_idx]
        for sample_idx in component:
            if split == "train":
                train_samples.append(samples[sample_idx])
            else:
                val_samples.append(samples[sample_idx])
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    return train_samples, val_samples


def load_all_documents(data_dir: Path, show_progress: bool = True) -> list[SpanFinderSample]:
    """Load all documents from train+val+test splits."""
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits" / "documents"
    
    all_samples: list[SpanFinderSample] = []
    for split_name in ["train", "val", "test"]:
        path = splits_dir / f"{split_name}.jsonl"
        if path.exists():
            samples = load_spanfinder_samples(path)
            all_samples.extend(samples)
            if show_progress:
                print(f"  Loaded {len(samples):,} documents from {split_name}")
    
    return all_samples


def load_and_split_mixed_windows(
    data_dir: Path,
    train_ratio: float = 0.88,
    val_ratio: float = 0.12,
    seed: int = 42,
    max_len: int = DEFAULT_MAX_LEN,
    prefix_len: int = DEFAULT_PREFIX_LEN,
    suffix_len: int = DEFAULT_SUFFIX_LEN,
    middle_len: int = DEFAULT_MIDDLE_LEN,
    multiline_max_lines: int = DEFAULT_MULTILINE_MAX_LINES,
    multiline_overlap: int = DEFAULT_MULTILINE_OVERLAP,
    multiline_min_lines: int = DEFAULT_MULTILINE_MIN_LINES,
    multiline_max_bytes: int = DEFAULT_MULTILINE_MAX_BYTES,
    show_progress: bool = True,
) -> tuple[tuple[list[LineSample], set[str]], tuple[list[LineSample], set[str]]]:
    """Load all documents, window into samples, and split with proper stratification.
    
    Returns:
        ((train_samples, train_secret_ids), (val_samples, val_secret_ids))
    """
    data_dir = Path(data_dir)
    
    if show_progress:
        print("  Loading all documents...")
    documents = load_all_documents(data_dir, show_progress=show_progress)
    
    if show_progress:
        print(f"  Windowing {len(documents):,} documents...")
    
    all_windows: list[LineSample] = []
    for sample in tqdm(documents, desc="Extracting windows", disable=not show_progress):
        single_lines = extract_lines_with_masks(sample)
        for ln in single_lines:
            windowed = window_line(ln, max_len, prefix_len, suffix_len, middle_len)
            all_windows.extend(w for w in windowed if len(w.bytes) >= MIN_WINDOW_LENGTH)
        
        multi_windows = extract_multiline_windows(
            sample,
            max_lines=multiline_max_lines,
            overlap=multiline_overlap,
            min_lines=multiline_min_lines,
            max_bytes=multiline_max_bytes,
        )
        all_windows.extend(w for w in multi_windows if len(w.bytes) >= MIN_WINDOW_LENGTH)
        
        secret_only = extract_secret_only_samples(sample, max_bytes=max_len)
        all_windows.extend(w for w in secret_only if len(w.bytes) >= MIN_WINDOW_LENGTH)
    
    if show_progress:
        print(f"  Total samples: {len(all_windows):,}")
        print(f"  Stratified splitting (seed={seed}, {train_ratio:.0%} train / {val_ratio:.0%} val)...")
    
    train, val = stratified_split_samples(
        all_windows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    
    def collect_secret_ids(samples: list[LineSample]) -> set[str]:
        ids: set[str] = set()
        for s in samples:
            if s.secret_ids:
                ids.update(s.secret_ids)
        return ids
    
    train_ids = collect_secret_ids(train)
    val_ids = collect_secret_ids(val)
    
    train_val_leak = train_ids & val_ids
    if train_val_leak:
        raise RuntimeError(f"Secret leakage detected! {len(train_val_leak)} secrets in both train and val")
    
    if show_progress:
        print(f"  Train: {len(train):,} samples, {len(train_ids):,} unique secrets")
        print(f"  Val: {len(val):,} samples, {len(val_ids):,} unique secrets")
        print("  No secret leakage detected.")
    
    return (train, train_ids), (val, val_ids)
