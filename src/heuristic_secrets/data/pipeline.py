"""Data pipeline utilities for deduplication and splitting."""

import json
import random
from pathlib import Path
from typing import Iterator, Type, TypeVar

from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample


T = TypeVar("T", ValidatorSample, SpanFinderSample)


def deduplicate_samples(samples: list[T]) -> list[T]:
    """Remove duplicate samples based on text content.

    Keeps the first occurrence of each unique text.

    Args:
        samples: List of samples to deduplicate

    Returns:
        Deduplicated list of samples
    """
    seen = set()
    result = []

    for sample in samples:
        if sample.text not in seen:
            seen.add(sample.text)
            result.append(sample)

    return result


def split_samples(
    samples: list[T],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int | None = None,
    stratify: bool = True,
) -> tuple[list[T], list[T], list[T]]:
    """Split samples into train/val/test sets with optional stratification by category.

    Args:
        samples: List of samples to split
        train: Fraction for training set
        val: Fraction for validation set
        test: Fraction for test set
        seed: Random seed for reproducibility
        stratify: If True, ensure proportional category representation in each split

    Returns:
        Tuple of (train, val, test) sample lists
    """
    assert abs(train + val + test - 1.0) < 0.01, "Splits must sum to 1.0"

    if seed is not None:
        random.seed(seed)

    if not stratify:
        shuffled = samples.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train)
        val_end = train_end + int(n * val)
        return (shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:])

    # Group samples by category
    by_category: dict[str, list[T]] = {}
    for sample in samples:
        cat = getattr(sample, "category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(sample)

    train_set: list[T] = []
    val_set: list[T] = []
    test_set: list[T] = []

    # Split each category proportionally
    for cat, cat_samples in by_category.items():
        random.shuffle(cat_samples)
        n = len(cat_samples)
        train_end = int(n * train)
        val_end = train_end + int(n * val)
        
        train_set.extend(cat_samples[:train_end])
        val_set.extend(cat_samples[train_end:val_end])
        test_set.extend(cat_samples[val_end:])

    # Shuffle within each split to mix categories
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return (train_set, val_set, test_set)


def write_jsonl(samples: list[T], filepath: Path) -> None:
    """Write samples to a JSONL file.

    Args:
        samples: Samples to write
        filepath: Path to output file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


def read_jsonl(filepath: Path, sample_type: Type[T]) -> Iterator[T]:
    """Read samples from a JSONL file.

    Args:
        filepath: Path to input file
        sample_type: The sample class to instantiate

    Yields:
        Sample objects
    """
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                yield sample_type.from_dict(data)
