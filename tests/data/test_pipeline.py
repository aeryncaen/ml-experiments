import pytest
import json
from pathlib import Path

from heuristic_secrets.data.pipeline import (
    deduplicate_samples,
    split_samples,
    write_jsonl,
    read_jsonl,
)
from heuristic_secrets.data.types import ValidatorSample


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        samples = [
            ValidatorSample("secret1", 1, "source1"),
            ValidatorSample("secret1", 1, "source2"),  # Duplicate text
            ValidatorSample("secret2", 1, "source1"),
        ]

        deduped = deduplicate_samples(samples)

        assert len(deduped) == 2
        texts = [s.text for s in deduped]
        assert "secret1" in texts
        assert "secret2" in texts

    def test_preserves_different_texts(self):
        samples = [
            ValidatorSample("a", 1, "s1"),
            ValidatorSample("b", 0, "s2"),
            ValidatorSample("c", 1, "s3"),
        ]

        deduped = deduplicate_samples(samples)
        assert len(deduped) == 3


class TestSplit:
    def test_default_split_ratios(self):
        samples = [ValidatorSample(f"s{i}", i % 2, "test") for i in range(100)]

        train, val, test = split_samples(samples, seed=42)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_custom_split_ratios(self):
        samples = [ValidatorSample(f"s{i}", i % 2, "test") for i in range(100)]

        train, val, test = split_samples(samples, train=0.7, val=0.15, test=0.15, seed=42)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_deterministic_with_seed(self):
        samples = [ValidatorSample(f"s{i}", i % 2, "test") for i in range(100)]

        train1, _, _ = split_samples(samples, seed=42)
        train2, _, _ = split_samples(samples, seed=42)

        assert [s.text for s in train1] == [s.text for s in train2]


class TestJsonl:
    def test_write_and_read(self, tmp_path):
        samples = [
            ValidatorSample("secret1", 1, "source1"),
            ValidatorSample("secret2", 0, "source2"),
        ]
        filepath = tmp_path / "test.jsonl"

        write_jsonl(samples, filepath)
        loaded = list(read_jsonl(filepath, ValidatorSample))

        assert len(loaded) == 2
        assert loaded[0].text == "secret1"
        assert loaded[1].label == 0
