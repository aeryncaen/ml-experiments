"""Integration tests for the data pipeline."""

import pytest
import json
from pathlib import Path

from heuristic_secrets.data.build import DataBuilder
from heuristic_secrets.data.pipeline import read_jsonl
from heuristic_secrets.data.types import ValidatorSample


class TestFullPipeline:
    def test_build_with_all_generators(self, tmp_path):
        """Test building data with all generated sources."""
        builder = DataBuilder(
            output_dir=tmp_path,
            sources=[
                "generated_uuids",
                "generated_md5",
                "generated_sha256",
                "generated_base64",
            ],
        )

        # Use small counts
        for name in builder.registry.names():
            collector = builder.registry.get(name)
            if hasattr(collector, "count"):
                collector.count = 50

        manifest = builder.build()

        # Verify all sources collected
        assert "generated_uuids" in manifest.sources
        assert "generated_md5" in manifest.sources
        assert "generated_sha256" in manifest.sources
        assert "generated_base64" in manifest.sources

        # Verify splits created
        train_path = tmp_path / "splits" / "validator" / "train.jsonl"
        assert train_path.exists()

        # Verify data is readable
        samples = list(read_jsonl(train_path, ValidatorSample))
        assert len(samples) > 0
        assert all(s.label == 0 for s in samples)  # All false positives

    def test_samples_have_correct_format(self, tmp_path):
        """Test that output samples have the expected format."""
        builder = DataBuilder(
            output_dir=tmp_path,
            sources=["generated_uuids"],
        )
        builder.registry.get("generated_uuids").count = 10
        builder.build()

        train_path = tmp_path / "splits" / "validator" / "train.jsonl"

        with open(train_path) as f:
            for line in f:
                data = json.loads(line)
                assert "text" in data
                assert "label" in data
                assert "source" in data
                assert isinstance(data["text"], str)
                assert data["label"] in [0, 1]

    def test_deduplication_works(self, tmp_path):
        """Test that duplicate samples are removed."""
        from heuristic_secrets.data.collector import Collector
        from heuristic_secrets.data.registry import CollectorRegistry

        # Create a collector that produces duplicates
        class DuplicateCollector(Collector):
            name = "duplicates"
            data_type = "validator"

            def collect(self):
                for _ in range(10):
                    yield ValidatorSample("same_text", 1, "duplicates")
                yield ValidatorSample("different_text", 1, "duplicates")

        registry = CollectorRegistry()
        registry.register(DuplicateCollector)

        builder = DataBuilder(output_dir=tmp_path, sources=["duplicates"])
        builder.registry = registry

        manifest = builder.build()

        # Should have deduplicated to 2 unique samples
        total = (
            manifest.splits["validator"]["train"]
            + manifest.splits["validator"]["val"]
            + manifest.splits["validator"]["test"]
        )
        assert total == 2
