import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from heuristic_secrets.data.build import DataBuilder, Manifest
from heuristic_secrets.data.types import ValidatorSample


class TestDataBuilder:
    def test_build_with_generated_only(self, tmp_path):
        builder = DataBuilder(
            output_dir=tmp_path,
            sources=["generated_uuids"],
        )

        # Use small counts for testing
        builder.registry.get("generated_uuids").count = 100

        manifest = builder.build()

        # Check files exist
        assert (tmp_path / "splits" / "validator" / "train.jsonl").exists()
        assert (tmp_path / "splits" / "validator" / "val.jsonl").exists()
        assert (tmp_path / "splits" / "validator" / "test.jsonl").exists()
        assert (tmp_path / "manifest.json").exists()

        # Check manifest
        assert manifest.sources["generated_uuids"]["samples"] == 100
        assert manifest.splits["validator"]["train"] == 80
        assert manifest.splits["validator"]["val"] == 10
        assert manifest.splits["validator"]["test"] == 10

    def test_build_creates_manifest(self, tmp_path):
        builder = DataBuilder(
            output_dir=tmp_path,
            sources=["generated_uuids"],
        )
        builder.registry.get("generated_uuids").count = 50

        builder.build()

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            data = json.load(f)

        assert "created_at" in data
        assert "sources" in data
        assert "splits" in data
        assert "deduplication" in data
