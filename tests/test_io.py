import pytest
from pathlib import Path

from heuristic_secrets.io import save_model_bundle, load_model_bundle
from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig
from heuristic_secrets.validator.features import FrequencyTable
from heuristic_secrets.pipeline.detector import DetectorConfig


class TestModelIO:
    def test_save_and_load_bundle(self, tmp_path):
        sf_config = SpanFinderConfig(embed_dim=32, num_layers=2)
        spanfinder = SpanFinderModel(sf_config)

        val_config = ValidatorConfig(hidden_dims=[8, 4])
        validator = ValidatorModel(val_config)

        det_config = DetectorConfig(chunk_size=256, chunk_overlap=32)

        freqs = {
            "text": FrequencyTable({ord("e"): 0.12}),
            "innocuous": FrequencyTable({ord("a"): 0.05}),
            "secret": FrequencyTable({ord("0"): 0.08}),
        }

        bundle_path = tmp_path / "test-model-v1"

        save_model_bundle(
            path=bundle_path,
            spanfinder=spanfinder,
            validator=validator,
            detector_config=det_config,
            frequencies=freqs,
        )

        assert (bundle_path / "config.json").exists()
        assert (bundle_path / "spanfinder.safetensors").exists()
        assert (bundle_path / "spanfinder.meta.json").exists()
        assert (bundle_path / "validator.safetensors").exists()
        assert (bundle_path / "validator.meta.json").exists()

        loaded = load_model_bundle(bundle_path)

        assert loaded.spanfinder is not None
        assert loaded.validator is not None
        assert loaded.config.chunk_size == 256
        assert loaded.frequencies["text"].get(ord("e")) == 0.12

    def test_load_preserves_architecture(self, tmp_path):
        sf_config = SpanFinderConfig(embed_dim=64, hidden_dim=64, num_layers=3)
        spanfinder = SpanFinderModel(sf_config)

        val_config = ValidatorConfig(hidden_dims=[32, 16, 8])
        validator = ValidatorModel(val_config)

        bundle_path = tmp_path / "arch-test"

        save_model_bundle(
            path=bundle_path,
            spanfinder=spanfinder,
            validator=validator,
            detector_config=DetectorConfig(),
            frequencies={
                "text": FrequencyTable({}),
                "innocuous": FrequencyTable({}),
                "secret": FrequencyTable({}),
            },
        )

        loaded = load_model_bundle(bundle_path)

        assert loaded.spanfinder.config.embed_dim == 64
        assert loaded.spanfinder.config.num_layers == 3
        assert loaded.validator.config.hidden_dims == [32, 16, 8]
