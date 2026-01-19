import pytest
import torch

from heuristic_secrets import (
    Detector,
    Detection,
    DetectorConfig,
    SpanFinderConfig,
    ValidatorConfig,
)
from heuristic_secrets.spanfinder.model import SpanFinderModel
from heuristic_secrets.validator.model import ValidatorModel
from heuristic_secrets.validator.features import FrequencyTable
from heuristic_secrets.io import save_model_bundle


class TestEndToEndPipeline:
    @pytest.fixture
    def model_bundle(self, tmp_path):
        sf = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        val = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))

        bundle_path = tmp_path / "integration-test-model"
        save_model_bundle(
            path=bundle_path,
            spanfinder=sf,
            validator=val,
            detector_config=DetectorConfig(
                chunk_size=256,
                chunk_overlap=32,
                spanfinder_threshold=0.3,
                validator_threshold=0.3,
            ),
            frequencies={
                "text": FrequencyTable({}),
                "innocuous": FrequencyTable({}),
                "secret": FrequencyTable({}),
            },
        )
        return bundle_path

    def test_full_pipeline_runs(self, model_bundle):
        detector = Detector.load(model_bundle)

        text = """
        const config = {
            apiKey: 'sk-1234567890abcdef',
            databaseUrl: 'postgres://user:password@localhost/db',
            debug: true
        };
        """

        detections = detector.scan(text)

        assert isinstance(detections, list)
        for d in detections:
            assert isinstance(d, Detection)
            assert 0 <= d.start < len(text)
            assert d.start < d.end <= len(text)
            assert 0.0 <= d.probability <= 1.0

    def test_batch_processing(self, model_bundle):
        detector = Detector.load(model_bundle)

        texts = [
            "api_key = 'secret123'",
            "normal text without secrets",
            "password: hunter2",
        ]

        results = detector.scan_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_empty_and_whitespace(self, model_bundle):
        detector = Detector.load(model_bundle)

        assert detector.scan("") == []
        assert detector.scan("   ") is not None
        assert detector.scan("\n\n\n") is not None

    def test_large_input(self, model_bundle):
        detector = Detector.load(model_bundle)

        large_text = "x" * 10000 + " api_key='secret' " + "y" * 10000

        detections = detector.scan(large_text)

        assert isinstance(detections, list)
        for d in detections:
            assert 0 <= d.start < len(large_text)
            assert d.start < d.end <= len(large_text)


class TestModelSaveLoadRoundtrip:
    def test_weights_preserved(self, tmp_path):
        sf = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        val = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))

        sf_weights_before = {k: v.clone() for k, v in sf.state_dict().items()}
        val_weights_before = {k: v.clone() for k, v in val.state_dict().items()}

        bundle_path = tmp_path / "roundtrip-test"
        save_model_bundle(
            path=bundle_path,
            spanfinder=sf,
            validator=val,
            detector_config=DetectorConfig(),
            frequencies={
                "text": FrequencyTable({65: 0.1, 66: 0.2}),
                "innocuous": FrequencyTable({67: 0.3}),
                "secret": FrequencyTable({68: 0.4}),
            },
        )

        detector = Detector.load(bundle_path)

        for name, weight in sf_weights_before.items():
            loaded_weight = detector.spanfinder.state_dict()[name]
            assert torch.allclose(weight, loaded_weight), f"SpanFinder {name} mismatch"

        for name, weight in val_weights_before.items():
            loaded_weight = detector.validator.state_dict()[name]
            assert torch.allclose(weight, loaded_weight), f"Validator {name} mismatch"
