import pytest
import torch
from heuristic_secrets.pipeline.detector import Detector, Detection, DetectorConfig
from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig
from heuristic_secrets.validator.features import FrequencyTable


class TestDetector:
    @pytest.fixture
    def detector(self):
        spanfinder = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        validator = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))

        config = DetectorConfig(
            chunk_size=100,
            chunk_overlap=10,
            spanfinder_threshold=0.5,
            validator_threshold=0.5,
        )

        return Detector(
            spanfinder=spanfinder,
            validator=validator,
            config=config,
            text_freq=FrequencyTable({}),
            innocuous_freq=FrequencyTable({}),
            secret_freq=FrequencyTable({}),
        )

    def test_scan_returns_detections(self, detector):
        text = "config = { api_key: 'sk-abc123xyz789' }"

        detections = detector.scan(text)

        assert isinstance(detections, list)
        for d in detections:
            assert isinstance(d, Detection)
            assert hasattr(d, 'start')
            assert hasattr(d, 'end')
            assert hasattr(d, 'text')
            assert hasattr(d, 'probability')

    def test_scan_empty_string(self, detector):
        detections = detector.scan("")
        assert detections == []

    def test_scan_batch(self, detector):
        texts = [
            "api_key = 'sk-abc123'",
            "password = 'hunter2'",
            "normal text here",
        ]

        results = detector.scan_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_detection_positions_are_valid(self, detector):
        text = "x" * 200

        detections = detector.scan(text)

        for d in detections:
            assert 0 <= d.start < len(text)
            assert d.start < d.end <= len(text)
            assert d.text == text[d.start:d.end]

    def test_probability_in_range(self, detector):
        text = "secret_key = 'ghp_xxxxxxxxxxxxxxxxxxxx'"

        detections = detector.scan(text)

        for d in detections:
            assert 0.0 <= d.probability <= 1.0


class TestDetectorLoad:
    def test_load_from_bundle(self, tmp_path):
        from heuristic_secrets.io import save_model_bundle

        sf = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        val = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))

        bundle_path = tmp_path / "test-bundle"
        save_model_bundle(
            path=bundle_path,
            spanfinder=sf,
            validator=val,
            detector_config=DetectorConfig(),
            frequencies={
                "text": FrequencyTable({}),
                "innocuous": FrequencyTable({}),
                "secret": FrequencyTable({}),
            },
        )

        detector = Detector.load(bundle_path)

        assert detector is not None
        assert detector.scan("test") is not None

    def test_load_with_config_override(self, tmp_path):
        from heuristic_secrets.io import save_model_bundle

        sf = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        val = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))

        bundle_path = tmp_path / "test-bundle-2"
        save_model_bundle(
            path=bundle_path,
            spanfinder=sf,
            validator=val,
            detector_config=DetectorConfig(validator_threshold=0.5),
            frequencies={
                "text": FrequencyTable({}),
                "innocuous": FrequencyTable({}),
                "secret": FrequencyTable({}),
            },
        )

        custom_config = DetectorConfig(validator_threshold=0.9)
        detector = Detector.load(bundle_path, config=custom_config)

        assert detector.config.validator_threshold == 0.9
