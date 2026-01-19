import pytest


class TestPublicAPI:
    def test_can_import_detector(self):
        from heuristic_secrets import Detector
        assert Detector is not None

    def test_can_import_detection(self):
        from heuristic_secrets import Detection
        assert Detection is not None

    def test_can_import_config(self):
        from heuristic_secrets import DetectorConfig
        assert DetectorConfig is not None

    def test_can_import_spanfinder_config(self):
        from heuristic_secrets import SpanFinderConfig
        assert SpanFinderConfig is not None

    def test_can_import_validator_config(self):
        from heuristic_secrets import ValidatorConfig
        assert ValidatorConfig is not None

    def test_version_exists(self):
        import heuristic_secrets
        assert hasattr(heuristic_secrets, "__version__")
