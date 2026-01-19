import pytest
from heuristic_secrets.data.types import ValidatorSample
from heuristic_secrets.data.collector import Collector


class TestValidatorSample:
    def test_create_sample(self):
        sample = ValidatorSample(text="secret123", label=1, source="test")
        assert sample.text == "secret123"
        assert sample.label == 1
        assert sample.source == "test"

    def test_to_dict(self):
        sample = ValidatorSample(text="secret", label=1, source="gitleaks")
        d = sample.to_dict()
        assert d == {"text": "secret", "label": 1, "source": "gitleaks", "category": "unknown"}

    def test_from_dict(self):
        d = {"text": "secret", "label": 0, "source": "uuids"}
        sample = ValidatorSample.from_dict(d)
        assert sample.text == "secret"
        assert sample.label == 0


class TestCollectorABC:
    def test_collector_is_abstract(self):
        with pytest.raises(TypeError):
            Collector()  # Can't instantiate abstract class

    def test_collector_subclass_must_implement_collect(self):
        class BadCollector(Collector):
            name = "bad"
            data_type = "validator"
            # Missing collect() method

        with pytest.raises(TypeError):
            BadCollector()
