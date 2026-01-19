import pytest

from heuristic_secrets.data.registry import CollectorRegistry, get_default_registry
from heuristic_secrets.data.collectors.generated import UUIDGenerator


class TestCollectorRegistry:
    def test_register_and_get(self):
        registry = CollectorRegistry()
        registry.register(UUIDGenerator)

        collector = registry.get("generated_uuids")
        assert collector is not None
        assert collector.name == "generated_uuids"

    def test_list_collectors(self):
        registry = CollectorRegistry()
        registry.register(UUIDGenerator)

        names = registry.names()
        assert "generated_uuids" in names

    def test_get_unknown_returns_none(self):
        registry = CollectorRegistry()
        assert registry.get("unknown") is None


class TestDefaultRegistry:
    def test_has_all_collectors(self):
        registry = get_default_registry()
        names = registry.names()

        # Should have all our collectors
        assert "generated_uuids" in names
        assert "generated_md5" in names
        assert "generated_sha256" in names
        assert "generated_base64" in names
        assert "detect_secrets" in names
        assert "gitleaks" in names
        assert "trufflehog" in names
