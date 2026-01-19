"""Registry for data collectors."""

from typing import Type

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.collectors.generated import (
    UUIDGenerator,
    HashGenerator,
    Base64Generator,
)
from heuristic_secrets.data.collectors.detect_secrets import DetectSecretsCollector
from heuristic_secrets.data.collectors.gitleaks import GitleaksCollector
from heuristic_secrets.data.collectors.trufflehog import TrufflehogCollector
from heuristic_secrets.data.collectors.creddata import CredDataCollector


class CollectorRegistry:
    """Registry for data collectors."""

    def __init__(self):
        self._collectors: dict[str, Collector] = {}

    def register(self, collector_class: Type[Collector], **kwargs) -> None:
        """Register a collector class.

        Args:
            collector_class: The collector class to register
            **kwargs: Arguments to pass to the constructor
        """
        instance = collector_class(**kwargs)
        self._collectors[instance.name] = instance

    def get(self, name: str) -> Collector | None:
        """Get a collector by name."""
        return self._collectors.get(name)

    def names(self) -> list[str]:
        """List all registered collector names."""
        return list(self._collectors.keys())

    def all(self) -> list[Collector]:
        """Get all registered collectors."""
        return list(self._collectors.values())


def get_default_registry() -> CollectorRegistry:
    """Get the default registry with all built-in collectors."""
    registry = CollectorRegistry()

    # Generated false positives
    registry.register(UUIDGenerator)
    registry.register(HashGenerator, hash_type="md5")
    registry.register(HashGenerator, hash_type="sha1")
    registry.register(HashGenerator, hash_type="sha256")
    registry.register(Base64Generator)

    # Tool fixtures (secrets)
    registry.register(DetectSecretsCollector)
    registry.register(GitleaksCollector)
    registry.register(TrufflehogCollector)

    # Research datasets
    registry.register(CredDataCollector)

    return registry
