from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
import os

from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample


Sample = ValidatorSample | SpanFinderSample


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded sources."""
    cache_dir = os.environ.get(
        "HEURISTIC_SECRETS_CACHE",
        Path.home() / ".cache" / "heuristic-secrets"
    )
    return Path(cache_dir)


class Collector(ABC):
    """Base class for data collectors."""

    name: str  # Unique identifier for this collector
    data_type: str  # "validator" or "spanfinder"

    @abstractmethod
    def collect(self) -> Iterator[Sample]:
        """Yield samples from this source."""
        pass

    def setup(self, force: bool = False) -> None:
        """Optional: Download/clone data sources. Called before collect()."""
        pass

    def cache_path(self) -> Path:
        """Return the cache directory for this collector's data."""
        return get_cache_dir() / "sources" / self.name

    def is_available(self) -> bool:
        """Check if this collector's data is available."""
        return True  # Override if needed

    def collect_spans(self) -> Iterator[SpanFinderSample]:
        """Yield SpanFinder samples with full file context.

        Override this to provide contextual data for SpanFinder training.
        Default: yields nothing (collector doesn't support context).
        """
        return iter([])
