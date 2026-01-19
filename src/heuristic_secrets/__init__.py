"""Heuristic Secrets - High-performance secret detection using ML."""

__version__ = "0.1.0"

from heuristic_secrets.pipeline.detector import Detector, Detection, DetectorConfig
from heuristic_secrets.spanfinder.model import SpanFinderConfig
from heuristic_secrets.validator.model import ValidatorConfig

__all__ = [
    "Detector",
    "Detection",
    "DetectorConfig",
    "SpanFinderConfig",
    "ValidatorConfig",
    "__version__",
]
