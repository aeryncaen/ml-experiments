"""Data collectors for secret detection training."""

from heuristic_secrets.data.collectors.detect_secrets import DetectSecretsCollector
from heuristic_secrets.data.collectors.gitleaks import GitleaksCollector
from heuristic_secrets.data.collectors.trufflehog import TrufflehogCollector
from heuristic_secrets.data.collectors.creddata import CredDataCollector
from heuristic_secrets.data.collectors.generated import (
    UUIDGenerator,
    HashGenerator,
    Base64Generator,
)

__all__ = [
    "DetectSecretsCollector",
    "GitleaksCollector",
    "TrufflehogCollector",
    "CredDataCollector",
    "UUIDGenerator",
    "HashGenerator",
    "Base64Generator",
]
