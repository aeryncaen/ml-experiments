from dataclasses import dataclass
from enum import Enum
from typing import Any


class SecretCategory(str, Enum):
    """Categories for secret classification.
    
    Based on SecretBench and CredData taxonomies.
    """
    # True secret categories
    PRIVATE_KEY = "private_key"           # RSA, EC, PGP private keys
    API_KEY = "api_key"                   # API keys (Stripe, Twilio, etc.)
    AUTH_TOKEN = "auth_token"             # Access tokens (AWS, Slack, GitHub, etc.)
    PASSWORD = "password"                 # Plain text passwords
    CONNECTION_STRING = "connection_string"  # Database URLs, connection strings
    GENERIC_SECRET = "generic_secret"     # Other secrets, signing keys, etc.
    
    # False positive categories (non-secrets)
    FP_UUID = "fp_uuid"                   # UUIDs, GUIDs
    FP_HASH = "fp_hash"                   # MD5, SHA1, SHA256 hashes
    FP_ENCODED = "fp_encoded"             # Base64, hex encoded data
    FP_PLACEHOLDER = "fp_placeholder"     # Example values, placeholders
    FP_OTHER = "fp_other"                 # Other false positives
    
    # Unknown/unclassified
    UNKNOWN = "unknown"


# Mapping of categories to label (1=secret, 0=non-secret)
CATEGORY_LABELS = {
    SecretCategory.PRIVATE_KEY: 1,
    SecretCategory.API_KEY: 1,
    SecretCategory.AUTH_TOKEN: 1,
    SecretCategory.PASSWORD: 1,
    SecretCategory.CONNECTION_STRING: 1,
    SecretCategory.GENERIC_SECRET: 1,
    SecretCategory.FP_UUID: 0,
    SecretCategory.FP_HASH: 0,
    SecretCategory.FP_ENCODED: 0,
    SecretCategory.FP_PLACEHOLDER: 0,
    SecretCategory.FP_OTHER: 0,
    SecretCategory.UNKNOWN: 0,  # Default to non-secret for unknown
}


def category_to_label(category: SecretCategory) -> int:
    """Convert a category to a binary label."""
    return CATEGORY_LABELS[category]


@dataclass
class ValidatorSample:
    """A sample for training the Validator model."""
    text: str
    label: int  # 1 = secret, 0 = not secret
    source: str
    category: str = "unknown"  # SecretCategory value

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "source": self.source,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidatorSample":
        return cls(
            text=d["text"],
            label=d["label"],
            source=d["source"],
            category=d.get("category", "unknown"),
        )


@dataclass
class SpanFinderSample:
    """A sample for training the SpanFinder model."""
    text: str
    starts: list[int]
    ends: list[int]
    source: str
    categories: list[str] | None = None  # Category per span

    def to_dict(self) -> dict[str, Any]:
        d = {
            "text": self.text,
            "starts": self.starts,
            "ends": self.ends,
            "source": self.source,
        }
        if self.categories is not None:
            d["categories"] = self.categories
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpanFinderSample":
        return cls(
            text=d["text"],
            starts=d["starts"],
            ends=d["ends"],
            source=d["source"],
            categories=d.get("categories"),
        )
