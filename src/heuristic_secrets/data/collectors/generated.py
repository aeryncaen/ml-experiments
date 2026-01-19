"""Collectors that generate synthetic false positives."""

import base64
import hashlib
import os
import uuid
from typing import Iterator

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.types import ValidatorSample, SecretCategory


class UUIDGenerator(Collector):
    """Generate random UUIDs as false positives."""

    name = "generated_uuids"
    data_type = "validator"

    def __init__(self, count: int = 5000):
        self.count = count

    def collect(self) -> Iterator[ValidatorSample]:
        for _ in range(self.count):
            yield ValidatorSample(
                text=str(uuid.uuid4()),
                label=0,
                source=self.name,
                category=SecretCategory.FP_UUID.value,
            )


class HashGenerator(Collector):
    """Generate random hash strings as false positives."""

    data_type = "validator"

    def __init__(self, hash_type: str = "sha256", count: int = 3000):
        self.hash_type = hash_type
        self.count = count
        self.name = f"generated_{hash_type}"

    def collect(self) -> Iterator[ValidatorSample]:
        for _ in range(self.count):
            random_bytes = os.urandom(32)

            if self.hash_type == "md5":
                hash_str = hashlib.md5(random_bytes).hexdigest()
            elif self.hash_type == "sha1":
                hash_str = hashlib.sha1(random_bytes).hexdigest()
            elif self.hash_type == "sha256":
                hash_str = hashlib.sha256(random_bytes).hexdigest()
            else:
                raise ValueError(f"Unknown hash type: {self.hash_type}")

            yield ValidatorSample(
                text=hash_str,
                label=0,
                source=self.name,
                category=SecretCategory.FP_HASH.value,
            )


class Base64Generator(Collector):
    """Generate random base64 strings as false positives."""

    name = "generated_base64"
    data_type = "validator"

    def __init__(self, count: int = 2000, min_length: int = 20, max_length: int = 100):
        self.count = count
        self.min_length = min_length
        self.max_length = max_length

    def collect(self) -> Iterator[ValidatorSample]:
        import random

        for _ in range(self.count):
            # Generate random bytes, then base64 encode
            length = random.randint(self.min_length, self.max_length)
            # base64 expands by ~4/3, so generate fewer raw bytes
            raw_length = (length * 3) // 4
            random_bytes = os.urandom(raw_length)
            b64_str = base64.b64encode(random_bytes).decode("ascii")

            yield ValidatorSample(
                text=b64_str,
                label=0,
                source=self.name,
                category=SecretCategory.FP_ENCODED.value,
            )
