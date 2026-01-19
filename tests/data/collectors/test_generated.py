import pytest
import re

from heuristic_secrets.data.collectors.generated import (
    UUIDGenerator,
    HashGenerator,
    Base64Generator,
)
from heuristic_secrets.data.types import ValidatorSample


class TestUUIDGenerator:
    def test_generates_uuids(self):
        gen = UUIDGenerator(count=10)
        samples = list(gen.collect())

        assert len(samples) == 10
        for sample in samples:
            assert isinstance(sample, ValidatorSample)
            assert sample.label == 0  # False positive
            assert sample.source == "generated_uuids"
            # Validate UUID format
            assert re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                sample.text,
            )

    def test_default_count(self):
        gen = UUIDGenerator()
        assert gen.count == 5000


class TestHashGenerator:
    def test_generates_md5(self):
        gen = HashGenerator(hash_type="md5", count=5)
        samples = list(gen.collect())

        assert len(samples) == 5
        for sample in samples:
            assert sample.label == 0
            assert len(sample.text) == 32  # MD5 hex length
            assert sample.source == "generated_md5"

    def test_generates_sha1(self):
        gen = HashGenerator(hash_type="sha1", count=5)
        samples = list(gen.collect())

        for sample in samples:
            assert len(sample.text) == 40  # SHA1 hex length

    def test_generates_sha256(self):
        gen = HashGenerator(hash_type="sha256", count=5)
        samples = list(gen.collect())

        for sample in samples:
            assert len(sample.text) == 64  # SHA256 hex length


class TestBase64Generator:
    def test_generates_base64(self):
        gen = Base64Generator(count=10, min_length=20, max_length=50)
        samples = list(gen.collect())

        assert len(samples) == 10
        for sample in samples:
            assert sample.label == 0
            assert sample.source == "generated_base64"
            # Base64 characters only
            assert re.match(r"^[A-Za-z0-9+/=]+$", sample.text)
