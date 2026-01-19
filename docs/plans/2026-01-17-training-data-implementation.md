# Training Data Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pluggable data collection pipeline that gathers training data for the Validator model from readily-available public sources.

**Architecture:** Plugin-based collector system with auto-fetch capability. Each collector clones/downloads its source, parses test fixtures, and yields normalized samples. A build script orchestrates collection, deduplication, and train/val/test splitting.

**Tech Stack:** Python 3.11+, pytest, gitpython (for cloning), PyYAML

---

## Task 1: Identify Readily-Available Data Sources

**Files:**
- Create: `docs/data-sources.md`

**Step 1: Document available sources**

Based on research, these sources are **readily available** (no credentials needed):

| Source | Location | Secret Format | Estimated Count |
|--------|----------|---------------|-----------------|
| **Gitleaks** | `cmd/generate/config/rules/*.go` | Go string literals with `GenerateSampleSecrets()` | ~500+ patterns |
| **detect-secrets** | `tests/plugins/*_test.py` + `test_data/` | Python constants + fixture files | ~200+ secrets |
| **TruffleHog** | `pkg/detectors/*/*_test.go` | Go string literals (unit tests only) | ~300+ patterns |
| **Generated UUIDs** | Programmatic | `uuid.uuid4()` | Unlimited |
| **Generated Hashes** | Programmatic | MD5, SHA1, SHA256 | Unlimited |
| **Generated Base64** | Programmatic | Random base64 strings | Unlimited |

**Not readily available** (skip for now):
- TruffleHog integration tests (require GCP Secret Manager access)
- FPSecretBench (requires DPA)
- SecretBench (requires DPA)

```markdown
# Data Sources for Training

## Readily Available (No Credentials)

### 1. Gitleaks (Secrets - Positive)
- **Repo:** https://github.com/gitleaks/gitleaks
- **Location:** `cmd/generate/config/rules/*.go`
- **Format:** Go functions calling `utils.GenerateSampleSecrets("identifier", "secret")`
- **Example:**
  ```go
  tps := utils.GenerateSampleSecrets("AWS", "AKIALALEMEL33243OLIB")
  ```

### 2. detect-secrets (Secrets - Positive)
- **Repo:** https://github.com/Yelp/detect-secrets
- **Locations:**
  - `tests/plugins/*_test.py` - Module-level constants like `EXAMPLE_SECRET = '...'`
  - `test_data/each_secret.py` - One example of each secret type
  - `test_data/config.yaml`, `config.ini`, etc. - Various formats
- **Example:**
  ```python
  EXAMPLE_SECRET = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
  ```

### 3. TruffleHog (Secrets - Positive, Unit Tests Only)
- **Repo:** https://github.com/trufflesecurity/trufflehog
- **Location:** `pkg/detectors/*/*_test.go` (unit tests, NOT integration tests)
- **Format:** Go string literals in test cases
- **Example:**
  ```go
  validPattern = `[{"test_secrets": {"github_secret": "ghs_RWGUZ6..."}}]`
  ```
- **Note:** Integration tests use GCP Secret Manager - skip those.

### 4. Generated False Positives
- UUIDs: `550e8400-e29b-41d4-a716-446655440000`
- MD5: `d41d8cd98f00b204e9800998ecf8427e`
- SHA1: `da39a3ee5e6b4b0d3255bfef95601890afd80709`
- SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Base64: Random bytes encoded

## Requires Special Access (Later)
- FPSecretBench: 1.5M false positives (requires DPA)
- SecretBench: 15K secrets with context (requires DPA)
```

**Step 2: Commit**

```bash
git add docs/data-sources.md
git commit -m "docs: document readily-available training data sources"
```

---

## Task 2: Base Infrastructure - Sample Types and Collector ABC

**Files:**
- Create: `src/heuristic_secrets/data/__init__.py`
- Create: `src/heuristic_secrets/data/types.py`
- Create: `src/heuristic_secrets/data/collector.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_collector.py`

**Step 1: Write the failing test**

```python
# tests/data/test_collector.py
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
        assert d == {"text": "secret", "label": 1, "source": "gitleaks"}

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_collector.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create directories and implement**

```bash
mkdir -p src/heuristic_secrets/data tests/data
touch src/heuristic_secrets/data/__init__.py
touch tests/data/__init__.py
```

```python
# src/heuristic_secrets/data/types.py
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidatorSample:
    """A sample for training the Validator model."""
    text: str
    label: int  # 1 = secret, 0 = not secret
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "label": self.label, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidatorSample":
        return cls(text=d["text"], label=d["label"], source=d["source"])


@dataclass
class SpanFinderSample:
    """A sample for training the SpanFinder model."""
    text: str
    starts: list[int]
    ends: list[int]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "starts": self.starts,
            "ends": self.ends,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpanFinderSample":
        return cls(
            text=d["text"],
            starts=d["starts"],
            ends=d["ends"],
            source=d["source"],
        )
```

```python
# src/heuristic_secrets/data/collector.py
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

    def setup(self) -> None:
        """Optional: Download/clone data sources. Called before collect()."""
        pass

    def cache_path(self) -> Path:
        """Return the cache directory for this collector's data."""
        return get_cache_dir() / "sources" / self.name

    def is_available(self) -> bool:
        """Check if this collector's data is available."""
        return True  # Override if needed
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_collector.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/ tests/data/
git commit -m "feat(data): add sample types and collector base class"
```

---

## Task 3: Git Clone Utility

**Files:**
- Create: `src/heuristic_secrets/data/git_utils.py`
- Create: `tests/data/test_git_utils.py`

**Step 1: Write the failing test**

```python
# tests/data/test_git_utils.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from heuristic_secrets.data.git_utils import clone_or_pull, get_repo_version


class TestCloneOrPull:
    def test_clone_new_repo(self, tmp_path):
        # Use a small, fast repo for testing
        repo_url = "https://github.com/octocat/Hello-World.git"
        dest = tmp_path / "hello-world"

        clone_or_pull(repo_url, dest)

        assert dest.exists()
        assert (dest / ".git").exists()

    def test_pull_existing_repo(self, tmp_path):
        repo_url = "https://github.com/octocat/Hello-World.git"
        dest = tmp_path / "hello-world"

        # Clone first
        clone_or_pull(repo_url, dest)
        # Pull again (should not fail)
        clone_or_pull(repo_url, dest)

        assert dest.exists()


class TestGetRepoVersion:
    def test_get_version_from_repo(self, tmp_path):
        repo_url = "https://github.com/octocat/Hello-World.git"
        dest = tmp_path / "hello-world"

        clone_or_pull(repo_url, dest)
        version = get_repo_version(dest)

        # Should return a commit hash (40 hex chars)
        assert len(version) == 40
        assert all(c in "0123456789abcdef" for c in version)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_git_utils.py -v`
Expected: FAIL with "cannot import name 'clone_or_pull'"

**Step 3: Implement**

```python
# src/heuristic_secrets/data/git_utils.py
import subprocess
from pathlib import Path


def clone_or_pull(repo_url: str, dest: Path) -> None:
    """Clone a git repository, or pull if it already exists.

    Args:
        repo_url: URL of the git repository
        dest: Destination directory
    """
    dest = Path(dest)

    if (dest / ".git").exists():
        # Pull existing repo
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=dest,
            capture_output=True,
            check=True,
        )
    else:
        # Clone new repo
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest)],
            capture_output=True,
            check=True,
        )


def get_repo_version(repo_path: Path) -> str:
    """Get the current commit hash of a git repository.

    Args:
        repo_path: Path to the repository

    Returns:
        The current commit hash (40 hex characters)
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_git_utils.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/git_utils.py tests/data/test_git_utils.py
git commit -m "feat(data): add git clone/pull utilities"
```

---

## Task 4: UUID Generator Collector

**Files:**
- Create: `src/heuristic_secrets/data/collectors/generated.py`
- Create: `tests/data/collectors/__init__.py`
- Create: `tests/data/collectors/test_generated.py`

**Step 1: Write the failing test**

```python
# tests/data/collectors/test_generated.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/collectors/test_generated.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create directories and implement**

```bash
mkdir -p src/heuristic_secrets/data/collectors tests/data/collectors
touch src/heuristic_secrets/data/collectors/__init__.py
touch tests/data/collectors/__init__.py
```

```python
# src/heuristic_secrets/data/collectors/generated.py
"""Collectors that generate synthetic false positives."""

import base64
import hashlib
import os
import uuid
from typing import Iterator

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.types import ValidatorSample


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
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/collectors/test_generated.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/collectors/ tests/data/collectors/
git commit -m "feat(data): add UUID, hash, and base64 generators"
```

---

## Task 5: detect-secrets Collector

**Files:**
- Create: `src/heuristic_secrets/data/collectors/detect_secrets.py`
- Create: `tests/data/collectors/test_detect_secrets.py`

**Step 1: Write the failing test**

```python
# tests/data/collectors/test_detect_secrets.py
import pytest
from pathlib import Path
from unittest.mock import patch

from heuristic_secrets.data.collectors.detect_secrets import DetectSecretsCollector
from heuristic_secrets.data.types import ValidatorSample


class TestDetectSecretsCollector:
    def test_name_and_type(self):
        collector = DetectSecretsCollector()
        assert collector.name == "detect_secrets"
        assert collector.data_type == "validator"

    def test_parse_python_constant(self):
        collector = DetectSecretsCollector()

        code = '''
EXAMPLE_SECRET = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
OTHER_VALUE = 123
CLOUD_IAM_KEY = "abcd1234abcd1234abcd1234ABCD1234ABCD1234--__"
'''
        secrets = list(collector._extract_constants_from_python(code))

        assert "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" in secrets
        assert "abcd1234abcd1234abcd1234ABCD1234ABCD1234--__" in secrets
        assert 123 not in secrets  # Not a string

    def test_parse_each_secret_file(self):
        collector = DetectSecretsCollector()

        # Simulated content from test_data/each_secret.py
        code = '''
base64_secret = 'c2VjcmV0IG1lc3NhZ2Ugc28geW91J2xsIG5ldmVyIGd1ZXNzIG15IHBhc3N3b3Jk'
hex_secret = '8b1118b376c313ed420e5133ba91307817ed52c2'
aws_access_key = 'AKIAIOSFODNN7EXAMPLE'
aws_secret_access_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
'''
        secrets = list(collector._extract_constants_from_python(code))

        assert len(secrets) == 4
        assert 'AKIAIOSFODNN7EXAMPLE' in secrets

    def test_filters_short_strings(self):
        collector = DetectSecretsCollector()

        code = '''
SHORT = 'abc'
EMPTY = ''
LONG_ENOUGH = 'this_is_long_enough_to_be_a_secret'
'''
        secrets = list(collector._extract_constants_from_python(code))

        assert 'abc' not in secrets
        assert '' not in secrets
        assert 'this_is_long_enough_to_be_a_secret' in secrets
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/collectors/test_detect_secrets.py -v`
Expected: FAIL with "cannot import name 'DetectSecretsCollector'"

**Step 3: Implement**

```python
# src/heuristic_secrets/data/collectors/detect_secrets.py
"""Collector for detect-secrets test fixtures."""

import ast
import re
from pathlib import Path
from typing import Iterator

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.git_utils import clone_or_pull, get_repo_version
from heuristic_secrets.data.types import ValidatorSample


class DetectSecretsCollector(Collector):
    """Collect secrets from detect-secrets test fixtures."""

    name = "detect_secrets"
    data_type = "validator"
    repo_url = "https://github.com/Yelp/detect-secrets.git"

    MIN_SECRET_LENGTH = 10  # Filter out short strings

    def setup(self) -> None:
        """Clone or update the detect-secrets repository."""
        clone_or_pull(self.repo_url, self.cache_path())

    def is_available(self) -> bool:
        """Check if the repository has been cloned."""
        return (self.cache_path() / ".git").exists()

    def collect(self) -> Iterator[ValidatorSample]:
        """Yield secrets from detect-secrets test files."""
        repo_path = self.cache_path()

        if not self.is_available():
            return

        seen = set()  # Deduplicate within this collector

        # 1. Parse test plugin files for constants
        tests_dir = repo_path / "tests" / "plugins"
        if tests_dir.exists():
            for test_file in tests_dir.glob("*_test.py"):
                content = test_file.read_text(errors="ignore")
                for secret in self._extract_constants_from_python(content):
                    if secret not in seen:
                        seen.add(secret)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                        )

        # 2. Parse test_data/each_secret.py
        each_secret = repo_path / "test_data" / "each_secret.py"
        if each_secret.exists():
            content = each_secret.read_text(errors="ignore")
            for secret in self._extract_constants_from_python(content):
                if secret not in seen:
                    seen.add(secret)
                    yield ValidatorSample(
                        text=secret,
                        label=1,
                        source=self.name,
                    )

        # 3. Parse test_data/config.yaml for secrets
        config_yaml = repo_path / "test_data" / "config.yaml"
        if config_yaml.exists():
            content = config_yaml.read_text(errors="ignore")
            for secret in self._extract_secrets_from_yaml(content):
                if secret not in seen:
                    seen.add(secret)
                    yield ValidatorSample(
                        text=secret,
                        label=1,
                        source=self.name,
                    )

    def _extract_constants_from_python(self, code: str) -> Iterator[str]:
        """Extract string constants from Python code.

        Looks for module-level assignments like:
            EXAMPLE_SECRET = 'secret_value'
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            # Look for assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if value is a string constant
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, str
                        ):
                            value = node.value.value
                            if len(value) >= self.MIN_SECRET_LENGTH:
                                yield value

    def _extract_secrets_from_yaml(self, content: str) -> Iterator[str]:
        """Extract potential secrets from YAML content.

        Looks for patterns like:
            key: 'AKIAIOSFODNN7EXAMPLE'
            api_key: "secret_value"
        """
        # Match quoted values that look like secrets
        # High-entropy looking strings, API keys, etc.
        patterns = [
            r":\s*['\"]([A-Za-z0-9+/=_-]{20,})['\"]",  # Long alphanumeric
            r":\s*['\"]?(AKIA[A-Z0-9]{16})['\"]?",  # AWS access key
            r":\s*['\"]?([a-f0-9]{32,})['\"]?",  # Hex strings
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content):
                value = match.group(1)
                if len(value) >= self.MIN_SECRET_LENGTH:
                    yield value
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/collectors/test_detect_secrets.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/collectors/detect_secrets.py tests/data/collectors/test_detect_secrets.py
git commit -m "feat(data): add detect-secrets collector"
```

---

## Task 6: Gitleaks Collector

**Files:**
- Create: `src/heuristic_secrets/data/collectors/gitleaks.py`
- Create: `tests/data/collectors/test_gitleaks.py`

**Step 1: Write the failing test**

```python
# tests/data/collectors/test_gitleaks.py
import pytest

from heuristic_secrets.data.collectors.gitleaks import GitleaksCollector
from heuristic_secrets.data.types import ValidatorSample


class TestGitleaksCollector:
    def test_name_and_type(self):
        collector = GitleaksCollector()
        assert collector.name == "gitleaks"
        assert collector.data_type == "validator"

    def test_extract_secrets_from_go(self):
        collector = GitleaksCollector()

        code = '''
func GitHubPat() *config.Rule {
    tps := utils.GenerateSampleSecrets("github", "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    tps = append(tps, utils.GenerateSampleSecrets("github", "ghp_1234567890abcdefABCDEF1234567890ab")...)
    fps := []string{"ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
    return utils.Validate(r, tps, fps)
}
'''
        secrets = list(collector._extract_secrets_from_go(code))

        assert "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in secrets
        assert "ghp_1234567890abcdefABCDEF1234567890ab" in secrets

    def test_extract_from_testdata(self):
        collector = GitleaksCollector()

        code = '''
awsToken := "AKIALALEMEL33243OLIA"
password := "super_secret_password_123"
'''
        secrets = list(collector._extract_string_literals_from_go(code))

        assert "AKIALALEMEL33243OLIA" in secrets
        assert "super_secret_password_123" in secrets

    def test_filters_short_and_common(self):
        collector = GitleaksCollector()

        code = '''
short := "abc"
empty := ""
valid := "this_is_a_valid_secret_string"
common := "true"
'''
        secrets = list(collector._extract_string_literals_from_go(code))

        assert "abc" not in secrets
        assert "" not in secrets
        assert "true" not in secrets
        assert "this_is_a_valid_secret_string" in secrets
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/collectors/test_gitleaks.py -v`
Expected: FAIL with "cannot import name 'GitleaksCollector'"

**Step 3: Implement**

```python
# src/heuristic_secrets/data/collectors/gitleaks.py
"""Collector for Gitleaks test fixtures."""

import re
from pathlib import Path
from typing import Iterator

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.git_utils import clone_or_pull
from heuristic_secrets.data.types import ValidatorSample


class GitleaksCollector(Collector):
    """Collect secrets from Gitleaks test fixtures."""

    name = "gitleaks"
    data_type = "validator"
    repo_url = "https://github.com/gitleaks/gitleaks.git"

    MIN_SECRET_LENGTH = 10

    # Common strings to filter out
    BLOCKLIST = {
        "true", "false", "null", "none", "undefined",
        "test", "example", "sample", "demo", "placeholder",
        "password", "secret", "token", "key", "api",
    }

    def setup(self) -> None:
        """Clone or update the Gitleaks repository."""
        clone_or_pull(self.repo_url, self.cache_path())

    def is_available(self) -> bool:
        """Check if the repository has been cloned."""
        return (self.cache_path() / ".git").exists()

    def collect(self) -> Iterator[ValidatorSample]:
        """Yield secrets from Gitleaks test files."""
        repo_path = self.cache_path()

        if not self.is_available():
            return

        seen = set()

        # 1. Parse rule files in cmd/generate/config/rules/
        rules_dir = repo_path / "cmd" / "generate" / "config" / "rules"
        if rules_dir.exists():
            for go_file in rules_dir.glob("*.go"):
                content = go_file.read_text(errors="ignore")
                for secret in self._extract_secrets_from_go(content):
                    if secret not in seen:
                        seen.add(secret)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                        )

        # 2. Parse testdata files
        testdata_dir = repo_path / "testdata"
        if testdata_dir.exists():
            for go_file in testdata_dir.rglob("*.go"):
                content = go_file.read_text(errors="ignore")
                for secret in self._extract_string_literals_from_go(content):
                    if secret not in seen:
                        seen.add(secret)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                        )

            # Also check .env files
            for env_file in testdata_dir.rglob("*.env*"):
                content = env_file.read_text(errors="ignore")
                for secret in self._extract_from_env(content):
                    if secret not in seen:
                        seen.add(secret)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                        )

    def _extract_secrets_from_go(self, code: str) -> Iterator[str]:
        """Extract secrets from GenerateSampleSecrets calls.

        Pattern: utils.GenerateSampleSecrets("identifier", "secret")
        """
        # Match GenerateSampleSecrets calls
        pattern = r'GenerateSampleSecrets\s*\(\s*"[^"]*"\s*,\s*"([^"]+)"\s*\)'
        for match in re.finditer(pattern, code):
            secret = match.group(1)
            if self._is_valid_secret(secret):
                yield secret

        # Also match secrets.NewSecret patterns for regex-generated ones
        # These often follow with actual test values
        literal_pattern = r'(?:tps|fps)\s*(?::=|=)\s*\[\]string\{([^}]+)\}'
        for match in re.finditer(literal_pattern, code):
            block = match.group(1)
            for str_match in re.finditer(r'"([^"]+)"', block):
                secret = str_match.group(1)
                if self._is_valid_secret(secret):
                    yield secret

    def _extract_string_literals_from_go(self, code: str) -> Iterator[str]:
        """Extract string literals that look like secrets."""
        # Match Go string assignments
        pattern = r':=\s*"([^"]+)"'
        for match in re.finditer(pattern, code):
            value = match.group(1)
            if self._is_valid_secret(value):
                yield value

        # Match backtick strings too
        pattern = r':=\s*`([^`]+)`'
        for match in re.finditer(pattern, code):
            value = match.group(1)
            if self._is_valid_secret(value):
                yield value

    def _extract_from_env(self, content: str) -> Iterator[str]:
        """Extract values from .env files."""
        for line in content.splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                _, _, value = line.partition("=")
                value = value.strip().strip("'\"")
                if self._is_valid_secret(value):
                    yield value

    def _is_valid_secret(self, value: str) -> bool:
        """Check if a string looks like a valid secret."""
        if len(value) < self.MIN_SECRET_LENGTH:
            return False
        if value.lower() in self.BLOCKLIST:
            return False
        # Filter out obvious placeholders
        if value.startswith("${") or value.startswith("{{"):
            return False
        return True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/collectors/test_gitleaks.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/collectors/gitleaks.py tests/data/collectors/test_gitleaks.py
git commit -m "feat(data): add Gitleaks collector"
```

---

## Task 7: TruffleHog Collector (Unit Tests Only)

**Files:**
- Create: `src/heuristic_secrets/data/collectors/trufflehog.py`
- Create: `tests/data/collectors/test_trufflehog.py`

**Step 1: Write the failing test**

```python
# tests/data/collectors/test_trufflehog.py
import pytest

from heuristic_secrets.data.collectors.trufflehog import TrufflehogCollector
from heuristic_secrets.data.types import ValidatorSample


class TestTrufflehogCollector:
    def test_name_and_type(self):
        collector = TrufflehogCollector()
        assert collector.name == "trufflehog"
        assert collector.data_type == "validator"

    def test_extract_from_unit_test(self):
        collector = TrufflehogCollector()

        code = '''
var (
    validPattern = `[{
        "test_secrets": {
            "github_secret": "ghs_RWGUZ6kS8_Ut7PbtR72k2miJwwYtxkpe8mOp"
        }
    }]`
    secret = "ghs_RWGUZ6kS8_Ut7PbtR72k2miJwwYtxkpe8mOp"
)

func TestGitHub_Pattern(t *testing.T) {
    tests := []struct {
        input string
        want  []string
    }{
        {
            input: `token := "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"`,
            want: []string{"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"},
        },
    }
}
'''
        secrets = list(collector._extract_from_go_test(code))

        assert "ghs_RWGUZ6kS8_Ut7PbtR72k2miJwwYtxkpe8mOp" in secrets
        assert "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" in secrets

    def test_skips_integration_tests(self):
        collector = TrufflehogCollector()

        # Integration test pattern - uses GCP Secret Manager
        code = '''
func TestGitlab_FromChunk(t *testing.T) {
    testSecrets, err := common.GetSecret(ctx, "trufflehog-testing", "detectors4")
    secret := testSecrets.MustGetField("GITLAB")
}
'''
        secrets = list(collector._extract_from_go_test(code))

        # Should not extract - these are references to GCP secrets
        assert "GITLAB" not in secrets
        assert "detectors4" not in secrets
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/collectors/test_trufflehog.py -v`
Expected: FAIL with "cannot import name 'TrufflehogCollector'"

**Step 3: Implement**

```python
# src/heuristic_secrets/data/collectors/trufflehog.py
"""Collector for TruffleHog test fixtures (unit tests only)."""

import re
from pathlib import Path
from typing import Iterator

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.git_utils import clone_or_pull
from heuristic_secrets.data.types import ValidatorSample


class TrufflehogCollector(Collector):
    """Collect secrets from TruffleHog unit test fixtures.

    Note: Integration tests require GCP Secret Manager access, so we skip those.
    We only parse unit tests (*_test.go files that don't use common.GetSecret).
    """

    name = "trufflehog"
    data_type = "validator"
    repo_url = "https://github.com/trufflesecurity/trufflehog.git"

    MIN_SECRET_LENGTH = 15  # TruffleHog secrets tend to be longer

    def setup(self) -> None:
        """Clone or update the TruffleHog repository."""
        clone_or_pull(self.repo_url, self.cache_path())

    def is_available(self) -> bool:
        """Check if the repository has been cloned."""
        return (self.cache_path() / ".git").exists()

    def collect(self) -> Iterator[ValidatorSample]:
        """Yield secrets from TruffleHog unit test files."""
        repo_path = self.cache_path()

        if not self.is_available():
            return

        seen = set()

        # Parse detector test files
        detectors_dir = repo_path / "pkg" / "detectors"
        if detectors_dir.exists():
            for test_file in detectors_dir.rglob("*_test.go"):
                # Skip integration tests
                if "integration" in test_file.name.lower():
                    continue

                content = test_file.read_text(errors="ignore")

                # Skip files that use GCP Secret Manager
                if "common.GetSecret" in content or "MustGetField" in content:
                    continue

                for secret in self._extract_from_go_test(content):
                    if secret not in seen:
                        seen.add(secret)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                        )

    def _extract_from_go_test(self, code: str) -> Iterator[str]:
        """Extract secret strings from Go unit test code.

        Looks for:
        - String literals in test cases
        - Variables assigned string values
        - want: []string{...} patterns
        """
        # Skip integration test patterns
        if "common.GetSecret" in code or "MustGetField" in code:
            return

        # Pattern 1: Direct string assignments
        # secret = "ghs_xxxx" or secret := "ghs_xxxx"
        pattern = r'(?:secret|token|key|password)\s*(?::=|=)\s*"([^"]+)"'
        for match in re.finditer(pattern, code, re.IGNORECASE):
            value = match.group(1)
            if self._is_valid_secret(value):
                yield value

        # Pattern 2: want: []string{"xxx", "yyy"}
        pattern = r'want:\s*\[\]string\{([^}]+)\}'
        for match in re.finditer(pattern, code):
            block = match.group(1)
            for str_match in re.finditer(r'"([^"]+)"', block):
                value = str_match.group(1)
                if self._is_valid_secret(value):
                    yield value

        # Pattern 3: input: `token := "xxx"` in test structs
        pattern = r'input:\s*`[^`]*"([^"]{15,})"[^`]*`'
        for match in re.finditer(pattern, code):
            value = match.group(1)
            if self._is_valid_secret(value):
                yield value

        # Pattern 4: validPattern = `...` backtick strings with secrets
        pattern = r'validPattern\s*=\s*`([^`]+)`'
        for match in re.finditer(pattern, code):
            block = match.group(1)
            # Extract quoted strings from within
            for str_match in re.finditer(r'"([^"]+)"', block):
                value = str_match.group(1)
                if self._is_valid_secret(value):
                    yield value

    def _is_valid_secret(self, value: str) -> bool:
        """Check if a string looks like a valid secret."""
        if len(value) < self.MIN_SECRET_LENGTH:
            return False

        # Filter out obvious non-secrets
        blocklist = {
            "mock_filename", "test", "example", "placeholder",
            "xxx", "yyy", "zzz", "abc", "123",
        }
        if value.lower() in blocklist:
            return False

        # Filter out repeated characters (xxxxx)
        if len(set(value.lower())) < 5:
            return False

        # Filter out URL-like strings without credentials
        if value.startswith("http") and "@" not in value:
            return False

        return True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/collectors/test_trufflehog.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/collectors/trufflehog.py tests/data/collectors/test_trufflehog.py
git commit -m "feat(data): add TruffleHog collector (unit tests only)"
```

---

## Task 8: Collector Registry

**Files:**
- Create: `src/heuristic_secrets/data/registry.py`
- Create: `tests/data/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/data/test_registry.py
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

        names = registry.list()
        assert "generated_uuids" in names

    def test_get_unknown_returns_none(self):
        registry = CollectorRegistry()
        assert registry.get("unknown") is None


class TestDefaultRegistry:
    def test_has_all_collectors(self):
        registry = get_default_registry()
        names = registry.list()

        # Should have all our collectors
        assert "generated_uuids" in names
        assert "generated_md5" in names
        assert "generated_sha256" in names
        assert "generated_base64" in names
        assert "detect_secrets" in names
        assert "gitleaks" in names
        assert "trufflehog" in names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_registry.py -v`
Expected: FAIL with "cannot import name 'CollectorRegistry'"

**Step 3: Implement**

```python
# src/heuristic_secrets/data/registry.py
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

    def list(self) -> list[str]:
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

    return registry
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_registry.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/registry.py tests/data/test_registry.py
git commit -m "feat(data): add collector registry"
```

---

## Task 9: Build Pipeline - Deduplication and Splitting

**Files:**
- Create: `src/heuristic_secrets/data/pipeline.py`
- Create: `tests/data/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/data/test_pipeline.py
import pytest
import json
from pathlib import Path

from heuristic_secrets.data.pipeline import (
    deduplicate_samples,
    split_samples,
    write_jsonl,
    read_jsonl,
)
from heuristic_secrets.data.types import ValidatorSample


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        samples = [
            ValidatorSample("secret1", 1, "source1"),
            ValidatorSample("secret1", 1, "source2"),  # Duplicate text
            ValidatorSample("secret2", 1, "source1"),
        ]

        deduped = deduplicate_samples(samples)

        assert len(deduped) == 2
        texts = [s.text for s in deduped]
        assert "secret1" in texts
        assert "secret2" in texts

    def test_preserves_different_texts(self):
        samples = [
            ValidatorSample("a", 1, "s1"),
            ValidatorSample("b", 0, "s2"),
            ValidatorSample("c", 1, "s3"),
        ]

        deduped = deduplicate_samples(samples)
        assert len(deduped) == 3


class TestSplit:
    def test_default_split_ratios(self):
        samples = [ValidatorSample(f"s{i}", i % 2, "test") for i in range(100)]

        train, val, test = split_samples(samples, seed=42)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_custom_split_ratios(self):
        samples = [ValidatorSample(f"s{i}", i % 2, "test") for i in range(100)]

        train, val, test = split_samples(samples, train=0.7, val=0.15, test=0.15, seed=42)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_deterministic_with_seed(self):
        samples = [ValidatorSample(f"s{i}", i % 2, "test") for i in range(100)]

        train1, _, _ = split_samples(samples, seed=42)
        train2, _, _ = split_samples(samples, seed=42)

        assert [s.text for s in train1] == [s.text for s in train2]


class TestJsonl:
    def test_write_and_read(self, tmp_path):
        samples = [
            ValidatorSample("secret1", 1, "source1"),
            ValidatorSample("secret2", 0, "source2"),
        ]
        filepath = tmp_path / "test.jsonl"

        write_jsonl(samples, filepath)
        loaded = list(read_jsonl(filepath, ValidatorSample))

        assert len(loaded) == 2
        assert loaded[0].text == "secret1"
        assert loaded[1].label == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_pipeline.py -v`
Expected: FAIL with "cannot import name 'deduplicate_samples'"

**Step 3: Implement**

```python
# src/heuristic_secrets/data/pipeline.py
"""Data pipeline utilities for deduplication and splitting."""

import json
import random
from pathlib import Path
from typing import Iterator, Type, TypeVar

from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample


T = TypeVar("T", ValidatorSample, SpanFinderSample)


def deduplicate_samples(samples: list[T]) -> list[T]:
    """Remove duplicate samples based on text content.

    Keeps the first occurrence of each unique text.

    Args:
        samples: List of samples to deduplicate

    Returns:
        Deduplicated list of samples
    """
    seen = set()
    result = []

    for sample in samples:
        if sample.text not in seen:
            seen.add(sample.text)
            result.append(sample)

    return result


def split_samples(
    samples: list[T],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int | None = None,
) -> tuple[list[T], list[T], list[T]]:
    """Split samples into train/val/test sets.

    Args:
        samples: List of samples to split
        train: Fraction for training set
        val: Fraction for validation set
        test: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val, test) sample lists
    """
    assert abs(train + val + test - 1.0) < 0.01, "Splits must sum to 1.0"

    # Shuffle with seed
    shuffled = samples.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled)

    # Calculate split indices
    n = len(shuffled)
    train_end = int(n * train)
    val_end = train_end + int(n * val)

    return (
        shuffled[:train_end],
        shuffled[train_end:val_end],
        shuffled[val_end:],
    )


def write_jsonl(samples: list[T], filepath: Path) -> None:
    """Write samples to a JSONL file.

    Args:
        samples: Samples to write
        filepath: Path to output file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")


def read_jsonl(filepath: Path, sample_type: Type[T]) -> Iterator[T]:
    """Read samples from a JSONL file.

    Args:
        filepath: Path to input file
        sample_type: The sample class to instantiate

    Yields:
        Sample objects
    """
    with open(filepath) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                yield sample_type.from_dict(data)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_pipeline.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/pipeline.py tests/data/test_pipeline.py
git commit -m "feat(data): add deduplication and splitting utilities"
```

---

## Task 10: Build Script

**Files:**
- Create: `src/heuristic_secrets/data/build.py`
- Create: `tests/data/test_build.py`

**Step 1: Write the failing test**

```python
# tests/data/test_build.py
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from heuristic_secrets.data.build import DataBuilder, Manifest
from heuristic_secrets.data.types import ValidatorSample


class TestDataBuilder:
    def test_build_with_generated_only(self, tmp_path):
        builder = DataBuilder(
            output_dir=tmp_path,
            sources=["generated_uuids"],
        )

        # Use small counts for testing
        builder.registry.get("generated_uuids").count = 100

        manifest = builder.build()

        # Check files exist
        assert (tmp_path / "splits" / "validator" / "train.jsonl").exists()
        assert (tmp_path / "splits" / "validator" / "val.jsonl").exists()
        assert (tmp_path / "splits" / "validator" / "test.jsonl").exists()
        assert (tmp_path / "manifest.json").exists()

        # Check manifest
        assert manifest.sources["generated_uuids"]["samples"] == 100
        assert manifest.splits["validator"]["train"] == 80
        assert manifest.splits["validator"]["val"] == 10
        assert manifest.splits["validator"]["test"] == 10

    def test_build_creates_manifest(self, tmp_path):
        builder = DataBuilder(
            output_dir=tmp_path,
            sources=["generated_uuids"],
        )
        builder.registry.get("generated_uuids").count = 50

        builder.build()

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            data = json.load(f)

        assert "created_at" in data
        assert "sources" in data
        assert "splits" in data
        assert "deduplication" in data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_build.py -v`
Expected: FAIL with "cannot import name 'DataBuilder'"

**Step 3: Implement**

```python
# src/heuristic_secrets/data/build.py
"""Build script for training data."""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.registry import get_default_registry, CollectorRegistry
from heuristic_secrets.data.pipeline import (
    deduplicate_samples,
    split_samples,
    write_jsonl,
)
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample


@dataclass
class Manifest:
    """Build manifest tracking sources and outputs."""

    created_at: str = ""
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    splits: dict[str, dict[str, int]] = field(default_factory=dict)
    deduplication: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "sources": self.sources,
            "splits": self.splits,
            "deduplication": self.deduplication,
        }


class DataBuilder:
    """Build training data from multiple sources."""

    def __init__(
        self,
        output_dir: Path,
        sources: list[str] | None = None,
        no_fetch: bool = False,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.no_fetch = no_fetch
        self.seed = seed
        self.registry = get_default_registry()

        # Filter to requested sources
        if sources:
            self._filter_sources(sources)

    def _filter_sources(self, sources: list[str]) -> None:
        """Keep only the specified sources."""
        filtered = CollectorRegistry()
        for name in sources:
            collector = self.registry.get(name)
            if collector:
                filtered._collectors[name] = collector
            else:
                print(f"[warn] Unknown source: {name}", file=sys.stderr)
        self.registry = filtered

    def build(self) -> Manifest:
        """Run the full build pipeline."""
        manifest = Manifest(created_at=datetime.utcnow().isoformat() + "Z")

        # Setup phase
        if not self.no_fetch:
            self._setup_collectors()

        # Collect phase
        validator_samples: list[ValidatorSample] = []
        spanfinder_samples: list[SpanFinderSample] = []

        for collector in self.registry.all():
            print(f"[collect] {collector.name}...", end=" ", flush=True)

            samples = list(collector.collect())
            count = len(samples)

            if collector.data_type == "validator":
                validator_samples.extend(samples)
            else:
                spanfinder_samples.extend(samples)

            manifest.sources[collector.name] = {"samples": count}
            print(f"{count} samples")

        # Deduplicate phase
        print("[dedupe] validator...", end=" ", flush=True)
        before = len(validator_samples)
        validator_samples = deduplicate_samples(validator_samples)
        after = len(validator_samples)
        manifest.deduplication["validator_before"] = before
        manifest.deduplication["validator_after"] = after
        print(f"{before} -> {after} (removed {before - after})")

        # Split phase
        if validator_samples:
            print("[split] validator...", end=" ", flush=True)
            train, val, test = split_samples(validator_samples, seed=self.seed)
            manifest.splits["validator"] = {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            }
            print(f"train={len(train)} val={len(val)} test={len(test)}")

            # Write output files
            splits_dir = self.output_dir / "splits" / "validator"
            write_jsonl(train, splits_dir / "train.jsonl")
            write_jsonl(val, splits_dir / "val.jsonl")
            write_jsonl(test, splits_dir / "test.jsonl")

        if spanfinder_samples:
            print("[split] spanfinder...", end=" ", flush=True)
            train, val, test = split_samples(spanfinder_samples, seed=self.seed)
            manifest.splits["spanfinder"] = {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            }
            print(f"train={len(train)} val={len(val)} test={len(test)}")

            splits_dir = self.output_dir / "splits" / "spanfinder"
            write_jsonl(train, splits_dir / "train.jsonl")
            write_jsonl(val, splits_dir / "val.jsonl")
            write_jsonl(test, splits_dir / "test.jsonl")

        # Write manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        print(f"[done] Wrote manifest to {manifest_path}")
        return manifest

    def _setup_collectors(self) -> None:
        """Run setup on all collectors."""
        for collector in self.registry.all():
            if hasattr(collector, "setup"):
                print(f"[setup] {collector.name}...", end=" ", flush=True)
                try:
                    collector.setup()
                    print("done")
                except Exception as e:
                    print(f"failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build training data")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data"),
        help="Output directory",
    )
    parser.add_argument(
        "--sources",
        type=str,
        help="Comma-separated list of sources to use",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Don't fetch/clone source repositories",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else None

    builder = DataBuilder(
        output_dir=args.output,
        sources=sources,
        no_fetch=args.no_fetch,
        seed=args.seed,
    )
    builder.build()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_build.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/build.py tests/data/test_build.py
git commit -m "feat(data): add build script for training data pipeline"
```

---

## Task 11: Integration Test - Full Pipeline

**Files:**
- Create: `tests/data/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/data/test_integration.py
"""Integration tests for the data pipeline."""

import pytest
import json
from pathlib import Path

from heuristic_secrets.data.build import DataBuilder
from heuristic_secrets.data.pipeline import read_jsonl
from heuristic_secrets.data.types import ValidatorSample


class TestFullPipeline:
    def test_build_with_all_generators(self, tmp_path):
        """Test building data with all generated sources."""
        builder = DataBuilder(
            output_dir=tmp_path,
            sources=[
                "generated_uuids",
                "generated_md5",
                "generated_sha256",
                "generated_base64",
            ],
        )

        # Use small counts
        for name in builder.registry.list():
            collector = builder.registry.get(name)
            if hasattr(collector, "count"):
                collector.count = 50

        manifest = builder.build()

        # Verify all sources collected
        assert "generated_uuids" in manifest.sources
        assert "generated_md5" in manifest.sources
        assert "generated_sha256" in manifest.sources
        assert "generated_base64" in manifest.sources

        # Verify splits created
        train_path = tmp_path / "splits" / "validator" / "train.jsonl"
        assert train_path.exists()

        # Verify data is readable
        samples = list(read_jsonl(train_path, ValidatorSample))
        assert len(samples) > 0
        assert all(s.label == 0 for s in samples)  # All false positives

    def test_samples_have_correct_format(self, tmp_path):
        """Test that output samples have the expected format."""
        builder = DataBuilder(
            output_dir=tmp_path,
            sources=["generated_uuids"],
        )
        builder.registry.get("generated_uuids").count = 10
        builder.build()

        train_path = tmp_path / "splits" / "validator" / "train.jsonl"

        with open(train_path) as f:
            for line in f:
                data = json.loads(line)
                assert "text" in data
                assert "label" in data
                assert "source" in data
                assert isinstance(data["text"], str)
                assert data["label"] in [0, 1]

    def test_deduplication_works(self, tmp_path):
        """Test that duplicate samples are removed."""
        from heuristic_secrets.data.collector import Collector
        from heuristic_secrets.data.registry import CollectorRegistry

        # Create a collector that produces duplicates
        class DuplicateCollector(Collector):
            name = "duplicates"
            data_type = "validator"

            def collect(self):
                for _ in range(10):
                    yield ValidatorSample("same_text", 1, "duplicates")
                yield ValidatorSample("different_text", 1, "duplicates")

        registry = CollectorRegistry()
        registry.register(DuplicateCollector)

        builder = DataBuilder(output_dir=tmp_path, sources=["duplicates"])
        builder.registry = registry

        manifest = builder.build()

        # Should have deduplicated to 2 unique samples
        total = (
            manifest.splits["validator"]["train"]
            + manifest.splits["validator"]["val"]
            + manifest.splits["validator"]["test"]
        )
        assert total == 2
```

**Step 2: Run integration tests**

Run: `pytest tests/data/test_integration.py -v`
Expected: All PASS

**Step 3: Run all data tests**

Run: `pytest tests/data/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/data/test_integration.py
git commit -m "test(data): add integration tests for data pipeline"
```

---

## Task 12: CLI Entry Point

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add CLI entry point**

Add to `pyproject.toml`:

```toml
[project.scripts]
heuristic-secrets-data = "heuristic_secrets.data.build:main"
```

**Step 2: Verify CLI works**

Run: `pip install -e . && heuristic-secrets-data --help`
Expected: Shows help text

**Step 3: Test CLI with generated data only**

Run: `heuristic-secrets-data --sources generated_uuids,generated_md5 --output /tmp/test-data`
Expected: Builds data successfully

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(data): add CLI entry point for data building"
```

---

## Summary

This plan creates a complete training data pipeline with:

1. **Task 1**: Document readily-available data sources
2. **Tasks 2-3**: Base infrastructure (types, collector ABC, git utils)
3. **Task 4**: Generated false positives (UUIDs, hashes, base64)
4. **Tasks 5-7**: Tool collectors (detect-secrets, Gitleaks, TruffleHog)
5. **Task 8**: Collector registry
6. **Task 9**: Deduplication and splitting utilities
7. **Task 10**: Build script
8. **Task 11**: Integration tests
9. **Task 12**: CLI entry point

**Total: 12 tasks, ~24 commits**

After completion, you can run:
```bash
# Build with all sources (will clone repos on first run)
heuristic-secrets-data --output data/

# Build with generated data only (instant, no network)
heuristic-secrets-data --sources generated_uuids,generated_md5,generated_sha256,generated_base64 --output data/
```
