"""Collector for TruffleHog test fixtures (unit tests only)."""

import re
from pathlib import Path
from typing import Iterator

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.git_utils import clone_or_pull
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample, SecretCategory


class TrufflehogCollector(Collector):
    """Collect secrets from TruffleHog unit test fixtures.

    Note: Integration tests require GCP Secret Manager access, so we skip those.
    We only parse unit tests (*_test.go files that don't use common.GetSecret).
    """

    name = "trufflehog"
    data_type = "validator"
    repo_url = "https://github.com/trufflesecurity/trufflehog.git"

    MIN_SECRET_LENGTH = 15  # TruffleHog secrets tend to be longer

    def setup(self, force: bool = False) -> None:
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
                        category = self._infer_category(secret, test_file.name)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                            category=category.value,
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

        # Filter out strings with whitespace (newlines, tabs) - these are parsing artifacts
        if "\n" in value or "\t" in value or "\r" in value:
            return False

        # Filter out strings that are mostly whitespace or start with whitespace
        stripped = value.strip()
        if len(stripped) < self.MIN_SECRET_LENGTH:
            return False

        # Filter out escaped whitespace sequences (literal \n, \t in string)
        if "\\n" in value or "\\t" in value:
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

    def _infer_category(self, secret: str, filename: str = "") -> SecretCategory:
        """Infer secret category from content and filename."""
        secret_lower = secret.lower()
        filename_lower = filename.lower()
        
        # Private keys
        if "-----BEGIN" in secret and "PRIVATE KEY" in secret:
            return SecretCategory.PRIVATE_KEY
        
        # AWS
        if secret.startswith("AKIA") or secret.startswith("ABIA"):
            return SecretCategory.AUTH_TOKEN
        if "aws" in filename_lower:
            return SecretCategory.AUTH_TOKEN
        
        # GitHub
        if secret.startswith("ghp_") or secret.startswith("ghs_") or secret.startswith("gho_"):
            return SecretCategory.AUTH_TOKEN
        if secret.startswith("github_pat_"):
            return SecretCategory.AUTH_TOKEN
        if "github" in filename_lower:
            return SecretCategory.AUTH_TOKEN
        
        # Slack
        if secret.startswith("xox"):
            return SecretCategory.AUTH_TOKEN
        if "slack" in filename_lower:
            return SecretCategory.AUTH_TOKEN
        
        # Stripe
        if secret.startswith("sk_live_") or secret.startswith("sk_test_"):
            return SecretCategory.API_KEY
        if "stripe" in filename_lower:
            return SecretCategory.API_KEY
        
        # SendGrid
        if secret.startswith("SG."):
            return SecretCategory.API_KEY
        
        # NPM/PyPI tokens
        if secret.startswith("npm_") or secret.startswith("pypi-"):
            return SecretCategory.AUTH_TOKEN
        
        # JWT
        if re.match(r"^eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$", secret):
            return SecretCategory.AUTH_TOKEN
        
        # Database connection strings
        if "://" in secret and "@" in secret:
            return SecretCategory.CONNECTION_STRING
        if any(secret_lower.startswith(p) for p in ["postgres://", "mysql://", "mongodb://", "redis://"]):
            return SecretCategory.CONNECTION_STRING
        
        # Filename hints
        if "private" in filename_lower and "key" in filename_lower:
            return SecretCategory.PRIVATE_KEY
        if "password" in filename_lower:
            return SecretCategory.PASSWORD
        if "database" in filename_lower or "db" in filename_lower:
            return SecretCategory.CONNECTION_STRING
        if "api" in filename_lower and "key" in filename_lower:
            return SecretCategory.API_KEY
        if "token" in filename_lower:
            return SecretCategory.AUTH_TOKEN
        
        return SecretCategory.GENERIC_SECRET

    def _find_all_positions(self, text: str, secret: str) -> list[tuple[int, int]]:
        """Find all (start, end) positions of a secret in text."""
        positions = []
        start = 0
        while True:
            idx = text.find(secret, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(secret)))
            start = idx + 1
        return positions

    def _file_to_span_sample(self, content: str) -> SpanFinderSample | None:
        """Convert file content to a SpanFinderSample with marked secret positions."""
        # Skip integration test files
        if "common.GetSecret" in content or "MustGetField" in content:
            return None

        secrets = set(self._extract_from_go_test(content))
        if not secrets:
            return None

        starts = []
        ends = []
        for secret in secrets:
            for start, end in self._find_all_positions(content, secret):
                starts.append(start)
                ends.append(end)

        if not starts:
            return None

        return SpanFinderSample(
            text=content,
            starts=starts,
            ends=ends,
            source=self.name,
        )

    def collect_spans(self) -> Iterator[SpanFinderSample]:
        """Yield SpanFinder samples with full file context."""
        repo_path = self.cache_path()

        if not self.is_available():
            return

        seen_files = set()

        detectors_dir = repo_path / "pkg" / "detectors"
        if detectors_dir.exists():
            for test_file in detectors_dir.rglob("*_test.go"):
                if "integration" in test_file.name.lower():
                    continue
                if test_file in seen_files:
                    continue
                seen_files.add(test_file)

                content = test_file.read_text(errors="ignore")
                sample = self._file_to_span_sample(content)
                if sample:
                    yield sample
