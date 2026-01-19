"""Collector for Gitleaks test fixtures."""

import re
from pathlib import Path
from typing import Iterator

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.git_utils import clone_or_pull
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample, SecretCategory


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

    def setup(self, force: bool = False) -> None:
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
                        category = self._infer_category(secret, go_file.name)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                            category=category.value,
                        )

        # 2. Parse testdata files
        testdata_dir = repo_path / "testdata"
        if testdata_dir.exists():
            for go_file in testdata_dir.rglob("*.go"):
                content = go_file.read_text(errors="ignore")
                for secret in self._extract_string_literals_from_go(content):
                    if secret not in seen:
                        seen.add(secret)
                        category = self._infer_category(secret, go_file.name)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                            category=category.value,
                        )

            # Also check .env files
            for env_file in testdata_dir.rglob("*.env*"):
                content = env_file.read_text(errors="ignore")
                for secret in self._extract_from_env(content):
                    if secret not in seen:
                        seen.add(secret)
                        category = self._infer_category(secret, env_file.name)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                            category=category.value,
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
        if secret.startswith("pk_live_") or secret.startswith("pk_test_"):
            return SecretCategory.API_KEY
        if "stripe" in filename_lower:
            return SecretCategory.API_KEY
        
        # SendGrid
        if secret.startswith("SG."):
            return SecretCategory.API_KEY
        if "sendgrid" in filename_lower:
            return SecretCategory.API_KEY
        
        # Twilio
        if "twilio" in filename_lower:
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

    def _file_to_span_sample(self, content: str, filename: str) -> SpanFinderSample | None:
        """Convert file content to a SpanFinderSample with marked secret positions."""
        suffix = Path(filename).suffix.lower()

        if suffix == ".go":
            secrets = set(self._extract_string_literals_from_go(content))
            secrets.update(self._extract_secrets_from_go(content))
        elif suffix in (".env",) or "env" in filename.lower():
            secrets = set(self._extract_from_env(content))
        else:
            return None

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

        # 1. Rule files
        rules_dir = repo_path / "cmd" / "generate" / "config" / "rules"
        if rules_dir.exists():
            for go_file in rules_dir.glob("*.go"):
                if go_file in seen_files:
                    continue
                seen_files.add(go_file)

                content = go_file.read_text(errors="ignore")
                sample = self._file_to_span_sample(content, go_file.name)
                if sample:
                    yield sample

        # 2. Testdata files
        testdata_dir = repo_path / "testdata"
        if testdata_dir.exists():
            for go_file in testdata_dir.rglob("*.go"):
                if go_file in seen_files:
                    continue
                seen_files.add(go_file)

                content = go_file.read_text(errors="ignore")
                sample = self._file_to_span_sample(content, go_file.name)
                if sample:
                    yield sample

            for env_file in testdata_dir.rglob("*.env*"):
                if env_file in seen_files:
                    continue
                seen_files.add(env_file)

                content = env_file.read_text(errors="ignore")
                sample = self._file_to_span_sample(content, env_file.name)
                if sample:
                    yield sample
