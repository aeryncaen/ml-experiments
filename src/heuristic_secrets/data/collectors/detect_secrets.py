"""Collector for detect-secrets test fixtures."""

import ast
import re
from pathlib import Path
from typing import Iterator

from heuristic_secrets.data.collector import Collector
from heuristic_secrets.data.git_utils import clone_or_pull, get_repo_version
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample, SecretCategory


class DetectSecretsCollector(Collector):
    """Collect secrets from detect-secrets test fixtures."""

    name = "detect_secrets"
    data_type = "validator"
    repo_url = "https://github.com/Yelp/detect-secrets.git"

    MIN_SECRET_LENGTH = 10  # Filter out short strings

    def setup(self, force: bool = False) -> None:
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
                        category = self._infer_category(secret, test_file.name)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                            category=category.value,
                        )

        # 2. Parse test_data/each_secret.py
        each_secret = repo_path / "test_data" / "each_secret.py"
        if each_secret.exists():
            content = each_secret.read_text(errors="ignore")
            for secret in self._extract_constants_from_python(content):
                if secret not in seen:
                    seen.add(secret)
                    category = self._infer_category(secret, each_secret.name)
                    yield ValidatorSample(
                        text=secret,
                        label=1,
                        source=self.name,
                        category=category.value,
                    )

        # 3. Parse test_data/ config files
        test_data_dir = repo_path / "test_data"
        if test_data_dir.exists():
            for config_file in test_data_dir.iterdir():
                if not config_file.is_file():
                    continue

                content = config_file.read_text(errors="ignore")
                suffix = config_file.suffix.lower()

                if suffix == ".yaml" or suffix == ".yml":
                    extractor = self._extract_secrets_from_yaml
                elif suffix == ".ini":
                    extractor = self._extract_secrets_from_ini
                elif suffix == ".env" or config_file.name.startswith("config.env"):
                    extractor = self._extract_secrets_from_env
                elif suffix == ".py":
                    extractor = self._extract_constants_from_python
                else:
                    continue

                for secret in extractor(content):
                    if secret not in seen:
                        seen.add(secret)
                        category = self._infer_category(secret, config_file.name)
                        yield ValidatorSample(
                            text=secret,
                            label=1,
                            source=self.name,
                            category=category.value,
                        )

    def _extract_constants_from_python(self, code: str) -> Iterator[str]:
        """Extract string constants from Python code.

        Looks for:
        - Module-level assignments: EXAMPLE_SECRET = 'secret_value'
        - Instance assignments: self.example_key = 'secret_value'
        - Strings in @pytest.mark.parametrize decorators
        - Other string literals in function calls
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            # Pattern 1: Module-level assignments (NAME = 'value')
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, str
                        ):
                            value = node.value.value
                            if self._is_valid_secret(value):
                                yield value

            # Pattern 2: Instance assignments (self.attr = 'value')
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, str
                        ):
                            value = node.value.value
                            if self._is_valid_secret(value):
                                yield value

            # Pattern 3: Strings in lists/tuples (e.g., pytest.mark.parametrize)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if self._is_valid_secret(value):
                    yield value

    def _is_valid_secret(self, value: str) -> bool:
        """Check if a string looks like a valid secret.

        Filters out:
        - Short strings
        - Lowercase-only strings (likely not secrets)
        - Common test placeholders
        """
        if len(value) < self.MIN_SECRET_LENGTH:
            return False

        # Skip pure lowercase strings - real secrets usually have mixed case or numbers
        if value.islower() and value.isalpha():
            return False

        # Skip common placeholders
        placeholders = {
            "mock_filename", "test", "example", "placeholder",
            "line,should_flag",  # pytest parametrize header
        }
        if value.lower() in placeholders:
            return False

        return True

    def _infer_category(self, secret: str, filename: str = "") -> SecretCategory:
        """Infer the category of a secret based on content and filename.
        
        Args:
            secret: The secret value
            filename: Optional filename where secret was found
            
        Returns:
            Inferred SecretCategory
        """
        # Normalize for matching
        secret_lower = secret.lower()
        filename_lower = filename.lower()
        
        # Private keys (content-based)
        if "-----BEGIN" in secret and "PRIVATE KEY" in secret:
            return SecretCategory.PRIVATE_KEY
        if "-----BEGIN RSA PRIVATE" in secret or "-----BEGIN EC PRIVATE" in secret:
            return SecretCategory.PRIVATE_KEY
        if "-----BEGIN PGP PRIVATE" in secret:
            return SecretCategory.PRIVATE_KEY
        
        # AWS credentials
        if secret.startswith("AKIA") or secret.startswith("ABIA") or secret.startswith("ACCA"):
            return SecretCategory.AUTH_TOKEN
        if re.match(r"^[A-Za-z0-9/+=]{40}$", secret):
            # Could be AWS secret key
            if "aws" in filename_lower:
                return SecretCategory.AUTH_TOKEN
        
        # GitHub tokens
        if secret.startswith("ghp_") or secret.startswith("ghs_") or secret.startswith("gho_"):
            return SecretCategory.AUTH_TOKEN
        if secret.startswith("github_pat_"):
            return SecretCategory.AUTH_TOKEN
        
        # Slack tokens
        if secret.startswith("xoxb-") or secret.startswith("xoxp-") or secret.startswith("xoxa-"):
            return SecretCategory.AUTH_TOKEN
        
        # Stripe keys
        if secret.startswith("sk_live_") or secret.startswith("sk_test_"):
            return SecretCategory.API_KEY
        if secret.startswith("pk_live_") or secret.startswith("pk_test_"):
            return SecretCategory.API_KEY
        
        # Twilio
        if secret.startswith("SK") and len(secret) == 34:
            return SecretCategory.API_KEY
        
        # SendGrid
        if secret.startswith("SG."):
            return SecretCategory.API_KEY
        
        # NPM tokens
        if secret.startswith("npm_"):
            return SecretCategory.AUTH_TOKEN
        
        # PyPI tokens
        if secret.startswith("pypi-"):
            return SecretCategory.AUTH_TOKEN
        
        # JWT tokens (have 3 base64 parts separated by dots)
        if re.match(r"^eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$", secret):
            return SecretCategory.AUTH_TOKEN
        
        # Database connection strings
        if "://" in secret and ("@" in secret or "password" in secret_lower):
            return SecretCategory.CONNECTION_STRING
        if secret_lower.startswith("postgres://") or secret_lower.startswith("mysql://"):
            return SecretCategory.CONNECTION_STRING
        if secret_lower.startswith("mongodb://") or secret_lower.startswith("redis://"):
            return SecretCategory.CONNECTION_STRING
        
        # Filename-based inference
        if "private" in filename_lower and "key" in filename_lower:
            return SecretCategory.PRIVATE_KEY
        if "aws" in filename_lower:
            return SecretCategory.AUTH_TOKEN
        if "slack" in filename_lower or "github" in filename_lower:
            return SecretCategory.AUTH_TOKEN
        if "stripe" in filename_lower or "twilio" in filename_lower:
            return SecretCategory.API_KEY
        if "password" in filename_lower:
            return SecretCategory.PASSWORD
        if "database" in filename_lower or "db" in filename_lower:
            return SecretCategory.CONNECTION_STRING
        
        # Default to generic secret
        return SecretCategory.GENERIC_SECRET

    def _extract_secrets_from_yaml(self, content: str) -> Iterator[str]:
        """Extract potential secrets from YAML content."""
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

    def _extract_secrets_from_ini(self, content: str) -> Iterator[str]:
        """Extract potential secrets from INI config files."""
        # Match key = value patterns (with or without quotes)
        pattern = r"^\s*\w+\s*=\s*['\"]?([A-Za-z0-9+/=_-]{10,})['\"]?\s*(?:#.*)?$"

        for line in content.splitlines():
            match = re.match(pattern, line)
            if match:
                value = match.group(1)
                if self._is_valid_secret(value):
                    yield value

    def _extract_secrets_from_env(self, content: str) -> Iterator[str]:
        """Extract potential secrets from .env files."""
        # Match KEY=value patterns
        pattern = r"^[A-Za-z_][A-Za-z0-9_]*=(.+)$"

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            match = re.match(pattern, line)
            if match:
                value = match.group(1).strip().strip("'\"")
                if self._is_valid_secret(value):
                    yield value

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

    def _get_extractor_for_file(self, filename: str):
        """Return the appropriate secret extractor for a file type."""
        suffix = Path(filename).suffix.lower()
        if suffix in (".py",):
            return self._extract_constants_from_python
        elif suffix in (".yaml", ".yml"):
            return self._extract_secrets_from_yaml
        elif suffix in (".ini",):
            return self._extract_secrets_from_ini
        elif suffix in (".env",) or filename.startswith("config.env"):
            return self._extract_secrets_from_env
        return None

    def _file_to_span_sample(self, content: str, filename: str) -> SpanFinderSample | None:
        """Convert file content to a SpanFinderSample with marked secret positions."""
        extractor = self._get_extractor_for_file(filename)
        if extractor is None:
            return None

        secrets = set(extractor(content))
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

        # 1. Test plugin files
        tests_dir = repo_path / "tests" / "plugins"
        if tests_dir.exists():
            for test_file in tests_dir.glob("*_test.py"):
                if test_file in seen_files:
                    continue
                seen_files.add(test_file)

                content = test_file.read_text(errors="ignore")
                sample = self._file_to_span_sample(content, test_file.name)
                if sample:
                    yield sample

        # 2. test_data/ files
        test_data_dir = repo_path / "test_data"
        if test_data_dir.exists():
            for config_file in test_data_dir.iterdir():
                if not config_file.is_file() or config_file in seen_files:
                    continue
                seen_files.add(config_file)

                content = config_file.read_text(errors="ignore")
                sample = self._file_to_span_sample(content, config_file.name)
                if sample:
                    yield sample
