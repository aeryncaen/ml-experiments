import pytest
from pathlib import Path
from unittest.mock import patch

from heuristic_secrets.data.collectors.detect_secrets import DetectSecretsCollector
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample


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
        secrets = set(collector._extract_constants_from_python(code))

        assert len(secrets) == 4  # Deduplicated
        assert 'AKIAIOSFODNN7EXAMPLE' in secrets
        assert 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY' in secrets

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

    def test_extract_pytest_parametrize_secrets(self):
        """Test extraction from @pytest.mark.parametrize decorators."""
        collector = DetectSecretsCollector()

        code = '''
import pytest

@pytest.mark.parametrize(
    'line,should_flag',
    [
        (
            'AKIAZZZZZZZZZZZZZZZZ',
            True,
        ),
        (
            'akiazzzzzzzzzzzzzzzz',
            False,
        ),
        (
            'A3T0ZZZZZZZZZZZZZZZZ',
            True,
        ),
        (
            'aws_access_key = "{}"'.format(EXAMPLE_SECRET),
            True,
        ),
    ],
)
def test_analyze(self, line, should_flag):
    pass
'''
        secrets = list(collector._extract_constants_from_python(code))

        assert 'AKIAZZZZZZZZZZZZZZZZ' in secrets
        assert 'A3T0ZZZZZZZZZZZZZZZZ' in secrets
        # Lowercase shouldn't match - it's not a real secret
        assert 'akiazzzzzzzzzzzzzzzz' not in secrets

    def test_extract_self_attribute_assignments(self):
        """Test extraction from self.attr = 'secret' patterns in setup methods."""
        collector = DetectSecretsCollector()

        code = '''
class TestAWSKeyDetector:

    def setup(self):
        self.example_key = 'AKIAZZZZZZZZZZZZZZZZ'
        self.secret_value = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'

    def test_something(self):
        pass
'''
        secrets = list(collector._extract_constants_from_python(code))

        assert 'AKIAZZZZZZZZZZZZZZZZ' in secrets
        assert 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY' in secrets

    def test_extract_from_ini_file(self):
        """Test extraction from INI config files."""
        collector = DetectSecretsCollector()

        content = '''
[credentials]
password = 123456789a1234

[aws]
aws_secret_key = 23456789a1

[key with id in name]
real_secret_which_isnt_an_i_d = vh987tyw9ehy8ghis7vwyhiwbwitefy7w3ASDGYDGUASDG
'''
        secrets = list(collector._extract_secrets_from_ini(content))

        assert '123456789a1234' in secrets
        assert 'vh987tyw9ehy8ghis7vwyhiwbwitefy7w3ASDGYDGUASDG' in secrets

    def test_extract_from_env_file(self):
        """Test extraction from .env files."""
        collector = DetectSecretsCollector()

        content = '''
mimi=gX69YO4CvBsVjzAwYxdGyDd30t5+9ez31gKATtj4
API_KEY=sk-1234567890abcdefghijklmnop
'''
        secrets = list(collector._extract_secrets_from_env(content))

        assert 'gX69YO4CvBsVjzAwYxdGyDd30t5+9ez31gKATtj4' in secrets
        assert 'sk-1234567890abcdefghijklmnop' in secrets


class TestDetectSecretsCollectorSpans:
    def test_find_secret_positions_in_text(self):
        collector = DetectSecretsCollector()

        text = "prefix AKIAIOSFODNN7EXAMPLE suffix"
        secret = "AKIAIOSFODNN7EXAMPLE"

        positions = collector._find_all_positions(text, secret)

        assert len(positions) == 1
        start, end = positions[0]
        assert text[start:end] == secret

    def test_find_multiple_occurrences(self):
        collector = DetectSecretsCollector()

        text = "key1=SECRET123ABC key2=SECRET123ABC"
        secret = "SECRET123ABC"

        positions = collector._find_all_positions(text, secret)

        assert len(positions) == 2
        for start, end in positions:
            assert text[start:end] == secret

    def test_extract_spans_from_python_file(self):
        collector = DetectSecretsCollector()

        code = '''EXAMPLE_SECRET = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
OTHER_KEY = 'AKIAIOSFODNN7EXAMPLE'
'''
        sample = collector._file_to_span_sample(code, "test.py")

        assert isinstance(sample, SpanFinderSample)
        assert sample.text == code
        assert sample.source == "detect_secrets"
        assert len(sample.starts) == 2
        assert len(sample.ends) == 2

        for start, end in zip(sample.starts, sample.ends):
            extracted = code[start:end]
            assert extracted in [
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "AKIAIOSFODNN7EXAMPLE",
            ]

    def test_extract_spans_from_ini_file(self):
        collector = DetectSecretsCollector()

        content = '''[aws]
secret_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
'''
        sample = collector._file_to_span_sample(content, "config.ini")

        assert isinstance(sample, SpanFinderSample)
        assert len(sample.starts) >= 1

        found_secret = False
        for start, end in zip(sample.starts, sample.ends):
            if content[start:end] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY":
                found_secret = True
        assert found_secret
