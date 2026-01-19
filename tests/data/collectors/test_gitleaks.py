import pytest

from heuristic_secrets.data.collectors.gitleaks import GitleaksCollector
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample


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


class TestGitleaksCollectorSpans:
    def test_file_to_span_sample(self):
        collector = GitleaksCollector()

        code = '''awsToken := "AKIALALEMEL33243OLIA"
password := "super_secret_password_123"
'''
        sample = collector._file_to_span_sample(code, "test.go")

        assert isinstance(sample, SpanFinderSample)
        assert sample.text == code
        assert sample.source == "gitleaks"
        assert len(sample.starts) >= 2

        for start, end in zip(sample.starts, sample.ends):
            extracted = code[start:end]
            assert extracted in ["AKIALALEMEL33243OLIA", "super_secret_password_123"]

    def test_span_positions_are_correct(self):
        collector = GitleaksCollector()

        code = 'token := "ghp_1234567890abcdefABCDEF1234567890ab"'
        sample = collector._file_to_span_sample(code, "test.go")

        assert sample is not None
        assert len(sample.starts) == 1

        start, end = sample.starts[0], sample.ends[0]
        assert code[start:end] == "ghp_1234567890abcdefABCDEF1234567890ab"
