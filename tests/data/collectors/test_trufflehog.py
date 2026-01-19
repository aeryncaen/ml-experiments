import pytest

from heuristic_secrets.data.collectors.trufflehog import TrufflehogCollector
from heuristic_secrets.data.types import ValidatorSample, SpanFinderSample


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

    def test_filters_garbage_strings(self):
        """Filter out strings with newlines, tabs, and code artifacts."""
        collector = TrufflehogCollector()

        # These are real patterns found in the output - garbage parsing artifacts
        garbage_strings = [
            "\n\t\t//},\n\t\t//\t\twant: []string{",
            "\\n\\t\\ttoken := \\\"abc123\\\"",
            "some_value\n\twith_newlines",
            "\t\t\t\tindented_garbage",
        ]

        for garbage in garbage_strings:
            assert not collector._is_valid_secret(garbage), f"Should filter: {repr(garbage)}"

    def test_accepts_valid_secrets(self):
        """Ensure valid secrets are still accepted."""
        collector = TrufflehogCollector()

        valid_secrets = [
            "ghs_RWGUZ6kS8_Ut7PbtR72k2miJwwYtxkpe8mOp",
            "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",
            "sk-1234567890abcdefghijklmnopqrstuvwxyz",
            "AKIAIOSFODNN7EXAMPLE",
        ]

        for secret in valid_secrets:
            assert collector._is_valid_secret(secret), f"Should accept: {secret}"


class TestTrufflehogCollectorSpans:
    def test_file_to_span_sample(self):
        collector = TrufflehogCollector()

        code = '''secret := "ghs_RWGUZ6kS8_Ut7PbtR72k2miJwwYtxkpe8mOp"
token := "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"
'''
        sample = collector._file_to_span_sample(code)

        assert isinstance(sample, SpanFinderSample)
        assert sample.text == code
        assert sample.source == "trufflehog"
        assert len(sample.starts) >= 2

        for start, end in zip(sample.starts, sample.ends):
            extracted = code[start:end]
            assert extracted in [
                "ghs_RWGUZ6kS8_Ut7PbtR72k2miJwwYtxkpe8mOp",
                "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",
            ]

    def test_skips_integration_test_files(self):
        collector = TrufflehogCollector()

        code = '''
testSecrets, err := common.GetSecret(ctx, "project", "secret-name")
secret := testSecrets.MustGetField("API_KEY")
'''
        sample = collector._file_to_span_sample(code)

        assert sample is None
