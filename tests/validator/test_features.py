import math
import os
import pytest
from heuristic_secrets.validator.features import (
    shannon_entropy,
    kolmogorov_complexity,
    char_frequency_difference,
    char_type_mix,
    extract_features,
    FrequencyTable,
    FeatureSet,
)


class TestShannonEntropy:
    def test_uniform_distribution_max_entropy(self):
        # All unique characters = maximum entropy
        # 256 unique bytes = log2(256) = 8 bits
        data = bytes(range(256))
        result = shannon_entropy(data)
        assert abs(result - 8.0) < 0.01

    def test_single_repeated_char_zero_entropy(self):
        # All same character = zero entropy
        data = b"aaaaaaaaaa"
        result = shannon_entropy(data)
        assert result == 0.0

    def test_two_chars_equal_frequency(self):
        # Two chars, equal frequency = 1 bit
        data = b"abababab"
        result = shannon_entropy(data)
        assert abs(result - 1.0) < 0.01

    def test_empty_string_zero_entropy(self):
        data = b""
        result = shannon_entropy(data)
        assert result == 0.0

    def test_typical_secret_high_entropy(self):
        # API key-like string should have high entropy (> 4 bits)
        data = b"sk-abc123XYZ789def456"
        result = shannon_entropy(data)
        assert result > 4.0

    def test_english_text_medium_entropy(self):
        # English text typically 3-4 bits
        data = b"the quick brown fox jumps over the lazy dog"
        result = shannon_entropy(data)
        assert 3.0 < result < 5.0


class TestKolmogorovComplexity:
    def test_repeated_char_low_complexity(self):
        # Highly compressible = low complexity ratio
        data = b"a" * 100
        result = kolmogorov_complexity(data)
        assert result < 0.2  # Should compress very well

    def test_random_bytes_high_complexity(self):
        # Random data = high complexity ratio (incompressible)
        data = os.urandom(100)
        result = kolmogorov_complexity(data)
        assert result > 0.7  # Should not compress much

    def test_empty_returns_zero(self):
        result = kolmogorov_complexity(b"")
        assert result == 0.0

    def test_single_byte_returns_one(self):
        result = kolmogorov_complexity(b"x")
        assert result == 1.0  # Can't compress single byte

    def test_typical_secret_high_complexity(self):
        data = b"ghp_1234567890abcdefABCDEF"
        result = kolmogorov_complexity(data)
        assert result > 0.5  # Secrets have moderate-high complexity


class TestCharFrequencyDifference:
    def test_identical_distribution_zero_diff(self):
        data = b"aabbcc"
        reference = FrequencyTable({ord("a"): 1/3, ord("b"): 1/3, ord("c"): 1/3})
        result = char_frequency_difference(data, reference)
        assert abs(result) < 0.01

    def test_completely_different_distribution(self):
        data = b"aaaa"  # Only 'a'
        reference = FrequencyTable({ord("z"): 1.0})  # Only 'z' expected
        result = char_frequency_difference(data, reference)
        assert result > 1.0  # Large difference

    def test_empty_data_zero_diff(self):
        reference = FrequencyTable({ord("a"): 1.0})
        result = char_frequency_difference(b"", reference)
        assert result == 0.0

    def test_partial_overlap(self):
        data = b"aabb"
        reference = FrequencyTable({ord("a"): 0.5, ord("c"): 0.5})
        result = char_frequency_difference(data, reference)
        assert result > 0.5  # Moderate difference


class TestCharTypeMix:
    def test_all_same_type_zero_transitions(self):
        # All lowercase letters - no transitions
        data = b"abcdefgh"
        result = char_type_mix(data)
        assert result == 0.0

    def test_alternating_types_max_transitions(self):
        # Alternating letter/digit = transition every char
        data = b"a1b2c3d4"
        result = char_type_mix(data)
        assert result > 0.8  # High transition rate

    def test_typical_secret_high_mix(self):
        # Secrets often mix letters, digits, special chars
        data = b"sk-abc123_XYZ"
        result = char_type_mix(data)
        assert result > 0.3

    def test_empty_returns_zero(self):
        result = char_type_mix(b"")
        assert result == 0.0

    def test_single_char_returns_zero(self):
        result = char_type_mix(b"a")
        assert result == 0.0

    def test_english_words_low_mix(self):
        # Normal words have few transitions
        data = b"hello world"
        result = char_type_mix(data)
        assert result < 0.3


class TestExtractFeatures:
    def test_returns_six_features(self):
        data = b"sk-abc123XYZ"
        text_freq = FrequencyTable({ord("e"): 0.12, ord("t"): 0.09})
        innocuous_freq = FrequencyTable({ord("a"): 0.05, ord("e"): 0.04})
        secret_freq = FrequencyTable({ord("a"): 0.08, ord("0"): 0.06})

        result = extract_features(data, text_freq, innocuous_freq, secret_freq)

        assert isinstance(result, FeatureSet)
        assert len(result.to_list()) == 6

    def test_feature_order_is_correct(self):
        data = b"test"
        text_freq = FrequencyTable({})
        innocuous_freq = FrequencyTable({})
        secret_freq = FrequencyTable({})

        result = extract_features(data, text_freq, innocuous_freq, secret_freq)
        features = result.to_list()

        assert all(isinstance(f, float) for f in features)

    def test_empty_data_returns_zeros(self):
        text_freq = FrequencyTable({})
        innocuous_freq = FrequencyTable({})
        secret_freq = FrequencyTable({})

        result = extract_features(b"", text_freq, innocuous_freq, secret_freq)
        features = result.to_list()

        assert features == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
