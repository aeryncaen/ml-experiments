import heapq
import math
from collections import Counter
from dataclasses import dataclass, field


def shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy in bits per byte."""
    if len(data) == 0:
        return 0.0

    counts = Counter(data)
    length = len(data)
    entropy = 0.0

    for count in counts.values():
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)

    return entropy


def _huffman_encoded_length(data: bytes) -> int:
    """Calculate the length in bits if Huffman encoded."""
    if len(data) == 0:
        return 0

    counts = Counter(data)

    if len(counts) == 1:
        return len(data)

    # Build Huffman tree using heap
    heap = [[count, [byte, ""]] for byte, count in counts.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Calculate total bits needed
    codes = {byte: code for byte, code in heap[0][1:]}
    total_bits = sum(len(codes[b]) * c for b, c in counts.items())

    return total_bits


def kolmogorov_complexity(data: bytes) -> float:
    """Estimate Kolmogorov complexity using Huffman coding.

    Returns ratio of compressed size to original size.
    Higher = more random/complex, Lower = more compressible/structured.
    """
    if len(data) == 0:
        return 0.0

    if len(data) == 1:
        return 1.0

    original_bits = len(data) * 8
    compressed_bits = _huffman_encoded_length(data)

    return compressed_bits / original_bits


@dataclass
class FrequencyTable:
    frequencies: dict[int, float] = field(default_factory=dict)

    def get(self, byte: int) -> float:
        return self.frequencies.get(byte, 0.0)

    def to_dict(self) -> dict[str, float]:
        return {str(k): v for k, v in self.frequencies.items()}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "FrequencyTable":
        return cls({int(k): v for k, v in d.items()})

    @classmethod
    def from_data(cls, data: bytes) -> "FrequencyTable":
        if len(data) == 0:
            return cls({})
        counts = Counter(data)
        total = len(data)
        return cls({byte: count / total for byte, count in counts.items()})


def char_frequency_difference(data: bytes, reference: FrequencyTable) -> float:
    """Calculate L2 distance between data's char frequency and reference."""
    if len(data) == 0:
        return 0.0

    data_freq = FrequencyTable.from_data(data)

    # Get all bytes present in either distribution
    all_bytes = set(data_freq.frequencies.keys()) | set(reference.frequencies.keys())

    # Calculate L2 distance
    sum_squared = 0.0
    for byte in all_bytes:
        diff = data_freq.get(byte) - reference.get(byte)
        sum_squared += diff * diff

    return math.sqrt(sum_squared)


def _char_type(byte: int) -> int:
    """Classify byte into type: 0=letter, 1=digit, 2=special."""
    if (65 <= byte <= 90) or (97 <= byte <= 122):  # A-Z or a-z
        return 0
    elif 48 <= byte <= 57:  # 0-9
        return 1
    else:
        return 2


def char_type_mix(data: bytes) -> float:
    """Calculate ratio of character type transitions.

    Measures how often the character type changes (letter->digit,
    digit->special, etc.) as a fraction of possible transitions.

    Args:
        data: Input bytes

    Returns:
        Transition ratio (0.0 to 1.0)
    """
    if len(data) <= 1:
        return 0.0

    transitions = 0
    prev_type = _char_type(data[0])

    for byte in data[1:]:
        curr_type = _char_type(byte)
        if curr_type != prev_type:
            transitions += 1
        prev_type = curr_type

    # Normalize by number of possible transitions
    return transitions / (len(data) - 1)


from typing import NamedTuple


class FeatureSet(NamedTuple):
    """The 6 features used by the validator model."""
    shannon_entropy: float
    kolmogorov_complexity: float
    text_freq_diff: float
    innocuous_freq_diff: float
    secret_freq_diff: float
    char_type_mix: float

    def to_list(self) -> list[float]:
        return list(self)


def extract_features(
    data: bytes,
    text_freq: FrequencyTable,
    innocuous_freq: FrequencyTable,
    secret_freq: FrequencyTable,
) -> FeatureSet:
    """Extract all 6 features from input data."""
    return FeatureSet(
        shannon_entropy=shannon_entropy(data),
        kolmogorov_complexity=kolmogorov_complexity(data),
        text_freq_diff=char_frequency_difference(data, text_freq),
        innocuous_freq_diff=char_frequency_difference(data, innocuous_freq),
        secret_freq_diff=char_frequency_difference(data, secret_freq),
        char_type_mix=char_type_mix(data),
    )
