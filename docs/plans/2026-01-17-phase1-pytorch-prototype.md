# Phase 1: Python + PyTorch Prototype Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a working prototype of the two-stage secret detection pipeline (SpanFinder + Heuristic Validator) in Python/PyTorch to validate the approach before optimizing in Rust.

**Architecture:** Byte-level SpanFinder model predicts start/end boundaries per byte, span assembly pairs and stitches candidates across chunks, then Heuristic Validator computes 6 features and classifies each candidate.

**Tech Stack:** Python 3.11+, PyTorch, pytest, numpy

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/heuristic_secrets/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "heuristic-secrets"
version = "0.1.0"
description = "High-performance secret detection using ML"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "safetensors>=0.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

**Step 2: Create directory structure and init files**

```bash
mkdir -p src/heuristic_secrets tests
touch src/heuristic_secrets/__init__.py
touch tests/__init__.py
```

**Step 3: Create .gitignore**

```
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/
*.egg
.pytest_cache/
.coverage
htmlcov/
.ruff_cache/
.venv/
venv/
*.safetensors
models/
```

**Step 4: Install dev dependencies**

Run: `pip install -e ".[dev]"`

**Step 5: Verify setup**

Run: `python -c "import heuristic_secrets; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add pyproject.toml src/ tests/ .gitignore
git commit -m "chore: initialize Python project structure"
```

---

## Task 2: Validator Feature Extraction - Shannon Entropy

**Files:**
- Create: `src/heuristic_secrets/validator/features.py`
- Create: `tests/validator/test_features.py`

**Step 1: Write the failing test**

```python
# tests/validator/test_features.py
import math
import pytest
from heuristic_secrets.validator.features import shannon_entropy


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validator/test_features.py -v`
Expected: FAIL with "ModuleNotFoundError" or "cannot import name 'shannon_entropy'"

**Step 3: Create directory and implement**

```bash
mkdir -p src/heuristic_secrets/validator tests/validator
touch src/heuristic_secrets/validator/__init__.py
touch tests/validator/__init__.py
```

```python
# src/heuristic_secrets/validator/features.py
import math
from collections import Counter


def shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy in bits per byte.
    
    Args:
        data: Input bytes
        
    Returns:
        Entropy in bits (0.0 to 8.0 for byte data)
    """
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validator/test_features.py::TestShannonEntropy -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/validator/ tests/validator/
git commit -m "feat(validator): add Shannon entropy feature extraction"
```

---

## Task 3: Validator Feature Extraction - Kolmogorov Complexity (Huffman)

**Files:**
- Modify: `src/heuristic_secrets/validator/features.py`
- Modify: `tests/validator/test_features.py`

**Step 1: Write the failing test**

```python
# tests/validator/test_features.py (append)
from heuristic_secrets.validator.features import kolmogorov_complexity


class TestKolmogorovComplexity:
    def test_repeated_char_low_complexity(self):
        # Highly compressible = low complexity ratio
        data = b"a" * 100
        result = kolmogorov_complexity(data)
        assert result < 0.2  # Should compress very well

    def test_random_bytes_high_complexity(self):
        # Random data = high complexity ratio (incompressible)
        import os
        data = os.urandom(100)
        result = kolmogorov_complexity(data)
        assert result > 0.8  # Should not compress much

    def test_empty_returns_zero(self):
        result = kolmogorov_complexity(b"")
        assert result == 0.0

    def test_single_byte_returns_one(self):
        result = kolmogorov_complexity(b"x")
        assert result == 1.0  # Can't compress single byte

    def test_typical_secret_high_complexity(self):
        data = b"ghp_1234567890abcdefABCDEF"
        result = kolmogorov_complexity(data)
        assert result > 0.7
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validator/test_features.py::TestKolmogorovComplexity -v`
Expected: FAIL with "cannot import name 'kolmogorov_complexity'"

**Step 3: Implement Huffman-based complexity**

```python
# src/heuristic_secrets/validator/features.py (append)
import heapq
from typing import Optional


def _huffman_encoded_length(data: bytes) -> int:
    """Calculate the length in bits if Huffman encoded."""
    if len(data) == 0:
        return 0
    
    counts = Counter(data)
    
    if len(counts) == 1:
        # Single unique byte - 1 bit per byte
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
    
    Args:
        data: Input bytes
        
    Returns:
        Complexity ratio (0.0 to 1.0+)
    """
    if len(data) == 0:
        return 0.0
    
    if len(data) == 1:
        return 1.0
    
    original_bits = len(data) * 8
    compressed_bits = _huffman_encoded_length(data)
    
    return compressed_bits / original_bits
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validator/test_features.py::TestKolmogorovComplexity -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/validator/features.py tests/validator/test_features.py
git commit -m "feat(validator): add Kolmogorov complexity via Huffman coding"
```

---

## Task 4: Validator Feature Extraction - Character Frequency Difference

**Files:**
- Modify: `src/heuristic_secrets/validator/features.py`
- Modify: `tests/validator/test_features.py`

**Step 1: Write the failing test**

```python
# tests/validator/test_features.py (append)
from heuristic_secrets.validator.features import (
    char_frequency_difference,
    FrequencyTable,
)


class TestCharFrequencyDifference:
    def test_identical_distribution_zero_diff(self):
        data = b"aabbcc"
        # Reference that matches data distribution
        reference = FrequencyTable({ord("a"): 1/3, ord("b"): 1/3, ord("c"): 1/3})
        result = char_frequency_difference(data, reference)
        assert abs(result) < 0.01

    def test_completely_different_distribution(self):
        data = b"aaaa"  # Only 'a'
        reference = FrequencyTable({ord("z"): 1.0})  # Only 'z' expected
        result = char_frequency_difference(data, reference)
        assert result > 1.5  # Large difference

    def test_empty_data_zero_diff(self):
        reference = FrequencyTable({ord("a"): 1.0})
        result = char_frequency_difference(b"", reference)
        assert result == 0.0

    def test_partial_overlap(self):
        data = b"aabb"
        reference = FrequencyTable({ord("a"): 0.5, ord("c"): 0.5})
        result = char_frequency_difference(data, reference)
        assert result > 0.5  # Moderate difference
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validator/test_features.py::TestCharFrequencyDifference -v`
Expected: FAIL with "cannot import name"

**Step 3: Implement**

```python
# src/heuristic_secrets/validator/features.py (append)
from dataclasses import dataclass, field


@dataclass
class FrequencyTable:
    """Character frequency distribution for comparison."""
    frequencies: dict[int, float] = field(default_factory=dict)
    
    def get(self, byte: int) -> float:
        """Get frequency for a byte, defaulting to 0."""
        return self.frequencies.get(byte, 0.0)
    
    @classmethod
    def from_data(cls, data: bytes) -> "FrequencyTable":
        """Build frequency table from sample data."""
        if len(data) == 0:
            return cls({})
        counts = Counter(data)
        total = len(data)
        return cls({byte: count / total for byte, count in counts.items()})


def char_frequency_difference(data: bytes, reference: FrequencyTable) -> float:
    """Calculate L2 distance between data's char frequency and reference.
    
    Args:
        data: Input bytes
        reference: Reference frequency distribution
        
    Returns:
        Euclidean distance between distributions
    """
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validator/test_features.py::TestCharFrequencyDifference -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/validator/features.py tests/validator/test_features.py
git commit -m "feat(validator): add character frequency difference metric"
```

---

## Task 5: Validator Feature Extraction - Character Type Mix

**Files:**
- Modify: `src/heuristic_secrets/validator/features.py`
- Modify: `tests/validator/test_features.py`

**Step 1: Write the failing test**

```python
# tests/validator/test_features.py (append)
from heuristic_secrets.validator.features import char_type_mix


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validator/test_features.py::TestCharTypeMix -v`
Expected: FAIL with "cannot import name 'char_type_mix'"

**Step 3: Implement**

```python
# src/heuristic_secrets/validator/features.py (append)

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validator/test_features.py::TestCharTypeMix -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/validator/features.py tests/validator/test_features.py
git commit -m "feat(validator): add character type mix metric"
```

---

## Task 6: Validator Feature Extraction - All 6 Features Combined

**Files:**
- Modify: `src/heuristic_secrets/validator/features.py`
- Modify: `tests/validator/test_features.py`

**Step 1: Write the failing test**

```python
# tests/validator/test_features.py (append)
from heuristic_secrets.validator.features import extract_features, FeatureSet


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
        
        # Verify types are floats
        assert all(isinstance(f, float) for f in features)

    def test_empty_data_returns_zeros(self):
        text_freq = FrequencyTable({})
        innocuous_freq = FrequencyTable({})
        secret_freq = FrequencyTable({})
        
        result = extract_features(b"", text_freq, innocuous_freq, secret_freq)
        features = result.to_list()
        
        assert features == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validator/test_features.py::TestExtractFeatures -v`
Expected: FAIL with "cannot import name 'extract_features'"

**Step 3: Implement**

```python
# src/heuristic_secrets/validator/features.py (append)
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
    """Extract all 6 features from input data.
    
    Args:
        data: Input bytes to analyze
        text_freq: Natural language character frequencies
        innocuous_freq: Typical source code character frequencies
        secret_freq: Known secret character frequencies
        
    Returns:
        FeatureSet with all 6 features
    """
    return FeatureSet(
        shannon_entropy=shannon_entropy(data),
        kolmogorov_complexity=kolmogorov_complexity(data),
        text_freq_diff=char_frequency_difference(data, text_freq),
        innocuous_freq_diff=char_frequency_difference(data, innocuous_freq),
        secret_freq_diff=char_frequency_difference(data, secret_freq),
        char_type_mix=char_type_mix(data),
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validator/test_features.py::TestExtractFeatures -v`
Expected: All PASS

**Step 5: Run all feature tests**

Run: `pytest tests/validator/test_features.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/heuristic_secrets/validator/features.py tests/validator/test_features.py
git commit -m "feat(validator): add combined feature extraction"
```

---

## Task 7: Validator Model - Configurable Architecture

**Files:**
- Create: `src/heuristic_secrets/validator/model.py`
- Create: `tests/validator/test_model.py`

**Step 1: Write the failing test**

```python
# tests/validator/test_model.py
import pytest
import torch
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig


class TestValidatorModel:
    def test_default_config(self):
        config = ValidatorConfig()
        assert config.input_dim == 6
        assert config.hidden_dims == [16, 8]

    def test_custom_config(self):
        config = ValidatorConfig(hidden_dims=[32, 16, 8])
        model = ValidatorModel(config)
        
        # Check layer structure
        assert len(model.layers) == 4  # 3 hidden + 1 output

    def test_forward_single_sample(self):
        model = ValidatorModel(ValidatorConfig())
        x = torch.randn(1, 6)  # batch=1, features=6
        
        output = model(x)
        
        assert output.shape == (1, 1)
        assert 0.0 <= output.item() <= 1.0  # Sigmoid output

    def test_forward_batch(self):
        model = ValidatorModel(ValidatorConfig())
        x = torch.randn(32, 6)  # batch=32
        
        output = model(x)
        
        assert output.shape == (32, 1)
        assert (output >= 0.0).all() and (output <= 1.0).all()

    def test_tiny_config(self):
        config = ValidatorConfig(hidden_dims=[8, 4])
        model = ValidatorModel(config)
        x = torch.randn(1, 6)
        
        output = model(x)
        assert output.shape == (1, 1)

    def test_large_config(self):
        config = ValidatorConfig(hidden_dims=[64, 32, 16, 8])
        model = ValidatorModel(config)
        x = torch.randn(1, 6)
        
        output = model(x)
        assert output.shape == (1, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validator/test_model.py -v`
Expected: FAIL with "cannot import name"

**Step 3: Implement**

```python
# src/heuristic_secrets/validator/model.py
from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass
class ValidatorConfig:
    """Configuration for the validator model architecture."""
    input_dim: int = 6
    hidden_dims: list[int] = field(default_factory=lambda: [16, 8])
    
    def to_dict(self) -> dict:
        return {"input_dim": self.input_dim, "hidden_dims": self.hidden_dims}
    
    @classmethod
    def from_dict(cls, d: dict) -> "ValidatorConfig":
        return cls(input_dim=d["input_dim"], hidden_dims=d["hidden_dims"])


class ValidatorModel(nn.Module):
    """Configurable neural network for secret validation.
    
    Takes 6 precomputed features as input, outputs probability [0, 1].
    """
    
    def __init__(self, config: ValidatorConfig):
        super().__init__()
        self.config = config
        
        layers = []
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 6)
            
        Returns:
            Output tensor of shape (batch, 1) with probabilities
        """
        return self.layers(x)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validator/test_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/validator/model.py tests/validator/test_model.py
git commit -m "feat(validator): add configurable validator model"
```

---

## Task 8: SpanFinder Model - Configurable Architecture

**Files:**
- Create: `src/heuristic_secrets/spanfinder/__init__.py`
- Create: `src/heuristic_secrets/spanfinder/model.py`
- Create: `tests/spanfinder/__init__.py`
- Create: `tests/spanfinder/test_model.py`

**Step 1: Write the failing test**

```python
# tests/spanfinder/test_model.py
import pytest
import torch
from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig


class TestSpanFinderModel:
    def test_default_config(self):
        config = SpanFinderConfig()
        assert config.vocab_size == 256
        assert config.embed_dim == 64
        assert config.hidden_dim == 64
        assert config.num_layers == 3

    def test_forward_single_sequence(self):
        config = SpanFinderConfig()
        model = SpanFinderModel(config)
        
        # Single sequence of 100 bytes
        x = torch.randint(0, 256, (1, 100))
        
        output = model(x)
        
        # Should output [start_prob, end_prob] per byte
        assert output.shape == (1, 100, 2)

    def test_forward_batch(self):
        config = SpanFinderConfig()
        model = SpanFinderModel(config)
        
        # Batch of 8 sequences, 512 bytes each
        x = torch.randint(0, 256, (8, 512))
        
        output = model(x)
        
        assert output.shape == (8, 512, 2)

    def test_tiny_config(self):
        config = SpanFinderConfig(embed_dim=32, hidden_dim=32, num_layers=2)
        model = SpanFinderModel(config)
        
        x = torch.randint(0, 256, (1, 100))
        output = model(x)
        
        assert output.shape == (1, 100, 2)

    def test_output_range(self):
        # Outputs should be valid probabilities after sigmoid
        config = SpanFinderConfig()
        model = SpanFinderModel(config)
        
        x = torch.randint(0, 256, (4, 50))
        output = model(x)
        
        # Raw output before threshold - check it's reasonable
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_medium_config(self):
        config = SpanFinderConfig(embed_dim=96, hidden_dim=96, num_layers=4)
        model = SpanFinderModel(config)
        
        x = torch.randint(0, 256, (1, 100))
        output = model(x)
        
        assert output.shape == (1, 100, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/spanfinder/test_model.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create directories and implement**

```bash
mkdir -p src/heuristic_secrets/spanfinder tests/spanfinder
touch src/heuristic_secrets/spanfinder/__init__.py
touch tests/spanfinder/__init__.py
```

```python
# src/heuristic_secrets/spanfinder/model.py
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class SpanFinderConfig:
    """Configuration for the SpanFinder model architecture."""
    vocab_size: int = 256  # Byte-level
    embed_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 3
    
    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SpanFinderConfig":
        return cls(**d)


class SpanFinderModel(nn.Module):
    """Byte-level span boundary prediction model.
    
    Takes a sequence of bytes, outputs [start_prob, end_prob] per byte.
    """
    
    def __init__(self, config: SpanFinderConfig):
        super().__init__()
        self.config = config
        
        # Byte embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Encoder layers
        encoder_layers = []
        prev_dim = config.embed_dim
        for _ in range(config.num_layers):
            encoder_layers.append(nn.Linear(prev_dim, config.hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = config.hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Output: 2 values per position (start_prob, end_prob)
        self.classifier = nn.Linear(config.hidden_dim, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len) with byte values 0-255
            
        Returns:
            Output tensor of shape (batch, seq_len, 2) with start/end logits
        """
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # (batch, seq_len, embed_dim) -> (batch, seq_len, hidden_dim)
        encoded = self.encoder(embedded)
        
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, 2)
        logits = self.classifier(encoded)
        
        return logits
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/spanfinder/test_model.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/spanfinder/ tests/spanfinder/
git commit -m "feat(spanfinder): add configurable SpanFinder model"
```

---

## Task 9: Pipeline - Chunker with Overlap

**Files:**
- Create: `src/heuristic_secrets/pipeline/__init__.py`
- Create: `src/heuristic_secrets/pipeline/chunker.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_chunker.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_chunker.py
import pytest
from heuristic_secrets.pipeline.chunker import Chunker, Chunk


class TestChunker:
    def test_small_input_single_chunk(self):
        chunker = Chunker(chunk_size=512, overlap=64)
        data = b"hello world"
        
        chunks = list(chunker.chunk(data))
        
        assert len(chunks) == 1
        assert chunks[0].data == data
        assert chunks[0].start == 0
        assert chunks[0].end == len(data)

    def test_exact_chunk_size(self):
        chunker = Chunker(chunk_size=10, overlap=2)
        data = b"0123456789"  # Exactly 10 bytes
        
        chunks = list(chunker.chunk(data))
        
        assert len(chunks) == 1
        assert chunks[0].data == data

    def test_two_chunks_with_overlap(self):
        chunker = Chunker(chunk_size=10, overlap=2)
        data = b"0123456789ABCDEF"  # 16 bytes
        
        chunks = list(chunker.chunk(data))
        
        assert len(chunks) == 2
        # First chunk: 0-10
        assert chunks[0].data == b"0123456789"
        assert chunks[0].start == 0
        # Second chunk: starts at 10-2=8 (overlap)
        assert chunks[1].start == 8
        assert chunks[1].data == data[8:]

    def test_many_chunks(self):
        chunker = Chunker(chunk_size=100, overlap=10)
        data = b"x" * 500
        
        chunks = list(chunker.chunk(data))
        
        # Should have multiple chunks with overlap
        assert len(chunks) >= 5
        
        # Verify coverage - every byte should be in at least one chunk
        covered = set()
        for chunk in chunks:
            for i in range(chunk.start, chunk.end):
                covered.add(i)
        assert covered == set(range(500))

    def test_empty_input(self):
        chunker = Chunker(chunk_size=512, overlap=64)
        
        chunks = list(chunker.chunk(b""))
        
        assert len(chunks) == 0

    def test_chunk_has_is_first_is_last(self):
        chunker = Chunker(chunk_size=10, overlap=2)
        data = b"0123456789ABCDEFGHIJ"  # 20 bytes
        
        chunks = list(chunker.chunk(data))
        
        assert chunks[0].is_first
        assert not chunks[0].is_last
        assert not chunks[-1].is_first
        assert chunks[-1].is_last
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_chunker.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create directories and implement**

```bash
mkdir -p src/heuristic_secrets/pipeline tests/pipeline
touch src/heuristic_secrets/pipeline/__init__.py
touch tests/pipeline/__init__.py
```

```python
# src/heuristic_secrets/pipeline/chunker.py
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Chunk:
    """A chunk of data with position metadata."""
    data: bytes
    start: int  # Start position in original data
    end: int    # End position in original data
    is_first: bool
    is_last: bool


class Chunker:
    """Split data into overlapping chunks for processing."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap
    
    def chunk(self, data: bytes) -> Iterator[Chunk]:
        """Split data into overlapping chunks.
        
        Args:
            data: Input bytes
            
        Yields:
            Chunk objects with data and position info
        """
        if len(data) == 0:
            return
        
        if len(data) <= self.chunk_size:
            yield Chunk(
                data=data,
                start=0,
                end=len(data),
                is_first=True,
                is_last=True,
            )
            return
        
        pos = 0
        is_first = True
        
        while pos < len(data):
            end = min(pos + self.chunk_size, len(data))
            is_last = end >= len(data)
            
            yield Chunk(
                data=data[pos:end],
                start=pos,
                end=end,
                is_first=is_first,
                is_last=is_last,
            )
            
            if is_last:
                break
            
            pos += self.stride
            is_first = False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_chunker.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/pipeline/ tests/pipeline/
git commit -m "feat(pipeline): add chunker with configurable overlap"
```

---

## Task 10: Pipeline - Span Assembly

**Files:**
- Create: `src/heuristic_secrets/pipeline/assembler.py`
- Create: `tests/pipeline/test_assembler.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_assembler.py
import pytest
import torch
from heuristic_secrets.pipeline.assembler import SpanAssembler, Span


class TestSpanAssembler:
    def test_simple_span_detection(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)
        
        # Simulate predictions: positions 2-5 are a span
        # Shape: (seq_len, 2) for [start_prob, end_prob]
        predictions = torch.zeros(10, 2)
        predictions[2, 0] = 0.9  # Start at 2
        predictions[5, 1] = 0.9  # End at 5
        
        spans = assembler.extract_spans(predictions, chunk_start=0)
        
        assert len(spans) == 1
        assert spans[0].start == 2
        assert spans[0].end == 6  # End is exclusive

    def test_multiple_spans(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)
        
        predictions = torch.zeros(20, 2)
        predictions[2, 0] = 0.9   # Start span 1
        predictions[4, 1] = 0.9   # End span 1
        predictions[10, 0] = 0.9  # Start span 2
        predictions[15, 1] = 0.9  # End span 2
        
        spans = assembler.extract_spans(predictions, chunk_start=0)
        
        assert len(spans) == 2
        assert spans[0].start == 2
        assert spans[0].end == 5
        assert spans[1].start == 10
        assert spans[1].end == 16

    def test_threshold_filtering(self):
        assembler = SpanAssembler(threshold=0.7, min_length=1, max_length=100)
        
        predictions = torch.zeros(10, 2)
        predictions[2, 0] = 0.8  # Above threshold
        predictions[5, 1] = 0.6  # Below threshold - should not be end
        predictions[7, 1] = 0.9  # Above threshold
        
        spans = assembler.extract_spans(predictions, chunk_start=0)
        
        assert len(spans) == 1
        assert spans[0].end == 8  # Uses position 7

    def test_min_length_filtering(self):
        assembler = SpanAssembler(threshold=0.5, min_length=5, max_length=100)
        
        predictions = torch.zeros(10, 2)
        predictions[2, 0] = 0.9
        predictions[3, 1] = 0.9  # Too short (length 2)
        predictions[5, 1] = 0.9  # Length 4, still too short
        predictions[8, 1] = 0.9  # Length 7, OK
        
        spans = assembler.extract_spans(predictions, chunk_start=0)
        
        assert len(spans) == 1
        assert spans[0].end - spans[0].start >= 5

    def test_max_length_filtering(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=5)
        
        predictions = torch.zeros(20, 2)
        predictions[2, 0] = 0.9
        predictions[15, 1] = 0.9  # Too long (length 14)
        predictions[5, 1] = 0.9   # Length 4, OK
        
        spans = assembler.extract_spans(predictions, chunk_start=0)
        
        assert len(spans) == 1
        assert spans[0].end - spans[0].start <= 5

    def test_chunk_offset_applied(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)
        
        predictions = torch.zeros(10, 2)
        predictions[2, 0] = 0.9
        predictions[5, 1] = 0.9
        
        # Chunk starts at position 100 in original document
        spans = assembler.extract_spans(predictions, chunk_start=100)
        
        assert spans[0].start == 102
        assert spans[0].end == 106

    def test_partial_span_start_only(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)
        
        predictions = torch.zeros(10, 2)
        predictions[7, 0] = 0.9  # Start near end, no matching end
        
        spans = assembler.extract_spans(predictions, chunk_start=0, is_last_chunk=False)
        
        # Should return partial span for cross-chunk stitching
        assert len(spans) == 1
        assert spans[0].end is None  # Partial - no end found

    def test_partial_span_end_only(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)
        
        predictions = torch.zeros(10, 2)
        predictions[2, 1] = 0.9  # End near start, no matching start
        
        spans = assembler.extract_spans(predictions, chunk_start=0, is_first_chunk=False)
        
        # Should return partial span for cross-chunk stitching
        assert len(spans) == 1
        assert spans[0].start is None  # Partial - no start found

    def test_no_spans(self):
        assembler = SpanAssembler(threshold=0.5, min_length=1, max_length=100)
        
        predictions = torch.zeros(10, 2)  # All below threshold
        
        spans = assembler.extract_spans(predictions, chunk_start=0)
        
        assert len(spans) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_assembler.py -v`
Expected: FAIL with "cannot import name"

**Step 3: Implement**

```python
# src/heuristic_secrets/pipeline/assembler.py
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Span:
    """A detected span with position info."""
    start: Optional[int]  # None if partial (continues from previous chunk)
    end: Optional[int]    # None if partial (continues to next chunk)
    
    @property
    def is_partial(self) -> bool:
        return self.start is None or self.end is None
    
    @property
    def length(self) -> Optional[int]:
        if self.start is None or self.end is None:
            return None
        return self.end - self.start


class SpanAssembler:
    """Extract spans from SpanFinder predictions."""
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_length: int = 8,
        max_length: int = 256,
    ):
        self.threshold = threshold
        self.min_length = min_length
        self.max_length = max_length
    
    def extract_spans(
        self,
        predictions: torch.Tensor,
        chunk_start: int = 0,
        is_first_chunk: bool = True,
        is_last_chunk: bool = True,
    ) -> list[Span]:
        """Extract spans from model predictions.
        
        Args:
            predictions: Tensor of shape (seq_len, 2) with [start_prob, end_prob]
            chunk_start: Offset of this chunk in the original document
            is_first_chunk: Whether this is the first chunk
            is_last_chunk: Whether this is the last chunk
            
        Returns:
            List of Span objects
        """
        # Apply sigmoid if not already done
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Find positions above threshold
        start_positions = (predictions[:, 0] >= self.threshold).nonzero(as_tuple=True)[0].tolist()
        end_positions = (predictions[:, 1] >= self.threshold).nonzero(as_tuple=True)[0].tolist()
        
        spans = []
        
        # Handle partial spans (end without start at chunk beginning)
        if not is_first_chunk and end_positions:
            first_end = end_positions[0]
            if not start_positions or start_positions[0] > first_end:
                spans.append(Span(start=None, end=chunk_start + first_end + 1))
        
        # Pair starts with ends (Cartesian product with constraints)
        for start_pos in start_positions:
            best_end = None
            for end_pos in end_positions:
                if end_pos < start_pos:
                    continue
                length = end_pos - start_pos + 1
                if length < self.min_length:
                    continue
                if length > self.max_length:
                    continue
                # Take first valid end
                best_end = end_pos
                break
            
            if best_end is not None:
                spans.append(Span(
                    start=chunk_start + start_pos,
                    end=chunk_start + best_end + 1,
                ))
            elif not is_last_chunk:
                # Partial span - start without end
                spans.append(Span(start=chunk_start + start_pos, end=None))
        
        return spans
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_assembler.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/pipeline/assembler.py tests/pipeline/test_assembler.py
git commit -m "feat(pipeline): add span assembly from predictions"
```

---

## Task 11: Pipeline - End-to-End Detector

**Files:**
- Create: `src/heuristic_secrets/pipeline/detector.py`
- Create: `tests/pipeline/test_detector.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_detector.py
import pytest
import torch
from heuristic_secrets.pipeline.detector import Detector, Detection, DetectorConfig
from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig
from heuristic_secrets.validator.features import FrequencyTable


class TestDetector:
    @pytest.fixture
    def detector(self):
        """Create a detector with small models for testing."""
        spanfinder = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        validator = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))
        
        config = DetectorConfig(
            chunk_size=100,
            chunk_overlap=10,
            spanfinder_threshold=0.5,
            validator_threshold=0.5,
        )
        
        return Detector(
            spanfinder=spanfinder,
            validator=validator,
            config=config,
            text_freq=FrequencyTable({}),
            innocuous_freq=FrequencyTable({}),
            secret_freq=FrequencyTable({}),
        )

    def test_scan_returns_detections(self, detector):
        text = "config = { api_key: 'sk-abc123xyz789' }"
        
        detections = detector.scan(text)
        
        assert isinstance(detections, list)
        for d in detections:
            assert isinstance(d, Detection)
            assert hasattr(d, 'start')
            assert hasattr(d, 'end')
            assert hasattr(d, 'text')
            assert hasattr(d, 'probability')

    def test_scan_empty_string(self, detector):
        detections = detector.scan("")
        assert detections == []

    def test_scan_batch(self, detector):
        texts = [
            "api_key = 'sk-abc123'",
            "password = 'hunter2'",
            "normal text here",
        ]
        
        results = detector.scan_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_detection_positions_are_valid(self, detector):
        text = "x" * 200  # Longer than chunk size
        
        detections = detector.scan(text)
        
        for d in detections:
            assert 0 <= d.start < len(text)
            assert d.start < d.end <= len(text)
            assert d.text == text[d.start:d.end]

    def test_probability_in_range(self, detector):
        text = "secret_key = 'ghp_xxxxxxxxxxxxxxxxxxxx'"
        
        detections = detector.scan(text)
        
        for d in detections:
            assert 0.0 <= d.probability <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_detector.py -v`
Expected: FAIL with "cannot import name"

**Step 3: Implement**

```python
# src/heuristic_secrets/pipeline/detector.py
from dataclasses import dataclass
from typing import Optional
import torch

from heuristic_secrets.spanfinder.model import SpanFinderModel
from heuristic_secrets.validator.model import ValidatorModel
from heuristic_secrets.validator.features import (
    FrequencyTable,
    extract_features,
)
from heuristic_secrets.pipeline.chunker import Chunker
from heuristic_secrets.pipeline.assembler import SpanAssembler


@dataclass
class Detection:
    """A detected secret with metadata."""
    start: int
    end: int
    text: str
    probability: float


@dataclass
class DetectorConfig:
    """Configuration for the detection pipeline."""
    chunk_size: int = 512
    chunk_overlap: int = 64
    spanfinder_threshold: float = 0.5
    validator_threshold: float = 0.5
    min_span_length: int = 8
    max_span_length: int = 256


class Detector:
    """End-to-end secret detection pipeline."""
    
    def __init__(
        self,
        spanfinder: SpanFinderModel,
        validator: ValidatorModel,
        config: DetectorConfig,
        text_freq: FrequencyTable,
        innocuous_freq: FrequencyTable,
        secret_freq: FrequencyTable,
    ):
        self.spanfinder = spanfinder
        self.validator = validator
        self.config = config
        self.text_freq = text_freq
        self.innocuous_freq = innocuous_freq
        self.secret_freq = secret_freq
        
        self.chunker = Chunker(config.chunk_size, config.chunk_overlap)
        self.assembler = SpanAssembler(
            threshold=config.spanfinder_threshold,
            min_length=config.min_span_length,
            max_length=config.max_span_length,
        )
        
        # Set models to eval mode
        self.spanfinder.eval()
        self.validator.eval()
    
    def scan(self, text: str) -> list[Detection]:
        """Scan text for secrets.
        
        Args:
            text: Input text to scan
            
        Returns:
            List of Detection objects
        """
        if not text:
            return []
        
        data = text.encode('utf-8')
        
        # Stage 1: Find candidate spans
        candidate_spans = self._find_spans(data)
        
        # Stage 2: Validate each candidate
        detections = []
        for span in candidate_spans:
            if span.is_partial:
                continue  # Skip partial spans for now
            
            span_data = data[span.start:span.end]
            probability = self._validate_span(span_data)
            
            if probability >= self.config.validator_threshold:
                detections.append(Detection(
                    start=span.start,
                    end=span.end,
                    text=span_data.decode('utf-8', errors='replace'),
                    probability=probability,
                ))
        
        return detections
    
    def scan_batch(self, texts: list[str]) -> list[list[Detection]]:
        """Scan multiple texts for secrets.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of detection lists, one per input text
        """
        return [self.scan(text) for text in texts]
    
    def _find_spans(self, data: bytes) -> list:
        """Run SpanFinder to find candidate spans."""
        all_spans = []
        
        for chunk in self.chunker.chunk(data):
            # Convert bytes to tensor
            byte_values = list(chunk.data)
            x = torch.tensor([byte_values], dtype=torch.long)
            
            # Run SpanFinder
            with torch.no_grad():
                predictions = self.spanfinder(x)
                predictions = torch.sigmoid(predictions[0])  # Remove batch dim
            
            # Extract spans from predictions
            spans = self.assembler.extract_spans(
                predictions,
                chunk_start=chunk.start,
                is_first_chunk=chunk.is_first,
                is_last_chunk=chunk.is_last,
            )
            all_spans.extend(spans)
        
        # TODO: Stitch partial spans across chunks
        return all_spans
    
    def _validate_span(self, data: bytes) -> float:
        """Run validator on a candidate span."""
        features = extract_features(
            data,
            self.text_freq,
            self.innocuous_freq,
            self.secret_freq,
        )
        
        x = torch.tensor([features.to_list()], dtype=torch.float32)
        
        with torch.no_grad():
            probability = self.validator(x)
        
        return probability.item()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_detector.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/pipeline/detector.py tests/pipeline/test_detector.py
git commit -m "feat(pipeline): add end-to-end detector"
```

---

## Task 12: Model Save/Load with Metadata

**Files:**
- Create: `src/heuristic_secrets/io.py`
- Create: `tests/test_io.py`

**Step 1: Write the failing test**

```python
# tests/test_io.py
import pytest
import tempfile
import os
from pathlib import Path

from heuristic_secrets.io import save_model_bundle, load_model_bundle
from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig
from heuristic_secrets.validator.features import FrequencyTable
from heuristic_secrets.pipeline.detector import DetectorConfig


class TestModelIO:
    def test_save_and_load_bundle(self, tmp_path):
        # Create models
        sf_config = SpanFinderConfig(embed_dim=32, num_layers=2)
        spanfinder = SpanFinderModel(sf_config)
        
        val_config = ValidatorConfig(hidden_dims=[8, 4])
        validator = ValidatorModel(val_config)
        
        det_config = DetectorConfig(chunk_size=256, chunk_overlap=32)
        
        freqs = {
            "text": FrequencyTable({ord("e"): 0.12}),
            "innocuous": FrequencyTable({ord("a"): 0.05}),
            "secret": FrequencyTable({ord("0"): 0.08}),
        }
        
        bundle_path = tmp_path / "test-model-v1"
        
        # Save
        save_model_bundle(
            path=bundle_path,
            spanfinder=spanfinder,
            validator=validator,
            detector_config=det_config,
            frequencies=freqs,
        )
        
        # Verify files exist
        assert (bundle_path / "config.json").exists()
        assert (bundle_path / "spanfinder.safetensors").exists()
        assert (bundle_path / "spanfinder.meta.json").exists()
        assert (bundle_path / "validator.safetensors").exists()
        assert (bundle_path / "validator.meta.json").exists()
        
        # Load
        loaded = load_model_bundle(bundle_path)
        
        assert loaded.spanfinder is not None
        assert loaded.validator is not None
        assert loaded.config.chunk_size == 256
        assert loaded.frequencies["text"].get(ord("e")) == 0.12

    def test_load_preserves_architecture(self, tmp_path):
        sf_config = SpanFinderConfig(embed_dim=64, hidden_dim=64, num_layers=3)
        spanfinder = SpanFinderModel(sf_config)
        
        val_config = ValidatorConfig(hidden_dims=[32, 16, 8])
        validator = ValidatorModel(val_config)
        
        bundle_path = tmp_path / "arch-test"
        
        save_model_bundle(
            path=bundle_path,
            spanfinder=spanfinder,
            validator=validator,
            detector_config=DetectorConfig(),
            frequencies={
                "text": FrequencyTable({}),
                "innocuous": FrequencyTable({}),
                "secret": FrequencyTable({}),
            },
        )
        
        loaded = load_model_bundle(bundle_path)
        
        # Verify architectures match
        assert loaded.spanfinder.config.embed_dim == 64
        assert loaded.spanfinder.config.num_layers == 3
        assert loaded.validator.config.hidden_dims == [32, 16, 8]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io.py -v`
Expected: FAIL with "cannot import name"

**Step 3: Implement**

```python
# src/heuristic_secrets/io.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file, load_file

from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig
from heuristic_secrets.validator.features import FrequencyTable
from heuristic_secrets.pipeline.detector import DetectorConfig


@dataclass
class ModelBundle:
    """Loaded model bundle with all components."""
    spanfinder: SpanFinderModel
    validator: ValidatorModel
    config: DetectorConfig
    frequencies: dict[str, FrequencyTable]


def save_model_bundle(
    path: Path,
    spanfinder: SpanFinderModel,
    validator: ValidatorModel,
    detector_config: DetectorConfig,
    frequencies: dict[str, FrequencyTable],
    version: str = "1.0.0",
) -> None:
    """Save a complete model bundle to disk.
    
    Args:
        path: Directory to save the bundle
        spanfinder: SpanFinder model
        validator: Validator model
        detector_config: Detection pipeline configuration
        frequencies: Character frequency tables
        version: Bundle version string
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save config.json
    config_data = {
        "version": version,
        "chunk_size": detector_config.chunk_size,
        "chunk_overlap": detector_config.chunk_overlap,
        "spanfinder_threshold": detector_config.spanfinder_threshold,
        "validator_threshold": detector_config.validator_threshold,
        "min_span_length": detector_config.min_span_length,
        "max_span_length": detector_config.max_span_length,
    }
    with open(path / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Save SpanFinder
    sf_state = {k: v for k, v in spanfinder.state_dict().items()}
    save_file(sf_state, path / "spanfinder.safetensors")
    
    sf_meta = spanfinder.config.to_dict()
    with open(path / "spanfinder.meta.json", "w") as f:
        json.dump(sf_meta, f, indent=2)
    
    # Save Validator
    val_state = {k: v for k, v in validator.state_dict().items()}
    save_file(val_state, path / "validator.safetensors")
    
    val_meta = {
        **validator.config.to_dict(),
        "frequencies": {
            name: table.frequencies
            for name, table in frequencies.items()
        },
    }
    with open(path / "validator.meta.json", "w") as f:
        json.dump(val_meta, f, indent=2)


def load_model_bundle(path: Path) -> ModelBundle:
    """Load a complete model bundle from disk.
    
    Args:
        path: Directory containing the bundle
        
    Returns:
        ModelBundle with all components loaded
    """
    path = Path(path)
    
    # Load config.json
    with open(path / "config.json") as f:
        config_data = json.load(f)
    
    detector_config = DetectorConfig(
        chunk_size=config_data["chunk_size"],
        chunk_overlap=config_data["chunk_overlap"],
        spanfinder_threshold=config_data["spanfinder_threshold"],
        validator_threshold=config_data["validator_threshold"],
        min_span_length=config_data["min_span_length"],
        max_span_length=config_data["max_span_length"],
    )
    
    # Load SpanFinder
    with open(path / "spanfinder.meta.json") as f:
        sf_meta = json.load(f)
    sf_config = SpanFinderConfig.from_dict(sf_meta)
    spanfinder = SpanFinderModel(sf_config)
    sf_state = load_file(path / "spanfinder.safetensors")
    spanfinder.load_state_dict(sf_state)
    
    # Load Validator
    with open(path / "validator.meta.json") as f:
        val_meta = json.load(f)
    
    frequencies = {
        name: FrequencyTable(freqs)
        for name, freqs in val_meta.pop("frequencies").items()
    }
    
    val_config = ValidatorConfig.from_dict(val_meta)
    validator = ValidatorModel(val_config)
    val_state = load_file(path / "validator.safetensors")
    validator.load_state_dict(val_state)
    
    return ModelBundle(
        spanfinder=spanfinder,
        validator=validator,
        config=detector_config,
        frequencies=frequencies,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_io.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/io.py tests/test_io.py
git commit -m "feat: add model bundle save/load with safetensors"
```

---

## Task 13: Public API - Detector.load()

**Files:**
- Modify: `src/heuristic_secrets/pipeline/detector.py`
- Modify: `tests/pipeline/test_detector.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_detector.py (append)
import tempfile
from heuristic_secrets.io import save_model_bundle


class TestDetectorLoad:
    def test_load_from_bundle(self, tmp_path):
        # Create and save a bundle
        from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
        from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig
        from heuristic_secrets.validator.features import FrequencyTable
        from heuristic_secrets.pipeline.detector import DetectorConfig
        
        sf = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        val = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))
        
        bundle_path = tmp_path / "test-bundle"
        save_model_bundle(
            path=bundle_path,
            spanfinder=sf,
            validator=val,
            detector_config=DetectorConfig(),
            frequencies={
                "text": FrequencyTable({}),
                "innocuous": FrequencyTable({}),
                "secret": FrequencyTable({}),
            },
        )
        
        # Load using Detector.load()
        detector = Detector.load(bundle_path)
        
        assert detector is not None
        assert detector.scan("test") is not None

    def test_load_with_config_override(self, tmp_path):
        from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
        from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig
        from heuristic_secrets.validator.features import FrequencyTable
        from heuristic_secrets.pipeline.detector import DetectorConfig
        
        sf = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        val = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))
        
        bundle_path = tmp_path / "test-bundle-2"
        save_model_bundle(
            path=bundle_path,
            spanfinder=sf,
            validator=val,
            detector_config=DetectorConfig(validator_threshold=0.5),
            frequencies={
                "text": FrequencyTable({}),
                "innocuous": FrequencyTable({}),
                "secret": FrequencyTable({}),
            },
        )
        
        # Load with custom threshold
        custom_config = DetectorConfig(validator_threshold=0.9)
        detector = Detector.load(bundle_path, config=custom_config)
        
        assert detector.config.validator_threshold == 0.9
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_detector.py::TestDetectorLoad -v`
Expected: FAIL with "AttributeError: type object 'Detector' has no attribute 'load'"

**Step 3: Add class method**

```python
# src/heuristic_secrets/pipeline/detector.py (add to Detector class)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: Optional[DetectorConfig] = None,
    ) -> "Detector":
        """Load a detector from a model bundle.
        
        Args:
            path: Path to model bundle directory
            config: Optional config override
            
        Returns:
            Configured Detector instance
        """
        from heuristic_secrets.io import load_model_bundle
        
        bundle = load_model_bundle(Path(path))
        
        return cls(
            spanfinder=bundle.spanfinder,
            validator=bundle.validator,
            config=config or bundle.config,
            text_freq=bundle.frequencies["text"],
            innocuous_freq=bundle.frequencies["innocuous"],
            secret_freq=bundle.frequencies["secret"],
        )
```

Also add import at top:
```python
from pathlib import Path
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/pipeline/test_detector.py::TestDetectorLoad -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/pipeline/detector.py tests/pipeline/test_detector.py
git commit -m "feat(detector): add Detector.load() class method"
```

---

## Task 14: Export Public API from Package

**Files:**
- Modify: `src/heuristic_secrets/__init__.py`
- Create: `tests/test_public_api.py`

**Step 1: Write the failing test**

```python
# tests/test_public_api.py
import pytest


class TestPublicAPI:
    def test_can_import_detector(self):
        from heuristic_secrets import Detector
        assert Detector is not None

    def test_can_import_detection(self):
        from heuristic_secrets import Detection
        assert Detection is not None

    def test_can_import_config(self):
        from heuristic_secrets import DetectorConfig
        assert DetectorConfig is not None

    def test_can_import_spanfinder_config(self):
        from heuristic_secrets import SpanFinderConfig
        assert SpanFinderConfig is not None

    def test_can_import_validator_config(self):
        from heuristic_secrets import ValidatorConfig
        assert ValidatorConfig is not None

    def test_version_exists(self):
        import heuristic_secrets
        assert hasattr(heuristic_secrets, "__version__")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_public_api.py -v`
Expected: FAIL with "cannot import name"

**Step 3: Implement**

```python
# src/heuristic_secrets/__init__.py
"""Heuristic Secrets - High-performance secret detection using ML."""

__version__ = "0.1.0"

from heuristic_secrets.pipeline.detector import Detector, Detection, DetectorConfig
from heuristic_secrets.spanfinder.model import SpanFinderConfig
from heuristic_secrets.validator.model import ValidatorConfig

__all__ = [
    "Detector",
    "Detection",
    "DetectorConfig",
    "SpanFinderConfig",
    "ValidatorConfig",
    "__version__",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_public_api.py -v`
Expected: All PASS

**Step 5: Run all tests**

Run: `pytest -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/heuristic_secrets/__init__.py tests/test_public_api.py
git commit -m "feat: export public API from package root"
```

---

## Task 15: Data Collection - Scaffold

**Files:**
- Create: `src/heuristic_secrets/data/__init__.py`
- Create: `src/heuristic_secrets/data/collect.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/test_collect.py`

**Step 1: Write the failing test**

```python
# tests/data/test_collect.py
import pytest
from heuristic_secrets.data.collect import SecretPattern, load_gitleaks_patterns


class TestLoadPatterns:
    def test_load_gitleaks_returns_patterns(self):
        # This will initially fail until we have real data
        # For now, test the structure
        patterns = load_gitleaks_patterns()
        
        assert isinstance(patterns, list)
        if patterns:  # May be empty if no data yet
            assert isinstance(patterns[0], SecretPattern)
            assert hasattr(patterns[0], 'name')
            assert hasattr(patterns[0], 'regex')
            assert hasattr(patterns[0], 'examples')
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_collect.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create directories and implement scaffold**

```bash
mkdir -p src/heuristic_secrets/data tests/data
touch src/heuristic_secrets/data/__init__.py
touch tests/data/__init__.py
```

```python
# src/heuristic_secrets/data/collect.py
"""Data collection utilities for training data."""

from dataclasses import dataclass, field


@dataclass
class SecretPattern:
    """A secret pattern with examples."""
    name: str
    regex: str
    examples: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)


def load_gitleaks_patterns() -> list[SecretPattern]:
    """Load patterns from Gitleaks test fixtures.
    
    Returns:
        List of SecretPattern objects
    """
    # TODO: Implement actual loading from gitleaks repo
    # For now, return empty list
    return []


def load_trufflehog_patterns() -> list[SecretPattern]:
    """Load patterns from TruffleHog test fixtures.
    
    Returns:
        List of SecretPattern objects
    """
    # TODO: Implement actual loading
    return []


def load_detect_secrets_patterns() -> list[SecretPattern]:
    """Load patterns from detect-secrets test fixtures.
    
    Returns:
        List of SecretPattern objects
    """
    # TODO: Implement actual loading
    return []
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_collect.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/data/ tests/data/
git commit -m "feat(data): add data collection scaffold"
```

---

## Task 16: Training Loop Scaffold - Validator

**Files:**
- Create: `src/heuristic_secrets/validator/train.py`
- Create: `tests/validator/test_train.py`

**Step 1: Write the failing test**

```python
# tests/validator/test_train.py
import pytest
import torch
from heuristic_secrets.validator.train import (
    ValidatorTrainer,
    TrainingConfig,
    TrainingResult,
)
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig


class TestValidatorTrainer:
    def test_train_single_epoch(self):
        model = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))
        
        # Dummy training data
        X = torch.randn(100, 6)
        y = torch.randint(0, 2, (100, 1)).float()
        
        trainer = ValidatorTrainer(
            model=model,
            config=TrainingConfig(epochs=1, batch_size=32, lr=0.01),
        )
        
        result = trainer.train(X, y)
        
        assert isinstance(result, TrainingResult)
        assert result.final_loss is not None
        assert len(result.loss_history) == 1

    def test_train_multiple_epochs(self):
        model = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))
        
        X = torch.randn(100, 6)
        y = torch.randint(0, 2, (100, 1)).float()
        
        trainer = ValidatorTrainer(
            model=model,
            config=TrainingConfig(epochs=5, batch_size=32, lr=0.01),
        )
        
        result = trainer.train(X, y)
        
        assert len(result.loss_history) == 5

    def test_loss_decreases(self):
        model = ValidatorModel(ValidatorConfig(hidden_dims=[16, 8]))
        
        # Create separable data
        X_pos = torch.randn(50, 6) + 2  # Positive class
        X_neg = torch.randn(50, 6) - 2  # Negative class
        X = torch.cat([X_pos, X_neg])
        y = torch.cat([torch.ones(50, 1), torch.zeros(50, 1)])
        
        trainer = ValidatorTrainer(
            model=model,
            config=TrainingConfig(epochs=20, batch_size=32, lr=0.1),
        )
        
        result = trainer.train(X, y)
        
        # Loss should generally decrease
        assert result.loss_history[-1] < result.loss_history[0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validator/test_train.py -v`
Expected: FAIL with "cannot import name"

**Step 3: Implement**

```python
# src/heuristic_secrets/validator/train.py
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from heuristic_secrets.validator.model import ValidatorModel


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 50
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.01


@dataclass
class TrainingResult:
    """Results from training."""
    final_loss: float
    loss_history: list[float] = field(default_factory=list)


class ValidatorTrainer:
    """Training loop for the validator model."""
    
    def __init__(self, model: ValidatorModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    
    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> TrainingResult:
        """Train the model.
        
        Args:
            X: Training features (N, 6)
            y: Training labels (N, 1)
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            TrainingResult with loss history
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        loss_history = []
        
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
        
        return TrainingResult(
            final_loss=loss_history[-1],
            loss_history=loss_history,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validator/test_train.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/validator/train.py tests/validator/test_train.py
git commit -m "feat(validator): add training loop"
```

---

## Task 17: Training Loop Scaffold - SpanFinder

**Files:**
- Create: `src/heuristic_secrets/spanfinder/train.py`
- Create: `tests/spanfinder/test_train.py`

**Step 1: Write the failing test**

```python
# tests/spanfinder/test_train.py
import pytest
import torch
from heuristic_secrets.spanfinder.train import (
    SpanFinderTrainer,
    TrainingConfig,
    TrainingResult,
)
from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig


class TestSpanFinderTrainer:
    def test_train_single_epoch(self):
        model = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        
        # Dummy training data: sequences of bytes and boundary labels
        # X: (batch, seq_len) byte values
        # y: (batch, seq_len, 2) boundary labels [start, end]
        X = torch.randint(0, 256, (10, 100))
        y = torch.zeros(10, 100, 2)
        # Mark some boundaries
        y[0, 20, 0] = 1.0  # Start at position 20
        y[0, 30, 1] = 1.0  # End at position 30
        
        trainer = SpanFinderTrainer(
            model=model,
            config=TrainingConfig(epochs=1, batch_size=4, lr=0.01),
        )
        
        result = trainer.train(X, y)
        
        assert isinstance(result, TrainingResult)
        assert result.final_loss is not None
        assert len(result.loss_history) == 1

    def test_loss_decreases(self):
        model = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        
        # Create training data with consistent patterns
        X = torch.randint(0, 256, (20, 50))
        y = torch.zeros(20, 50, 2)
        # Always mark position 10 as start, 20 as end
        y[:, 10, 0] = 1.0
        y[:, 20, 1] = 1.0
        
        trainer = SpanFinderTrainer(
            model=model,
            config=TrainingConfig(epochs=20, batch_size=4, lr=0.01),
        )
        
        result = trainer.train(X, y)
        
        # Loss should generally decrease
        assert result.loss_history[-1] < result.loss_history[0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/spanfinder/test_train.py -v`
Expected: FAIL with "cannot import name"

**Step 3: Implement**

```python
# src/heuristic_secrets/spanfinder/train.py
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from heuristic_secrets.spanfinder.model import SpanFinderModel


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.01


@dataclass
class TrainingResult:
    """Results from training."""
    final_loss: float
    loss_history: list[float] = field(default_factory=list)


class SpanFinderTrainer:
    """Training loop for the SpanFinder model."""
    
    def __init__(self, model: SpanFinderModel, config: TrainingConfig):
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()  # As per spaCy's SpanFinder
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    
    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> TrainingResult:
        """Train the model.
        
        Args:
            X: Training sequences (N, seq_len) with byte values
            y: Training labels (N, seq_len, 2) with [start, end] boundaries
            X_val: Optional validation sequences
            y_val: Optional validation labels
            
        Returns:
            TrainingResult with loss history
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        loss_history = []
        
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)  # (batch, seq_len, 2)
                outputs = torch.sigmoid(outputs)  # Convert to probabilities
                
                # Compute loss
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
        
        return TrainingResult(
            final_loss=loss_history[-1],
            loss_history=loss_history,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/spanfinder/test_train.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/heuristic_secrets/spanfinder/train.py tests/spanfinder/test_train.py
git commit -m "feat(spanfinder): add training loop"
```

---

## Task 18: Final Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_integration.py
"""End-to-end integration tests."""

import pytest
import tempfile
from pathlib import Path
import torch

from heuristic_secrets import (
    Detector,
    Detection,
    DetectorConfig,
    SpanFinderConfig,
    ValidatorConfig,
)
from heuristic_secrets.spanfinder.model import SpanFinderModel
from heuristic_secrets.validator.model import ValidatorModel
from heuristic_secrets.validator.features import FrequencyTable
from heuristic_secrets.io import save_model_bundle


class TestEndToEndPipeline:
    @pytest.fixture
    def model_bundle(self, tmp_path):
        """Create a model bundle for testing."""
        sf = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        val = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))
        
        bundle_path = tmp_path / "integration-test-model"
        save_model_bundle(
            path=bundle_path,
            spanfinder=sf,
            validator=val,
            detector_config=DetectorConfig(
                chunk_size=256,
                chunk_overlap=32,
                spanfinder_threshold=0.3,
                validator_threshold=0.3,
            ),
            frequencies={
                "text": FrequencyTable({}),
                "innocuous": FrequencyTable({}),
                "secret": FrequencyTable({}),
            },
        )
        return bundle_path

    def test_full_pipeline_runs(self, model_bundle):
        """Test that the full pipeline runs without errors."""
        detector = Detector.load(model_bundle)
        
        text = """
        const config = {
            apiKey: 'sk-1234567890abcdef',
            databaseUrl: 'postgres://user:password@localhost/db',
            debug: true
        };
        """
        
        detections = detector.scan(text)
        
        # With random weights, we may or may not get detections
        # The important thing is it runs without errors
        assert isinstance(detections, list)
        for d in detections:
            assert isinstance(d, Detection)
            assert 0 <= d.start < len(text)
            assert d.start < d.end <= len(text)
            assert 0.0 <= d.probability <= 1.0

    def test_batch_processing(self, model_bundle):
        """Test batch processing works."""
        detector = Detector.load(model_bundle)
        
        texts = [
            "api_key = 'secret123'",
            "normal text without secrets",
            "password: hunter2",
        ]
        
        results = detector.scan_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_empty_and_whitespace(self, model_bundle):
        """Test edge cases."""
        detector = Detector.load(model_bundle)
        
        assert detector.scan("") == []
        assert detector.scan("   ") is not None  # Should not crash
        assert detector.scan("\n\n\n") is not None

    def test_large_input(self, model_bundle):
        """Test handling of large inputs."""
        detector = Detector.load(model_bundle)
        
        # 10KB of text
        large_text = "x" * 10000 + " api_key='secret' " + "y" * 10000
        
        detections = detector.scan(large_text)
        
        assert isinstance(detections, list)
        # Positions should be valid
        for d in detections:
            assert 0 <= d.start < len(large_text)
            assert d.start < d.end <= len(large_text)


class TestModelSaveLoadRoundtrip:
    def test_weights_preserved(self, tmp_path):
        """Test that model weights are preserved after save/load."""
        # Create models with specific weights
        sf = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))
        val = ValidatorModel(ValidatorConfig(hidden_dims=[8, 4]))
        
        # Store original weights
        sf_weights_before = {k: v.clone() for k, v in sf.state_dict().items()}
        val_weights_before = {k: v.clone() for k, v in val.state_dict().items()}
        
        bundle_path = tmp_path / "roundtrip-test"
        save_model_bundle(
            path=bundle_path,
            spanfinder=sf,
            validator=val,
            detector_config=DetectorConfig(),
            frequencies={
                "text": FrequencyTable({65: 0.1, 66: 0.2}),
                "innocuous": FrequencyTable({67: 0.3}),
                "secret": FrequencyTable({68: 0.4}),
            },
        )
        
        # Load and compare
        detector = Detector.load(bundle_path)
        
        for name, weight in sf_weights_before.items():
            loaded_weight = detector.spanfinder.state_dict()[name]
            assert torch.allclose(weight, loaded_weight), f"SpanFinder {name} mismatch"
        
        for name, weight in val_weights_before.items():
            loaded_weight = detector.validator.state_dict()[name]
            assert torch.allclose(weight, loaded_weight), f"Validator {name} mismatch"
```

**Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `pytest -v --tb=short`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests"
```

---

## Summary

This plan creates the complete Phase 1 prototype with:

1. **Validator module** (Tasks 2-7): Feature extraction and configurable classifier
2. **SpanFinder module** (Task 8): Byte-level boundary prediction model
3. **Pipeline module** (Tasks 9-11): Chunking, span assembly, end-to-end detector
4. **IO module** (Tasks 12-13): SafeTensors save/load with metadata
5. **Public API** (Task 14): Clean exports from package root
6. **Data collection scaffold** (Task 15): Structure for gathering training data
7. **Training loops** (Tasks 16-17): Training infrastructure for both models
8. **Integration tests** (Task 18): End-to-end validation

**Total: 18 tasks, ~50 commits**

Each task follows TDD: write failing test → implement → verify → commit.
