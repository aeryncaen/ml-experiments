# Heuristic Secrets: High-Performance Secret Detection

**Design Document**  
**Date:** 2026-01-17  
**Status:** Approved

---

## Overview & Goals

A two-stage ML pipeline for detecting secrets in arbitrary text, based on the research paper "Beyond RegEx – Heuristic-based Secret Detection" (Burdick-Pless, 2025).

### Goals

- Detect secrets without relying solely on regex patterns
- Handle pattern-less secrets (passwords, random tokens) that regex can't catch
- Hyper-fast inference: millions of strings per second
- CPU and GPU capable, highly parallelizable
- Configurable model architectures to find optimal size/accuracy tradeoff

### Non-Goals (v1)

- Real-time streaming (batch-oriented first)
- Secret type classification (binary: secret or not)
- Automatic remediation/revocation

### Target Users

- Security teams integrating into CI/CD pipelines
- IDE plugin developers
- Secret scanning service operators

---

## Architecture

### Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Input: Arbitrary text (code, config, logs, etc.)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Chunking Layer                                                         │
│  - Split into 512-byte chunks with 64-byte overlap (configurable)       │
│  - Track chunk boundaries for stitching                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: SpanFinder                                                    │
│  - Byte-level sequence model                                            │
│  - Predicts [start_prob, end_prob] per byte                             │
│  - Supports partial spans (start-only, end-only) at chunk boundaries    │
│  - Configurable: embed_dim, num_layers, hidden_dim                      │
│  - Goal: HIGH RECALL - find all candidates (secrets AND false positives)│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Span Assembly                                                          │
│  - Threshold start/end predictions                                      │
│  - Pair starts x ends within length constraints                         │
│  - Stitch partial spans across chunk boundaries                         │
│  - Output: candidate spans [(start, end, text), ...]                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Heuristic Validator                                           │
│  - Compute 6 features per candidate span:                               │
│    1. Shannon Entropy                                                   │
│    2. Kolmogorov Complexity (Huffman)                                   │
│    3. Text Character Frequency Difference                               │
│    4. Innocuous Code Frequency Difference                               │
│    5. Secret Frequency Difference                                       │
│    6. Character Type Mix                                                │
│  - Feed into configurable NN: input(6) -> hidden... -> output(1)        │
│  - Goal: HIGH PRECISION - filter false positives                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Output: Detected secrets with positions and confidence scores          │
│  [(start, end, text, probability), ...]                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Byte-level tokenization** - No tokenizer dependencies, UTF-8 native, precise boundaries
2. **Chunking with overlap** - Handles large files, configurable (default 512/64)
3. **Partial span support** - SpanFinder can predict start-only or end-only at boundaries
4. **Two-stage separation** - SpanFinder for recall, Validator for precision

---

## Model Specifications

### SpanFinder Model

Based on spaCy's SpanFinder architecture: a CNN-based sequence model that predicts span boundaries.

```
Input: Byte sequence (0-255)
    │
    ▼
┌─────────────────────────────────────┐
│  Byte Embedding                     │
│  - vocab_size: 256                  │
│  - embed_dim: configurable (64/96)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  CNN Encoder (1D Convolutions)      │
│  - window_size: 3 (context ±3)      │
│  - depth: 4 layers                  │
│  - residual connections             │
│  - activation: ReLU or Maxout       │
│  Receptive field = window * depth   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Linear + Sigmoid                   │
│  - Output: 2 values per byte        │
│  - [start_prob, end_prob]           │
└─────────────────────────────────────┘
```

| Component | Description |
|-----------|-------------|
| Input | Byte sequence (0-255), length up to chunk_size |
| Embedding | Learned byte embeddings, dim = width (64/96) |
| Encoder | **1D CNN** with sliding window, residual connections, depth layers |
| Output | 2 values per byte: [start_prob, end_prob] via linear + sigmoid |
| Loss | Binary cross-entropy per position |
| Inference | Threshold (default 0.5) -> Cartesian product of starts × ends |

**Key differences from spaCy's token-level SpanFinder:**
- Operates on **bytes** (vocab=256) instead of tokens
- Uses **1D convolutions** directly on byte embeddings
- Same boundary prediction approach: 2 outputs per position

**Size Presets:**

| Preset | Width | CNN Depth | Window | Receptive Field | Approx Params |
|--------|-------|-----------|--------|-----------------|---------------|
| Tiny   | 32    | 2         | 2      | 4 bytes         | ~30K          |
| Small  | 64    | 3         | 3      | 9 bytes         | ~100K         |
| Medium | 96    | 4         | 3      | 12 bytes        | ~300K         |

### Heuristic Validator Model

| Component | Description |
|-----------|-------------|
| Input | 6 floats (precomputed features) |
| Hidden Layers | Configurable: e.g., [16, 8] or [32, 16, 8] |
| Activation | ReLU (hidden), Sigmoid (output) |
| Output | 1 float: probability [0.0 - 1.0] |
| Loss | Binary cross-entropy |

### The 6 Features

1. **Shannon Entropy** - Bits of uncertainty per character
2. **Kolmogorov Complexity** - Huffman compression ratio (incompressibility)
3. **Text Frequency Diff** - Distance from natural language character distribution
4. **Innocuous Frequency Diff** - Distance from typical source code distribution
5. **Secret Frequency Diff** - Distance from known secret distribution
6. **Character Type Mix** - Ratio of transitions between letters/digits/special chars

Frequency tables are computed from training data and stored with the model.

---

## Model Packaging

### Bundle Directory Structure

```
heuristic-secrets-v1/
├── config.json               # Shared runtime configuration
├── spanfinder.safetensors    # SpanFinder weights
├── spanfinder.meta.json      # SpanFinder architecture config
├── validator.safetensors     # Validator weights
├── validator.meta.json       # Validator architecture + frequency tables
└── README.md                 # Model card (version, training info, metrics)
```

### config.json

```json
{
  "version": "1.0.0",
  "chunk_size": 512,
  "chunk_overlap": 64,
  "spanfinder_threshold": 0.5,
  "validator_threshold": 0.5,
  "min_span_length": 8,
  "max_span_length": 256
}
```

### spanfinder.meta.json

```json
{
  "architecture": "cnn",
  "vocab_size": 256,
  "width": 64,
  "cnn_depth": 3,
  "cnn_window": 3,
  "trained_on": "2026-01-17",
  "metrics": {
    "recall": 0.94,
    "candidates_per_kb": 12.3
  }
}
```

### validator.meta.json

```json
{
  "input_dim": 6,
  "hidden_dims": [16, 8],
  "frequencies": {
    "text": { "a": 0.082, "b": 0.015, "...": "..." },
    "innocuous": { "a": 0.045, "e": 0.038, "...": "..." },
    "secret": { "a": 0.062, "0": 0.058, "...": "..." }
  },
  "metrics": {
    "accuracy": 0.989,
    "precision": 0.985,
    "recall": 0.992
  }
}
```

---

## Training Data Strategy

### Hybrid Approach (Progressive)

**Phase 1: Bootstrap from Existing Tools**
- TruffleHog test fixtures (900+ secret patterns)
- Gitleaks test cases
- detect-secrets test suite
- Known data leaks (public disclosures)

**Phase 2: Academic Datasets**
- SecretBench (15K labeled true secrets, requires DPA)
- FPSecretBench (1.5M false positive examples)

**Phase 3: Synthetic Augmentation**
- Generate secrets with known patterns
- Insert into realistic code/config contexts
- Intentionally split across chunk boundaries for SpanFinder training

### Training Data Split

| Model | Positive Examples | Negative Examples |
|-------|-------------------|-------------------|
| **SpanFinder** | True secrets + False positives (both are valid candidates) | Normal code/text with nothing suspicious |
| **Validator** | True secrets only | False positives (UUIDs, hashes, base64, etc.) |

**SpanFinder's job:** "Is this a candidate worth checking?"
- Real API key -> find it
- UUID that looks like a key -> find it
- Base64 blob -> find it
- Normal variable names -> ignore

**Validator's job:** "Is this candidate actually a secret?"
- Real API key -> yes (0.95)
- UUID -> no (0.12)
- Base64 config data -> no (0.08)

### Frequency Tables

- **Text frequencies:** from NLP corpus (Wikipedia, books)
- **Innocuous frequencies:** from open-source code (GitHub top repos)
- **Secret frequencies:** from positive training examples

---

## Implementation Phases

### Phase 1: Python + PyTorch Prototype

**Goal:** Validate approach, iterate fast

```
Deliverables:
├── spanfinder/
│   ├── model.py          # Byte-level SpanFinder
│   ├── dataset.py        # Training data loading
│   └── train.py          # Training loop
├── validator/
│   ├── features.py       # 6 metric extraction
│   ├── model.py          # Configurable classifier
│   └── train.py          # Training loop
├── pipeline/
│   ├── chunker.py        # Chunking with overlap
│   ├── assembler.py      # Span stitching logic
│   └── detector.py       # End-to-end pipeline
├── data/
│   ├── collect.py        # Scrape existing tool test suites
│   └── augment.py        # Synthetic data generation
└── eval/
    ├── benchmark.py      # Accuracy/speed evaluation
    └── compare_sizes.py  # Model size comparison
```

### Phase 2: Rust + Candle Production

**Goal:** Maximum performance

```
Deliverables:
├── crates/
│   ├── heuristic-secrets/        # Core library
│   │   ├── spanfinder.rs         # SpanFinder in Candle
│   │   ├── validator.rs          # Validator in Candle
│   │   ├── features.rs           # SIMD-optimized metrics
│   │   ├── pipeline.rs           # Chunking + assembly
│   │   └── lib.rs                # Public API
│   ├── heuristic-secrets-cli/    # CLI tool
│   └── heuristic-secrets-train/  # Training tools
└── Cargo.toml
```

### Phase 3: Tooling & Bindings

**Goal:** Integration ecosystem

```
Deliverables:
├── python/
│   ├── heuristic_secrets/
│   │   ├── _rust.pyi     # Rust extension stubs
│   │   ├── _torch.py     # PyTorch fallback
│   │   └── __init__.py   # Auto-selects backend
│   └── pyproject.toml
├── CLI features
│   ├── scan files/directories
│   ├── scan git history
│   └── output formats (JSON, SARIF, text)
└── CI integrations
    ├── GitHub Action
    └── Pre-commit hook
```

---

## API Design

### Rust Library API

```rust
use heuristic_secrets::{Detector, Config, Detection};

// Load model bundle
let detector = Detector::load("./models/heuristic-secrets-v1")?;

// Or with custom config
let config = Config::builder()
    .chunk_size(1024)
    .chunk_overlap(128)
    .spanfinder_threshold(0.4)
    .validator_threshold(0.6)
    .device(Device::Cuda(0))
    .build();
let detector = Detector::load_with_config("./models/v1", config)?;

// Single string
let detections: Vec<Detection> = detector.scan("api_key = 'sk-abc123xyz'")?;

// Batch processing (parallel)
let inputs: Vec<&str> = vec![...];
let results: Vec<Vec<Detection>> = detector.scan_batch(&inputs)?;

// File scanning
let detections = detector.scan_file("config.yaml")?;

// Detection struct
pub struct Detection {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub probability: f32,
}
```

### Python API

```python
from heuristic_secrets import Detector, Config

# Auto-selects Rust backend if available, else PyTorch
detector = Detector.load("./models/heuristic-secrets-v1")

# Force specific backend
detector = Detector.load("./models/v1", backend="torch")  # or "rust"

# Scan
detections = detector.scan("api_key = 'sk-abc123xyz'")
for d in detections:
    print(f"{d.start}:{d.end} ({d.probability:.2f}): {d.text}")

# Batch (parallel in Rust, vectorized in PyTorch)
results = detector.scan_batch(["string1", "string2", ...])

# File
detections = detector.scan_file("config.yaml")
```

### CLI

```bash
# Scan file
heuristic-secrets scan config.yaml

# Scan directory
heuristic-secrets scan ./src --recursive

# JSON output
heuristic-secrets scan ./src -o json

# Custom thresholds
heuristic-secrets scan ./src --spanfinder-threshold 0.4 --validator-threshold 0.7

# Git history
heuristic-secrets scan-git --since "2024-01-01"
```

---

## Backend Selection

The Python package supports multiple backends:

```
Python Package
├── Pure PyTorch backend (works everywhere: CPU, CUDA, MPS, ROCm)
└── Optional Rust extension (PyO3) - used when available for max performance

Runtime selection:
if rust_extension_available():
    use_rust_backend()  # Fastest
else:
    use_torch_backend()  # Broadest compatibility
```

**Benefits:**
- MPS (Apple Silicon) support via PyTorch
- ROCm (AMD) support via PyTorch
- Easier debugging during development
- Graceful degradation on exotic platforms

---

## References

- **Paper:** "Beyond RegEx – Heuristic-based Secret Detection" (Burdick-Pless, 2025)
- **SpanFinder:** spaCy boundary prediction component (Explosion AI)
- **SecretBench:** https://github.com/setu1421/SecretBench
- **Candle:** https://github.com/huggingface/candle
