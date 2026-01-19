# Heuristic Secrets: Technical Architecture

A hybrid secret detection model combining handcrafted heuristic features with learned byte-level attention.

## Overview

The Validator model classifies candidate text spans as secrets or false positives using:

1. **Heuristic Features**: Two precomputed frequency-distance features
2. **Byte Attention Head**: Attention over raw bytes learns patterns heuristics miss
3. **Adaptive Gating**: Input-dependent gates weight feature importance

Current performance: **F1=0.926, Recall=0.942** with 11K parameters.

## Architecture

```
                                    ┌─────────────────────┐
                                    │    Raw Bytes        │
                                    │   (0-255, padded)   │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  Byte Embedding     │
                                    │  (256 → 32 dim)     │
                                    └──────────┬──────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          │                    │                    │
               ┌──────────▼──────────┐  ┌──────▼──────┐  ┌──────────▼──────────┐
               │  Feature Attention  │  │   Shared    │  │   Gate Attention    │
               │  (learned query)    │  │   K, V      │  │  (learned query)    │
               └──────────┬──────────┘  └─────────────┘  └──────────┬──────────┘
                          │                                         │
               ┌──────────▼──────────┐                   ┌──────────▼──────────┐
               │  Linear → 1 dim     │                   │  Linear → 2 dim     │
               │  (attention output) │                   │  (gate adjustments) │
               └──────────┬──────────┘                   └──────────┬──────────┘
                          │                                         │
                          │                                         ▼
                          │                              ┌─────────────────────┐
┌─────────────────────┐   │                              │   Learned Gate      │
│  Heuristic Features │   │                              │   Bias (2 dim)      │
│  text_freq_diff     │   │                              └──────────┬──────────┘
│  secret_freq_diff   │   │                                         │
└──────────┬──────────┘   │                              ┌──────────▼──────────┐
           │              │                              │  sigmoid(bias +     │
           │              │                              │  adjustments)       │
           │              │                              └──────────┬──────────┘
           │              │                                         │
           ▼              │                                         ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    Element-wise Gating                               │
    │              gated_features = features × gates                       │
    └──────────────────────────────────┬───────────────────────────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   Concatenate           │
                          │ [gated_features(2),     │
                          │  attn_output(1)]        │
                          └────────────┬────────────┘
                                       │
                          ┌────────────▼────────────┐
                          │   MLP Classifier        │
                          │   3 → 32 → 16 → 1       │
                          │   ReLU, Sigmoid         │
                          └────────────┬────────────┘
                                       │
                                       ▼
                               P(secret | input)
```

## Heuristic Features

After extensive ablation (63 feature combinations tested), only **2 features** are used:

### 1. Text Frequency Difference
Euclidean distance from English character frequencies.

```python
def char_frequency_difference(data: bytes, reference: FrequencyTable) -> float:
    observed = compute_frequencies(data)
    return euclidean_distance(observed, reference)
```

### 2. Secret Frequency Difference
Distance from known secret patterns (API keys, tokens, etc.).

### Why Only 2 Features?

The ablation study found that adding more features (entropy, kolmogorov, char_type_mix, innocuous_freq_diff) actually **hurt** performance. The byte attention head learns everything those features captured, making them redundant noise.

| Features | F1 | Recall |
|----------|-----|--------|
| All 6 features | 0.885 | 0.905 |
| text_freq + secret_freq only | **0.926** | **0.942** |

## Byte Attention Head

A single-layer attention mechanism over raw bytes:

```python
class ByteAttentionHead(nn.Module):
    def __init__(self, embed_dim=32, output_dim=1, num_gates=2):
        self.embed = nn.Embedding(256, embed_dim)
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, output_dim)
        
        # Second head for gate adjustments
        self.gate_query = nn.Parameter(torch.randn(embed_dim))
        self.gate_out = nn.Linear(embed_dim, num_gates)
```

The attention learns to focus on discriminative byte patterns:
- Base64 padding (`=`)
- Hex character distribution
- Token prefixes (`sk-`, `ghp_`, etc.)
- Length and structure patterns

### Dual-Head Design

Two attention heads share K/V projections:
- **Feature Head**: 1-dim output concatenated to heuristics
- **Gate Head**: 2-dim adjustments to learned gate biases

## Architecture Search Results

Tested attention dimensions [4, 8, 16, 32, 64] × MLP configs:

| Attention Dim | MLP | Params | F1 | Recall |
|---------------|-----|--------|-----|--------|
| 32 | [32, 16] | 11,142 | **0.926** | **0.942** |
| 16 | [32, 16] | 5,558 | 0.922 | 0.934 |
| 64 | [32, 16] | 22,310 | 0.921 | 0.938 |

The 32-dim attention with [32, 16] MLP was selected for best recall (fewer missed secrets).

### DeformConv Head Experiment

Tested replacing attention with 1D deformable convolutions (DCNv3-style):

| Head Type | Params | F1 | Recall |
|-----------|--------|-----|--------|
| Attention | 11,142 | 0.920 | 0.934 |
| DeformConv (d=2, k=5) | 13,078 | 0.915 | 0.937 |

DeformConv achieved slightly higher recall but worse precision. Attention is simpler and performs better overall.

## Training

### Configuration

```python
TrainingConfig(
    epochs=15,
    batch_size=64,
    lr=0.001,
    weight_decay=0.01,
)
```

### Loss Function

Binary cross-entropy with sigmoid output.

### Data

| Split | Total | Secrets | Non-Secrets |
|-------|-------|---------|-------------|
| Train | 14,103 | 1,300 | 12,803 |
| Val | 1,762 | 181 | 1,581 |
| Test | 1,764 | 148 | 1,616 |

**Sources**:
- Secrets: detect-secrets, gitleaks, trufflehog test suites
- Non-secrets: Generated UUIDs, MD5, SHA1, SHA256, base64

## Performance Summary

### Final Model

| Metric | Value |
|--------|-------|
| F1 Score | 0.926 |
| Precision | 0.906 |
| Recall | 0.942 |
| Parameters | 11,142 |

---

## ByteMasker: Byte-Level Span Detection

ByteMasker finds the exact byte positions of secrets within lines. While the Validator classifies pre-extracted candidates, ByteMasker operates on raw text to locate secret boundaries.

### Architecture

```
Input: (batch, seq_len) byte values 0-255

    ┌─────────────────────────────────────┐
    │         Byte Embedding              │
    │         (256 → width)               │
    └──────────────────┬──────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │      DeformConv1d Block × depth     │◄──── Residual + LayerNorm
    │                                     │
    │  ┌───────────────────────────────┐  │
    │  │  Depthwise Conv (k=3)         │  │
    │  │  → LayerNorm → GELU           │  │
    │  └───────────────┬───────────────┘  │
    │                  │                  │
    │    ┌─────────────┴─────────────┐    │
    │    ▼                           ▼    │
    │  Offset Net                 Mask Net│
    │  (G×K offsets)            (G×K weights)
    │    │                           │    │
    │    └─────────────┬─────────────┘    │
    │                  ▼                  │
    │         Deformable Sampling         │
    │    (bilinear interp at offsets)     │
    │                  │                  │
    │                  ▼                  │
    │           Output Proj               │
    └──────────────────┬──────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────┐
    │           Output Head               │
    │   Linear(width → width/2) → GELU    │
    │   Linear(width/2 → 1)               │
    └──────────────────┬──────────────────┘
                       │
                       ▼
Output: (batch, seq_len) per-byte logits
```

### 1D Deformable Convolution (DCNv3-style)

Standard convolutions sample at fixed offsets. Deformable convolutions learn input-dependent offsets, allowing the receptive field to adapt to content structure.

**Key components:**

1. **Learned Offsets**: Each position predicts K offsets (one per kernel point) that shift sampling locations
2. **Attention Mask**: Softmax weights over K sampling points (like DCNv3, not DCNv1/v2 scalar modulation)
3. **Groups**: G independent groups specialize on different patterns (similar to multi-head attention)

```python
# Per-position sampling
for k in range(kernel_size):
    # Base offset + learned offset
    sample_pos = position + base_offset[k] + learned_offset[k]
    
    # Bilinear interpolation for sub-pixel sampling
    value = bilinear_sample(input, sample_pos)
    
    # Weighted contribution
    output += value * softmax_mask[k]
```

**Why deformable conv for span detection?**

- Secrets have variable structure (API keys vs JWTs vs passwords)
- Fixed receptive fields can't adapt to different token lengths
- Offsets learn to "reach" for relevant context (e.g., `=` padding, prefix patterns)

### Model Configurations

| Preset | Width | Depth | Kernel | Groups | Params |
|--------|-------|-------|--------|--------|--------|
| Tiny   | 32    | 2     | 5      | 4      | ~15K   |
| Small  | 64    | 3     | 7      | 4      | ~60K   |
| Medium | 96    | 4     | 7      | 8      | ~150K  |

### Training

**Data**: Lines extracted from documents, with byte-level masks indicating secret positions.

**Loss**: Binary cross-entropy per byte, masked for padding.

**Sampling**: Train primarily on secret-containing lines to focus learning on positive examples. Validate FP rate on clean lines.

```python
trainer = ByteMaskerTrainer(
    model=ByteMaskerModel(ByteMaskerConfig.small()),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    lr=1e-3,
    batch_size=64,
    train_on_secret_lines_only=True,
)
```

### Dataset

Lines are extracted from SpanFinder documents and cached as JSONL:

```json
{"b": "c2stYWJjMTIz...", "m": [0,0,0,1,1,1,1,0,0], "s": true}
```

- `b`: Base64-encoded line bytes
- `m`: Per-byte mask (1 = secret)
- `s`: Has any secret

Bucketed batching groups lines by length to minimize padding overhead.

---

## Implementation

### Core Files

```
src/heuristic_secrets/
├── validator/
│   ├── model.py      # ValidatorModel, ValidatorConfig, ByteAttentionHead
│   ├── features.py   # Heuristic feature extraction
│   ├── train.py      # Training loop with MPS/CUDA support
│   └── dataset.py    # Data loading with bucketed batching
├── bytemasker/       # Byte-level span detection (WIP)
│   ├── model.py      # 1D DeformConv architecture
│   ├── dataset.py    # Line-level dataset with JSONL caching
│   └── train.py      # Trainer
└── io.py             # Model serialization
```

### Usage

```python
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig

config = ValidatorConfig()  # Uses optimal defaults
model = ValidatorModel(config)

# Forward pass
features = torch.tensor([[0.5, 0.3]])  # [text_freq_diff, secret_freq_diff]
byte_ids = torch.tensor([[115, 107, 45, ...]])  # "sk-..."
prob = model(features, byte_ids=byte_ids)
```

### Training

```bash
python scripts/train_validator.py --epochs 15
python scripts/arch_search.py  # Architecture search
python scripts/head_comparison.py  # Attention vs DeformConv
```
