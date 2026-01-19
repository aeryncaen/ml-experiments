# Mamba3 and State Space Models for Long Sequences

## Overview

Mamba is a family of state space models (SSMs) that achieve linear complexity in sequence length while matching or exceeding Transformer performance. This makes them ideal for processing very long sequences (100k+ tokens).

## Mamba Evolution

### Mamba 1 (Gu & Dao, December 2023)

**Key innovations**:
- Selective state spaces: input-dependent dynamics (unlike fixed S4)
- Hardware-aware implementation: fused CUDA kernels
- Linear complexity: O(L) vs O(L²) for attention

```python
# Core selective scan
def selective_scan(x, A, B, C, delta):
    # x: input [B, L, D]
    # A, B, C: learned matrices (input-dependent in Mamba)
    # delta: discretization step (also input-dependent)
    h = torch.zeros(B, D, N)  # State
    outputs = []
    for t in range(L):
        h = A * h + B * x[:, t]  # State update
        y = C @ h  # Output
        outputs.append(y)
    return torch.stack(outputs, dim=1)
```

### Mamba 2 (Dao & Gu, May 2024)

**Key innovations**:
- State Space Duality (SSD): connection between SSMs and attention
- 2-8x faster than Mamba 1
- Simplified architecture, easier to implement
- Better scaling to larger models

```python
# Mamba-2 uses structured matrices for faster computation
# Key insight: selective SSM ≈ linear attention with specific structure
```

### Mamba 3 (Anticipated 2025-2026)

**Expected features** (based on research direction):
- MIMO (Multi-Input Multi-Output) formulation
- Trapezoidal discretization for stability
- Complex-valued dynamics
- Even faster kernels

**Current status**: 
- Paper under review (ICLR 2026 submission noted)
- Not yet in official `mamba-ssm` repo
- GitHub issue #809 tracks release

## Installation & Usage

### Official Package

```bash
pip install mamba-ssm
# Requires CUDA, won't work on CPU/MPS
```

### Basic Usage (Mamba 2)

```python
from mamba_ssm import Mamba2

# Single Mamba layer
mamba = Mamba2(
    d_model=256,    # Model dimension
    d_state=64,     # SSM state dimension (larger = more capacity)
    d_conv=4,       # Local convolution width
    expand=2,       # Expansion factor for inner dimension
)

x = torch.randn(batch, length, 256)
y = mamba(x)  # Same shape output
```

### Stacking Layers

```python
class MambaStack(nn.Module):
    def __init__(self, d_model, n_layers, d_state=64):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))  # Pre-norm residual
        return x
```

### Replacing Attention in Existing Models

```python
# Before: Transformer
class OldModel(nn.Module):
    def __init__(self):
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4),
            num_layers=4
        )

# After: Mamba
class NewModel(nn.Module):
    def __init__(self):
        self.encoder = MambaStack(d_model=256, n_layers=4)
```

## Performance Characteristics

### Memory Usage

| Sequence Length | Attention Memory | Mamba Memory |
|-----------------|------------------|--------------|
| 1K | 4 MB | 1 MB |
| 8K | 256 MB | 8 MB |
| 32K | 4 GB | 32 MB |
| 128K | 64 GB | 128 MB |
| 800K | OOM | ~800 MB |

Mamba scales linearly; attention scales quadratically.

### Speed (Tokens/Second)

| Model | 1K seq | 8K seq | 32K seq |
|-------|--------|--------|---------|
| Transformer | 50K | 12K | 1K |
| Mamba-1 | 100K | 90K | 80K |
| Mamba-2 | 200K | 180K | 160K |

Mamba maintains throughput at long sequences.

### Quality

- Matches Transformers on most NLP benchmarks
- Slightly weaker on tasks requiring precise "lookup" (attention excels here)
- Better on very long range dependencies
- Excellent for streaming/online processing

## Limitations & Gotchas

### 1. CUDA Required

```python
# This will fail on CPU/MPS
from mamba_ssm import Mamba2  # Needs CUDA

# Workaround for development: use pure PyTorch implementation (slower)
# See: https://github.com/johnma2006/mamba-minimal
```

### 2. Bidirectional Processing

Mamba is inherently causal (left-to-right). For bidirectional:

```python
class BiMamba(nn.Module):
    def __init__(self, d_model):
        self.forward_mamba = Mamba2(d_model)
        self.backward_mamba = Mamba2(d_model)
        self.combine = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x):
        fwd = self.forward_mamba(x)
        bwd = self.backward_mamba(x.flip(1)).flip(1)
        return self.combine(torch.cat([fwd, bwd], dim=-1))
```

### 3. State Size Tuning

- `d_state=16`: Fast, lower capacity
- `d_state=64`: Good balance (recommended)
- `d_state=256`: Slower, higher capacity for complex patterns

### 4. No Attention-Style Interpretability

Can't visualize "what attends to what" - state is compressed.

## Integration for Secret Detection

### Recommended Architecture

```python
class SecretMaskPredictor(nn.Module):
    """
    Long-sequence secret mask prediction using Mamba.
    Handles sequences up to 800K+ tokens.
    """
    def __init__(
        self,
        vocab_size=256,  # Byte-level
        d_model=256,
        d_state=64,
        n_layers=6,
    ):
        super().__init__()
        
        # Byte embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Bidirectional Mamba encoder
        self.encoder = nn.ModuleList([
            BiMamba(d_model) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Mask prediction head
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: [B, L] byte indices
        h = self.embed(x)  # [B, L, D]
        
        for layer, norm in zip(self.encoder, self.norms):
            h = h + layer(norm(h))
        
        return self.head(h).squeeze(-1)  # [B, L] logits
```

### Handling 800K+ Sequences

For truly massive sequences, use chunked processing:

```python
def predict_long_sequence(model, x, chunk_size=32768, overlap=1024):
    """Process very long sequences with overlapping chunks."""
    L = x.shape[1]
    if L <= chunk_size:
        return model(x)
    
    outputs = []
    for start in range(0, L, chunk_size - overlap):
        end = min(start + chunk_size, L)
        chunk = x[:, start:end]
        out = model(chunk)
        
        # Handle overlap: use first chunk's predictions for overlap region
        if start > 0:
            out = out[:, overlap:]
        outputs.append(out)
    
    return torch.cat(outputs, dim=1)
```

### Memory-Efficient Training

```python
# Gradient checkpointing for long sequences
from torch.utils.checkpoint import checkpoint

class MemoryEfficientMamba(nn.Module):
    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = x + checkpoint(layer, norm(x))  # Trade compute for memory
        return x
```

## Alternative: Mamba-Minimal (Pure PyTorch)

For development without CUDA:

```bash
git clone https://github.com/johnma2006/mamba-minimal
```

```python
# Slower but works everywhere
from mamba_minimal import Mamba

mamba = Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
```

## Comparison with Alternatives

| Approach | Complexity | Quality | Long Seq | Ease of Use |
|----------|------------|---------|----------|-------------|
| Full Attention | O(L²) | Best | Poor | Easy |
| Sparse Attention | O(L√L) | Good | OK | Medium |
| Linear Attention | O(L) | OK | Good | Easy |
| Mamba | O(L) | Great | Great | Medium |
| RWKV | O(L) | Good | Good | Easy |

**Recommendation**: Mamba for quality + speed on long sequences.

## References

1. Gu & Dao "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
2. Dao & Gu "Transformers are SSMs" (2024) - Mamba-2
3. Official repo: https://github.com/state-spaces/mamba
4. Mamba-minimal: https://github.com/johnma2006/mamba-minimal
5. S4: Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces" (2022)

## Next Steps for This Project

1. **Prototype with Mamba-2**: Already released, well-tested
2. **Use BiMamba for mask prediction**: Secrets need bidirectional context
3. **Chunk processing for 800K+**: 32K chunks with 1K overlap
4. **Upgrade to Mamba-3 when released**: Watch issue #809
