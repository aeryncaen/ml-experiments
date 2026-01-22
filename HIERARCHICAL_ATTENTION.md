# Hierarchical N-Dimensional Local Attention

An efficient O(L·K) alternative to O(L²) full attention that maintains global receptive field through multi-scale pooling and an integrated adaptive convolution system.

## Architecture Overview

```
Input (B, *spatial_dims, C)
         │
         ▼
    ┌─────────────────────────────────────┐
    │            Level 0                  │
    │  ┌─────────────┐                    │
    │  │AdaptiveConv │ ──► Position cross-│
    │  │   + Norm    │     attention conv │
    │  └──────┬──────┘     with residual  │
    │         │                           │
    │         ▼                           │
    │  ┌─────────────┐                    │
    │  │LocalAttnND  │ ──► Full resolution│
    │  └──────┬──────┘                    │
    └─────────┼───────────────────────────┘
              │
         RMSNorm + strided conv
              │
              ▼
    ┌─────────────────────────────────────┐
    │            Level 1                  │
    │  AdaptiveConv + LocalAttnND         │
    │  (half resolution)                  │
    └─────────┬───────────────────────────┘
              │
             ...
              │
              ▼
    ┌─────────────────────────────────────┐
    │          Level N-1                  │
    │  (coarsest resolution)              │
    └─────────┬───────────────────────────┘
              │
              ▼
    ┌───────────────────────────────────┐
    │ Learned Query Level Aggregation   │
    │   level_query @ level_means.T     │
    │          ──► ctx                  │
    │   ctx @ level0_hidden.T           │
    │          ──► update + residual    │
    └───────────────┬───────────────────┘
                    │
                    ▼
Output (B, *spatial_dims, C)
```

## Core Components

### 1. AdaptiveConvND

Position cross-attention convolution integrated at every hierarchy level. Uses `grid_sample` for efficient N-dimensional sampling with learned deformation.

**Parameters:**
- `channels`: Model dimension
- `ndim`: Number of spatial dimensions (1, 2, or 3)
- `max_kernel_size`: Maximum kernel window size
- `num_channels`: Number of depthwise channel groups (heads)

**Key Features:**

| Feature | Implementation |
|---------|----------------|
| **Position Cross-Attention** | Q from content, K from scaled positions, V from gathered window |
| **Adaptive Width** | Per-position width scales the coordinate space |
| **Adaptive Sharpness** | Per-position inverse temperature for attention |
| **Deformation** | Per-position spatial offset via `grid_sample` |
| **SE Block** | Per-(position, channel) reweighting |
| **SiLU Activation** | All projections activated before final nonlinearity |

**Position Cross-Attention Mechanism:**

The kernel weights are computed via cross-attention where:
- **Query**: Content-derived embedding at output position
- **Keys**: Relative positions scaled by 1/width (smaller width → positions appear farther → attention focuses on center)
- **Values**: Input features gathered from the deformed window
- **Sharpness**: Scales attention logits (inverse temperature)

```python
# Per-position adaptive parameters
width = sigmoid(adapt_proj(x)[..., 0]) * half_k + 0.5      # [0.5, half_k+0.5]
sharpness = sigmoid(adapt_proj(x)[..., 1]) * 9.5 + 0.5    # [0.5, 10]
deform = tanh(silu(deform_proj(x))) * half_k              # [-half_k, half_k]

# Gather window with deformation via grid_sample
values = grid_sample(x, centers + rel_pos + deform)       # (B, L, H, K, D)

# Position cross-attention
queries = silu(query_proj(x))                              # (B, L, H, pos_dim)
scaled_pos = rel_pos / width                               # smaller width = positions farther
keys = key_proj(scaled_pos)                                # (B, L, H, K, pos_dim)

attn_logits = einsum('blhd,blhkd->blhk', Q, K) * scale * sharpness
attn_weights = softmax(attn_logits)

output = einsum('blhkd,blhk->blhd', values, attn_weights)

# SE per-(position, channel)
se_weights = sigmoid(se_fc2(silu(se_fc1(output))))         # (B, L, C)
output = output * se_weights
```

**Supported Dimensions:**
- 1D: Sequences (B, L, C)
- 2D: Images (B, H, W, C)
- 3D: Volumes (B, D, H, W, C)

All use `F.grid_sample` with bilinear interpolation and border padding.

### 2. LocalAttentionND

The base attention mechanism operating on local windows.

**Parameters:**
- `embed_dim`: Model dimension
- `kernel_size`: Window size (int or tuple for each spatial dim)
- `ndim`: Number of spatial dimensions (1, 2, or 3)
- `num_channels`: Number of attention heads

**Key Features:**

| Feature | Implementation |
|---------|----------------|
| **QK Normalization** | RMSNorm on Q and K before attention |
| **RoPE** | Rotary position embeddings on Q and K |
| **Learned Adaptive Width** | Per-token, per-channel attention width |
| **Soft Masking** | `sigmoid((width - rel_dist) * sharpness)` for differentiable window |
| **Unfold-based** | Efficient windowed attention via tensor unfolding |

### 3. HierarchicalLocalAttentionND

Multi-scale wrapper that applies AdaptiveConv + LocalAttention at progressively coarser resolutions.

**Parameters:**
- `embed_dim`: Model dimension
- `window_size`: Local attention window (int or tuple)
- `ndim`: Spatial dimensions (1, 2, or 3)
- `num_channels`: Attention heads
- `poolable_dims`: Which dimensions to pool (default: all)
- `min_size`: Stop pooling when dim < min_size (default: 4)

**Key Features:**

| Feature | Implementation |
|---------|----------------|
| **Auto Levels** | `n_levels = bit_length(min_poolable // min_size)` |
| **Selective Reduction** | Only reduce specified dims (e.g., H/W but not depth) |
| **AdaptiveConv at Each Level** | Position cross-attention conv before local attention |
| **RMSNorm Before Reduction** | Prevents exploding activations across levels |
| **Strided Reduce Conv** | Learned downsampling between levels |
| **Shared Weights** | Same attention + conv + reduce conv at all levels |
| **Two-Stage Aggregation** | Learned query → level means → context → hidden states |

**Forward Pass:**
```python
h = x
levels = []

for i in range(n_levels):
    # AdaptiveConv with residual
    conv_out, _ = self.conv(h)
    conv_out = conv_norm(conv_out)
    h = h + conv_out
    
    # Local attention
    h = local_attn(h)
    levels.append(h)
    
    # Reduce for next level (with RMSNorm)
    if i < n_levels - 1:
        h = reduce_norm(h)
        h = reduce_conv(h)

# Stage 1: Learned query attends to level means → global context
level_means = stack([lvl.mean(spatial_dims) for lvl in levels])
ctx = softmax(level_query @ level_k.T * scale) @ level_v

# Stage 2: Context cross-attends to level0 hidden states
update = softmax(ctx_proj(ctx) @ hidden_k.T * scale) @ hidden_v

# Residual (broadcasts to all positions)
out = level0 + out_proj(update)
```

## Complexity Analysis

| Method | Time | Space | Receptive Field |
|--------|------|-------|-----------------|
| Full Attention | O(L²) | O(L²) | Global |
| Local Attention | O(L·K) | O(L·K) | K tokens |
| **Hierarchical Local** | O(2L·K) | O(L·K) | Global |

The hierarchical version does ~2x the work of plain local (geometric series: L + L/2 + L/4 + ... ≈ 2L) but gains global receptive field through coarse levels.

## Usage Examples

### 1D Sequences
```python
attn = HierarchicalLocalAttentionND(
    embed_dim=128,
    window_size=17,
    ndim=1,
    num_channels=4,
)
x = torch.randn(B, L, 128)
out = attn(x)  # (B, L, 128)
```

### 2D Images
```python
attn = HierarchicalLocalAttentionND(
    embed_dim=128,
    window_size=(7, 7),
    ndim=2,
    num_channels=4,
)
x = torch.randn(B, H, W, 128)
out = attn(x)  # (B, H, W, 128)
```

### 3D Volumes (Selective Pooling)
```python
attn = HierarchicalLocalAttentionND(
    embed_dim=128,
    window_size=(5, 5, 5),
    ndim=3,
    num_channels=4,
    poolable_dims=(1, 2),  # pool H, W only; preserve depth
)
x = torch.randn(B, D, H, W, 128)
out = attn(x)  # (B, D, H, W, 128)
```

## Design Decisions

### Why Position Cross-Attention for Adaptive Conv?
- **Width scales coordinate space**: Smaller width makes positions appear farther, naturally focusing attention on center without hard masking
- **Sharpness as temperature**: Controls how peaked vs. uniform the kernel is
- **Content-dependent**: Query derived from content, not just fixed positional embedding
- **Unified framework**: Same cross-attention pattern as the local attention, just operating on kernel positions

### Why Deformation via grid_sample?
- Hardware-optimized bilinear interpolation
- Supports 1D, 2D, 3D uniformly
- Gradients flow through sampling locations
- Per-position learned spatial offsets

### Why RMSNorm Before Reduction?
- Prevents activation explosion across hierarchy levels
- Each level sees normalized inputs regardless of previous level's scale
- Critical for training stability

### Why SiLU on All Projections?
- Smooth nonlinearity before final activation (sigmoid/tanh/softmax)
- Consistent with modern architectures (SwiGLU, etc.)
- Better gradient flow than ReLU

### Why SE Per-(Position, Channel)?
- Original SE: global pool → fc → fc → per-channel scale (same for all positions)
- Ours: fc → fc → per-(position, channel) scale
- Allows position-specific channel reweighting, more expressive

### Why Two-Stage Learned Query Aggregation?
- **Stage 1**: Learned query attends to level means → extracts scale-relevant context
- **Stage 2**: Context cross-attends to full-resolution hidden states → localizes global info
- Better than FiLM: allows position-specific updates rather than global scale/bias
- Better than averaging: learns which scales matter per-input

## File Locations

| Component | Path |
|-----------|------|
| AdaptiveConvND | `src/heuristic_secrets/models/scatter_attention.py` |
| LocalAttentionND | `src/heuristic_secrets/models/scatter_attention.py` |
| HierarchicalLocalAttentionND | `src/heuristic_secrets/models/scatter_attention.py` |
| Benchmark | `scripts/bench_scatter_vs_attn.py` |
