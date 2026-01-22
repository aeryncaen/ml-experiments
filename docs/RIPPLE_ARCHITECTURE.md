# Ripple: A Novel Attention + SSM Architecture

Ripple combines **SGSB attention** (Scatter-Gather-Scatter-Broadcast) with a **MIMO Jacobi SSM** (State Space Model), featuring cross-layer state accumulation via low-rank attention. This document provides a comprehensive technical breakdown.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
   - [AdaptiveConvND](#adaptiveconvnd)
   - [SIRENDownsampleND](#sirendownsamplend)
   - [LocalAttentionND](#localattentionnd)
   - [LowRankAttentionMergeND](#lowrankattentionmergend)
   - [SGSBAttentionND](#sgsbattentionnd)
   - [MIMOJacobiSSM_ND](#mimojacobissm_nd)
   - [RippleLayerND](#ripplelayernd)
3. [Full Model](#full-model)
4. [Key Innovations](#key-innovations)

---

## Architecture Overview

Ripple uses a **shared SSM** at the model level that performs one Jacobi iteration per layer. Each layer maintains its own **RNN-style hidden state** that accumulates within a single forward pass (reset between forwards).

```
Input: (B, *spatial, C)
         │
         ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      RippleModelND                          │
    │                                                             │
    │   ┌─────────────────────────────────────────────────────┐   │
    │   │              MIMOJacobiSSM_ND (SHARED)              │   │
    │   │        H = ssm.init_state(x)  # model-level         │   │
    │   └─────────────────────────────────────────────────────┘   │
    │                           │                                 │
    │         ┌─────────────────┼─────────────────┐               │
    │         │                 │                 │               │
    │         ▼                 ▼                 ▼               │
    │   ┌───────────┐    ┌───────────┐    ┌───────────┐          │
    │   │  Layer 0  │    │  Layer 1  │    │  Layer 2  │   ...    │
    │   │           │    │           │    │           │          │
    │   │ layer_s₀  │    │ layer_s₁  │    │ layer_s₂  │          │
    │   └───────────┘    └───────────┘    └───────────┘          │
    │         │                 │                 │               │
    │         └────────┬────────┴────────┬────────┘               │
    │                  │                 │                        │
    │            accumulated        output                        │
    └─────────────────────────────────────────────────────────────┘

Per-Layer Detail (RippleLayerND):
    ┌─────────────────────────────────────────┐
    │  ┌─────────────────────────────────┐   │
    │  │  Accumulated Attention (if L>1) │   │  ← attends to reduced states from ALL previous layers
    │  └─────────────────────────────────┘   │
    │                  │                      │
    │                  ▼                      │
    │  ┌─────────────────────────────────┐   │
    │  │       SGSBAttentionND           │   │  ← scatter → gather → scatter → broadcast
    │  └─────────────────────────────────┘   │
    │                  │                      │
    │                  ▼                      │
    │             + ssm_out                   │  ← from shared SSM (one step this layer)
    │                  │                      │
    │                  ▼                      │
    │  ┌─────────────────────────────────┐   │
    │  │    LowRankAttentionMergeND      │   │  ← merge with original embedding
    │  └─────────────────────────────────┘   │
    │                  │                      │
    │                  ▼                      │
    │  ┌─────────────────────────────────┐   │
    │  │   State Gate + Layer State      │   │  ← RNN-style accumulation within forward
    │  └─────────────────────────────────┘   │
    │                  │                      │
    │                  ▼                      │
    │  ┌─────────────────────────────────┐   │
    │  │   SIRENDownsampleND → reduced   │───┼──→ accumulated list (for future layers)
    │  └─────────────────────────────────┘   │
    └─────────────────────────────────────────┘
```

---

## Core Components

### AdaptiveConvND

**Purpose**: Data-dependent spatial sampling with learned frequencies.

**Key Insight**: Instead of fixed convolution kernels, sample at positions determined by the input itself. Each position learns its own sampling frequency, phase, and decay.

**Parameters**:
- `wave_proj`: Linear(C, 3H) → frequency, phase, decay per channel
- `query_proj`: Linear(C, H×pos_dim) → position queries
- `key_proj`: Linear(1, pos_dim) → relative distance encoding
- `out_proj`: Linear(C, C) → output projection
- `se_fc1/2`: Squeeze-excitation for channel recalibration

**Forward Pass**:

```
Input x: (B, *spatial, C)

1. Compute wave parameters per position:
   wave_params = silu(wave_proj(x))  → (B, L, 3, H)
   freq = sigmoid(wave_params[..., 0]) * (max_freq - min_freq) + min_freq
   phase = tanh(wave_params[..., 1]) * max_freq
   decay = sigmoid(wave_params[..., 2]) * 9.5 + 0.5

2. Compute sampling positions:
   For each center position c and offset k in stride_grid:
     sample_pos[c, k] = c + stride_grid[k] * freq[c] + phase[c]

3. Gather values at sample positions (with boundary clamping):
   values = x[sample_pos]  → (B, L, S, H, D)

4. Compute attention over samples:
   queries = silu(query_proj(x))
   keys = silu(key_proj(rel_dist))
   attn = softmax(queries @ keys) * decay_envelope * valid_mask
   output = einsum(attn, values)

5. Squeeze-excitation recalibration:
   se_weights = sigmoid(se_fc2(silu(se_fc1(output))))
   output = output * se_weights
```

**Why it matters**: The network learns WHERE to look, not just WHAT to look for. Positions with high-frequency content can sample densely; smooth regions can sample sparsely.

---

### SIRENDownsampleND

**Purpose**: Learned downsampling with SIREN-generated kernels that adapt to target resolution.

**Key Insight**: Use a small coordinate MLP (SIREN) to generate depthwise convolution kernels on-the-fly based on the desired output shape.

**Architecture**:
```
KernelNetND:
  Linear(ndim, 32) → sin → Linear(32, 32) → sin → Linear(32, 32) → sin → Linear(32, C)
```

**Forward Pass**:
```
Input x: (B, *spatial, C)
Target shape: (t1, t2, ...)

1. Compute kernel size: kernel_size[i] = spatial[i] // target[i]

2. Generate kernel coordinates:
   positions = meshgrid(linspace(-1, 1, k) for k in kernel_size)

3. Generate kernel weights via SIREN:
   kernel = kernel_net(positions * omega_0)  → (K, C) reshaped to (C, 1, *kernel_size)

4. Apply depthwise strided convolution:
   output = conv_nd(x, kernel, stride=kernel_size, groups=C)

5. Interpolate if needed to exact target shape
```

**Why it matters**: The kernel adapts to any downsampling ratio. Unlike fixed pooling, the network learns optimal spatial aggregation patterns.

---

### LocalAttentionND

**Purpose**: Windowed attention with learned, position-adaptive window shape.

**Key Insight**: Each position learns its own effective window width and sharpness, allowing the receptive field to vary spatially.

**Learned Parameters**:
- `window_proj`: Linear(C, 2H) → width, sharpness per head
- Standard Q/K/V/O projections with RMSNorm

**Window Masking**:
```
width = sigmoid(width_raw) * max_dist + 0.5      # [0.5, max_dist+0.5]
sharpness = sigmoid(sharpness_raw) * 9.5 + 0.5   # [0.5, 10.0]

soft_mask = sigmoid((width - rel_dist) * sharpness)
scores = scores - (1 - soft_mask) * 1e4  # soft masking
```

**Additional Features**:
- RoPE positional encoding (1D)
- V-gating: `merged = gate * v + (1 - gate) * attn_out`

---

### LowRankAttentionMergeND

**Purpose**: Efficiently merge two representations via attention in reduced space.

**Key Insight**: Downsample both inputs to sqrt(L) per dimension, compute full attention in reduced space, then upsample.

**Forward Pass** (standard merge):
```
Input embed: (B, *spatial, C)  - original embedding
Input processed: (B, *spatial, C)  - processed representation

1. Downsample both to (sqrt(s1), sqrt(s2), ...):
   embed_down = avg_pool(embed, target_shape)
   processed_down = avg_pool(processed, target_shape)

2. Cross-attention in reduced space:
   q = silu(q_proj(processed_down))
   k = silu(k_proj(embed_down))
   v = silu(v_proj(embed_down))
   out = scaled_dot_product_attention(q, k, v)

3. Upsample and residual:
   out = interpolate(out, spatial_shape)
   return processed + out
```

**Forward Pass** (accumulated - for cross-layer attention):
```
Input x_full: (B, *spatial, C)  - current layer's input
Input accumulated: list of reduced tensors from previous layers

1. Query from most recent reduced state:
   q = silu(q_proj(accumulated[-1]))

2. Keys/values from ALL previous reduced states:
   kv_cat = concat([t.reshape(B, r, C) for t in accumulated[:-1]])
   k = silu(k_proj(kv_cat))
   v = silu(v_proj(kv_cat))

3. Attend, project, upsample:
   out = scaled_dot_product_attention(q, k, v)
   out = interpolate(out, spatial_shape)
   return x_full + out
```

---

### SGSBAttentionND

**Purpose**: Multi-scale attention via Scatter-Gather-Scatter-Broadcast pattern.

**Architecture**:
```
┌─────────────────────────────────────────────────────┐
│                   SGSBAttentionND                   │
│                                                     │
│   Input h                                           │
│      │                                              │
│      ▼                                              │
│   ┌──────────────────┐                              │
│   │  AdaptiveConvND  │  SCATTER: adaptive sampling  │
│   └──────────────────┘                              │
│      │ + residual                                   │
│      ▼                                              │
│   ┌──────────────────┐                              │
│   │ LocalAttentionND │  GATHER: window attention    │
│   └──────────────────┘                              │
│      │ + residual                                   │
│      ▼                                              │
│   ┌──────────────────┐                              │
│   │  AdaptiveConvND  │  SCATTER: same module!       │
│   └──────────────────┘                              │
│      │ + residual                                   │
│      ▼                                              │
│   ┌──────────────────┐                              │
│   │ LowRankAttention │  BROADCAST: global context   │
│   └──────────────────┘                              │
│      │ + residual                                   │
│      ▼                                              │
│   out_proj → output                                 │
└─────────────────────────────────────────────────────┘
```

**Why this pattern**:
1. **First Scatter**: Adaptive sampling spreads local information outward
2. **Gather**: Window attention collects and integrates local neighborhoods  
3. **Second Scatter**: Further spreading with refined features (same weights = regularization)
4. **Broadcast**: Global context via low-rank attention (sqrt(L) complexity)

**Weight Sharing**: The two scatter operations share the same `AdaptiveConvND` module—this acts as implicit regularization and reduces parameters.

---

### MIMOJacobiSSM_ND

**Purpose**: Multi-Input Multi-Output State Space Model with Jacobi-style iterative refinement. **Shared at model level** — runs ONE iteration per layer, with depth providing the iteration count.

**Inspiration**: Mamba-3's MIMO formulation + Jacobi iterative solvers + RoPE-style state rotation.

**Key Design Change**: Instead of running K iterations internally, the SSM is shared across all layers. Each layer calls `ssm.step()` once, so **iterations = n_layers**. This lets the deep model naturally diffuse state through the SSM.

**State Dynamics**:
```
H ∈ (B, *spatial, N, R)  - hidden state with R ranks (persists across layers)
B_mat ∈ (B, *spatial, N, R)  - input projection matrix  
C_mat ∈ (B, *spatial, N, R)  - output projection matrix
X_r ∈ (B, *spatial, R)  - per-rank input gate
decay ∈ (B, *spatial, N)  - state decay
theta ∈ (B, *spatial, N//2)  - rotation angles (scaled by layer_idx)
```

**API**:
```python
# At model level:
H = ssm.init_state(x)  # zeros: (B, *spatial, N, R)

# Per layer (one Jacobi step):
H, ssm_out = ssm.step(x, H, layer_idx)
```

**Single Step** (`step` method):
```python
def step(self, x, H, layer_idx):
    # Input projections
    B_mat = silu(to_B(x)).reshape(B, *spatial, N, R)
    C_mat = silu(to_C(x)).reshape(B, *spatial, N, R)
    X_r = silu(to_X(x))
    decay = sigmoid(to_decay(x))
    theta = to_theta(x)
    
    # Layer-scaled rotation (deeper = more rotation)
    theta_k = theta * (layer_idx + 1)
    B_rot = apply_rope(B_mat, theta_k)
    inject = B_rot * X_r.unsqueeze(-2)
    
    # Spatial diffusion (all R ranks batched as B*R)
    H = H.permute(batch R to front)
    H = diffuse(H)  # AdaptiveConvND
    H = H.permute(back)
    
    # Jacobi update: decay + inject
    H = decay * H + inject
    
    # Output projection
    out = silu(out_proj(H.reshape(B, *spatial, N*R)))
    return H, out
```

**Key Optimizations**:
- Single `AdaptiveConvND` shared across all R ranks (batched as B×R)
- 4x parameter reduction vs R separate modules
- Layer-scaled theta provides different rotation per depth

**Why One Step Per Layer**: 
- Depth IS the iteration count — no hyperparameter to tune
- State naturally diffuses through the network
- Gradients flow cleanly (no internal loop to checkpoint)
- Each layer sees the SSM at a different "iteration" of convergence

---

### RippleLayerND

**Purpose**: Single layer combining attention with SSM output injection, cross-layer state accumulation, and **per-layer RNN-style hidden state**.

**Key Changes**:
- Layer does NOT contain its own SSM — receives `ssm_out` from model-level shared SSM
- Each layer has its own hidden state that accumulates within a forward pass (resets between forwards)
- Single attention call (not two)

**Architecture**:
```python
def init_state(self, x):
    return torch.zeros_like(x)  # per-layer hidden state

def forward(self, x, embed, ssm_out, accumulated, layer_state):
    # Cross-layer attention (if we have history from previous layers)
    if len(accumulated) > 1:
        x = accum_attn.forward_accumulated(x, accumulated)
    
    # SGSB attention
    x = x + attn(norm1(x))
    
    # Add SSM output (from shared model-level SSM)
    x = x + ssm_out
    
    # Merge with original embedding
    x = merge(embed, x)
    
    # RNN-style state accumulation (gated)
    gate = sigmoid(state_gate(x))
    layer_state = state_norm(layer_state + x)  # accumulate
    x = gate * layer_state + (1 - gate) * x    # gated mix
    
    # Reduce for next layer's accumulated attention
    spatial_shape = x.shape[1:-1]
    target_shape = tuple(int(s ** 0.5) for s in spatial_shape)
    x_reduced = downsample(reduce_norm(x), target_shape)
    
    return x, x_reduced, layer_state
```

**Per-Layer State**: Each layer maintains its own hidden state that:
- Starts at zero for each forward pass
- Accumulates information via normalized residual: `state = norm(state + x)`
- Gets mixed into output via learned gate (initialized to favor fresh computation)
- Resets between forward calls (not persistent like true RNN)

**Accumulated State**: Each layer produces a sqrt(L)-reduced representation appended to the accumulated list. Future layers attend to ALL previous layers' reduced states via `forward_accumulated`.

---

## Full Model

### RippleModelND / RippleClassifierND

```python
def forward(self, x):
    embed = embed_norm(silu(embed_proj(x)))
    
    x = embed
    
    # Initialize shared SSM state (model-level)
    H = ssm.init_state(x)
    
    # Initialize per-layer RNN states
    layer_states = [layer.init_state(x) for layer in layers]
    
    accumulated = []
    
    for i, layer in enumerate(layers):
        # One SSM step per layer (shared SSM)
        H, ssm_out = ssm.step(ssm_norm(x), H, i)
        
        # Layer forward with SSM output and layer state
        x, x_reduced, layer_states[i] = layer(x, embed, ssm_out, accumulated, layer_states[i])
        accumulated.append(x_reduced)
    
    # Pool spatial dims and classify
    x = out_norm(x).mean(spatial_dims)
    return head(x)
```

**Three State Streams**:
1. **SSM State (H)**: Shared across all layers, one Jacobi step per layer. Provides global recurrence.
2. **Layer States**: Per-layer RNN-style accumulation within forward pass. Each layer has its own state.
3. **Accumulated Reduced**: Cross-layer attention to all previous layers' downsampled representations.

**Information Flow**:
```
                    ┌─────────────────────────────────────────────────────┐
                    │                    SSM State H                       │
                    │  H₀ ──step──> H₁ ──step──> H₂ ──step──> H₃ ...     │
                    └─────────────────────────────────────────────────────┘
                           ↓            ↓            ↓
Layer 0: x₀ + ssm_out₀ ─────────────────────────────────────> x₁, r₀, s₀
Layer 1: x₁ + ssm_out₁ + attend(r₀) ────────────────────────> x₂, r₁, s₁  
Layer 2: x₂ + ssm_out₂ + attend(r₁ → [r₀]) ─────────────────> x₃, r₂, s₂
Layer 3: x₃ + ssm_out₃ + attend(r₂ → [r₀, r₁]) ─────────────> x₄, r₃, s₃
                    └──────────────────────────────────────────────────────┘
                                 Per-layer state sᵢ accumulates within forward
```

- **H**: Global SSM state diffuses through the network (one step per layer)
- **rᵢ**: Reduced representations enable cross-layer attention
- **sᵢ**: Per-layer states accumulate local information within the forward pass

---

## Key Innovations

### 1. Adaptive Spatial Sampling
`AdaptiveConvND` learns WHERE to sample based on input content—frequency, phase, and decay are all data-dependent. This is fundamentally different from fixed convolution kernels.

### 2. SIREN-Generated Kernels  
`SIRENDownsampleND` generates convolution kernels on-the-fly via a coordinate MLP, enabling smooth adaptation to any downsampling ratio.

### 3. Scatter-Gather-Scatter-Broadcast Pattern
SGSB combines local spreading (scatter), local integration (gather), and global context (broadcast) in a single attention block with shared weights.

### 4. Shared SSM with Depth-as-Iterations
Instead of running K iterations internally, the MIMO Jacobi SSM is **shared at the model level**:
- One SSM step per layer → iterations = n_layers
- Layer-scaled rotation angles (deeper = more rotation)
- No iteration hyperparameter to tune
- Gradients flow cleanly without internal checkpointing

### 5. Per-Layer RNN-Style State
Each `RippleLayerND` maintains its own hidden state within a forward pass:
- Accumulates via normalized residual: `state = norm(state + x)`
- Gated mixing with output (learned gate, initialized conservative)
- Resets between forward calls (not persistent)
- Provides local temporal-like accumulation complementing the global SSM

### 6. Cross-Layer State Accumulation
Each layer produces a sqrt(L)-reduced state. Future layers attend to ALL previous reduced states, creating dense information flow with O(sqrt(L)) memory per layer.

### 7. Weight Sharing
- SGSBAttentionND: single scatter module called twice
- MIMOJacobiSSM: single diffuse module for all R ranks (batched as B×R)
- SSM shared across all layers (not per-layer instances)

This reduces parameters while acting as implicit regularization.

---

## Complexity Analysis

| Component | Time | Space |
|-----------|------|-------|
| AdaptiveConvND | O(L × S) | O(L × S) |
| LocalAttentionND | O(L × K) | O(L × K) |
| LowRankAttention | O(r²) where r = √L | O(r²) |
| MIMOJacobiSSM (per step) | O(L × N × R) | O(L × N × R) |
| Cross-layer attn | O(L_layers × r²) | O(L_layers × r) |
| Per-layer state | O(L) | O(L) |

Where:
- L = sequence length (product of spatial dims)
- S = number of samples in AdaptiveConvND
- K = window size for LocalAttentionND
- N = state dim, R = MIMO rank
- r = √L (reduced dimension)

Total per layer: **O(L × max(S, K, N×R))** — linear in sequence length.

Note: SSM iterations are now implicit (one step per layer), so total SSM cost across model = O(n_layers × L × N × R).
