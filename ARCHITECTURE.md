# Unified Adaptive Binary Model Architecture

A deep dive into the unified architecture for byte-level secret detection, combining two-phase attention with adaptive multi-scale convolutions and state space models.

## Executive Summary

The `UnifiedBinaryModel` processes raw byte sequences through a novel two-phase attention pipeline:

1. **Phase 1: ContextualAttentionBlock** - discovers short-medium range patterns, generates per-branch biases for adaptive convolutions
2. **Parallel AdaptiveConvBranches** - multi-scale feature extraction with different receptive fields
3. **Phase 2: Merge Attention** - attends to all feature extractions in context of residuals
4. **Single SSM** - temporal/sequential modeling over the merged, attended features
5. **Classification** - pooling + gated MLP + binary head

The key innovations:
- **Two-phase attention**: Phase 1 pre-biases extraction, Phase 2 merges all features contextually
- **Parallel convolutions, single SSM**: Multi-scale convs feed into ONE SSM (not parallel SSMs)
- **Residual-biased merge**: Residuals from Phase 1 inform Phase 2's attention

```
Raw Bytes [B, L]
    │
    ▼
ByteEmbedding + Dropout
    │
    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                 PHASE 1: ContextualAttentionBlock                             ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  Self-Attention + SwiGLU FFN                                            │  ║
║  │  • Discovers short-medium range patterns not found by convolutions      │  ║
║  │  • Pre-biases downstream feature extraction                             │  ║
║  │                                                                         │  ║
║  │  Outputs:                                                               │  ║
║  │  • h: processed features [B, L, W] ─────────────────────────────────┐   │  ║
║  │  • biases: (σ, offset_scale, ω) per branch [B, n_branches]          │   │  ║
║  │  • pooler_context: [B, context_dim]                                 │   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                    │                                                    │
                    │ biases                                             │ residual (h)
                    ▼                                                    │
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    MULTI-SCALE ADAPTIVE CONVOLUTIONS                          ║
║  ┌───────────────────────────────────────────────────────────────────────┐    ║
║  │  in_proj: [B, L, W] → [B, L, total_width]                             │    ║
║  │  split into n_branches chunks                                         │    ║
║  │      │                                                                │    ║
║  │      ├──→ AdaptiveConvBranch(σ=0.05)  ◄── biases[:, 0]                │    ║
║  │      ├──→ AdaptiveConvBranch(σ=0.275) ◄── biases[:, 1]                │    ║
║  │      └──→ AdaptiveConvBranch(σ=0.50)  ◄── biases[:, 2]                │    ║
║  │                  │                                                    │    ║
║  │                  ▼                                                    │    ║
║  │          concat: [B, L, total_width]                                  │    ║
║  │                  │                                                    │    ║
║  │                  ▼                                                    │    ║
║  │          proj_down: [B, L, W]                                         │    ║
║  └───────────────────────────────────────────────────────────────────────┘    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                    │                                                    │
                    ▼                                                    │
              + residual ◄───────────────────────────────────────────────┘
                    │
                    ▼
                RMSNorm
                    │
                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                 PHASE 2: Merge Attention                                      ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  AttentionBlock (Self-Attention + SwiGLU FFN)                           │  ║
║  │  • Attends to ALL feature extractions (conv features + attn features)   │  ║
║  │  • Context-aware via residual from Phase 1                              │  ║
║  │  • Decides "what matters" from all extraction methods                   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                    │
                    ▼
                RMSNorm
                    │
                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                 SINGLE SSM (SSMMixer3, use_conv=False)                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │  State Space Model with:                                                │  ║
║  │  • Trapezoidal discretization                                           │  ║
║  │  • Data-dependent RoPE                                                  │  ║
║  │  • QK-norm on B/C projections                                           │  ║
║  │  • Gated output                                                         │  ║
║  │                                                                         │  ║
║  │  Takes the attended, merged features and models temporal dependencies   │  ║
║  └─────────────────────────────────────────────────────────────────────────┘  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                    │
                    ▼
                RMSNorm
                    │
                    ▼
        Learned Query Pooling (context-gated)
                    │
                    ▼
              feature_proj → ssm_features [B, num_features]
                    │
                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         Feature Aggregation                                   ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               ║
║  │  attn_features  │  │  ssm_features   │  │ embed_features  │               ║
║  │  [B, 4]         │  │  [B, 4]         │  │ [B, 4]          │               ║
║  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘               ║
║           │                    │                    │                         ║
║           └────────────┬───────┴────────────────────┘                         ║
║                        │                                                      ║
║                        ▼                                                      ║
║                 concat → GatedMLP → BinaryHead → logit [B]                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 1. ByteEmbedding

Converts raw bytes to dense vectors:

```python
byte (0-255) → Embedding(256, width) → Dropout → [B, L, width]
```

---

## 2. Phase 1: ContextualAttentionBlock

The first attention phase serves dual purposes:
1. **Sequence processing** - discovers short-medium range patterns via self-attention + FFN
2. **Bias generation** - produces per-branch biases for downstream adaptive convolutions

### Architecture

```python
class ContextualAttentionBlock(nn.Module):
    def __init__(self, width, num_heads, n_branches, context_dim, ffn_mult, dropout):
        # Standard attention components
        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.norm2 = RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_mult, dropout)
        
        # Pooling for bias generation
        self.pool_query = nn.Parameter(torch.randn(1, width))
        
        # Bias projections (zero-initialized for stable training)
        self.sigma_proj = nn.Linear(width, n_branches)       # Envelope width
        self.offset_scale_proj = nn.Linear(width, n_branches) # Offset magnitude
        self.omega_proj = nn.Linear(width, n_branches)       # SIREN frequency
        self.context_proj = nn.Linear(width, context_dim)    # Pooler context
```

### Bias Semantics

| Bias | Shape | Controls | Effect |
|------|-------|----------|--------|
| `sigma` | [B, n_branches] | Gaussian envelope σ | Larger = wider receptive field |
| `offset_scale` | [B, n_branches] | Deformable offset magnitude | Larger = more position shifting |
| `omega` | [B, n_branches] | KernelNet SIREN ω₀ | Larger = higher-frequency kernels |

All biases are **zero-initialized** so the model starts with default behavior and learns to modulate.

---

## 3. AdaptiveConvBranch

Lightweight wrapper around AdaptiveDeformConv1d:

```python
class AdaptiveConvBranch(nn.Module):
    def __init__(self, width, init_sigma, min_sigma, max_sigma, groups, kernel_size):
        self.norm = RMSNorm(width)
        self.conv = AdaptiveDeformConv1d(
            width, kernel_size=kernel_size, groups=groups,
            init_sigma=init_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
        )
    
    def forward(self, x, mask=None, biases=None):
        h = self.norm(x)
        out, aux = self.conv(h, mask, biases)
        return F.silu(out), aux
```

Each branch has a different initial σ, creating a multi-scale feature extraction bank.

---

## 4. AdaptiveDeformConv1d

The core convolution combining **learned continuous kernels**, **deformable sampling**, **attention masking**, and **adaptive envelope width**.

### Components

```
Input [B, L, C]
    │
    ├──→ input_proj ──→ x_proj [B, L, C]  (value projection)
    │
    └──→ dw_conv ──→ x_dw [B, L, C]  (context for offset/mask prediction)
           │
           ├──→ offset_net ──→ offsets [B, L, G, K]  (WHERE to sample)
           │
           └──→ mask_net ──→ raw_mask [B, L, G, K]  (HOW MUCH to weight)
                   │
                   ▼
            × Gaussian envelope (from sigma bias)
                   │
                   ▼
            softmax → attention mask [B, L, G, K]

KernelNet (SIREN):
    grid [-0.5, 0.5] ──→ kernel_net ──→ kernel_weights [G, gc, K]
                          (omega bias modulates frequency)

Sampling:
    for k in range(K):
        abs_pos = pos + ref_offset[k] + learned_offset[:,:,:,k]
        sampled = bilinear_interpolate(x_proj, abs_pos)
        sampled = sampled * kernel_weights[:, :, k]
        output += sampled * attn_mask[:,:,:,k]

    output ──→ SE block ──→ output_proj ──→ [B, L, C]
```

### Dynamic Kernel Sizing

Kernel size adapts to σ using the 3σ rule:

```python
def _get_effective_kernel_size(self, sigma):
    sigma_val = sigma.mean().item()
    raw_k = int(6 * sigma_val / self.max_sigma * self.max_kernel_size)
    raw_k = max(self.min_kernel_size, min(self.max_kernel_size, raw_k))
    return raw_k // 2 * 2 + 1  # Ensure odd
```

- Small σ (0.05) → small kernel (fewer positions needed)
- Large σ (0.5) → large kernel (wider receptive field)

### KernelNet (SIREN)

Generates continuous kernel weights for any position:

```python
class KernelNet1d(nn.Module):
    def __init__(self, out_channels, hidden_channels=32, num_layers=3, omega_0=30.0):
        layers = [SIRENLayer(1, hidden_channels, omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_channels, hidden_channels, omega_0))
        self.net = nn.Sequential(*layers)
        self.output_linear = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
```

Key insight: Fixed parameters but can generate kernels of **any size** by varying input positions.

### Regularization Losses

The conv returns auxiliary losses for training:

```python
offset_reg = (offsets ** 2).mean()  # Penalize large offsets
entropy = -(attn_mask * (attn_mask + 1e-8).log()).sum(dim=-1).mean()  # Encourage spread
aux_losses = {'offset_reg': offset_reg, 'entropy_reg': -entropy}
```

---

## 5. MultiKernelSSMBlock

The refactored block that implements: **parallel convs → merge → attention → single SSM**.

### Architecture

```python
class MultiKernelSSMBlock(nn.Module):
    def __init__(self, width, init_sigmas, state_size, n_heads, ...):
        # Input projection
        self.in_norm = RMSNorm(width)
        self.in_proj = nn.Linear(width, total_width)  # total_width = n_branches * branch_width
        
        # Parallel adaptive conv branches (NOT SSMs)
        self.branches = nn.ModuleList([
            AdaptiveConvBranch(branch_width, init_sigma=sigma, ...)
            for sigma in init_sigmas
        ])
        
        # Merge: conv outputs → residual size
        self.proj_down = nn.Linear(total_width, width)
        self.norm_merge = RMSNorm(width)
        
        # Phase 2: Merge attention
        self.merge_attn = AttentionBlock(width, n_heads, ffn_mult=4, dropout=dropout)
        self.norm_attn = RMSNorm(width)
        
        # Single SSM (no conv inside)
        self.ssm = SSMMixer3(width, state_size, n_heads, expand, dropout, use_conv=False)
        self.norm_ssm = RMSNorm(width)
        
        # Pooling for features
        self.pool_queries = nn.Parameter(torch.randn(1, width))
        self.context_gate = nn.Linear(context_dim, width)  # Optional context bias
        self.feature_proj = nn.Sequential(nn.Linear(width, num_features), nn.SiLU())
```

### Forward Pass

```python
def forward(self, x, residual, mask=None, biases=None, pooler_context=None):
    B, L, _ = x.shape
    
    # 1. Project and split for parallel convs
    h = self.in_proj(self.in_norm(x))
    chunks = h.chunk(self.n_branches, dim=-1)
    
    # 2. Parallel adaptive convolutions
    branch_outputs = []
    all_aux_losses = []
    for i, (branch, chunk) in enumerate(zip(self.branches, chunks)):
        if biases is not None:
            branch_bias = AdaptiveConvBiases(
                sigma=biases.sigma[:, i:i+1].squeeze(-1),
                offset_scale=biases.offset_scale[:, i:i+1].squeeze(-1),
                omega=biases.omega[:, i:i+1].squeeze(-1),
            )
            out, aux = branch(chunk, mask, branch_bias)
        else:
            out, aux = branch(chunk, mask)
        branch_outputs.append(out)
        all_aux_losses.append(aux)
    
    # 3. Merge conv outputs + residual
    combined = torch.cat(branch_outputs, dim=-1)
    h = self.proj_down(combined)
    h = self.norm_merge(h + residual)  # Residual from Phase 1 attention
    
    # 4. Phase 2: Merge attention
    h = self.merge_attn(h, mask)
    h = self.norm_attn(h)
    
    # 5. Single SSM
    h, ssm_aux = self.ssm(h, mask)
    h = self.norm_ssm(h)
    
    # 6. Context-gated pooling
    q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
    if pooler_context is not None:
        q = q + self.context_gate(pooler_context).unsqueeze(1)
    
    attn = torch.bmm(q, h.transpose(1, 2)) * self.pool_scale
    if mask is not None:
        attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
    pooled = torch.bmm(F.softmax(attn, dim=-1), h).squeeze(1)
    features = self.feature_proj(pooled)
    
    # Aggregate aux losses
    aux_losses = {}
    for key in all_aux_losses[0]:
        aux_losses[key] = sum(d[key] for d in all_aux_losses) / len(all_aux_losses)
    for k, v in ssm_aux.items():
        aux_losses[f'ssm_{k}'] = v
    
    return self.dropout(h), features, aux_losses
```

### Key Design: Parallel Convs → Single SSM

**Old design (incorrect):**
```
Input → [Conv + SSM₀] → out₀
      → [Conv + SSM₁] → out₁  → concat → pool
      → [Conv + SSM₂] → out₂
```
Each branch had its own SSM - they couldn't share temporal information across scales.

**New design (correct):**
```
Input → [Conv₀(σ=0.05)]  →
      → [Conv₁(σ=0.275)] → concat → proj_down → +residual → Attention → SSM → pool
      → [Conv₂(σ=0.50)]  →
```
Multi-scale convolutions extract features, then ONE SSM sees all scales together.

---

## 6. SSMMixer3 (Mamba-3 Style)

The SSM component with advanced features:

### Core Structure

```python
class SSMMixer3(nn.Module):
    def __init__(self, width, state_size=64, n_heads=4, expand=2, use_conv=False, ...):
        self.intermediate_size = width * expand
        
        # Input projection with gating
        self.in_proj = nn.Linear(width, intermediate_size * 2)
        
        # Optional conv (disabled in MultiKernelSSMBlock)
        if use_conv:
            self.conv = AdaptiveDeformConv1d(intermediate_size, ...)
        
        # SSM parameters
        self.B_proj = nn.Linear(intermediate_size, n_heads * state_size)
        self.C_proj = nn.Linear(intermediate_size, n_heads * state_size)
        self.b_bias = nn.Parameter(torch.ones(n_heads, state_size))
        self.c_bias = nn.Parameter(torch.ones(n_heads, state_size))
        self.norm_b = RMSNorm(state_size)
        self.norm_c = RMSNorm(state_size)
        
        # Discretization
        self.dt_proj = nn.Linear(intermediate_size, n_heads)
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        
        # Data-dependent RoPE
        self.theta_proj = nn.Linear(intermediate_size, n_heads * (state_size // 2))
        
        # Trapezoidal discretization
        self.lambda_proj = nn.Linear(intermediate_size, n_heads)
        
        # Skip and output
        self.D = nn.Parameter(torch.ones(n_heads, head_dim))
        self.out_proj = nn.Linear(intermediate_size, width)
```

### Trapezoidal Discretization

Standard Euler: `h[t] = A·h[t-1] + B·x[t]`

Trapezoidal (combines current and previous):
```
h[t] = α·h[t-1] + γ·B·x[t] + β·B·x[t-1]

where:
  α = exp(dt·A)       # Decay factor
  γ = λ·dt            # Current input weight
  β = (1-λ)·dt·α      # Lookback coefficient
```

The λ parameter interpolates between Euler (λ=1) and trapezoidal (λ=0.5).

### Data-Dependent RoPE

Instead of fixed positional frequencies, RoPE angles are content-dependent:

```python
theta = self.theta_proj(x_ssm).view(B, L, H, N//2)
delta_theta = dt.unsqueeze(-1) * theta
cum_theta = torch.cumsum(delta_theta, dim=1)
B_rot = apply_rotary_emb(B_param, cum_theta)
C_rot = apply_rotary_emb(C_param, cum_theta)
```

---

## 7. Training Configuration

### Weight Decay Groups

Parameters are split into decay/no-decay groups:

```python
no_decay_keywords = {
    'bias', 'LayerNorm.weight', 'norm.weight',
    'log_sigma', 'A_log', '.D',
    'b_bias', 'c_bias',
    'pool_query', 'pool_queries',
    'lambda_proj.bias', 'dt_proj.bias',
}
```

These parameters are excluded from weight decay to allow free learning of:
- Envelope widths (log_sigma)
- SSM state decay rates (A_log)
- Skip connection strengths (D)
- Discretization parameters

### Regularization

```python
TrainConfig(
    weight_decay=0.01,        # Applied to non-excluded params
    offset_reg_weight=0.01,   # Penalize large deformable offsets
    entropy_reg_weight=0.001, # Encourage attention spread
)
```

### Training Command

```bash
python scripts/train_line_filter.py --adaptive --n-branches 3 --epochs 20
```

---

## 8. RMSNorm Strategy

RMSNorm is applied at key transition points:

| Location | Purpose |
|----------|---------|
| Before in_proj | Normalize input to conv branches |
| After residual add | Stabilize merged conv + attention features |
| After merge attention | Prepare for SSM |
| After SSM | Prepare for pooling |
| Inside AttentionBlock | Pre-norm for self-attention and FFN |
| Inside AdaptiveConvBranch | Pre-norm for conv |

---

## 9. Design Rationale

### Why Two-Phase Attention?

1. **Phase 1 pre-biases extraction** - learns what scales/patterns to focus on
2. **Phase 2 merges intelligently** - sees all extractions in context of Phase 1 features
3. **Separation of concerns** - extraction bias vs. feature selection

### Why Parallel Convs → Single SSM?

1. **Multi-scale convolutions** capture different receptive field patterns
2. **Single SSM** models temporal dependencies across ALL scales simultaneously
3. **Information flow** - scales can interact through the shared SSM state

### Why Residual into Merge?

1. **Context preservation** - Phase 1 attention features inform Phase 2
2. **Gradient flow** - direct path from early attention to late stages
3. **Feature mixing** - conv features blend with attention features

### Why Dynamic Kernel Sizing?

1. **Efficiency** - small σ doesn't need large kernel
2. **Matched receptive field** - kernel size tracks envelope width
3. **KernelNet flexibility** - continuous generation handles any size

### Why KernelNet (SIREN)?

1. **Continuous kernel generation** - resize without retraining
2. **Implicit regularization** - network smoothness constrains kernel
3. **Frequency modulation** - ω₀ bias enables content-dependent frequency

---

## 10. Model Configuration

```python
ModelConfig(
    task="binary",
    arch_type="unified",
    embed_width=48,
    dropout=0.1,
    
    # Phase 1 Attention
    attn_heads=4,
    attn_ffn_mult=4,
    num_attn_features=4,
    
    # SSM (single, after merge attention)
    ssm_state_size=64,
    ssm_n_heads=4,
    ssm_expand=2,
    num_ssm_features=4,
    
    # Adaptive convolutions
    adaptive_conv=True,
    n_adaptive_branches=3,
    adaptive_kernel_size=15,
    adaptive_init_sigmas=None,  # Auto: (0.05, 0.275, 0.5)
    adaptive_min_sigma=0.05,
    adaptive_max_sigma=0.5,
    context_dim=16,
    
    # Feature aggregation
    num_embed_features=4,
    num_hidden_features=8,
    num_heuristic_features=4,
    mlp_hidden_mult=4,
    mlp_output_dim=32,
)
```

When `adaptive_init_sigmas=None`, sigmas are auto-generated linearly:
```python
# For n=3: (0.05, 0.275, 0.5)
init_sigmas = tuple(
    min_sigma + (max_sigma - min_sigma) * i / (n - 1)
    for i in range(n)
)
```

---

## 11. Data Flow Summary

```
Input bytes [B, L]
    │
    ▼
Embedding [B, L, 48]
    │
    ▼
ContextualAttentionBlock ──► biases [B, 3] each
    │                    ──► pooler_context [B, 16]
    │
    ▼ h [B, L, 48] (also used as residual)
    │
    ▼
in_proj [B, L, 84] (3 branches × 28 width)
    │
    ├─► AdaptiveConvBranch(σ=0.05)  [B, L, 28]
    ├─► AdaptiveConvBranch(σ=0.275) [B, L, 28]
    └─► AdaptiveConvBranch(σ=0.50)  [B, L, 28]
            │
            ▼
        concat [B, L, 84]
            │
            ▼
        proj_down [B, L, 48]
            │
            ▼
        + residual (h from attention)
            │
            ▼
        RMSNorm
            │
            ▼
        AttentionBlock (merge attention)
            │
            ▼
        RMSNorm
            │
            ▼
        SSMMixer3 (use_conv=False)
            │
            ▼
        RMSNorm
            │
            ▼
        Learned Query Pooling [B, 48]
            │
            ▼
        feature_proj [B, 4] (ssm_features)
            │
            ▼
        + attn_features [B, 4]
        + embed_features [B, 4]
        + hidden_features [B, 8]
        + heuristic_features [B, 4]
            │
            ▼
        concat [B, 24]
            │
            ▼
        GatedMLP [B, 32]
            │
            ▼
        BinaryHead [B]
```
