# Unified Adaptive Binary Model Architecture

A deep dive into the unified architecture for byte-level secret detection, combining adaptive multi-scale convolutions with state space models.

## Executive Summary

The `UnifiedModel` (default configuration for `train_line_filter.py --adaptive`) processes raw byte sequences through a streamlined pipeline:

1. **BiasProjection** - SwiGLU MLP that mean-pools embeddings to generate per-branch biases (sigma, offset_scale, omega, se_bias)
2. **Parallel AdaptiveConvBranches** - multi-scale feature extraction with different receptive fields
3. **Single SSM (Mamba-3 style)** - temporal/sequential modeling over the merged features
4. **Feature aggregation** - combines SSM, embedding, hidden, and heuristic features
5. **Classification** - GatedMLP + binary head

The key design:
- **No attention** (default): BiasProjection replaces ContextualAttentionBlock for faster training
- **SwiGLU everywhere**: BiasProjection uses SwiGLU MLP; depthwise mode adds per-channel SwiGLU projections
- **Parallel convolutions, single SSM**: Multi-scale convs feed into ONE SSM
- **Precomputed heuristic features**: char_frequency_difference provides domain knowledge

```
Raw Bytes [B, L]
    |
    v
ByteEmbedding + Dropout [B, L, 8]
    |
    v
+==============================================================================+
|                     BiasProjection (use_attention=False)                     |
|  +------------------------------------------------------------------------+  |
|  |  Mean pooling over sequence -> pooled [B, 8]                           |  |
|  |  SwiGLU MLP -> h [B, 8]                                                 |  |
|  |                                                                        |  |
|  |  Outputs:                                                              |  |
|  |  - biases: (sigma, offset_scale, omega, se_bias) per branch [B, 3]     |  |
|  |  - biases.context: h [B, 8] (for depthwise per-channel SwiGLUs)        |  |
|  |  - pooler_context: [B, 16]                                             |  |
|  +------------------------------------------------------------------------+  |
+==============================================================================+
                    |                                                    
                    | biases                                             
                    v                                                    
+==============================================================================+
|                    MULTI-SCALE ADAPTIVE CONVOLUTIONS                         |
|  +------------------------------------------------------------------------+  |
|  |  in_proj: [B, L, 8] -> [B, L, total_width]                             |  |
|  |  split into 3 branches                                                 |  |
|  |      |                                                                 |  |
|  |      +---> AdaptiveConvBranch(sigma=0.05)  <-- biases[:, 0]            |  |
|  |      +---> AdaptiveConvBranch(sigma=0.275) <-- biases[:, 1]            |  |
|  |      +---> AdaptiveConvBranch(sigma=0.50)  <-- biases[:, 2]            |  |
|  |                  |                                                     |  |
|  |                  v                                                     |  |
|  |          concat: [B, L, total_width]                                   |  |
|  |                  |                                                     |  |
|  |                  v                                                     |  |
|  |          proj_down: [B, L, 8]                                          |  |
|  +------------------------------------------------------------------------+  |
+==============================================================================+
                    |
                    v
                RMSNorm
                    |
                    v
+==============================================================================+
|                 SINGLE SSM (SSMMixer3, use_conv=False)                       |
|  +------------------------------------------------------------------------+  |
|  |  State Space Model with:                                               |  |
|  |  - Trapezoidal discretization                                          |  |
|  |  - Data-dependent RoPE                                                 |  |
|  |  - QK-norm on B/C projections                                          |  |
|  |  - Gated output                                                        |  |
|  |                                                                        |  |
|  |  Takes the merged features and models temporal dependencies            |  |
|  +------------------------------------------------------------------------+  |
+==============================================================================+
                    |
                    v
                RMSNorm
                    |
                    v
        Learned Query Pooling
                    |
                    v
              feature_proj -> ssm_features [B, 16]
                    |
                    v
+==============================================================================+
|                         Feature Aggregation                                  |
|  +-----------------+  +-----------------+  +-----------------+               |
|  |  ssm_features   |  | embed_features  |  | hidden_features |               |
|  |  [B, 16]        |  | [B, 16]         |  | [B, 16]         |               |
|  +--------+--------+  +--------+--------+  +--------+--------+               |
|           |                    |                    |                        |
|           +----------+---------+---------+----------+                        |
|                      |                   |                                   |
|                      v                   v                                   |
|              heuristic_features [B, 16]                                      |
|              (from precomputed char_frequency_difference)                    |
|                      |                                                       |
|                      v                                                       |
|              concat -> [B, 64]                                               |
|                      |                                                       |
|                      v                                                       |
|              GatedMLP -> [B, 128]                                            |
|                      |                                                       |
|                      v                                                       |
|              ClassifierHead -> logit [B]                                     |
+==============================================================================+
```

---

## 1. ByteEmbedding

Converts raw bytes to dense vectors:

```python
byte (0-255) -> Embedding(256, 8) -> Dropout(0.1) -> [B, L, 8]
```

---

## 2. BiasProjection (No-Attention Mode)

When `use_attention=False`, BiasProjection generates biases without attention overhead using a **SwiGLU MLP**:

### Architecture

```python
class BiasProjection(nn.Module):
    def __init__(self, width, n_branches, context_dim=0, bias_ffn_mult=2, dropout=0.1):
        # SwiGLU MLP for shared representation
        self.bias_mlp = SwiGLU(width, bias_ffn_mult, dropout, output_dim=width)
        
        # Bias heads (zero-initialized for stable training)
        self.sigma_head = nn.Linear(width, n_branches)       # Envelope width
        self.offset_scale_head = nn.Linear(width, n_branches) # Offset magnitude
        self.omega_head = nn.Linear(width, n_branches)       # SIREN frequency
        self.se_bias_head = nn.Linear(width, n_branches * width)  # SE block bias
        self.context_head = nn.Linear(width, context_dim) if context_dim > 0 else None
```

### SwiGLU MLP

The bias MLP uses SwiGLU gating for better gradient flow:

```python
class SwiGLU(nn.Module):
    def __init__(self, width, ffn_mult=4, dropout=0.1, output_dim=None):
        output_dim = output_dim or width
        hidden = int(width * ffn_mult)
        self.w1 = nn.Linear(width, hidden)      # Value path
        self.w2 = nn.Linear(width, hidden)      # Gate path
        self.w3 = nn.Linear(hidden, output_dim) # Output projection
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))
```

### Forward Pass

```python
def forward(self, x, mask):
    # Mean pool (with mask handling)
    pooled = x.mean(dim=1)  # [B, width]
    
    # SwiGLU MLP for shared representation
    h = self.bias_mlp(pooled)  # [B, width]
    
    biases = AdaptiveConvBiases(
        sigma=self.sigma_head(h),           # [B, n_branches]
        offset_scale=self.offset_scale_head(h),
        omega=self.omega_head(h),
        se_bias=self.se_bias_head(h).view(B, n_branches, width),
        context=h,  # Pass representation for depthwise per-channel projections
    )
    context = self.context_head(h) if self.context_head else None  # [B, context_dim]
    return biases, context
```

The `bias_ffn_mult` parameter (default: 2) controls the expansion ratio in the SwiGLU hidden layer.

### Bias Semantics

| Bias | Shape | Controls | Effect |
|------|-------|----------|--------|
| `sigma` | [B, 3] | Gaussian envelope sigma | Larger = wider receptive field |
| `offset_scale` | [B, 3] | Deformable offset magnitude | Larger = more position shifting |
| `omega` | [B, 3] | KernelNet SIREN omega_0 | Larger = higher-frequency kernels |
| `se_bias` | [B, 3, 8] | SE block output bias | Content-dependent channel scaling |

All biases are **zero-initialized** so the model starts with default behavior and learns to modulate.

### How Biases Modulate Behavior (Per-Sample)

All biases are applied **per-sample** (no `.mean()` collapsing):

**Sigma modulation:**
```python
raw = self.raw_sigma + sigma_bias  # sigma_bias: [N] or [N, C] for depthwise
sigma = softplus(raw).clamp(min=1e-3, max=max_sigma)  # [N] or [N, C]
```

**Offset scale modulation:**
```python
# offset_scale_bias: [N] -> broadcast to [N, 1, 1, 1] for offsets [N, L, G, K]
per_sample_scale = base_offset_scale * (1.0 + 0.2 * offset_scale_bias.view(-1, 1, 1, 1))
offsets = offsets * per_sample_scale
```
The 0.2 multiplier limits the modulation range to ±20% of base.

**Omega modulation (KernelNet frequency):**
```python
# omega_bias: [N] -> per-sample kernel generation
omega_scale = 1.0 + tanh(omega_bias).view(N, 1, 1) * omega_max  # omega_max=2.0
positions = positions.expand(N, 1, K) * omega_scale
kernel_weights = kernel_net(positions)  # [N, C, K] - per-sample kernels
```
Each sample gets its own scaled kernel weights.

**SE bias modulation:**
```python
# Inside SEBlock.forward():
w = fc2(silu(fc1(pooled)))  # Standard SE computation
if bias is not None:
    w = w + bias  # Add pre-computed bias BEFORE sigmoid
w = sigmoid(w)
return x * w.unsqueeze(1)
```
The SE bias shifts channel importance based on sequence content.

---

## 3. AdaptiveConvBranch

Wrapper around AdaptiveDeformConv1d with SE block for channel recalibration:

```python
class AdaptiveConvBranch(nn.Module):
    def __init__(self, width, init_sigma, min_sigma, max_sigma, groups, kernel_size, se_bias_dim, depthwise=False):
        self.norm = RMSNorm(width)
        self.conv = AdaptiveDeformConv1d(..., depthwise=depthwise)
        self.se = SEBlock(width)  # SE block with pre-bias from BiasProjection
        if se_bias_dim != width:
            self.se_bias_proj = nn.Linear(se_bias_dim, width)
    
    def forward(self, x, mask=None, biases=None):
        out, aux = self.conv(x, mask, biases)
        out = self.norm(silu(out))
        
        # SE with pre-computed bias from BiasProjection
        se_bias = None
        if biases is not None and biases.se_bias is not None:
            se_bias = biases.se_bias
            if self.se_bias_proj is not None:
                se_bias = self.se_bias_proj(se_bias)
        out = self.se(out, mask, se_bias)
        return out, aux
```

Each branch has a different initial sigma, creating a multi-scale feature extraction bank:
- Branch 0: sigma=0.05 (narrow, local patterns)
- Branch 1: sigma=0.275 (medium receptive field)
- Branch 2: sigma=0.50 (wide, global patterns)

---

## 4. AdaptiveDeformConv1d

The core convolution combining **learned continuous kernels**, **deformable sampling**, **attention masking**, and **adaptive envelope width**.

### Components

```
Input [B, L, C]
    |
    +---> input_proj ---> x_proj [B, L, C]  (value projection)
    |
    +---> dw_conv ---> x_dw [B, L, C]  (context for offset/mask prediction)
           |
           +---> offset_net ---> offsets [B, L, G, K] * offset_scale  (WHERE to sample)
           |
           +---> mask_net ---> raw_mask [B, L, G, K]  (HOW MUCH to weight)
                   |
                   v
            x Gaussian envelope (from sigma - per-sample or per-channel)
            x kernel_weight_mask (dynamic kernel sizing)
                   |
                   v
            softmax -> attention mask [B, L, G, K]

KernelNet (SIREN):
    grid [-0.5, 0.5] ---> kernel_net ---> kernel_weights
                          Grouped mode: [N, G*gc, K] per-sample
                          Depthwise mode: [N, C, K] per-sample, per-channel omega scaling

Chunked Deformable Gather (memory optimization):
    for l_start in range(0, L, chunk_size=128):
        # Process 128 positions at a time instead of all L
        # Reduces memory from O(N*L*G*K*gc) to O(N*chunk*G*K*gc)
        
        abs_pos = pos + ref_offset[k] + learned_offset[:,:,:,k]
        sampled = bilinear_interpolate(x_proj, abs_pos)  # Floor/ceil weights
        sampled = sampled * kernel_weights[:, :, k]
        output_chunk = sum(sampled * attn_mask, dim=K)
    
    output = concat(output_chunks)
    output ---> output_proj ---> [B, L, C]
```

### Grouped vs Depthwise Mode

**Grouped (default):**
- `G` = number of groups (default: 2)
- `gc` = channels per group = C / G
- Each group has its own offsets and attention mask
- KernelNet generates `G * gc` weights
- Single scalar biases per sample: sigma [N], offset_scale [N], omega [N]

**Depthwise (`--depthwise` flag):**
- `G` = C (each channel is its own group)
- `gc` = 1 (one channel per group)
- Per-channel learned baselines + projected biases (see below)

### SwiGLU Projections in AdaptiveDeformConv1d

Both grouped and depthwise modes have dedicated SwiGLU projections for each bias type. The BiasProjection's `context` output feeds into these projections.

**Grouped mode (4 SwiGLUs, scalar outputs):**
```python
self.sigma_proj = SwiGLU(ctx_dim, bias_ffn_mult, 0.0, output_dim=1)
self.offset_scale_proj = SwiGLU(ctx_dim, bias_ffn_mult, 0.0, output_dim=1)
self.omega_proj = SwiGLU(ctx_dim, bias_ffn_mult, 0.0, output_dim=1)
self.se_bias_proj = SwiGLU(ctx_dim, bias_ffn_mult, 0.0, output_dim=channels)
```

**Depthwise mode (4 batched SwiGLUs, per-channel outputs):**
```python
self.raw_sigma = nn.Parameter(torch.full((C,), init_raw))  # Per-channel baseline
self.base_offset_scale_per_channel = nn.Parameter(torch.full((C,), offset_scale))
self.base_omega_per_channel = nn.Parameter(torch.zeros(C))

self.sigma_proj = SwiGLU(ctx_dim, bias_ffn_mult, 0.0, output_dim=channels)
self.offset_scale_proj = SwiGLU(ctx_dim, bias_ffn_mult, 0.0, output_dim=channels)
self.omega_proj = SwiGLU(ctx_dim, bias_ffn_mult, 0.0, output_dim=channels)
self.se_bias_proj = SwiGLU(ctx_dim, bias_ffn_mult, 0.0, output_dim=channels)
```

**Forward pass (depthwise example):**
```python
if context is not None:
    sigma = F.softplus(
        self.raw_sigma.unsqueeze(0) + self.sigma_proj(context)  # [1,C] + [N,C]
    ).clamp(min=1e-3, max=self.max_sigma)
    offset_scale = self.base_offset_scale_per_channel.unsqueeze(0) * (
        1.0 + 0.2 * self.offset_scale_proj(context)
    )
    omega = self.base_omega_per_channel.unsqueeze(0) + self.omega_proj(context)
    se_bias = self.se_bias_proj(context)  # [N, C]
```

**Architecture flow:**
```
BiasProjection: pooled → shared SwiGLU → context [B, width]
                                            ↓
AdaptiveDeformConv1d: context → sigma_proj → sigma [B] or [B,C]
                      context → offset_scale_proj → offset_scale [B] or [B,C]
                      context → omega_proj → omega [B] or [B,C]
                      context → se_bias_proj → se_bias [B, C]
```

Each bias type has its own dedicated SwiGLU pathway for maximum expressiveness.

### Dynamic Kernel Sizing (Batched)

Kernel size adapts to sigma using a weight mask computed per-sample:

```python
def _compute_kernel_weight_mask(self, sigma: torch.Tensor) -> torch.Tensor:
    # sigma: [N] (per-sample) or scalar
    # Returns: [N, K] or [K] weight mask
    
    t = (sigma - self.min_sigma) / (self.max_sigma - self.min_sigma)
    t = torch.clamp(t, 0.0, 1.0)
    effective_K = self.min_kernel_size + t * (self.max_kernel_size - self.min_kernel_size)
    
    # Each position k gets weight based on whether it's within effective_K
    k_idx = torch.arange(self.max_kernel_size, device=sigma.device)
    center = (self.max_kernel_size - 1) / 2
    dist_from_center = torch.abs(k_idx - center)
    
    # Smooth falloff at edges
    mask = torch.clamp(1.0 - (dist_from_center - effective_K/2) / 2, 0.0, 1.0)
    return mask  # [N, K] for batched, [K] for scalar
```

- Small sigma (0.05) -> small effective kernel (fewer positions used)
- Large sigma (0.5) -> large kernel (wider receptive field)
- Weights applied per-sample, no `.item()` calls for GPU efficiency

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

The block that implements: **parallel convs -> merge -> SSM**.

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
        
        # Merge: conv outputs -> residual size
        self.proj_down = nn.Linear(total_width, width)
        self.norm_merge = RMSNorm(width)
        
        # Single SSM (no conv inside when use_merge_attention=False)
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
    h = self.in_proj(x)  # Note: in_norm is applied inside
    chunks = h.chunk(self.n_branches, dim=-1)
    
    # 2. Parallel adaptive convolutions
    branch_outputs = []
    for i, (branch, chunk) in enumerate(zip(self.branches, chunks)):
        branch_bias = AdaptiveConvBiases(
            sigma=biases.sigma[:, i],
            offset_scale=biases.offset_scale[:, i],
            omega=biases.omega[:, i],
            se_bias=biases.se_bias[:, i, :],  # Per-branch SE bias
        )
        out, aux = branch(chunk, mask, branch_bias)
        branch_outputs.append(out)
    
    # 3. Merge conv outputs + residual
    combined = torch.cat(branch_outputs, dim=-1)
    h = self.proj_down(combined)
    h = self.norm_merge(h + residual)
    
    # 4. Merge path (attention OR SwiGLU based on use_merge_attention)
    if self.use_merge_attention:
        h = self.merge_attn(h, mask)
    else:
        h = h + self.merge_proj(h)  # SwiGLU with residual
    
    # 5. Single SSM with residual connection
    h_res = h
    h, ssm_aux = self.ssm(h, mask)
    h = self.norm_ssm(h + h_res)  # Residual around SSM
    
    # 6. Context-gated pooling
    q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
    if pooler_context is not None:
        q = q + self.context_gate(pooler_context).unsqueeze(1)
    
    attn = torch.bmm(q, h.transpose(1, 2)) * self.pool_scale
    attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
    pooled = torch.bmm(attn, h).squeeze(1)
    features = self.feature_proj(pooled)
    
    return self.dropout(h), features, aux_losses
```

**Note:** When `use_attention=False` in LayerConfig, `use_merge_attention=False` is passed to MultiKernelSSMBlock, so SwiGLU is used instead of merge attention.

### Key Design: Parallel Convs -> Single SSM

```
Input -> [Conv_0(sigma=0.05)]  ->
      -> [Conv_1(sigma=0.275)] -> concat -> proj_down -> +residual -> SSM -> pool
      -> [Conv_2(sigma=0.50)]  ->
```

Multi-scale convolutions extract features at different scales, then ONE SSM sees all scales together.

---

## 6. SSMMixer3 (Mamba-3 Style)

The SSM component with advanced features from Mamba-2/3 papers.

### Core Structure

```python
class SSMMixer3(nn.Module):
    def __init__(self, width, state_size=16, n_heads=2, expand=2, use_conv=False, ...):
        self.intermediate_size = width * expand  # 8 * 2 = 16
        self.head_dim = intermediate_size // n_heads  # 16 / 2 = 8
        
        # Input projection with gating
        self.in_proj = nn.Linear(width, intermediate_size * 2)  # x_branch and z (gate)
        
        # B, C projections with QK-style normalization
        self.B_proj = nn.Linear(intermediate_size, n_heads * state_size)
        self.C_proj = nn.Linear(intermediate_size, n_heads * state_size)
        self.b_bias = nn.Parameter(torch.ones(n_heads, state_size))  # Learned bias
        self.c_bias = nn.Parameter(torch.ones(n_heads, state_size))
        self.norm_b = RMSNorm(state_size)  # QK-norm for stability
        self.norm_c = RMSNorm(state_size)
        
        # Discretization (per-head)
        self.dt_proj = nn.Linear(intermediate_size, n_heads)
        self.A_log = nn.Parameter(torch.zeros(n_heads))  # Learned decay rate
        
        # Data-dependent RoPE angles
        self.theta_proj = nn.Linear(intermediate_size, n_heads * (state_size // 2))
        
        # Trapezoidal interpolation factor
        self.lambda_proj = nn.Linear(intermediate_size, n_heads)
        
        # Skip connection (per-head, per-channel)
        self.D = nn.Parameter(torch.ones(n_heads, head_dim))
        
        self.out_proj = nn.Linear(intermediate_size, width)
```

### BC Biases with QK-Norm

The B and C parameters get learnable biases and normalization:

```python
# Project to [B, L, H, N]
B_param = self.B_proj(x_ssm).view(B, L, H, N)
C_param = self.C_proj(x_ssm).view(B, L, H, N)

# Add learned bias (broadcasts over B, L)
B_param = B_param + self.b_bias  # [H, N] broadcasts to [B, L, H, N]
C_param = C_param + self.c_bias

# QK-style normalization (stabilizes training)
B_param = self.norm_b(B_param)
C_param = self.norm_c(C_param)
```

### Trapezoidal Discretization

Standard Euler: `h[t] = A*h[t-1] + B*x[t]`

Trapezoidal (combines current and previous input):
```
h[t] = alpha*h[t-1] + gamma*B*x[t] + beta*B*x[t-1]

where:
  alpha = exp(dt*A)             # Decay factor
  lambda = sigmoid(lambda_proj) # Interpolation (0.5 = trapezoidal, 1.0 = Euler)
  gamma = lambda * dt           # Current input weight
  beta = (1-lambda) * dt * alpha  # Lookback coefficient
```

Lambda is initialized to sigmoid(2.0) ≈ 0.88, close to Euler but with some lookback.

### Data-Dependent RoPE

Instead of fixed positional frequencies, RoPE angles depend on content:

```python
# Content-dependent rotation angles
theta = self.theta_proj(x_ssm).view(B, L, H, N//2)  # [B, L, H, N/2]
delta_theta = dt.unsqueeze(-1) * theta  # Scale by timestep
cum_theta = torch.cumsum(delta_theta, dim=1)  # Cumulative rotation

# Apply RoPE to B and C (pairs of elements rotate together)
B_rot = apply_rotary_emb(B_param, cum_theta)
C_rot = apply_rotary_emb(C_param, cum_theta)
```

This allows the SSM to learn position-dependent phase relationships.

### Skip Connection

The D parameter provides a direct skip from input to output:
```python
y = ssm_output + x_conv * self.D[None, None, :, :]  # [B, L, H, P]
y = y * silu(z)  # Gate with z branch
```

---

## 7. Precomputed Features

The model uses one precomputed heuristic feature:

### char_frequency_difference

Measures how different a line's character distribution is from normal code:

```python
def char_frequency_difference(line_bytes: bytes, text_freq: FrequencyTable) -> float:
    # Build frequency distribution of characters in this line
    line_freq = Counter(line_bytes)
    total = len(line_bytes)
    
    # Compare against expected text frequencies (built from training data)
    # Returns Jensen-Shannon divergence or similar metric
```

This single float is expanded through HeuristicEmbed to 16 features:

```python
class HeuristicEmbed(nn.Module):
    def __init__(self, output_dim=16, hidden_dim=16):
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
```

---

## 8. Training Configuration

### Current Default (train_line_filter.py --adaptive)

```python
LayerConfig(
    embed_width=8,
    conv_groups=2,
    ssm_state_size=16,
    ssm_n_heads=2,
    ssm_expand=2,
    ssm_kernel_sizes=(7,),
    num_ssm_features=16,
    num_embed_features=16,
    num_hidden_features=16,
    num_heuristic_features=16,
    mlp_hidden_mult=16,
    mlp_output_dim=128,
    dropout=0.1,
    adaptive_conv=True,
    n_adaptive_branches=3,
    depthwise_conv=False,      # Set True with --depthwise for per-channel biases
    bias_ffn_mult=2,           # SwiGLU expansion ratio for BiasProjection and depthwise projections
    use_attention=False,
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bias_ffn_mult` | 2 | SwiGLU hidden expansion ratio in BiasProjection and per-channel depthwise projections |
| `depthwise_conv` | False | Use per-channel biases with individual SwiGLU projections |
| `n_adaptive_branches` | 3 | Number of parallel conv branches with different init_sigma values |

### Training Commands

```bash
# Standard adaptive conv (grouped, per-sample biases)
python scripts/train_line_filter.py --adaptive --n-branches 3 --epochs 20

# Depthwise adaptive conv (per-channel biases)
python scripts/train_line_filter.py --adaptive --depthwise --epochs 20
```

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
    threshold_search=False,   # Disabled for speed
)
```

---

## 9. RMSNorm Strategy

RMSNorm is applied at key transition points:

| Location | Purpose |
|----------|---------|
| Before in_proj | Normalize input to conv branches |
| After residual add | Stabilize merged conv features |
| After SSM | Prepare for pooling |
| Inside AdaptiveConvBranch | Pre-norm for conv |

---

## 10. Feature Dimensions Summary

With `use_attention=False`:

```
Input: [B, L] raw bytes

Embedding: [B, L, 8]

SSM output pooled: [B, 8] -> feature_proj -> ssm_features [B, 16]
Embed pooled: [B, 8] -> proj -> embed_features [B, 16]
Hidden pooled: [B, 8] -> proj -> hidden_features [B, 16]
Precomputed: [B, 1] -> HeuristicEmbed -> heuristic_features [B, 16]

Total features: 16 + 16 + 16 + 16 = 64
After GatedMLP: [B, 128]
Final logit: [B, 1]
```

---

## 11. Architecture Variants

The model supports multiple configurations via CLI flags:

| Flag | Effect |
|------|--------|
| `--adaptive` | Enable adaptive conv with KernelNet (default training mode) |
| `--depthwise` | Use depthwise conv with per-channel biases (requires --adaptive) |
| `--unified` | Enable unified architecture (implied by --adaptive) |
| `--ssm-kernels X` | Set SSM conv kernel sizes (0=no conv, 7=single, 3,5,7,9=multi) |
| `--n-branches N` | Number of adaptive conv branches (default: 3) |
| `--n-layers N` | Number of unified layers to stack (default: 1) |

### Grouped vs Depthwise Comparison

| Aspect | Grouped (default) | Depthwise (`--depthwise`) |
|--------|-------------------|---------------------------|
| Groups | G=2 | G=C (8) |
| Channels/group | C/G=4 | 1 |
| Bias shape | [N] per-sample scalar | [N, C] per-channel |
| Learned baselines | 1 scalar (raw_sigma) | C scalars each |
| SwiGLU projections | 4 SwiGLUs (output_dim=1 or C) | 4 batched SwiGLUs (output_dim=C) |
| Cross-channel mixing | Within groups | Only via input/output proj |
| Parameters | ~25k | ~28k |

---

## 12. Data Pipeline & Stratified Splitting

### The Duplicate Problem

The CredData dataset contains **massive duplication** of secrets across files. The same API key, password, or token can appear in dozens of different source files. This creates a critical data leakage risk: if we naively split by file, the same secret text ends up in both train and val sets.

**Example**: An AWS key `AKIA...` might appear in:
- `repo1/config.yaml` -> assigned to train
- `repo2/.env` -> assigned to val
- `repo3/settings.py` -> assigned to train

The model sees the exact same secret in both splits. This inflates validation metrics and gives false confidence.

### Content-Based Secret IDs

To fix this, we generate secret IDs based on **content hash**, not file location:

```python
def _content_hash_secret_id(secret_bytes: bytes) -> str:
    return hashlib.sha256(secret_bytes).hexdigest()[:16]
```

Now all occurrences of `AKIA...` get the same ID regardless of which file they're in. When we split, ALL samples containing that secret stay together.

### Multi-Label Stratified Splitting

Secrets have categories (api_key, password, auth_token, generic_secret). We want each category represented proportionally in both splits.

**Algorithm:**
1. **Window all documents** into samples (single-line, multi-line, secrets-only)
2. **Build connected components** using union-find: samples sharing ANY secret_id are grouped
3. **Assign categories** to each component (union of all member categories)
4. **Greedy stratified assignment**: process components from rarest category first, assign to split that most needs that category

This ensures:
- No secret leakage (same secret never in both splits)
- Proportional category representation
- Rare categories get priority placement

### Split Ratio Analysis

We analyzed unique secrets per category:

| Category | Unique Secrets |
|----------|----------------|
| generic_secret | 2,187 |
| password | 929 |
| api_key | 843 |
| auth_token | 820 |

**Total: 4,638 unique secrets**

The rarest category (auth_token) has only 820 secrets. For meaningful validation, we want ~100+ unique secrets per category in val.

| Target Min in Val | Train % | Val % |
|-------------------|---------|-------|
| 30 | 96.3% | 3.7% |
| 50 | 93.9% | 6.1% |
| 100 | 87.8% | 12.2% |
| 150 | 81.7% | 18.3% |

**Decision: 88% train / 12% val**

This gives ~100 unique auth_tokens in val (the rarest category), with proportionally more for other categories. No separate test set—it would just waste data for a model that's iteratively tuned on val anyway.

### Final Split Statistics

```
Train: 5,266,782 samples, 4,116 unique secrets
Val:     718,405 samples,   522 unique secrets

Unique secrets in val per category:
  auth_token:     91
  api_key:       102
  password:      108
  generic_secret: 252

Secret leakage: 0 (verified at runtime)
```

### Sample Types in Mixed Mode

The dataset combines three sample types:

1. **Single-line windows**: Each line from source files, windowed with prefix/suffix preservation if >512 bytes
2. **Multi-line windows**: 2-9 consecutive lines, up to 512 bytes total, with overlap
3. **Secrets-only**: Just the secret/FP span bytes, properly labeled by category

This gives the model diverse views: secrets in context, secrets across line boundaries, and secrets in isolation.

### Filtering

Windows are filtered to remove noise:

- **Whitespace-only lines**: Filtered in `extract_lines_with_masks()`
- **Short windows**: `MIN_WINDOW_LENGTH = 7` - windows <= 6 bytes filtered everywhere

---

## 13. Performance Optimizations

### Validation Loop

- Tensors stay on device during validation (no per-batch `.cpu()` calls)
- Single `torch.cat().cpu()` after loop instead of hundreds of sync points
- Metrics computed in numpy (no torch tensor operations)
- Threshold search disabled by default (`threshold_search=False`)

### Batch Timing

- ms-per-token tracking instead of raw batch time
- Slow batch detection: `ms_per_token > mean + stddev`
- tqdm smoothing=0 for accurate average speeds

### Per-Sample/Per-Channel Biases

All biases are now properly batched without `.mean()` collapsing:

```python
# Grouped mode: per-sample biases
sigma: [N] -> envelope [N, K], kernel_weight_mask [N, K]
offset_scale: [N] -> broadcast [N, 1, 1, 1] for offsets [N, L, G, K]
omega: [N] -> kernel_weights [N, G*gc, K]

# Depthwise mode: per-channel biases  
sigma: [N, C] -> envelope [N, C, K], kernel_weight_mask [N, C, K]
offset_scale: [N, C] -> broadcast [N, 1, G, 1] for offsets [N, L, G, K] (G=C)
omega: [N, C] -> kernel_weights [N, C, K] via post-scaling
```

No `.item()` calls, fully GPU-compatible.
