# Architecture Document — Diffusive-State Relay MoE (DS-MoE)

## Overview

A clean-room rewrite of the core attention/SSM/MLP components for a hybrid diffusion-based architecture. Designed to be:
- Modular: each op is a standalone `nn.Module(x) -> y` with no residual or norm baked in
- Composable: a Block class applies pre-norm + residual around each op, plus a gated anchor to the original embedding
- Testable: each op can be benchmarked and validated independently
- Compatible: should slot into both a modded-nanogpt-style trainer and zoology
- Multi-modal generation: supports autoregressive LM, classification, and text diffusion

**Core philosophy**: Signal denoising via differential operators & global anchoring. The architecture creates a "Dense-Highway" topology where signal processing is handled by a hybrid of Diffusive State SSMs and Differential MHA, while semantic routing is managed by a Segmented (Relay) Mixture-of-Experts system driven by Squeeze-Excite logic.

## Embedding Strategy

All ops operate on flat `(B, L, D)` tensors. Dimensionality is purely an embedding concern:

**1D sequences** (text, audio): Token embedding only. No learned positional embedding — position information comes entirely from RoPE applied inside DiffAttn and DS1. Embed dropout is applied after embedding.

**2D inputs** (images): Learned sinusoidal 2D positional embedding. Per-axis learnable frequencies and phases produce a continuous position encoding: `pos_embed = proj(cat(sin/cos(h_freq * h_pos + h_phase), sin/cos(w_freq * w_pos + w_phase)))`. Added to pixel/patch embeddings, then flattened to `(B, H*W, D)` for processing. All ops see 1D sequences.

**3D inputs**: Same sinusoidal approach extended to 3 axes, flattened to `(B, D1*D2*D3, D)`.

## Core Ops

### 1. Differential Multi-Head Attention — "The Relation Engine"

Causal self-attention with differential output, QK-norm, and RoPE. Cancels common-mode noise to isolate sharp, long-range dependencies. Compensates for the SSM's lack of selectivity — handles copying and specific associative recall that static SSMs cannot perform.

**Differential output**: QKV projects to `2 * num_heads` sub-heads. Sub-heads are paired — each pair computes two independent attention outputs, and the final output is `head1 - λ * head2`. This cancels common-mode noise in the attention pattern.

**λ reparameterization** (from Ye et al., ICLR 2025): Per-head λ is NOT a simple sigmoid. It uses a dot-product parameterization:
```
λ = exp(dot(λ_q1, λ_k1) / sqrt(half_dim)) - exp(dot(λ_q2, λ_k2) / sqrt(half_dim)) + λ_init
```
where `λ_q1, λ_k1, λ_q2, λ_k2` are learnable vectors of size `half_dim`, initialized to `randn * 0.1`. Dots are scaled by `1/sqrt(half_dim)` to prevent exp blowup as dim grows. `λ_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)` provides layer-dependent initialization.

**Head normalization**: After the differential subtraction, the output is normalized per-head and scaled: `diff = head_norm(diff) * (1 - λ_init)`. `head_norm` is a learnable RMSNorm over `head_dim`.

**QK-norm**: Q and K halves are RMS-normalized per sub-head (over `half_dim`) with learnable scale before RoPE. Separate `q_norm` and `k_norm` modules, each RMSNorm(half_dim). The same norm is applied to both halves of each (q1/q2 share q_norm, k1/k2 share k_norm).

**RoPE**: Applied to Q and K sub-heads after QK-norm.

**Activation**: `silu` applied to QKV projection output and to `out_proj` output.

**Biases**: `qkv` and `out_proj` both use `bias=True`.

**Dropout**: Configurable attention dropout (applied during training only).

```
Input:  x (B, L, D)
Output: y (B, L, D)

qkv = silu(linear(x))        # bias=True, D -> 3*D
q, k, v = reshape to (B, L, H, head_dim) each  # H = num_heads
q1, q2 = q[..., :half], q[..., half:]           # half = head_dim // 2
k1, k2 = k[..., :half], k[..., half:]

q1, q2 = q_norm(q1), q_norm(q2)   # shared RMSNorm(half_dim)
k1, k2 = k_norm(k1), k_norm(k2)   # shared RMSNorm(half_dim)
q1, q2 = rope(q1), rope(q2)
k1, k2 = rope(k1), rope(k2)

out1 = sdpa(q1, k1, v, causal=True)  # (B, H, L, head_dim)
out2 = sdpa(q2, k2, v, causal=True)

λ = exp(dot(λ_q1, λ_k1) / sqrt(half_dim)) - exp(dot(λ_q2, λ_k2) / sqrt(half_dim)) + λ_init
diff = head_norm(out1 - λ * out2) * (1 - λ_init)
y = silu(out_proj(diff.reshape(B, L, D)))  # bias=True
```

Parameters: `qkv` (D -> 3*D, bias), `out_proj` (D -> D, bias), `λ_q1/k1/q2/k2` (half_dim each), `q_norm` RMSNorm(half_dim), `k_norm` RMSNorm(half_dim), `head_norm` RMSNorm(head_dim).

Non-differential mode: When `differential=False`, standard MHA with full `head_dim` QK-norm, no λ, no head_norm.

### 2. Diffusive State SSM (DS1) — "The Texture Engine"

**DS1 — Diffusive State 1.** An open-loop SSM that uses iterative spatial convolutions instead of sequential scanning. The state evolves through fixed-point iterations with trapezoidal integration. Acts as a stable "carrier wave" convolution for text structure — smooths local noise and manages continuous signal features.

**Open-loop (state-independent projections)**: The state `H` is not fed back into input projections (making it parallelizable), but the decay, theta, and lambda ARE input-dependent (projected from x). The projections are open-loop w.r.t. state — not LTI, since parameters vary with input.

**Why "diffusive state"**: Instead of scanning left-to-right like Mamba, the state evolves through iterative spatial diffusion — depthwise convolutions that spread information across sequence positions. No scan, no recurrence. State is refined through `n_iters` iterations of: (1) project input to state space, (2) diffuse state via depthwise conv, (3) decay + inject. Each iteration sees the same input but rotates the B/C projections via RoPE with iteration-dependent frequency, giving each iteration a different "view" of the input.

**Trapezoidal integration**: The state update uses a trapezoidal rule: `H = α*H + β*prev_inject + γ*curr_inject`, blending the previous and current injection with learned mixing. This is more stable than Euler integration.

**Activation**: `silu` by default (configurable to `relu_squared` via `relu2` flag). Applied to B, C, and X projections.

**B/C bias**: Learnable bias parameters `B_bias` and `C_bias`, both initialized to `ones(N*R)`. Added before activation: `B = act(to_B(x) + B_bias)`.

**BC-norm** (optional): RMSNorm(N) applied to B and C after reshaping to `(B, R, L, N)`, before RoPE. Separate `b_norm` and `c_norm` modules.

**RoPE (interleaved)**: Uses interleaved even/odd index rotation, NOT the standard concat-half approach:
```
x1, x2 = x[..., ::2], x[..., 1::2]   # even and odd indices
out[..., ::2] = x1 * cos - x2 * sin
out[..., 1::2] = x1 * sin + x2 * cos
```
Theta is per-iteration: `theta_k = theta * (i + 1)`, applied to B and C each iteration.

**Differential inject** (optional): B is split in half along state dim. Both subtraction AND addition are concatenated:
```
B1, B2 = B_rot[..., :N//2], B_rot[..., N//2:]
inject = cat[B1*X - λ_inject*B2*X, B1*X + λ_inject*B2*X]
```
`inject_lambda` is a scalar parameter initialized to `0.5`.

**Differential readout** (optional): Same pattern as inject:
```
C1, C2 = C_rot[..., :N//2], C_rot[..., N//2:]
H1, H2 = H[..., :N//2], H[..., N//2:]
readout = cat[C1*H1 - λ_readout*C2*H2, C1*H1 + λ_readout*C2*H2]
```
`readout_lambda` is a scalar parameter initialized to `0.5`.

**Squeeze-excite on diffusion** (optional): `SqueezeExciteND` applied to H after depthwise conv, before trapezoidal update. Reduction factor 4, min hidden 8.

**Projection shapes**:
- `to_B`: D -> N*R (reshaped to B, L, N, R then permuted to B, R, L, N)
- `to_C`: D -> N*R (same reshape)
- `to_X`: D -> R (broadcast to B, R, L, 1)
- `to_decay`: D -> N (per state dimension, sigmoid applied)
- `to_theta`: D -> N//2 (half state dim, content-conditioned base frequency; multiplied by position index for positional RoPE)
- `to_lambda`: D -> 1 (scalar per position, sigmoid applied, used as trapezoidal blend)

```
Input:  x (B, L, D)
Output: y (B, L, D)

B = act(to_B(x) + B_bias)  -> (B, L, N, R) -> permute to (B, R, L, N)
C = act(to_C(x) + C_bias)  -> same
X_r = act(to_X(x))         -> (B, L, R) -> permute to (B, R, L, 1)
decay = sigmoid(to_decay(x))  -> (B, L, N) -> unsqueeze to (B, 1, L, N)
theta = to_theta(x)           -> (B, L, N//2)  # content-conditioned base freq
pos = arange(L)               -> (L,)          # position indices
lam = sigmoid(to_lambda(x))   -> (B, L, 1) -> unsqueeze to (B, 1, L, 1)

# Optional: BC-norm before any rotation
B = b_norm(B)  # RMSNorm over N dim
C = c_norm(C)

H = zeros(B, R, L, N)
prev_inject = None

for i in range(n_iters):
    # theta: (B, L, N//2), pos: (L,) -> angles: (B, 1, L, N//2) broadcast over R
    angles = theta.unsqueeze(1) * pos[None, None, :, None] * (i + 1)
    B_rot = interleaved_rope(B, angles)
    C_rot = interleaved_rope(C, angles)

    # Injection (with optional differential)
    if diff_inject:
        B1, B2 = B_rot[..., :N//2], B_rot[..., N//2:]
        inj1, inj2 = B1 * X_r, B2 * X_r
        inject = cat[inj1 - λ_inject * inj2, inj1 + λ_inject * inj2]
    else:
        inject = B_rot * X_r

    # Diffuse state spatially — Conv1d needs (batch, channels, length)
    H = H.permute(0, 1, 3, 2).reshape(B*R, N, L)       # (B, R, L, N) -> (B*R, N, L)
    H = depthwise_conv1d(H)                             # kernel_size=3, padding=1, groups=N
    H = H.reshape(B, R, N, L).permute(0, 1, 3, 2)      # back to (B, R, L, N)
    if diffuse_se:
        H = squeeze_excite(H)  # over N channels

    # Trapezoidal update
    α = decay    # (B, 1, L, N)
    γ = lam      # (B, 1, L, 1)
    if prev_inject is not None:
        β = (1 - γ) * α
        H = α * H + β * prev_inject + γ * inject
    else:
        H = α * H + inject
    prev_inject = inject

# Readout (with optional differential)
if diff_readout:
    C1, C2 = C_rot[..., :N//2], C_rot[..., N//2:]
    H1, H2 = H[..., :N//2], H[..., N//2:]
    g1, g2 = C1 * H1, C2 * H2
    gated = cat[g1 - λ_readout * g2, g1 + λ_readout * g2]
else:
    gated = C_rot * H

# (B, R, L, N) -> permute to (B, L, R, N) -> reshape (B, L, N*R)
y = act(out_proj(gated))  # N*R -> D
```

Parameters: `to_B` (D -> N*R), `to_C` (D -> N*R), `to_X` (D -> R), `to_decay` (D -> N), `to_theta` (D -> N//2), `to_lambda` (D -> 1), `B_bias` (N*R, init=ones), `C_bias` (N*R, init=ones), `out_proj` (N*R -> D), `diffuse` conv1d (N groups, kernel 3, padding 1), optional `diffuse_se` SqueezeExcite(N), optional `b_norm/c_norm` RMSNorm(N), optional `inject_lambda` (scalar, init=0.5), optional `readout_lambda` (scalar, init=0.5).

Defaults: `state_dim(N)=64, mimo_rank(R)=4, n_iters=2`.

**Open question**: Does this actually behave like an SSM? We need to validate on standard SSM benchmarks (selective copying, induction heads, long-range arena) to confirm the iterative-conv mechanism captures sequential dependencies.

### 3. Differential SwiGLU MLP — "The Logic Engine"

Dual-path projection with subtractive activation. High-pass filtering of features — amplifies specific semantic variances while suppressing background noise.

**Hidden dim**: Uses the `2/3` scaling from LLaMA-style SwiGLU: `hidden = int(D * mult * 2/3)`, rounded up to the nearest multiple of 8.

**λ parameterization**: Additive offset — `lam = sigmoid(lambda_logit) + lambda_init` where `lambda_init = 0.5` and `lambda_logit` is initialized to `0.0`. This means λ starts at `1.0` (sigmoid(0) + 0.5 = 1.0).

**Biases**: `gate` and `up` are bias-free. `down` is bias-free.

```
Input:  x (B, L, D)
Output: y (B, L, D)

hidden = round_up_to_8(int(D * mult * 2/3))
h = silu(gate(x)) * up(x)     # gate, up: D -> hidden (no bias)
o1, o2 = down(h).chunk(2, -1) # down: hidden -> 2*D (no bias)
λ = sigmoid(lambda_logit) + lambda_init
y = o1 - λ * o2
```

Parameters: `gate` (D -> hidden, no bias), `up` (D -> hidden, no bias), `down` (hidden -> 2*D, no bias), `lambda_logit` (scalar, init=0.0), `lambda_init` (float, default=0.5).

Default `mult=4` → hidden = round_up_to_8(int(D * 4 * 2/3)) = round_up_to_8(int(D * 2.667)).

## Block Structure: The Super-Block

### Default Super-Block: SSM → MHA → MLP

Every block contains all three core ops in sequence. This is the mandatory default configuration:

1. **DS1** — establishes sequential context, smooths local noise (texture)
2. **DiffAttn** — precise token-level retrieval, long-range dependencies (relations)
3. **DiffMLP** — nonlinear feature transformation, high-pass semantic filtering (logic)

Each op is wrapped with pre-norm and residual:

```
x = x + op(norm(x))  # norm = RMSNorm(dim), learnable
```

### Gated Anchor to Embedding

After all ops in a block, the original input embedding `embed` is injected via a learnable gate:

```
x = x + gate_l * embed
```

`gate_l` is a learnable scalar (or vector) per layer, **initialized near 0** (e.g., `init_constant_(bias, -2.0)` so `sigmoid(gate) ≈ 0.12`). This lets the model slowly learn to rely on the anchor injection.

**Why this is critical:**
- In diffusion models, deep layers lose track of the conditional prompt as they process noisy state. The anchor ensures every layer can "check the original prompt."
- Provides a **depth-1 gradient highway** from the loss function to the embedding layer, preventing vanishing gradients.
- For autoregressive LM, prevents residual stream drift by maintaining grounding to the input signal.
- This was `embed_residual` in old RippleAttention but is now a proper gated mechanism at block level.

### Full Block Pseudocode

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, ssm_cfg):
        # Ops are stateless — no stored weights (bank pattern)
        self.attn = DiffAttn(dim, num_heads)   # config only, no nn.Linear
        self.ssm = DS1(dim, **ssm_cfg)     # config only, no nn.Linear
        self.mlp = DiffMLP(dim)                 # config only, no nn.Linear
        # Learnable pre-norms per op
        self.norm_attn = RMSNorm(dim)
        self.norm_ssm = RMSNorm(dim)
        self.norm_mlp = RMSNorm(dim)
        # Gated anchor (too small for banking)
        self.anchor_gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.anchor_gate.weight)
        nn.init.constant_(self.anchor_gate.bias, -2.0)

    def forward(self, x, embed, attn_w, ssm_w, mlp_w):
        x = x + self.attn(self.norm_attn(x), attn_w)
        x = x + self.ssm(self.norm_ssm(x), ssm_w)
        x = x + self.mlp(self.norm_mlp(x), mlp_w)
        # Gated anchor: inject original embedding
        gate = torch.sigmoid(self.anchor_gate(x))
        x = x + gate * embed
        return x
```

Each op has its own learnable `RMSNorm(dim)` pre-norm. Norms and anchor gate are stored on the Block (too small for banking). Ops receive weight slices from banks — the Block stores no projection weights.

## Macro-Architecture: Relay MoE

### Segmented Expert Routing

The model uses a **Segmented (Relay) Mixture-of-Experts** strategy. Instead of routing every token at every layer, experts are "locked in" for fixed-depth segments and rerouted at segment boundaries.

**Routing cadence:**
- **Total depth:** `L` layers (e.g., 24)
- **Segment size:** `S` layers (e.g., 6)
- **Routing points:** Layers `0, S, 2S, ...` (i.e., 4 routing decisions for 24 layers with S=6)

### Expert Design

Each expert is a **full-width super-block chain** — every expert sees and produces the full `D`-dimensional representation. This is standard MoE, not channel-split.

Expert chains can have different op compositions (e.g., one expert uses `[DS1, DiffAttn, DiffMLP]`, another uses `[DiffAttn, DiffAttn, DiffMLP]`), or they can all share the same architecture with independent weights.

### The SE Router

At each segment boundary, a Squeeze-Excite block determines optimal experts for the next segment:

1. **Squeeze (Compression):** Aggregate global context of the current residual stream via Global Average Pooling.
   ```
   z = x.mean(dim=1)  # (B, D) — pool over sequence length
   ```

2. **Excite (Selection):** Project compressed state to expert logits, select top-k.
   ```
   logits = router(silu(fc1(z)))  # (B, n_experts)
   routing = top_k_softmax(logits, k)
   ```

3. **Lock-in:** Selected top-k experts process ALL tokens for the next `S` layers. No per-token routing — this is per-sequence routing locked for the segment duration.

### SE Mix-and-Reroute Between Segments

After each expert segment completes:
1. Expert outputs are combined via weighted sum using routing weights
2. A **SqueezeExcite block** recalibrates the combined output (global channel attention) — this is the "mix" step that lets experts share knowledge
3. A new router scores experts for the next segment — this is the "reroute" step

```python
# At segment boundary:
for segment_idx in range(n_segments):
    # Route
    z = x.mean(dim=1)
    logits = router[segment_idx](silu(fc1(z)))
    topk_vals, topk_idx = logits.topk(k, dim=-1)
    routing = softmax(topk_vals)

    # Execute locked experts for S layers (per-sequence routing)
    # In practice: group batch elements by expert set for efficient execution
    expert_outputs = []
    for i in range(k):
        expert_idx = topk_idx[:, i]  # (B,) — per-sequence expert selection
        expert_out = x
        for layer_offset in range(S):
            layer_idx = segment_idx * S + layer_offset
            expert_out = expert_chains[expert_idx][layer_idx](expert_out, embed)
        expert_outputs.append(expert_out)

    # Weighted combine: routing is (B, k) from softmax over topk values
    x = sum(routing[:, i:i+1].unsqueeze(-1) * expert_outputs[i] for i in range(k))

    # SE mix-and-reroute
    x = squeeze_excite(x)  # recalibrate channels across expert contributions
```

**Why relay routing works:** It allows the model to switch processing "modes" based on depth (e.g., early segments = denoising mode, late segments = polishing mode). Per-sequence routing (not per-token) avoids the load balancing headaches of per-token MoE.

## Output Heads

### Autoregressive LM Head

Standard linear projection with weight tying:
```
logits = linear(norm(x))  # head.weight = embed.weight
```

For MoE models, a final SE block before the linear projection recalibrates channel importance from expert outputs:
```
x = squeeze_excite(norm(x))
logits = linear(x)
```

### ML-Decoder Head (for Classification)

Instead of simple `mean-pool → linear`, uses a learned cross-attention decoder inspired by ML-Decoder (Ridnik et al.):

```python
class MLDecoderHead(nn.Module):
    def __init__(self, width, n_classes, num_groups=None, num_heads=8):
        num_groups = min(n_classes, 100)  # group classes to reduce query count
        group_size = ceil(n_classes / num_groups)

        self.queries = nn.Parameter(randn(num_groups, width) * 0.02)  # learnable class queries
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.group_fc = nn.Parameter(randn(num_groups, width, group_size) * 0.02)

    def forward(self, x):  # x: (B, L, D)
        q = self.q_proj(self.queries)  # (num_groups, D) -> (B, num_groups, H, head_dim)
        k = self.k_proj(x)             # (B, L, D) -> (B, L, H, head_dim)
        v = self.v_proj(x)

        attn = scaled_dot_product_attention(q, k, v)  # (B, num_groups, D)
        logits = einsum('bgc,gco->bgo', attn, group_fc)  # (B, num_groups, group_size)
        return logits.reshape(B, -1)[:, :n_classes]
```

**Why ML-Decoder over mean-pool + linear:**
- Learnable class queries attend to the full sequence — each query group specializes on different parts of the input
- Much better than collapsing all sequence info into a single mean vector
- Grouped structure is parameter-efficient even for large n_classes

### Text Diffusion Head

Denoising diffusion for text generation. The same block backbone processes noised token embeddings conditioned on a noise level, and the model predicts the clean embeddings (or noise).

**Signal dynamics for diffusion:**
- The **Gated Anchor** ensures `x_t` is always strictly conditioned on the prompt embedding, preventing deep layers from losing the conditioning signal
- **Differential ops** (DiffAttn, DiffMLP) are intrinsically subtractive — they *compare* features rather than *add* them. This is mathematically aligned with the diffusion objective of predicting noise `ε`, creating a network of "Difference Engines" that isolate signal from noise
- The **DS1** acts as an input-independent carrier wave — fast and stable for processing the continuous noise structure

Design TBD for specific noise schedule, conditioning mechanism, and sampling strategy. The core ops and block structure are reusable; only the model shell and training loop differ.

## Model Shells

All shells share the same block backbone. The shell determines embedding, output head, and whether MoE is used. The shell owns parameter banks and passes weight slices to blocks.

```python
class LM(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, num_heads, ssm_cfg):
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, dim)
        self.embed_drop = nn.Dropout(embed_dropout)
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ssm_cfg) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        # Transposed weight storage: (in_features, out_features)
        self.lm_head = nn.Parameter(empty(dim, vocab_size, dtype=bfloat16))

        # --- Parameter Banks (transposed storage, bfloat16) ---
        # DiffAttn: QKVO weights for all layers
        self.attn_bank = nn.Parameter(empty(n_layers, 4 * dim, dim, dtype=bfloat16))
        self.attn_bank.label = 'attn'
        self.attn_bank.reshape = (n_layers * 4, dim, dim)

        # DS1: all projection weights for all layers
        ssm_bank_size = ...  # sum of all SSM projection sizes
        self.ssm_bank = nn.Parameter(empty(n_layers, ssm_bank_size, dtype=bfloat16))
        self.ssm_bank.label = 'ssm'
        self.ssm_bank.reshape = (n_layers * n_ssm_mats, ...)

        # DiffMLP: gate, up, down for all layers
        hidden = round_up_to_8(int(dim * 4 * 2/3))
        self.mlp_bank = nn.Parameter(empty(n_layers, 3, hidden, dim, dtype=bfloat16))
        self.mlp_bank.label = 'mlp'
        self.mlp_bank.reshape = (n_layers * 3, hidden, dim)

        # Packed scalars for optimizer efficiency
        self.scalars = nn.Parameter(cat([...]))  # anchor gates, lambdas, etc.
        self.scalars.label = 'scalars'

        # Weight tying: embed mirrors lm_head.T during tied phase
        with torch.no_grad():
            self.embed.weight.copy_(self.lm_head.T)

    def forward(self, idx):
        embed = self.embed_drop(self.embed(idx))
        x = embed

        # Unbind banks to avoid select_backwards kernel
        attn_ws = self.attn_bank.unbind(0)
        ssm_ws = self.ssm_bank.unbind(0)
        mlp_ws = self.mlp_bank.unbind(0)

        for i in range(self.n_layers):
            x = self.blocks[i](x, embed, attn_ws[i], ssm_ws[i], mlp_ws[i])

        x = self.norm(x)
        return x @ self.lm_head.type_as(x)  # transposed storage: x @ W
```

**Classifier** variant: Uses `MLDecoderHead` instead of linear head. For 2D/3D inputs, uses `LearnedSinusoidal2DEmbed` / `3DEmbed` then flattens to 1D. Uses learned positional embedding (`pos_embed`) + `embed_norm` for non-RoPE position info.

**MoE** variant: Wraps blocks into segments with SE routing at boundaries. Final head gets SE before projection.

## Weight Initialization

All models use the following initialization strategy:

```python
def _init_weights(self):
    nn.init.normal_(self.embed.weight, std=0.02)
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    # Scale output projections by depth to prevent signal explosion
    for layer in self.blocks:
        for name, p in layer.named_parameters():
            if "out_proj.weight" in name:
                with torch.no_grad():
                    p.mul_(1.0 / (2 * n_layers) ** 0.5)
```

Key points:
- All linear weights: `normal_(std=0.02)`
- All biases: `zeros_()`
- Output projection weights in each op: scaled down by `1 / sqrt(2 * n_layers)` to control residual stream growth
- Embedding weights: `normal_(std=0.02)`
- Anchor gate bias: `constant_(-2.0)` so sigmoid starts near 0
- Anchor gate weight: `zeros_()` so anchor is initially a no-op

## Compatibility with Modded-NanoGPT Training

The architecture must support the training speed techniques from modded-nanogpt. These are not optional optimizations — they are structural constraints that affect how ops and models are written.

### Parameter Banks (Critical)

**Ops must not own their weights.** Instead of `self.qkv = nn.Linear(...)`, each op receives weight tensors as forward arguments. All weights of the same type across all layers are stored in a single contiguous "bank" tensor owned by the model shell.

```python
# BAD: op owns weights (standard PyTorch pattern)
class DiffAttn(nn.Module):
    def __init__(self, dim):
        self.qkv = nn.Linear(dim, 3 * dim)
    def forward(self, x):
        qkv = self.qkv(x)

# GOOD: op receives weights (bank pattern)
class DiffAttn(nn.Module):
    def __init__(self, dim):
        pass  # no stored weights
    def forward(self, x, qkvo_w):
        q, k, v = (x @ qkvo_w[:dim*3]).chunk(3, -1)
```

**Bank layout for DS-MoE:**

```python
# Model shell owns these banks:
self.attn_bank = nn.Parameter(empty(n_layers, 4 * D, D))       # DiffAttn: QKVO
self.ssm_bank = nn.Parameter(empty(n_layers, n_ssm_mats, ...)) # DS1: B,C,X,decay,theta,lambda,out
self.mlp_bank = nn.Parameter(empty(n_layers, 3, hidden, D))    # DiffMLP: gate, up, down

# In forward:
attn_weights = self.attn_bank.unbind(0)  # avoid select_backwards kernel
ssm_weights = self.ssm_bank.unbind(0)
mlp_weights = self.mlp_bank.unbind(0)

for i in range(n_layers):
    x = self.blocks[i](x, embed, attn_weights[i], ssm_weights[i], mlp_weights[i])
```

**Why banks matter:**
- Single `reduce_scatter` / `all_gather` per bank instead of per-layer per-projection
- NorMuon optimizer operates on the entire bank as a batched matrix (one Polar Express call)
- Contiguous memory layout for optimizer and communication

### Transposed Weight Storage

All projection weights stored as `(in_features, out_features)` — the transpose of PyTorch's default `(out_features, in_features)`. Forward computes `x @ W` instead of `F.linear(x, W)`.

**Why:** Gradient accumulation `grad_w = x.T @ grad_out` produces a result already in the stored layout. This eliminates a slow transpose kernel during backward.

### Parameter Labels and Reshape

Every bank parameter needs two custom attributes for the optimizer:

```python
self.attn_bank.label = 'attn'
self.attn_bank.reshape = (n_layers * 4, D, D)  # dim[0] must be divisible by world_size
```

- `.label`: unique identifier for optimizer config lookup (lr_mul, wd_mul, betas, etc.)
- `.reshape`: shape for sharding. Leading dimension must be divisible by `world_size` for `reduce_scatter` / `all_gather`

### Packed Scalars

All per-layer learnable scalars (anchor gates, residual lambdas, differential λ params, etc.) should be packed into a single `self.scalars` parameter:

```python
self.scalars = nn.Parameter(torch.cat([
    anchor_gate_inits,           # (n_layers,) — anchor gate biases
    resid_lambdas,               # (n_layers,) — residual stream scaling
    # ... other per-layer scalars
]))
self.scalars.label = 'scalars'
```

This allows a single Adam update on all scalars with shared optimizer config (higher momentum, no weight decay, custom lr_mul).

### bfloat16 Weights

All bank parameters are stored in `bfloat16`. The NorMuon optimizer uses mantissa tracking (`uint16` buffer) to recover float32 precision during updates. Ops cast weights with `.type_as(x)` when needed.

### torch.compile Compatibility

The entire model must compile as one graph (`torch.compile(dynamic=False, fullgraph=True)`):

- **No dynamic control flow:** The `n_iters` loop in DS1 is fine (fixed count known at compile time). Any `if` based on input values or shapes that change between calls is NOT okay.
- **No Python-level conditionals on tensor values:** Use `torch.where` instead.
- **Bank `.unbind(0)` at start of forward:** Avoids `select_backwards` kernel overhead (a `torch.compile` optimization).
- **Fixed tensor shapes:** All bank shapes, sequence lengths, batch sizes must be known at compile time (or use `dynamic=True` with performance cost).

### FP8 Matmul (Optional, Phase 2)

Large projections (LM head, potentially DiffAttn QKV) can use FP8 matmul with custom `_scaled_mm` kernels. The weight-as-argument pattern makes this trivial to add: just quantize the weight slice before passing to the op.

### Weight Tying with Untie Support

LM head and embedding are tied during early training, then split at a configurable step (e.g., 2/3 of training):

```python
# Tied phase: embed.weight = lm_head.weight.T (transposed storage)
# Optimizer only updates lm_head, copies .T to embed after each step
# Untie step: copy optimizer state from lm_head to embed, mark as split
```

The model shell must support both tied and untied modes without code changes — controlled by the optimizer/training manager.

### Block Forward Signature

Due to parameter banks, the Block's forward signature is:

```python
def forward(self, x, embed, attn_w, ssm_w, mlp_w):
    x = x + diff_attn(norm_attn(x), attn_w)
    x = x + ds1(norm_ssm(x), ssm_w)
    x = x + diff_mlp(norm_mlp(x), mlp_w)
    gate = torch.sigmoid(self.anchor_gate(x))
    x = x + gate * embed
    return x
```

Each op is a pure function of `(input, weights)` — stateless except for fixed config (num_heads, state_dim, etc.). The anchor gate is the only weight stored on the Block itself (it's a small scalar/vector, not a bank candidate).

### Fused Kernels (Phase 2 Optimization)

Candidates for Triton kernel fusion:
- **DiffMLP:** Fuse `silu(x @ gate_w) * (x @ up_w)` into one kernel (analogous to `FusedLinearReLUSquare`)
- **DS1 inner loop:** Fuse diffuse → inject → trapezoidal update per iteration
- **Softcapped cross entropy:** Reuse modded-nanogpt's `FusedSoftcappedCrossEntropy` or adapt for our loss

### Summary: Structural Requirements

| Requirement | Affects | Non-Negotiable? |
|------------|---------|-----------------|
| Ops receive weights as args (no stored nn.Linear) | All ops | ✅ Yes |
| Transposed weight storage `(in, out)` | All projections | ✅ Yes |
| `.label` and `.reshape` on bank params | Model shell | ✅ Yes |
| Packed scalars tensor | Model shell | ✅ Yes |
| bfloat16 weight storage | All params | ✅ Yes |
| `torch.compile(fullgraph=True)` compatible | All ops | ✅ Yes |
| Bank `.unbind(0)` in forward | Model shell | ✅ Yes |
| FP8 matmul support | LM head, large projections | Optional |
| Weight tying with untie | LM shell | ✅ Yes |
| Fused Triton kernels | Hot paths | Optional (Phase 2) |

## Engineering Constraints & Risks

1. **VRAM Bandwidth:** The Dense-Highway (gated anchor + per-op residuals) requires keeping `embed` in memory across all layers for backpropagation. MoE compounds this by running multiple expert chains.
2. **Signal magnitude:** Dense residuals + gated anchor can cause signal explosion. Mitigated by: learnable `RMSNorm` before every op, anchor gate initialized near 0, output projection depth scaling.
3. **MoE load balancing:** Per-sequence routing (not per-token) avoids standard load balancing issues but may underutilize experts if routing becomes degenerate. Monitor expert utilization during training.

## Differences from Old Ripple

| Aspect | Old Ripple | New (DS-MoE) |
|--------|-----------|-----|
| Norm placement | Post-norm inside RippleAttention per-op | Pre-norm at block level, learnable RMSNorm |
| Residual | Internal to RippleAttention + block-level | Only at block level: `x = x + op(norm(x))` + gated anchor |
| Block norms | RMSNorm with learnable scale per-op | Learnable RMSNorm per-op at block level |
| Composition | Single RippleAttention with order string | Explicit `[DS1, DiffAttn, DiffMLP]` super-block |
| Embed anchor | Optional `embed_residual` flag (additive) | Gated anchor with learnable gate initialized near 0 |
| MoE | Channel-split experts (D // n_channels per expert) | Full-width relay MoE with SE rerouting at segment boundaries |
| ND support | Separate ND op variants (Conv2d, Conv3d in SSM) | 1D ops only; ND handled at embedding level |
| Positional encoding (1D) | Learned pos_embed + RoPE | RoPE only (no learned positional) |
| Default ops per block | Configurable (usually SSM + Attn) | Always SSM + Attn + MLP |
| Classification head | GAP + Linear | ML-Decoder (cross-attention with learned class queries) |

## What We Know Works (from zoology experiments)

- `[DS1, DiffAttn]` order beats attention alone on MQAR
- Differential inject + readout + BC-norm on the SSM are all beneficial
- Squeeze-excite on diffusion helps
- `n_iters=2` is sufficient (more doesn't help much, 1 is too few)
- `state_dim=64, mimo_rank=4` are good defaults for d_model=32-512

## What We Need to Validate

1. **Does DS1 behave like an SSM?** Run on selective copying, induction heads, S4-style benchmarks. Compare to Mamba, S4, H3.
2. **Does the differential mechanism help?** Ablate diff inject/readout/λ on the new architecture (old results were with broken training loop).
3. **DiffMLP in the super-block**: Previous benchmarks showed DiffMLP within noise on classification. Does the 3-op super-block (SSM+Attn+MLP) outperform 2-op (SSM+Attn)?
4. **Gated anchor**: Does the embed anchor improve convergence? Compare with/without on MQAR and language modeling.
5. **Training loop correctness**: Match zoology accuracy on MQAR with the new architecture before moving to harder tasks.
6. **Speed**: Profile each op individually. The SSM's 6 projections * n_iters could be expensive — consider fusing or caching.
7. **Relay MoE**: Does segment-locked routing outperform dense (every-layer) routing? What's the optimal segment size?
8. **Text diffusion**: Does the backbone work well in a denoising diffusion setting for text? Does the gated anchor help convergence?

## Implementation Components

Each component is listed with its dependencies and acceptance test. Implementation order follows dependency order — foundation first, then ops, then composition, then shells.

### Phase 1: Foundation (no dependencies)

**C1. RMSNorm** — Learnable RMSNorm class with `weight` parameter. Used for block pre-norms, QK-norm, head_norm, BC-norm. Test: output has unit RMS, gradient flows through weight.

**C2. RoPE (standard)** — `apply_rope(x, freqs)` using concat-half layout: `x1, x2 = x[..., :half], x[..., half:]`. For DiffAttn Q/K. Test: rotation is norm-preserving, different positions get different rotations.

**C3. RoPE (interleaved)** — `apply_interleaved_rope(x, theta)` using even/odd layout: `x[..., ::2], x[..., 1::2]`. For DS1 B/C with iteration-scaled theta. Test: same rotation properties as C2, interleaved index layout confirmed.

**C4. SqueezeExcite** — `SqueezeExcite(channels, reduction=4)`: GAP → fc1 → silu → fc2 → sigmoid → scale. Min hidden=8. Reused by DS1 diffusion, MoE routing, and SE final head. Test: output shape == input shape, scale is sigmoid-bounded.

**C5. Weight init** — `init_weights(model, n_layers)` function: all Linear weights `normal_(std=0.02)`, all biases `zeros_()`, `out_proj.weight` scaled by `1/sqrt(2*n_layers)`, anchor gate `weight=zeros, bias=-2.0`. Test: verify statistics on a constructed model.

### Phase 2: Core Ops (depend on Phase 1)

**C6. DiffAttn** — Differential multi-head attention. Depends on C1 (RMSNorm for q/k/head norms), C2 (RoPE for Q/K). Interface: `forward(x, qkvo_w) → y` — receives weight slice from attn_bank, no stored projections. Only stores config (num_heads, layer_idx) and small learned params (λ vectors, QK-norm scales, head_norm). Configurable: `num_heads`, `layer_idx` (for λ_init), `differential` flag, `dropout`. Full spec in §Core Ops.1. Test: causal mask correct (future tokens get zero attention), λ is learnable, output matches old `CausalSelfAttention` on identical weights/input.

**C7. DS1** — Diffusive State SSM with iterative convolution. Depends on C1 (RMSNorm for BC-norm), C3 (interleaved RoPE), C4 (SqueezeExcite for diffusion). Interface: `forward(x, ssm_w) → y` — receives weight slice from ssm_bank. Only stores config and small params (B/C bias, diffuse conv, optional SE, optional BC-norm, optional diff λ). Configurable: `state_dim`, `mimo_rank`, `n_iters`, `diffuse_se`, `diff_inject`, `diff_readout`, `bc_norm`, `relu2`. Full spec in §Core Ops.2. Test: gradient flows through all 6 projections (the bug we found), output matches old `MIMOJacobiSSM` on identical weights/input.

**C8. DiffMLP** — Differential SwiGLU. Interface: `forward(x, mlp_w) → y` — receives weight slice from mlp_bank. Only stores `lambda_logit` scalar. Configurable: `mult`, `lambda_init`. Full spec in §Core Ops.3. Test: output matches old `DifferentialSwiGLU` on identical weights/input, hidden dim uses 2/3 scaling.

**Note on bank vs. stored params:** Large projection matrices (QKV, out_proj, gate, up, down, to_B, to_C, etc.) live in banks for optimizer efficiency. Small per-op params (λ vectors, norm scales, conv weights, biases) remain stored on the op module — they are too small to benefit from banking and have different optimizer configs (Adam, not NorMuon).

### Phase 3: Composition (depend on Phase 2)

**C9. Block (Super-Block)** — Pre-norm + residual for each op + gated anchor. Depends on C6, C7, C8 (ops), C1 (learnable RMSNorm per op). Interface: `forward(x, embed, attn_w, ssm_w, mlp_w) → x`. Contains: `norm_attn, norm_ssm, norm_mlp: RMSNorm(dim)`, `anchor_gate: Linear(dim, dim)` with `weight=zeros, bias=-2.0`. Full spec in §Block Structure. Test: anchor gate sigmoid starts ≈0.12, all 3 ops fire in sequence, `embed` gradient flows through anchor.

**C10. Token Embedding (1D)** — `nn.Embedding(vocab_size, dim)` + `nn.Dropout(embed_dropout)`. No positional embedding — RoPE handles position. Test: correct shape `(B, L, D)`, dropout active in train mode.

**C11. Sinusoidal 2D Embedding** — `LearnedSinusoidal2DEmbed(height, width, dim)` with learnable per-axis frequencies, phases, and projection. Outputs `(B, H*W, D)`. Port from old `LearnedSinusoidal2DEmbed` in `ripple_attention.py`. Depends on C1 (RMSNorm). Test: correct flattened shape, gradients flow to freq/phase params.

**C12. ML-Decoder Head** — Cross-attention classification head with grouped learnable queries. Port from old `MLDecoderHead` in bench script. Interface: `forward(x: (B,L,D)) → (B, n_classes)`. Test: correct output shape, gradient flows through learned queries.

**C13. SE Final Head** — SqueezeExcite recalibration before linear vocab projection (for MoE models). Depends on C4. Interface: `forward(x: (B,L,D)) → (B, L, vocab_size)`. Test: SE is applied before projection, output shape correct.

### Phase 4: Model Shells (depend on Phase 3)

**C14. LM** — Autoregressive language model. Depends on C9 (Block), C10 (embed), C1 (final RMSNorm), C5 (init). `forward(idx) → (B, L, vocab_size)`. Weight tying: `head.weight = embed.weight`. Passes `embed` to every block for gated anchor. Test: full forward pass produces correct shape, loss decreases on toy data.

**C15. Classifier** — Sequence classifier. Depends on C9 (Block), C10 or C11 (embed), C1 (norms), C12 (ML-Decoder head), C5 (init). `forward(x) → (B, n_classes)`. Uses learned positional embedding for classifier (not RoPE-only). Test: forward produces correct shape, trains on MQAR.

**C16. Zoology Mixer Wrapper** — Thin wrapper so DS-MoE slots into zoology's `TransformerBlock`. Interface matches zoology's `sequence_mixer(hidden_states) → hidden_states`. Wraps a single Block's ops (without anchor, since zoology handles residual/norm). Test: zoology training loop runs, matches old RippleMixer accuracy.

### Phase 5: MoE (depend on Phase 4)

**C17. Relay MoE** — Segment-locked routing with SE mix-and-reroute. Depends on C9 (Block), C4 (SE). Contains: per-segment routers, expert block chains, SE mix layers. Configurable: `n_experts`, `top_k`, `segment_size`. Full spec in §Macro-Architecture. Test: routing decisions are per-sequence (not per-token), experts locked for full segment, SE fires at boundaries.

**C18. MoE LM** — MoE variant of LM shell. Depends on C17, C13 (SE head), C10, C1, C5. Final head uses SE before projection. Test: full forward pass with routing, expert utilization is non-degenerate.

### Phase 6: Diffusion (depend on Phase 4)

**C19. Diffusion LM** — Noise-conditioned backbone for text diffusion. Depends on C9, C10, C1, C5. Design TBD — needs noise schedule, conditioning mechanism, sampling strategy. The gated anchor is expected to be critical for conditioning preservation.

### Validation Milestones

| Milestone | Components | Criterion |
|-----------|-----------|-----------|
| **M1: Ops match old code** | C6, C7, C8 | Given identical weights and input, new ops produce same output as old `CausalSelfAttention`, `MIMOJacobiSSM`, `DifferentialSwiGLU` |
| **M2: Gradient health** | C7 especially | DS1 input projection gradients are same order of magnitude as zoology (the root cause we diagnosed) |
| **M3: MQAR baseline** | C14 or C16 | Match zoology 99%+ on seq_len=256, num_kv_pairs=16 with known-good config |
| **M4: Super-block value** | C14 | 3-op (SSM+Attn+MLP) outperforms 2-op (SSM+Attn) on MQAR or LM loss |
| **M5: Anchor value** | C14 | Gated anchor improves convergence vs no anchor |
| **M6: MoE works** | C18 | Relay MoE trains without degenerate routing, matches or beats dense baseline |

## File Layout

Following the convention of production LLM codebases (LLaMA, modded-nanogpt), the implementation is consolidated into minimal files:

```
src/ds_moe/
    __init__.py
    model.py              # Everything: norms, RoPE, SE, ops (DiffAttn, DS1, DiffMLP),
                          #   Block, embeds, heads (MLDecoder, SE head), model shells
                          #   (LM, Classifier), weight init
    moe.py                # Relay MoE only: segment routing, SE mix-and-reroute,
                          #   expert chains, MoE model shells
```

`model.py` is the single-file core — contains all components needed for a non-MoE model. Reading one file gives you the complete architecture.

`moe.py` is optional — only needed when using expert routing. Imports from `model.py`.
