# Feature Attention: Replacing SwiGLU Gating with Self-Attention Over Token Features

Anonymous draft (Markdown preprint)

---

## Abstract

We investigate **feature attention** — replacing the element-wise gating in a SwiGLU MLP with self-attention over the token's own hidden features. In a standard SwiGLU block, `gate_proj` produces per-feature gates and `up_proj` produces values, combined via element-wise multiplication: $\text{SiLU}(g) \odot v$. Feature attention replaces this with cross-feature interaction: `gate_proj` output serves as shared Q=K (Reformer-style), `up_proj` output serves as V, and attention over the feature dimension determines how values are mixed. The `down_proj` is unchanged.

This replacement adds **zero parameters** — it reinterprets the same three projection matrices (gate/up/down) that SwiGLU already uses. We sweep factorization ratios of the hidden dimension from 32:1 to 1:32 across three activation functions (softmax, SiLU, SiLU²) on a ~124M-parameter model trained on FineWeb.

---

## 1. Introduction

The SwiGLU MLP block in modern transformers applies element-wise gating between two parallel projections:

$$
\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x)
$$

This gating operates independently per hidden dimension — feature $i$'s gate only controls feature $i$'s value. There is no cross-feature interaction at this stage; the only mixing happens in the linear projections themselves.

We ask: **what happens if we replace element-wise gating with cross-feature attention?**

Feature attention reinterprets the three SwiGLU projections as an attention operation:

- $W_{\text{gate}}$ → **shared Q = K** (Reformer-style): determines which features are similar
- $W_{\text{up}}$ → **V**: provides the values to be mixed
- $W_{\text{down}}$ → output projection (unchanged)

The gate and up outputs are reshaped into $(N_f, D_f)$ feature matrices, and attention over $N_f$ features replaces the element-wise product. This introduces **content-dependent cross-feature gating** at zero additional parameter cost.

### 1.1 Motivation

In SwiGLU, each feature independently decides its own gate value. This is efficient but local — feature $i$ cannot influence whether feature $j$ passes through.

Feature attention makes gating global: whether feature $i$ contributes to the output depends on its relationship to all other features in the same token. A feature with a strong gate signal can amplify correlated features and suppress anti-correlated ones, enabling within-token feature coordination that element-wise gating cannot express.

### 1.2 Relation to prior work

- **Reformer** (Kitaev et al. 2020): Introduced shared-QK attention (Q and K use the same projection matrix) for sequence attention. They found that with shared QK, a token's dot product with itself dominates, requiring diagonal masking. Our setup uses shared QK from `gate_proj` with separate V from `up_proj`, which avoids the worst of this collapse — even if attention concentrates on the diagonal, it retrieves `up_proj` values rather than the gate values themselves.
- **GLU variants** (Shazeer 2020, Dauphin et al. 2017): Gated linear units introduce feature interaction through element-wise gating ($\sigma(Wx) \odot Vx$), but the interaction is pairwise between matched dimensions, not all-to-all.
- **Mixture of Experts**: MoE routes tokens to different MLPs but does not change the within-MLP nonlinearity.
- **Feature attention** is orthogonal to all of the above — it replaces the within-MLP gating mechanism with attention over the feature dimension, leaving the sequence-level mechanism and the projection matrices unchanged.

---

## 2. Method

### 2.1 SwiGLU baseline

For model dimension $D = 768$ with standard $8/3$ expansion (hidden dim $H = 2048$, snapped to 256 for hardware efficiency):

$$
g = W_{\text{gate}} \cdot x, \quad v = W_{\text{up}} \cdot x, \qquad W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{D \times H}
$$
$$
\text{SwiGLU}(x) = W_{\text{down}} \cdot (\text{SiLU}(g) \odot v), \qquad W_{\text{down}} \in \mathbb{R}^{H \times D}
$$

### 2.2 Feature attention replacement

We replace $\text{SiLU}(g) \odot v$ with attention where $g$ provides Q=K and $v$ provides V:

1. **Project**: $g = W_{\text{gate}} \cdot x$, $v = W_{\text{up}} \cdot x$ — same projections as SwiGLU.

2. **Reshape**: $g \to \hat{g} \in \mathbb{R}^{(BT) \times N_f \times D_f}$, $v \to \hat{v} \in \mathbb{R}^{(BT) \times N_f \times D_f}$ where $N_f \cdot D_f = H$.

3. **Score** (shared Q=K from gate): $S = \hat{g} \hat{g}^\top / \sqrt{D_f} \in \mathbb{R}^{(BT) \times N_f \times N_f}$

4. **Activate**: $A = f(S)$ where $f$ is one of:
   - **Softmax**: $A_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})}$ — normalized, non-negative.
   - **SiLU**: $A_{ij} = \text{SiLU}(S_{ij})$ — unnormalized, can be negative.
   - **SiLU²**: $A_{ij} = \text{SiLU}(S_{ij})^2$ — unnormalized, non-negative, sharper.

5. **Attend** (values from up_proj): $\hat{o} = A \hat{v} \in \mathbb{R}^{(BT) \times N_f \times D_f}$

6. **Reshape and project**: $o \in \mathbb{R}^{B \times T \times H}$, then $\text{out} = W_{\text{down}} \cdot o$.

### 2.3 Comparison to SwiGLU

The key structural difference:
- **SwiGLU**: $\text{out}_i = \text{SiLU}(g_i) \cdot v_i$ — each feature gated independently by its own gate value.
- **Feature attention**: $\text{out}_i = \sum_j A_{ij} \cdot v_j$ — each feature is a weighted combination of *all* features' values, weighted by gate similarity.

When the attention matrix $A$ is diagonal, feature attention degenerates to per-feature scaling (similar to SwiGLU). When $A$ has off-diagonal mass, it enables cross-feature mixing that SwiGLU cannot express.

### 2.4 Factorization ratios

For $H = 2048$ ($= 2^{11}$), we test six power-of-two factorizations:

| Label | $N_f$ | $D_f$ | Ratio $N_f : D_f$ | Attention matrix size |
|-------|--------|--------|--------------------|-----------------------|
| 32:1  | 256    | 8      | 32                 | $256 \times 256$      |
| 8:1   | 128    | 16     | 8                  | $128 \times 128$      |
| 2:1   | 64     | 32     | 2                  | $64 \times 64$        |
| 1:2   | 32     | 64     | 0.5                | $32 \times 32$        |
| 1:8   | 16     | 128    | 0.125              | $16 \times 16$        |
| 1:32  | 8      | 256    | 0.03               | $8 \times 8$          |

At $B = 8, T = 2048$, the largest attention matrix ($256 \times 256 \times 16384$ tokens) occupies ~2 GB in bf16. All others are smaller.

### 2.5 Properties

**Parameter count**: Identical to SwiGLU — 123,587,328 for all configurations. Same three projection matrices, same 12 layers.

**Compute cost**: Feature attention replaces the element-wise multiply ($O(H)$ per token) with two matmuls: scoring $O(N_f^2 \cdot D_f)$ and attending $O(N_f^2 \cdot D_f)$. For the largest factorization ($N_f = 256, D_f = 8$): $2 \times 256^2 \times 8 = 1,\!048,\!576$ FLOPs per token per layer — small relative to the projection matmuls ($3 \times 2 \times D \times H \approx 9.4$M FLOPs).

**Shared Q=K and diagonal dominance**: With Q = K = gate_proj output, the score matrix diagonal is $\|g_i\|^2 / \sqrt{D_f}$, which may dominate. Under softmax, this biases attention toward the diagonal. However, since V comes from a separate projection (`up_proj`), even diagonal-dominant attention performs meaningful computation — it's reweighting `up_proj` values by `gate_proj` norms, not collapsing to identity.

---

## 3. Experimental Setup

### 3.1 Model

- **Architecture**: GPT-style decoder-only transformer
- **Parameters**: 123,587,328 (~124M, shared across all configurations)
- **Sequence attention**: Standard multi-head self-attention, 12 heads, 64 dim/head, RoPE
- **MLP**: SwiGLU with 8/3 expansion (768 → 2048 → 768), gating varies by experiment
- **Normalization**: RMSNorm (pre-norm)
- **Sequence length**: 2048

### 3.2 Training

- **Data**: FineWeb-10B
- **Steps**: 20,000
- **Batch size**: 8
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95)
- **Learning rate**: 3e-4 with cosine decay, 200 step warmup
- **Weight decay**: 0.1
- **Gradient clipping**: 1.0
- **Precision**: bf16 autocast

### 3.3 Configurations

18 feature attention variants (6 ratios × 3 activations) plus 1 SwiGLU baseline = **19 runs**.

| Run ID | $N_f$ | $D_f$ | Activation | Notes |
|--------|--------|--------|------------|-------|
| baseline | — | — | SwiGLU | Element-wise SiLU gating |
| fa-256x8-softmax | 256 | 8 | softmax | |
| fa-256x8-silu | 256 | 8 | silu | |
| fa-256x8-silu2 | 256 | 8 | silu² | |
| fa-128x16-softmax | 128 | 16 | softmax | |
| fa-128x16-silu | 128 | 16 | silu | |
| fa-128x16-silu2 | 128 | 16 | silu² | |
| fa-64x32-softmax | 64 | 32 | softmax | |
| fa-64x32-silu | 64 | 32 | silu | |
| fa-64x32-silu2 | 64 | 32 | silu² | |
| fa-32x64-softmax | 32 | 64 | softmax | |
| fa-32x64-silu | 32 | 64 | silu | |
| fa-32x64-silu2 | 32 | 64 | silu² | |
| fa-16x128-softmax | 16 | 128 | softmax | |
| fa-16x128-silu | 16 | 128 | silu | |
| fa-16x128-silu2 | 16 | 128 | silu² | |
| fa-8x256-softmax | 8 | 256 | softmax | |
| fa-8x256-silu | 8 | 256 | silu | |
| fa-8x256-silu2 | 8 | 256 | silu² | |

---

## 4. Results

### 4.1 Validation loss curves

<!-- TODO: Insert loss curves. One plot per activation function with all 6 ratios overlaid, plus baseline. -->

#### Softmax activation

```
[PLACEHOLDER: loss curve plot — softmax variants vs baseline]
```

#### SiLU activation

```
[PLACEHOLDER: loss curve plot — silu variants vs baseline]
```

#### SiLU² activation

```
[PLACEHOLDER: loss curve plot — silu² variants vs baseline]
```

#### Best of each activation

```
[PLACEHOLDER: loss curve plot — best ratio from each activation + baseline]
```

### 4.2 Final validation loss

<!-- TODO: Fill in after runs complete -->

| Run ID | Val loss (20k steps) | Δ vs baseline |
|--------|----------------------|---------------|
| baseline | | — |
| fa-256x8-softmax | | |
| fa-256x8-silu | | |
| fa-256x8-silu2 | | |
| fa-128x16-softmax | | |
| fa-128x16-silu | | |
| fa-128x16-silu2 | | |
| fa-64x32-softmax | | |
| fa-64x32-silu | | |
| fa-64x32-silu2 | | |
| fa-32x64-softmax | | |
| fa-32x64-silu | | |
| fa-32x64-silu2 | | |
| fa-16x128-softmax | | |
| fa-16x128-silu | | |
| fa-16x128-silu2 | | |
| fa-8x256-softmax | | |
| fa-8x256-silu | | |
| fa-8x256-silu2 | | |

### 4.3 Effect of factorization ratio

<!-- TODO: Plot val loss at 20k steps vs N_f, one line per activation. This is the key plot — does the optimal ratio differ by activation? -->

```
[PLACEHOLDER: scatter/line plot — final val loss vs N_f, grouped by activation]
```

### 4.4 Effect of activation function

<!-- TODO: For each factorization ratio, which activation wins? Is there a consistent winner? -->

```
[PLACEHOLDER: grouped bar chart — final val loss by activation, grouped by ratio]
```

### 4.5 Training dynamics

<!-- TODO: Do any variants show instability? Do silu/silu² variants diverge at certain ratios? How does loss curvature differ early vs late in training? -->

```
[PLACEHOLDER: training loss (not val) curves for any interesting dynamics — divergence, instability, etc.]
```

### 4.6 Throughput

<!-- TODO: Wall-clock time per step for each factorization. Feature attention adds matmuls — how much does it actually cost? -->

| $N_f$ | sec/step (feature attn) | sec/step (baseline) | Overhead |
|--------|--------------------------|----------------------|----------|
| 256 | | | |
| 128 | | | |
| 64 | | | |
| 32 | | | |
| 16 | | | |
| 8 | | | |

---

## 5. Analysis

### 5.1 What does feature attention learn?

<!-- TODO: Visualize attention patterns. For a trained model, extract the feature attention matrices and look at: (1) are they sparse or dense? (2) do certain features consistently attend to each other? (3) does the pattern differ by layer? (4) how diagonal-dominant is the attention? -->

```
[PLACEHOLDER: heatmaps of feature attention matrices at different layers]
```

### 5.2 Diagonal dominance

<!-- TODO: Measure what fraction of attention weight lands on the diagonal across layers and training. Does it start diagonal and diversify, or stay diagonal? Compare softmax (which normalizes, forcing diagonal dominance) vs silu/silu² (which don't). -->

```
[PLACEHOLDER: diagonal mass fraction over training steps, by activation]
```

### 5.3 Feature specialization

<!-- TODO: Do individual features specialize? Compare feature activation distributions between SwiGLU baseline and feature attention variants. Does feature attention encourage or suppress dead features? -->

```
[PLACEHOLDER: feature activation histograms — baseline vs best feature attention variant]
```

### 5.4 Gradient norms

<!-- TODO: Compare gradient norms through the MLP across variants. Does feature attention change gradient flow meaningfully? -->

```
[PLACEHOLDER: per-layer gradient norm comparison]
```

---

## 6. Discussion

<!-- TODO: Write after experiments. Key questions to address:

1. Does feature attention beat SwiGLU at any configuration?
2. Is there a clear best activation function?
3. Is there a clear best factorization ratio, and does it interact with activation choice?
4. Is the compute overhead worth it?
5. How diagonal-dominant is the learned attention? Does it effectively collapse to SwiGLU?
6. Does the answer change with scale? (motivation for future work at larger sizes)
-->

### 6.1 Follow-up: projection variants

The initial study uses shared Q=K from `gate_proj` and V from `up_proj` — a direct reinterpretation of SwiGLU's three matrices. If feature attention shows promise, there are natural follow-ups that introduce additional learned projections within the feature space:

**Separate Q and K projections**:
- Learn $W_Q, W_K \in \mathbb{R}^{D_f \times D_f}$ applied per-feature to the gate output.
- Breaks the symmetry of the score matrix (Q=K gives symmetric scores).
- Adds $2 \times D_f^2$ parameters per layer. At $D_f = 32$: 2,048 params/layer.

**Grouped-query feature attention (GQA-style)**:
- Q projections per feature, but K/V shared across groups of features.
- E.g., with $N_f = 64$ and 8 KV groups: each group of 8 features shares K/V.
- Reduces attention matrix cost at high $N_f$.

**Multi-head feature attention**:
- Split each feature's descriptor into multiple heads.
- E.g., with $D_f = 32$ and 4 heads: each head attends over $D_f/4 = 8$ dimensions.
- Allows different attention patterns at different descriptor scales.

These would be run at whichever factorization ratio(s) win the initial sweep.

<!-- TODO: If initial results are promising, add rows to the results table for these variants. -->

| Run ID | Variant | Extra params/layer | Val loss (20k) | Δ vs shared QK |
|--------|---------|--------------------|----------------|----------------|
| | separate Q/K | | | |
| | GQA-style | | | |
| | multi-head | | | |

---

## 7. Conclusion

<!-- TODO: Write after experiments. -->

---

## Appendix A: Implementation Details

Feature attention is implemented as a drop-in replacement for SwiGLU gating. The `FeatureAttentionMLP` class uses the same three projection matrices:

```python
class FeatureAttentionMLP(nn.Module):
    def __init__(self, d_model, n_features, activation='softmax'):
        hidden = int(d_model * 8 / 3)
        hidden = ((hidden + 255) // 256) * 256  # snap for hardware
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)  # → Q=K
        self.up_proj   = nn.Linear(d_model, hidden, bias=False)  # → V
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qk = self.gate_proj(x).view(B*T, n_features, desc_dim)  # shared Q=K
        v  = self.up_proj(x).view(B*T, n_features, desc_dim)    # V

        scores = bmm(qk, qk.T) / sqrt(desc_dim)  # (B*T, NF, NF)
        weights = activation(scores)               # softmax / silu / silu²
        out = bmm(weights, v)                      # (B*T, NF, DD)

        return self.down_proj(out.view(B, T, hidden))
```

Configuration is via environment variables:
- `MODEL_TYPE=feat_attn` — select feature attention model
- `FA_N_FEATURES=64` — number of features (must divide 2048)
- `FA_ACTIVATION=softmax` — attention activation function

Valid `FA_N_FEATURES` values for hidden=2048: 8, 16, 32, 64, 128, 256.

All other hyperparameters (layers, heads, dim, training schedule) are identical to the baseline.

## Appendix B: Run commands

```bash
# SwiGLU Baseline
MODEL_TYPE=transformer TRAIN_STEPS=20000 python train_fineweb_vanilla.py

# Feature attention sweep (example for one ratio)
for act in softmax silu silu2; do
    MODEL_TYPE=feat_attn FA_N_FEATURES=64 FA_ACTIVATION=$act \
        TRAIN_STEPS=20000 python train_fineweb_vanilla.py
done

# Full sweep
for nf in 256 128 64 32 16 8; do
    for act in softmax silu silu2; do
        MODEL_TYPE=feat_attn FA_N_FEATURES=$nf FA_ACTIVATION=$act \
            TRAIN_STEPS=20000 python train_fineweb_vanilla.py
    done
done
```
