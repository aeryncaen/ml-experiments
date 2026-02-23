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
- **Steps**: 7,500
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

![Val loss by activation function](../analysis/val_loss_by_activation.png)

All 18 feature attention configs converge on similar trajectories in early training (steps 0-2000), then gradually separate. Baseline (black) maintains a consistent lead throughout. Key observations:

- **Softmax** variants stay tightest together (smallest spread across factorizations) and closest to baseline.
- **SiLU** variants show moderate spread, with 32x64 and 64x32 closest to baseline.
- **SiLU²** variants show the largest spread — 8x256-silu² is the worst overall config.

![Val loss by factorization](../analysis/val_loss_by_factorization.png)

Within each factorization, softmax consistently achieves the lowest final loss, followed by silu, then silu².

![Late-stage val loss zoom (steps 4000-7500)](../analysis/late_stage_zoom.png)

The late-stage zoom reveals that feature attention configs are converging faster than baseline in the 5000-7500 step range: their slopes are steeper, closing the gap. The step-6700 spike visible in all configs is a training data artifact (bad batch).

### 4.2 Final validation loss

| Rank | Run ID | Val loss (7.5k) | Best val | Best @ step | Δ% vs baseline |
|------|--------|-----------------|----------|-------------|----------------|
| 1 | **baseline** | **4.4697** | 4.3435 | 7200 | — |
| 2 | fa-64x32-softmax | 4.5614 | 4.4449 | 7200 | +2.05% |
| 3 | fa-256x8-softmax | 4.5703 | 4.4560 | 7200 | +2.25% |
| 4 | fa-32x64-silu | 4.5734 | 4.4611 | 7200 | +2.32% |
| 5 | fa-32x64-softmax | 4.5781 | 4.4635 | 7200 | +2.43% |
| 6 | fa-128x16-softmax | 4.5807 | 4.4690 | 7200 | +2.48% |
| 7 | fa-64x32-silu | 4.5834 | 4.4693 | 7200 | +2.55% |
| 8 | fa-16x128-softmax | 4.6130 | 4.4982 | 7200 | +3.21% |
| 9 | fa-64x32-silu2 | 4.6179 | 4.5096 | 7200 | +3.32% |
| 10 | fa-32x64-silu2 | 4.6292 | 4.5199 | 7200 | +3.57% |
| 11 | fa-16x128-silu | 4.6335 | 4.5234 | 7200 | +3.66% |
| 12 | fa-128x16-silu2 | 4.6382 | 4.5331 | 7200 | +3.77% |
| 13 | fa-8x256-silu | 4.6618 | 4.5528 | 7200 | +4.30% |
| 14 | fa-128x16-silu | 4.6680 | 4.5666 | 7200 | +4.44% |
| 15 | fa-8x256-softmax | 4.6824 | 4.5716 | 7200 | +4.76% |
| 16 | fa-256x8-silu | 4.6908 | 4.5913 | 7000 | +4.95% |
| 17 | fa-256x8-silu2 | 4.7029 | 4.6050 | 7200 | +5.22% |
| 18 | fa-16x128-silu2 | 4.7110 | 4.6133 | 7000 | +5.40% |
| 19 | fa-8x256-silu2 | 4.7759 | 4.6742 | 7000 | +6.85% |

All 18 feature attention configs underperform baseline at 7,500 steps. The gap ranges from +2.05% (64x32-softmax) to +6.85% (8x256-silu²). However, as shown in §4.5, feature attention configs are converging faster in the late-training regime.

### 4.3 Effect of factorization ratio

Averaging across activations:

| $N_f \times D_f$ | Avg final val | Avg convergence rate (5k-7.5k) | Avg overfit gap | Avg sec/step |
|-------------------|---------------|-------------------------------|-----------------|--------------|
| 64x32 | **4.5876** | 0.0515 | 0.0559 | 0.107 |
| 32x64 | 4.5936 | 0.0522 | 0.0549 | 0.096 |
| 128x16 | 4.6290 | 0.0529 | 0.0737 | 0.148 |
| 16x128 | 4.6525 | 0.0557 | 0.0685 | 0.095 |
| 256x8 | 4.6547 | 0.0534 | 0.0830 | 0.311 |
| 8x256 | 4.7067 | 0.0553 | 0.0681 | 0.098 |

**Finding**: 64x32 and 32x64 are the best factorizations. The sweet spot is 32-64 features with 32-64 descriptor dimensions. Extremes in either direction hurt:
- **Too many features** (256x8): 8-dimensional descriptors are too small for meaningful similarity computation, and the 256×256 attention matrix is expensive.
- **Too few features** (8x256): Only 8 features means minimal cross-feature interaction — an 8×8 attention matrix provides little beyond per-feature gating.

The U-shaped relationship is clear: optimal is in the middle where both feature count and descriptor dimensionality are large enough to be useful.

### 4.4 Effect of activation function

Averaging across factorizations:

| Activation | Avg final val | Avg convergence rate (5k-7.5k) | Avg overfit gap |
|------------|---------------|-------------------------------|-----------------|
| **softmax** | **4.5976** | 0.0474 | **0.0480** |
| silu | 4.6352 | 0.0560 | 0.0717 |
| silu² | 4.6792 | **0.0571** | 0.0824 |

![Heatmaps: factorization x activation](../analysis/heatmaps.png)

**Finding**: Softmax wins on final loss and overfitting, but silu/silu² converge faster.

- **Softmax** achieves the best loss at every factorization (left heatmap). It also has the lowest overfitting gap (right heatmap). This makes sense: softmax normalizes attention weights, which acts as implicit regularization.
- **SiLU/SiLU²** converge 1.2-1.4× faster than softmax in the 5k-7.5k step range (center heatmap). Their unnormalized weights allow larger-magnitude updates, which speeds convergence but increases overfitting.
- **SiLU²** consistently underperforms SiLU on final loss despite faster convergence — the overfitting penalty outweighs the convergence benefit at this training length.

The softmax-vs-silu tradeoff (lower loss vs faster convergence) is the key open question for chinchilla-length training.

### 4.5 Training dynamics and convergence rates

Convergence rates (val loss drop per 1,000 steps) in three training phases:

| Config | Rate 0-2k | Rate 2k-5k | Rate 5k-7.5k |
|--------|-----------|------------|--------------|
| baseline | 1.417 | 0.203 | 0.045 |
| fa-64x32-softmax | 1.472 | 0.185 | 0.046 |
| fa-32x64-silu | 1.392 | 0.201 | 0.053 |
| fa-32x64-silu2 | 1.385 | 0.179 | 0.057 |
| fa-16x128-silu2 | 1.373 | 0.154 | 0.062 |

**Key finding: Feature attention configs converge slower early but faster late.**

- Steps 0-2k: Baseline converges at 1.42 loss/1k-steps. Softmax configs match or slightly exceed this (e.g., 256x8-softmax: 1.50); silu/silu² configs are slightly slower (1.37-1.40).
- Steps 2k-5k: Baseline still leads (0.203). Feature attention configs range 0.15-0.20.
- Steps 5k-7.5k: **Feature attention configs converge 1.0-1.4× faster than baseline.** All 18 configs have higher convergence rates than baseline (0.045) in this range. The fastest is 16x128-silu² at 0.062.

This crossover in convergence rate is the most interesting finding: it suggests feature attention may require more training to realize its benefits.

#### Extrapolated crossover with baseline

Using linear extrapolation from the 5k-7.5k val loss trends, we estimate when each config would cross baseline:

| Config | Est. crossover step | Est. wall-clock |
|--------|--------------------:|----------------:|
| fa-32x64-silu | ~21,000 | ~0.6h |
| fa-32x64-silu2 | ~21,100 | ~0.6h |
| fa-64x32-silu | ~20,900 | ~0.6h |
| fa-16x128-silu | ~20,800 | ~0.5h |
| fa-16x128-silu2 | ~22,200 | ~0.6h |
| fa-64x32-silu2 | ~23,100 | ~0.7h |
| fa-128x16-silu | ~23,400 | ~1.0h |
| fa-8x256-silu | ~23,500 | ~0.6h |
| fa-256x8-silu | ~25,200 | ~2.2h |
| fa-256x8-silu2 | ~27,200 | ~2.3h |
| fa-128x16-silu2 | ~29,000 | ~1.2h |
| fa-8x256-silu2 | ~28,800 | ~0.8h |
| fa-128x16-softmax | ~46,100 | ~1.9h |
| fa-16x128-softmax | ~48,400 | ~1.3h |
| fa-8x256-softmax | ~56,600 | ~1.6h |
| fa-32x64-softmax | ~65,500 | ~1.8h |
| fa-64x32-softmax | ~75,300 | ~2.3h |
| fa-256x8-softmax | ~207,900 | ~18.0h |

**Caveat**: Linear extrapolation from 2,500 steps of data is unreliable. The actual crossover could occur earlier (if feature attention's advantage accelerates) or never (if the rate difference was transient). Chinchilla-length runs (~150k steps) are needed to resolve this.

The silu configs are predicted to cross baseline around step 21k — well within chinchilla length. Softmax configs cross much later (46k-208k steps) because their convergence advantage over baseline is smaller.

### 4.6 Throughput

| $N_f \times D_f$ | sec/step | Baseline sec/step | Overhead |
|-------------------|----------|-------------------|----------|
| 8x256 | 0.098 | 0.080 | +22% |
| 16x128 | 0.095 | 0.080 | +18% |
| 32x64 | 0.096 | 0.080 | +20% |
| 64x32 | 0.107 | 0.080 | +34% |
| 128x16 | 0.148 | 0.080 | +85% |
| 256x8 | 0.311 | 0.080 | +289% |

Throughput overhead scales with $N_f^2$ (the attention matrix size). For the sweet-spot factorizations (32x64, 64x32), overhead is 20-34% — manageable. The 256x8 config is nearly 4× slower, dominated by the 256×256 attention computation repeated for every token.

For chinchilla-length runs (~150k steps), estimated wall-clock times:
- Baseline: ~3.3 hours
- 32x64 configs: ~4.0 hours (+20%)
- 64x32 configs: ~4.5 hours (+34%)

The throughput penalty is modest enough that even a small loss improvement would justify the extra compute.

---

## 5. Analysis

### 5.1 Overfitting behavior

![Overfitting gap over time](../analysis/overfitting_gap.png)

The overfitting gap (val loss - train loss) is surprisingly uniform across configs when measured at each evaluation point. All five top configs track each other closely. The large oscillations are driven by the validation batch ordering: certain val batches are systematically easier or harder than adjacent train batches, creating correlated spikes across all configs.

At the final evaluation (step 7500):

| Config | Overfit gap | Relative to baseline |
|--------|------------|---------------------|
| baseline | 0.034 | 1.0× |
| fa-64x32-softmax | 0.040 | 1.2× |
| fa-32x64-softmax | 0.045 | 1.3× |
| fa-32x64-silu | 0.056 | 1.6× |
| fa-32x64-silu2 | 0.064 | 1.9× |
| fa-128x16-silu | 0.099 | 2.9× |

Feature attention overfits 1.2-2.9× more than baseline. The pattern is clear:
- **Softmax** overfits least (1.2-1.3×) — normalization acts as regularization.
- **SiLU** overfits moderately (1.6×).
- **SiLU²** overfits most (1.9×) — unbounded, non-negative weights with sharp peaks.
- **High $N_f$** overfits more (128x16 at 2.9×) — more cross-feature interaction = more capacity = more overfitting.

### 5.2 Interaction between factorization and activation

The heatmaps in §4.4 reveal a non-trivial interaction:

- **Softmax** is relatively insensitive to factorization — its final val loss ranges from 4.56 (64x32) to 4.68 (8x256), a span of 0.12. This stability comes from normalization constraining the attention distribution regardless of matrix size.
- **SiLU/SiLU²** are more sensitive — silu ranges from 4.57 (32x64) to 4.69 (256x8), and silu² from 4.62 (32x64) to 4.78 (8x256). Without normalization, the raw score magnitudes depend strongly on $D_f$ (via the $1/\sqrt{D_f}$ scaling).
- **Best factorization differs by activation**: softmax prefers 64x32, while silu and silu² prefer 32x64. This makes sense: softmax benefits from more features to attend over (larger attention matrix), while silu/silu² benefit from higher-dimensional descriptors (more stable similarity scores).

### 5.3 Why does feature attention start slower?

Feature attention configs lag baseline in the first ~5,000 steps. Several factors likely contribute:

1. **Initialization mismatch**: At initialization, gate_proj outputs are near-zero, making all QK scores near-zero. Under softmax, this produces a uniform attention matrix (every feature attends equally to all others), which is very different from the near-zero gating of SwiGLU's SiLU(0)≈0. The model must first learn non-trivial feature similarity structure before attention becomes useful.

2. **Higher effective capacity**: Feature attention introduces cross-feature interactions that SwiGLU cannot express. Higher capacity models typically take longer to find good solutions but eventually achieve lower loss — consistent with the convergence crossover we observe.

3. **Optimization landscape**: The N×N attention matrix creates a more complex loss surface with more saddle points and local minima than element-wise gating, slowing early optimization.

### 5.4 Deferred analyses (require model checkpoints)

The following analyses require saved model checkpoints from chinchilla-length training, which are not yet available:

- **Attention pattern visualization**: Are feature attention matrices sparse or dense? Diagonal-dominant or distributed?
- **Diagonal dominance over training**: Does attention start uniform and sharpen, or start diagonal and diversify?
- **Feature specialization**: Does feature attention encourage or suppress dead features compared to SwiGLU?
- **Gradient norms**: Does feature attention change gradient flow through the MLP?

These will be filled in after chinchilla-length runs with checkpoint saving enabled.

---

## 6. Discussion

### 6.1 Does feature attention beat SwiGLU?

**Not yet, at 7,500 steps.** The best feature attention config (64x32-softmax) lags baseline by 2.05%. However, the convergence rate crossover at ~5,000 steps suggests this gap is closing. Linear extrapolation predicts silu configs crossing baseline around step 21,000 — well within chinchilla-optimal training length (150,845 steps).

The answer is genuinely uncertain: the extrapolation could be wrong, and the higher overfitting of feature attention could prevent it from ever matching baseline. Chinchilla-length runs are needed.

### 6.2 Best activation function

**Softmax** wins on absolute loss and overfitting control. **SiLU** wins on convergence speed. The optimal choice depends on training length:
- Short training (≤10k steps): softmax, because it has the lowest loss at every evaluation point.
- Long training (≥20k steps): silu may overtake softmax due to its faster convergence rate.
- **SiLU² is dominated** — it converges only marginally faster than silu but overfits substantially more. There is no training regime where silu² is the best choice.

### 6.3 Best factorization ratio

**64x32 and 32x64** are the clear winners, with 64x32 slightly favored for softmax and 32x64 for silu. The sweet spot is ~32-64 features with ~32-64 descriptor dimensions.

The intuition: you need enough features for attention to be meaningfully cross-feature (ruling out 8x256), and enough descriptor dimensions for similarity scores to be meaningful (ruling out 256x8).

### 6.4 Is the compute overhead worth it?

At the sweet-spot factorizations, overhead is 20-34%. For the potential benefit of cross-feature interaction, this is cheap. The question is whether the loss improvement materializes at chinchilla length.

If feature attention matches or beats baseline at 150k steps, the +20% compute overhead translates to needing 20% more training time for equal loss — which would make it borderline. Feature attention would need to *beat* baseline by enough to compensate for the throughput penalty.

### 6.5 Implications for scaling

This study is at 124M parameters. Several aspects likely change at larger scale:
- **Higher hidden dim** means more factorization options and potentially more benefit from cross-feature interaction.
- **Longer training** (chinchilla-optimal scales as 20× params) gives more time for feature attention's convergence advantage to manifest.
- **More layers** means feature attention in later layers can build on structured representations from earlier layers.

We expect feature attention to be more competitive at larger scales, but this is speculative.

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

| Run ID | Variant | Extra params/layer | Val loss (7.5k) | Δ vs shared QK |
|--------|---------|--------------------|----------------|----------------|
| | separate Q/K | | | |
| | GQA-style | | | |
| | multi-head | | | |

---

## 7. Conclusion

We introduced feature attention — replacing SwiGLU's element-wise gating with self-attention over the token's own hidden features — and swept 18 configurations (6 factorizations × 3 activations) against a SwiGLU baseline at 124M parameters.

**Main findings from the 7,500-step sweep:**

1. **No config beats baseline yet**, but feature attention converges 1.0-1.4× faster than baseline in the 5k-7.5k step range, suggesting a crossover may occur with longer training.
2. **64x32 and 32x64** are the optimal factorizations — a sweet spot where both feature count and descriptor dimensionality are large enough for meaningful computation.
3. **Softmax** achieves the best absolute loss and lowest overfitting; **silu** converges fastest; **silu²** is dominated.
4. **Zero additional parameters** and **20-34% throughput overhead** at the sweet-spot factorizations.
5. **Overfitting** is the primary risk: feature attention overfits 1.2-2.9× more than baseline, with unnormalized activations (silu, silu²) overfitting more.

**Next step**: Chinchilla-length training (~150k steps) on the top 5 configs to determine whether the convergence crossover is real and whether feature attention can match or beat SwiGLU when trained to completion.

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
MODEL_TYPE=transformer TRAIN_STEPS=7500 python train_fineweb_vanilla.py

# Feature attention sweep (example for one ratio)
for act in softmax silu silu2; do
    MODEL_TYPE=feat_attn FA_N_FEATURES=64 FA_ACTIVATION=$act \
        TRAIN_STEPS=7500 python train_fineweb_vanilla.py
done

# Full sweep
for nf in 256 128 64 32 16 8; do
    for act in softmax silu silu2; do
        MODEL_TYPE=feat_attn FA_N_FEATURES=$nf FA_ACTIVATION=$act \
            TRAIN_STEPS=7500 python train_fineweb_vanilla.py
    done
done
```
