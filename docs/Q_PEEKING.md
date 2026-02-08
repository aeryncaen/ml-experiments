# Q-Peeking: Bounded Acausal Query Mixing with Tail-Recompute Decoding

Anonymous draft (Markdown preprint)

## Abstract

We introduce **Q-peeking**, a local acausal mechanism for autoregressive attention blocks that mixes each query with its immediate future query. Unlike fully bidirectional attention, Q-peeking exposes only a bounded forward receptive field and admits **cache-compatible decoding via fixed-radius tail recomputation**. We formalize the mechanism, prove a depth-dependent dependency bound, and show how to decode with constant memory growth and bounded extra compute per generated token. In our ULB/FusedGate setting, Q-peeking is the dominant source of acausal capability on induction-style tasks, while data-dependent rotary phase alone is not sufficient. We provide a practical verification protocol and implementation checklist for ensuring bounded leakage under layer accumulation.

## 1. Problem Statement

Standard causal self-attention enforces

$$
\hat{y}_t = f(x_{\le t}),
$$

which is strictly compatible with one-step KV-cache decoding.

Many tasks, however, require limited future context. Full bidirectional attention is expensive and invalidates standard autoregressive decoding. We seek a middle ground:

- local acausality for capability,
- bounded dependency radius,
- decoding that updates only a small tail region.

## 2. Q-Peeking Mechanism

Let $q_t \in \mathbb{R}^{d_h}$ be per-head query channels selected for peeking (typically the last half of head dimension). Q-peeking defines:

$$
\tilde{q}_t = (1 - g_t^{f} - g_t^{b}) q_t + g_t^{f} q_{t+1} + g_t^{b} q_{t-1},
$$

with gates

$$
g_t^{f} = \sigma(W_f x_t + b_f), \qquad g_t^{b} = \sigma(W_b x_t + b_b).
$$

In implementation:

- forward neighbor uses zero-padding at sequence end,
- backward neighbor uses zero-padding at sequence start,
- only a subset of channels are mixed (others pass through unchanged).

### 2.1 Why this is acausal

Because $\tilde{q}_t$ depends on $q_{t+1}$, any downstream attention output at position $t$ can depend on token $x_{t+1}$ even under causal masking in the attention matrix. Thus masking alone no longer guarantees strict causality.

### 2.2 Equivalence to a constrained acausal attention operator

Consider one head (drop head index) with causal key/value access:

$$
y_t = \sum_{j \le t} \alpha_{tj} v_j,
\qquad
\alpha_{tj} = \operatorname{softmax}_j\!\left(\frac{\tilde q_t^\top k_j}{\sqrt d}\right).
$$

With Q-peeking,

$$
\tilde q_t = a_t q_t + b_t q_{t+1} + c_t q_{t-1},
\qquad
a_t = 1-g_t^f-g_t^b,\; b_t=g_t^f,\; c_t=g_t^b.
$$

Hence each logit is

$$
\ell_{tj}
= \frac{\tilde q_t^\top k_j}{\sqrt d}
= \frac{a_t q_t^\top k_j + b_t q_{t+1}^\top k_j + c_t q_{t-1}^\top k_j}{\sqrt d}. \tag{10}
$$

Equation (10) is an exact decomposition into three bilinear score terms, one of which uses the **future** query $q_{t+1}$. Therefore the output $y_t$ is a function of $x_{t+1}$ in general, i.e. acausal.

Define the augmented query state

$$
\bar q_t := \begin{bmatrix} q_{t-1} \\ q_t \\ q_{t+1} \end{bmatrix} \in \mathbb{R}^{3d},
\qquad
\bar k_{tj} := \begin{bmatrix} c_t k_j \\ a_t k_j \\ b_t k_j \end{bmatrix} \in \mathbb{R}^{3d}.
$$

Then

$$
\ell_{tj} = \frac{\bar q_t^\top \bar k_{tj}}{\sqrt d}. \tag{11}
$$

So Q-peeking is exactly equivalent to standard dot-product attention on an augmented query feature map with a position-dependent key scaling. This is a restricted (banded, +1) acausal attention family:

- **key/value index constraint:** still $j \le t$ (causal in retrieval index),
- **query content constraint:** depends on $\{t-1,t,t+1\}$ (acausal in representation index).

This asymmetric construction is why the receptive field is small but genuinely acausal.

### 2.3 Operator-equivalence vs task-equivalence

It is critical to distinguish two claims:

1. **Operator-equivalence** (false in general):
   one-layer Q-peeking is not equivalent to full bidirectional attention, since
   its future dependency is bounded (+1 here).

2. **Task-equivalence** (can be true):
   one-layer Q-peeking can match full-acausal performance on tasks whose target
   depends primarily on short-horizon future context.

For induction-style next-token tasks used here, this second claim is the relevant one.

#### Proposition 3 (Induction task-equivalence under one-step future signal)

Assume a task family where the Bayes-optimal predictor at position $t$ is a measurable
function of $(x_{\le t}, x_{t+1})$ and does not require $x_{>t+1}$. Then a one-layer
Q-peeking attention block with sufficient width is representationally sufficient to
implement the Bayes-optimal predictor, while strict causal attention (without access
to $x_{t+1}$) is not.

**Proof sketch.**

- Q-peeking exposes $x_{t+1}$ to position $t$ through $q_{t+1}$ in Eq. (10).
- Attention and MLP maps are universal approximators over finite-dimensional inputs,
  so the block can approximate any target function of $(x_{\le t}, x_{t+1})$.
- Strict causal models cannot condition on $x_{t+1}$ by definition, so any target that
  has non-zero conditional dependence on $x_{t+1}$ cannot be matched exactly.

Hence one-step peeking can be **task-equivalent** to full acausal attention on such
tasks despite not being operator-equivalent.

### 2.4 Formal result for the induction benchmark

We now state a benchmark-specific theorem to make the above precise.

#### Setup

For a sample sequence length $L$, choose pattern length $P$ and random tokens
$s_0,\dots,s_{P-1} \overset{i.i.d.}{\sim} \mathrm{Unif}(\mathcal V)$ with
$|\mathcal V|=V$. Construct

$$
\text{seq} = (s_0,\dots,s_{P-1}, s_0,\dots,s_{P-1}),
$$

and train/evaluate next-token prediction on windowed tensors

$$
x_t = \text{seq}_t, \qquad y_t = \text{seq}_{t+1}, \qquad t=0,\dots,L-1,
$$

with $2P \ge L+1$ (as in code).

#### Theorem 2 (Causal Bayes ceiling)

For any strictly causal predictor $\hat y_t = f_t(x_{\le t})$, the Bayes-optimal
token accuracy conditioned on $P$ is

$$
\mathrm{Acc}^*_\mathrm{causal}(P)
= \frac{L-P}{L} + \frac{P}{L}\cdot\frac{1}{V}
= 1 - \frac{P}{L}\left(1-\frac{1}{V}\right). \tag{13}
$$

**Proof.**

Split positions into two sets:

- **Second-copy positions** $t \in \{P,\dots,L-1\}$ (count $L-P$):
  here $y_t=s_{t-P+1}$ and $x_{t-P}=s_{t-P}$ is in prefix, so the mapping is
  deterministically recoverable from past context; Bayes accuracy is $1$.

- **First-copy positions** $t \in \{0,\dots,P-1\}$ (count $P$):
  $y_t=s_{t+1}$ is independent of $x_{\le t}=(s_0,\dots,s_t)$ under i.i.d.
  sampling, so Bayes rule is uniform guessing with accuracy $1/V$.

Average over positions gives Eq. (13). QED.

For $L=32$, $V=32$, and $P\in[17,21]$, Eq. (13) gives ceilings in
approximately $[0.364, 0.485]$, matching the observed causal wall (~41%).

#### Theorem 3 (One-layer Q-peeking sufficiency for this benchmark)

Assume a model class where representation at position $t$ can realize
$h_t = \phi(x_{t+1})$ whenever $x_{t+1}$ is available in-query (realizable by
Q-peeking with nonzero forward gate on at least one channel), and output head
can realize token decoding from $\phi(x_{t+1})$.

Then Bayes-optimal accuracy is

$$
\mathrm{Acc}^*_{\mathrm{Qpeek}} = 1. \tag{14}
$$

**Proof.**

By construction of the task, $y_t=x_{t+1}$ for all scored positions. If
$h_t$ can represent $x_{t+1}$ and the head can decode it, prediction is exact.
Q-peeking provides access to $q_{t+1}$ at position $t$ through Eq. (10), hence
the realizability condition is satisfied in this model family. QED.

#### Corollary (Task-equivalence to full acausal attention)

For this induction benchmark, full acausal attention and one-layer Q-peeking
share the same Bayes optimum (100%), despite different operator classes.

This is exactly the sense in which Q-peeking is "equivalent for induction":
**equal optimal task performance, not equal global receptive field.**

#### Multi-layer composition

Let $h_t^{(\ell)}$ be layer-$\ell$ hidden states. If each layer applies +1 Q-peeking, then recursively

$$
h_t^{(\ell)} = F_\ell\big(h_{\le t+1}^{(\ell-1)}\big),
$$

which yields

$$
h_t^{(L)} = \Phi\big(h_{\le t+L}^{(0)}\big). \tag{12}
$$

Thus Q-peeking is equivalent to attention with an **effective upper dependency band** of width $L$ (future radius $R=L$ in this configuration). It is not fully bidirectional attention; it is a bounded acausal operator.

## 3. Dependency Radius and Layer Accumulation

Define the **future dependency radius** $R$ such that output at position $t$ depends only on $x_{\le t+R}$.

### Proposition 1 (Single-layer bound)

For one Q-peeking layer with +1 forward peek, output at position $t$ depends on at most $x_{\le t+1}$. Hence $R_1 = 1$.

**Sketch.** Query mixing introduces only $q_{t+1}$ as future input. Causal attention weights still restrict key/value indices to $\le t$ in attention-time index space, but query content itself can encode one-step future information.

### Proposition 2 (Depth accumulation)

For a stack of $L$ layers where each layer applies +1 Q-peeking, the effective radius is

$$
R_L \le L.
$$

If every layer has active peek channels, $R_L = L$ empirically in ULB.

**Proof (induction).**

- Base: $R_1 = 1$ from Proposition 1.
- Step: assume layer $\ell-1$ has radius $R_{\ell-1}$. Layer $\ell$ at position $t$ uses representation at $t$ and $t+1$ from layer $\ell-1$, so it can depend on inputs up to

$$
\max((t + R_{\ell-1}), (t+1 + R_{\ell-1})) = t + (R_{\ell-1}+1).
$$

Thus $R_\ell \le R_{\ell-1}+1$, yielding $R_L \le L$.

More generally, if only a subset of layers peeks, radius is bounded by the count of peeking layers.

## 4. Decoding: Tail-Recompute Instead of Full Recompute

Strict KV-cache decoding assumes $R=0$. For $R>0$, appending one token invalidates a tail of previous outputs.

### Theorem 1 (Tail invalidation bound)

When extending sequence by one token, only positions in

$$
\{T-R, \dots, T\}
$$

may change (for current length index $T$). Positions $< T-R$ are invariant.

This enables **fixed-radius re-attend**:

1. Keep cached prefix states/outputs unchanged up to $T-R-1$.
2. Recompute only tail window $[T-R, \dots, T]$.
3. Emit token at $T$.

### Complexity

Per generated token:

- standard causal decode: $O(1)$ tokens recomputed,
- Q-peeking decode: $O(R)$ tokens recomputed.

For fixed model depth/config, $R$ is constant w.r.t. sequence length, so decode remains linear in generated length with constant-factor overhead.

Memory:

- prefix cache remains linear in context length,
- additional rolling tail buffer of size $R$.

## 5. KV-Cache Compatibility: Precise Claim

Q-peeking is **not** compatible with *strict* one-step KV-cache semantics ($R=0$).

Q-peeking **is compatible** with an **R-tail cache** protocol:

- maintain normal KV cache for stable prefix,
- re-materialize last $R$ positions on each append,
- update tail cache and emitted logits.

We call this **bounded-cache-compatible acausality**.

## 6. Empirical Verification Protocol

We provide `scripts/check_q_peeking_causality.py` with two diagnostics:

1. **Future perturbation map**: perturb suffix $[\text{split}, T)$ and measure per-position drift on prefix.
2. **Prefix consistency**: compare output at position $t$ from full pass vs prefix-only pass.

For Q-peeking, strict causality should fail, but bounded leakage should pass if drift is confined to the boundary band.

### Observed radius sweep (ULB, paired blend mode)

Using the checker across 5 seeds:

| Layers | Max empirical leakage radius | Bounded leakage pass |
|---|---:|---|
| 1 | 1 | Yes |
| 2 | 2 | Yes |
| 3 | 3 | Yes |
| 4 | 4 | Yes |
| 5 | 5 | Yes |
| 6 | 6 | Yes |

This matches the $R=L$ accumulation law in this configuration.

## 7. Interaction with Other ULB Components

In current ablations:

- disabling Q-peeking collapses induction performance to causal ceiling,
- enabling Q-peeking restores strong acausal performance,
- data-dependent rotary phase alone does not recover acausal performance.

Interpretation: in this architecture, Q-peeking is the primary acausal channel; other mechanisms remain causal or state-tracking.

## 8. Implementation Notes

### 8.1 Numerics and training stability

- Gate init near identity is critical (small initial peeking).
- Monitor gate saturation; very high forward gate can over-amplify boundary leakage.

### 8.2 Decode-time contract

Any production decode path must explicitly declare one of:

1. strict causal cache (`R=0`), or
2. bounded acausal cache with configured radius `R`.

Silent fallback to strict cache when `R>0` is incorrect.

## 9. Limitations

- Radius grows with number of peeking layers unless explicitly capped.
- Tail recompute adds constant-factor decode latency.
- Current proofs assume local (+1) peeking; larger offsets require generalized bounds.

## 10. Release Checklist (TODO)

### Core figures (TODO)

- **Figure 1:** Q-peeking computation graph at one layer (show $q_{t-1}, q_t, q_{t+1}$ mixing).
- **Figure 2:** Layer accumulation diagram showing receptive field growth from $R=1$ to $R=L$.
- **Figure 3:** Decode timeline showing tail recomputation window per generated token.
- **Figure 4:** Future-perturbation heatmap (position vs layer depth).

### Main tables (TODO)

- **Table 1:** Synthetic benchmark results (causal baselines vs Q-peeking variants).
- **Table 2:** Ablation matrix:
  - Q-peeking on/off
  - dd-rotate on/off
  - paired/unpaired
  - induction accuracy and other tasks.
- **Table 3:** Decode overhead vs radius ($R=0..L$): tokens/sec, latency/token, memory.
- **Table 4:** Radius verification sweep (seeds, layers, empirical radius, pass rate).

### Theory appendix (TODO)

- Full formal proof of Theorem 1 under residual + prenorm stacking.
- Generalized radius theorem for k-step peek ($q_{t+k}$) and mixed offsets.
- Error propagation bound under finite precision and normalization layers.

### Engineering appendix (TODO)

- Reference pseudocode for R-tail cache decode API.
- Unit tests for boundary invariance (`far_prefix`) and radius checks.
- Kernel-level plan for efficient tail recompute (fused small-window kernels).

## 11. Suggested Citation Block (placeholder)

```
@article{qpeeking2026,
  title={Q-Peeking: Bounded Acausal Query Mixing with Tail-Recompute Decoding},
  author={Anonymous},
  year={2026}
}
```

---

## Appendix A: Minimal Decoder Pseudocode (R-tail cache)

```text
state: prefix_cache, tail_tokens (length <= R), model

for each new token x_T:
  append x_T to tail_tokens
  window = concat(last stable prefix token needed, tail_tokens)
  recompute model outputs for window positions [T-R, ..., T]
  update tail cache entries
  emit logits at position T
  if tail_tokens length > R:
      move oldest tail token/cache to stable prefix cache
```

## Appendix B: Practical Assertions

For configured radius `R`:

- perturbing tokens at positions `>= split` must not change outputs `< split-R`.
- perturbing tokens at positions `>= split` may change outputs in `[split-R, split)`.
- empirical leakage radius should remain `<= R` across seeds.
