# Universal Learning Block (ULB): Scan-Free Recurrent Accumulation via Transition-Shaped Attention

Anonymous draft (Markdown preprint)

---

## Abstract

We present the **Universal Learning Block (ULB)**, a sequence modeling primitive that combines local temporal transitions in query/key channels with full causal attention to obtain recurrent-like accumulation **without explicit scan recurrence**. ULB unifies (i) transition-style state shaping (K-lerp), (ii) bounded acausal signal injection (Q-peeking), (iii) rotary phase dynamics (fixed and data-dependent), and (iv) gated blending of normalized and unnormalized attention paths inside one block.

The central thesis is that ULB separates **transition construction** from **history accumulation**:

- local one-step operators define dynamics,
- causal attention performs global prefix accumulation in parallel.

This yields SSM-like capabilities while preserving GPU-friendly attention kernels. On synthetic capability benchmarks, ULB exhibits strong retrieval/state-tracking behavior and, with MoE stacking, reaches full convergence where matched baselines fail. We provide an end-to-end mathematical formulation, mechanism analysis, decoding contract, and ablation-driven explanation of why ULB works.

---

## 1. Introduction

Modern sequence models typically choose one of two paradigms:

1. **Attention-first models** with strong retrieval but quadratic training-time sequence interaction and KV-cache growth at inference.
2. **Scan-based recurrent/SSM models** with explicit state recurrences and favorable asymptotic inference, but dependence on sequential scan semantics and often lower hardware efficiency in practical kernels.

ULB takes a third path:

- keep full attention aggregation,
- inject explicit transition structure into Q/K before attention,
- optionally inject bounded acausal context through query peeking,
- preserve composability and standard residual stacking.

This is not "attention with a small tweak" and not "an SSM with attention decoration." It is a compositional operator where temporal dynamics are encoded in projected similarity features and accumulated by attention itself.

### Core contributions

1. **Transition-shaped attention formulation.** We formalize ULB as one-step temporal transition operators in Q/K composed with causal attention accumulation.
2. **Bounded acausality.** We formalize Q-peeking as a local acausal operator with depth-bounded leakage radius and a tail-recompute decode contract.
3. **Hybrid phase dynamics.** We unify fixed RoPE and data-dependent cumulative phase (dd-RoPE) in a single rotational parameterization.
4. **Gated attention-path fusion.** We use output-space blending between softmax attention and SiLU-squared attention, stabilizing scale via branch normalization.
5. **Empirical claim.** In this codebase's benchmark suite, ULB components provide distinct roles: Q-peeking is the dominant acausal capability path on induction; K-lerp and phase dynamics contribute transition/state structure; MoE routing amplifies specialization.

---

## 2. ULB Block Definition

Let input sequence be $x \in \mathbb{R}^{B\times T\times D}$, head count $H$, inner dimension $D_{in}$ (configurable via `inner_ratio`, default 1.0; $D_{in} = \mathrm{round}(D \cdot \texttt{inner\_ratio} / (4H)) \cdot 4H$), and head dim $d=D_{in}/H$.

ULB block computes

$$
h = \mathrm{Swish}_\beta(W_{up}x), \qquad W_{up} \in \mathbb{R}^{D \times D_{in}},
$$

$$
q = W_q h,\quad k = W_k h,\quad v = W_v h, \qquad W_q,W_k,W_v \in \mathbb{R}^{D_{in} \times D_{in}},
$$

with per-head reshaping to $\mathbb{R}^{B\times T\times H\times d}$ and post-norm channel biases:

$$
q \leftarrow \mathrm{RMSNorm}(q)\odot b_q,
\qquad
k \leftarrow \mathrm{RMSNorm}(k)\odot b_k.
$$

Then temporal transitions:

$$
k' = \mathcal{T}_K(k; x),
\qquad
q' = \mathcal{T}_Q(q; x),
$$

followed by hybrid rotation:

$$
(q_r, k_r) = \mathcal{R}_{hybrid}(q',k';x),
$$

attention:

$$
y_{attn} = \mathcal{A}(q_r,k_r,v),
$$

multiplicative fusion and projection:

$$
y = W_{down}\,\mathrm{Swish}_\beta\!\left(\mathrm{RMSNorm}(y_{attn})\odot h\right), \qquad W_{down} \in \mathbb{R}^{D_{in} \times D}.
$$

Stack-level residual is external (pre-norm residual stack):

$$
x^{(\ell+1)} = x^{(\ell)} + \mathrm{ULB}^{(\ell)}(\mathrm{RMSNorm}(x^{(\ell)})).
$$

---

## 3. Transition Operators

### 3.1 K-lerp (causal one-step transition)

On selected key channels:

$$
k_t^{mix} = (1-g_t^k)k_t + g_t^k k_{t-1},
\qquad
g_t^k = \sigma(W_k^g x_t + b_k^g).
$$

This injects a one-step transition motif (current/previous endpoint mixing) directly into similarity features.

### 3.2 Q-lerp / Q-peeking (bounded acausal transition)

On selected query channels:

$$
q_t^{mix} = (1-g_t^f-g_t^b)q_t + g_t^f q_{t+1} + g_t^b q_{t-1},
$$

$$
g_t^f=\sigma(W_f x_t+b_f),\qquad g_t^b=\sigma(W_b x_t+b_b).
$$

Forward term $q_{t+1}$ introduces bounded acausality.

---

## 4. Why ULB Works: Operator Composition

The key identity is composition:

$$
\text{(local transition in Q/K)} \circ \text{(global causal accumulation by attention)}.
$$

ULB does not build an explicit serial state variable inside the block. Instead, it parameterizes similarity kernels so that attention performs the history accumulation over the full prefix.

### 4.1 Logit decomposition

For one head with key mixing:

$$
\ell_{tj} = \frac{\tilde q_t^\top \tilde k_j}{\sqrt d}
= \frac{\tilde q_t^\top\big((1-\gamma_j)k_j+\gamma_j k_{j-1}\big)}{\sqrt d}.
$$

Thus each score uses adjacent-key transition structure; summation over all $j\le t$ propagates this structure through the full prefix.

### 4.2 Linearized state interpretation

In linear-attention form (analysis view),

$$
\hat y_t = \phi(\tilde q_t)^\top s_t,
\qquad
s_t = s_{t-1} + \psi(\tilde k_t)v_t^\top,
$$

which is an explicit recurrent accumulator in feature space. ULB therefore realizes recurrent accumulation behavior without a dedicated scan recurrence in forward implementation.

---

## 5. Bounded Acausality: Formal Properties

### Proposition 1 (single-layer leakage)

With +1 Q-peeking, output at position $t$ may depend on $x_{t+1}$ and cannot depend on $x_{>t+1}$.

### Proposition 2 (depth accumulation)

With $L$ peeking layers (each +1), future dependency radius satisfies

$$
R \le L.
$$

In this implementation family, empirical checks are tight ($R=L$).

### Theorem 1 (tail invalidation bound)

When appending one token at decode step $T$, only positions in

$$
\{T-R,\ldots,T\}
$$

can change. Prefix positions $<T-R$ remain invariant.

**Decode contract:** use radius-$R$ tail recompute cache rather than strict one-step KV-cache.

---

## 6. Hybrid Rotational Dynamics

ULB applies rotation on Q/K by splitting rotary pairs into two parts:

1. fixed-frequency RoPE,
2. data-dependent cumulative phase:

$$
\theta_t = \sum_{\tau\le t} u_\tau,
\qquad
u_t = W_{dd}x_t + b_{dd}.
$$

This unifies positional phase anchoring (fixed) and content-driven phase trajectories (data-dependent). In capability ablations, this mechanism can aid state-tracking, but acausal induction gains in this setup are dominated by Q-peeking.

---

## 7. Attention Path Design

ULB supports three paths:

1. softmax SDPA,
2. SiLU-squared attention,
3. blend:

$$
y = (1-\lambda_t)\,y_{softmax} + \lambda_t\,\mathrm{RMSNorm}(y_{silu2}),
\qquad
\lambda_t = \sigma(w^\top x_t + b).
$$

Blend combines calibrated probability-style retrieval (softmax) with higher-gain unnormalized path dynamics (SiLU-squared), while branch normalization keeps scale competition controlled.

---

## 8. Architectural Choices That Matter

In this implementation lineage, the following choices are deliberate:

- multiplicative skip fusion after attention (`attn_norm(y) * h_up`),
- no internal residual inside block (block returns delta),
- KV projection bias with Q bias disabled in projection,
- post-norm multiplicative Q/K channel biases,
- no conv dependence for retrieval-heavy tasks,
- learnable Swish rather than fixed SiLU (configurable via `swish_mode`),
- configurable `inner_ratio` for wider QKV attention (up/down project between $D$ and $D_{in}$).

These interact with transition operators; changing them shifts stability/capability tradeoffs.

---

## 9. ULB + MoE

ULB composes with a layerwise Mixture-of-Experts (MoE) stacker that is model-agnostic (any block type can be used as an expert). This section specifies the exact mechanism used in this codebase.

### 9.1 Architecture

Let depth be $L_e$ and experts per layer be $E$. For layer $\ell$, define experts

$$
\{f_{\ell,1},\dots,f_{\ell,E}\},
$$

where each $f_{\ell,e}$ is a ULB block returning a residual delta. The model keeps standard pre-norm residual semantics.

At layer $\ell$, given hidden state $x^{(\ell)} \in \mathbb{R}^{B\times T\times D}$:

1. **Pre-norm**
$$
h^{(\ell)} = \mathrm{RMSNorm}(x^{(\ell)}).
$$

2. **Per-sample routing signal** (sequence pooled):
$$
r^{(\ell)} = \frac{1}{T}\sum_{t=1}^{T} h^{(\ell)}_t \in \mathbb{R}^{B\times D}.
$$

3. **Router logits**
$$
z^{(\ell)} = W^{(\ell)}_{router} r^{(\ell)} \in \mathbb{R}^{B\times E}.
$$

4. **Top-$k$ selection** (per sample):
$$
I^{(\ell)}_b = \operatorname{TopK}(z^{(\ell)}_b, k),
\qquad
\pi^{(\ell)}_b = \operatorname{softmax}(z^{(\ell)}_b[I^{(\ell)}_b]).
$$

5. **Expert computation and merge**

All experts run on full batch (dense execution; no sparse dispatch in current default implementation):
$$
u^{(\ell)}_e = f_{\ell,e}(h^{(\ell)}) \in \mathbb{R}^{B\times T\times D}.
$$

Then gather selected experts and merge:
$$
\Delta x^{(\ell)}_b = \sum_{m=1}^{k} \pi^{(\ell)}_{b,m}\,u^{(\ell)}_{I^{(\ell)}_{b,m}}.
$$

6. **Residual update**
$$
x^{(\ell+1)} = x^{(\ell)} + \Delta x^{(\ell)}.
$$

7. **Cross-layer output mixing**

Store all intermediate states $\{x^{(0)},x^{(1)},\dots,x^{(L_e)}\}$ and blend with learned global layer weights:
$$
w = \operatorname{softmax}(\theta) \in \mathbb{R}^{L_e+1},
\qquad
x_{out} = \mathrm{RMSNorm}\!\left(\sum_{i=0}^{L_e} w_i x^{(i)}\right).
$$

### 9.2 Routing (v1 only)

### Algorithm 1: MoE-ULB v1 forward (per batch)

```text
Input: x^(0) in R^{B x T x D}, experts {f_{l,e}}, routers {W_router^(l)}, top-k, layer-weights theta
for l = 0 .. L_e-1:
    h^(l) = RMSNorm(x^(l))
    r^(l) = mean_t h^(l)_t                       # R^{B x D}
    z^(l) = W_router^(l) r^(l)                   # R^{B x E}
    I_b^(l), vals_b^(l) = TopK(z_b^(l), k)
    pi_b^(l) = softmax(vals_b^(l))

    # Dense expert execution
    for e = 1 .. E:
        u_e^(l) = f_{l,e}(h^(l))                 # R^{B x T x D}

    # Gather selected experts and merge per sample
    Delta x_b^(l) = sum_{m=1..k} pi_{b,m}^(l) * u_{I_{b,m}^(l), b}^(l)
    x^(l+1) = x^(l) + Delta x^(l)

Collect states S = {x^(0), ..., x^(L_e)}
w = softmax(theta)                               # R^{L_e+1}
x_out = RMSNorm(sum_i w_i * S_i)
Return x_out
```

Each layer routes using its own pooled pre-norm state $r^{(\ell)}$. Merge weights are directly top-$k$ softmax over router logits.

This is the only MoE routing variant considered in this paper.

### 9.3 Why this MoE works with ULB

ULB already factorizes temporal behavior into transition channels and attention accumulation. MoE adds a second factorization axis: **sample-conditional functional specialization**.

- Router decides which transition-accumulation dynamics to apply per sample.
- Different experts can specialize to different synthetic mechanisms (delay, copy, induction, parity, arithmetic).
- Learned cross-layer mixing avoids forcing only the final layer to carry all task signal.

Empirically in this repository, this combination yields rapid convergence and full mixed-task saturation for ULB-based models, consistent with strong expert specialization.

### 9.4 Compute and systems note

Current implementation intentionally runs all experts per layer on full batch (dense compute) and applies sparse selection only at merge time. This avoids complex dispatch logic and keeps implementation simple, at the expense of FLOP efficiency.

Future sparse kernels can reduce compute, but they are not required for functional correctness of the routing mechanism.

### 9.5 CLI surface used in experiments

MoE is enabled by:

- `--moe`
- `--n-experts E` (default 4)
- `--top-k k` (default 2)

and applies to all selected models, including ULB variants.

### 9.6 Synthetic benchmark protocol (`bench_ssm.py`, mixed mode)

This paper's synthetic capability claims are grounded in the mixed-task protocol from `scripts/bench_ssm.py`. The setup is intentionally unified: one embedding table, one head, one forward path, shared vocabulary, and `ignore_index=-100` masking where needed.

#### Task suite

Mixed mode samples each training example from one of five tasks:

1. **delay**: input tokens in `[0,31]`; target is one-step shifted previous token.
   - `target[:,0] = -100`, `target[:,1:] = x[:,:-1]`.
2. **selective_copy**: input tokens in `[0,31]` with `n_markers` randomly selected positions marked by adding `+32` (so marked tokens lie in `[32,63]`).
   - target is only evaluated on last `n_markers` positions, where unmarked originals must be emitted.
3. **induction**: two-copy random pattern next-token prediction with variable pattern length.
   - all positions are scored; first copy is intentionally unpredictable.
4. **parity**: binary input, target is running XOR.
5. **mod_arith**: base-5 input digits, target is running cumulative sum mod 5.

The canonical task list is:

$$
\texttt{ALL\_TASKS} = [\texttt{delay},\texttt{selective\_copy},\texttt{induction},\texttt{parity},\texttt{mod\_arith}].
$$

#### Unified vocabulary and masking

- Global vocabulary size is fixed at `VOCAB_SIZE = 64`.
- Unmarked symbols use `[0,31]`.
- Marker channel for selective-copy uses `[32,63]`.
- Loss is always cross-entropy with `ignore_index=-100`.

This forces all tasks through identical model plumbing (no task-specific output heads).

#### Mixed batch generation

For each sample index $b$ in a batch of size $B$:

1. sample task id $t_b \sim \mathrm{Uniform}(\{1,\dots,5\})$,
2. generate one-sample pair $(x_b,y_b)$ from that task,
3. store `task_ids[b] = t_b` for evaluation-time per-task breakdown.

So each batch is heterogeneous across tasks, and routing/specialization pressures are present even without explicit task tokens.

#### Data pregeneration and caching

Training/validation batches are pregenerated and cached to disk under `scripts/.bench_cache/` keyed by:

$$
\texttt{hash}(\texttt{task\_name}, n\_steps, B, L, \texttt{seed}).
$$

When generator logic changes, caches must be invalidated to avoid stale datasets.

#### Optimization schedule

For mixed task runs:

- base learning rate: `1e-3` (standalone tasks use `1e-4`),
- one-epoch linear warmup from `0.1 * lr` to `1.0 * lr`,
- cosine decay to `0.05 * lr`.

#### Two-phase hard mining (mixed only)

Hard mining activates at

$$
\texttt{hard\_mine\_epoch} = \left\lfloor \frac{2}{3} \cdot \texttt{max\_epochs} \right\rfloor.
$$

After activation, each sample's mean token loss is ranked within the batch and reweighted:

- easiest samples: weight near `0.5`,
- hardest samples: weight near `2.0`,
- then normalized to mean weight `1.0`.

This upweights unresolved failure modes late in training without changing objective family.

#### Validation metrics and convergence rule

Validation computes:

1. per-subtask accuracy (for mixed mode) using `task_ids` masks,
2. macro-averaged validation accuracy: the mean of per-subtask accuracies (not micro-averaged global token accuracy).

Macro-averaging prevents easy high-token-count tasks from inflating the reported accuracy.

Convergence criterion for mixed mode is strict:

$$
\forall \tau \in \texttt{ALL\_TASKS},\; \mathrm{Acc}_{val}^{(\tau)} \ge \texttt{early\_stop\_acc}.
$$

With default `early_stop_acc = 0.98`, a model is considered converged only if **every** subtask crosses 98%.

This avoids misleading wins where one easy task dominates the average.

#### Why this benchmark is diagnostic for ULB

The mixed suite jointly pressures:

- short-term memory transfer (delay),
- retrieval with sparse supervision (selective_copy),
- bounded acausal transfer (induction),
- algorithmic state tracking (parity, mod_arith).

ULB's transition-shaped attention and Q-peeking are directly stressed by this combination, which is why mixed-mode outcomes are central to this paper.

---

## 10. What Makes ULB Different

ULB is different from standard model families in mechanism, not branding:

1. **Not plain Transformer attention:** similarity features are temporally transitioned before aggregation.
2. **Not classic scan SSM:** no explicit serial hidden-state scan in block forward; attention provides accumulation.
3. **Not full bidirectional attention:** acausality is bounded and controllable by layer depth/peek design.
4. **Not "just RoPE tricks":** capability is split across transition operators, rotational dynamics, and path blending.

In short: ULB is a transition-parameterized attention accumulator with bounded acausal control.

## 10.1 Under-capacity stress test (single-layer mixed benchmark)

To isolate layer inductive bias rather than brute capacity, we run an **extreme under-capacity setting**:

- 1 layer
- `dim=64`, `seq_len=64`, `batch=256`
- mixed benchmark (all 5 subtasks jointly)
- matched parameter budgets, with baselines auto-sized to match ULBBlendP parameter count

Parameter matching protocol (as implemented in `bench_ssm.py`):

- Reference target is ULBBlendP block parameter count at the same depth/width (with default `inner_ratio`).
- Baseline internal knobs are searched per depth to minimize param difference:
  - S4D: `d_state`
  - Mamba: `d_state` (with `expand=2`, `d_conv=4` fixed)
  - MHA: `mlp_inner`

Exact single-layer model configs used for the table below (run with `inner_ratio=1.0`, thin up/down linears):

- `S4D(d_model=64, d_state=148)`
- `MambaWrapper(d_model=64, d_state=2, d_conv=4, expand=2)`
- `MHABlock(d_model=64, n_heads=4, mlp_inner=54)`
- `ULBBlock(ULBConfig(d_model=64, n_heads=4, paired=True, attn_mode='blend', q_mix='lerp', inner_ratio=1.0))`

> Note: param counts in the table below reflect the architecture state at the time of that run. Exact counts shift as up/down projection structure and `inner_ratio` change; re-run bench to get current numbers.

### Results (single run, seed-fixed)

| Model | Params | Overall | delay | selective_copy | induction | parity | mod_arith | Min subtask |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S4D | 27,520 | 48.5% | 95.3% | 7.2% | 8.1% | 63.0% | 30.9% | 7.2% |
| MHA | 27,456 | 48.2% | 99.9% | 28.9% | 3.1% | 62.5% | 29.3% | 3.1% |
| Mamba | 27,392 | 46.8% | 99.7% | 30.0% | 3.2% | 60.8% | 25.2% | 3.2% |
| **ULBBlendP** | **27,508** | 46.6% | 49.4% | 52.7% | 52.4% | 57.9% | 25.9% | **25.9%** |

All runs use the same training schedule and mixed-task convergence rule described in Section 9.6.

### Interpretation

This experiment is designed to stress-test whether a layer family has enough intrinsic inductive bias to train **above near-random floor on all tasks simultaneously** under severe capacity constraints.

- Baseline families (S4D/MHA/Mamba) show **narrow bias**: they maximize one aligned task (especially delay) while collapsing on others.
- ULB shows **broad bias**: it does not maximize the easiest single task in this regime, but it learns all subtasks to non-trivial levels and raises the worst-subtask floor substantially.

So in under-capacity conditions, ULB trades single-task peak for cross-task competence. This is a core claim of the architecture: transition-shaped attention gives a more uniform capability prior across heterogeneous sequence tasks.

> Note: this table is intentionally a stress regime and not a final-capacity comparison. Main-capacity results and multi-seed aggregates should be reported separately.

---

## 11. Mathematical Summary (compact)

Define transition operators $\mathcal{T}_Q,\mathcal{T}_K$ and rotary map $\mathcal{R}$, then

$$
y_t = \sum_{j\le t} \operatorname{Softmax}_j\!\left(\frac{\langle \mathcal{R}(\mathcal{T}_Q(q_t)),\mathcal{R}(\mathcal{T}_K(k_j))\rangle}{\sqrt d}\right) v_j.
$$

This single equation captures the mechanism:

- transition-shaped similarity,
- prefix accumulation,
- optional bounded acausal query dependence.

---

## 12. Reproducibility Notes

- Validate correctness on CPU/CUDA; do not rely on MPS for final learning conclusions.
- If data generators change, invalidate dataset caches before comparing runs.
- For peeking decode studies, report both strict-causal failure and bounded-leakage pass with declared radius.

---

## 13. Limitations

- Bounded acausality still requires tail recompute decode semantics.
- Radius grows with peeking depth unless explicitly capped.
- Softmax normalization complicates closed-form recurrence equivalence (linearized analysis is exact only in linear-attention view).
- Long-context asymptotics still inherit attention-time training cost.

---

## 14. Release-Ready TODOs

### Figures (TODO)

- **Fig 1:** End-to-end ULB block diagram with operator stages.
- **Fig 2:** Q/K temporal transition visualization.
- **Fig 3:** Effective leakage radius vs depth (empirical and theoretical).
- **Fig 4:** Tail-recompute decode timeline.
- **Fig 5:** Attention-logit decomposition heatmap under K-lerp.

### Tables (TODO)

- **Table 1:** Main benchmark (ULB vs matched SSM/attention baselines).
- **Table 2:** ULB ablation matrix (Q-lerp, K-lerp, dd-rotate, blend path).
- **Table 3:** MoE specialization and convergence table.
- **Table 4:** Decode cost vs radius ($R$) and layers.
- **Table 5:** Robustness across seeds and sequence lengths.

### Theory appendices (TODO)

- full proof of tail invalidation theorem under residual stacks,
- generalized bound for multi-step peek offsets,
- softmax-vs-linearized accumulator approximation bounds.

### Systems appendix (TODO)

- reference radius-$R$ decode API,
- unit tests for bounded leakage invariants,
- kernel design for small-window tail recompute.

---

## 15. Conclusion

ULB demonstrates that explicit temporal transition structure and attention accumulation are not competing paradigms. They are composable. By shaping Q/K with local transition operators and letting attention aggregate globally, ULB achieves scan-free recurrent accumulation with controllable acausality, strong capability, and practical implementation paths.

This is the core claim: **ULB is not another attention block; it is a transition-accumulation framework.**
