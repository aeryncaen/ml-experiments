# QK-Lerp + Causal Attention as Scan-Free Recurrent Accumulation

Anonymous draft (Markdown preprint)

## Abstract

We formalize a mechanism in which **one-step temporal mixing in Q/K** (QK-lerp) combined with **full causal attention** yields SSM-like recurrent accumulation **without an explicit sequential scan**. The key point is operator composition: local temporal dynamics are injected into query/key channels, then propagated over the full causal prefix by attention’s global aggregation. This produces an effective recurrent kernel over history while retaining parallel training-time execution.

## 1. Setup

For one head, let tokenwise features be $x_t \in \mathbb{R}^d$, with projected channels

$$
q_t = W_Q x_t, \quad k_t = W_K x_t, \quad v_t = W_V x_t.
$$

Define one-step temporal mixing (content-gated):

$$
\tilde{k}_t = (1-\gamma_t)k_t + \gamma_t k_{t-1},
$$

$$
\tilde{q}_t = (1-\alpha_t-\beta_t)q_t + \alpha_t q_{t+1} + \beta_t q_{t-1}.
$$

For strictly causal analysis, set $\alpha_t=0$ (Q-causal case). For acausal analysis keep $\alpha_t\ge0$.

Attention output is

$$
y_t = \sum_{j\le t} A_{tj} v_j,
\qquad
A_{tj} = \operatorname{softmax}_{j\le t}\!\left(\frac{\tilde q_t^\top \tilde k_j}{\sqrt d}\right).
$$

## 2. Mechanism: Local Dynamics + Global Accumulation

QK-lerp is local (one-step), but attention is global over the causal prefix. Their composition yields:

1. **Local transition injection** into similarity scores via $\tilde q_t, \tilde k_j$.
2. **Prefix-wide propagation** via $\sum_{j\le t} A_{tj}(\cdot)$.

So the model behaves as a recurrent accumulator over history even though no explicit scan state $h_t$ is updated serially.

## 3. Equivalent Kernel View

Expand key mixing in the logits:

$$
\tilde q_t^\top \tilde k_j
= (1-\gamma_j)\tilde q_t^\top k_j + \gamma_j\tilde q_t^\top k_{j-1}. \tag{1}
$$

Thus each score at index $j$ blends two adjacent key states. Attention then integrates these blended scores over all $j\le t$.

Interpretation: Eq. (1) is a one-step transition operator; causal attention performs the forward accumulation of that operator over the entire prefix. This is the scan-free analogue of recurrent accumulation.

## 4. SSM-Like Recurrence Emergence

Consider linearized attention (drop softmax normalization for analysis):

$$
\hat y_t = \sum_{j\le t} \phi(\tilde q_t)^\top \psi(\tilde k_j)\, v_j. \tag{2}
$$

Define feature-state

$$
s_t = s_{t-1} + \psi(\tilde k_t) v_t^\top,
\qquad
\hat y_t = \phi(\tilde q_t)^\top s_t. \tag{3}
$$

Eq. (3) is an explicit recurrent accumulator in feature space (standard linear-attention state form). QK-lerp modifies $\psi(\tilde k_t), \phi(\tilde q_t)$ through one-step temporal transitions, yielding an SSM-like input-dependent recurrence kernel without scanning over a separate hidden state in the block implementation.

For softmax attention, this exact additive state form is replaced by normalized weights, but the same causal accumulation principle holds: history contributions are integrated through prefix attention with transition-shaped scores.

## 5. Why This Is Scan-Free

Training/prefill complexity uses standard attention kernels:

- build mixed Q/K via shift-and-blend (parallel tensor ops),
- run SDPA/attention over full sequence (parallel matmul kernels).

No explicit sequential recurrence loop over $t$ is required in the block forward pass.

Hence: recurrent-like accumulation behavior, scan-free implementation path.

## 6. Relation to Trapezoidal/SSM Intuition

The K-side one-step blend

$$
\tilde k_t = (1-\gamma_t)k_t + \gamma_t k_{t-1}
$$

matches a data-dependent two-point discretization motif (current + previous endpoint). Q-side blend plays the analogous role on readout alignment. Attention then carries these locally discretized signals across the full prefix, acting as the accumulation engine.

## 7. Causal vs Acausal Modes

- **Causal mode**: $\alpha_t=0$; still scan-free recurrent accumulation over past.
- **Acausal mode (Q-peeking)**: $\alpha_t>0$ introduces bounded future dependency; accumulation remains attention-driven.

With $L$ layers of +1 forward peeking, effective future radius is bounded by $L$ (empirically tight in ULB).

## 8. Practical Consequences

1. You can get recurrent/SSM-like accumulation behavior without implementing scan kernels.
2. K-lerp shapes memory dynamics (state carry/smoothing) at score level.
3. Q-lerp can inject bounded acausal signal while preserving full-prefix accumulation.
4. Decode requires either strict causal cache (no forward peek) or bounded tail-recompute cache when peeking is enabled.

## 9. Minimal Mathematical Claim (for paper body)

> **Claim.** A one-step temporal transition operator applied to Q/K, composed with causal attention aggregation, defines a scan-free model whose output is equivalent to applying a prefix-sum accumulation over transition-shaped similarity kernels. This is functionally recurrent (history-accumulating) though not implemented as an explicit serial state scan.

## 10. Proof Targets (TODO)

### Main theorem targets (TODO)

- Prove softmax-case kernel factorization bounds that connect Eq. (1) to effective convolutional-recurrent memory kernels.
- Quantify approximation gap between normalized softmax accumulator and unnormalized linear-attention state form.
- Derive stability conditions under gate ranges $(\alpha_t,\beta_t,\gamma_t)$.

### Empirical theorem checks (TODO)

- Layer-depth vs effective memory timescale under fixed gate priors.
- Compare with explicit scan-SSM on synthetic state tasks at matched params.
- Ablate K-only, Q-only, QK together to isolate accumulation mechanism.

## 11. Release Figures/Tables (TODO)

### Figures (TODO)

- **Fig 1:** Operator diagram: one-step QK transition followed by causal prefix aggregation.
- **Fig 2:** Logit decomposition heatmap showing adjacent-key coupling term from Eq. (1).
- **Fig 3:** Effective receptive/memory growth across layers (causal and acausal variants).
- **Fig 4:** Decode protocol comparison: strict cache vs bounded tail-recompute.

### Tables (TODO)

- **Table 1:** Synthetic tasks with/without QK-lerp at fixed attention backend.
- **Table 2:** Matched-parameter comparison to scan-based SSM baselines.
- **Table 3:** Throughput/latency for scan-free attention path vs scan kernels.
- **Table 4:** Gate initialization sensitivity and stability metrics.

## 12. Appendix: Decoder Contract (engineering)

- If forward peek disabled ($\alpha_t=0$): strict causal KV-cache valid.
- If forward peek enabled: use radius-$R$ tail recompute (typically $R=\#$peeking layers).
