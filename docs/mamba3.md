Under review as a conference paper at ICLR 2026

# MAMBA-3: IMPROVED SEQUENCE MODELING USING STATE SPACE PRINCIPLES

**Anonymous authors**  
Paper under double-blind review

---

### ABSTRACT

The recent scaling of test-time compute for LLMs has restricted the practical deployment of models to those with strong capabilities that can generate high-quality outputs in an inference-efficient manner. While current Transformer-based models are the standard, their quadratic compute and linear memory bottlenecks have spurred the development of sub-quadratic models with linear-scaling compute with constant memory requirements. However, many recent linear-style models lack certain capabilities or lag behind in quality, and even their linear-time inference is not hardware-efficient. Guided by an inference-first perspective, we introduce three core methodological improvements inspired by the state-space model viewpoint of linear models. We combine a: 1) **more expressive recurrence** derived from discretization, 2) **complex-valued state update rule** that enables richer state tracking, and 3) **multi-input, multi-output formulation** together, resulting in a stronger model. Together with architectural refinements, our **Mamba-3** model achieves significant gains across retrieval, state-tracking, and downstream language modeling tasks. Our new architecture sets the Pareto-frontier for performance under a fixed inference budget and outperforms strong baselines in a head-to-head comparison.

---

## 1 INTRODUCTION

Test-time compute has emerged as a key driver of progress in AI, with techniques like chain-of-thought reasoning and iterative refinement demonstrating that inference-time scaling can unlock new capabilities (Wu et al., 2025; Snell et al., 2024). This paradigm shift makes inference efficiency (Kwon et al., 2023; Li et al., 2024) paramount, as the practical impact of AI systems now depends critically on their ability to perform large-scale inference during deployment. Model architecture design plays a fundamental role in determining inference efficiency, as architectural choices directly dictate the computational and memory requirements during generation. While Transformer-based models (Vaswani et al., 2017) are the current industry standard, they are fundamentally bottlenecked by linearly increasing memory demands through the KV cache and quadratically increasing compute requirements through the self-attention mechanism. These drawbacks have motivated recent lines of work on sub-quadratic models, e.g., state-space models (SSMs), which, despite utilizing only constant memory and linear compute, have comparable or better performance than their Transformer counterparts. Models that benefit the most from this new scaling paradigm perform well on the following three axes: (i) quality, (ii) capability, and (iii) inference efficiency.

Recent model architectures have tried to strike a balance between the three, but many fall short on at least one of these three axes. In particular, Mamba-2 and Gated DeltaNet (GDN), which have gained significant traction and adoption due to their inference efficiency, made architectural design choices that enable their linear compute requirements but sacrifice quality and capabilities (Dao & Gu, 2024; Yang et al., 2025a). For example, Mamba-2 was developed to improve training speed and simplicity over Mamba-1 (Gu & Dao, 2024), opting out of more expressive parameterizations of the underlying SSM and hindering the quality of the model (Dao & Gu, 2024). Linear attention-style models (Katharopoulos et al., 2020) have also been shown to lack certain capabilities, with poor state-tracking abilities, e.g., determining parity of bit sequences, being one of the most notable (Grazzi et al., 2025; Sarrof et al., 2024). In addition, despite these sub-quadratic models being prized for theoretically efficient inference, these inference algorithms are not hardware efficient. In particular, because these algorithms were developed from a training perspective, their decoding phase has low arithmetic intensity (the ratio of FLOPs to memory traffic), resulting in large portions of hardware remaining idle.

To develop more performant models from an inference-first paradigm, we introduce three core methodological changes on top of Mamba-2, influenced by a SSM-centric viewpoint of sub-quadratic models. While many recent models fall into the linear attention framework (Dao & Gu, 2024; Yang et al., 2025a; Sun et al., 2023), we find that the classical SSM toolbox (Kalman, 1960; Gopal, 1993) leads to natural interpretations and improvements on modeling.

**Trapezoidal Discretization.** We discretize the underlying continuous-time dynamical system with a trapezoidal methodology. The final recurrence is a more expressive superset of Mamba-2’s recurrence and can be viewed as a convolution. We combine this new discretization with applied biases on the $B$, $C$, inspired by Yu & Erichson (2025), and find that their synergy is able to empirically replace the short causal convolution in language modeling which was previously hypothesized to be essential for recurrent models.

**Complex-valued State-Space Model.** By viewing the underlying SSM of Mamba-3 as complex-valued, we enable a more expressive state update than Mamba-2’s. This change in update rule, designed to be lightweight for training and inference, overcomes the lack of state-tracking ability common in many current linear models. We emphasize that our complex-valued update rule is equivalent to a data-dependent rotary embedding and can be efficiently computed (Su et al., 2023).

**Multi-Input, Multi-Output SSM.** To improve FLOP-efficiency during decoding, we shift from outer-product-based state update to matrix-multiplication-based state update. In view of the signal processing foundations of SSMs, such a transition exactly coincides with the generalization from a single-input single-output (SISO) sequence dynamic to a multiple-input multiple-output (MIMO) one. Here, we found that MIMO is particularly suitable for inference, as the extra expressivity allows for more compute during state update, without increasing the state size and hence compromising speed.

These three SSM-centric methodological changes are core to our **Mamba-3** mixer primitive. We also make adjustments to the overall architecture to ensure more similarity to the baseline Transformer architecture. Mamba-3 swaps the pre-output projection norm with the more common QK-normalization (Team et al., 2025; OLMo et al., 2025) and makes the short convolution, a common component found in many other sub-quadratic models (Gu & Dao, 2024; Yang et al., 2025a; von Oswald et al., 2025), optional.

We empirically validate our new model on a suite of synthetic and language-modeling tasks.

*   **Better Quality.** Mamba-3 matches or outperforms Mamba-2 and other open-source architectures on standard downstream language modeling evaluations. For example, Mamba-3-1.5B’s average accuracy on all downstream tasks is better than that of its Transformer, Mamba-2, and Gated DeltaNet counterparts.
*   **New Capabilities.** Mamba-3’s complexification of the SSM state enables the model to solve synthetic state-tracking tasks that Mamba-2 cannot. We empirically demonstrate that the efficient RoPE-like calculation is able to near perfectly solve arithmetic tasks, while Mamba-3 without RoPE and Mamba-2 perform not better than random guessing.
*   **Stronger Inference Efficiency.** Mamba-3’s MIMO variant retains the same state size while enabling better hardware utilization compared to standard Mamba-3 and other models. Its improved performance without increased memory requirements pushes the pareto-frontier of inference efficiency.

## 2 PRELIMINARIES

### 2.1 NOTATION
Scalars are denoted by plain-text letters (e.g., $x, y$). Tensors, including vectors and matrices, are denoted by bold letters (e.g., $\mathbf{h}, \mathbf{C}$). The shape of the tensor can be inferred from the context. We denote the input sequence length as $T$, the model dimension as $D$, and the SSM state size as $N$. For time indices, we use subscripts (e.g., $x_t$ for the input at time $t$). The Hadamard product between two tensors is denoted by $\odot$. For a vector of size $\mathbf{v} \in \mathbb{R}^d$, we denote $\text{Diag}(\mathbf{v}) \in \mathbb{R}^{d \times d}$ as the diagonal matrix with the vector $\mathbf{v}$ as the diagonal, and for products of scalars across time steps, we use the notation $\alpha_{t \cdots s} = \alpha_{t:s}^\times = \prod_{i=s}^t \alpha_i$.

### 2.2 SSM PRELIMINARIES
State Space Models (SSMs) describe continuous-time linear dynamics via
$$\mathbf{\dot{h}}(t) = \mathbf{A}(t) \mathbf{h}(t) + \mathbf{B}(t) x(t), \quad y(t) = \mathbf{C}(t)^\top \mathbf{h}(t),$$
where $\mathbf{h}(t) \in \mathbb{R}^N$ is the hidden state, $x(t) \in \mathbb{R}$ the input, and $\mathbf{A}(t) \in \mathbb{R}^{N \times N}, \mathbf{B}(t), \mathbf{C}(t) \in \mathbb{R}^N$. For discrete sequences with step size $\Delta_t$, Euler’s discretization gives the recurrence
$$\mathbf{h}_t = e^{\Delta_t \mathbf{A}_t} \mathbf{h}_{t-1} + \Delta_t \mathbf{B}_t x_t, \quad y_t = \mathbf{C}_t^\top \mathbf{h}_t.$$

**Mamba-2’s parameterization.** Mamba-2 (Dao & Gu, 2024) makes the SSM *data-dependent* and hardware-efficient by (i) projecting $\mathbf{A} = A \in \mathbb{R}_{<0}$, and $\mathbf{B}, \mathbf{C} \in \mathbb{R}^N$ from the current token and (ii) choosing transition matrix $\mathbf{A} = A$ as a data-dependent scalar. Writing $\alpha_t := e^{\Delta_t A_t} \in (0, 1)$ and $\gamma_t := \Delta_t$, the update becomes
$$\mathbf{h}_t = \alpha_t \mathbf{h}_{t-1} + \gamma_t \mathbf{B}_t x_t, \quad y_t = \mathbf{C}_t^\top \mathbf{h}_t.$$
The scalar $A_t < 0$ is an input-dependent forget-gate (decay) $\alpha_t$, and the parameter selectivity $\Delta_t$ jointly controls the forget-gate ($\alpha_t = \exp(\Delta_t A_t)$) and the input-gate ($\gamma_t = \Delta_t$): larger $\Delta_t$ forgets faster and up-weights the current token more strongly, while smaller $\Delta_t$ retains the hidden state with minimal contributions from the current token.

### 2.3 STRUCTURED MASKED REPRESENTATION AND STATE SPACE DUALITY
Dao & Gu (2024) show that a large class of SSMs admit a *matrix* form that vectorizes the time-step recurrence. For instance, Mamba-2’s recurrence can be vectorized as a masked matrix multiplication,
$$\mathbf{Y} = (\mathbf{L} \odot \mathbf{C}\mathbf{B}^\top)\mathbf{X} = \left( \begin{bmatrix} 1 & & & \\ \alpha_1 & 1 & & \\ \vdots & \ddots & \ddots & \\ \alpha_{T \cdots 1} & \cdots & \alpha_T & 1 \end{bmatrix} \odot \mathbf{C}\mathbf{B}^\top \right) \mathbf{X}, \tag{1}$$
where $\mathbf{L} \in \mathbb{R}^{T \times T}$ is the structured mask, $\mathbf{B}, \mathbf{C} \in \mathbb{R}^{T \times N}$, $\mathbf{X} \in \mathbb{R}^{T \times D}$ is the input to the SSM and $\mathbf{Y} \in \mathbb{R}^{T \times D}$ is its output. Within this form, Mamba-2 can be viewed as a type of linear attention by setting $\mathbf{Q} = \mathbf{C}$, $\mathbf{K} = \mathbf{B}$, $\mathbf{V} = \mathbf{X}$ and viewing $\mathbf{L}$ as a causal, data-dependent mask. When all $\alpha = 1$, the expression reduces to (causal) linear attention (Katharopoulos et al., 2020).

## 3 MODEL DESIGN FROM A STATE-SPACE VIEWPOINT
We introduce Mamba-3, with three new innovations rooted in classical state-space theory: trapezoidal discretization for more expressive dynamics, complex-valued state spaces for state-tracking, and multi-input multi-output (MIMO) to improve hardware utilization. These advances address the quality, capability, and efficiency limitations of current sub-quadratic architectures.

### 3.1 TRAPEZOIDAL DISCRETIZATION
Structured SSMs are naturally defined as continuous-time dynamical systems that map input functions, $x(t) \in \mathbb{R}$, to output functions, $y(t) \in \mathbb{R}$, for time $t > 0$. In sequence modeling, however, the data is only observed at discrete time steps, which then requires applying a *discretization step* to the SSM to transform its continuous-time dynamics into a discrete recurrence. The preliminary step in deriving Mamba-3’s discretization is to apply the Variation of Constants formula (Proposition 5), which decomposes the hidden state into an exponentially decay term and a state update term “information” term dependent on the most recent inputs.

The first step in deriving the discretized recurrence is to approximate the “state-update” integral in equation 10. A straightforward choice, used in Mamba-2, is applying Euler’s rule (Suli & Mayers, 2003), which approximates the integral by holding the (right) endpoint constant throughout the interval (Fig. 1). This yields Mamba-2’s recurrence,
$$\mathbf{h}_t = e^{\Delta_t A_t} \mathbf{h}_{t-1} + (\tau_t - \tau_{t-1})e^{(\tau_t - \tau_t) A_t} \mathbf{B}_t x_t \approx e^{\Delta_t A_t} \mathbf{h}_{t-1} + \Delta_t \mathbf{B}_t x_t. \tag{2}$$

![Figure 1: Left: The structured mask induced by the generalized trapezoid rule is a product of the decay and convolutional mask. Right: Euler (hold endpoint) vs trapezoidal rule (average endpoints).](figure1_placeholder)

However, Euler’s rule provides only a first-order approximation to the “state-update” integral: local truncation error is $O(\Delta_t^2)$, which accumulates across steps to yield a global error of $O(\Delta_t)$ over the sequence. In contrast, we adopt a *generalized trapezoidal rule*, which provides a second-order accurate approximation of the integral, offering improved accuracy over the Euler’s rule. Specifically, it approximates the integral with a *data-dependent, convex combination* of both interval endpoints. This generalization extends the classical trapezoidal rule (Suli & Mayers, 2003), which simply averages the interval endpoints, by allowing for a *data-dependent convex combination* (Fig. 1).

**Proposition 1 (Generalized Trapezoidal Discretization).** *Approximating the state-update integral in equation 10 by the general trapezoidal rule yields the recurrence,*
$$\mathbf{h}_t = e^{\Delta_t A_t} \mathbf{h}_{t-1} + (1 - \lambda_t)\Delta_t e^{\Delta_t A_t} \mathbf{B}_{t-1} x_{t-1} + \lambda_t \Delta_t \mathbf{B}_t x_t, \tag{3}$$
$$:= \alpha_t \mathbf{h}_{t-1} + \beta_t \mathbf{B}_{t-1} x_{t-1} + \gamma_t \mathbf{B}_t x_t, \tag{4}$$
*where $\lambda_t \in [0, 1]$ is a data-dependent scalar, $\alpha_t := e^{\Delta_t A_t}$, $\beta_t := (1 - \lambda_t)\Delta_t e^{\Delta_t A_t}$, $\gamma_t := \lambda_t \Delta_t$.*

**Remark 1 (Expressivity).** Our scheme is a generalization of a) The classical trapezoid rule which is recovered when $\lambda_t = \frac{1}{2}$. b) Mamba-2’s Euler’s rule, which is recovered when $\lambda_t = 1$.

**Remark 2 (Error Rate).** This is a second-order discretization with local truncation error $O(\Delta_t^3)$ and global error $O(\Delta_t^2)$ over the sequence under standard stability assumptions, provided that the trapezoidal parameter satisfies $\lambda_t = \frac{1}{2} + O(\Delta_t)$. However, our ablations indicate that not enforcing this constraint is the best for empirical performance. See Appendix B.2, B.3 for details.

#### 3.1.1 TRAPEZOIDAL DISCRETIZATION IS A CONVOLUTIONAL MASK
We can view the generalized trapezoidal discretization as applying a *data-dependent* convolution of size two on the projected input, $\mathbf{B}_t x_t$, to the SSM. We now show that a similar vectorization to Equation (1) holds with the generalized trapezoidal discretization. Unrolling the recurrence starting from $\mathbf{h}_0 = \gamma_0 \mathbf{B}_0 x_0$ results in $\mathbf{h}_T = \alpha_{T \cdots 2}(\gamma_0 \alpha_1 + \beta_1) \mathbf{B}_0 x_0 + \cdots + \gamma_T \mathbf{B}_T x_T$.

Unrolling these rows shows that the mask induced by the trapezoidal update is no longer a fixed averaging of endpoints (as in the classical trapezoidal rule), but a *data-dependent convex combination* of the two interval endpoints. In the SSD representation, this corresponds to a mask $\mathbf{L}$:
$$\begin{bmatrix} \gamma_0 & & \\ (\gamma_0 \alpha_1 + \beta_1) & \gamma_1 & \\ \alpha_2(\gamma_0 \alpha_1 + \beta_1) & \gamma_2 & \\ \vdots & \vdots & \ddots \\ \alpha_{T \cdots 2}(\gamma_0 \alpha_1 + \beta_1) & \cdots & \gamma_T \end{bmatrix} = \begin{bmatrix} 1 & & \\ \alpha_1 & 1 & \\ \alpha_2 \alpha_1 & \vdots & \ddots \\ \vdots & \vdots & \\ \alpha_{T \cdots 1} & \cdots & 1 \end{bmatrix} \begin{bmatrix} \gamma_0 & & \\ \beta_1 & \gamma_1 & \\ 0 & \gamma_2 & \\ \vdots & \vdots & \ddots \\ 0 & \cdots & \gamma_T \end{bmatrix}. \tag{5}$$
Here, the first factor is precisely the lower-triangular decay mask from Mamba-2, while the second factor encodes the size two convolution induced by the trapezoidal rule through the coefficients $(\beta_t, \gamma_t)$. We provide a rigorous proof for this decomposition in Appendix B.1.

### 3.2 COMPLEX-VALUED SSMS
Modern SSMs are designed with efficiency as the central goal, motivated by the need to scale to larger models and longer sequences. For instance, successive architectures have progressively simplified the state transition matrix: S4 (Gu et al., 2022a) used complex-valued Normal plus Low Rank (NPLR) matrices, Mamba (Gu & Dao, 2024) reduced this to a diagonal of reals, and Mamba-2 (Dao & Gu, 2024) further simplified it to a single scalar. Although these simplifications largely maintain language modeling performance, recent works (Merrill et al., 2025; Sarrof et al., 2024; Grazzi et al., 2025) have shown that they degrade the capabilities of the model on simple state-tracking tasks such as parity and modular arithmetic, which can be solved by a one-layer LSTM.

This limitation, formalized in Theorem-1 of (Grazzi et al., 2024), arises from restricting the eigenvalues of the transition matrix to real numbers, which cannot represent “rotational” hidden state dynamics. For instance, consider the parity function on binary inputs $\{0, 1\}$, defined as $\sum_i x_t \text{ mod } 2$. This task can be performed using update: $\mathbf{h}_t = \mathbf{R}(\pi x_t)\mathbf{h}_{t-1}$, where $\mathbf{R}(\cdot)$ is a 2-D rotation matrix. Such rotational dynamics cannot be expressed with real eigenvalues.

To recover this capability, we begin with complex SSMs (6), which are capable of representing state-tracking dynamics. We show that, under discretization (Proposition 5), complex SSMs can be formulated as a real SSMs with a *block-diagonal transition matrix composed of $2 \times 2$ rotation matrices* (Proposition 2). We then show that this is equivalent to applying *data-dependent rotary embeddings* on both the input and output projections $\mathbf{B}, \mathbf{C}$ respectively. This result establishes a theoretical connection between complex SSMs and data-dependent RoPE embeddings (Proposition 3). Finally, this allows for an efficient implementation of the complex-valued SSM via the “RoPE trick”, enabling efficient complex-valued state transition matrix with minimal computational overhead over real-valued SSMs.

**Proposition 2 (Complex-to-Real SSM Equivalence).** *Consider a complex-valued SSM*
$$\mathbf{\dot{h}}(t) = \text{Diag}(\mathbf{A}(t) + i\mathbf{\theta}(t)) \mathbf{h}(t) + (\mathbf{B}(t) + i\mathbf{\hat{B}}(t)) x(t), \tag{6}$$
$$y(t) = \text{Re}((\mathbf{C}(t) + i\mathbf{\hat{C}}(t))^\top \mathbf{h}(t)),$$
*where $\mathbf{h}(t) \in \mathbb{C}^{N/2}$, $\mathbf{\theta}(t), \mathbf{B}(t), \mathbf{\hat{B}}(t), \mathbf{C}(t), \mathbf{\hat{C}}(t) \in \mathbb{R}^{N/2}$, and $x(t), A(t) \in \mathbb{R}$. Under Euler discretization, this system is equivalent to a real-valued SSM*
$$\mathbf{h}_t = e^{\Delta_t A_t} \mathbf{R}_t \mathbf{h}_{t-1} + \Delta_t \mathbf{B}_t x_t, \tag{7}$$
$$y_t = \mathbf{C}_t^\top \mathbf{h}_t,$$
*with state $\mathbf{h}_t \in \mathbb{R}^N$, projections*
$$\mathbf{B}_t = \begin{bmatrix} \mathbf{B}_t \\ \mathbf{\hat{B}}_t \end{bmatrix} \in \mathbb{R}^N, \quad \mathbf{C}_t = \begin{bmatrix} \mathbf{C}_t \\ -\mathbf{\hat{C}}_t \end{bmatrix} \in \mathbb{R}^N,$$
*and a transition matrix*
$$\mathbf{R}_t = \text{Block}\{R(\Delta_t \theta_t[i])\}_{i=1}^{N/2} \in \mathbb{R}^{N \times N}, \quad R(\Theta) = \begin{bmatrix} \cos(\Theta) & -\sin(\Theta) \\ \sin(\Theta) & \cos(\Theta) \end{bmatrix}.$$
*The proof is in Appendix C.1.*

Proposition 2 shows that the discretized complex SSM has an equivalent real SSM with doubled state dimension ($N$), and a block-diagonal transition matrix multiplied with a scalar decay, where each $2 \times 2$ block is a data-dependent rotation matrix ($e^{\Delta_t A_t} \mathbf{R}_t$). We now show that the rotations can equivalently be absorbed into the input and output projections $\mathbf{B}_t, \mathbf{C}_t$, yielding an equivalent view that complex SSMs are real SSMs equipped with data-dependent rotary embeddings (RoPE).

**Proposition 3 (Complex SSM, Data-Dependent RoPE Equivalence).** *Under the notation established in Proposition 2, consider the real SSM defined in Eq. 7 unrolled for $T$ time-steps. The output of the above SSM is equivalent to that of a vanilla scalar transition matrix-based SSM (Eq. 2) with a data-dependent rotary embedding applied on the $\mathbf{B}, \mathbf{C}$ components of the SSM defined as:*
$$\mathbf{h}_t = e^{\Delta_t A_t} \mathbf{h}_{t-1} + \left( \prod_{i=0}^t \mathbf{R}_i^\top \right) \mathbf{B}_t x_t, \quad y_t = \left( \left( \prod_{i=0}^t \mathbf{R}_i^\top \right) \mathbf{C}_t \right)^\top \mathbf{h}_t \tag{8}$$
*where the matrix production represents right matrix multiplication, e.g., $\prod_{i=0}^1 \mathbf{R}_i = \mathbf{R}_0 \mathbf{R}_1$. We denote employing the vanilla SSM to compute the Complex SSM as “RoPE trick”.*
*The proof is in Appendix C.2.*

To observe the connection of complex SSMs to RoPE embeddings, note that in the above proposition, the data-dependent rotations $\mathbf{R}_i$ are aggregated across time-steps and applied to $\mathbf{C}, \mathbf{B}$, which, by the State Space Duality of Dao & Gu (2024), correspond to the Query ($\mathbf{Q}$) and Key ($\mathbf{K}$) components of Attention. Analogously, vanilla RoPE (Su et al., 2023) applies data-independent rotation matrices, where the rotation angles follow a fixed frequency schedule $\theta[i] = 10000^{-2i/N}$.

**Remark 3 (Generality).** Proposition 3 extends to the fully general case where the transition is given by any complex matrix. By the complex diagonalization theorem, such a matrix is unitarily equivalent to a complex diagonal matrix, $\text{Diag}(\mathbf{A}(t) + i\mathbf{\theta}(t))$ with $\mathbf{A}(t) \in \mathbb{R}^N$. However, in practice, we restrict $\mathbf{A}(t)$ to a scalar, mirroring the simplification from Mamba to Mamba-2, to enable faster implementation by avoiding GPU memory bottlenecks.

**Proposition 4 (Rotary Embedding Equivalence with Trapezoidal Discretization).** *Discretizing a complex SSM with the trapezoidal rule (Proposition 1) yields the recurrence*
$$\mathbf{h}_t = \alpha_t \mathbf{h}_{t-1} + \beta_t \left( \prod_{i=0}^{t-1} \mathbf{R}_i^\top \right) \mathbf{B}_{t-1} x_{t-1} + \gamma_t \left( \prod_{i=0}^t \mathbf{R}_i^\top \right) \mathbf{B}_t x_t,$$
$$y_t = \left( \left( \prod_{i=0}^t \mathbf{R}_i^\top \right) \mathbf{C}_t \right)^\top \mathbf{h}_t. \tag{9}$$
*Here $\mathbf{R}_t$ is the block-diagonal rotation matrix defined in Proposition 3.*
*The proof is in Appendix C.3.*

**Remark 4 (RoPE Trick).** Complex SSMs discretized with the general trapezoidal rule of a complex SSM naturally admit the RoPE trick we established for SSMs discretized with Euler’s rule.

### 3.3 MULTI-INPUT, MULTI-OUTPUT
During the decoding phase of autoregressive inference, outputs are generated one token at a time, and performance is typically measured using in *Tokens generated Per Second (TPS)*. In this metric, sub-quadratic models, such as Mamba-2 (Dao & Gu, 2024), have a significant advantage over standard Transformer-style attention, since they feature a fixed-size hidden state (Equation (2)) rather than maintaining a key–value (KV) cache that grows linearly with the sequence length.

TPS, however, does not explicitly factor in hardware efficiency, where we aim to be in a compute-bound regime (as opposed to memory-bound) in order to fully utilize on-chip accelerators. To better characterize hardware efficiency, we would need to consider the arithmetic intensity of token generation. Recall that arithmetic intensity is defined as FLOPs divided by the number of input-output bytes, for a given op. In order to fully utilize both the accelerators and the bandwidth, we would like the arithmetic intensity to match the ops:byte ratio of the hardware, which in the case of NVIDIA H100-SXM5, is 295.2 bfloat16 ops per second with respect to the DRAM, and 31.9 bfloat16 ops per second with respect to the SRAM [Fleetwood].

Table 2(a) shows the arithmetic intensity for a single generation in the SSM component of Mamba (with respect to 2-byte data). We see that it falls far short of a compute-bound regime, and moreover it is not clear how one can adjust the existing parameters in Mamba to mitigate the lack of hardware efficiency. We note that this observation applies generally to other sub-quadratic models, such as causal linear attention.

| Input | Output | FLOPs | Arithmetic Intensity |
| :--- | :--- | :--- | :--- |
| $\mathbf{h}_t: (n, p)$ | $y_t: (p)$ | $5pn$ | $\frac{5pn}{2(1 + 2n + p + np)} \approx 2.5 = \Theta(1)$ |
| $x_t: (p)$ | | | |
| $a_t: (1)$ | | | |
| $b_t: (n)$ | | | |
| $c_t: (n)$ | | | |

**(a) SISO (2-byte data).**

| Input | Output | FLOPs | Arithmetic Intensity |
| :--- | :--- | :--- | :--- |
| $\mathbf{h}_t: (n, p)$ | $y_t: (p, r)$ | $4nrp + 2np$ | $\frac{p(4nr + 2n)}{2(1 + 2nr + pr + np)} \approx 2r = \Theta(r)$ |
| $x_t: (p, r)$ | | | |
| $a_t: (1)$ | | | |
| $b_t: (n, r)$ | | | |
| $c_t: (n, r)$ | | | |

**(b) MIMO (2-byte data).**

**Figure 2: Arithmetic Intensity for (a) SISO, (b) MIMO. Batch and head dimensions cancel out.**

In light of this, we made the following simple adjustment to our recurrent relation: instead of transforming the input $x_t \in \mathbb{R}^p$ to state $\mathbf{H}_t \in \mathbb{R}^{n \times p}$ via an outer product, i.e., $\mathbf{H}_t \leftarrow a_t \mathbf{H}_{t-1} + b_t \otimes x_t$, we made such a transformation via a matrix product, i.e., $\mathbf{H}_t \leftarrow a_t \mathbf{H}_{t-1} + \mathbf{B}_t \mathbf{X}_t^\top$, where $\mathbf{B}_t \in \mathbb{R}^{n \times r}$ and $\mathbf{X}_t \in \mathbb{R}^{p \times r}$ are now matrices with an additional rank $r$. The emission from state to output similarly acquire an extra rank $r$, i.e., $\mathbf{Y}_t \in \mathbb{R}^{r \times p} \leftarrow \mathbf{C}_t^\top \mathbf{H}_t$, where $\mathbf{C}_t \in \mathbb{R}^{n \times r}, \mathbf{H}_t \in \mathbb{R}^{n \times p}$. This simple change increases the arithmetic intensity of recurrence, which now scales with the rank $r$ (Figure 2(b)). Hence, by increasing $r$, arithmetic intensity improves and shifts decode generation towards a more compute-bound regime. This increase in FLOPs during decode does not compromise runtime, as the operation is bounded by the I/O of state $\mathbf{H}_t \in \mathbb{R}^{n \times p}$.

Moreover, moving from outer-product-based state update to matrix-product-based coincides exactly with generalizing from SISO to MIMO SSM, with the rank $r$ being the MIMO rank. Such a generalization recovers a key expressive feature of SSMs in classical literature; indeed, there has been previous work, namely Smith et al. (2023), that explored MIMO SSM as a drop-in replacement of attention, albeit not in the context of Mamba and not necessarily with inference in view. We note that training and prefilling is generally compute bound, resulting in MIMO incurring increased costs during these stages, while decoding, a memory-bound operation, sees very little increase in latency when utilizing MIMO over SISO.

Details of the MIMO formulation for Mamba-3 are provided in Appendix D.

### 3.4 MAMBA-3 ARCHITECTURE
The Mamba-3 block retains the overall layout of its predecessor while introducing several key modifications. Most notably, the SSD layer is replaced with the more expressive trapezoidal SSM defined in Proposition 4. The extra normalization layer, first introduced between Mamba-1 and Mamba-2 for training stability, is repositioned to follow the $\mathbf{B}, \mathbf{C}$ projection, mirroring the QK-Norm commonly used in modern Transformers (Henry et al., 2020; Wortsman et al., 2023). Inspired by the findings of Yu & Erichson (2025), which prove adding channel-specific bias to $\mathbf{B}$ in a blockwise variant of Mamba-1 grants universal approximation capabilities, Mamba-3 incorporates a head-specific, channel-wise bias into both the $\mathbf{B}$ and $\mathbf{C}$ components after its normalization. These learnable biases are data-independent parameters that are initialized to all ones and independent across $\mathbf{B}$ and $\mathbf{C}$ (ablations for bias parameterization can be found in Appendix G). Our trapezoidal discretization complements this bias, empirically eliminating the need for the original short causal convolution and its accompanying activation function (Section 4.3). Mamba-3 employs the SISO SSM by default, though we view its MIMO variant as a flexible option that can be toggled depending on inference requirements. The overall architecture follows the Llama design (Grattafiori et al., 2024), alternating Mamba-3 and SwiGLU blocks with pre-normalization.

## 4 EMPIRICAL VALIDATION
We empirically validate our SSM-centric methodological changes through the Mamba-3 model on a host of synthetic and real world tasks. Section 4.1 compares our SISO-variant of Mamba-3 on language modeling and retrieval-based tasks, while Section 4.2 demonstrates inference efficiency of Mamba-3 and MIMO Mamba-3’s benefits over SISO Mamba-3 under fixed inference compute. We ablate the impact of our new discretization and BC bias on performance and show that complexification of the SSM leads capabilities that prior SSMs such as Mamba-2 lacked in Section 4.3.

### 4.1 LANGUAGE MODELING
All models are pretrained with 100B tokens of the FineWeb-Edu dataset (Penedo et al., 2024) with the Llama-3.1 tokenizer (Grattafiori et al., 2024) at a 2K context length with the same standard training protocol. Training and evaluation details can be found in Appendix E.

Across all four model scales, Mamba-3 outperforms popular baselines at various downstream tasks (Table 1). We highlight that Mamba-3 does not utilize the short convolution that has been empirically identified as an important component in many performant linear models (Allen-Zhu, 2025).

#### 4.1.1 RETRIEVAL CAPABILITIES
Beyond standard language modeling, an important measure for linear models is their retrieval ability — how well they can recall information from earlier in the sequence (Arora et al., 2025a;b). Unlike attention models, which can freely revisit past context with the growing KV cache, linear models must compress context into a fixed-size state. This trade-off is reflected in the Transformer baseline’s substantially stronger retrieval scores. To evaluate Mamba-3 under this lens, Table 2 compares it against baselines on both real-world and synthetic needle-in-a-haystack (NIAH) tasks (Hsieh et al., 2024), using our pretrained 1.5B models from Section 4.1. We restrict the task sequence length to 2K tokens to match the training setup and adopt the cloze-style format for our real-world tasks to mirror the next-token-prediction objective, following Arora et al. (2025b; 2024).

Mamba-3 is competitive on real-world associative recall and question-answering but struggles when extracting information from semi-structured or unstructured data. On synthetic NIAH tasks, however, Mamba-3 surpasses or matches baselines on most cases and notably demonstrates markedly better out-of-distribution retrieval abilities than its Mamba-2 predecessor.

**Table 1: Downstream language modeling evaluations on models trained with 100B FineWeb-Edu tokens.** Best results for each size are **bolded**, and second best are underlined. All models are trained with the same procedure. Mamba-3 outperforms Mamba-2 and others at every model scale.

| Model | FW-Edu ppl ↓ | LAMB. ppl ↓ | LAMB. acc ↑ | HellaS. acc n ↑ | PIQA acc ↑ | Arc-E acc ↑ | Arc-C acc n ↑ | WinoGr. acc ↑ | OBQA acc ↑ | Average acc ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformer-180M | 16.89 | 45.0 | 32.5 | 39.0 | **67.1** | 59.8 | 27.9 | 51.2 | 21.8 | 42.8 |
| Gated DeltaNet-180M | 16.61 | **35.9** | **33.7** | 40.2 | 66.8 | 59.6 | **28.5** | 51.2 | 21.6 | 43.1 |
| Mamba-2-180M | 16.76 | 41.8 | 30.9 | 40.1 | 66.8 | 60.1 | 27.3 | **52.0** | **23.2** | 42.9 |
| **Mamba-3-180M (SISO)** | **16.59** | 37.7 | 32.5 | **40.8** | 66.1 | **61.5** | 27.9 | **52.0** | 22.8 | **43.4** |
| | | | | | | | | | | |
| Transformer-440M | 13.03 | 21.2 | **41.7** | 50.5 | 69.9 | 67.6 | 34.6 | **56.7** | **26.0** | 49.6 |
| Gated DeltaNet-440M | 13.12 | **19.0** | 40.4 | 50.5 | 70.5 | 67.5 | 34.0 | 55.3 | 25.8 | 49.1 |
| Mamba-2-440M | 13.00 | 19.6 | 40.8 | **51.7** | 70.6 | 68.8 | **35.0** | 54.1 | **26.0** | 49.6 |
| **Mamba-3-440M (SISO)** | **12.87** | 19.6 | 40.2 | **51.7** | **71.9** | **68.9** | 34.4 | 55.8 | **26.0** | **49.8** |
| | | | | | | | | | | |
| Transformer-880M | 11.42 | 15.0 | 44.7 | 57.2 | 72.6 | 71.6 | 39.2 | 57.7 | 26.8 | 52.8 |
| Gated DeltaNet-880M | 11.39 | **12.7** | 47.1 | 57.5 | 72.6 | 72.5 | 38.8 | 57.9 | **30.6** | 53.9 |
| Mamba-2-880M | 11.35 | 13.8 | 45.0 | 58.1 | 72.5 | 72.3 | 38.7 | 56.8 | 30.2 | 53.4 |
| **Mamba-3-880M (SISO)** | **11.23** | 12.9 | **47.2** | **58.8** | **73.6** | **72.7** | **40.2** | **58.4** | 30.0 | **54.4** |
| | | | | | | | | | | |
| Transformer-1.5B | 10.51 | 11.1 | **50.3** | 60.6 | 73.8 | 74.0 | 40.4 | 58.7 | 29.6 | 55.4 |
| Gated DeltaNet-1.5B | 10.51 | **10.8** | 49.9 | 60.5 | **74.3** | 73.3 | 40.4 | **61.5** | 30.4 | 55.7 |
| Mamba-2-1.5B | 10.47 | 12.0 | 47.8 | 61.4 | 73.6 | 75.3 | 41.8 | 57.5 | **32.6** | 55.7 |
| **Mamba-3-1.5B (SISO)** | **10.35** | 10.9 | 49.4 | **61.9** | 73.6 | **75.9** | **42.7** | 59.4 | 32.0 | **56.4** |

**Table 2: Retrieval capabilities measured by a mixture of real-world and synthetic retrieval tasks.**

| Model (1.5B) | SWDE | SQUAD | FDA | TQA | NQ | Drop | NIAH-S1 (1k/2k/4k) | NIAH-S2 (1k/2k/4k) | NIAH-S3 (1k/2k/4k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Transformer | 48.9 | 46.6 | 58.4 | 67.5 | 31.7 | 26.4 | 100/100/0 | 92.2/100/0 | 98.6/99.4/0 |
| Gated DeltaNet | 32.7 | 40.0 | 28.3 | 63.5 | 25.7 | 24.5 | 100/100/99.8| 100/93.8/49.8 | 83.8/68.4/34.2 |
| Mamba-2 | 30.7 | 39.1 | 23.7 | 64.3 | 25.1 | 28.5 | 100/99.6/62.0| 100/53.8/11.8 | 95.8/87.4/13.4 |
| **Mamba-3 (SISO)** | 28.5 | 40.1 | 23.4 | 64.5 | 26.5 | 27.4 | 100/100/88.2| 100/95.4/50.6 | 92.4/81.4/34.2 |

### 4.2 INFERENCE EFFICIENCY
In this section, we investigate our methodological changes in the context of inference performance. We first present our inference benchmark in Section 4.2.1; we then establish a framework for comparing the inference performance in Section 4.2.2. Finally, we focus on the effectiveness of MIMO in Section 4.2.3.

#### 4.2.1 FAST MAMBA-3 KERNELS
We complement Mamba-3’s methodological advances with optimized kernels that deliver fast inference in practical settings. Specifically, we implement a new series of inference kernels for Mamba-3—using Triton for the forward (prefill) path and CuTe-DSL for decode—and compare their per-token decode latency against the released Triton kernels for Mamba-2 and Gated DeltaNet (GDN) in Table 3. The evaluation uses the setting: a decode step at batch size 128 on a single H100 for 1.5B-parameter models with model dimension 2048, state dimension $\in \{64, 128\}$ in both FP32 and BF16 datatypes. Across all configurations, SISO achieves the lowest latency amongst baselines, while MIMO incurs only a minor overhead relative to SISO. This indicates that our CuTe-DSL decode implementation is competitive and that the additional components of Mamba-3 (trapezoidal update, complex-valued state, and MIMO projections) are lightweight. This supports our overall inference-first perspective: the Mamba-3 admits **simple, low-latency implementation** while providing strong empirical performance. A thorough analysis, including prefill and prefill with decode results are provided in Appendix H.

**Table 3: Latency (in milliseconds) comparison across models, precision, and $d_{\text{state}}$ values.** Both Mamba-3 SISO and MIMO are faster than the Mamba-2 and Gated DeltaNet at the commonly used bf16, $d_{\text{state}} = 128$ setting.

| Model | FP32 ($d_{\text{state}}=64$) | FP32 ($d_{\text{state}}=128$) | BF16 ($d_{\text{state}}=64$) | BF16 ($d_{\text{state}}=128$) |
| :--- | :---: | :---: | :---: | :---: |
| Mamba-2 | 0.295 | 0.409 | 0.127 | 0.203 |
| Gated DeltaNet | 0.344 | 0.423 | 0.176 | 0.257 |
| **Mamba-3 (SISO)** | **0.261** | **0.356** | **0.106** | **0.152** |
| **Mamba-3 (MIMO)** | 0.285 | 0.392 | 0.136 | 0.185 |

#### 4.2.2 PARETO FRONTIER FOR INFERENCE EFFICIENCY
For Mamba and many variants of sub-quadratic models, the generation of tokens during decoding is heavily dominated by memory I/O due to the low arithmetic intensity of computing the recurrent update (c.f. Section 3.3). Furthermore, among the data being transferred, the latent state $\mathbf{H}_t$ dominates in terms of size. Indeed, from Table 3, we see that the runtime scales with $d_{\text{state}}$, which configures the size of the hidden state.

As $d_{\text{state}}$ dominates the decode runtime for the sub-quadratic models considered in this paper, we opt to use it as a proxy for inference speed. By plotting the validation perplexity (itself a proxy for model performance) as a function of $d_{\text{state}}$, we aim to formulate a holistic picture about how the sub-quadratic models can trade off performance with inference speed.

Figure 3 shows such a Pareto front for the Mamba variants models considered in this paper. For each data point, we train a 440M parameter model to $2 \times$ Chinchilla optimal tokens on the Fineweb-Edu dataset, where the model is configured with a $d_{\text{state}}$ of $\{16, 32, 64, 128\}$. As expected, we observe an inverse correlation between validation loss and $d_{\text{state}}$; moreover, we noticed a general downward shift on the Pareto front moving from Mamba-2 to Mamba-3. A further downward shift is observed when moving from the SISO variant of Mamba-3 to the MIMO variant of Mamba-3 (where we set the Mimo rank $r = 4$ and decrease our MLP inner dimension to parameter match the SISO variants). We expand the comparison to include the Gated DeltaNet baseline in Figure 7. The results highlight both the expressivity gain coming our methodology change as well as the effectiveness of the MIMO mechanism in improving decoding efficiency.

#### 4.2.3 MIMO ENHANCES INFERENCE EFFICIENCY
MIMO, with its higher arithmetic intensity, increases the decoding FLOPs without significantly increasing decode runtime (Table 3). The implication is that any performance gain from MIMO translates into efficiency gain in decoding: a conclusion supported by the downward shift of the MIMO pareto curve we observed in Section 4.2.2.

We aim to further verify the gain from MIMO by investigating its language-modeling capabilities. To that end, we train a 440M and 820M parameter MIMO models with MIMO rank $r = 4$ on 100B tokens on Fineweb-Edu (i.e., same setting as the 440M parameter run in Section 4.1; we are currently training the 1.5B model). To ensure the total parameter count equals SISO, we decrease the inner dimension of the MLP layers to compensate for the increase due to the MIMO projections.

On both validation perplexity and our suite of language evaluation tasks (Table 6), we see significant gain when moving from SISO to MIMO. Namely, we attain a perplexity gain of 0.16 on the 100B tokens run, and Figure 3 illustrates the downward shift in our validation loss. On the language evaluation front, we see significant gain on most tasks when compared to SISO, resulting in an overall gain of 1.2 point over SISO. This strongly supports MIMO as a SSM-centric technique to improve model quality without compromising decoding speed.

### 4.3 SSM-CENTRIC METHODOLOGICAL ABLATIONS
Table 4a ablates the changes made to the core SSM component, mainly the introduction of BC bias and trapezoidal discretization. We report the pretraining test perplexity on models at the 440M scale, trained for Chinchilla optimal tokens. We find that the bias and trapezoidal SSM synergize well and make the short convolution utilized by many current linear models redundant.

We empirically demonstrate that data-dependent RoPE in Mamba-3 enables state tracking. Following Grazzi et al. (2025), we evaluate on tasks from the Chomsky hierarchy—Parity, Modular Arithmetic (without brackets), and Modular Arithmetic (with brackets)—and report scaled accuracies in Table 4b. Mamba-3 solves Parity and Modular Arithmetic (without brackets), and nearly closes the accuracy gap on Modular Arithmetic (with brackets). In contrast, Mamba-3 without RoPE, Mamba-3 with standard RoPE (Su et al., 2023), and Mamba-2 fail to learn these tasks. We use the state tracking–enabled Gated DeltaNet variant of and observe that Mamba-3 is competitive—matching parity and approaching its performance on both modular-arithmetic tasks. Experimental settings are covered in Appendix E.

**Table 4: Left: Ablations on core modeling components of Mamba-3. Right: Formal language evaluation (scaled accuracy, %).**

| Model Variant (SISO) | ppl ↓ |
| :--- | :---: |
| Mamba-3 $-$ bias $-$ trap | 16.68 |
| Mamba-3 $-$ bias | 16.49 |
| **Mamba-3** | **15.72** |
| Mamba-3 $+$ conv | 15.85 |

**(a) Component ablation (350M).**

| Model | Parity ↑ | Arith. w/o ↑ brackets | Arith. w/ ↑ brackets |
| :--- | :---: | :---: | :---: |
| **Mamba-3** | **100.00** | **98.51** | **87.75** |
| Mamba-3 (w/o RoPE) | 2.27 | 1.49 | 0.72 |
| Mamba-3 (w/ Std. RoPE) | 1.56 | 20.70 | 2.62 |
| Mamba-2 | 0.90 | 47.81 | 0.88 |
| Gated DeltaNet [-1,1] | 100.00 | 99.25 | 93.50 |

**(b) Performance comparison on formal language tasks.**

---

## 5 CONCLUSION AND FUTURE WORK
We introduce Mamba-3, an SSM model with three axes of improvement rooted in SSM principles: (i) *improved quality*, via trapezoidal discretization; (ii) *new capabilities*, through complex SSMs that recover state-tracking; and (iii) *higher inference efficiency*, with a MIMO formulation that raises arithmetic intensity. Mamba-3 delivers strong language modeling results and establishes a new Pareto frontier on the performance-efficiency axes with respect to strong baseline models. A limitation remains in retrieval, where fixed-state architectures lags attention-based models. We see hybrid Mamba-3 architectures that integrate retrieval mechanisms as a promising path, alongside broader application of our design principles to linear-time sequence models.

---

### B TRAPEZOIDAL DISCRETIZATION
**Proposition 5 (Variation of Constants (Tenenbaum & Pollard, 1985)).** *Consider the linear SSM*
$$\mathbf{\dot{h}}(t) = A(t) \mathbf{h}(t) + \mathbf{B}(t) x(t),$$
*where $\mathbf{h}(t) \in \mathbb{R}^N, A(t) \in \mathbb{R}$ is a scalar decay, and $\mathbf{B}(t)x(t) \in \mathbb{R}^N$. For $\Delta_t$ discretized time grid $\tau_t = \tau_{t-1} + \Delta_t$, the hidden state satisfies*
$$\mathbf{h}_t \approx e^{\Delta_t A_t} \mathbf{h}_{t-1} + \int_{\tau_{t-1}}^{\tau_t} e^{(\tau_t - \tau) A_t} \mathbf{B}(\tau) x(\tau) d\tau. \tag{10}$$

---

### G ARCHITECTURE ABLATIONS
**Table 7: Ablations on $B, C$ bias initialization (left) and presence (right) for Mamba-3.**

| Bias Init. | Trainable | ppl ↓ |
| :--- | :---: | :---: |
| **1.0** | **✓** | **15.72** |
| 0.0 | ✓ | 16.57 |
| 1.0 | × | 15.80 |
| $\mathcal{U}(0, 1)$ | ✓ | 15.76 |
| $\mathcal{U}(-1, 1)$ | ✓ | 16.07 |

**(a) Effect of parameterization of the $B$ and $C$ bias on model performance.**

| $B$ Bias | $C$ Bias | ppl ↓ |
| :---: | :---: | :---: |
| × | × | 16.52 |
| ✓ | × | 16.68 |
| × | ✓ | 15.98 |
| **✓** | **✓** | **15.69** |

**(b) Applying a bias to both $B$ and $C$ leads to the best performance.**

---

### H.2 EXTENDED PREFILL AND PREFILL+DECODE LATENCY MEASUREMENTS

**Table 10: Prefill and Prefill+Decode latency across sequence lengths (in seconds).**

| Model | 512 tokens (Pref/P+D) | 1024 tokens (Pref/P+D) | 2048 tokens (Pref/P+D) | 4096 tokens (Pref/P+D) | 16384 tokens (Pref/P+D) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| vLLM (Llama-3.2-1B) | **0.26 / 4.45** | **0.52 / 9.60** | **1.08 / 20.37** | **2.08 / 58.64** | **1.52 / 122.06** |
| Gated DeltaNet | 0.48 / 4.52 | 0.95 / 9.04 | 1.90 / 18.07 | 3.79 / 36.14 | 1.91 / 71.66 |
| Mamba-2 | 0.48 / 4.62 | 0.96 / 9.24 | 1.91 / 18.48 | 3.81 / 36.94 | 1.92 / 57.90 |
| **Mamba-3 (SISO)** | 0.48 / **4.33** | 0.95 / **8.64** | 1.90 / **17.29** | 3.80 / **34.57** | 1.91 / **53.97** |
