# LUNA: LINEAR UNIVERSAL NEURAL ATTENTION WITH GENERALIZATION GUARANTEES

**Ashkan Shahbazi\* $^{1}$, Ping He\* $^{1}$, Ali Abbasi$^{1}$**
**Yikun Bai$^{1}$, Xinran Liu$^{1}$, Elaheh Akbari$^{1}$**
**Darian Salehi$^{3}$, Navid NaderiAlizadeh$^{4}$, Soheil Kolouri$^{1,2}$**

$^{1}$Department of Computer Science, Vanderbilt University, Nashville, TN, USA
$^{2}$Department of Electrical & Computer Engineering, Vanderbilt University, Nashville, TN, USA
$^{3}$Department of Computer Science, Duke University, Durham, NC, USA
$^{4}$Department of Biostatistics & Bioinformatics, Duke University, Durham, NC, USA

\*Equal contribution

December 10, 2025

---

### ABSTRACT

Scaling attention faces a critical bottleneck: the $O(n^2)$ quadratic computational cost of softmax attention, which limits its application in long-sequence domains. While linear attention mechanisms reduce this cost to $O(n)$, they typically rely on fixed random feature maps, such as random Fourier features or hand-crafted functions. This reliance on static, data-agnostic kernels creates a fundamental trade-off, forcing practitioners to sacrifice significant model accuracy for computational efficiency. We introduce LUNA, a kernelized linear attention mechanism that eliminates this trade-off, retaining linear cost while matching and surpassing the accuracy of quadratic attention. LUNA is built on the key insight that the kernel feature map itself should be learned rather than fixed a priori. By parameterizing the kernel, LUNA learns a feature basis tailored to the specific data and task, overcoming the expressive limitations of fixed-feature methods. LUNA implements this with a learnable feature map that induces a positive-definite kernel and admits a streaming form, yielding linear time and memory scaling in the sequence length. Empirical evaluations validate our approach across diverse settings. On the Long Range Arena (LRA), LUNA achieves state-of-the-art average accuracy among efficient Transformers under compute parity, using the same parameter count, training steps, and approximate FLOPs. LUNA also excels at post-hoc conversion: replacing softmax in fine-tuned BERT and ViT-B/16 checkpoints and briefly fine-tuning recovers most of the original performance, substantially outperforming fixed linearizations.

---

## 1 Introduction

Transformers [27] underpin state-of-the-art systems across language [7], vision [14], audio [8], multi-modal learning [16], and scientific domains [6, 11]. Their core mechanism, attention, models long-range token dependencies but incurs quadratic cost in the sequence length, which limits context scaling. This has motivated a large literature on linear attention, which reduces complexity via structured sparsity, low-rank compression, or kernel feature expansions [2, 3, 13, 17, 24, 30, 31]. However, these architectures typically commit to a fixed kernel or feature map, whether derived from the softmax exponential kernel or hand-crafted nonlinearities, and thus cannot adapt their inductive bias to the statistics of a given task or dataset. Our work follows this line but asks a different question: rather than fixing the kernel feature map *a priori* (random or engineered), can we learn the feature family directly from data—preserving linear complexity while tailoring the kernel to the task?

Learning such kernels while preserving the linear-attention regime is non-trivial. The streaming formulations that enable linear time and memory typically rely on rigid algebraic structure: the kernel must admit a non-negative feature representation and a stable decomposition into key- and query-side statistics. Naively parameterizing the feature map can break positive-definiteness, destroy the streaming factorization, or lead to brittle training dynamics. Instead, we seek a learnable kernel family that preserves the linear computation while exposing enough flexibility to adjust its inductive bias across architectures and domains.

We address this challenge by introducing LUNA, a linear-time attention mechanism that replaces hand-crafted random features with a fully learnable kernel feature family. Concretely, LUNA parameterizes (i) input projection matrices that capture distributional structure and (ii) a bank of channel functions with a token-wise envelope that together define the kernel nonlinearity. Attention is computed via the standard kernelized factorization in linear time, making LUNA a drop-in replacement for softmax or prior linear modules. We train all components end-to-end on each task using the task loss. This separation of representation (the learned nonlinearity) and mixing (the projections) yields expressive efficiency: the model retains linear-time complexity in the sequence length while adapting its feature basis to data rather than sampling from a fixed spectral measure.

Beyond training from scratch, LUNA also supports *post-hoc conversion* of quadratic models: given a fine-tuned checkpoint (e.g., BERT-base on GLUE or ViT-B/16 on ImageNet-1K), we replace each softmax attention layer with its LUNA linear counterpart and briefly fine-tune to recover accuracy, thereby avoiding reliance on the exponential softmax kernel while remaining compatible with existing architectures. Theoretically, for a single-layer instantiation, we derive a feature-level Rademacher complexity bound showing that, under standard norm and Lipschitz assumptions, the hypothesis class induced by our learned kernel family has complexity scaling as $\tilde{O}(1/\sqrt{n})$ with controlled dependence on that family.

**Contributions in this work are summarized as:**
* We introduce a positive-definite, kernelized attention with *learnable* feature maps that subsume fixed random-feature schemes (e.g., random Fourier features) while preserving linear-time/memory structure. This design enables the model to discover kernels auto-adaptive to each modality and task distribution.
* We provide a concise PD guarantee for our construction and an approximation-error decomposition (parametrization vs. sampling), with high-probability bounds under bounded and unbounded feature regimes. This design provides theoretical justification for the method's generalization capabilities.
* We show that softmax attention in *finetuned* models can be replaced by LUNA with brief task-specific finetuning, recovering most of the original performance for BERT on GLUE and ViT-B/16 on ImageNet-1K, and outperforming exponential-feature linearizations. This post-hoc conversion capability enables practical deployment of linear attention in existing production systems without re-training from scratch.
* Under matched compute, LUNA sets a new state-of-the-art average accuracy on the Long Range Arena.

## 2 Related Work

### 2.1 Quadratic Attention and Stochastic Structure
The original Transformer uses a quadratic-time softmax attention whose cost becomes prohibitive at long context lengths [27]. Within this quadratic regime, some studies exploit the stochastic structure of attention matrices. Standard softmax attention is row-stochastic but not column-normalized; doubly-stochastic variants enforce approximate bi-stochasticity via Sinkhorn-style normalization and related transport-inspired constraints, mainly to improve stability and interpretability while keeping $O(n^2)$ complexity [21, 22]. These methods act on the normalization of the attention matrix and are largely orthogonal to linearization: such constraints can be combined with either quadratic softmax attention or linearized kernels [23].

Beyond such normalization-based approaches, Teo and Nguyen view self-attention through kernel PCA and propose a robust quadratic variant (RPC-Attention), whereas we work in the linear-attention setting and learn the kernel feature family itself [26].

### 2.2 From Quadratic to Efficient Approximations
To remove the $O(n^2)$ bottleneck, efficient attention mechanisms replace dense all-to-all interactions with sparse, low-rank, or kernelized surrogates. Structured-sparsity methods restrict each token to a subset of neighbors using local windows, dilations, or block patterns (e.g., local attention, Longformer, and hashing/block-sparse designs) [1, 15, 18, 32]. Kernelized and low-rank approaches factorize similarity via feature maps or learned projections: Linear Transformers and Performer re-express softmax attention through (random or deterministic) kernel features, turning $n \times n$ accumulations into $n \times D$ computations [3, 13], while Linformer, Nyströmformer, and related methods learn or approximate a low-rank structure in the keys, values, or kernel matrix [30, 31]. Subsequent work proposes orthogonal or structured bases and stabilized feature maps [2, 4, 34], and Synthesizer departs from query-key matching by learning synthetic mixing weights [25]. “Post-hoc conversion” techniques replace softmax with an exponential feature map in pretrained models and finetune to recover performance, prioritizing compatibility with existing checkpoints over expressivity of the feature map itself [12, 33]. These studies define the linear-attention paradigm: approximate softmax attention while reducing complexity to linear or near-linear in token length.

![Figure 1](https://example.com/placeholder_figure1)
*Figure 1: (a) Softmax attention requires computing all pairwise interactions among tokens, which causes the cost to grow quadratically with the sequence length. (b) LUNA introduces a learnable kernel method for linearizing the attention mechanism, shifting the expensive step from the sequence length $n$ to the feature map size $mL$. (c) For a given set of tokens, LUNA applies $m$ linear projections $W_i \in \mathbb{R}^d$, producing $m$ scalar values. Each scalar is then passed through a shared MLP $\psi: \mathbb{R} \to \mathbb{R}^L$. By concatenating the $L$ outputs across all $m$ projections, we obtain the kernel feature map $\phi \in \mathbb{R}^{mL}$. The plots on the right show several learned components $\psi_i$ for the LRA-Text task...*

### 2.3 Positioning LUNA
LUNA operates on the *feature-map* side of linear attention. Instead of fixing the kernel feature map *a priori* (e.g., via random Fourier features or hand-crafted nonlinearities) or focusing primarily on compressing queries, keys, or values, LUNA learns a task-specific family of kernel features. Concretely, it combines learnable input projections with a shared bank of scalar channel functions, trained end-to-end on a single task. This design separates specialization over inputs from the representation power of the nonlinearity, allowing the feature basis to adapt to the data while preserving the drop-in efficiency of linear attention (linear complexity in the sequence length and feature dimension). In this sense, LUNA differs from fixed or purely random feature maps used in Performer/Cosformer/Skyformer-style designs and from exponential-feature conversions used for post-hoc linearization, providing a learned kernel representation tailored to the task within a standard linear-attention pipeline.

Importantly, this differs from post-hoc conversion approaches such as T2R and Hedgehog [12, 33], which replace softmax with a fixed exponential feature map and fine-tune to mimic the softmax kernel. LUNA does not attempt to match softmax; it learns a task-aligned kernel feature family directly from data. The exponential map is a special case within our parameterization, but it is not a constraint, enabling LUNA to retain linear complexity while searching a richer space of kernels better suited to the downstream task.

## 3 Learning Kernels for Linear Attention

We adopt the kernelized formulation of dot-product attention and its random-feature linearizations. This view makes explicit the sufficient-statistics factorization that underlies linear-time, linear-memory variants and recovers softmax attention as a specific positive-definite kernel induced by an exponential feature map. Building on this formulation, we introduce a parametric family of learnable feature maps that (i) define a valid positive-definite kernel and (ii) retain the linear compute pattern of kernelized attention layers.

### 3.1 Preliminaries: Attention as a Kernel Method
Let $Q, K, V \in \mathbb{R}^{n \times d}$ denote the query, key, and value matrices, respectively, with sequence length $n$ and latent dimension $d$. Scaled dot-product attention is given by
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V = \frac{AV}{A\mathbf{1}_n}, \quad A = \exp\left(\frac{QK^\top}{\sqrt{d}}\right), \tag{1}$$
where the softmax and the normalization by $\mathbf{1}_n \in \mathbb{R}^n$ act row-wise. This admits a kernel view with $k_{SM}(x, y) = \exp\left(\frac{x^\top y}{\sqrt{d}}\right), x = Q[i, :], y = K[j, :]$. Although $k_{SM}$ is not shift-invariant, it is linked to the Gaussian kernel $k_G(x, y) = \exp(-\frac{1}{2\sqrt{d}}\|x - y\|^2)$ via
$$k_{SM}(x, y) = e^{\|x\|^2/2\sqrt{d}} k_G(x, y) e^{\|y\|^2/2\sqrt{d}} \tag{2}$$
By Bochner’s theorem [19, 20], a continuous, shift-invariant, positive-definite kernel admits
$$k(x - y) = \int_{\mathbb{R}^d} e^{i\omega^\top(x-y)} d\mu(\omega) = \mathbb{E}_{\omega \sim \mu} [e^{i\omega^\top(x-y)}] = \mathbb{E}_{\omega \sim \mu, b \sim \text{Unif}(0,2\pi)} [\zeta_{\omega,b}(x) \zeta_{\omega,b}(y)], \tag{3}$$
with $\zeta_{\omega,b}(x) = \sqrt{2} \cos(\omega^\top x + b)$, where $b \sim \text{Unif}(0, 2\pi)$ is added as a variance-reduction trick to get cosine-shifted features [19]. Writing $k_\mu(x, y) := k(x - y)$ for the kernel induced by the spectral measure $\mu$, we approximate the expectation in (3) with $m$ Monte Carlo samples $(\omega_i, b_i) \overset{i.i.d.}{\sim} \mu \times \text{Unif}(0, 2\pi)$ obtaining the finite feature map:
$$\phi_m(x) = \sqrt{\frac{2}{m}} [\cos(\omega_1^\top x + b_1), \dots, \cos(\omega_m^\top x + b_m)]^\top, \tag{4}$$
yielding the empirical kernel estimator
$$\hat{k}_\mu^{(m)}(x, y) := \phi_m(x)^\top \phi_m(y) = \frac{2}{m} \sum_{i=1}^m \cos(\omega_i^\top x + b_i) \cos(\omega_i^\top y + b_i), \tag{5}$$
which converges to $k_\mu(x, y)$ as $m \to \infty$. Specializing to the Gaussian spectral measure $\mu = \mathcal{N}(0, \frac{1}{\sqrt{d}} I_d)$ yields the standard random Fourier features (RFF) approximation to the Gaussian kernel, $k_G(x, y) \approx \hat{k}_\mu^{(m)}(x, y) = \phi_m(x)^\top \phi_m(y)$.

**Performer features.** To linearize the exponential dot-product kernel $k_{exp}(x, y) = \exp(x^\top y / \sqrt{d})$, Performer [3] uses the fact that this kernel admits a Gaussian-moment factorization:
$$k_{exp}(x, y) = \exp\left(\frac{x^\top y}{\sqrt{d}}\right) = \mathbb{E}_{\omega \sim \mathcal{N}(0, I_d)} [e^{\frac{2\omega^\top x - \|x\|^2}{2\sqrt{d}}} e^{\frac{2\omega^\top y - \|y\|^2}{2\sqrt{d}}}]. \tag{7}$$
This representation expresses the kernel as an expectation of two separated functions of $x$ and $y$, enabling a random-feature approximation. Drawing $\omega_1, \dots, \omega_m \overset{i.i.d.}{\sim} \mathcal{N}(0, \frac{1}{\sqrt{d}} I_d)$ produces the Performer feature map
$$\phi_m^P(x) = \frac{1}{\sqrt{m}} [e^{\frac{-\|x\|^2 + 2\omega_1^\top x}{2\sqrt{d}}}, \dots, e^{\frac{-\|x\|^2 + 2\omega_m^\top x}{2\sqrt{d}}}]^\top, \tag{8}$$
for which $\phi_m^P(x)^\top \phi_m^P(y) \approx k_{exp}(x, y)$.

**Linear attention.** With a (possibly learned) feature map $\phi : \mathbb{R}^d \to \mathbb{R}^D$ applied row-wise on the query and key matrices, we have
$$\text{Attn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1}_n)}, \tag{9}$$
reducing complexity from $O(n^2 d)$ to $O(nD^2)$. The denominator is computed element-wise per row.

### 3.2 Our Method: Fully Learnable Kernels
The constructions above show that many efficient Transformers admit a kernel of the form $k(x, y) = \mathbb{E}_{\omega \sim \mu} [\zeta_\omega(x) \zeta_\omega(y)]$, where $\zeta_\omega$ is a *fixed*, scalar feature determined by the choice of spectral measure $\mu$. In all such cases the kernel class is hard-coded by $\zeta_\omega$. We generalize this template by replacing the scalar feature $\zeta_\omega(x)$ with a vector-valued, *learnable* feature family $\phi_\omega(x)$. Formally, we define
$$k(x, y) := \mathbb{E}_{\omega \sim \mathcal{N}(0, I_d)} \langle \phi_\omega(x), \phi_\omega(y) \rangle_{\mathcal{H}} \approx \frac{1}{m} \sum_{i=1}^m \langle \phi(x; \omega_i), \phi(y; \omega_i) \rangle_{\mathcal{H}}, \tag{10}$$
where each $\phi_\omega : \mathbb{R}^d \to \mathcal{H}$ maps an input to a feature in a Hilbert space $\mathcal{H} \cong \mathbb{R}^{mL}$ endowed with the Euclidean inner product. This preserves the positive-definite kernel structure and the linear-time, streaming form of kernel attention, while allowing the feature family and the kernel itself to adapt to the data.

**Proposition 1.** *The construction in (10) yields a positive-definite kernel. Conversely, by Mercer’s theorem, any positive-definite kernel admits such a representation for an appropriate $\phi(\cdot; \omega)$ into an RKHS $\mathcal{H}$. See Appendix 8.*

We replace fixed RFF components with learnable projections and channels. Let $W \in \mathbb{R}^{m \times d}$ with rows $\{w_i^\top\}_{i=1}^m$, channel functions $\{\psi_\ell : \mathbb{R} \to \mathbb{R}\}_{\ell=1}^L$ (each instantiated as a small MLP on the scalar projection $u = w_i^\top x$), and a tokenwise envelope $h : \mathbb{R}^d \to \mathbb{R}$. Our feature map is
$$\phi(x; W, \psi, h) = \frac{h(x)}{\sqrt{m}} [\psi_\ell(w_i^\top x)]_{i=1,\dots,m, \ell=1,\dots,L} \in \mathbb{R}^{mL}. \tag{11}$$
This template strictly generalizes RFF and Performer features: both are recovered by fixing $h$ and the $\psi_\ell$ by hand, whereas in our case $W, h$, and the channel functions $\{\psi_\ell\}$ are learned from data.

**Remark 1 (Neural Approximation of Multiplicatively Decomposable Kernels).** *Let $k : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ be a positive-definite kernel, and suppose it admits the multiplicative form $k(x, y) = h(x) k'(x - y) h(y)$... there exists a neural feature map $\phi(x; \omega) \in \mathbb{R}^L$, of the form described in (11), such that with high probability... $|k(x, y) - k_{NN}(x, y; W)| \leq \varepsilon$.*

---

## 4 Runtime Analysis
Let $n$ be the sequence length, $d$ the key/query width, $d_v$ the value width, and $D = mL$ the feature dimension of $\phi(x; W, \psi, h) \in \mathbb{R}^D$. Using the kernelized form of Eq. (9), the per-head cost decomposes into (i) evaluating $\phi(\cdot)$ for all tokens and (ii) forming two key-side sufficient statistics followed by a single query-side application:
$$S_{KV} = \phi(K)^\top V \in \mathbb{R}^{D \times d_v}, \quad S_{K1} = \phi(K)^\top \mathbf{1}_n \in \mathbb{R}^D.$$
The resulting complexity is $T_{total} = O(n(md + D c_\psi)) + O(n D d_v)$, where $c_\psi$ denotes the per-channel MLP cost in $\psi$. In the common regime $d_v = \Theta(D)$, this simplifies to
$$T_{total} = O(nD^2) = O(n(mL)^2). \tag{12}$$
The computation is *linear* in $n$ because the $n \times n$ attention matrix is never formed. Figure 2 compares the resulting compute flow and empirical scaling against representative linear-attention baselines.

---

## 5 Experiments

### 5.1 Long Range Arena
Table 1 summarizes test accuracy by task and averaged across LRA. LUNA matches or surpasses prior efficient Transformers on four of five tasks.

**Table 1: Results on the LRA benchmark.** Best numbers per column are in bold.
| Model | Text | ListOps | Retrieval | Pathfinder | Image | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformer | 61.55 | 38.71 | 80.93 | 70.39 | 39.14 | 58.14 |
| Performer | 65.40 | 18.01 | 53.82 | **77.05** | 42.77 | 51.41 |
| Skyformer | 64.70 | 38.69 | 82.06 | 70.73 | 40.77 | 59.39 |
| LOTFormer | 71.1 | 38.5 | 80.9 | 69.9 | 54.1 | 62.9 |
| **LUNA (Ours)** | **73.41** | **38.94** | 81.02 | 69.52 | **64.32** | **65.44** |

### 5.2 Post-hoc Conversion of Finetuned Quadratic Transformers
**Table 2: Post-hoc conversion of BERT-base on GLUE.**
| Method | CoLA | SST2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE | (%) Recover |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BERT-FT | 58.8 | 93.2 | 90.2 | 88.8 | 91.0 | 84.7 | 91.3 | 68.2 | 100.0 |
| Hedgehog | 59.2 | 92.6 | 90.1 | 87.4 | 91.0 | 82.6 | 89.6 | 69.3 | 99.3 |
| **LUNA** | 58.8 | 93.4 | 90.1 | 88.5 | 90.7 | 83.5 | 90.6 | 68.8 | **99.5** |

**Table 3: Post-hoc conversion of ViT-B/16 on ImageNet-1K validation accuracy (%).**
| Top-1 | ViT-B/16 | T2R-HH | Hedgehog | LUNA |
| :--- | :--- | :--- | :--- | :--- |
| Accuracy (%) | 80.3 | 77.0 | 79.5 | **80.5** |

---

## 6 Conclusion
We introduced LUNA, a linear attention mechanism that replaces fixed, data-agnostic feature maps with a learned kernel feature map while preserving a positive-definite kernel and an associative, streaming formulation. This yields linear time and memory complexity and substantially narrows the accuracy gap to quadratic softmax attention.

---

### Listing 1: Our MLP-based feature map and linear attention.

```python
class TaskSpecificProjections(nn.Module):
    # shared W_i^T x + b_i over tokens/heads
    def __init__(self, d: int, M: int):
        super().__init__()
        self.M = int(M)
        self.W = nn.Parameter(torch.randn(self.M, d) / math.sqrt(d))
        self.b = nn.Parameter(torch.zeros(self.M))

    def forward(self, x): # (B,H,N,d) -> (B,H,N,M)
        return torch.einsum("md,bhnd->bhnm", self.W, x) + self.b

class ScalarMLP(nn.Module):
    # maps scalar u to L channels in one pass (L=1 gives OneMLP)
    def __init__(self, L=1, hidden=64, act="relu", nonneg=True):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, int(L))
        self.act = _act(act)
        self.nonneg = bool(nonneg)

    def forward(self, u): # (T,1) -> (T,L)
        y = self.fc2(self.act(self.fc1(u)))
        if self.nonneg:
            y = F.relu(y)
        return y

class MLPLearnableFeatureMap(nn.Module):
    # phi(x) = (1/sqrt(M)) * [ f_l(w_i^T x) ]_{i,l}
    def __init__(self, M, L, hidden=64, act="relu", nonneg=True,
                 chunk=1_000_000, shared_channels=False,
                 ch_rms=False, ch_rms_target=0.1):
        super().__init__()
        self.M, self.L = int(M), int(L)
        self.scale = 1.0 / math.sqrt(self.M)
        self.chunk = int(chunk)
        self.shared = bool(shared_channels)
        self.ch_rms = bool(ch_rms)
        self.ch_rms_target = float(ch_rms_target)

        if self.shared:
            self.shared_mlp = ScalarMLP(self.L, hidden, act, nonneg)
            self.mlps = None
        else:
            self.mlps = nn.ModuleList(
                [ScalarMLP(1, hidden, act, nonneg) for _ in range(self.L)]
            )

    def forward(self, proj): # (B,H,N,M) -> (B,H,N,M*L)
        B, H, N, M = proj.shape
        u = proj.reshape(-1, 1).float() # (T,1)
        if self.shared:
            y = _chunked(self.shared_mlp, u, self.chunk) # (T,L)
        else:
            ys = [_chunked(mlp, u, self.chunk) for mlp in self.mlps]
            y = torch.cat(ys, dim=1) # (T,L)
        y = y.view(B, H, N, M, self.L)
        if self.ch_rms:
            eps = 1e-6
            rms = torch.sqrt(y.pow(2).mean((0, 1, 2, 3)) + eps) # (L,)
            s = (self.ch_rms_target / (rms + eps)).clamp(max=1.0)
            y = y * s.view(1, 1, 1, 1, self.L)
        return (y * self.scale).reshape(B, H, N, M * self.L)
```
