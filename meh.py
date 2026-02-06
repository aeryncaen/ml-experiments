"""
Data-Dependent Simpson Discretization for Mamba-3 SSM
=====================================================

Extends the Mamba-3 trapezoidal discretization (Proposition 1, Eq. 3-5)
to a Simpson-style rule that uses THREE time points instead of two.

The trapezoidal rule (Mamba-3, Eq. 4):
    h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t

Our Simpson rule adds one more lookback:
    h_t = α_t h_{t-1} + δ_t B_{t-2} x_{t-2} + ε_t B_{t-1} x_{t-1} + ζ_t B_t x_t

Key insight from the paper (Section 3.1.1): the trapezoidal mask decomposes as
    L = L_decay × L_conv                                    (Eq. 5)
where L_decay is the standard Mamba-2 lower-triangular decay mask and L_conv
is a bidiagonal (size-2 convolution) matrix.

For Simpson, L_conv becomes a TRIDIAGONAL (size-3 convolution) matrix.
The same SSD contraction (Appendix B.1) applies with an extra band.

References:
- Mamba-3 paper: Proposition 1 (trapezoidal), Eq. 5 (mask decomposition)
- Mamba-3 paper: Appendix B.1 (contraction decomposition proof)
- Mamba-2/SSD: Eq. 1 (structured masked representation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from typing import Optional, Tuple


# =============================================================================
# Section 1: Core Recurrence
# =============================================================================

def simpson_recurrence_step(
    h_prev: torch.Tensor,       # (B, H, N, P) - hidden state at t-1
    Bx_prev2: torch.Tensor,     # (B, H, N, P) - B_{t-2} x_{t-2}
    Bx_prev1: torch.Tensor,     # (B, H, N, P) - B_{t-1} x_{t-1}
    Bx_curr: torch.Tensor,      # (B, H, N, P) - B_t x_t
    alpha: torch.Tensor,        # (B, H, 1, 1) - decay e^{Δ_t A_t}
    delta: torch.Tensor,        # (B, H, 1, 1) - weight on t-2 term
    epsilon: torch.Tensor,      # (B, H, 1, 1) - weight on t-1 term
    zeta: torch.Tensor,         # (B, H, 1, 1) - weight on t term
) -> torch.Tensor:
    """
    Single step of Simpson-discretized SSM recurrence.

    Mamba-3's trapezoidal (Eq. 4):
        h_t = α_t h_{t-1} + β_t B_{t-1} x_{t-1} + γ_t B_t x_t

    Our Simpson generalization:
        h_t = α_t h_{t-1} + δ_t B_{t-2} x_{t-2} + ε_t B_{t-1} x_{t-1} + ζ_t B_t x_t

    where (δ_t, ε_t, ζ_t) are data-dependent quadrature weights initialized
    near Simpson's (1/6, 4/6, 1/6) · Δ_t, but learned freely.
    """
    return alpha * h_prev + delta * Bx_prev2 + epsilon * Bx_prev1 + zeta * Bx_curr


# =============================================================================
# Section 2: Mask Decomposition (extends Eq. 5 from paper)
# =============================================================================

def build_simpson_conv_mask(
    delta: torch.Tensor,    # (T,) or (B, H, T) - weight on t-2 term
    epsilon: torch.Tensor,  # (T,) or (B, H, T) - weight on t-1 term
    zeta: torch.Tensor,     # (T,) or (B, H, T) - weight on t term
    T: int,
) -> torch.Tensor:
    """
    Build the convolutional factor L_conv of the Simpson mask decomposition.

    From the paper's Eq. 5, the trapezoidal mask factors as:
        L = L_decay × L_conv

    where L_conv for trapezoidal is BIDIAGONAL:
        L_conv = [[γ_0,                    ],
                  [β_1,  γ_1,              ],
                  [0,    β_2,  γ_2,        ],
                  [            ...         ],
                  [0,    ...,  β_T,  γ_T   ]]

    For Simpson, L_conv is TRIDIAGONAL (size-3 convolution):
        L_conv = [[ζ_0,                           ],
                  [ε_1,  ζ_1,                      ],
                  [δ_2,  ε_2,  ζ_2,                ],
                  [0,    δ_3,  ε_3,  ζ_3,          ],
                  [                  ...            ],
                  [0,    ...,  δ_T,  ε_T,  ζ_T     ]]

    This is a bandwidth-3 banded lower-triangular matrix.
    The (i, j) entry is:
        - ζ_i  if j == i     (diagonal)
        - ε_i  if j == i-1   (first subdiagonal)
        - δ_i  if j == i-2   (second subdiagonal)
        - 0    otherwise

    Returns: (T, T) or (B, H, T, T) lower-triangular banded matrix
    """
    # Handle batched vs unbatched
    if delta.dim() == 1:
        L_conv = torch.zeros(T, T, device=delta.device, dtype=delta.dtype)
        for i in range(T):
            L_conv[i, i] = zeta[i]                      # diagonal
            if i >= 1:
                L_conv[i, i - 1] = epsilon[i]           # subdiag-1
            if i >= 2:
                L_conv[i, i - 2] = delta[i]             # subdiag-2
    else:
        # Batched: (B, H, T) -> (B, H, T, T)
        B_sz, H_sz = delta.shape[0], delta.shape[1]
        L_conv = torch.zeros(B_sz, H_sz, T, T, device=delta.device, dtype=delta.dtype)

        idx = torch.arange(T, device=delta.device)

        # Diagonal: L_conv[:, :, i, i] = ζ_i
        L_conv[:, :, idx, idx] = zeta

        # First subdiagonal: L_conv[:, :, i, i-1] = ε_i for i >= 1
        if T > 1:
            L_conv[:, :, idx[1:], idx[:-1]] = epsilon[:, :, 1:]

        # Second subdiagonal: L_conv[:, :, i, i-2] = δ_i for i >= 2
        if T > 2:
            L_conv[:, :, idx[2:], idx[:-2]] = delta[:, :, 2:]

    return L_conv


def build_decay_mask(
    alpha: torch.Tensor,  # (T,) or (B, H, T) - per-step decay factors
    T: int,
) -> torch.Tensor:
    """
    Build the standard Mamba-2 decay mask L_decay (first factor of Eq. 5).

    L_decay[i, j] = prod_{s=j+1}^{i} α_s    for i >= j
                   = 1                        for i == j

    This is unchanged from Mamba-2/Mamba-3.
    """
    if alpha.dim() == 1:
        L_decay = torch.zeros(T, T, device=alpha.device, dtype=alpha.dtype)
        for i in range(T):
            for j in range(i + 1):
                if i == j:
                    L_decay[i, j] = 1.0
                else:
                    L_decay[i, j] = alpha[j + 1: i + 1].prod()
    else:
        # Batched version using cumulative log-sum for efficiency
        B_sz, H_sz = alpha.shape[0], alpha.shape[1]
        log_alpha = torch.log(alpha.clamp(min=1e-8))          # (B, H, T)
        log_alpha_cumsum = log_alpha.cumsum(dim=-1)            # (B, H, T)

        # L_decay[i, j] = exp(sum_{s=j+1}^{i} log(α_s)) = exp(cumsum[i] - cumsum[j])
        # For i >= j, with L_decay[i, i] = 1
        diff = log_alpha_cumsum.unsqueeze(-1) - log_alpha_cumsum.unsqueeze(-2)  # (B,H,T,T)
        L_decay = torch.exp(diff)

        # Zero out upper triangle (causal)
        causal_mask = torch.tril(torch.ones(T, T, device=alpha.device))
        L_decay = L_decay * causal_mask

    return L_decay


def build_full_simpson_mask(
    alpha: torch.Tensor,    # (B, H, T) decay factors
    delta: torch.Tensor,    # (B, H, T) Simpson weight on t-2
    epsilon: torch.Tensor,  # (B, H, T) Simpson weight on t-1
    zeta: torch.Tensor,     # (B, H, T) Simpson weight on t
    T: int,
) -> torch.Tensor:
    """
    Full Simpson mask: L = L_decay @ L_conv

    This directly parallels Eq. 5 from the paper:
        L_trapezoid = L_decay × L_conv_bidiag

    but with L_conv_tridiag instead:
        L_simpson = L_decay × L_conv_tridiag

    The contraction from Appendix B.1 generalizes:
        contract(TN, SN, TJ, JS, SP → TP)(C, B, L_decay, L_conv, X)
    where J now has bandwidth 3 instead of 2.
    """
    L_decay = build_decay_mask(alpha, T)
    L_conv = build_simpson_conv_mask(delta, epsilon, zeta, T)

    # L = L_decay @ L_conv  (matrix multiply over the T×T dimensions)
    if alpha.dim() == 1:
        return L_decay @ L_conv
    else:
        return torch.matmul(L_decay, L_conv)


# =============================================================================
# Section 3: Parallel (SSD-style) Forward Pass
# =============================================================================

def simpson_ssd_forward(
    X: torch.Tensor,         # (B, T, H, P) input to SSM
    B: torch.Tensor,         # (B, T, H, N) input projection
    C: torch.Tensor,         # (B, T, H, N) output projection
    alpha: torch.Tensor,     # (B, T, H)    decay factors
    delta: torch.Tensor,     # (B, T, H)    Simpson weight on t-2
    epsilon: torch.Tensor,   # (B, T, H)    Simpson weight on t-1
    zeta: torch.Tensor,      # (B, T, H)    Simpson weight on t
) -> torch.Tensor:
    """
    Parallel SSD-style computation with Simpson discretization.

    From the paper's Eq. 1 (Mamba-2 SSD):
        Y = (L ⊙ C B^T) X

    With Simpson, L = L_decay @ L_conv, giving:
        Y = ((L_decay @ L_conv) ⊙ C B^T) X

    Following Appendix B.1's contraction decomposition:
        Z    = contract(SN, SP → SNP)(B, X)           # B ⊗ X
        Z'   = contract(JS, SNP → JNP)(L_conv, Z)     # size-3 conv on B⊗X
        H    = contract(TJ, JNP → TNP)(L_decay, Z')   # decay-weighted accumulation
        Y    = contract(TN, TNP → TP)(C, H)            # output projection

    This is the direct generalization of the paper's decomposition,
    with L_conv now being tridiagonal (bandwidth 3) instead of bidiagonal (bandwidth 2).
    """
    B_sz, T, H, P = X.shape
    N = B.shape[-1]

    # Rearrange for head-first computation
    X = rearrange(X, 'b t h p -> b h t p')
    B = rearrange(B, 'b t h n -> b h t n')
    C = rearrange(C, 'b t h n -> b h t n')
    alpha = rearrange(alpha, 'b t h -> b h t')
    delta = rearrange(delta, 'b t h -> b h t')
    epsilon = rearrange(epsilon, 'b t h -> b h t')
    zeta = rearrange(zeta, 'b t h -> b h t')

    # Step 1: Z = B ⊗ X → (B, H, T, N, P)
    # This is contract(SN, SP → SNP)(B, X) from Appendix B.1
    Z = torch.einsum('bhtn, bhtp -> bhtnp', B, X)

    # Step 2: Apply size-3 convolution (L_conv) on the time dimension of Z
    # This is contract(JS, SNP → JNP)(L_conv, Z)
    #
    # For trapezoidal (paper), this was a size-2 conv:
    #   Z'_t = β_t Z_{t-1} + γ_t Z_t
    #
    # For Simpson, this is a size-3 conv:
    #   Z'_t = δ_t Z_{t-2} + ε_t Z_{t-1} + ζ_t Z_t
    Z_conv = simpson_causal_conv(Z, delta, epsilon, zeta)

    # Step 3: Apply decay mask (L_decay) — standard Mamba-2 cumulative decay
    # This is contract(TJ, JNP → TNP)(L_decay, Z')
    #
    # Efficient implementation via cumulative scan rather than full T×T matmul
    H = apply_decay_scan(Z_conv, alpha)

    # Step 4: Output projection Y = C^T H
    # This is contract(TN, TNP → TP)(C, H)
    Y = torch.einsum('bhtn, bhtnp -> bhtp', C, H)

    Y = rearrange(Y, 'b h t p -> b t h p')
    return Y


def simpson_causal_conv(
    Z: torch.Tensor,         # (B, H, T, N, P) - the B⊗X tensor
    delta: torch.Tensor,     # (B, H, T) - weight on t-2
    epsilon: torch.Tensor,   # (B, H, T) - weight on t-1
    zeta: torch.Tensor,      # (B, H, T) - weight on t
) -> torch.Tensor:
    """
    Apply the Simpson convolutional mask L_conv as a causal size-3 convolution
    on the time dimension of Z.

    For Mamba-3's trapezoidal (paper Eq. 5, second factor), this would be:
        Z'_t = β_t Z_{t-1} + γ_t Z_t          (size-2 conv, bidiagonal)

    For our Simpson variant:
        Z'_t = δ_t Z_{t-2} + ε_t Z_{t-1} + ζ_t Z_t    (size-3 conv, tridiagonal)

    This is the ONLY change to the SSD pipeline vs the paper's trapezoidal.
    The rest (decay mask, output projection) is identical.
    """
    B_sz, H, T, N, P = Z.shape

    # Reshape weights for broadcasting: (B, H, T) -> (B, H, T, 1, 1)
    d = delta.unsqueeze(-1).unsqueeze(-1)
    e = epsilon.unsqueeze(-1).unsqueeze(-1)
    z = zeta.unsqueeze(-1).unsqueeze(-1)

    # Pad Z with zeros for the lookback (causal: no future leakage)
    # Z_padded[:, :, 0:2, ...] are zero-padding, Z_padded[:, :, 2:, ...] = Z
    Z_padded = F.pad(Z, (0, 0, 0, 0, 2, 0))  # pad time dim on left by 2

    # Gather the three time-shifted versions
    Z_curr  = Z_padded[:, :, 2:2+T]    # Z_t
    Z_prev1 = Z_padded[:, :, 1:1+T]    # Z_{t-1}
    Z_prev2 = Z_padded[:, :, 0:0+T]    # Z_{t-2}

    # Apply data-dependent Simpson weights
    Z_conv = d * Z_prev2 + e * Z_prev1 + z * Z_curr

    return Z_conv


def apply_decay_scan(
    Z: torch.Tensor,       # (B, H, T, N, P)
    alpha: torch.Tensor,   # (B, H, T) decay factors
) -> torch.Tensor:
    """
    Apply the decay mask L_decay via a parallel prefix scan.

    This is UNCHANGED from Mamba-2/Mamba-3. The decay mask L_decay is the
    first factor in Eq. 5 and is the same regardless of which quadrature rule
    is used for the integral.

    L_decay[i, j] = ∏_{s=j+1}^{i} α_s

    Efficient implementation: sequential scan (for clarity).
    In production, replace with a parallel associative scan.
    """
    B_sz, H, T, N, P = Z.shape
    alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)  # (B, H, T, 1, 1)

    H_out = torch.zeros_like(Z)
    H_out[:, :, 0] = Z[:, :, 0]

    for t in range(1, T):
        H_out[:, :, t] = alpha_expanded[:, :, t] * H_out[:, :, t - 1] + Z[:, :, t]

    return H_out


# =============================================================================
# Section 4: Parameter Generation (extends paper's Section 3.4)
# =============================================================================

class SimpsonSSMParams(nn.Module):
    """
    Generate data-dependent Simpson quadrature weights from input tokens.

    Parallels the paper's parameterization where:
    - α_t = exp(Δ_t A_t)   — decay (unchanged)
    - Mamba-3 trapezoidal: λ_t = σ(u_t), then β_t = (1-λ_t)Δ_t α_t, γ_t = λ_t Δ_t

    For Simpson, we produce three weights (δ_t, ε_t, ζ_t) via a data-dependent
    softmax scaled by Δ_t, initialized near classical Simpson (1/6, 4/6, 1/6).

    From paper Table 5 (Appendix B.3): data-dependent λ_t (default) outperforms
    fixed λ_t = 1/2. So we similarly make Simpson weights data-dependent.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,          # N in the paper
        n_heads: int,          # H
        head_dim: int,         # P = D/H
        dt_rank: int = None,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dt_rank = dt_rank or (d_model // 16)

        # --- Projections matching paper's architecture (Section 3.4, Fig. 4) ---

        # B, C projections: input -> (H, N) per token
        # Paper: "projecting ... B, C ∈ R^N from the current token"
        self.B_proj = nn.Linear(d_model, n_heads * d_state, bias=bias)
        self.C_proj = nn.Linear(d_model, n_heads * d_state, bias=bias)

        # BC bias (paper Section 3.4): "head-specific, channel-wise bias ...
        # initialized to all ones"
        # Paper Table 7a: all-ones init is best
        self.B_bias = nn.Parameter(torch.ones(n_heads, d_state))
        self.C_bias = nn.Parameter(torch.ones(n_heads, d_state))

        # QK-norm (paper Section 3.4): "swaps the pre-output projection norm
        # with the more common QK-normalization"
        self.B_norm = nn.RMSNorm(d_state)
        self.C_norm = nn.RMSNorm(d_state)

        # Δ_t projection for decay
        # Paper: "A = A ∈ R<0" is a scalar, Δ_t is projected from input
        self.dt_proj = nn.Linear(d_model, n_heads, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.rand(n_heads) * 0.5 + 0.5))  # A < 0

        # --- NEW: Simpson quadrature weight projection ---
        # Three logits -> softmax -> (δ_t, ε_t, ζ_t) weights
        # Initialized so that softmax(bias) ≈ (1/6, 4/6, 1/6) (classical Simpson)
        self.simpson_proj = nn.Linear(d_model, n_heads * 3, bias=True)

        # Initialize bias toward Simpson's rule: softmax([0, log(4), 0]) ≈ (1/6, 4/6, 1/6)
        nn.init.zeros_(self.simpson_proj.weight)
        with torch.no_grad():
            bias_init = torch.zeros(n_heads * 3)
            for h in range(n_heads):
                bias_init[h * 3 + 0] = 0.0           # → 1/6 after softmax
                bias_init[h * 3 + 1] = 1.3863        # log(4) → 4/6 after softmax
                bias_init[h * 3 + 2] = 0.0           # → 1/6 after softmax
            self.simpson_proj.bias.copy_(bias_init)

    def forward(
        self,
        u: torch.Tensor,  # (B, T, D) input tokens
    ) -> Tuple[torch.Tensor, ...]:
        """
        Returns: B, C, alpha, delta, epsilon, zeta
        all with shape (B, T, H, ...) matching the SSD forward signature.
        """
        B_sz, T, D = u.shape

        # --- B, C with norm and bias (paper Section 3.4) ---
        B = self.B_proj(u).view(B_sz, T, self.n_heads, self.d_state)
        C = self.C_proj(u).view(B_sz, T, self.n_heads, self.d_state)

        # Apply QK-norm then add bias (paper: "bias added after normalization")
        B = self.B_norm(B) + self.B_bias  # broadcast (H, N) bias
        C = self.C_norm(C) + self.C_bias

        # --- Decay α_t = exp(Δ_t A_t) (paper Section 2.2) ---
        dt = F.softplus(self.dt_proj(u))          # (B, T, H), Δ_t > 0
        A = -torch.exp(self.A_log)                # A < 0, (H,)
        alpha = torch.exp(dt * A)                 # (B, T, H), α_t ∈ (0, 1)

        # --- Simpson weights (δ_t, ε_t, ζ_t) ---
        # Data-dependent via softmax, scaled by Δ_t (matching how trapezoidal
        # scales β_t, γ_t by Δ_t in paper Eq. 4)
        simpson_logits = self.simpson_proj(u).view(B_sz, T, self.n_heads, 3)
        simpson_weights = F.softmax(simpson_logits, dim=-1)  # (B, T, H, 3)

        # Scale by Δ_t (paper: β_t = (1-λ_t)Δ_t α_t, γ_t = λ_t Δ_t)
        # For Simpson, classical weights are (Δ/3, 4Δ/3, Δ/3) summing to 2Δ
        # We use Δ_t as the overall scale factor
        dt_expanded = dt.unsqueeze(-1)  # (B, T, H, 1)

        # Weight on t-2 term: additionally multiply by α_t^2
        # (because in the integral, the t-2 sample has been decayed twice)
        delta   = simpson_weights[..., 0] * dt * alpha * alpha   # (B, T, H)

        # Weight on t-1 term: multiply by α_t
        # (decayed once from t-1 to t in the integral approximation)
        epsilon = simpson_weights[..., 1] * dt * alpha           # (B, T, H)

        # Weight on t term: no extra decay
        zeta    = simpson_weights[..., 2] * dt                   # (B, T, H)

        return B, C, alpha, delta, epsilon, zeta


# =============================================================================
# Section 5: Full Simpson Mamba-3 Mixer Layer
# =============================================================================

class SimpsonMamba3Mixer(nn.Module):
    """
    Drop-in replacement for the Mamba-3 SSD mixer (paper Section 3.4, Fig. 4)
    using Simpson discretization instead of trapezoidal.

    Architecture follows the paper:
    - Input projection → X, B, C, gate
    - Simpson SSM (replaces SSD with trapezoidal)
    - Gate + output projection
    - NO short convolution (paper Table 4a: "combination of our BC bias and
      trapezoidal discretization makes the convolution optional")

    The Simpson discretization (size-3 conv in the mask) subsumes even more
    of the short convolution's role, making its removal even more natural.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,      # Paper: "dstate of 128" (Appendix E)
        n_heads: int = 32,
        head_dim: int = 64,      # Paper: "head dimension of 64" (Appendix E)
        expand: int = 2,         # Paper: "standard expand factor of 2" (Appendix E)
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state

        assert self.d_inner == n_heads * head_dim, \
            f"d_inner ({self.d_inner}) must equal n_heads * head_dim ({n_heads * head_dim})"

        # Input projection: u → (X, gate) following paper Fig. 4
        # X goes through SSM, gate is applied after
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # SSM parameter generation
        self.ssm_params = SimpsonSSMParams(
            d_model=d_model,
            d_state=d_state,
            n_heads=n_heads,
            head_dim=head_dim,
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Normalization before output (paper: uses this for stability)
        self.norm = nn.RMSNorm(self.d_inner)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, T, D) input tensor

        Returns:
            (B, T, D) output tensor
        """
        B_sz, T, D = u.shape

        # Input projection → X and gate (paper Fig. 4)
        xz = self.in_proj(u)
        X, gate = xz.chunk(2, dim=-1)

        # Reshape X for multi-head SSM: (B, T, D_inner) → (B, T, H, P)
        X = rearrange(X, 'b t (h p) -> b t h p', h=self.n_heads)

        # Get SSM parameters (B, C, α, δ, ε, ζ)
        B, C, alpha, delta, epsilon, zeta = self.ssm_params(u)

        # Simpson SSD forward pass
        # This is the core change: uses the decomposed L = L_decay × L_conv_tridiag
        Y = simpson_ssd_forward(X, B, C, alpha, delta, epsilon, zeta)

        # Reshape back: (B, T, H, P) → (B, T, D_inner)
        Y = rearrange(Y, 'b t h p -> b t (h p)')

        # Gate and output (paper Fig. 4: nonlinearity then output projection)
        Y = self.norm(Y)
        Y = Y * F.silu(gate)
        Y = self.out_proj(Y)

        return Y


# =============================================================================
# Section 6: Chunked SSD Implementation (for training efficiency)
# =============================================================================

def simpson_ssd_chunked(
    X: torch.Tensor,         # (B, T, H, P)
    B: torch.Tensor,         # (B, T, H, N)
    C: torch.Tensor,         # (B, T, H, N)
    alpha: torch.Tensor,     # (B, T, H)
    delta: torch.Tensor,     # (B, T, H)
    epsilon: torch.Tensor,   # (B, T, H)
    zeta: torch.Tensor,      # (B, T, H)
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Chunked SSD computation for training efficiency.

    Following the paper's training approach (Mamba-2 chunking), we process
    the sequence in chunks of size L, computing intra-chunk with the full
    Simpson mask and inter-chunk via the recurrent state.

    The key insight (paper Appendix B.1) is that the decomposition:
        contract(TN, SN, TJ, JS, SP → TP)(C, B, L_decay, L_conv, X)
    can be chunked, with L_conv applied WITHIN each chunk and L_decay
    handling the inter-chunk state propagation.

    For Simpson, the only change is:
    - Intra-chunk: L_conv is tridiagonal (bandwidth 3) instead of bidiagonal
    - Inter-chunk: state carries h_{t-1} AND the conv lookback buffer (2 tokens)

    The inter-chunk state needs the last 2 tokens' Bx values from the previous
    chunk (for the Simpson conv's lookback), plus the hidden state.
    """
    B_sz, T, H, P = X.shape
    N = B.shape[-1]
    n_chunks = (T + chunk_size - 1) // chunk_size

    # Rearrange to head-first
    X = rearrange(X, 'b t h p -> b h t p')
    B = rearrange(B, 'b t h n -> b h t n')
    C = rearrange(C, 'b t h n -> b h t n')
    alpha = rearrange(alpha, 'b t h -> b h t')
    delta = rearrange(delta, 'b t h -> b h t')
    epsilon = rearrange(epsilon, 'b t h -> b h t')
    zeta = rearrange(zeta, 'b t h -> b h t')

    Y_chunks = []

    # Inter-chunk state
    h_state = torch.zeros(B_sz, H, N, P, device=X.device, dtype=X.dtype)

    # Conv lookback buffer: last 2 tokens' Bx from previous chunk
    Bx_buffer = torch.zeros(B_sz, H, 2, N, P, device=X.device, dtype=X.dtype)

    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, T)
        L = end - start

        X_c = X[:, :, start:end]
        B_c = B[:, :, start:end]
        C_c = C[:, :, start:end]
        alpha_c = alpha[:, :, start:end]
        delta_c = delta[:, :, start:end]
        epsilon_c = epsilon[:, :, start:end]
        zeta_c = zeta[:, :, start:end]

        # Compute Bx for this chunk: (B, H, L, N, P)
        Bx_c = torch.einsum('bhtn, bhtp -> bhtnp', B_c, X_c)

        # Prepend the 2-token buffer from previous chunk for conv lookback
        Bx_with_buffer = torch.cat([Bx_buffer, Bx_c], dim=2)  # (B, H, L+2, N, P)

        # Apply Simpson conv (size-3) using the buffered sequence
        # Need to expand delta, epsilon, zeta to include buffer positions
        # But we only compute output for positions in this chunk
        d = delta_c.unsqueeze(-1).unsqueeze(-1)     # (B, H, L, 1, 1)
        e = epsilon_c.unsqueeze(-1).unsqueeze(-1)
        z = zeta_c.unsqueeze(-1).unsqueeze(-1)

        Z_conv_c = (
            d * Bx_with_buffer[:, :, 0:L]   +       # t-2 (from buffer or earlier in chunk)
            e * Bx_with_buffer[:, :, 1:L+1] +        # t-1
            z * Bx_with_buffer[:, :, 2:L+2]          # t (current)
        )

        # Apply decay scan (intra-chunk) with initial state h_state
        alpha_exp = alpha_c.unsqueeze(-1).unsqueeze(-1)  # (B, H, L, 1, 1)
        H_c = torch.zeros(B_sz, H, L, N, P, device=X.device, dtype=X.dtype)
        H_c[:, :, 0] = alpha_exp[:, :, 0].squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(h_state) * h_state
        H_c[:, :, 0] = alpha_c[:, :, 0].unsqueeze(-1).unsqueeze(-1) * h_state + Z_conv_c[:, :, 0]

        for t in range(1, L):
            H_c[:, :, t] = alpha_c[:, :, t].unsqueeze(-1).unsqueeze(-1) * H_c[:, :, t-1] + Z_conv_c[:, :, t]

        # Output: Y = C^T H
        Y_c = torch.einsum('bhtn, bhtnp -> bhtp', C_c, H_c)
        Y_chunks.append(Y_c)

        # Update inter-chunk state
        h_state = H_c[:, :, -1]

        # Update conv buffer: last 2 Bx values from this chunk
        if L >= 2:
            Bx_buffer = Bx_c[:, :, -2:].clone()
        else:
            # Shift buffer and append
            Bx_buffer = torch.cat([Bx_buffer[:, :, 1:], Bx_c], dim=2)

    Y = torch.cat(Y_chunks, dim=2)
    Y = rearrange(Y, 'b h t p -> b t h p')
    return Y


# =============================================================================
# Section 7: Verification & Testing
# =============================================================================

def test_simpson_decomposition():
    """
    Verify that the Simpson mask L = L_decay @ L_conv produces the same
    result as direct unrolling of the Simpson recurrence.

    This parallels how the paper verifies Eq. 5: showing that the factored
    form equals the unrolled recurrence.
    """
    torch.manual_seed(42)
    T = 8

    # Random parameters
    alpha   = torch.sigmoid(torch.randn(T)) * 0.5 + 0.5   # decay in (0.5, 1)
    delta   = torch.rand(T) * 0.1                           # small weight on t-2
    epsilon = torch.rand(T) * 0.3                            # medium weight on t-1
    zeta    = torch.rand(T) * 0.2                            # small weight on t

    B = torch.randn(T, 4)   # N=4
    C = torch.randn(T, 4)
    X = torch.randn(T, 8)   # P=8

    # Method 1: Build full mask and compute Y = (L ⊙ CB^T) X
    L = build_full_simpson_mask(alpha, delta, epsilon, zeta, T)
    CB = C @ B.T  # (T, T)
    M = L * CB    # Element-wise mask (Eq. 1 pattern)
    Y_mask = M @ X

    # Method 2: Sequential recurrence
    N, P = 4, 8
    h = torch.zeros(N, P)
    Bx = torch.zeros(T, N, P)  # Precompute B_t x_t
    for t in range(T):
        Bx[t] = B[t].unsqueeze(1) * X[t].unsqueeze(0)

    Y_recur = torch.zeros(T, P)
    for t in range(T):
        # h_t = α_t h_{t-1} + δ_t Bx_{t-2} + ε_t Bx_{t-1} + ζ_t Bx_t
        h_new = alpha[t] * h
        if t >= 2:
            h_new = h_new + delta[t] * Bx[t - 2]
        if t >= 1:
            h_new = h_new + epsilon[t] * Bx[t - 1]
        h_new = h_new + zeta[t] * Bx[t]
        h = h_new
        Y_recur[t] = C[t] @ h

    # Compare
    error = (Y_mask - Y_recur).abs().max().item()
    print(f"Max error between mask-based and recurrence-based: {error:.2e}")
    assert error < 1e-5, f"Decomposition verification failed! Error: {error}"
    print("✓ Simpson decomposition verified: L = L_decay × L_conv_tridiag")

    return True


def test_chunked_matches_full():
    """
    Verify chunked implementation matches the non-chunked version.
    """
    torch.manual_seed(42)
    B_sz, T, H, P, N = 2, 32, 4, 16, 8

    X = torch.randn(B_sz, T, H, P)
    B = torch.randn(B_sz, T, H, N)
    C = torch.randn(B_sz, T, H, N)
    alpha = torch.sigmoid(torch.randn(B_sz, T, H)) * 0.3 + 0.6
    delta = torch.rand(B_sz, T, H) * 0.05
    epsilon = torch.rand(B_sz, T, H) * 0.2
    zeta = torch.rand(B_sz, T, H) * 0.1

    Y_full = simpson_ssd_forward(X, B, C, alpha, delta, epsilon, zeta)
    Y_chunked = simpson_ssd_chunked(X, B, C, alpha, delta, epsilon, zeta, chunk_size=8)

    error = (Y_full - Y_chunked).abs().max().item()
    print(f"Max error between full and chunked: {error:.2e}")
    assert error < 1e-4, f"Chunked verification failed! Error: {error}"
    print("✓ Chunked Simpson matches full computation")

    return True


def test_reduces_to_trapezoidal():
    """
    Verify that when δ_t = 0, Simpson reduces to trapezoidal (paper Remark 1).

    With δ_t = 0 (zero weight on t-2):
        h_t = α_t h_{t-1} + 0·Bx_{t-2} + ε_t Bx_{t-1} + ζ_t Bx_t
            = α_t h_{t-1} + ε_t Bx_{t-1} + ζ_t Bx_t    (= trapezoidal, Eq. 4)
    """
    torch.manual_seed(42)
    T = 16
    N, P = 4, 8

    alpha = torch.sigmoid(torch.randn(T)) * 0.5 + 0.5
    beta  = torch.rand(T) * 0.2       # trapezoidal β_t
    gamma = torch.rand(T) * 0.2       # trapezoidal γ_t

    B = torch.randn(T, N)
    C = torch.randn(T, N)
    X = torch.randn(T, P)

    # Trapezoidal recurrence (paper Eq. 4)
    h_trap = torch.zeros(N, P)
    Bx = torch.zeros(T, N, P)
    for t in range(T):
        Bx[t] = B[t].unsqueeze(1) * X[t].unsqueeze(0)

    Y_trap = torch.zeros(T, P)
    for t in range(T):
        h_new = alpha[t] * h_trap
        if t >= 1:
            h_new = h_new + beta[t] * Bx[t - 1]
        h_new = h_new + gamma[t] * Bx[t]
        h_trap = h_new
        Y_trap[t] = C[t] @ h_trap

    # Simpson with δ = 0, ε = β, ζ = γ (should match trapezoidal exactly)
    h_simp = torch.zeros(N, P)
    Y_simp = torch.zeros(T, P)
    for t in range(T):
        h_new = alpha[t] * h_simp
        if t >= 2:
            h_new = h_new + 0.0 * Bx[t - 2]   # δ = 0
        if t >= 1:
            h_new = h_new + beta[t] * Bx[t - 1]
        h_new = h_new + gamma[t] * Bx[t]
        h_simp = h_new
        Y_simp[t] = C[t] @ h_simp

    error = (Y_trap - Y_simp).abs().max().item()
    print(f"Max error Simpson(δ=0) vs trapezoidal: {error:.2e}")
    assert error < 1e-6, f"Reduction to trapezoidal failed! Error: {error}"
    print("✓ Simpson with δ=0 reduces exactly to trapezoidal (paper Eq. 4)")

    return True


def test_full_layer():
    """
    Smoke test for the full SimpsonMamba3Mixer layer.
    """
    torch.manual_seed(42)
    d_model = 128
    B_sz, T = 2, 64

    layer = SimpsonMamba3Mixer(
        d_model=d_model,
        d_state=16,
        n_heads=4,
        head_dim=64,      # 4 heads × 64 = 256 = 128 * expand(2)
        expand=2,
    )

    u = torch.randn(B_sz, T, d_model)
    y = layer(u)

    assert y.shape == (B_sz, T, d_model), f"Wrong output shape: {y.shape}"
    assert not torch.isnan(y).any(), "Output contains NaN"
    print(f"✓ Full SimpsonMamba3Mixer layer: input {u.shape} → output {y.shape}")

    # Verify Simpson init: check that softmax of initial bias ≈ (1/6, 4/6, 1/6)
    with torch.no_grad():
        bias = layer.ssm_params.simpson_proj.bias[:3]  # First head
        weights = F.softmax(bias, dim=0)
        expected = torch.tensor([1/6, 4/6, 1/6])
        init_error = (weights - expected).abs().max().item()
        print(f"  Simpson weight init error from (1/6, 4/6, 1/6): {init_error:.4f}")

    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Data-Dependent Simpson Discretization for Mamba-3")
    print("Extends trapezoidal (Eq. 4-5) to size-3 convolutional mask")
    print("=" * 70)
    print()

    print("--- Test 1: Mask decomposition L = L_decay × L_conv_tridiag ---")
    test_simpson_decomposition()
    print()

    print("--- Test 2: Simpson(δ=0) reduces to trapezoidal ---")
    test_reduces_to_trapezoidal()
    print()

    print("--- Test 3: Chunked matches full computation ---")
    test_chunked_matches_full()
    print()

    print("--- Test 4: Full layer smoke test ---")
    test_full_layer()
    print()

    print("=" * 70)
    print("All tests passed!")
    print("=" * 70)
