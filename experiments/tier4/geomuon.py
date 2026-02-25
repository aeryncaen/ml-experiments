"""
GeodesicMuon: Riemannian optimizer with tangent-space spectral whitening.

Combines three ideas derived from axiomatic optimization on L²(μ):

1. NATIVELY SPHERICAL: Parameters live on S^{d-1}. Movement is via exponential
   map (geodesics). No retraction. No normalization. No spectral distortion.

2. TANGENT-SPACE WHITENING: Muon-style Newton-Schulz polar decomposition applied
   to the tangent gradient, not the ambient gradient. This equalizes the spectrum
   *within* the manifold, respecting the constraint geometry.

3. PARALLEL-TRANSPORTED MOMENTUM: First moments are transported between tangent
   spaces along geodesics. No ambient-space momentum contamination.

The key insight: current whitening optimizers (Shampoo, SOAP, Muon) equalize the
spectrum in ambient R^d. But spherical parameters have d-1 tangential degrees of
freedom and 1 radial (frozen) degree. Whitening in R^d wastes capacity equalizing
the radial direction and distorts the tangential spectrum. Whitening in the tangent
space respects the actual geometry.

For matrix parameters (2D), uses Newton-Schulz polar decomposition of tangent gradient.
For vector parameters (1D), uses Adam-style diagonal preconditioning in tangent space.
For non-spherical parameters, uses standard AdamW.

Usage:
    optimizer = GeodesicMuon(
        model.parameters(),
        lr=0.02,
        momentum=0.95,
        spherical_params={id(p) for p in spherical_list},
    )

    Or with helper:
    optimizer = GeodesicMuon.from_model(
        model, lr=0.02,
        spherical_param_names=["wq", "wk", "wv", "wo", "wu", "wv_mlp", "embed"],
    )
"""

import torch
from torch.optim import Optimizer
import math
from typing import Set, Optional, List, Dict, Any, Tuple


# ============================================================================
# Newton-Schulz iteration for polar decomposition
# ============================================================================

@torch.no_grad()
def newton_schulz_polar(
    M: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute the polar factor U of M = U P via Newton-Schulz iteration.

    U = M (M^T M)^{-1/2} is the nearest orthogonal matrix to M.
    All singular values of U are 1: it whitens the spectrum completely.

    Newton-Schulz iteration (cubic convergence):
        X_0 = M / ||M||_F
        X_{k+1} = X_k (3I - X_k^T X_k) / 2

    After ~5 steps, X_k ≈ U to high precision.

    For non-square M ∈ R^{m×n} with m >= n:
        Iterates on M, converges to the partial isometry factor.
    For m < n:
        Transpose, iterate, transpose back.

    Args:
        M: Input matrix [m, n]
        steps: Number of Newton-Schulz iterations (default 5)
        eps: Regularization for norm (default 1e-7)

    Returns:
        U: Polar factor with orthonormal columns (or rows if m < n)
    """
    m, n = M.shape
    transposed = False

    if m < n:
        M = M.t()
        m, n = n, m
        transposed = True

    # Normalize to spectral radius ≈ 1 for convergence
    norm = M.norm()
    if norm < eps:
        return torch.zeros_like(M.t() if transposed else M)

    X = M / norm

    # Newton-Schulz iterations
    I = torch.eye(n, device=M.device, dtype=M.dtype)
    for _ in range(steps):
        A = X.t() @ X          # [n, n]
        X = X @ (3 * I - A) / 2  # [m, n]

    if transposed:
        X = X.t()

    return X


@torch.no_grad()
def newton_schulz_polar_fused(
    M: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Fused Newton-Schulz that avoids materializing intermediate [m,m] matrices.

    Uses the Zhu et al. (2020) variant with better numerical stability:
        a, b, c coefficients tuned for faster convergence.

    For Muon-style optimizers, this is the hot path.
    """
    m, n = M.shape
    transposed = False

    if m < n:
        M = M.t()
        m, n = n, m
        transposed = True

    norm = M.norm()
    if norm < eps:
        return torch.zeros_like(M.t() if transposed else M)

    X = M / norm

    # Quintic coefficients from Muon paper for faster convergence
    # These are optimized for the spectral distribution of NN gradients
    a, b, c = (3.4445, -4.7750, 2.0315)

    I = torch.eye(n, device=M.device, dtype=M.dtype)
    for _ in range(steps):
        A = X.t() @ X
        X = X @ (a * I + b * A + c * A @ A)

    if transposed:
        X = X.t()

    return X


# ============================================================================
# Manifold operations on the product of spheres
# ============================================================================

class SphereManifold:
    """
    Operations on the product of unit spheres (S^{d-1})^k.

    A matrix W ∈ R^{d×k} with each column on S^{d-1}.
    Tangent space at W: matrices V where <V_i, W_i> = 0 for each column i.

    All operations are O(dk) — same as normalization.
    """

    @staticmethod
    def tangent_project(grad: torch.Tensor, point: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Project gradient to tangent space: remove radial component per column.
        g_T = g - <g, w> w  for each column w.
        """
        dot = (grad * point).sum(dim=dim, keepdim=True)
        return grad - dot * point

    @staticmethod
    def exp_map(point: torch.Tensor, tangent: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Geodesic step: exp_w(v) = cos(||v||) w + sin(||v||) v/||v||
        Exact. Never leaves the sphere.
        """
        t_norm = tangent.norm(dim=dim, keepdim=True).clamp(min=1e-12)
        return torch.cos(t_norm) * point + torch.sin(t_norm) * (tangent / t_norm)

    @staticmethod
    def parallel_transport(
        vec: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """
        Parallel transport vec from T_{origin}S to T_{destination}S.
        P_{u→u'}(v) = v − <v, u'+u>/(1+<u,u'>) (u'+u)
        """
        u_sum = origin + destination
        denom = 1.0 + (origin * destination).sum(dim=dim, keepdim=True)

        # Antipodal guard
        safe = denom.abs() > 1e-7
        denom_safe = torch.where(safe, denom, torch.ones_like(denom))

        coeff = (vec * u_sum).sum(dim=dim, keepdim=True) / denom_safe
        transported = vec - coeff * u_sum

        # Fallback to tangent projection if near-antipodal
        fallback = vec - (vec * destination).sum(dim=dim, keepdim=True) * destination
        return torch.where(safe, transported, fallback)

    @staticmethod
    def geodesic_distance(a: torch.Tensor, b: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """d(a,b) = arccos(<a,b>) per column."""
        dot = (a * b).sum(dim=dim).clamp(-1.0, 1.0)
        return torch.acos(dot)


# ============================================================================
# GeodesicMuon Optimizer
# ============================================================================

class GeodesicMuon(Optimizer):
    """
    Natively spherical optimizer with tangent-space spectral whitening.

    For 2D spherical params: Newton-Schulz whitening of tangent gradient + momentum.
    For 1D spherical params: Adam-style diagonal preconditioning in tangent space.
    For non-spherical params: standard AdamW.

    Args:
        params: Model parameters
        lr: Learning rate (default: 0.02, Muon-style — higher than Adam)
        momentum: Nesterov momentum coefficient (default: 0.95)
        ns_steps: Newton-Schulz iterations (default: 5)
        betas: Adam betas for 1D spherical and non-spherical params (default: (0.9, 0.999))
        eps: Adam epsilon (default: 1e-8)
        weight_decay: For non-spherical params only (default: 0.0)
        spherical_params: Set of param ids living on the sphere
        normalize_dim: Dimension of unit-norm columns (default: 0)
        nesterov: Use Nesterov momentum for matrix params (default: True)
        backend: "fused" uses quintic NS, "standard" uses cubic (default: "fused")
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        spherical_params: Optional[Set[int]] = None,
        normalize_dim: int = 0,
        nesterov: bool = True,
        backend: str = "fused",
    ):
        if spherical_params is None:
            spherical_params = set()

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            spherical_params=spherical_params,
            normalize_dim=normalize_dim,
            nesterov=nesterov,
            backend=backend,
        )
        super().__init__(params, defaults)

        self.manifold = SphereManifold()

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        lr: float = 0.02,
        momentum: float = 0.95,
        spherical_param_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "GeodesicMuon":
        """
        Convenience constructor. Mark params as spherical by name substring.

        Example:
            opt = GeodesicMuon.from_model(
                model, lr=0.02,
                spherical_param_names=["wq", "wk", "wv", "wo", "wu", "wv_mlp", "embed"]
            )
        """
        if spherical_param_names is None:
            spherical_param_names = []

        spherical_ids = set()
        for name, param in model.named_parameters():
            for sname in spherical_param_names:
                if sname in name:
                    spherical_ids.add(id(param))
                    break

        return cls(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            spherical_params=spherical_ids,
            **kwargs,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            spherical_ids = group["spherical_params"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                is_spherical = id(p) in spherical_ids

                if is_spherical and p.dim() >= 2:
                    self._step_muon_spherical(p, group)
                elif is_spherical and p.dim() == 1:
                    self._step_adam_spherical(p, group)
                else:
                    self._step_adamw(p, group)

        return loss

    def _step_muon_spherical(self, p: torch.Tensor, group: Dict[str, Any]):
        """
        Matrix spherical parameter: tangent-space Muon with geodesic step.

        1. Project gradient to tangent space
        2. Whiten tangent gradient via Newton-Schulz polar decomposition
        3. Nesterov momentum with parallel transport
        4. Exponential map
        """
        grad = p.grad
        dim = group["normalize_dim"]
        lr = group["lr"]
        mu = group["momentum"]
        ns_steps = group["ns_steps"]
        nesterov = group["nesterov"]
        backend = group["backend"]
        M = self.manifold

        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["momentum_buf"] = torch.zeros_like(p)
            state["prev_point"] = p.data.clone()

        state["step"] += 1
        mom = state["momentum_buf"]
        prev_point = state["prev_point"]
        curr_point = p.data

        # --- 1. Tangent projection ---
        g_tan = M.tangent_project(grad, curr_point, dim)

        # --- 2. Newton-Schulz whitening in tangent space ---
        # Reshape to 2D if needed for NS (handles multi-head etc.)
        shape = g_tan.shape
        if g_tan.dim() > 2:
            g_2d = g_tan.reshape(shape[0], -1) if dim == 0 else g_tan.reshape(-1, shape[-1])
        else:
            g_2d = g_tan

        # Apply polar decomposition to tangent gradient
        ns_fn = newton_schulz_polar_fused if backend == "fused" else newton_schulz_polar
        g_white = ns_fn(g_2d, steps=ns_steps)

        if g_tan.dim() > 2:
            g_white = g_white.reshape(shape)

        # Re-project to tangent space (NS may introduce tiny radial component)
        g_white = M.tangent_project(g_white, curr_point, dim)

        # --- 3. Momentum with parallel transport ---
        if state["step"] > 1:
            mom_transported = M.parallel_transport(mom, prev_point, curr_point, dim)
        else:
            mom_transported = mom

        # Update momentum
        mom.copy_(mu * mom_transported + g_white)

        # Nesterov lookahead
        if nesterov:
            update = mu * mom + g_white
        else:
            update = mom.clone()

        # Ensure update is tangent
        update = M.tangent_project(update, curr_point, dim)

        # --- 4. Save position, then geodesic step ---
        state["prev_point"] = curr_point.clone()
        p.data.copy_(M.exp_map(curr_point, -lr * update, dim))

    def _step_adam_spherical(self, p: torch.Tensor, group: Dict[str, Any]):
        """
        1D spherical parameter (vector on S^{d-1}): tangent-space Adam.
        Newton-Schulz needs matrix structure; use diagonal preconditioning for vectors.
        """
        grad = p.grad
        dim = 0  # 1D: normalize along dim 0
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        M = self.manifold

        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["m"] = torch.zeros_like(p)
            state["v"] = torch.zeros_like(p)
            state["prev_point"] = p.data.clone()

        state["step"] += 1
        m, v = state["m"], state["v"]
        prev_point = state["prev_point"]
        curr_point = p.data

        # Tangent projection
        g_tan = M.tangent_project(grad, curr_point, dim)

        # Parallel transport moments
        if state["step"] > 1:
            m_t = M.parallel_transport(m, prev_point, curr_point, dim)
            v_t = M.parallel_transport(v, prev_point, curr_point, dim).abs()
        else:
            m_t, v_t = m, v

        # Adam update in tangent space
        m.copy_(beta1 * m_t + (1 - beta1) * g_tan)
        v.copy_(beta2 * v_t + (1 - beta2) * g_tan.square())

        bc1 = 1 - beta1 ** state["step"]
        bc2 = 1 - beta2 ** state["step"]
        direction = (m / bc1) / ((v / bc2).sqrt() + eps)

        direction = M.tangent_project(direction, curr_point, dim)

        state["prev_point"] = curr_point.clone()
        p.data.copy_(M.exp_map(curr_point, -lr * direction, dim))

    def _step_adamw(self, p: torch.Tensor, group: Dict[str, Any]):
        """Standard AdamW for non-spherical parameters."""
        grad = p.grad
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["m"] = torch.zeros_like(p)
            state["v"] = torch.zeros_like(p)

        state["step"] += 1
        m, v = state["m"], state["v"]

        if wd != 0:
            p.data.add_(p.data, alpha=-lr * wd)

        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bc1 = 1 - beta1 ** state["step"]
        bc2 = 1 - beta2 ** state["step"]
        p.data.addcdiv_(m / bc1, (v / bc2).sqrt() + eps, value=-lr)


# ============================================================================
# Diagnostics
# ============================================================================

def full_diagnostics(
    model: torch.nn.Module,
    spherical_param_names: List[str],
    normalize_dim: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive diagnostics for spherical parameters.

    Reports:
    - Sphere fidelity: how close column norms are to 1
    - Spectral health: condition number, singular value spread
    - Angular structure: pairwise cosine statistics
    - Effective rank: how many dimensions carry meaningful energy
    """
    stats = {}
    M = SphereManifold()

    for name, param in model.named_parameters():
        is_spherical = any(s in name for s in spherical_param_names)
        if not is_spherical:
            continue

        with torch.no_grad():
            d = param.data

            # Sphere fidelity
            norms = d.norm(dim=normalize_dim)
            norm_dev = (norms - 1.0).abs()

            info = {
                "sphere_max_deviation": norm_dev.max().item(),
                "sphere_mean_deviation": norm_dev.mean().item(),
            }

            if param.dim() >= 2:
                # Spectral health
                S = torch.linalg.svdvals(d)
                info["condition_number"] = (S[0] / S[-1].clamp(min=1e-10)).item()
                info["singular_max"] = S[0].item()
                info["singular_min"] = S[-1].item()

                # Effective rank: exp(entropy of normalized singular values)
                S_norm = S / S.sum()
                entropy = -(S_norm * S_norm.clamp(min=1e-12).log()).sum()
                info["effective_rank"] = entropy.exp().item()
                info["full_rank"] = min(d.shape[0], d.shape[1])

                # Angular structure (pairwise cosines)
                if normalize_dim == 0:
                    W = d.t()  # columns to rows for gram computation
                else:
                    W = d
                G = W @ W.t()
                n = G.shape[0]
                if n > 1 and n <= 4096:  # skip if too large
                    mask = ~torch.eye(n, dtype=torch.bool, device=G.device)
                    off = G[mask]
                    info["cosine_mean"] = off.mean().item()
                    info["cosine_std"] = off.std().item()
                    info["cosine_absmax"] = off.abs().max().item()

            stats[name] = info

    return stats


def print_diagnostics(stats: Dict[str, Dict[str, float]]):
    """Pretty-print diagnostic output."""
    for name, info in stats.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        for k, v in info.items():
            if isinstance(v, float):
                if abs(v) < 0.01 or abs(v) > 1000:
                    print(f"  {k:30s}: {v:.4e}")
                else:
                    print(f"  {k:30s}: {v:.6f}")
            else:
                print(f"  {k:30s}: {v}")


# ============================================================================
# Quick benchmark
# ============================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Test problem: matrix of unit-norm columns, optimize dot products with targets
    d_model = 256
    n_cols = 64
    n_targets = 8

    # Random targets on sphere
    targets = torch.randn(d_model, n_targets, device=device)
    targets = targets / targets.norm(dim=0, keepdim=True)

    def make_param():
        W = torch.randn(d_model, n_cols, device=device)
        W = W / W.norm(dim=0, keepdim=True)
        return torch.nn.Parameter(W.clone())

    def loss_fn(W, targets):
        # Maximize sum of max dot products per target
        dots = W.t() @ targets  # [n_cols, n_targets]
        return -dots.max(dim=0).values.sum()

    n_steps = 300

    # --- GeodesicMuon ---
    W_gm = make_param()
    opt_gm = GeodesicMuon(
        [W_gm], lr=0.02, momentum=0.95,
        spherical_params={id(W_gm)}, normalize_dim=0,
    )
    gm_losses = []
    gm_norms = []
    t0 = time.time()
    for step in range(n_steps):
        opt_gm.zero_grad()
        loss = loss_fn(W_gm, targets)
        loss.backward()
        opt_gm.step()
        gm_losses.append(loss.item())
        gm_norms.append((W_gm.data.norm(dim=0) - 1).abs().max().item())
    t_gm = time.time() - t0

    # --- GeodesicAdam (from our previous optimizer) ---
    from geodesic_adam import GeodesicAdam

    W_ga = make_param()
    opt_ga = GeodesicAdam(
        [W_ga], lr=0.01,
        spherical_params={id(W_ga)}, normalize_dim=0,
    )
    ga_losses = []
    ga_norms = []
    t0 = time.time()
    for step in range(n_steps):
        opt_ga.zero_grad()
        loss = loss_fn(W_ga, targets)
        loss.backward()
        opt_ga.step()
        ga_losses.append(loss.item())
        ga_norms.append((W_ga.data.norm(dim=0) - 1).abs().max().item())
    t_ga = time.time() - t0

    # --- Standard Adam + retraction ---
    W_ret = make_param()
    opt_ret = torch.optim.Adam([W_ret], lr=0.01)
    ret_losses = []
    ret_norms = []
    t0 = time.time()
    for step in range(n_steps):
        opt_ret.zero_grad()
        loss = loss_fn(W_ret, targets)
        loss.backward()
        opt_ret.step()
        with torch.no_grad():
            drift = (W_ret.data.norm(dim=0) - 1).abs().max().item()
            W_ret.data.copy_(W_ret.data / W_ret.data.norm(dim=0, keepdim=True))
        ret_losses.append(loss.item())
        ret_norms.append(drift)
    t_ret = time.time() - t0

    # --- Standard Muon-style (ambient NS + retraction) ---
    W_muon = make_param()
    muon_mom = torch.zeros_like(W_muon.data)
    muon_lr = 0.02
    muon_mu = 0.95
    muon_losses = []
    muon_norms = []
    t0 = time.time()
    for step in range(n_steps):
        W_muon.grad = None
        loss = loss_fn(W_muon, targets)
        loss.backward()
        with torch.no_grad():
            g = W_muon.grad
            # Ambient NS (no tangent projection)
            g_white = newton_schulz_polar_fused(g, steps=5)
            muon_mom = muon_mu * muon_mom + g_white
            update = muon_mu * muon_mom + g_white  # nesterov
            W_muon.data.add_(update, alpha=-muon_lr)
            # Retract
            drift = (W_muon.data.norm(dim=0) - 1).abs().max().item()
            W_muon.data.copy_(W_muon.data / W_muon.data.norm(dim=0, keepdim=True))
        muon_losses.append(loss.item())
        muon_norms.append(drift)
    t_muon = time.time() - t0

    # Results
    print(f"{'':30s} {'GeoMuon':>12s} {'GeoAdam':>12s} {'Adam+Ret':>12s} {'Muon+Ret':>12s}")
    print("-" * 80)
    print(f"{'Final loss':30s} {gm_losses[-1]:12.4f} {ga_losses[-1]:12.4f} {ret_losses[-1]:12.4f} {muon_losses[-1]:12.4f}")
    print(f"{'Best loss':30s} {min(gm_losses):12.4f} {min(ga_losses):12.4f} {min(ret_losses):12.4f} {min(muon_losses):12.4f}")
    print(f"{'Steps to -7.0 (or never)':30s}", end="")
    for losses in [gm_losses, ga_losses, ret_losses, muon_losses]:
        reached = next((i for i, l in enumerate(losses) if l < -7.0), -1)
        if reached >= 0:
            print(f" {reached:11d}s", end="")
        else:
            print(f" {'never':>12s}", end="")
    print()
    print(f"{'Max sphere deviation':30s} {max(gm_norms):12.2e} {max(ga_norms):12.2e} {max(ret_norms):12.2e} {max(muon_norms):12.2e}")
    print(f"{'Wall time (s)':30s} {t_gm:12.3f} {t_ga:12.3f} {t_ret:12.3f} {t_muon:12.3f}")

    # Spectral comparison
    print(f"\n{'Condition numbers':30s}", end="")
    for W in [W_gm, W_ga, W_ret, W_muon]:
        S = torch.linalg.svdvals(W.data)
        cond = (S[0] / S[-1].clamp(min=1e-10)).item()
        print(f" {cond:12.2f}", end="")
    print()

    # Effective rank
    print(f"{'Effective rank':30s}", end="")
    for W in [W_gm, W_ga, W_ret, W_muon]:
        S = torch.linalg.svdvals(W.data)
        S_n = S / S.sum()
        ent = -(S_n * S_n.clamp(min=1e-12).log()).sum()
        print(f" {ent.exp().item():12.2f}", end="")
    print(f"  (of {min(d_model, n_cols)})")

    print("\n=== Loss at milestones ===")
    for step_i in [10, 50, 100, 200, 299]:
        print(f"  Step {step_i+1:3d}:  "
              f"GeoMuon={gm_losses[step_i]:8.4f}  "
              f"GeoAdam={ga_losses[step_i]:8.4f}  "
              f"Adam+Ret={ret_losses[step_i]:8.4f}  "
              f"Muon+Ret={muon_losses[step_i]:8.4f}")

    print("\nDone.")
