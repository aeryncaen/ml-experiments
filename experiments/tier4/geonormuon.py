"""
GeodesicNorMuon: GeoMuon + NorMuon per-row adaptive scaling.

Combines four ideas:

1. NATIVELY SPHERICAL: Parameters live on S^{d-1}. Movement is via exponential
   map (geodesics). No retraction. No normalization. No spectral distortion.

2. TANGENT-SPACE WHITENING: Muon-style Newton-Schulz polar decomposition applied
   to the momentum-blended tangent update, respecting the constraint geometry.

3. PARALLEL-TRANSPORTED MOMENTUM: First moments are transported between tangent
   spaces along geodesics via in-place lerp (matching Muon's momentum scheme).

4. PER-ROW ADAPTIVE SCALING (NorMuon): After NS whitening, each row of the update
   is normalized by its exponential moving average of squared values, then rescaled
   to preserve the original matrix norm.  This gives per-row calibrated step sizes
   while staying on the manifold.

For matrix parameters (2D), uses NS whitening + NorMuon scaling in tangent space.
For vector parameters (1D), uses Adam-style diagonal preconditioning in tangent space.
For non-spherical parameters, uses standard AdamW.

Usage:
    optimizer = GeodesicNorMuon(
        model.parameters(),
        lr=0.02,
        momentum=0.95,
        beta2=0.95,
        spherical_params={id(p) for p in spherical_list},
    )
"""

import torch
from torch.optim import Optimizer
import math
from typing import Set, Optional, List, Dict, Any, Tuple


# ============================================================================
# Newton-Schulz iteration (aligned with Muon canonical impl)
# ============================================================================

@torch.no_grad()
def newton_schulz_polar_fused(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Fused Newton-Schulz quintic iteration in bfloat16.
    Produces approximate polar factor US'V^T.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.no_grad()
def newton_schulz_polar(
    M: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Cubic Newton-Schulz iteration (slower, more precise)."""
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

    I = torch.eye(n, device=M.device, dtype=M.dtype)
    for _ in range(steps):
        A = X.t() @ X
        X = X @ (3 * I - A) / 2

    if transposed:
        X = X.t()

    return X


# ============================================================================
# Manifold operations on the product of spheres
# ============================================================================

class SphereManifold:
    """
    Operations on the product of unit spheres (S^{d-1})^k.
    """

    @staticmethod
    def tangent_project(grad: torch.Tensor, point: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Project gradient to tangent space: remove radial component."""
        dot = (grad * point).sum(dim=dim, keepdim=True)
        return grad - dot * point

    @staticmethod
    def exp_map(point: torch.Tensor, tangent: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Geodesic step: exp_w(v) = cos(||v||) w + sin(||v||) v/||v||"""
        t_norm = tangent.norm(dim=dim, keepdim=True).clamp(min=1e-12)
        return torch.cos(t_norm) * point + torch.sin(t_norm) * (tangent / t_norm)

    @staticmethod
    def parallel_transport(
        vec: torch.Tensor,
        origin: torch.Tensor,
        destination: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Parallel transport vec from T_{origin}S to T_{destination}S."""
        u_sum = origin + destination
        denom = 1.0 + (origin * destination).sum(dim=dim, keepdim=True)

        safe = denom.abs() > 1e-7
        denom_safe = torch.where(safe, denom, torch.ones_like(denom))

        coeff = (vec * u_sum).sum(dim=dim, keepdim=True) / denom_safe
        transported = vec - coeff * u_sum

        fallback = vec - (vec * destination).sum(dim=dim, keepdim=True) * destination
        return torch.where(safe, transported, fallback)


# ============================================================================
# GeodesicNorMuon Optimizer
# ============================================================================

class GeodesicNorMuon(Optimizer):
    """
    Natively spherical optimizer with tangent-space NS whitening + NorMuon scaling.

    For 2D spherical params: NS whitening + per-row second-moment normalization.
    For 1D spherical params: Adam-style diagonal preconditioning in tangent space.
    For non-spherical params: standard AdamW.

    Args:
        params: Model parameters
        lr: Learning rate (default: 0.02)
        momentum: Nesterov momentum coefficient (default: 0.95)
        beta2: NorMuon per-row second moment EMA coefficient (default: 0.95)
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
        beta2: float = 0.95,
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
            beta2=beta2,
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
    ) -> "GeodesicNorMuon":
        """Convenience constructor. Mark params as spherical by name substring."""
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
        Matrix spherical parameter: tangent-space NorMuon with geodesic step.

        1. Project gradient to tangent space
        2. Parallel transport momentum, in-place lerp blend
        3. Nesterov lookahead (mutates grad)
        4. Newton-Schulz whitening of momentum-blended update
        5. NorMuon per-row adaptive scaling (norm-preserving)
        6. Aspect ratio scaling
        7. Tangent re-projection
        8. Exponential map
        """
        grad = p.grad
        dim = group["normalize_dim"]
        lr = group["lr"]
        mu = group["momentum"]
        b2 = group["beta2"]
        ns_steps = group["ns_steps"]
        nesterov = group["nesterov"]
        backend = group["backend"]
        M = self.manifold

        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["momentum_buf"] = torch.zeros_like(p)
            state["second_momentum_buf"] = torch.zeros_like(p[..., 0:1])
            state["prev_point"] = p.data.clone()

        state["step"] += 1
        mom = state["momentum_buf"]
        smom = state["second_momentum_buf"]
        prev_point = state["prev_point"]
        curr_point = p.data

        # --- 1. Tangent projection ---
        g_tan = M.tangent_project(grad, curr_point, dim)

        # --- 2. Parallel transport momentum, then blend (in-place lerp) ---
        if state["step"] > 1:
            mom.copy_(M.parallel_transport(mom, prev_point, curr_point, dim))

        mom.lerp_(g_tan, 1 - mu)

        # Nesterov lookahead (mutates g_tan)
        if nesterov:
            update = g_tan.lerp_(mom, mu)
        else:
            update = mom

        # --- 3. Newton-Schulz whitening of momentum-blended update ---
        shape = update.shape
        if update.dim() > 2:
            u_2d = update.reshape(shape[0], -1) if dim == 0 else update.reshape(-1, shape[-1])
        else:
            u_2d = update

        ns_fn = newton_schulz_polar_fused if backend == "fused" else newton_schulz_polar
        u_white = ns_fn(u_2d, steps=ns_steps)

        if update.dim() > 2:
            u_white = u_white.reshape(shape)

        # --- 4. NorMuon per-row adaptive scaling (norm-preserving) ---
        vnorm = u_white.norm(dim=(-2, -1), keepdim=True)
        v_mean = torch.mean(u_white * u_white, dim=-1, keepdim=True)
        smom.lerp_(v_mean, 1 - b2)
        step_size = 1 / smom.sqrt().add_(1e-10)
        u_white = u_white * step_size
        vnorm_new = u_white.norm(dim=(-2, -1), keepdim=True)
        u_white = u_white * (vnorm / vnorm_new.add_(1e-10))

        # --- 5. Aspect ratio scaling (match Muon) ---
        u_white = u_white * max(1, u_white.size(-2) / u_white.size(-1)) ** 0.5

        # Re-project to tangent space (NS + scaling may introduce tiny radial component)
        update = M.tangent_project(u_white.to(dtype=curr_point.dtype), curr_point, dim)

        # --- 6. Save position, then geodesic step ---
        state["prev_point"] = curr_point.clone()
        p.data.copy_(M.exp_map(curr_point, -lr * update, dim))

    def _step_adam_spherical(self, p: torch.Tensor, group: Dict[str, Any]):
        """1D spherical parameter: tangent-space Adam."""
        grad = p.grad
        dim = 0
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

        g_tan = M.tangent_project(grad, curr_point, dim)

        if state["step"] > 1:
            m_t = M.parallel_transport(m, prev_point, curr_point, dim)
            v_t = M.parallel_transport(v, prev_point, curr_point, dim).abs()
        else:
            m_t, v_t = m, v

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
