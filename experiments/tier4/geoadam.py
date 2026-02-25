"""
GeodesicAdam: A natively spherical optimizer for hypersphere-constrained parameters.

Instead of stepping in ambient space and retracting (normalizing) back to the sphere,
this optimizer works entirely in tangent spaces and moves via the exponential map.

Key operations:
  1. Tangent projection: remove radial gradient component
  2. Parallel transport: move momentum/second moments between tangent spaces  
  3. Adam step: computed entirely in tangent space
  4. Exponential map: geodesic step on S^{d-1}, never leaving the sphere

No retraction. No normalization. No spectral distortion.

Usage:
    optimizer = GeodesicAdam(model.parameters(), lr=1e-3, spherical_params={id(p) for p in spherical_list})
    
    Or with the helper:
    optimizer = GeodesicAdam.from_model(model, lr=1e-3, spherical_param_names=["embed", "wq", "wk", ...])

For nGPT-style models, mark all normalized weight matrices as spherical.
Non-spherical parameters (biases, scaling factors, eigen learning rates) use standard Adam.
"""

import torch
from torch.optim import Optimizer
import math
from typing import Set, Optional, List, Dict, Any


class GeodesicAdam(Optimizer):
    """
    Adam optimizer that moves spherical parameters via geodesics on S^{d-1}.
    
    Parameters marked as spherical are assumed to have unit-norm columns (or unit-norm
    vectors if 1D). The optimizer maintains momentum and second moments in the tangent
    space and parallel-transports them between steps.
    
    Non-spherical parameters use standard Adam.
    
    Args:
        params: Iterable of parameters or param groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        spherical_params: Set of parameter ids that live on the sphere.
                         These must be pre-normalized to unit norm along normalize_dim.
        normalize_dim: Dimension along which columns are unit-normalized (default: 0).
                      For embedding matrices [V, d], use dim=1.
                      For weight matrices [d_out, d_in], use dim=0 (each column of W^T).
        weight_decay: L2 penalty for non-spherical params only (default: 0).
                     Ignored for spherical params (they don't need it).
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        spherical_params: Optional[Set[int]] = None,
        normalize_dim: int = 0,
        weight_decay: float = 0.0,
        adam_lr: Optional[float] = None,
    ):
        if spherical_params is None:
            spherical_params = set()
        
        defaults = dict(
            lr=lr,
            adam_lr=adam_lr if adam_lr is not None else lr,
            betas=betas,
            eps=eps,
            spherical_params=spherical_params,
            normalize_dim=normalize_dim,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        spherical_param_names: Optional[List[str]] = None,
        normalize_dim: int = 0,
        weight_decay: float = 0.0,
    ) -> "GeodesicAdam":
        """
        Convenience constructor. Pass substrings of parameter names to mark as spherical.
        
        Example:
            optimizer = GeodesicAdam.from_model(
                model, lr=1e-3,
                spherical_param_names=["wq", "wk", "wv", "wo", "wu", "wv_mlp", "wo_mlp", "embed"]
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
            betas=betas,
            eps=eps,
            spherical_params=spherical_ids,
            normalize_dim=normalize_dim,
            weight_decay=weight_decay,
        )

    @staticmethod
    def _tangent_project(grad: torch.Tensor, param: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Project gradient onto tangent space of S^{d-1} at param.
        
        For a unit vector u, the tangent space is {v : <v, u> = 0}.
        Projection: g_T = g - <g, u> u
        
        Applied independently along each column (slice along normalize_dim).
        """
        # Inner product along the normalize dimension
        # dot shape: same as param but with dim collapsed
        dot = (grad * param).sum(dim=dim, keepdim=True)
        return grad - dot * param
    
    @staticmethod
    def _parallel_transport(
        v: torch.Tensor,
        u_old: torch.Tensor,
        u_new: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """
        Parallel transport tangent vector v from T_{u_old}S to T_{u_new}S.
        
        Closed form on the sphere:
            P_{u->u'}(v) = v - <v, u' + u> / (1 + <u, u'>) * (u' + u)
        
        This preserves norms and angles during transport.
        Handles the edge case where u_old ≈ -u_new (antipodal).
        """
        u_sum = u_old + u_new
        dot_uu = (u_old * u_new).sum(dim=dim, keepdim=True)
        denom = 1.0 + dot_uu
        
        # Antipodal guard: if u_old ≈ -u_new, transport is undefined.
        # In practice this shouldn't happen with small learning rates.
        # Fall back to tangent projection at u_new if it does.
        safe = denom.abs() > 1e-7
        denom = torch.where(safe, denom, torch.ones_like(denom))
        
        dot_v_sum = (v * u_sum).sum(dim=dim, keepdim=True)
        transported = v - (dot_v_sum / denom) * u_sum
        
        # Fallback: just project onto new tangent space
        fallback = v - (v * u_new).sum(dim=dim, keepdim=True) * u_new
        
        return torch.where(safe, transported, fallback)
    
    @staticmethod
    def _exp_map(
        u: torch.Tensor,
        v: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """
        Exponential map on S^{d-1}: move from u along tangent vector v.
        
            exp_u(v) = cos(||v||) * u + sin(||v||) * v / ||v||
        
        This is an exact geodesic step. No retraction. No normalization.
        The result is exactly on the sphere (up to floating point).
        """
        v_norm = v.norm(dim=dim, keepdim=True).clamp(min=1e-12)
        
        cos_vn = torch.cos(v_norm)
        sin_vn = torch.sin(v_norm)
        
        return cos_vn * u + sin_vn * (v / v_norm)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Spherical parameters: tangent projection -> parallel transport momentum ->
                             Adam in tangent space -> exponential map
        
        Non-spherical parameters: standard Adam (with optional weight decay)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            adam_lr = group["adam_lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            spherical_ids = group["spherical_params"]
            dim = group["normalize_dim"]
            wd = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                is_spherical = id(p) in spherical_ids
                
                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)  # first moment
                    state["v"] = torch.zeros_like(p)  # second moment
                    if is_spherical:
                        # Store previous position for parallel transport
                        state["u_prev"] = p.data.clone()
                
                state["step"] += 1
                m, v = state["m"], state["v"]
                
                if is_spherical:
                    self._step_spherical(
                        p, grad, m, v, state, lr, beta1, beta2, eps, dim
                    )
                else:
                    self._step_euclidean(
                        p, grad, m, v, state, adam_lr, beta1, beta2, eps, wd
                    )
        
        return loss
    
    def _step_spherical(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        state: Dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        dim: int,
    ):
        """Geodesic Adam step for spherical parameters."""
        step = state["step"]
        u_prev = state["u_prev"]
        u_curr = p.data
        
        # --- Step 1: Project gradient to tangent space at current point ---
        g_tan = self._tangent_project(grad, u_curr, dim)
        
        # --- Step 2: Parallel transport momentum from previous tangent space ---
        if step > 1:
            # Transport first moment
            m_transported = self._parallel_transport(m, u_prev, u_curr, dim)
            # Transport second moment
            # v is element-wise squared gradient stats - transport the sqrt,
            # square after. This preserves the metric interpretation better.
            # However for diagonal Adam, component-wise transport then
            # component-wise squaring is the pragmatic choice.
            v_transported = self._parallel_transport(v, u_prev, u_curr, dim)
            # Ensure v stays non-negative after transport
            # (transport can introduce tiny negatives from floating point)
            v_transported = v_transported.abs()
        else:
            m_transported = m
            v_transported = v
        
        # --- Step 3: Adam update in tangent space ---
        m.copy_(beta1 * m_transported + (1 - beta1) * g_tan)
        v.copy_(beta2 * v_transported + (1 - beta2) * g_tan.square())
        
        # Bias correction
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        m_hat = m / bc1
        v_hat = v / bc2
        
        # Adam direction in tangent space
        direction = m_hat / (v_hat.sqrt() + eps)
        
        # Re-project to tangent space (numerical hygiene - 
        # m and v should already be tangent but floating point drifts)
        direction = self._tangent_project(direction, u_curr, dim)
        
        # Scale by learning rate (negative for descent)
        step_vec = -lr * direction
        
        # --- Step 4: Save current position before moving ---
        state["u_prev"] = u_curr.clone()
        
        # --- Step 5: Exponential map - geodesic step on sphere ---
        p.data.copy_(self._exp_map(u_curr, step_vec, dim))
    
    def _step_euclidean(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        state: Dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        wd: float,
    ):
        """Standard Adam step for non-spherical parameters."""
        step = state["step"]
        
        # Weight decay (decoupled, AdamW-style) — use adam_lr not lr
        if wd != 0:
            p.data.add_(p.data, alpha=-lr * wd)  # lr is already adam_lr from caller
        
        # Standard Adam
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        m_hat = m / bc1
        v_hat = v / bc2
        
        p.data.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


# ============================================================================
# Diagnostic utilities
# ============================================================================

def sphere_diagnostics(model, spherical_param_names: List[str], normalize_dim: int = 0):
    """
    Check how well spherical parameters stay on the sphere.
    
    With GeodesicAdam, column norms should be exactly 1.0 (up to float precision).
    With retraction-based methods, they'd be exactly 1.0 after retraction but
    drift between retractions.
    
    Returns dict with per-parameter stats.
    """
    stats = {}
    for name, param in model.named_parameters():
        is_spherical = any(s in name for s in spherical_param_names)
        if not is_spherical:
            continue
        
        with torch.no_grad():
            norms = param.norm(dim=normalize_dim)
            stats[name] = {
                "mean_norm": norms.mean().item(),
                "std_norm": norms.std().item(),
                "max_deviation": (norms - 1.0).abs().max().item(),
                "min_norm": norms.min().item(),
                "max_norm": norms.max().item(),
            }
    
    return stats


def spectral_diagnostics(model, spherical_param_names: List[str], normalize_dim: int = 0):
    """
    Compute condition numbers and pairwise cosine statistics for spherical params.
    
    In a natively spherical optimizer, spectral structure should be cleaner
    than with retraction because no spectral distortion is introduced.
    """
    stats = {}
    for name, param in model.named_parameters():
        is_spherical = any(s in name for s in spherical_param_names)
        if not is_spherical or param.dim() < 2:
            continue
        
        with torch.no_grad():
            # Reshape so columns are along normalize_dim
            if normalize_dim == 0:
                W = param.data  # [d_out, d_in], columns are rows
            else:
                W = param.data.t()
            
            # Gram matrix
            G = W @ W.t()
            
            # Condition number via SVD
            S = torch.linalg.svdvals(W)
            cond = (S[0] / S[-1]).item() if S[-1] > 1e-10 else float("inf")
            
            # Pairwise cosine stats (off-diagonal of Gram matrix for unit-norm cols)
            n = G.shape[0]
            if n > 1:
                mask = ~torch.eye(n, dtype=torch.bool, device=G.device)
                off_diag = G[mask]
                cos_mean = off_diag.mean().item()
                cos_std = off_diag.std().item()
                cos_max = off_diag.abs().max().item()
            else:
                cos_mean = cos_std = cos_max = 0.0
            
            stats[name] = {
                "condition_number": cond,
                "singular_max": S[0].item(),
                "singular_min": S[-1].item(),
                "cosine_mean": cos_mean,
                "cosine_std": cos_std,
                "cosine_max_abs": cos_max,
            }
    
    return stats


def geodesic_distance(u: torch.Tensor, v: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Geodesic distance on S^{d-1} between unit vectors u and v.
    
    d(u, v) = arccos(<u, v>)
    
    This is the actual distance on the sphere, not the ambient Euclidean distance.
    """
    dot = (u * v).sum(dim=dim).clamp(-1.0, 1.0)
    return torch.acos(dot)


# ============================================================================
# Quick test / demo
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create a simple test: optimize vectors on S^{d-1} to maximize dot product with target
    d = 128
    n_vectors = 32
    
    # Target direction (unit norm)
    target = torch.randn(d, device=device)
    target = target / target.norm()
    
    # Learnable matrix: n_vectors columns on S^{d-1}
    W = torch.randn(d, n_vectors, device=device)
    W = W / W.norm(dim=0, keepdim=True)  # Initialize on sphere
    W = torch.nn.Parameter(W)
    
    # --- Test with GeodesicAdam ---
    W_geo = torch.nn.Parameter(W.data.clone())
    opt_geo = GeodesicAdam(
        [W_geo], lr=0.01, spherical_params={id(W_geo)}, normalize_dim=0
    )
    
    geo_losses = []
    geo_norm_devs = []
    for step in range(200):
        opt_geo.zero_grad()
        # Loss: negative mean dot product with target
        dots = target @ W_geo  # [n_vectors]
        loss = -dots.mean()
        loss.backward()
        opt_geo.step()
        
        geo_losses.append(loss.item())
        norm_dev = (W_geo.data.norm(dim=0) - 1.0).abs().max().item()
        geo_norm_devs.append(norm_dev)
    
    # --- Test with standard Adam + retraction ---
    W_ret = torch.nn.Parameter(W.data.clone())
    opt_ret = torch.optim.Adam([W_ret], lr=0.01)
    
    ret_losses = []
    ret_norm_devs = []
    for step in range(200):
        opt_ret.zero_grad()
        dots = target @ W_ret
        loss = -dots.mean()
        loss.backward()
        opt_ret.step()
        
        # Retract (normalize)
        with torch.no_grad():
            norm_before = (W_ret.data.norm(dim=0) - 1.0).abs().max().item()
            W_ret.data.copy_(W_ret.data / W_ret.data.norm(dim=0, keepdim=True))
        
        ret_losses.append(loss.item())
        ret_norm_devs.append(norm_before)
    
    print("\n=== Results (200 steps) ===")
    print(f"{'':30s} {'GeodesicAdam':>14s} {'Adam+Retract':>14s}")
    print(f"{'Final loss':30s} {geo_losses[-1]:14.6f} {ret_losses[-1]:14.6f}")
    print(f"{'Max norm deviation (final)':30s} {geo_norm_devs[-1]:14.2e} {ret_norm_devs[-1]:14.2e}")
    print(f"{'Max norm deviation (worst)':30s} {max(geo_norm_devs):14.2e} {max(ret_norm_devs):14.2e}")
    
    # Spectral comparison
    S_geo = torch.linalg.svdvals(W_geo.data)
    S_ret = torch.linalg.svdvals(W_ret.data)
    cond_geo = (S_geo[0] / S_geo[-1]).item()
    cond_ret = (S_ret[0] / S_ret[-1]).item()
    print(f"{'Condition number':30s} {cond_geo:14.4f} {cond_ret:14.4f}")
    
    # Check geodesic distances from init
    W_init = W.data.clone()
    W_init = W_init / W_init.norm(dim=0, keepdim=True)
    
    geo_dist = geodesic_distance(W_init, W_geo.data, dim=0).mean().item()
    ret_dist = geodesic_distance(W_init, W_ret.data, dim=0).mean().item()
    print(f"{'Mean geodesic dist from init':30s} {geo_dist:14.4f} {ret_dist:14.4f}")
    
    # Verify sphere constraint held throughout for geodesic
    print(f"\nGeodesicAdam sphere constraint: max deviation = {max(geo_norm_devs):.2e}")
    print(f"Adam+Retract pre-retraction drift: max deviation = {max(ret_norm_devs):.2e}")
    
    print("\n=== Norm deviation over training ===")
    for i in [0, 49, 99, 149, 199]:
        print(f"  Step {i+1:3d}: Geodesic {geo_norm_devs[i]:.2e}  |  Retract {ret_norm_devs[i]:.2e}")
    
    print("\nDone.")
