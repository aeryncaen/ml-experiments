# MagMuon optimizer — Momentum-Adaptive Gradient Muon
# SI-EDP canonical flow: θ ← θ + η · J† · (Proj_{T_U}(p̂) − ∂R_T(u))
#
# Correct ordering per Prop 6.6 (face-restricted susceptibility):
#   δu = (H_R|_T)⁻¹ P_T δp — project first, then operate within T.
#
# Operational steps:
#   1. Nesterov momentum → p̂ (estimate of signal p = Π_S Φ°)
#   2. Proj_{T_U}(p̂) — project onto tangent plane of unit sphere per neuron
#   3. Adaptive gain within tangent space — scalar rescaling from gnorm deviation
#   4. AOL + NS5 → J† (pullback to θ-space)
import torch
import torch.distributed as dist


def magmuon_update(grad, weight, momentum, gnorm_ema, beta=0.95, gnorm_beta=0.99, lambda_reg=1.0, ns_steps=5, nesterov=True):
    """
    MagMuon update — SI-EDP canonical flow:
      θ ← θ + η · J† · (Proj_{T_U}(p̂) − ∂R_T(u))

    1. Nesterov momentum → p̂ (out-of-place, does not mutate grad)
    2. Proj_{T_U}(p̂): tangent plane projection per neuron row
    3. Adaptive gain: scalar rescaling per neuron from tangent gnorm deviation
       gain = (1 - λ) + λ · ema / tangent_gnorm
       Shrinks large tangent gradients, boosts small ones toward EMA.
    4. AOL warm start + NS5 refinement → J† pullback
    """
    # Step 1: Nesterov momentum → p̂ (out-of-place to avoid mutating p.grad)
    momentum.lerp_(grad, 1 - beta)
    p_hat = grad.lerp(momentum, beta) if nesterov else momentum

    original_shape = None
    if p_hat.ndim == 4:  # conv filters
        original_shape = p_hat.shape
        p_hat = p_hat.reshape(p_hat.size(0), -1)
        weight = weight.reshape(weight.size(0), -1)

    # Step 2: Proj_{T_U}(p̂) — project to tangent plane FIRST
    p_flat = p_hat.reshape(p_hat.size(0), -1)
    w_flat = weight.reshape(weight.size(0), -1).to(p_flat.dtype)
    w_norms = w_flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    radial = w_flat / w_norms
    p_radial = (p_flat * radial).sum(dim=-1, keepdim=True)
    p_tangent = p_flat - p_radial * radial

    # Step 3: Adaptive gain within tangent space
    tangent_gnorm = p_tangent.norm(dim=-1, keepdim=True)

    # Cold-start: gnorm_ema initialized to -1 (impossible for norms), detect first step
    if gnorm_ema.min() < 0:
        gnorm_ema.copy_(tangent_gnorm.to(gnorm_ema.dtype))
    else:
        gnorm_ema.lerp_(tangent_gnorm.to(gnorm_ema.dtype), 1 - gnorm_beta)

    # Scalar gain per neuron: (1-λ) + λ·ema/||p_tangent||
    # When tangent_gnorm > ema: gain < 1 (dampen spike)
    # When tangent_gnorm < ema: gain > 1 (boost quiet neuron)
    # When tangent_gnorm == ema: gain = 1 (no change)
    gain = (1 - lambda_reg) + lambda_reg * gnorm_ema / tangent_gnorm.clamp(min=1e-10)
    v = p_tangent * gain

    # Reshape back
    v = v.reshape(p_hat.shape)

    # Step 4: J† pullback — AOL warm start + NS5 refinement
    X = v.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # AOL diagonal preconditioning (warm start for NS5)
    A = X @ X.mT
    s = torch.rsqrt(A.abs().sum(dim=-1, keepdim=True) + 1e-7)
    X = X * s

    # NS5 refinement toward true orthogonal factor (no Frobenius renorm — AOL already scaled)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT

    update = X.to(p_hat.dtype)
    if original_shape is not None:
        update = update.reshape(original_shape)

    update *= max(1, p_hat.size(-2) / p_hat.size(-1))**0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMagMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed MagMuon+AdamW combo. Pass param_groups with use_muon=True/False.

    MagMuon: SI-EDP canonical flow optimizer.
      θ ← θ + η · J† · (Proj_{T_U}(p̂) − ∂R_T(u))

    MagMuon groups: params, lr, momentum, gnorm_beta, lambda_reg, weight_decay, use_muon
    Adam groups: params, lr, betas, eps, weight_decay, use_muon
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["gnorm_beta"] = group.get("gnorm_beta", 0.99)
                group["lambda_reg"] = group.get("lambda_reg", 1.0)
                group["weight_decay"] = group.get("weight_decay", 0)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue  # skip params with no gradient (don't decay momentum)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        # Initialize to -1 so first step is detected robustly
                        state["gnorm_ema"] = torch.full((p.size(0), 1), -1.0, device=p.device, dtype=p.dtype)
                    update = magmuon_update(p.grad, p.data, state["momentum_buffer"], state["gnorm_ema"],
                                            beta=group["momentum"], gnorm_beta=group["gnorm_beta"],
                                            lambda_reg=group.get("lambda_reg", 1.0),
                                            ns_steps=group.get("ns_steps", 5))
                    if group["weight_decay"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue  # skip params with no gradient
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
