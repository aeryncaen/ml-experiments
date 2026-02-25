# AutoNorMuon optimizer
# NorMuon (zichongli5) + cosine LR ceiling
# + dual-EMA gradient norm tracking for adaptive LR attenuation
# + row retraction to the product of spheres.
import math
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if original_shape is not None:
        update = update.reshape(original_shape)
    # NorMuon: per-row adaptive scaling with norm preservation
    vnorm = update.norm(dim=(-2,-1), keepdim=True)
    v_mean = torch.mean(update * update, dim=-1, keepdim=True).to(second_momentum.dtype)
    second_momentum.lerp_(v_mean, 1 - beta2)
    step_size = 1 / second_momentum.sqrt().add_(1e-10)
    update.mul_(step_size)
    vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
    update.mul_(vnorm / (vnorm_new.add_(1e-10)))
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class AutoNorMuon(torch.optim.Optimizer):
    """
    AutoNorMuon: NorMuon+AdamW with:
      - Cosine LR ceiling (decays base_lr to 0 over total_steps)
      - Grad norm tracking: fast EMA / max → cosine-mapped attenuation
      - Row retraction to the product of spheres (Muon params only)

    The effective LR each step uses:
      cosine_lr = base_lr * 0.5 * (1 + cos(pi * k / total_steps))
      ratio     = fast_ema(gnorm) / max(gnorm)
      ema_lr    = base_lr * 0.5 * (1 + cos(pi * (1 - ratio)))
      lr        = geomean(cosine_lr, ema_lr) if cosine < ema, else ema_lr

    NorMuon groups: params, lr, momentum, beta2, weight_decay, use_muon=True
    Adam groups: params, lr, betas, eps, weight_decay, use_muon=False

    Constructor args:
        total_steps: total training steps (required)
        retract: retract Muon params to product of spheres after each step (default True)
        gnorm_beta: EMA decay for grad norm tracking (default 0.99)
    """
    def __init__(self, param_groups, total_steps, retract=True,
                 gnorm_beta=0.9):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
            group["k"] = 0
            group["scheduled_lr"] = 0.0
            group["gnorm_ema"] = 0.0
            group["gnorm_max"] = 0.0
        self.total_steps = total_steps
        self.retract = retract
        self.gnorm_beta = gnorm_beta
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            decay = group["weight_decay"]
            k = group["k"]
            beta = self.gnorm_beta
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * k / self.total_steps))

            if group["use_muon"]:
                # --- Per-matrix: fast EMA / max gnorm ---
                ratios = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        state["gnorm_ema"] = 0.0
                        state["gnorm_max"] = 0.0

                    gnorm = p.grad.norm().item()
                    if k == 0:
                        state["gnorm_ema"] = gnorm
                    else:
                        state["gnorm_ema"] = beta * state["gnorm_ema"] + (1 - beta) * gnorm
                    state["gnorm_max"] = max(state["gnorm_max"], gnorm)
                    ratio = state["gnorm_ema"] / state["gnorm_max"] if state["gnorm_max"] > 0 else 1.0
                    ratios.append(ratio)

                    # Per-matrix LR: ratio mapped through cosine curve
                    cosine_lr = group["lr"] * cosine_factor
                    ema_factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - ratio)))
                    ema_lr = group["lr"] * ema_factor
                    if cosine_lr < ema_lr:
                        lr = math.sqrt(cosine_lr * ema_lr)
                    else:
                        lr = ema_lr

                    update = normuon_update(p.grad, state["momentum_buffer"],
                                            state["second_momentum_buffer"],
                                            beta=group["momentum"],
                                            beta2=group["beta2"])
                    update = update.reshape(p.shape)

                    if decay != 0:
                        p.mul_(1 - lr * decay)
                    p.add_(update, alpha=-lr)

                    # Retract to product of spheres
                    if self.retract:
                        p.div_(p.norm(dim=-1, keepdim=True).clamp(min=1e-8))

                # Log median ratio for the group (for CSV/plotting)
                if ratios:
                    ratios.sort()
                    n = len(ratios)
                    med_ratio = ratios[n // 2] if n % 2 else (ratios[n // 2 - 1] + ratios[n // 2]) / 2
                    group["gnorm_ratio"] = med_ratio
                    group["gnorm_ratio_raw"] = med_ratio
                    group["gnorm_median"] = 0.0
                    group["gnorm_ema"] = 0.0
                # Log the group-level effective LR (use median ratio)
                med_ratio = group.get("gnorm_ratio", 1.0)
                cosine_lr = group["lr"] * cosine_factor
                ema_factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - med_ratio)))
                ema_lr = group["lr"] * ema_factor
                if cosine_lr < ema_lr:
                    group["scheduled_lr"] = math.sqrt(cosine_lr * ema_lr)
                else:
                    group["scheduled_lr"] = ema_lr

            else:
                # --- Adam: per-group fast EMA / max gnorm ---
                gnorms = []
                for p in group["params"]:
                    if p.grad is not None:
                        gnorms.append(p.grad.norm().item())
                if gnorms:
                    gnorms.sort()
                    n = len(gnorms)
                    median_gnorm = gnorms[n // 2] if n % 2 else (gnorms[n // 2 - 1] + gnorms[n // 2]) / 2
                    if k == 0:
                        group["gnorm_ema"] = median_gnorm
                    else:
                        group["gnorm_ema"] = beta * group["gnorm_ema"] + (1 - beta) * median_gnorm
                    group["gnorm_max"] = max(group["gnorm_max"], median_gnorm)
                    ratio = group["gnorm_ema"] / group["gnorm_max"] if group["gnorm_max"] > 0 else 1.0
                    group["gnorm_ratio"] = ratio
                    group["gnorm_ratio_raw"] = ratio
                    group["gnorm_median"] = median_gnorm

                cosine_lr = group["lr"] * cosine_factor
                ratio = group.get("gnorm_ratio", 1.0)
                ema_factor = 0.5 * (1.0 + math.cos(math.pi * (1.0 - ratio)))
                ema_lr = group["lr"] * ema_factor
                if cosine_lr < ema_lr:
                    lr = math.sqrt(cosine_lr * ema_lr)
                else:
                    lr = ema_lr
                group["scheduled_lr"] = lr

                beta1, beta2_adam = group["betas"]
                eps = group["eps"]
                bias_correction1 = 1 - beta1 ** (k + 1)
                bias_correction2 = 1 - beta2_adam ** (k + 1)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2_adam).addcmul_(grad, grad, value=1 - beta2_adam)

                    step_val = exp_avg.div(bias_correction1)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    if decay != 0:
                        p.mul_(1 - lr * decay)
                    p.addcdiv_(step_val, denom, value=-lr)

            group["k"] = k + 1

        return loss
