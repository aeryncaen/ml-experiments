"""SpicyAdam: Adam with first moment but ratio-based adaptive LR replacing the second moment.

Standard Adam uses v = beta2*v + (1-beta2)*g^2 and divides by sqrt(v).
SpicyAdam replaces that with EMA/max ratio tracking of per-unit gradient norms.

For 1D params: per-element ratio.
For 2D+ params: per-row ratio.
"""
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import List, Tuple


class SpicyAdam(Optimizer):
    """Adam with ratio-based adaptive LR replacing the second moment.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        beta1: Momentum EMA beta (like Adam's beta1).
        gnorm_beta: EMA beta for gradient norm tracking (replaces beta2).
        weight_decay: Weight decay (decoupled, like AdamW).
        nesterov: Use Nesterov momentum extrapolation.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        beta1: float = 0.9,
        gnorm_beta: float = 0.55,
        weight_decay: float = 0.0,
        nesterov: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, beta1=beta1, gnorm_beta=gnorm_beta,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            gnorm_beta = group["gnorm_beta"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                g = p.grad.float()
                state = self.state[p]

                # Lazy init
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)
                    n_units = p.shape[0]
                    state["gnorm_ema"] = torch.zeros(n_units, device=p.device, dtype=torch.float32)
                    state["gnorm_max"] = torch.zeros(n_units, device=p.device, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]
                m = state["momentum"]
                g_ema = state["gnorm_ema"]
                g_max = state["gnorm_max"]

                # First moment: EMA momentum
                m.lerp_(g.to(dtype=m.dtype), 1.0 - beta1)

                # Update direction
                if nesterov:
                    update = torch.lerp(g, m.float(), beta1)
                else:
                    update = m.float()

                # Per-unit gradient norms for ratio tracking
                if p.ndim >= 2:
                    units = update.reshape(update.shape[0], -1).norm(dim=-1)
                else:
                    units = update.abs()

                # EMA / running-max ratio (replaces Adam's second moment)
                if step == 1:
                    signal_u = units
                    max_u = units
                else:
                    signal_u = gnorm_beta * g_ema + (1 - gnorm_beta) * units
                    max_u = torch.maximum(g_max, units)

                g_ema.copy_(signal_u.detach())
                g_max.copy_(max_u.detach())

                ratio_u = signal_u / max_u.clamp(min=1e-12)

                # Apply ratio-scaled update
                if p.ndim >= 2:
                    lr_shape = (ratio_u.shape[0],) + (1,) * (p.ndim - 1)
                    p.add_(-(update * (lr * ratio_u).view(lr_shape).to(dtype=update.dtype)))
                else:
                    p.add_(-(update * (lr * ratio_u).to(dtype=update.dtype)))

        return loss
