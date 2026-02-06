"""
Chi2 trust-region optimizer for Tier 3 experiments.

Implements updates of the form:
  d_t = G_t^{-1} g_t
  step = sqrt(rho) * d_t / sqrt(g_t^T d_t)

with a diagonal metric estimate G_t from EMA of squared gradients.
"""

from __future__ import annotations

import math
import torch
from torch.optim import Optimizer


class Chi2TrustRegion(Optimizer):
    def __init__(
        self,
        params,
        trust_radius: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_scale: float = 1.0,
        max_step_l2: float = 1.0,
    ):
        if trust_radius <= 0:
            raise ValueError("trust_radius must be > 0")
        if not (0.0 < beta2 < 1.0):
            raise ValueError("beta2 must be in (0, 1)")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError("beta1 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        defaults = dict(
            trust_radius=trust_radius,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            max_step_l2=max_step_l2,
        )
        super().__init__(params, defaults)
        self.last_stats = {
            "denom_mean": 0.0,
            "scale": 0.0,
            "n_tensors": 0,
            "n_params": 0,
            "step_l2": 0.0,
            "step_linf": 0.0,
            "grad_l2": 0.0,
            "precond_grad_l2": 0.0,
            "pred_linear": 0.0,
            "groups": [],
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        global_step_sq = 0.0
        global_step_linf = 0.0
        global_grad_sq = 0.0
        global_pre_sq = 0.0
        global_denom = 0.0
        global_pred = 0.0
        global_params = 0
        global_tensors = 0
        group_stats = []

        for gi, group in enumerate(self.param_groups):
            trust_radius = group["trust_radius"]
            beta2 = group["beta2"]
            beta1 = group["beta1"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            lr_scale = group["lr_scale"]
            max_step_l2 = group["max_step_l2"]
            group_name = group.get("name", f"group_{gi}")

            # First pass: update metric states and compute denom = g^T G^{-1} g
            denom_sum = 0.0
            denom_count = 0
            tensors = []
            grad_sq_sum = 0.0
            pre_sq_sum = 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Chi2TrustRegion does not support sparse gradients")

                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                step_t = state["step"]
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                if beta1 > 0:
                    m_hat = m / (1.0 - beta1 ** step_t)
                else:
                    m_hat = m
                v_hat = v / (1.0 - beta2 ** step_t)

                inv_metric_grad = m_hat / (torch.sqrt(v_hat) + eps)
                denom_sum += torch.sum(m_hat * inv_metric_grad).item()
                denom_count += g.numel()
                grad_sq_sum += torch.sum(g * g).item()
                pre_sq_sum += torch.sum(inv_metric_grad * inv_metric_grad).item()
                tensors.append((p, inv_metric_grad))

            if denom_sum <= 0.0 or denom_count == 0 or len(tensors) == 0:
                continue

            scale = lr_scale * math.sqrt(trust_radius) / math.sqrt(denom_sum)

            # Optional global step-norm clipping for stability.
            pre_l2 = math.sqrt(max(pre_sq_sum, 0.0))
            if pre_l2 > 0:
                step_l2_est = scale * pre_l2
                if step_l2_est > max_step_l2:
                    scale = scale * (max_step_l2 / step_l2_est)

            # Second pass: apply normalized trust-region step
            step_sq_sum = 0.0
            step_linf = 0.0
            for p, inv_metric_grad in tensors:
                delta = -scale * inv_metric_grad
                p.add_(delta)
                step_sq_sum += torch.sum(delta * delta).item()
                step_linf = max(step_linf, torch.max(torch.abs(delta)).item())

            group_pred = float(max(scale * denom_sum, 0.0))
            global_step_sq += step_sq_sum
            global_step_linf = max(global_step_linf, step_linf)
            global_grad_sq += grad_sq_sum
            global_pre_sq += pre_sq_sum
            global_denom += denom_sum
            global_pred += group_pred
            global_params += denom_count
            global_tensors += len(tensors)
            group_stats.append(
                {
                    "index": int(gi),
                    "name": str(group_name),
                    "trust_radius": float(trust_radius),
                    "denom_mean": float(denom_sum),
                    "scale": float(scale),
                    "n_tensors": int(len(tensors)),
                    "n_params": int(denom_count),
                    "step_l2": float(math.sqrt(max(step_sq_sum, 0.0))),
                    "step_linf": float(step_linf),
                    "grad_l2": float(math.sqrt(max(grad_sq_sum, 0.0))),
                    "precond_grad_l2": float(math.sqrt(max(pre_sq_sum, 0.0))),
                    "pred_linear": group_pred,
                }
            )

        self.last_stats = {
            "denom_mean": float(global_denom),
            "scale": float(sum(gs["scale"] for gs in group_stats) / max(len(group_stats), 1)),
            "n_tensors": int(global_tensors),
            "n_params": int(global_params),
            "step_l2": float(math.sqrt(max(global_step_sq, 0.0))),
            "step_linf": float(global_step_linf),
            "grad_l2": float(math.sqrt(max(global_grad_sq, 0.0))),
            "precond_grad_l2": float(math.sqrt(max(global_pre_sq, 0.0))),
            "pred_linear": float(global_pred),
            "groups": group_stats,
        }

        return loss
