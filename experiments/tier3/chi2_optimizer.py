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
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            trust_radius = group["trust_radius"]
            beta2 = group["beta2"]
            beta1 = group["beta1"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            lr_scale = group["lr_scale"]

            # First pass: update metric states and compute denom = g^T G^{-1} g
            denom_sum = 0.0
            denom_count = 0
            tensors = []
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
                tensors.append((p, inv_metric_grad))

            if denom_sum <= 0.0 or denom_count == 0 or len(tensors) == 0:
                continue

            denom_mean = denom_sum / float(denom_count)
            scale = lr_scale * math.sqrt(trust_radius) / math.sqrt(denom_mean)

            # Second pass: apply normalized trust-region step
            for p, inv_metric_grad in tensors:
                p.add_(inv_metric_grad, alpha=-scale)

        return loss
