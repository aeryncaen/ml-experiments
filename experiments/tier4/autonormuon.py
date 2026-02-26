# AutoNorMuon optimizer
# NorMuon (zichongli5) + cosine LR ceiling
# + fast-EMA/max gradient norm tracking for adaptive LR attenuation
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


def normuon_update(
    grad,
    momentum,
    beta=0.95,
    ns_steps=5,
    nesterov=True,
) -> torch.Tensor:
    """Momentum + orthogonalization + aspect-ratio scaling.

    Returns the (pre-LR) update tensor.

    # TODO: second-moment hook point
    # Previously supported row_mean / matrix_mean_square / matrix_norm_square
    # second-moment modes with an EMA buffer.  Hard-disabled after ablation
    # showed no benefit in AutoNorMuon.  Re-add here if exploring new
    # second-moment strategies (e.g. per-neuron RMS, Fisher-style, …).
    """
    momentum.lerp_(grad, 1 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if original_shape is not None:
        update = update.reshape(original_shape)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def _blend_with_cosine(base_lr_t, cosine_factor, ratio_t):
    cosine_lr = base_lr_t * cosine_factor
    ema_lr = base_lr_t * ratio_t
    harmonic_lr = 2.0 * cosine_lr * ema_lr / (cosine_lr + ema_lr).clamp(min=1e-20)
    return torch.where(cosine_lr < ema_lr, harmonic_lr, ema_lr)


def _safe_item(x):
    if isinstance(x, torch.Tensor):
        return x.detach().item()
    return float(x)


def _flatten_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        return x.reshape(1, -1)
    if x.ndim == 2:
        return x
    return x.reshape(x.size(0), -1)


def _unit_norms(x: torch.Tensor, scope: str) -> torch.Tensor:
    rows = _flatten_rows(x)
    if scope == "matrix":
        return rows.norm().float().reshape(())
    if scope == "neuron":
        return rows.norm(dim=-1).float()
    raise ValueError(f"Unsupported adaptation_scope={scope}; expected 'matrix' | 'neuron'")


class AutoNorMuon(torch.optim.Optimizer):
    """
    AutoNorMuon: NorMuon+AdamW with:
      - Cosine LR ceiling (decays base_lr to 0 over total_steps)
      - Grad norm tracking: fast EMA / max → cosine-mapped attenuation
      - Optional row-normalization of weights or updates (Muon params only)

    The effective LR each step uses:
      cosine_lr = base_lr * 0.5 * (1 + cos(pi * k / total_steps))
      ratio     = fast_ema(gnorm) / max(gnorm)
      ema_lr    = base_lr * 0.5 * (1 + cos(pi * (1 - ratio)))
      lr        = harmonic(cosine_lr, ema_lr) if cosine < ema, else ema_lr

    NorMuon groups: params, lr, momentum, weight_decay, use_muon=True
    Adam groups: params, lr, betas, eps, weight_decay, use_muon=False

    Constructor args:
        total_steps: total training steps (required)
        beta: EMA decay for grad norm tracking (default 0.55)
        adaptation_scope: granularity for gnorm tracking & LR application:
            "neuron" (per output-neuron) | "matrix" (per weight matrix)
        retract: what to normalize to the product of spheres (row-norm = 1):
            "weights"          — normalize W rows after each step (nGPT-style)
            "grad_pre_ortho"   — normalize raw grad rows before momentum/NS5
            "grad_post_ortho"  — normalize post-ortho update rows after NS5
            "off"              — no normalization
            True               — same as "weights" (backward compat)
            False              — same as "off" (backward compat)
            Both grad modes normalize before gnorm tracking sees the norms.
        ratio_pow: exponent applied to adaptation ratio before LR scaling
        min_ratio: lower clamp for adaptation ratio
    """
    def __init__(self, param_groups, total_steps,
                 beta=0.55,
                 adaptation_scope="neuron",
                 retract=True,
                 ratio_pow=1.0,
                 min_ratio=0.0):
        if adaptation_scope not in ("neuron", "matrix"):
            raise ValueError(
                f"Unsupported adaptation_scope={adaptation_scope}; expected 'neuron' | 'matrix'"
            )
        # Normalize retract to string form
        if retract is True:
            retract = "weights"
        elif retract is False:
            retract = "off"
        if retract not in ("weights", "grad_pre_ortho", "grad_post_ortho", "off"):
            raise ValueError(
                f"Unsupported retract={retract!r}; expected 'weights' | 'grad_pre_ortho' | 'grad_post_ortho' | 'off' (or bool)"
            )

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
            group["k"] = 0
            group["scheduled_lr"] = 0.0
            group["gnorm_ema"] = None
            group["gnorm_max"] = None
            group["signal_ratio"] = None
            group["lr_mult"] = None

        self.total_steps = total_steps
        self.retract = retract
        self.gnorm_beta = beta
        self.adaptation_scope = adaptation_scope
        self.ratio_pow = ratio_pow
        self.min_ratio = min_ratio
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            is_compiling = torch.compiler.is_compiling()
        elif hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "is_compiling"):
            is_compiling = torch._dynamo.is_compiling()
        else:
            is_compiling = False

        for group in self.param_groups:
            decay = group["weight_decay"]
            k = group["k"]
            beta = self.gnorm_beta
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * k / self.total_steps))

            if group["use_muon"]:
                params = []
                grads = []
                states = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["gnorm_ema"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max_matrix"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max_neuron"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    params.append(p)
                    grads.append(p.grad)
                    states.append(state)

                if params:
                    # --- Compute updates via NorMuon (post-ortho) ---
                    updates = []
                    for p, state, g in zip(params, states, grads):
                        # Normalize raw grad rows before momentum/NS5
                        if self.retract == "grad_pre_ortho":
                            g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                        update = normuon_update(
                            g,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                        )
                        update = update.reshape(p.shape)

                        # Normalize post-ortho update rows after NS5
                        if self.retract == "grad_post_ortho":
                            update = update / update.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                        updates.append(update)

                    # --- Per-unit gradient norms (post-ortho, post-retract) ---
                    gnorm_units = [
                        _unit_norms(u, self.adaptation_scope) for u in updates
                    ]

                    flat_units = torch.cat([u.reshape(-1) for u in gnorm_units])
                    median_gnorm_t = flat_units.median()
                    mean_t = flat_units.mean()
                    std_t = flat_units.std(unbiased=False)

                    # --- Per-unit EMA / max tracking ---
                    ema_units = []
                    gmax_units = []
                    ratio_units = []
                    for state, units in zip(states, gnorm_units):
                        prev_ema = state.get("gnorm_ema")
                        if not isinstance(prev_ema, torch.Tensor) or prev_ema.shape != units.shape:
                            prev_ema = torch.zeros_like(units)
                        ema_u = units if k == 0 else (beta * prev_ema + (1 - beta) * units)

                        if self.adaptation_scope == "matrix":
                            prev_mat = state.get("gnorm_max_matrix")
                            if not isinstance(prev_mat, torch.Tensor) or prev_mat.numel() != 1:
                                prev_mat = units.new_zeros(())
                            cur_mat = units.max()
                            gmax_u = cur_mat if k == 0 else torch.maximum(prev_mat.to(cur_mat), cur_mat)
                            gmax_u_store = gmax_u
                            gmax_u = torch.ones_like(units) * gmax_u
                            state["gnorm_max_matrix"] = gmax_u_store.detach()
                        else:  # neuron
                            prev_neuron = state.get("gnorm_max_neuron")
                            if not isinstance(prev_neuron, torch.Tensor) or prev_neuron.shape != units.shape:
                                prev_neuron = torch.zeros_like(units)
                            gmax_u = units if k == 0 else torch.maximum(prev_neuron, units)
                            state["gnorm_max_neuron"] = gmax_u.detach()

                        ratio_u = (ema_u / gmax_u.clamp(min=1e-12)).clamp(min=0.0, max=1.0)

                        state["gnorm_ema"] = ema_u.detach()
                        state["gnorm_max"] = gmax_u.detach()

                        ema_units.append(ema_u)
                        gmax_units.append(gmax_u)
                        ratio_units.append(ratio_u)

                    flat_ema = torch.cat([u.reshape(-1) for u in ema_units])
                    flat_gmax = torch.cat([u.reshape(-1) for u in gmax_units])
                    flat_ratio = torch.cat([u.reshape(-1) for u in ratio_units])

                    ratio_gnorm_t = flat_ratio.median()

                    # --- Per-unit adaptive LR via cosine blend ---
                    lr_applies = []
                    if self.adaptation_scope == "neuron":
                        cosine_lr = flat_units.new_full((), group["lr"] * cosine_factor)
                        for ratio_u in ratio_units:
                            ema_factor_u = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - ratio_u)))
                            ema_lr_u = group["lr"] * ema_factor_u
                            harmonic_u = 2.0 * cosine_lr * ema_lr_u / (cosine_lr + ema_lr_u).clamp(min=1e-20)
                            lr_applies.append(torch.where(cosine_lr < ema_lr_u, harmonic_u, ema_lr_u))
                    else:  # matrix
                        ratio_vec = torch.stack([r.median() if r.ndim > 0 else r for r in ratio_units])
                        cosine_lr = ratio_vec.new_full((), group["lr"] * cosine_factor)
                        ema_factor = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - ratio_vec)))
                        ema_lr = group["lr"] * ema_factor
                        harmonic_lr = 2.0 * cosine_lr * ema_lr / (cosine_lr + ema_lr).clamp(min=1e-20)
                        lr_vec = torch.where(cosine_lr < ema_lr, harmonic_lr, ema_lr)
                        lr_applies = [lr_t for lr_t in lr_vec]

                    flat_lr = torch.cat([
                        (lr_t.reshape(-1) if isinstance(lr_t, torch.Tensor) and lr_t.ndim > 0 else torch.as_tensor(lr_t, device=flat_units.device, dtype=flat_units.dtype).reshape(1))
                        for lr_t in lr_applies
                    ])

                    signal_ratio_t = ratio_gnorm_t.clamp(min=self.min_ratio, max=1.0)
                    lr_mult_t = signal_ratio_t.pow(self.ratio_pow)

                    # --- Apply updates ---
                    for p, update, lr_t in zip(params, updates, lr_applies):
                        if isinstance(lr_t, torch.Tensor) and lr_t.ndim > 0:
                            lr_decay = lr_t.median()
                        else:
                            lr_decay = lr_t

                        if decay != 0:
                            p.mul_(1 - lr_decay * decay)

                        if isinstance(lr_t, torch.Tensor) and lr_t.ndim > 0:
                            view_shape = (lr_t.shape[0],) + (1,) * (update.ndim - 1)
                            update.mul_(lr_t.view(view_shape))
                        else:
                            update.mul_(lr_t)

                        p.sub_(update)

                        # Normalize weight rows to unit norm after step
                        if self.retract == "weights":
                            p.div_(p.norm(dim=-1, keepdim=True).clamp(min=1e-8))

                    # --- Group logging ---
                    group["gnorm_mean"] = mean_t.detach()
                    group["gnorm_std"] = std_t.detach()
                    group["signal_ratio"] = signal_ratio_t.detach()
                    group["lr_mult"] = lr_mult_t.detach()
                    group["ratio_gnorm"] = ratio_gnorm_t.detach()
                    if is_compiling:
                        group["gnorm_ratio"] = signal_ratio_t.detach()
                        group["gnorm_ratio_raw"] = signal_ratio_t.detach()
                        group["gnorm_median"] = median_gnorm_t.detach()
                        group["gnorm_ema"] = flat_ema.median().detach()
                        group["gnorm_max"] = flat_gmax.max().detach()
                        group["scheduled_lr"] = flat_lr.median().detach()
                    else:
                        group["gnorm_ratio"] = signal_ratio_t.item()
                        group["gnorm_ratio_raw"] = group["gnorm_ratio"]
                        group["gnorm_median"] = median_gnorm_t.item()
                        group["gnorm_ema"] = flat_ema.median().item()
                        group["gnorm_max"] = flat_gmax.max().item()
                        group["scheduled_lr"] = flat_lr.median().item()
                        group["gnorm_mean"] = _safe_item(mean_t)
                        group["gnorm_std"] = _safe_item(std_t)
                        group["signal_ratio"] = signal_ratio_t.item()
                        group["lr_mult"] = lr_mult_t.item()
                        group["ratio_gnorm"] = _safe_item(ratio_gnorm_t)

            else:
                # --- Adam: per-group fast EMA / max gnorm ---
                grads = [p.grad for p in group["params"] if p.grad is not None]
                if grads:
                    gnorms = torch.stack([g.norm().float() for g in grads])
                    median_gnorm_t = gnorms.median()
                    mean_t = gnorms.mean()
                    std_t = gnorms.std(unbiased=False)

                    if group["gnorm_ema"] is None or k == 0:
                        gnorm_ema_t = median_gnorm_t
                        gnorm_max_t = median_gnorm_t
                    else:
                        gnorm_ema_t = beta * group["gnorm_ema"] + (1 - beta) * median_gnorm_t
                        gnorm_max_t = torch.maximum(group["gnorm_max"], median_gnorm_t)
                    ratio_gnorm_t = (gnorm_ema_t / gnorm_max_t.clamp(min=1e-12)).clamp(min=0.0, max=1.0)

                    ratio_active = ratio_gnorm_t
                    cosine_lr_t = median_gnorm_t.new_full((), group["lr"] * cosine_factor)
                    ema_factor_t = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - ratio_active)))
                    ema_lr_t = group["lr"] * ema_factor_t
                    harmonic_t = 2.0 * cosine_lr_t * ema_lr_t / (cosine_lr_t + ema_lr_t).clamp(min=1e-20)
                    lr_t = torch.where(cosine_lr_t < ema_lr_t, harmonic_t, ema_lr_t)

                    signal_ratio_t = ratio_active.clamp(min=self.min_ratio, max=1.0)
                    lr_mult_t = signal_ratio_t.pow(self.ratio_pow)

                    group["gnorm_ema"] = gnorm_ema_t.detach()
                    group["gnorm_max"] = gnorm_max_t.detach()
                    group["gnorm_mean"] = mean_t.detach()
                    group["gnorm_std"] = std_t.detach()
                    group["signal_ratio"] = signal_ratio_t.detach()
                    group["lr_mult"] = lr_mult_t.detach()
                    group["ratio_gnorm"] = ratio_gnorm_t.detach()
                    if is_compiling:
                        group["gnorm_ratio"] = signal_ratio_t.detach()
                        group["gnorm_ratio_raw"] = signal_ratio_t.detach()
                        group["gnorm_median"] = median_gnorm_t.detach()
                        group["scheduled_lr"] = lr_t.detach()
                    else:
                        group["gnorm_ratio"] = _safe_item(signal_ratio_t)
                        group["gnorm_ratio_raw"] = group["gnorm_ratio"]
                        group["gnorm_median"] = _safe_item(median_gnorm_t)
                        group["scheduled_lr"] = _safe_item(lr_t)
                        group["gnorm_mean"] = _safe_item(mean_t)
                        group["gnorm_std"] = _safe_item(std_t)
                        group["signal_ratio"] = _safe_item(signal_ratio_t)
                        group["lr_mult"] = _safe_item(lr_mult_t)
                        group["ratio_gnorm"] = _safe_item(ratio_gnorm_t)
                    lr = lr_t
                else:
                    lr = group["scheduled_lr"] if group["scheduled_lr"] is not None else group["lr"]

                beta1, beta2_adam = group["betas"]
                eps = group["eps"]
                bias_correction1 = 1 - beta1 ** (k + 1)
                bias_correction2 = 1 - beta2_adam ** (k + 1)

                params = []
                exp_avgs = []
                exp_avg_sqs = []
                grads2 = []
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    params.append(p)
                    grads2.append(p.grad)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                use_foreach = (not is_compiling) and len(params) > 0 and params[0].device.type in ("cuda", "cpu")
                if use_foreach:
                    lr_val = float(lr)
                    torch._foreach_mul_(exp_avgs, beta1)
                    torch._foreach_add_(exp_avgs, grads2, alpha=1 - beta1)
                    torch._foreach_mul_(exp_avg_sqs, beta2_adam)
                    torch._foreach_addcmul_(exp_avg_sqs, grads2, grads2, value=1 - beta2_adam)

                    step_vals = torch._foreach_div(exp_avgs, bias_correction1)
                    denoms = torch._foreach_div(exp_avg_sqs, bias_correction2)
                    denoms = torch._foreach_sqrt(denoms)
                    torch._foreach_add_(denoms, eps)

                    if decay != 0:
                        torch._foreach_mul_(params, 1 - lr_val * decay)
                    torch._foreach_addcdiv_(params, step_vals, denoms, value=-lr_val)
                else:
                    for p, grad, exp_avg, exp_avg_sq in zip(params, grads2, exp_avgs, exp_avg_sqs):
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2_adam).addcmul_(grad, grad, value=1 - beta2_adam)

                        step_val = exp_avg.div(bias_correction1)
                        denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                        step_val = step_val.div(denom)
                        step_val.mul_(lr)

                        if decay != 0:
                            p.mul_(1 - lr * decay)
                        p.sub_(step_val)

            group["k"] = k + 1

        return loss
