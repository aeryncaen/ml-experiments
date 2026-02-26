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
    second_momentum,
    beta=0.95,
    beta2=0.95,
    ns_steps=5,
    nesterov=True,
    second_moment_mode="none",
    gnorm_scope="matrix",
) -> tuple[torch.Tensor, torch.Tensor]:
    momentum.lerp_(grad, 1 - beta)
    update = torch.lerp(grad, momentum, beta) if nesterov else momentum
    original_shape = None
    if update.ndim == 4:  # for the case of conv filters
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if original_shape is not None:
        update = update.reshape(original_shape)

    source_norms = _unit_norms(update, gnorm_scope)

    # NorMuon: per-row adaptive scaling with norm preservation
    if second_moment_mode != "none":
        vnorm = update.norm(dim=(-2, -1), keepdim=True)
        if second_moment_mode == "row_mean":
            v_mean = torch.mean(update * update, dim=-1, keepdim=True).to(second_momentum.dtype)
        elif second_moment_mode == "matrix_mean_square":
            msq = torch.mean(update * update).to(second_momentum.dtype)
            v_mean = torch.ones_like(second_momentum) * msq
        elif second_moment_mode == "matrix_norm_square":
            nsq = torch.sum(update * update).to(second_momentum.dtype)
            v_mean = torch.ones_like(second_momentum) * nsq
        else:
            raise ValueError(
                f"Unsupported second_moment_mode={second_moment_mode}; "
                "expected none | row_mean | matrix_mean_square | matrix_norm_square"
            )
        second_momentum.lerp_(v_mean, 1 - beta2)
        step_size = 1 / second_momentum.sqrt().add_(1e-10)
        update.mul_(step_size)
        vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
        update.mul_(vnorm / (vnorm_new.add_(1e-10)))
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update, source_norms


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
    if scope == "group":
        return rows.norm().float().reshape(())
    if scope == "matrix":
        return rows.norm().float().reshape(())
    if scope == "neuron":
        return rows.norm(dim=-1).float()
    raise ValueError(f"Unsupported gnorm_scope={scope}; expected 'group' | 'matrix' | 'neuron'")


def _matrix_rows_for_shape(shape: tuple[int, ...]) -> int:
    if len(shape) == 0:
        return 1
    n = 1
    for s in shape:
        n *= int(s)
    if n <= 1:
        return 1

    # Heuristic squarification for matrixified Adam params:
    # prefer fewer rows / longer channels, but avoid extremely skinny layouts.
    # We target rows near n^(1/3), with a tie-break toward fewer rows.
    target = max(2, int(round(float(n) ** (1.0 / 3.0))))

    cands = set()
    lim = int(math.sqrt(float(n)))
    for d in range(2, lim + 1):
        if n % d != 0:
            continue
        a = d
        b = n // d
        if a <= b:
            cands.add(a)
        if b <= a:
            cands.add(b)

    if not cands:
        return 1

    def _score(r: int) -> tuple[float, int]:
        return (abs(math.log((r + 1e-12) / float(target))), r)

    return min(cands, key=_score)


def _view_as_matrix(x: torch.Tensor, rows: int) -> torch.Tensor:
    return x.reshape(rows, -1)


def _restore_from_matrix(xm: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return xm.reshape(shape)


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
      lr        = harmonic(cosine_lr, ema_lr) if cosine < ema, else ema_lr

    NorMuon groups: params, lr, momentum, beta2, weight_decay, use_muon=True
    Adam groups: params, lr, betas, eps, weight_decay, use_muon=False

    Constructor args:
        total_steps: total training steps (required)
        retract: retract Muon params to product of spheres after each step (default True)
        gnorm_beta: EMA decay for grad norm tracking (default 0.9)
        adapt_mode: LR adaptation mode:
            gnorm | mu_var | hybrid | cv | surge
        ratio_pow: exponent applied to adaptation ratio before LR scaling
        min_ratio: lower clamp for adaptation ratio
        var_eps: epsilon used in mu/var ratio denominator
        conflict_proj: if True, project update to avoid update/grad conflict
        lr_scope: retained for compatibility; AutoNorMuon now hardcodes
            per-neuron LR application ("neuron")
        gnorm_source: Muon gnorm source for adaptation: "grad" | "post_ortho"
        gnorm_scope: retained for compatibility; AutoNorMuon now hardcodes
            neuron-scoped gnorm tracking ("neuron")
        gmax_scope: retained for compatibility; AutoNorMuon now hardcodes
            neuron-scoped gmax tracking ("neuron")
        second_moment_mode: retained for compatibility; AutoNorMuon now hard-disables
            NorMuon second-moment scaling and always uses "none"
        adam_matrixify: if True, non-scalar Adam params run through
            Muon-style orthogonalized update in matrix view + row retraction;
            reshape heuristic favors fewer longer channels over many short ones
        adam_second_moment_mode: second-moment handling for Adam-side params:
            adam | none
    """
    def __init__(self, param_groups, total_steps, retract=True,
                 gnorm_beta=0.9,
                 adapt_mode="gnorm",
                 ratio_pow=1.0,
                 min_ratio=0.0,
                 var_eps=1e-12,
                 conflict_proj=False,
                 lr_scope="neuron",
                 gnorm_source="grad",
                 gnorm_scope="neuron",
                 gmax_scope="neuron",
                 second_moment_mode="none",
                 adam_matrixify=True,
                 adam_second_moment_mode="adam"):
        valid_modes = ("gnorm", "mu_var", "hybrid", "cv", "surge")
        if adapt_mode not in valid_modes:
            raise ValueError(f"Unsupported adapt_mode={adapt_mode}; expected one of {valid_modes}")
        if lr_scope not in ("matrix", "group", "neuron"):
            raise ValueError(f"Unsupported lr_scope={lr_scope}; expected 'matrix' | 'group' | 'neuron'")
        if gnorm_source not in ("grad", "post_ortho"):
            raise ValueError(f"Unsupported gnorm_source={gnorm_source}; expected 'grad' or 'post_ortho'")
        if gnorm_scope not in ("group", "matrix", "neuron"):
            raise ValueError(f"Unsupported gnorm_scope={gnorm_scope}; expected 'group' | 'matrix' | 'neuron'")
        if gmax_scope not in ("group", "global", "matrix", "neuron"):
            raise ValueError(f"Unsupported gmax_scope={gmax_scope}; expected 'group' | 'matrix' | 'neuron' (or legacy 'global')")
        if not isinstance(adam_matrixify, bool):
            raise ValueError(f"Unsupported adam_matrixify={adam_matrixify}; expected bool")
        if adam_second_moment_mode not in ("adam", "none"):
            raise ValueError(
                f"Unsupported adam_second_moment_mode={adam_second_moment_mode}; expected 'adam' or 'none'"
            )

        # Hard-code neuron granularity across the full adaptation stack.
        lr_scope = "neuron"
        gnorm_scope = "neuron"
        gmax_scope = "neuron"
        # Hard-disable NorMuon second-moment scaling in AutoNorMuon.
        # Keep the argument for config/backward compatibility, but ignore it.
        second_moment_mode = "none"
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
            group["gnorm_ema"] = None
            group["gnorm_max"] = None
            group["mu2_ema"] = None
            group["var_ema"] = None
            group["signal_ratio"] = None
            group["lr_mult"] = None
            group["adapt_mode"] = adapt_mode
            group["lr_scope"] = lr_scope
            group["gnorm_source"] = gnorm_source
            group["gnorm_scope"] = gnorm_scope
            group["gmax_scope"] = gmax_scope
            group["second_moment_mode"] = second_moment_mode
            group["adam_matrixify"] = adam_matrixify
            group["adam_second_moment_mode"] = adam_second_moment_mode
            group["gnorm_mean"] = None
            group["gnorm_std"] = None
            group["gnorm_cv"] = None
            group["gnorm_cv_ema"] = None
            group["gnorm_mean_ema"] = None
            group["gnorm_surge"] = None
            group["ratio_gnorm"] = None
            group["ratio_muvar"] = None
            group["ratio_hybrid"] = None
            group["ratio_cv"] = None
            group["ratio_surge"] = None
            group["conflict_frac"] = None
        self.total_steps = total_steps
        self.retract = retract
        self.gnorm_beta = gnorm_beta
        self.adapt_mode = adapt_mode
        self.ratio_pow = ratio_pow
        self.min_ratio = min_ratio
        self.var_eps = var_eps
        self.conflict_proj = conflict_proj
        self.lr_scope = lr_scope
        self.gnorm_source = gnorm_source
        self.gnorm_scope = gnorm_scope
        self.gmax_scope = gmax_scope
        self.second_moment_mode = second_moment_mode
        self.adam_matrixify = adam_matrixify
        self.adam_second_moment_mode = adam_second_moment_mode
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
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        state["gnorm_ema"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max_matrix"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max_neuron"] = torch.zeros((), device=p.device, dtype=torch.float32)
                    params.append(p)
                    grads.append(p.grad)
                    states.append(state)

                if params:
                    updates = []
                    for p, state, g in zip(params, states, grads):
                        update, _ = normuon_update(
                            g,
                            state["momentum_buffer"],
                            state["second_momentum_buffer"],
                            beta=group["momentum"],
                            beta2=group["beta2"],
                            second_moment_mode=self.second_moment_mode,
                            gnorm_scope="matrix",
                        )

                        update = update.reshape(p.shape)
                        updates.append(update)

                    if self.gnorm_scope == "group":
                        source_tensors = updates if self.gnorm_source == "post_ortho" else grads
                        sq_sum = None
                        for t in source_tensors:
                            cur = (t.float() * t.float()).sum()
                            sq_sum = cur if sq_sum is None else (sq_sum + cur)
                        assert sq_sum is not None
                        group_norm = sq_sum.clamp(min=0.0).sqrt()
                        gnorm_units = [group_norm for _ in updates]
                    else:
                        gnorm_units = []
                        for update, g in zip(updates, grads):
                            if self.gnorm_source == "post_ortho":
                                gnorm_units.append(_unit_norms(update, self.gnorm_scope))
                            else:
                                gnorm_units.append(_unit_norms(g, self.gnorm_scope))

                    flat_units = torch.cat([u.reshape(-1) for u in gnorm_units])
                    median_gnorm_t = flat_units.median()
                    mean_t = flat_units.mean()
                    std_t = flat_units.std(unbiased=False)
                    cv_t = std_t / mean_t.clamp(min=1e-12)
                    if group["gnorm_mean_ema"] is None or k == 0:
                        mean_ema_t = mean_t
                    else:
                        mean_ema_t = beta * group["gnorm_mean_ema"] + (1 - beta) * mean_t
                    if group["gnorm_cv_ema"] is None or k == 0:
                        cv_ema_t = cv_t
                    else:
                        cv_ema_t = beta * group["gnorm_cv_ema"] + (1 - beta) * cv_t
                    surge_t = (mean_t - mean_ema_t).clamp(min=0.0) / mean_ema_t.clamp(min=1e-12)

                    mu2_t = mean_t * mean_t
                    second_t = (flat_units * flat_units).mean()
                    var_t = (second_t - mu2_t).clamp(min=0.0)
                    if group["mu2_ema"] is None or k == 0:
                        mu2_ema_t = mu2_t
                        var_ema_t = var_t
                    else:
                        mu2_ema_t = beta * group["mu2_ema"] + (1 - beta) * mu2_t
                        var_ema_t = beta * group["var_ema"] + (1 - beta) * var_t

                    gmax_global = flat_units.new_zeros(())
                    if self.gmax_scope == "group":
                        cur_global_max = flat_units.max()
                        if group["gnorm_max"] is None or k == 0:
                            gmax_global = cur_global_max
                        else:
                            gprev = torch.as_tensor(group["gnorm_max"], device=flat_units.device, dtype=flat_units.dtype)
                            gmax_global = torch.maximum(gprev, cur_global_max)

                    ema_units = []
                    gmax_units = []
                    ratio_units = []
                    for state, units in zip(states, gnorm_units):
                        gmax_state = units.new_zeros(())
                        prev_ema = state.get("gnorm_ema")
                        if not isinstance(prev_ema, torch.Tensor) or prev_ema.shape != units.shape:
                            prev_ema = torch.zeros_like(units)
                        ema_u = units if k == 0 else (beta * prev_ema + (1 - beta) * units)

                        if self.gmax_scope == "group":
                            gmax_u = torch.ones_like(units) * gmax_global
                            gmax_state = gmax_global
                        elif self.gmax_scope == "matrix":
                            prev_mat = state.get("gnorm_max_matrix")
                            if not isinstance(prev_mat, torch.Tensor) or prev_mat.numel() != 1:
                                prev_mat = units.new_zeros(())
                            cur_mat = units.max()
                            gmax_mat = cur_mat if k == 0 else torch.maximum(prev_mat.to(cur_mat), cur_mat)
                            gmax_u = torch.ones_like(units) * gmax_mat
                            state["gnorm_max_matrix"] = gmax_mat.detach()
                            gmax_state = gmax_mat
                        else:
                            prev_neuron = state.get("gnorm_max_neuron")
                            if not isinstance(prev_neuron, torch.Tensor) or prev_neuron.shape != units.shape:
                                prev_neuron = torch.zeros_like(units)
                            gmax_u = units if k == 0 else torch.maximum(prev_neuron, units)
                            state["gnorm_max_neuron"] = gmax_u.detach()
                            gmax_state = gmax_u

                        ratio_u = (ema_u / gmax_u.clamp(min=1e-12)).clamp(min=0.0, max=1.0)

                        state["gnorm_ema"] = ema_u.detach()
                        state["gnorm_max"] = gmax_state.detach()

                        ema_units.append(ema_u)
                        gmax_units.append(gmax_u)
                        ratio_units.append(ratio_u)

                    flat_ema = torch.cat([u.reshape(-1) for u in ema_units])
                    flat_gmax = torch.cat([u.reshape(-1) for u in gmax_units])
                    flat_ratio = torch.cat([u.reshape(-1) for u in ratio_units])

                    ratio_gnorm_t = flat_ratio.median()
                    ratio_muvar_t = (mu2_ema_t / (mu2_ema_t + var_ema_t + self.var_eps)).clamp(min=0.0, max=1.0)
                    ratio_cv_t = (1.0 / (1.0 + cv_ema_t)).clamp(min=0.0, max=1.0)
                    ratio_surge_t = (1.0 / (1.0 + surge_t)).clamp(min=0.0, max=1.0)
                    ratio_hybrid_t = torch.sqrt((ratio_gnorm_t * ratio_muvar_t).clamp(min=0.0, max=1.0))

                    lr_applies = []

                    if self.adapt_mode == "gnorm":
                        ratio_active = ratio_gnorm_t
                        if self.lr_scope == "matrix":
                            ratio_vec = torch.stack([r.median() for r in ratio_units])
                            cosine_lr = ratio_vec.new_full((), group["lr"] * cosine_factor)
                            ema_factor = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - ratio_vec)))
                            ema_lr = group["lr"] * ema_factor
                            harmonic_lr = 2.0 * cosine_lr * ema_lr / (cosine_lr + ema_lr).clamp(min=1e-20)
                            lr_vec = torch.where(cosine_lr < ema_lr, harmonic_lr, ema_lr)
                            lr_applies = [lr_t for lr_t in lr_vec]
                        elif self.lr_scope == "neuron":
                            cosine_lr = flat_units.new_full((), group["lr"] * cosine_factor)
                            for ratio_u in ratio_units:
                                ema_factor_u = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - ratio_u)))
                                ema_lr_u = group["lr"] * ema_factor_u
                                harmonic_u = 2.0 * cosine_lr * ema_lr_u / (cosine_lr + ema_lr_u).clamp(min=1e-20)
                                lr_applies.append(torch.where(cosine_lr < ema_lr_u, harmonic_u, ema_lr_u))
                        else:
                            lr_scalar = _blend_with_cosine(flat_units.new_full((), group["lr"]), cosine_factor, ratio_active)
                            lr_applies = [lr_scalar for _ in params]
                    else:
                        if self.adapt_mode == "mu_var":
                            ratio_active = ratio_muvar_t
                        elif self.adapt_mode == "hybrid":
                            ratio_active = ratio_hybrid_t
                        elif self.adapt_mode == "cv":
                            ratio_active = ratio_cv_t
                        elif self.adapt_mode == "surge":
                            ratio_active = ratio_surge_t
                        else:
                            ratio_active = ratio_gnorm_t
                        ratio_active = ratio_active.clamp(min=self.min_ratio, max=1.0)
                        lr_scalar = _blend_with_cosine(flat_units.new_full((), group["lr"]), cosine_factor, ratio_active)
                        if self.lr_scope == "neuron":
                            lr_applies = [torch.ones_like(u) * lr_scalar for u in gnorm_units]
                        else:
                            lr_applies = [lr_scalar for _ in params]

                    flat_lr = torch.cat([
                        (lr_t.reshape(-1) if isinstance(lr_t, torch.Tensor) and lr_t.ndim > 0 else torch.as_tensor(lr_t, device=flat_units.device, dtype=flat_units.dtype).reshape(1))
                        for lr_t in lr_applies
                    ])

                    signal_ratio_t = ratio_active.clamp(min=self.min_ratio, max=1.0)
                    lr_mult_t = signal_ratio_t.pow(self.ratio_pow)

                    conflict_hits = flat_units.new_zeros(())
                    conflict_total = 0
                    for p, g, update, lr_t in zip(params, grads, updates, lr_applies):

                        if self.conflict_proj:
                            dot = (update * g).sum()
                            denom = g.pow(2).sum().clamp(min=1e-12)
                            coeff = torch.minimum(dot / denom, torch.zeros_like(dot))
                            update = update - coeff * g
                            conflict_hits = conflict_hits + (dot < 0).to(conflict_hits.dtype)
                            conflict_total += 1

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

                        # Retract to product of spheres
                        if self.retract:
                            p.div_(p.norm(dim=-1, keepdim=True).clamp(min=1e-8))

                    # Group logging (median across Muon matrices)
                    conflict_frac_t = conflict_hits / max(1, conflict_total)
                    group["gnorm_mean"] = mean_t.detach()
                    group["gnorm_std"] = std_t.detach()
                    group["gnorm_cv"] = cv_t.detach()
                    group["gnorm_cv_ema"] = cv_ema_t.detach()
                    group["gnorm_mean_ema"] = mean_ema_t.detach()
                    group["gnorm_surge"] = surge_t.detach()
                    group["mu2_ema"] = mu2_ema_t.detach()
                    group["var_ema"] = var_ema_t.detach()
                    group["signal_ratio"] = signal_ratio_t.detach()
                    group["lr_mult"] = lr_mult_t.detach()
                    group["ratio_gnorm"] = ratio_gnorm_t.detach()
                    group["ratio_muvar"] = ratio_muvar_t.detach()
                    group["ratio_hybrid"] = ratio_hybrid_t.detach()
                    group["ratio_cv"] = ratio_cv_t.detach()
                    group["ratio_surge"] = ratio_surge_t.detach()
                    group["conflict_frac"] = conflict_frac_t.detach()
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
                        group["gnorm_cv"] = _safe_item(cv_t)
                        group["gnorm_cv_ema"] = _safe_item(cv_ema_t)
                        group["gnorm_mean_ema"] = _safe_item(mean_ema_t)
                        group["gnorm_surge"] = _safe_item(surge_t)
                        group["mu2_ema"] = mu2_ema_t.item()
                        group["var_ema"] = var_ema_t.item()
                        group["signal_ratio"] = signal_ratio_t.item()
                        group["lr_mult"] = lr_mult_t.item()
                        group["ratio_gnorm"] = _safe_item(ratio_gnorm_t)
                        group["ratio_muvar"] = _safe_item(ratio_muvar_t)
                        group["ratio_hybrid"] = _safe_item(ratio_hybrid_t)
                        group["ratio_cv"] = _safe_item(ratio_cv_t)
                        group["ratio_surge"] = _safe_item(ratio_surge_t)
                        group["conflict_frac"] = _safe_item(conflict_frac_t)

            else:
                # --- Adam component: scalar params stay Adam; matrix-like params can run Muon path ---
                beta1, beta2_adam = group["betas"]
                eps = group["eps"]
                adam2_mode = str(group.get("adam_second_moment_mode", self.adam_second_moment_mode))
                use_adam_second_moment = adam2_mode == "adam"
                use_matrixify = bool(group.get("adam_matrixify", self.adam_matrixify))

                matrix_entries = []
                scalar_entries = []
                all_entries = []

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "gnorm_ema_adam" not in state:
                        state["gnorm_ema_adam"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max_adam"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max_matrix_adam"] = torch.zeros((), device=p.device, dtype=torch.float32)
                        state["gnorm_max_neuron_adam"] = torch.zeros((), device=p.device, dtype=torch.float32)

                    is_matrix = use_matrixify and p.numel() > 1
                    if is_matrix:
                        rows = int(state.get("matrix_rows", _matrix_rows_for_shape(tuple(p.shape))))
                        if rows < 1:
                            rows = 1
                        if p.numel() % rows != 0:
                            rows = 1
                        state["matrix_rows"] = rows
                        grad_m = _view_as_matrix(p.grad, rows)
                        if "momentum_buffer_matrix" not in state or state["momentum_buffer_matrix"].shape != grad_m.shape:
                            state["momentum_buffer_matrix"] = torch.zeros_like(grad_m)
                            state["second_momentum_buffer_matrix"] = torch.zeros_like(grad_m[:, :1])
                        update_m, _ = normuon_update(
                            grad_m,
                            state["momentum_buffer_matrix"],
                            state["second_momentum_buffer_matrix"],
                            beta=beta1,
                            beta2=beta2_adam,
                            second_moment_mode="none",
                            gnorm_scope="matrix",
                        )
                        units = _unit_norms(update_m if self.gnorm_source == "post_ortho" else grad_m, self.gnorm_scope)
                        ent = {
                            "kind": "matrix",
                            "p": p,
                            "state": state,
                            "grad": grad_m,
                            "update": update_m,
                            "rows": rows,
                            "units": units,
                        }
                        matrix_entries.append(ent)
                        all_entries.append(ent)
                    else:
                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                        grad = p.grad
                        units = _unit_norms(grad, self.gnorm_scope)
                        ent = {
                            "kind": "scalar",
                            "p": p,
                            "state": state,
                            "grad": grad,
                            "units": units,
                        }
                        scalar_entries.append(ent)
                        all_entries.append(ent)

                if all_entries:
                    gnorm_units = [e["units"] for e in all_entries]
                    flat_units = torch.cat([u.reshape(-1) for u in gnorm_units])
                    median_gnorm_t = flat_units.median()
                    mean_t = flat_units.mean()
                    std_t = flat_units.std(unbiased=False)
                    cv_t = std_t / mean_t.clamp(min=1e-12)
                    if group["gnorm_mean_ema"] is None or k == 0:
                        mean_ema_t = mean_t
                    else:
                        mean_ema_t = beta * group["gnorm_mean_ema"] + (1 - beta) * mean_t
                    if group["gnorm_cv_ema"] is None or k == 0:
                        cv_ema_t = cv_t
                    else:
                        cv_ema_t = beta * group["gnorm_cv_ema"] + (1 - beta) * cv_t
                    surge_t = (mean_t - mean_ema_t).clamp(min=0.0) / mean_ema_t.clamp(min=1e-12)

                    mu2_t = mean_t * mean_t
                    second_t = (flat_units * flat_units).mean()
                    var_t = (second_t - mu2_t).clamp(min=0.0)
                    if group["mu2_ema"] is None or k == 0:
                        mu2_ema_t = mu2_t
                        var_ema_t = var_t
                    else:
                        mu2_ema_t = beta * group["mu2_ema"] + (1 - beta) * mu2_t
                        var_ema_t = beta * group["var_ema"] + (1 - beta) * var_t

                    gmax_global = flat_units.new_zeros(())
                    if self.gmax_scope == "group":
                        cur_global_max = flat_units.max()
                        if group["gnorm_max"] is None or k == 0:
                            gmax_global = cur_global_max
                        else:
                            gprev = torch.as_tensor(group["gnorm_max"], device=flat_units.device, dtype=flat_units.dtype)
                            gmax_global = torch.maximum(gprev, cur_global_max)

                    ema_units = []
                    gmax_units = []
                    ratio_units = []
                    for e in all_entries:
                        state = e["state"]
                        units = e["units"]
                        gmax_state = units.new_zeros(())
                        prev_ema = state.get("gnorm_ema_adam")
                        if not isinstance(prev_ema, torch.Tensor) or prev_ema.shape != units.shape:
                            prev_ema = torch.zeros_like(units)
                        ema_u = units if k == 0 else (beta * prev_ema + (1 - beta) * units)

                        if self.gmax_scope == "group":
                            gmax_u = torch.ones_like(units) * gmax_global
                            gmax_state = gmax_global
                        elif self.gmax_scope == "matrix":
                            prev_mat = state.get("gnorm_max_matrix_adam")
                            if not isinstance(prev_mat, torch.Tensor) or prev_mat.numel() != 1:
                                prev_mat = units.new_zeros(())
                            cur_mat = units.max()
                            gmax_mat = cur_mat if k == 0 else torch.maximum(prev_mat.to(cur_mat), cur_mat)
                            gmax_u = torch.ones_like(units) * gmax_mat
                            state["gnorm_max_matrix_adam"] = gmax_mat.detach()
                            gmax_state = gmax_mat
                        else:
                            prev_neuron = state.get("gnorm_max_neuron_adam")
                            if not isinstance(prev_neuron, torch.Tensor) or prev_neuron.shape != units.shape:
                                prev_neuron = torch.zeros_like(units)
                            gmax_u = units if k == 0 else torch.maximum(prev_neuron, units)
                            state["gnorm_max_neuron_adam"] = gmax_u.detach()
                            gmax_state = gmax_u

                        ratio_u = (ema_u / gmax_u.clamp(min=1e-12)).clamp(min=0.0, max=1.0)
                        state["gnorm_ema_adam"] = ema_u.detach()
                        state["gnorm_max_adam"] = gmax_state.detach()
                        ema_units.append(ema_u)
                        gmax_units.append(gmax_u)
                        ratio_units.append(ratio_u)

                    flat_ema = torch.cat([u.reshape(-1) for u in ema_units])
                    flat_gmax = torch.cat([u.reshape(-1) for u in gmax_units])
                    flat_ratio = torch.cat([u.reshape(-1) for u in ratio_units])

                    ratio_gnorm_t = flat_ratio.median()
                    ratio_muvar_t = (mu2_ema_t / (mu2_ema_t + var_ema_t + self.var_eps)).clamp(min=0.0, max=1.0)
                    ratio_cv_t = (1.0 / (1.0 + cv_ema_t)).clamp(min=0.0, max=1.0)
                    ratio_surge_t = (1.0 / (1.0 + surge_t)).clamp(min=0.0, max=1.0)
                    ratio_hybrid_t = torch.sqrt((ratio_gnorm_t * ratio_muvar_t).clamp(min=0.0, max=1.0))

                    if self.adapt_mode == "gnorm":
                        ratio_active = ratio_gnorm_t
                        cosine_lr_t = median_gnorm_t.new_full((), group["lr"] * cosine_factor)
                        ema_factor_t = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - ratio_active)))
                        ema_lr_t = group["lr"] * ema_factor_t
                        harmonic_t = 2.0 * cosine_lr_t * ema_lr_t / (cosine_lr_t + ema_lr_t).clamp(min=1e-20)
                        lr_t = torch.where(cosine_lr_t < ema_lr_t, harmonic_t, ema_lr_t)
                    else:
                        if self.adapt_mode == "mu_var":
                            ratio_active = ratio_muvar_t
                        elif self.adapt_mode == "hybrid":
                            ratio_active = ratio_hybrid_t
                        elif self.adapt_mode == "cv":
                            ratio_active = ratio_cv_t
                        elif self.adapt_mode == "surge":
                            ratio_active = ratio_surge_t
                        else:
                            ratio_active = ratio_gnorm_t
                        ratio_active = ratio_active.clamp(min=self.min_ratio, max=1.0)
                        lr_t = _blend_with_cosine(median_gnorm_t.new_full((), group["lr"]), cosine_factor, ratio_active)

                    signal_ratio_t = ratio_active.clamp(min=self.min_ratio, max=1.0)
                    lr_mult_t = signal_ratio_t.pow(self.ratio_pow)

                    group["gnorm_ema"] = flat_ema.median().detach()
                    group["gnorm_max"] = flat_gmax.max().detach()
                    group["gnorm_mean"] = mean_t.detach()
                    group["gnorm_std"] = std_t.detach()
                    group["gnorm_cv"] = cv_t.detach()
                    group["gnorm_cv_ema"] = cv_ema_t.detach()
                    group["gnorm_mean_ema"] = mean_ema_t.detach()
                    group["gnorm_surge"] = surge_t.detach()
                    group["mu2_ema"] = mu2_ema_t.detach()
                    group["var_ema"] = var_ema_t.detach()
                    group["signal_ratio"] = signal_ratio_t.detach()
                    group["lr_mult"] = lr_mult_t.detach()
                    group["ratio_gnorm"] = ratio_gnorm_t.detach()
                    group["ratio_muvar"] = ratio_muvar_t.detach()
                    group["ratio_hybrid"] = ratio_hybrid_t.detach()
                    group["ratio_cv"] = ratio_cv_t.detach()
                    group["ratio_surge"] = ratio_surge_t.detach()
                    group["conflict_frac"] = median_gnorm_t.new_zeros(())
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
                        group["gnorm_cv"] = _safe_item(cv_t)
                        group["gnorm_cv_ema"] = _safe_item(cv_ema_t)
                        group["gnorm_mean_ema"] = _safe_item(mean_ema_t)
                        group["gnorm_surge"] = _safe_item(surge_t)
                        group["mu2_ema"] = _safe_item(mu2_ema_t)
                        group["var_ema"] = _safe_item(var_ema_t)
                        group["signal_ratio"] = _safe_item(signal_ratio_t)
                        group["lr_mult"] = _safe_item(lr_mult_t)
                        group["ratio_gnorm"] = _safe_item(ratio_gnorm_t)
                        group["ratio_muvar"] = _safe_item(ratio_muvar_t)
                        group["ratio_hybrid"] = _safe_item(ratio_hybrid_t)
                        group["ratio_cv"] = _safe_item(ratio_cv_t)
                        group["ratio_surge"] = _safe_item(ratio_surge_t)
                    lr = lr_t
                else:
                    lr = group["scheduled_lr"] if group["scheduled_lr"] is not None else group["lr"]

                bias_correction1 = 1 - beta1 ** (k + 1)
                bias_correction2 = 1 - beta2_adam ** (k + 1) if use_adam_second_moment else 1.0
                conflict_hits = 0.0
                conflict_total = 0

                for e in matrix_entries:
                    p = e["p"]
                    grad_m = e["grad"]
                    update_m = e["update"]
                    rows = int(e["rows"])

                    if self.conflict_proj:
                        dot = (update_m * grad_m).sum()
                        denom_g = grad_m.pow(2).sum().clamp(min=1e-12)
                        coeff = torch.minimum(dot / denom_g, torch.zeros_like(dot))
                        update_m = update_m - coeff * grad_m
                        conflict_hits += float((dot < 0).item())
                        conflict_total += 1

                    if decay != 0:
                        p.mul_(1 - lr * decay)

                    p_m = _view_as_matrix(p, rows)
                    p_m.sub_(update_m * lr)
                    if self.retract:
                        p_m.div_(p_m.norm(dim=-1, keepdim=True).clamp(min=1e-8))

                for e in scalar_entries:
                    p = e["p"]
                    grad = e["grad"]
                    state = e["state"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    step_val = exp_avg.div(bias_correction1)

                    if use_adam_second_moment:
                        exp_avg_sq.mul_(beta2_adam).addcmul_(grad, grad, value=1 - beta2_adam)
                        denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                        step_val = step_val.div(denom)

                    if self.conflict_proj:
                        dot = (step_val * grad).sum()
                        denom_g = grad.pow(2).sum().clamp(min=1e-12)
                        coeff = torch.minimum(dot / denom_g, torch.zeros_like(dot))
                        step_val = step_val - coeff * grad
                        conflict_hits += float((dot < 0).item())
                        conflict_total += 1

                    step_val.mul_(lr)
                    if decay != 0:
                        p.mul_(1 - lr * decay)
                    p.sub_(step_val)

                if conflict_total > 0:
                    group["conflict_frac"] = conflict_hits / conflict_total

            group["k"] = k + 1

        return loss
