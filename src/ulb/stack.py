"""Stacking models: StackedULB, MoEStackedULB, and PoolOfExperts.

StackedULB: simple pre-norm residual stacking.
    x = x + block(norm(x))  for each layer
    return final_norm(x)

MoEStackedULB: Mixture-of-Experts grid (n_experts x n_layers).
    Each layer has multiple expert blocks. A router picks top-k experts
    per sample, and the outputs are weighted-summed. Learned layer
    weighting blends all layer outputs + input before final norm.

    Two routing versions:
    - v1 (default): each layer routes itself — router picks experts from
      current layer's pre-normed input, blind weighting from router logits.
    - v2 (sender-routed): previous layer's output decides next layer's routing.
      Merge weights computed AFTER expert outputs via output_scorer + router_logits.

    Sample-level router supports 'topk' or 'relu' (ReMoE) mode.
    When router_mode='relu', adaptive L1 sparsity regularization controls
    computational cost. aux_loss is stored on the module after each forward.
"""

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class StackedULB(nn.Module):
    """Stack N identical layers with pre-norm residual connections.

    Args:
        make_layer: Callable that creates a single block (no arguments).
        n_layers:   Number of layers to stack.
        dim:        Model dimension (for RMSNorm).
    """

    def __init__(self, make_layer: Callable[[], nn.Module], n_layers: int, dim: int):
        super().__init__()
        self.layers = nn.ModuleList([make_layer() for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aux = 0.0
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
            # Collect aux_loss from blocks (e.g. sub-expert ReLU routing)
            block_aux = getattr(layer, 'aux_loss', 0.0)
            aux = aux + block_aux
        self.aux_loss = aux
        return self.final_norm(x)


class MoEStackedULB(nn.Module):
    """MoE grid: n_experts wide x n_layers tall.

    A non-routed stem layer runs first to build features from raw embeddings
    before any routing decisions are made.

    v1 (default): each layer routes itself.
        - Router applied to mean-pooled pre-normed hidden state.
        - Top-k experts selected per sample, softmax over selected logits.
        - All experts run on full batch (no sparse dispatch).
        - Weighted sum of top-k outputs + residual skip.
        - Learned layer weighting: softmax(layer_weights) blends all layer
          outputs + input before final norm.

    v2 (sender-routed): previous layer's output routes next layer's experts.
        - Routing comes from stem output (not raw embeddings).
        - Merge weights computed AFTER expert outputs via output_scorer + router_logits.
        - Same learned layer weighting.

    Args:
        make_layer:   Callable that creates a single block (no arguments).
        n_layers:     Number of layers.
        dim:          Model dimension.
        n_experts:    Number of experts per layer (default 4).
        top_k:        Top-k expert selection per sample (default 2).
        version:      Routing version: 1 or 2 (default 1).
        router_mode:  'topk' (default) or 'relu' (ReMoE differentiable routing).
        relu_lb:      Load-balanced L1 regularization (default True). Only for relu mode.
    """

    def __init__(self, make_layer: Callable[[], nn.Module], n_layers: int, dim: int,
                 n_experts: int = 4, top_k: int = 2, version: Literal[1, 2] = 1,
                 router_mode: Literal['topk', 'relu'] = 'relu', relu_lb: bool = True):
        super().__init__()
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.version = version
        self.router_mode = router_mode
        self.relu_lb = relu_lb

        # Stem: single non-routed layer to build features before routing
        self.stem_norm = RMSNorm(dim)
        self.stem_layer = make_layer()

        # Grid of experts: [layer][expert]
        self.experts = nn.ModuleList([
            nn.ModuleList([make_layer() for _ in range(n_experts)])
            for _ in range(n_layers)
        ])
        # Per-layer pre-norm
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)

        if version == 1:
            # v1: per-layer router decides own experts, blind weighting
            self.routers = nn.ModuleList([
                nn.Linear(dim, n_experts, bias=False) for _ in range(n_layers)
            ])
            self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))
        elif version == 2:
            # v2: sender-routed — layer l's output routes to layer l+1
            self.routers = nn.ModuleList([
                nn.Linear(dim, n_experts, bias=False) for _ in range(n_layers)
            ])
            # Per-layer merge scorer: looks at expert outputs to decide weights
            self.merge_scorers = nn.ModuleList([
                nn.Linear(dim, 1, bias=False) for _ in range(n_layers)
            ])
            self.layer_weights = nn.Parameter(torch.zeros(n_layers + 1))

        # ReLU routing state (ReMoE: adaptive L1 per layer)
        if router_mode == 'relu':
            self._target_sparsity = 1.0 - top_k / n_experts
            self._relu_alpha = 1.2
            for l in range(n_layers):
                self.register_buffer(f'_relu_lambda_{l}', torch.tensor(1e-8))

        # aux_loss is always available; 0.0 for topk mode
        self.aux_loss = 0.0

    def _collect_block_aux(self, experts: nn.ModuleList):
        """Sum aux_loss from all expert blocks in a layer."""
        aux = 0.0
        for e in experts:
            block_aux = getattr(e, 'aux_loss', 0.0)
            aux = aux + block_aux
        return aux

    def _topk_sample_route(self, logits: torch.Tensor):
        """TopK + Softmax sample-level routing. Returns (topk_idx, topk_weights)."""
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (B, top_k)
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B, top_k)
        return topk_idx, topk_weights, 0.0

    def _relu_sample_route(self, logits: torch.Tensor, relu_lambda: torch.Tensor):
        """ReLU sample-level routing (ReMoE). Returns (weights, aux_loss).

        Unlike topk which returns indices + weights for gather, relu returns
        full (B, n_experts) weight vector — zero entries mean inactive.
        """
        weights = F.relu(logits)  # (B, n_experts)

        aux_loss = 0.0
        if self.training:
            B, E = weights.shape
            with torch.no_grad():
                sparsity = (weights == 0).float().mean().item()
                sign = 1.0 if (self._target_sparsity - sparsity) > 0 else -1.0
                relu_lambda.mul_(self._relu_alpha ** sign)

            if self.relu_lb:
                with torch.no_grad():
                    active_counts = (weights > 0).float().sum(dim=0)  # (E,)
                    desired_ratio = self.top_k / self.n_experts
                    f_e = desired_ratio * B / active_counts.clamp(min=1)  # (E,)
                l_reg = (weights * f_e.unsqueeze(0)).mean()
            else:
                l_reg = weights.mean()

            aux_loss = relu_lambda.detach() * l_reg

        return weights, aux_loss

    def _v1_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Stem: build features before routing
        x = x + self.stem_layer(self.stem_norm(x))

        layer_outputs = [x]
        total_aux = 0.0

        for l, (norm, router, experts) in enumerate(
                zip(self.norms, self.routers, self.experts)):
            h = norm(x)
            h_pool = h.mean(dim=1)  # (B, D)
            logits = router(h_pool)  # (B, n_experts)

            # Run all experts on full batch
            expert_outs = torch.stack([e(h) for e in experts], dim=2)  # (B, T, n_experts, D)
            total_aux = total_aux + self._collect_block_aux(experts)

            if self.router_mode == 'relu':
                relu_lambda = getattr(self, f'_relu_lambda_{l}')
                weights, route_aux = self._relu_sample_route(logits, relu_lambda)
                total_aux = total_aux + route_aux
                # weights is (B, n_experts), multiply and sum
                out = (expert_outs * weights[:, None, :, None]).sum(dim=2)  # (B, T, D)
            else:
                topk_idx, topk_weights, _ = self._topk_sample_route(logits)
                idx_expanded = topk_idx[:, None, :, None].expand(-1, T, -1, D)
                selected = expert_outs.gather(2, idx_expanded)  # (B, T, top_k, D)
                out = (selected * topk_weights[:, None, :, None]).sum(dim=2)

            x = x + out
            layer_outputs.append(x)

        # Learned layer weighting
        w = F.softmax(self.layer_weights, dim=0)
        x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))
        self.aux_loss = total_aux
        return self.final_norm(x)

    def _v2_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Stem: build features before routing
        x = x + self.stem_layer(self.stem_norm(x))

        layer_outputs = [x]
        total_aux = 0.0

        # Initial routing decision from stem output
        route_signal = x.mean(dim=1)  # (B, D)

        for l, (norm, experts) in enumerate(zip(self.norms, self.experts)):
            h = norm(x)
            logits = self.routers[l](route_signal)  # (B, n_experts)

            # Run all experts, gather per sample
            expert_outs = torch.stack([e(h) for e in experts], dim=2)  # (B, T, n_experts, D)
            total_aux = total_aux + self._collect_block_aux(experts)

            if self.router_mode == 'relu':
                relu_lambda = getattr(self, f'_relu_lambda_{l}')
                weights, route_aux = self._relu_sample_route(logits, relu_lambda)
                total_aux = total_aux + route_aux

                # For v2, we still want merge scoring on top of relu weights.
                # Compute output-scorer contribution for non-zero experts.
                # weighted output first:
                out = (expert_outs * weights[:, None, :, None]).sum(dim=2)  # (B, T, D)
            else:
                topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
                idx_expanded = topk_idx[:, None, :, None].expand(-1, T, -1, D)
                selected = expert_outs.gather(2, idx_expanded)  # (B, T, top_k, D)

                # Merge: score each expert output AFTER computation
                selected_pooled = selected.mean(dim=1)  # (B, top_k, D)
                output_scores = self.merge_scorers[l](selected_pooled).squeeze(-1)  # (B, top_k)
                router_scores = topk_vals
                scores = output_scores + router_scores
                merge_weights = F.softmax(scores, dim=-1)  # (B, top_k)
                out = (selected * merge_weights[:, None, :, None]).sum(dim=2)

            x = x + out
            layer_outputs.append(x)
            route_signal = x.mean(dim=1)

        # Learned layer weighting
        w = F.softmax(self.layer_weights, dim=0)
        x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))
        self.aux_loss = total_aux
        return self.final_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.version == 2:
            return self._v2_forward(x)
        return self._v1_forward(x)


class PoolOfExperts(nn.Module):
    """Dynamic-depth expert pool with learned routing and exit.

    Instead of a fixed grid (n_experts x n_layers), there is a flat pool of
    N experts.  A non-routed stem layer builds features, then a stem router
    selects the first K experts.  Each expert has its own outbound router
    (Linear → pool_size+1 logits).  After running K experts, their outputs
    are weighted-merged (softmax over top-k logits) into one state (residual
    add), and their outbound logits are merged with the same weights.  The
    merged logits are then top-k'd to pick the next K experts.  The loop
    stops when exit appears in the top-K.  A non-routed exit layer produces
    the final output.

    Experts can be revisited (no exclusion mask).

    Args:
        make_layer:   Callable that creates a single block (no arguments).
        pool_size:    Number of experts in the pool.
        dim:          Model dimension.
        top_k:        Number of experts selected per hop (default 2).
        max_hops:     Loop bound on routing depth.  Default = 2 * pool_size.
        router_mode:  Exit slot density in the router space:
                      'squared' (default) — pool_size² options (pool_size expert + pool_size*(pool_size-1) exit).
                          Random pick has 1/pool_size chance of expert.
                      'single' — pool_size+1 options (pool_size expert + 1 exit).
                          Minimal exit pressure.
                      'half' — 2*pool_size options (pool_size expert + pool_size exit).
                          50/50 expert-vs-exit odds at random.
        router_noise: Gaussian noise scale added to logits during training
                      (default 1.0).  Annealable via .router_noise_scale.
        router_dropout: Enables random exit-logit dropout during training.
                      Drops a random number (0..75%) of exit logits per sample.
                      Expert logits are never dropped.
        block_shared_fraction: Fraction of expert block output dims shared across
                      all experts (0.0 = independent, 1.0 = fully shared).
        router_shared_fraction: Fraction of expert router output dims shared
                      across all expert routers (0.0 = independent, 1.0 = fully shared).
        shared_fraction: Convenience — sets both block and router fractions
                      when the individual ones are not specified.
    """

    # Index convention: router outputs logits over n_router_options.
    # Indices 0..pool_size-1 are experts, pool_size..n_router_options-1 are exit slots.
    # Exit slot count depends on router_mode.

    ROUTER_MODES = ('squared', 'single', 'half')

    def __init__(self, make_layer: Callable[[], nn.Module], pool_size: int, dim: int,
                 top_k: int = 2, max_hops: int | None = None,
                 router_mode: str = 'squared',
                 router_noise: float = 1.0, router_dropout: float = 0.0,
                 shared_fraction: float = 0.0,
                 block_shared_fraction: float | None = None,
                 router_shared_fraction: float | None = None,
                 hop_shared_fraction: float | None = None):
        super().__init__()
        if router_mode not in self.ROUTER_MODES:
            raise ValueError(f"router_mode must be one of {self.ROUTER_MODES}, got {router_mode!r}")
        self.pool_size = pool_size
        self.router_mode = router_mode
        if router_mode == 'squared':
            self.n_router_options = pool_size * pool_size
        elif router_mode == 'single':
            self.n_router_options = pool_size + 1
        elif router_mode == 'half':
            self.n_router_options = 2 * pool_size
        self.top_k = top_k
        self.max_hops = max_hops if max_hops is not None else 2 * pool_size
        self.router_noise_scale = router_noise  # settable for annealing
        self.router_dropout = router_dropout
        self.exit_ramp_scale = 3.0

        # Stem: non-routed entry layer
        self.stem_norm = RMSNorm(dim)
        self.stem_layer = make_layer()

        # Stem router: kicks off the first hop
        self.stem_router = nn.Linear(dim, self.n_router_options, bias=False)

        # Expert pool
        self.experts = nn.ModuleList([make_layer() for _ in range(pool_size)])

        # Per-expert outbound router: each expert votes on next destination
        self.expert_routers = nn.ModuleList([
            nn.Linear(dim, self.n_router_options, bias=False) for _ in range(pool_size)
        ])

        # Per-hop pre-norm (shared across hops)
        self.hop_norm = RMSNorm(dim)

        # Per-expert hop conditioning: content-gated hop embedding
        # Each expert has a (max_hops, dim) embedding and a small gate projection.
        # gate = sigmoid(linear(x[..., :hop_gate_dim]))  →  (B, T, 1)
        # conditioning = gate * hop_embed[hop]            →  (B, T, dim)
        hop_gate_dim = min(dim, 12)
        self.hop_gate_dim = hop_gate_dim
        from .shared import ParamHolder
        self.hop_embeds = nn.ModuleList([
            ParamHolder(torch.randn(self.max_hops, dim) * 0.02) for _ in range(pool_size)
        ])
        self.hop_gates = nn.ModuleList([
            nn.Linear(hop_gate_dim, 1, bias=False) for _ in range(pool_size)
        ])

        # Exit: non-routed output layer
        self.exit_norm = RMSNorm(dim)
        self.exit_layer = make_layer()

        self.final_norm = RMSNorm(dim)

        # Weight sharing across experts
        _block_frac = block_shared_fraction if block_shared_fraction is not None else shared_fraction
        _router_frac = router_shared_fraction if router_shared_fraction is not None else shared_fraction
        self.block_shared_fraction = _block_frac
        self.router_shared_fraction = _router_frac
        from .shared import share_expert_weights, share_linear_list, share_parameter_list
        self._shared_block_params = (
            share_expert_weights(self.experts, _block_frac) if _block_frac > 0.0
            else nn.ParameterDict()
        )
        self._shared_router_params = (
            share_linear_list(self.expert_routers, _router_frac) if _router_frac > 0.0
            else nn.ParameterDict()
        )
        _hop_frac = hop_shared_fraction if hop_shared_fraction is not None else shared_fraction
        self.hop_shared_fraction = _hop_frac
        self._shared_hop_embed_params = (
            share_parameter_list(self.hop_embeds, _hop_frac, prefix='hop_embed') if _hop_frac > 0.0
            else nn.ParameterDict()
        )
        self._shared_hop_gate_params = (
            share_linear_list(self.hop_gates, _hop_frac, prefix='hop_gate') if _hop_frac > 0.0
            else nn.ParameterDict()
        )

        self.aux_loss = 0.0
        self.trace = False  # set True to record per-sample routing decisions
        self.last_trace = None  # populated when trace=True

    def _perturb_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply noise and dropout to router logits during training."""
        if not self.training:
            return logits
        # Gaussian noise
        if self.router_noise_scale > 0:
            logits = logits + self.router_noise_scale * torch.randn_like(logits)
        # Dropout: drop a random number of exit logits (0..n_exit), never touch expert logits
        if self.router_dropout > 0:
            n_exit = self.n_router_options - self.pool_size
            max_drop = int(n_exit * 0.25)
            n_drop = torch.randint(0, max_drop + 1, (logits.shape[0],), device=logits.device)
            # Per-sample random permutation of exit slot indices
            rand = torch.rand(logits.shape[0], n_exit, device=logits.device)
            exit_rank = rand.argsort(dim=-1)  # (B, n_exit)
            exit_mask = exit_rank < n_drop[:, None]  # (B, n_exit)
            logits[:, self.pool_size:] = logits[:, self.pool_size:].masked_fill(exit_mask, float('-inf'))
        return logits

    def stem(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run stem layer and produce initial router logits.

        Args:
            x: Input tensor (B, T, D).

        Returns:
            x: Stem-processed tensor (B, T, D).
            logits: Perturbed stem router logits (B, n_router_options).
        """
        x = x + self.stem_layer(self.stem_norm(x))
        stem_pool = x.mean(dim=1)  # (B, D)
        logits = self._perturb_logits(self.stem_router(stem_pool))
        return x, logits

    def route(self, logits: torch.Tensor, hop: int
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply exit ramp, top-k selection, and exit detection.

        Per-sample exit handling: if a sample picks exit in some but not all
        top-k slots, the exit slots get zero weight and the real expert slots
        are renormalized to sum to 1. The sample still runs its real experts
        this hop. Only when ALL top-k slots are exit does the sample fully
        exit (gets zero output).

        Args:
            logits: Router logits (B, n_router_options).
            hop: Current hop index (0-based).

        Returns:
            topk_idx: Selected indices (B, top_k).
            topk_weights: Softmax weights over selected (B, top_k).
            has_exit: Boolean per sample — True if ALL selected indices are exit (B,).
        """
        exit_bias = self.exit_ramp_scale * (hop / self.max_hops)
        if exit_bias > 0:
            bias = torch.zeros_like(logits)
            bias[:, self.pool_size:] = exit_bias
            logits = logits + bias

        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)

        # Zero out exit slot weights, renormalize real expert weights
        is_exit = topk_idx >= self.pool_size  # (B, top_k)
        topk_weights = topk_weights.masked_fill(is_exit, 0.0)
        weight_sum = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        topk_weights = topk_weights / weight_sum

        # Full exit: ALL top-k slots are exit
        has_exit = is_exit.all(dim=-1)
        return topk_idx, topk_weights, has_exit

    def execute_hop(self, x: torch.Tensor, topk_idx: torch.Tensor,
                    topk_weights: torch.Tensor, hop: int = 0
                    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Run selected experts and produce weighted-merged output and next logits.

        Args:
            x: Current hidden state (B, T, D).
            topk_idx: Selected expert/exit indices (B, top_k).
            topk_weights: Softmax weights (B, top_k).
            hop: Current hop index (0-based). Subclasses can use for conditioning.

        Returns:
            out: Weighted-merged expert output (B, T, D) — added to x by caller.
            next_logits: Perturbed merged outbound router logits (B, n_router_options).
            hop_aux: Accumulated aux loss from experts this hop.
        """
        B = x.shape[0]
        h = self.hop_norm(x)

        # Find which experts are active this hop (indices < pool_size)
        active_eids = topk_idx.unique()
        active_eids = active_eids[active_eids < self.pool_size]

        # Run active experts on full batch, build lookup
        hop_aux = 0.0
        expert_outs = {}   # eid -> (B, T, D)
        expert_logits = {} # eid -> (B, n_router_options)
        for eid in active_eids.tolist():
            # Content-gated hop conditioning: gate(x_slice) * hop_embed[hop]
            gate = torch.sigmoid(self.hop_gates[eid](h[..., :self.hop_gate_dim]))  # (B, T, 1)
            hop_cond = gate * self.hop_embeds[eid][hop]  # (B, T, dim)
            e_out = self.experts[eid](h + hop_cond)  # (B, T, D)
            expert_outs[eid] = e_out
            e_pool = e_out.mean(dim=1)  # (B, D)
            expert_logits[eid] = self.expert_routers[eid](e_pool)

            block_aux = getattr(self.experts[eid], 'aux_loss', 0.0)
            hop_aux = hop_aux + block_aux

        # Weighted merge
        out = torch.zeros_like(x)
        next_logits = torch.zeros(B, self.n_router_options, device=x.device, dtype=x.dtype)

        for k_idx in range(self.top_k):
            w = topk_weights[:, k_idx]
            eids = topk_idx[:, k_idx]
            for eid in active_eids.tolist():
                mask = eids == eid
                if not mask.any():
                    continue
                out = out + (mask[:, None, None].float() * w[:, None, None]) * expert_outs[eid]
                next_logits = next_logits + (mask[:, None].float() * w[:, None]) * expert_logits[eid]

        next_logits = self._perturb_logits(next_logits)
        return out, next_logits, hop_aux

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        """Run exit layer and final norm.

        Args:
            x: Hidden state after routing loop (B, T, D).

        Returns:
            Normalized output (B, T, D).
        """
        x = x + self.exit_layer(self.exit_norm(x))
        return self.final_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Stem
        x, logits = self.stem(x)

        total_aux = 0.0
        non_exit_decisions = 0
        trace_hops: list[dict] = []

        for hop in range(self.max_hops):
            topk_idx, topk_weights, has_exit = self.route(logits, hop)

            if self.trace:
                trace_hops.append({
                    'topk_idx': topk_idx.detach().cpu(),
                    'topk_weights': topk_weights.detach().cpu(),
                    'has_exit': has_exit.detach().cpu(),
                })

            non_exit_decisions += (~has_exit).sum()

            if has_exit.all():
                break

            out, logits, hop_aux = self.execute_hop(x, topk_idx, topk_weights, hop)
            x = x + out
            total_aux = total_aux + hop_aux

        # Exit
        x = self.finalize(x)

        self.aux_loss = total_aux
        self.last_mean_hops = non_exit_decisions / B
        if self.trace:
            self.last_trace = trace_hops
        return x
