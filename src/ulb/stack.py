"""Stacking models: StackedULB and MoEStackedULB.

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
