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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return self.final_norm(x)


class MoEStackedULB(nn.Module):
    """MoE grid: n_experts wide x n_layers tall.

    v1 (default): each layer routes itself per-position.
        - Router applied to pre-normed hidden state at each position.
        - Top-k experts selected per (sample, position), softmax over selected logits.
        - All experts run on full batch (no sparse dispatch).
        - Weighted sum of top-k outputs + residual skip.
        - Learned layer weighting: softmax(layer_weights) blends all layer
          outputs + input before final norm.

    v2 (sender-routed): previous layer's output routes next layer's experts,
    also per-position.
        - Layer 0 routing comes from input content at each position.
        - Merge weights from output_scorer(expert_output) + router_logits, per-position.
        - Same learned layer weighting.

    Args:
        make_layer: Callable that creates a single block (no arguments).
        n_layers:   Number of layers.
        dim:        Model dimension.
        n_experts:  Number of experts per layer (default 4).
        top_k:      Top-k expert selection per sample (default 2).
        version:    Routing version: 1 or 2 (default 1).
    """

    def __init__(self, make_layer: Callable[[], nn.Module], n_layers: int, dim: int,
                 n_experts: int = 4, top_k: int = 2, version: Literal[1, 2] = 1):
        super().__init__()
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.version = version

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

    def _v1_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        layer_outputs = [x]

        for norm, router, experts in zip(self.norms, self.routers, self.experts):
            h = norm(x)
            # Causal-compatible routing: per-position expert choice
            logits = router(h)  # (B, T, n_experts)
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (B, T, top_k)
            topk_weights = F.softmax(topk_vals, dim=-1)  # (B, T, top_k)

            # Run all experts on full batch, gather top-k
            expert_outs = torch.stack([e(h) for e in experts], dim=2)  # (B, T, n_experts, D)
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, T, top_k, D)
            selected = expert_outs.gather(2, idx_expanded)  # (B, T, top_k, D)
            out = (selected * topk_weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)
            x = x + out
            layer_outputs.append(x)

        # Learned layer weighting
        w = F.softmax(self.layer_weights, dim=0)
        x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))
        return self.final_norm(x)

    def _v2_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        layer_outputs = [x]

        # Initial routing decision from input content (per-position)
        route_signal = x  # (B, T, D)

        for l, (norm, experts) in enumerate(zip(self.norms, self.experts)):
            h = norm(x)

            # Routing: decided by previous layer's output (or input for layer 0)
            logits = self.routers[l](route_signal)  # (B, T, n_experts)
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (B, T, top_k)

            # Run all experts, gather top-k per position
            expert_outs = torch.stack([e(h) for e in experts], dim=2)  # (B, T, n_experts, D)
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
            selected = expert_outs.gather(2, idx_expanded)  # (B, T, top_k, D)

            # Merge: score each expert output AFTER computation, per-position
            output_scores = self.merge_scorers[l](selected).squeeze(-1)  # (B, T, top_k)
            # Router logits for selected experts give gradient signal
            router_scores = topk_vals  # (B, T, top_k)
            scores = output_scores + router_scores
            merge_weights = F.softmax(scores, dim=-1)  # (B, T, top_k)

            out = (selected * merge_weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)
            x = x + out
            layer_outputs.append(x)

            # This layer's output becomes routing signal for next layer
            route_signal = x

        # Learned layer weighting
        w = F.softmax(self.layer_weights, dim=0)
        x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))
        return self.final_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.version == 2:
            return self._v2_forward(x)
        return self._v1_forward(x)
