"""RoutingPool --- shared routing mechanics for LLooM's dual-paradigm pools.

Base class providing:
- Banked outbound routers (with sharing)
- Content-gated hop embeddings
- Hop norm (RMSNorm)
- Router noise injection (annealed)
- Exit/bridge bias application (exit ramp + fixed bridge bias)
- Top-k selection with exit/bridge weight zeroing and renormalization
- route(): logit interpretation with rank-1 exit/bridge rule
- Entry router (for bridge crossings)

Subclasses (SequencePool, TokenPool) provide:
- execute_hop() --- dispatch experts, outbound routers on output, weighted merge
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dispatch import _make_bank, _gather_weights, _make_1d_bank, _gather_1d


class RoutingPool(nn.Module, ABC):
    """Base class for routed expert pools.

    Args:
        pool_size: Number of experts.
        dim: Model hidden dimension.
        top_k: Number of experts selected per routing decision.
        max_hops: Maximum hops (cumulative across visits).
        exit_bias_init: Starting scalar bias for exit slot.
        bridge_bias_init: Starting scalar bias for bridge slot.
        exit_ramp_scale: Scale for exit bias ramp over hops.
        router_noise: Gaussian noise scale for router exploration.
        router_shared_fraction: Fraction of router + hop embed params shared
            across experts.
        hop_gate_dim: Prefix slice of hidden dim used for hop gating.
    """

    def __init__(self, pool_size: int, dim: int, top_k: int,
                 max_hops: int, exit_bias_init: float | None = None,
                 bridge_bias_init: float | None = None,
                 exit_ramp_scale: float = 2.0,
                 router_noise: float = 1.0,
                 router_shared_fraction: float = 0.5,
                 hop_gate_dim: int = 12,
                 global_max_hops: int | None = None):
        super().__init__()
        self.pool_size = pool_size
        self.dim = dim
        self.top_k = top_k
        self.max_hops = max_hops

        # None = auto: log(pool_size) gives equal 1/3 init probability for
        # (any expert, exit, bridge) since pool_size * exp(0) = exp(log(P))
        auto_bias = math.log(pool_size) if pool_size > 1 else 0.0
        self.exit_bias_init = exit_bias_init if exit_bias_init is not None else auto_bias
        self.bridge_bias_init = bridge_bias_init if bridge_bias_init is not None else auto_bias
        self.exit_ramp_scale = exit_ramp_scale
        self.router_noise_scale = router_noise
        self.router_shared_fraction = router_shared_fraction
        self.hop_gate_dim = hop_gate_dim

        # Hop embedding table is sized to global_max_hops (total across both
        # sides) so that global hop indices address distinct embeddings.
        # max_hops is the per-side budget used for exit ramp and budget checks.
        self.global_max_hops = global_max_hops if global_max_hops is not None else max_hops

        self.n_options = pool_size + 2  # experts + exit + bridge
        self.exit_idx = pool_size
        self.bridge_idx = pool_size + 1

        # --- Banked outbound routers (one per expert) ---
        # Router: dim -> n_options, with shared/private split
        self.router_shared, self.router_bank, \
            self.router_shared_out, self.router_private_out = \
            _make_bank(pool_size, dim, self.n_options, router_shared_fraction)

        # --- Entry router: for samples arriving via bridge ---
        # Standalone Linear, produces logits in this pool's space
        self.entry_router = nn.Linear(dim, self.n_options, bias=False)
        nn.init.normal_(self.entry_router.weight, std=dim ** -0.5)

        # --- Per-expert hop embeddings: (pool_size, global_max_hops, dim) ---
        # Shared/private split on last dim (the embedding dim)
        hop_shared_dim = round(dim * router_shared_fraction) if router_shared_fraction > 0 else 0
        hop_private_dim = dim - hop_shared_dim

        self.hop_embed_shared = None
        if hop_shared_dim > 0:
            self.hop_embed_shared = nn.Parameter(
                torch.randn(self.global_max_hops, hop_shared_dim) * 0.02)

        self.hop_embed_bank = nn.Parameter(
            torch.randn(pool_size, self.global_max_hops, hop_private_dim) * 0.02)

        self.hop_shared_dim = hop_shared_dim
        self.hop_private_dim = hop_private_dim

        # --- Content gate for hop embeddings ---
        # gate = sigmoid(linear(h[..., :hop_gate_dim]))
        self.hop_gate_proj = nn.Linear(hop_gate_dim, 1, bias=True)
        nn.init.zeros_(self.hop_gate_proj.weight)
        nn.init.zeros_(self.hop_gate_proj.bias)

        # --- Hop norm (RMSNorm) ---
        self.hop_norm = nn.RMSNorm(dim)

    # ------------------------------------------------------------------
    # Router helpers
    # ------------------------------------------------------------------

    def get_router_logits(self, x: torch.Tensor, expert_idx: torch.Tensor
                          ) -> torch.Tensor:
        """Compute outbound router logits for given expert(s).

        Args:
            x: (..., dim) hidden states.
            expert_idx: (...,) which expert's router to use.

        Returns:
            (..., n_options) raw logits.
        """
        # Gather router weights for the selected experts
        flat_idx = expert_idx.reshape(-1)
        N = flat_idx.shape[0]

        # Private: (N, dim, private_out)
        priv_w = self.router_bank[flat_idx]  # (N, dim, private_out)

        # x might be (..., dim), flatten to (N, dim)
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(N, self.dim)

        # Matmul: (N, dim) @ (N, dim, private_out) -> (N, private_out)
        priv_logits = torch.bmm(
            x_flat.unsqueeze(1), priv_w).squeeze(1)  # (N, private_out)

        if self.router_shared is not None:
            # (N, dim) @ (dim, shared_out) -> (N, shared_out)
            shared_logits = x_flat @ self.router_shared
            logits = torch.cat([shared_logits, priv_logits], dim=-1)
        else:
            logits = priv_logits

        return logits.view(*orig_shape, self.n_options)

    def perturb_logits(self, logits: torch.Tensor,
                       noise_scale: float | None = None) -> torch.Tensor:
        """Add Gaussian noise to router logits for exploration.

        Args:
            logits: (..., n_options) raw router logits.
            noise_scale: Override noise scale (None = use self.router_noise_scale).

        Returns:
            (..., n_options) perturbed logits.
        """
        scale = noise_scale if noise_scale is not None else self.router_noise_scale
        if scale <= 0 or not self.training:
            return logits
        noise = torch.randn_like(logits) * scale
        return logits + noise

    def apply_biases(self, logits: torch.Tensor, hops_used: int | torch.Tensor
                     ) -> torch.Tensor:
        """Apply exit ramp bias and fixed bridge bias.

        Exit bias ramps linearly: exit_bias_init + exit_ramp_scale * (hops_used / max_hops)
        Bridge bias is fixed: bridge_bias_init

        Args:
            logits: (..., n_options) router logits.
            hops_used: Number of hops already consumed on this side (int or scalar tensor).

        Returns:
            (..., n_options) biased logits.
        """
        exit_bias = self.exit_bias_init + self.exit_ramp_scale * (hops_used / self.max_hops)
        logits = logits.clone()
        logits[..., self.exit_idx] = logits[..., self.exit_idx] + exit_bias
        logits[..., self.bridge_idx] = logits[..., self.bridge_idx] + self.bridge_bias_init
        return logits

    def select_topk(self, logits: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-k selection with exit/bridge weight zeroing and renormalization.

        Args:
            logits: (..., n_options) biased router logits.

        Returns:
            topk_idx: (..., top_k) selected indices.
            topk_weights: (..., top_k) normalized weights (exit/bridge zeroed).
            topk_raw_weights: (..., top_k) softmax weights before zeroing.
        """
        topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_raw_weights = F.softmax(topk_logits, dim=-1)

        # Zero out exit/bridge weights
        is_expert = topk_idx < self.pool_size
        topk_weights = topk_raw_weights * is_expert.float()

        # Renormalize over expert slots only (avoid div-by-zero for all-exit/bridge)
        weight_sum = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        topk_weights = topk_weights / weight_sum

        # If ALL top-k are non-expert, zero everything (no dispatch)
        all_non_expert = ~is_expert.any(dim=-1, keepdim=True)
        topk_weights = topk_weights.masked_fill(all_non_expert, 0.0)

        return topk_idx, topk_weights, topk_raw_weights

    def classify_topk(self, topk_idx: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classify each top-k pick as expert, exit, or bridge.

        Args:
            topk_idx: (..., top_k) selected indices.

        Returns:
            is_expert: (..., top_k) bool
            is_exit: (..., top_k) bool
            is_bridge: (..., top_k) bool
        """
        is_expert = topk_idx < self.pool_size
        is_exit = topk_idx == self.exit_idx
        is_bridge = topk_idx == self.bridge_idx
        return is_expert, is_exit, is_bridge

    def route(self, logits: torch.Tensor, hops_used: int | torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interpret incoming logits into routing decisions.

        Applies biases, top-k selection, and the rank-1 exit/bridge rule:
        exit or bridge triggers only if it is the rank-1 pick (index 0
        in the top-k). Otherwise it is zeroed out and expert weights
        renormalize.

        Args:
            logits: (B, n_options) incoming router logits.
            hops_used: Number of hops already consumed on this side.

        Returns:
            topk_idx: (B, top_k) selected indices.
            topk_weights: (B, top_k) normalized expert weights (exit/bridge zeroed).
            has_exit: (B,) bool -- rank-1 is exit.
            has_bridge: (B,) bool -- rank-1 is bridge.
            has_continue: (B,) bool -- rank-1 is an expert.
        """
        logits = self.apply_biases(logits, hops_used)
        topk_idx, topk_weights, _ = self.select_topk(logits)

        # Rank-1 exit/bridge rule: only the top-1 pick matters for action
        rank1 = topk_idx[..., 0]  # (B,)
        has_exit = rank1 == self.exit_idx
        has_bridge = rank1 == self.bridge_idx
        has_continue = rank1 < self.pool_size

        return topk_idx, topk_weights, has_exit, has_bridge, has_continue

    # ------------------------------------------------------------------
    # Hop conditioning
    # ------------------------------------------------------------------

    def apply_hop_conditioning(self, x: torch.Tensor,
                               expert_idx: torch.Tensor,
                               hop: int | torch.Tensor) -> torch.Tensor:
        """Apply hop norm + content-gated hop embedding.

        Args:
            x: (..., dim) hidden states.
            expert_idx: (...,) expert indices (for per-expert hop embeddings).
            hop: Current hop index (0-based). Can be int or scalar tensor
                (scalar tensor avoids recompilation under torch.compile).

        Returns:
            (..., dim) conditioned hidden states.
        """
        # Hop norm
        x = self.hop_norm(x)

        # Gather hop embedding for (expert, hop)
        # Use torch.clamp instead of min() -- Python min() with tensor causes graph break
        if isinstance(hop, torch.Tensor):
            hop_clamped = torch.clamp(hop, max=self.global_max_hops - 1)
        else:
            hop_clamped = min(hop, self.global_max_hops - 1)
        flat_idx = expert_idx.reshape(-1).clamp(max=self.pool_size - 1)

        # Private: (N, dim_private)
        priv_embed = self.hop_embed_bank[flat_idx, hop_clamped]

        if self.hop_embed_shared is not None:
            shared_embed = self.hop_embed_shared[hop_clamped].unsqueeze(0).expand_as(priv_embed)
            hop_embed = torch.cat([shared_embed, priv_embed], dim=-1)
        else:
            hop_embed = priv_embed

        # Reshape to match x
        hop_embed = hop_embed.view(*expert_idx.shape, self.dim)

        # Content gate: sigmoid(linear(h[..., :hop_gate_dim]))
        gate = torch.sigmoid(self.hop_gate_proj(x[..., :self.hop_gate_dim]))

        return x + gate * hop_embed

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def execute_hop(self, x: torch.Tensor, topk_idx: torch.Tensor,
                    topk_weights: torch.Tensor,
                    hop: int | torch.Tensor = 0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run selected experts and produce weighted-merged output + next logits.

        Following PoE's pattern:
        1. Hop conditioning (hop norm + gated hop embed)
        2. Dispatch through selected experts
        3. Each expert's outbound router processes expert output -> logits
        4. Weighted merge of both expert outputs and outbound logits

        Args:
            x: (B, T, D) current hidden state.
            topk_idx: (B, top_k) selected expert/exit indices.
            topk_weights: (B, top_k) softmax weights (exit/bridge zeroed).
            hop: Global hop index for hop conditioning.

        Returns:
            out: (B, T, D) weighted-merged expert output (delta for residual).
            next_logits: (B, n_options) perturbed merged outbound router logits.
        """
        ...

    @abstractmethod
    def prepare_bridge_out(self, x: torch.Tensor) -> torch.Tensor:
        """Format hidden states for the other side."""
        ...

    @abstractmethod
    def accept_bridge_in(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Receive hidden states from the other side, set up context."""
        ...
