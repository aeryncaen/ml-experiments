"""SequencePool — sample-level routing through banked attention experts.

Implements the sequence side of LLooM: entire samples route through a pool
of attention experts with top-k selection. Decision aggregation uses the
all-top-k agreement rule: ALL top-k must be exit for sample to exit,
ALL must be bridge for sample to bridge, otherwise continue.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dispatch import AttentionParamBank
from .routing_pool import RoutingPool


class SequencePool(RoutingPool):
    """Pool of attention experts with sample-level routing.

    Per-hop flow:
        1. Hop norm + content-gated hop embedding
        2. Banked attention dispatch for top-k experts
        3. Weighted merge across top-k + residual add
        4. Outbound router logits
        5. Apply exit bias (ramping) and bridge bias (fixed)
        6. Top-k selection + classify: all exit → exit, all bridge → bridge

    Args:
        pool_size: Number of attention experts.
        dim: Model hidden dimension.
        inner_dim: Expert inner dimension after up-projection.
        n_heads: Number of attention heads.
        top_k: Experts selected per routing decision.
        max_hops: Maximum hops (cumulative across all visits).
        exit_bias_init: Starting exit bias.
        bridge_bias_init: Starting bridge bias.
        exit_ramp_scale: Scale for exit bias ramp.
        router_noise: Gaussian noise scale for exploration.
        expert_shared_fraction: Fraction of expert bank weights shared.
        router_shared_fraction: Fraction of router + hop embed params shared.
        hop_gate_dim: Prefix slice for hop gating.
        is_causal: Whether attention is causal.
    """

    def __init__(self, pool_size: int, dim: int, inner_dim: int,
                 n_heads: int, top_k: int = 2, max_hops: int = 16,
                 exit_bias_init: float | None = None,
                 bridge_bias_init: float | None = None,
                 exit_ramp_scale: float = 10.0, router_noise: float = 1.0,
                 expert_shared_fraction: float = 0.5,
                 router_shared_fraction: float = 0.5,
                 hop_gate_dim: int = 12,
                 is_causal: bool = True, global_max_hops: int | None = None):
        super().__init__(
            pool_size=pool_size, dim=dim, top_k=top_k, max_hops=max_hops,
            exit_bias_init=exit_bias_init, bridge_bias_init=bridge_bias_init,
            exit_ramp_scale=exit_ramp_scale, router_noise=router_noise,
            router_shared_fraction=router_shared_fraction,
            hop_gate_dim=hop_gate_dim,
            global_max_hops=global_max_hops,
        )
        self.inner_dim = inner_dim
        self.n_heads = n_heads
        self.is_causal = is_causal

        self.expert_bank = AttentionParamBank(
            pool_size=pool_size, dim=dim, inner_dim=inner_dim,
            n_heads=n_heads, shared_fraction=expert_shared_fraction,
            is_causal=is_causal,
        )

    def dispatch(self, x: torch.Tensor, expert_idx: torch.Tensor,
                 expert_weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """Banked attention dispatch.

        Args:
            x: (B, T, D) hidden states.
            expert_idx: (B, top_k) expert indices.
            expert_weights: (B, top_k) routing weights.

        Returns:
            (B, T, D) expert outputs.
        """
        return self.expert_bank(x, expert_idx, expert_weights)

    def aggregate_decisions(self, topk_idx: torch.Tensor,
                            is_expert: torch.Tensor,
                            is_exit: torch.Tensor,
                            is_bridge: torch.Tensor,
                            **kwargs
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All-top-k agreement rule for sample-level decisions.

        ALL top-k must be exit → sample exits.
        ALL top-k must be bridge → sample bridges.
        Otherwise → sample continues.

        Args:
            topk_idx: (B, top_k) selected indices.
            is_expert/is_exit/is_bridge: (B, top_k) boolean masks.

        Returns:
            (do_continue, do_exit, do_bridge): each (B,) bool.
        """
        all_exit = is_exit.all(dim=-1)       # (B,)
        all_bridge = is_bridge.all(dim=-1)   # (B,)
        do_continue = ~all_exit & ~all_bridge
        return do_continue, all_exit, all_bridge

    def prepare_bridge_out(self, x: torch.Tensor) -> torch.Tensor:
        """Identity — raw hidden state passes to token side."""
        return x

    def accept_bridge_in(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Identity — receive raw hidden state from token side."""
        return x

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor,
                hops_used: int | torch.Tensor, current_expert: torch.Tensor,
                noise_scale: float | None = None,
                global_hop: int | torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, int | torch.Tensor]:
        """Run one hop of sequence-side routing.

        Args:
            x: (B, T, D) hidden states.
            active_mask: (B,) bool — which samples are still active on this side.
            hops_used: Cumulative hops consumed on this side (for exit ramp / budget).
                Can be int or scalar tensor (scalar tensor avoids recompilation
                under torch.compile).
            current_expert: (B,) int — which expert each sample is currently at
                (used for outbound router and hop embedding selection).
            noise_scale: Override router noise (None = use default).
            global_hop: Global hop index across both sides (for hop embeddings).
                Falls back to hops_used if None. Can be int or scalar tensor.

        Returns:
            x: (B, T, D) updated hidden states (inactive samples unchanged).
            active_mask: (B,) updated active mask.
            do_exit: (B,) bool — samples that chose to exit.
            do_bridge: (B,) bool — samples that chose to bridge.
            next_expert: (B,) int — expert assignment for next hop (top-1 of experts).
            hops_used: Updated hop count.
        """
        B, T, D = x.shape
        gh = global_hop if global_hop is not None else hops_used

        if hops_used >= self.max_hops:
            # Budget exhausted — force exit
            return (x, torch.zeros(B, dtype=torch.bool, device=x.device),
                    active_mask.clone(), torch.zeros(B, dtype=torch.bool, device=x.device),
                    current_expert, hops_used)

        # --- Hop conditioning (for active samples) ---
        # Uses global hop for embedding index (computational lifetime)
        x_cond = self.apply_hop_conditioning(
            x.view(B * T, D),
            current_expert.repeat_interleave(T),
            hop=gh,
        ).view(B, T, D)

        # Blend: active samples get conditioned, inactive keep original
        x_work = torch.where(active_mask[:, None, None], x_cond, x)

        # --- Router logits from current expert ---
        # Pool over tokens for sample-level routing: mean pool
        x_pooled = x_work.mean(dim=1)  # (B, D)
        logits = self.get_router_logits(x_pooled, current_expert)
        logits = self.perturb_logits(logits, noise_scale)
        logits = self.apply_biases(logits, hops_used)

        # --- Top-k selection ---
        topk_idx, topk_weights, _ = self.select_topk(logits)
        is_expert, is_exit, is_bridge = self.classify_topk(topk_idx)

        # --- Dispatch through experts ---
        expert_out = self.dispatch(x_work, topk_idx, topk_weights)

        # Residual add (only for active samples)
        x_new = x + torch.where(active_mask[:, None, None], expert_out, torch.zeros_like(expert_out))

        # --- Aggregate decisions ---
        do_continue, do_exit, do_bridge = self.aggregate_decisions(
            topk_idx, is_expert, is_exit, is_bridge)

        # Apply only to currently active samples
        do_exit = do_exit & active_mask
        do_bridge = do_bridge & active_mask
        do_continue = do_continue & active_mask

        # Update active mask
        new_active = active_mask & do_continue

        # Next expert = top-1 expert slot (first expert in top-k)
        # For samples that exit/bridge, doesn't matter
        first_expert_mask = is_expert[:, 0]
        next_expert = torch.where(
            first_expert_mask,
            topk_idx[:, 0],
            current_expert,  # keep current if top-1 wasn't an expert
        )

        return x_new, new_active, do_exit, do_bridge, next_expert, hops_used + 1
