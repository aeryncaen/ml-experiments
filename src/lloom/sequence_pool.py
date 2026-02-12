"""SequencePool --- sample-level routing through banked attention experts.

Implements the sequence side of LLooM: entire samples route through a pool
of attention experts with top-k selection.  Post-dispatch routing: experts
run first, then their outbound routers produce logits from expert output,
which are weighted-merged to form the next hop's incoming logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dispatch import AttentionParamBank
from .routing_pool import RoutingPool


class SequencePool(RoutingPool):
    """Pool of attention experts with sample-level routing.

    Per execute_hop():
        1. Hop norm + content-gated hop embedding
        2. Banked attention dispatch for top-k experts
        3. Each expert's outbound router produces logits from expert output
        4. Weighted merge of both expert outputs and outbound logits
        Returns (out, next_logits) for the caller to residual-add and feed
        into the next route() call.

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
                 exit_ramp_scale: float = 2.0, router_noise: float = 1.0,
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

    def execute_hop(self, x: torch.Tensor, topk_idx: torch.Tensor,
                    topk_weights: torch.Tensor,
                    hop: int | torch.Tensor = 0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run selected experts and produce weighted-merged output + next logits.

        Flow (following PoE):
            1. hop_norm(x) + gated hop embedding -> h
            2. h dispatched through AttentionParamBank (banked gather + einsum)
               Each top-k slot produces an output.  The bank merges them
               weighted by topk_weights internally.
            3. Outbound router: mean-pool the merged output, compute logits
               per selected expert, weighted-merge logits.

        Because AttentionParamBank already does the weighted merge across
        top-k, we can't get per-expert outputs from it.  Instead we compute
        the merged output, and approximate outbound logits by running the
        router for the *dominant* expert (rank-1) on the merged output.
        This is a pragmatic choice -- banked dispatch means we don't have
        individual expert outputs to feed separate routers.

        Actually, to do it properly we loop over top-k slots (fixed small
        loop: 2-3), run the expert bank for each slot individually, get
        the output, run the outbound router, and merge.

        Args:
            x: (B, T, D) current hidden state.
            topk_idx: (B, top_k) selected expert/exit indices.
            topk_weights: (B, top_k) normalized expert weights.
            hop: Global hop index for hop conditioning.

        Returns:
            out: (B, T, D) weighted-merged expert output (delta for residual).
            next_logits: (B, n_options) merged outbound router logits (perturbed).
        """
        B, T, D = x.shape
        top_k = topk_idx.shape[1]
        safe_idx = topk_idx.clamp(max=self.pool_size - 1)

        # --- Hop conditioning ---
        # Use rank-1 expert for hop embedding (all top-k get same conditioning)
        rank1_expert = safe_idx[:, 0]  # (B,)
        x_cond = self.apply_hop_conditioning(
            x.reshape(B * T, D),
            rank1_expert.repeat_interleave(T),
            hop=hop,
        ).reshape(B, T, D)

        # --- Per-slot expert dispatch + outbound router ---
        # Loop over top-k slots (fixed small: 2-3 iterations)
        # For each slot, dispatch single-expert, get output, run outbound router
        merged_out = torch.zeros_like(x)  # (B, T, D)
        merged_logits = torch.zeros(B, self.n_options, device=x.device, dtype=x.dtype)

        for k in range(top_k):
            slot_idx = safe_idx[:, k:k+1]  # (B, 1)
            slot_weight = topk_weights[:, k]  # (B,)

            # Dispatch this single expert slot
            # expert_bank expects (B, top_k, ...) so we pass top_k=1
            slot_weights_for_bank = torch.ones(B, 1, device=x.device, dtype=x.dtype)
            e_out = self.expert_bank(x_cond, slot_idx, slot_weights_for_bank)  # (B, T, D)

            # Outbound router: mean-pool expert output, run banked router
            e_pooled = e_out.mean(dim=1)  # (B, D)
            e_logits = self.get_router_logits(e_pooled, slot_idx.squeeze(1))  # (B, n_options)

            # Weighted accumulate
            w = slot_weight  # (B,)
            merged_out = merged_out + w[:, None, None] * e_out
            merged_logits = merged_logits + w[:, None] * e_logits

        # Perturb merged logits
        merged_logits = self.perturb_logits(merged_logits)

        return merged_out, merged_logits

    def prepare_bridge_out(self, x: torch.Tensor) -> torch.Tensor:
        """Identity --- raw hidden state passes to token side."""
        return x

    def accept_bridge_in(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Identity --- receive raw hidden state from token side."""
        return x
