"""TokenPool --- token-level routing through banked SwiGLU MLP experts.

Implements the token side of LLooM: individual tokens route through a pool
of MLP experts with top-k selection.  Post-dispatch routing: experts run
first, then their outbound routers produce logits from expert output,
which are weighted-merged to form the next hop's incoming logits.

Decision aggregation uses Ranked Choice Voting with sticky votes: tokens
that vote exit/bridge lock permanently, and elimination rounds resolve ties.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dispatch import MLPParamBank
from .routing_pool import RoutingPool


class TokenPool(RoutingPool):
    """Pool of SwiGLU MLP experts with token-level routing + RCV.

    Per execute_hop():
        1. Hop norm + content-gated hop embedding (per token)
        2. Token-level dispatch through MLPParamBank
        3. Each token's outbound router produces logits from expert output
        4. Weighted merge of outbound logits across top-k
        5. Residual add of expert output (only for tokens with expert in top-k)
        Returns (out, next_logits) at token level, plus RCV sample decision.

    The caller (LLooM.forward) handles:
        - Passing incoming logits into route()
        - Calling execute_hop() with the route results
        - Interpreting has_exit/has_bridge from route()
        - RCV aggregation across tokens for sample-level decisions

    Args:
        pool_size: Number of MLP experts.
        dim: Model hidden dimension.
        inner_dim: Expert inner dimension.
        top_k: Experts selected per routing decision.
        max_hops: Maximum hops (cumulative).
        exit_bias_init: Starting exit bias.
        bridge_bias_init: Starting bridge bias.
        exit_ramp_scale: Exit bias ramp scale.
        router_noise: Gaussian noise for exploration.
        expert_shared_fraction: Fraction of expert bank weights shared.
        router_shared_fraction: Fraction of router + hop embed params shared.
        hop_gate_dim: Prefix slice for hop gating.
    """

    def __init__(self, pool_size: int, dim: int, inner_dim: int,
                 top_k: int = 2, max_hops: int = 32,
                 exit_bias_init: float | None = None,
                 bridge_bias_init: float | None = None,
                 exit_ramp_scale: float = 2.0, router_noise: float = 1.0,
                 expert_shared_fraction: float = 0.5,
                 router_shared_fraction: float = 0.5,
                 hop_gate_dim: int = 12,
                 global_max_hops: int | None = None):
        super().__init__(
            pool_size=pool_size, dim=dim, top_k=top_k, max_hops=max_hops,
            exit_bias_init=exit_bias_init, bridge_bias_init=bridge_bias_init,
            exit_ramp_scale=exit_ramp_scale, router_noise=router_noise,
            router_shared_fraction=router_shared_fraction,
            hop_gate_dim=hop_gate_dim,
            global_max_hops=global_max_hops,
        )
        self.inner_dim = inner_dim

        # Expert bank
        self.expert_bank = MLPParamBank(
            pool_size=pool_size, dim=dim, inner_dim=inner_dim,
            shared_fraction=expert_shared_fraction,
        )

    def execute_hop(self, x: torch.Tensor, topk_idx: torch.Tensor,
                    topk_weights: torch.Tensor,
                    hop: int | torch.Tensor = 0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run selected MLP experts and produce merged output + next logits.

        Token-level dispatch: each token independently routes through experts.

        Flow:
            1. hop_norm(x) + gated hop embedding -> h  (per token)
            2. Dispatch all tokens through MLPParamBank
               topk_idx/topk_weights are (B*T, top_k)
            3. For each top-k slot, run outbound router on expert output
            4. Weighted merge of outbound logits across top-k

        Args:
            x: (B, T, D) current hidden state.
            topk_idx: (B, T, top_k) selected expert/exit indices per token.
            topk_weights: (B, T, top_k) normalized expert weights per token.
            hop: Global hop index for hop conditioning.

        Returns:
            out: (B, T, D) expert output (delta for residual).
            next_logits: (B, T, n_options) merged outbound router logits (perturbed).
        """
        B, T, D = x.shape
        top_k = topk_idx.shape[2]
        safe_idx = topk_idx.clamp(max=self.pool_size - 1)

        # --- Hop conditioning ---
        # Use rank-1 expert for each token's hop embedding
        rank1_expert = safe_idx[:, :, 0]  # (B, T)
        x_cond = self.apply_hop_conditioning(
            x.reshape(B * T, D),
            rank1_expert.reshape(B * T),
            hop=hop,
        ).reshape(B, T, D)

        # --- Per-slot dispatch + outbound router ---
        # Flatten to token level
        x_flat = x_cond.reshape(B * T, D)  # (N, D) where N = B*T

        merged_out = torch.zeros(B * T, D, device=x.device, dtype=x.dtype)
        merged_logits = torch.zeros(B * T, self.n_options, device=x.device, dtype=x.dtype)

        for k in range(top_k):
            slot_idx = safe_idx[:, :, k:k+1].reshape(B * T, 1)  # (N, 1)
            slot_weight = topk_weights[:, :, k].reshape(B * T)  # (N,)

            # Dispatch this single expert slot
            slot_weights_for_bank = torch.ones(B * T, 1, device=x.device, dtype=x.dtype)
            e_out = self.expert_bank(x_flat, slot_idx, slot_weights_for_bank)  # (N, D)

            # Outbound router on expert output
            e_logits = self.get_router_logits(e_out, slot_idx.squeeze(1))  # (N, n_options)

            # Weighted accumulate
            merged_out = merged_out + slot_weight[:, None] * e_out
            merged_logits = merged_logits + slot_weight[:, None] * e_logits

        # Perturb merged logits
        merged_logits = self.perturb_logits(merged_logits)

        return merged_out.reshape(B, T, D), merged_logits.reshape(B, T, self.n_options)

    # ------------------------------------------------------------------
    # RCV: Ranked Choice Voting for sample-level decisions
    # ------------------------------------------------------------------

    @staticmethod
    def ranked_choice_vote(token_has_exit: torch.Tensor,
                           token_has_bridge: torch.Tensor,
                           token_has_continue: torch.Tensor,
                           vote_state: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      torch.Tensor]:
        """Vectorized RCV across tokens within each sample.

        Uses the rank-1 decision from route() for each token as the token's
        vote.  Sticky votes: once a token votes exit or bridge, it locks.

        Args:
            token_has_exit: (B, T) bool -- rank-1 is exit for this token.
            token_has_bridge: (B, T) bool -- rank-1 is bridge for this token.
            token_has_continue: (B, T) bool -- rank-1 is expert for this token.
            vote_state: (B, T) int8 sticky state (0=active, 1=exit, 2=bridge).
                Created fresh if None.  Modified in-place.

        Returns:
            do_continue: (B,) bool
            do_exit: (B,) bool
            do_bridge: (B,) bool
            vote_state: (B, T) int8 updated sticky state.
        """
        B, T = token_has_exit.shape
        device = token_has_exit.device

        if vote_state is None:
            vote_state = torch.zeros(B, T, dtype=torch.int8, device=device)

        # --- Sticky votes: lock newly decided tokens ---
        was_active = vote_state == 0
        new_exit = was_active & token_has_exit
        new_bridge = was_active & token_has_bridge
        vote_state[new_exit] = 1
        vote_state[new_bridge] = 2

        # --- Build vote tensor: 0=continue, 1=exit, 2=bridge ---
        # Active tokens vote based on current routing decision
        # Locked tokens use their locked vote
        token_vote = torch.zeros(B, T, dtype=torch.int8, device=device)
        token_vote = torch.where(token_has_exit & was_active, torch.ones_like(token_vote), token_vote)
        token_vote = torch.where(token_has_bridge & was_active, torch.full_like(token_vote, 2), token_vote)
        # Override with locked votes
        locked = vote_state > 0
        token_vote = torch.where(locked, vote_state, token_vote)

        # --- RCV tallying ---
        do_continue, do_exit, do_bridge = _ranked_choice_vote(token_vote, B, T, device)

        return do_continue, do_exit, do_bridge, vote_state

    def prepare_bridge_out(self, x: torch.Tensor) -> torch.Tensor:
        """Identity --- pass hidden state to sequence side."""
        return x

    def accept_bridge_in(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Identity --- receive hidden state from sequence side."""
        return x


def _ranked_choice_vote(votes: torch.Tensor, B: int, T: int,
                        device: torch.device
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized RCV across tokens within each sample.

    Args:
        votes: (B, T) int8 --- 0=continue, 1=exit, 2=bridge.

    Returns:
        (do_continue, do_exit, do_bridge): each (B,) bool.
    """
    # Round 1: tally first choices
    n_continue = (votes == 0).sum(dim=-1)  # (B,)
    n_exit = (votes == 1).sum(dim=-1)
    n_bridge = (votes == 2).sum(dim=-1)
    half = T / 2.0

    # Check for majority
    has_majority_continue = n_continue > half
    has_majority_exit = n_exit > half
    has_majority_bridge = n_bridge > half

    has_any_majority = has_majority_continue | has_majority_exit | has_majority_bridge

    # Round 1 results
    r1_continue = has_majority_continue
    r1_exit = has_majority_exit
    r1_bridge = has_majority_bridge

    # Round 2: eliminate lowest, transfer votes
    tallies = torch.stack([n_continue, n_exit, n_bridge], dim=-1)  # (B, 3)
    min_cat = tallies.argmin(dim=-1)  # (B,)

    r2_continue = n_continue.clone()
    r2_exit = n_exit.clone()
    r2_bridge = n_bridge.clone()

    eliminated_votes = torch.gather(tallies, 1, min_cat.unsqueeze(-1)).squeeze(-1)

    is_elim_continue = min_cat == 0
    is_elim_exit = min_cat == 1
    is_elim_bridge = min_cat == 2

    r2_continue = torch.where(is_elim_continue, torch.zeros_like(r2_continue), r2_continue)
    r2_exit = torch.where(is_elim_exit, torch.zeros_like(r2_exit), r2_exit)
    r2_bridge = torch.where(is_elim_bridge, torch.zeros_like(r2_bridge), r2_bridge)

    # Transfer: eliminated -> second choice
    transfer_to_exit = is_elim_continue & (n_exit >= n_bridge)
    transfer_to_bridge = is_elim_continue & (n_bridge > n_exit)
    transfer_to_continue = is_elim_exit | is_elim_bridge

    r2_exit = r2_exit + torch.where(transfer_to_exit, eliminated_votes, torch.zeros_like(r2_exit))
    r2_bridge = r2_bridge + torch.where(transfer_to_bridge, eliminated_votes, torch.zeros_like(r2_bridge))
    r2_continue = r2_continue + torch.where(transfer_to_continue, eliminated_votes, torch.zeros_like(r2_continue))

    # Round 2 majority check
    r2_majority_continue = r2_continue > half
    r2_majority_exit = r2_exit > half
    r2_majority_bridge = r2_bridge > half

    # Combine: use R1 if majority, else R2
    do_continue = torch.where(has_any_majority, r1_continue, r2_majority_continue)
    do_exit = torch.where(has_any_majority, r1_exit, r2_majority_exit)
    do_bridge = torch.where(has_any_majority, r1_bridge, r2_majority_bridge)

    # Fallback: if still no majority after R2, continue (conservative)
    no_decision = ~(do_continue | do_exit | do_bridge)
    do_continue = do_continue | no_decision

    return do_continue, do_exit, do_bridge
