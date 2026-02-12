"""TokenPool — token-level routing through banked SwiGLU MLP experts.

Implements the token side of LLooM: individual tokens route through a pool
of MLP experts with top-k selection. Decision aggregation uses Ranked Choice
Voting with sticky votes: tokens that vote exit/bridge lock permanently,
and elimination rounds resolve ties.

FiLM conditioning is generated at every entry to the token side and remains
static throughout the token-side loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dispatch import MLPParamBank, _make_bank
from .routing_pool import RoutingPool


class TokenPool(RoutingPool):
    """Pool of SwiGLU MLP experts with token-level routing + RCV.

    Per-hop flow:
        1. Hop norm + content-gated hop embedding
        2. Token-level router → per-token top-k selection
        3. Classify each token's picks as expert/exit/bridge
        4. Dispatch tokens with any expert in top-k (exit/bridge zeroed)
        5. Parked tokens (only exit/bridge in top-k) keep state unchanged
        6. Sticky votes: newly parked tokens lock their vote
        7. Ranked Choice Vote across all tokens → sample action

    Args:
        pool_size: Number of MLP experts.
        dim: Model hidden dimension.
        inner_dim: Expert inner dimension.
        top_k: Experts selected per routing decision.
        max_hops: Maximum hops (cumulative).
        film_rank: Low-rank bottleneck for FiLM projection.
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
                 film_rank: int = 16,
                 exit_bias_init: float | None = None,
                 bridge_bias_init: float | None = None,
                 exit_ramp_scale: float = 10.0, router_noise: float = 1.0,
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
        self.film_rank = film_rank

        # Expert bank
        self.expert_bank = MLPParamBank(
            pool_size=pool_size, dim=dim, inner_dim=inner_dim,
            shared_fraction=expert_shared_fraction,
        )

        # Entry router: standalone token-level router for first hop
        # (analogous to stem but for token granularity)
        self.entry_router = nn.Linear(dim, self.n_options, bias=False)
        nn.init.normal_(self.entry_router.weight, std=dim ** -0.5)

        # FiLM generator: low-rank projection from mean-pooled sample state
        # Output: (gamma_up[D_inner], beta_up[D_inner], gamma_down[D], beta_down[D])
        # gamma_up/beta_up modulate the up-proj output (D_inner)
        # gamma_down/beta_down modulate the down-proj output (D)
        film_output = 2 * inner_dim + 2 * dim
        self.film_output_split = (inner_dim, inner_dim, dim, dim)
        self.film_down = nn.Linear(dim, film_rank, bias=False)
        self.film_up = nn.Linear(film_rank, film_output, bias=True)
        nn.init.normal_(self.film_down.weight, std=dim ** -0.5)
        nn.init.zeros_(self.film_up.weight)
        # Init bias so gamma=1, beta=0 (identity FiLM at init)
        with torch.no_grad():
            bias = torch.zeros(film_output)
            # gamma_up init to 1
            bias[:inner_dim] = 1.0
            # beta_up init to 0 (already zero)
            # gamma_down init to 1
            bias[2 * inner_dim:2 * inner_dim + dim] = 1.0
            # beta_down init to 0 (already zero)
            self.film_up.bias.copy_(bias)

    def generate_film(self, x: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor]:
        """Generate FiLM conditioning from sample state.

        Args:
            x: (B, T, D) hidden states.

        Returns:
            (gamma_up, beta_up, gamma_down, beta_down):
                each (B, inner_dim), broadcast over T by the expert bank.
        """
        sample_repr = x.mean(dim=1)  # (B, D)
        film_raw = self.film_up(F.silu(self.film_down(sample_repr)))  # (B, film_output)
        gamma_up, beta_up, gamma_down, beta_down = \
            film_raw.split(self.film_output_split, dim=-1)
        return gamma_up, beta_up, gamma_down, beta_down

    def dispatch(self, x: torch.Tensor, expert_idx: torch.Tensor,
                 expert_weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """Banked SwiGLU dispatch with FiLM.

        Args:
            x: (N, D) flattened tokens.
            expert_idx: (N, top_k) expert indices.
            expert_weights: (N, top_k) routing weights.
            **kwargs: film_params — tuple of (gamma_up, beta_up, gamma_down, beta_down).

        Returns:
            (N, D) expert outputs.
        """
        film_params = kwargs.get('film_params', None)
        return self.expert_bank(x, expert_idx, expert_weights, film_params=film_params)

    def aggregate_decisions(self, topk_idx: torch.Tensor,
                            is_expert: torch.Tensor,
                            is_exit: torch.Tensor,
                            is_bridge: torch.Tensor,
                            **kwargs
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ranked Choice Voting with sticky votes.

        This operates at the sample level using per-token votes.

        Args:
            topk_idx: (B, T, top_k) selected indices.
            is_expert/is_exit/is_bridge: (B, T, top_k) boolean masks.
            **kwargs:
                vote_state: (B, T) int8 — 0=active, 1=parked-exit, 2=parked-bridge
                    Modified in-place to update sticky votes.

        Returns:
            (do_continue, do_exit, do_bridge): each (B,) bool.
        """
        vote_state = kwargs.get('vote_state', None)
        B, T, K = topk_idx.shape

        # --- Determine which tokens are newly parked ---
        has_any_expert = is_expert.any(dim=-1)  # (B, T)
        newly_parked = ~has_any_expert  # tokens with only exit/bridge in top-k

        # --- Sticky votes: lock newly parked tokens ---
        if vote_state is not None:
            was_active = vote_state == 0

            # Newly parked tokens: their top-1 determines their locked vote
            top1_is_exit = is_exit[:, :, 0]
            top1_is_bridge = is_bridge[:, :, 0]

            new_exit_parked = was_active & newly_parked & top1_is_exit
            new_bridge_parked = was_active & newly_parked & top1_is_bridge

            vote_state[new_exit_parked] = 1
            vote_state[new_bridge_parked] = 2

        # --- Compute per-token first-choice category ---
        # Active tokens: categorize based on majority of their top-k
        # Category: 0=continue, 1=exit, 2=bridge
        n_exit = is_exit.sum(dim=-1)     # (B, T)
        n_bridge = is_bridge.sum(dim=-1)  # (B, T)

        # Default: continue (has at least one expert)
        token_vote = torch.zeros(B, T, dtype=torch.int8, device=topk_idx.device)
        # Tokens with more exit votes than bridge: vote exit
        token_vote = torch.where(
            (n_exit > n_bridge) & ~has_any_expert,
            torch.ones_like(token_vote),  # exit
            token_vote)
        # Tokens with more bridge votes: vote bridge
        token_vote = torch.where(
            (n_bridge >= n_exit) & ~has_any_expert,
            torch.full_like(token_vote, 2),  # bridge
            token_vote)

        # Override with locked votes from vote_state
        if vote_state is not None:
            locked = vote_state > 0
            token_vote = torch.where(locked, vote_state, token_vote)

        # --- Ranked Choice Voting ---
        return self._ranked_choice_vote(token_vote, B, T, topk_idx.device)

    @staticmethod
    def _ranked_choice_vote(votes: torch.Tensor, B: int, T: int,
                            device: torch.device
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized RCV across tokens within each sample.

        Args:
            votes: (B, T) int8 — 0=continue, 1=exit, 2=bridge.

        Returns:
            (do_continue, do_exit, do_bridge): each (B,) bool.
        """
        # Round 1: tally first choices
        # Count votes for each category
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
        # Find category with fewest votes
        tallies = torch.stack([n_continue, n_exit, n_bridge], dim=-1)  # (B, 3)
        min_cat = tallies.argmin(dim=-1)  # (B,)

        # Transfer eliminated votes to their second choice
        # Second choice for eliminated category:
        #   continue voters → distribute to exit or bridge based on which is leading
        #   exit voters → continue (prefer to stay if can't exit)
        #   bridge voters → continue (prefer to stay if can't bridge)
        # Simplification: eliminated → continue (conservative default)
        # This is vectorized — no per-sample loops.
        r2_continue = n_continue.clone()
        r2_exit = n_exit.clone()
        r2_bridge = n_bridge.clone()

        # Eliminated category's votes go to continue (conservative)
        eliminated_votes = torch.gather(tallies, 1, min_cat.unsqueeze(-1)).squeeze(-1)

        # Zero out eliminated category
        is_elim_continue = min_cat == 0
        is_elim_exit = min_cat == 1
        is_elim_bridge = min_cat == 2

        r2_continue = torch.where(is_elim_continue, torch.zeros_like(r2_continue), r2_continue)
        r2_exit = torch.where(is_elim_exit, torch.zeros_like(r2_exit), r2_exit)
        r2_bridge = torch.where(is_elim_bridge, torch.zeros_like(r2_bridge), r2_bridge)

        # Transfer: eliminated → second choice (continue for exit/bridge, leading for continue)
        # If continue eliminated: votes go to whichever of exit/bridge is leading
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

    def prepare_bridge_out(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape flat tokens back to (B, T, D) for sequence side."""
        return x  # already (B, T, D) when called from forward

    def accept_bridge_in(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Receive hidden states from sequence side. FiLM is generated externally."""
        return x  # raw passthrough

    def forward(self, x: torch.Tensor, active_mask: torch.Tensor,
                hops_used: int | torch.Tensor,
                film_params: tuple[torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor] | None = None,
                vote_state: torch.Tensor | None = None,
                current_expert: torch.Tensor | None = None,
                is_first_hop: bool = False,
                noise_scale: float | None = None,
                global_hop: int | torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, torch.Tensor | None,
                           int | torch.Tensor]:
        """Run one hop of token-side routing.

        Args:
            x: (B, T, D) hidden states.
            active_mask: (B,) bool — which samples are active.
            hops_used: Cumulative hops consumed on this side (for exit ramp / budget).
                Can be int or scalar tensor (scalar tensor avoids recompilation
                under torch.compile).
            film_params: FiLM conditioning tuple or None.
            vote_state: (B, T) int8 sticky vote state (0=active, 1=exit, 2=bridge).
                Created if None. Modified in-place.
            current_expert: (B, T) int — per-token expert assignment.
                None on first hop (entry router used).
            is_first_hop: Whether this is the first hop on this visit.
            noise_scale: Override router noise.
            global_hop: Global hop index across both sides (for hop embeddings).
                Falls back to hops_used if None. Can be int or scalar tensor.

        Returns:
            x: (B, T, D) updated hidden states.
            active_mask: (B,) updated mask.
            do_exit: (B,) samples that chose to exit.
            do_bridge: (B,) samples that chose to bridge.
            current_expert: (B, T) updated expert assignments.
            vote_state: (B, T) updated vote state.
            hops_used: Updated hop count.
        """
        B, T, D = x.shape
        device = x.device
        gh = global_hop if global_hop is not None else hops_used

        # Init vote state if needed
        if vote_state is None:
            vote_state = torch.zeros(B, T, dtype=torch.int8, device=device)

        if hops_used >= self.max_hops:
            # Budget exhausted — force exit
            return (x, torch.zeros(B, dtype=torch.bool, device=device),
                    active_mask.clone(), torch.zeros(B, dtype=torch.bool, device=device),
                    current_expert if current_expert is not None else torch.zeros(B, T, dtype=torch.long, device=device),
                    vote_state, hops_used)

        # --- Hop conditioning (before routing and dispatch) ---
        # Uses global hop for embedding index (computational lifetime)
        if current_expert is None:
            cond_expert = torch.zeros(B, T, dtype=torch.long, device=device)
        else:
            cond_expert = current_expert

        x_cond = self.apply_hop_conditioning(
            x.view(B * T, D),
            cond_expert.view(B * T),
            hop=gh,
        ).view(B, T, D)

        # Blend: active samples get conditioned, inactive keep original
        x_work = torch.where(active_mask[:, None, None], x_cond, x)

        # --- Token-level routing ---
        x_flat = x_work.view(B * T, D)

        if is_first_hop or current_expert is None:
            # Use entry router
            logits = self.entry_router(x_flat)  # (B*T, n_options)
        else:
            # Use outbound router from current expert
            logits = self.get_router_logits(
                x_flat, current_expert.view(B * T))  # (B*T, n_options)

        logits = logits.view(B, T, self.n_options)
        logits = self.perturb_logits(logits, noise_scale)
        logits = self.apply_biases(logits.view(B * T, self.n_options), hops_used).view(B, T, self.n_options)

        # --- Per-token top-k selection ---
        logits_flat = logits.view(B * T, self.n_options)
        topk_idx_flat, topk_weights_flat, _ = self.select_topk(logits_flat)
        topk_idx = topk_idx_flat.view(B, T, self.top_k)
        topk_weights = topk_weights_flat.view(B, T, self.top_k)

        # --- Classify ---
        is_expert_bt, is_exit_bt, is_bridge_bt = self.classify_topk(topk_idx)

        # --- Dispatch tokens that have any expert in top-k ---
        has_any_expert = is_expert_bt.any(dim=-1)  # (B, T)

        # Expand film_params to (B*T, inner_dim) if provided
        film_flat = None
        if film_params is not None:
            g_up, b_up, g_down, b_down = film_params
            # (B, inner_dim) → (B*T, inner_dim) via repeat_interleave
            film_flat = (
                g_up.repeat_interleave(T, dim=0),
                b_up.repeat_interleave(T, dim=0),
                g_down.repeat_interleave(T, dim=0),
                b_down.repeat_interleave(T, dim=0),
            )

        # Dispatch all tokens (parked tokens will have zero weights from select_topk)
        expert_out = self.dispatch(
            x_flat, topk_idx_flat, topk_weights_flat,
            film_params=film_flat)
        expert_out = expert_out.view(B, T, D)

        # --- Next-hop expert tracking ---
        top1_expert = topk_idx[:, :, 0].clamp(max=self.pool_size - 1)

        # Residual add only for dispatched tokens in active samples
        dispatch_mask = has_any_expert & active_mask.unsqueeze(-1)  # (B, T)
        x_new = x + expert_out * dispatch_mask.unsqueeze(-1).float()

        # --- Aggregate decisions via RCV ---
        do_continue, do_exit, do_bridge = self.aggregate_decisions(
            topk_idx, is_expert_bt, is_exit_bt, is_bridge_bt,
            vote_state=vote_state,
        )

        # Apply only to active samples
        do_exit = do_exit & active_mask
        do_bridge = do_bridge & active_mask
        do_continue = do_continue & active_mask

        new_active = active_mask & do_continue

        return (x_new, new_active, do_exit, do_bridge,
                top1_expert, vote_state, hops_used + 1)
