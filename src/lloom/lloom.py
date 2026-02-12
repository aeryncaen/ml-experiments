"""LLooM --- dual-paradigm adaptive routing model.

Top-level model combining:
- Entry/exit stems (full transformer blocks)
- Stem router (sample-level -> seq pool initial logits)
- Sequence pool (attention experts, sample-routed, post-dispatch routing)
- Token pool (SwiGLU MLP experts, token-routed + RCV, post-dispatch routing)
- Bridge (raw passthrough + entry routers on each pool)

Forward flow:
    embed -> entry_stem -> stem_router -> initial logits
    -> [logit chain: route() -> execute_hop() -> merged logits -> route() -> ...]
    -> exit_stem -> final_norm -> head

The routing loop follows PoE's pattern:
    logits -> route(logits) -> topk_idx, topk_weights, has_exit, has_bridge
    -> execute_hop(x, topk_idx, topk_weights) -> out, next_logits
    -> x = x + out; logits = next_logits
    -> repeat
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LLooMConfig
from .sequence_pool import SequencePool
from .token_pool import TokenPool


# ---------------------------------------------------------------------------
# Megatron-style init for LLooM
# ---------------------------------------------------------------------------

def lloom_megatron_init_(model: nn.Module, n_layers: int, std: float = 0.02,
                         cutoff_factor: float = 2.0):
    """Megatron-style weight init for LLooM-based models.

    Follows the same convention as ulb_megatron_init_:
    - Input projections: trunc_normal(std)
    - Output projections (o_proj, down_proj, o_shared, o_bank, down_shared,
      down_bank): trunc_normal(std / sqrt(2 * n_layers))
    - Embeddings: trunc_normal(std)
    - Norm weights: left at their init (1.0)
    - Biases: zero (except stem_router bias which is set separately)
    - Banked params (3-D tensors in expert banks): same rule, applied to the
      raw parameter tensor regardless of shape.

    Args:
        model: The top-level model (e.g. LLooMLM) containing LLooM + embed/head.
        n_layers: Effective depth for output scaling (stems + expected hops).
        std: Base init std for input projections and embeddings.
        cutoff_factor: Truncation range = cutoff_factor * std.
    """
    out_std = std / math.sqrt(2.0 * n_layers)
    cutoff = cutoff_factor * std
    out_cutoff = cutoff_factor * out_std

    # Names that indicate output projections (residual-contributing)
    _output_suffixes = ('.o_proj.weight', '.down_proj.weight',
                        '.o_shared', '.o_bank', '.down_shared', '.down_bank')
    # Names to skip (norm weights, hop embeds, router biases with special init)
    _skip_suffixes = ('.weight',)  # nn.RMSNorm / nn.LayerNorm
    _skip_names = set()

    for name, param in model.named_parameters():
        # Skip norm weights (1-D, from RMSNorm — but also expert norm banks)
        if name.endswith('.norm_shared') or name.endswith('.norm_bank'):
            continue
        if name.endswith('.hop_norm.weight') or name.endswith('.attn_norm.weight') \
                or name.endswith('.mlp_norm.weight') or name.endswith('.final_norm.weight'):
            continue
        # Skip hop embeddings (have their own small init)
        if 'hop_embed' in name:
            continue
        # Skip stem_router bias (set separately for exit/bridge)
        if name == 'stem_router.bias' or name.endswith('.stem_router.bias'):
            continue
        # Skip hop_gate_proj bias
        if 'hop_gate_proj' in name:
            continue

        is_output = any(name.endswith(s) for s in _output_suffixes)

        if is_output:
            nn.init.trunc_normal_(param, std=out_std,
                                  a=-out_cutoff, b=out_cutoff)
        else:
            nn.init.trunc_normal_(param, std=std,
                                  a=-cutoff, b=cutoff)

        # Zero biases that we didn't skip
        # (nn.Linear biases are separate params ending in .bias)

    # Zero all biases except the ones we skipped
    for name, param in model.named_parameters():
        if name.endswith('.bias') and 'stem_router' not in name \
                and 'hop_gate_proj' not in name:
            nn.init.zeros_(param)

    # Rescale exit/bridge biases and exit ramp to match the new router logit
    # scale.  Original router init is dim^{-0.5}; Megatron sets it to `std`.
    # Without rescaling, the ramp overwhelms the tiny post-Megatron logits.
    for name, module in model.named_modules():
        if hasattr(module, 'exit_ramp_scale') and hasattr(module, 'dim'):
            original_std = module.dim ** -0.5
            scale = std / original_std
            module.exit_ramp_scale *= scale
            module.exit_bias_init *= scale
            module.bridge_bias_init *= scale


# ---------------------------------------------------------------------------
# Stem block: a full transformer block (attention + SwiGLU MLP)
# ---------------------------------------------------------------------------

class StemBlock(nn.Module):
    """Non-routed transformer block for entry/exit stems.

    Architecture: pre-norm attention sublayer + pre-norm SwiGLU MLP sublayer,
    each with residual connections.
    """

    def __init__(self, dim: int, n_heads: int, inner_dim: int,
                 is_causal: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.inner_dim = inner_dim
        self.is_causal = is_causal

        # Attention sublayer
        self.attn_norm = nn.RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # MLP sublayer (SwiGLU)
        self.mlp_norm = nn.RMSNorm(dim)
        self.gate_up_proj = nn.Linear(dim, 2 * inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape

        # --- Attention sublayer ---
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.o_proj(attn_out)
        x = x + self.dropout(attn_out)

        # --- MLP sublayer ---
        h = self.mlp_norm(x)
        gate_up = self.gate_up_proj(h)
        gate, up = gate_up.split(self.inner_dim, dim=-1)
        mlp_out = self.down_proj(F.silu(gate) * up)
        x = x + self.dropout(mlp_out)

        return x


# ---------------------------------------------------------------------------
# LLooM
# ---------------------------------------------------------------------------

class LLooM(nn.Module):
    """LLooM: dual-paradigm adaptive routing model.

    Args:
        config: LLooMConfig or None (kwargs used if None).
    """

    def __init__(self, config: LLooMConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = LLooMConfig(**kwargs)
        self.config = config

        # --- Entry stem ---
        self.entry_stem = StemBlock(
            dim=config.dim, n_heads=config.stem_n_heads,
            inner_dim=config.stem_inner_dim,
            is_causal=config.is_causal, dropout=config.dropout,
        )

        # --- Stem router: produces initial logits in seq pool's space ---
        # Options: seq_pool_size experts + exit + bridge (same layout as seq pool)
        # Index seq_pool_size = exit, seq_pool_size + 1 = bridge-to-token
        # Zero bias: the pool's apply_biases() already adds exit/bridge bias,
        # so the stem router should produce unbiased logits to avoid double-counting.
        self.stem_router = nn.Linear(config.dim, config.stem_n_options, bias=True)
        nn.init.normal_(self.stem_router.weight, std=config.dim ** -0.5)
        with torch.no_grad():
            self.stem_router.bias.zero_()

        # --- Sequence pool ---
        self.seq_pool = SequencePool(
            pool_size=config.seq_pool_size,
            dim=config.dim,
            inner_dim=config.seq_inner_dim,
            n_heads=config.seq_n_heads,
            top_k=config.seq_top_k,
            max_hops=config.seq_max_hops,
            exit_bias_init=config.exit_bias_init,
            bridge_bias_init=config.bridge_bias_init,
            exit_ramp_scale=config.exit_ramp_scale,
            router_noise=config.router_noise,
            expert_shared_fraction=config.resolved_seq_expert_share,
            router_shared_fraction=config.resolved_seq_router_share,
            hop_gate_dim=config.hop_gate_dim,
            is_causal=config.is_causal,
            global_max_hops=config.global_max_hops,
        )

        # --- Token pool ---
        self.tok_pool = TokenPool(
            pool_size=config.tok_pool_size,
            dim=config.dim,
            inner_dim=config.tok_inner_dim,
            top_k=config.tok_top_k,
            max_hops=config.tok_max_hops,
            exit_bias_init=config.exit_bias_init,
            bridge_bias_init=config.bridge_bias_init,
            exit_ramp_scale=config.exit_ramp_scale,
            router_noise=config.router_noise,
            expert_shared_fraction=config.resolved_tok_expert_share,
            router_shared_fraction=config.resolved_tok_router_share,
            hop_gate_dim=config.hop_gate_dim,
            global_max_hops=config.global_max_hops,
        )

        # --- Exit stem ---
        self.exit_stem = StemBlock(
            dim=config.dim, n_heads=config.stem_n_heads,
            inner_dim=config.stem_inner_dim,
            is_causal=config.is_causal, dropout=config.dropout,
        )

        # --- Final norm ---
        self.final_norm = nn.RMSNorm(config.dim)

    @property
    def router_noise_scale(self) -> float:
        """Current noise scale (for annealing from benchmark harness)."""
        return self.seq_pool.router_noise_scale

    @router_noise_scale.setter
    def router_noise_scale(self, val: float) -> None:
        self.seq_pool.router_noise_scale = val
        self.tok_pool.router_noise_scale = val

    def _stem_forward(self, x: torch.Tensor, noise_scale: float
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run entry stem and produce initial logits in seq pool's space.

        The stem router produces logits over seq_pool_size + 2 options,
        which is the same as seq_pool.n_options.  These logits feed
        directly into seq_pool.route() for the first routing decision.

        Args:
            x: (B, T, D) raw input.
            noise_scale: Router noise scale.

        Returns:
            x: (B, T, D) after entry stem.
            logits: (B, seq_n_options) initial logits for seq pool routing.
        """
        x = self.entry_stem(x)
        x_pooled = x.mean(dim=1)  # (B, D)
        logits = self.stem_router(x_pooled)  # (B, stem_n_options)

        if noise_scale > 0 and self.training:
            logits = logits + torch.randn_like(logits) * noise_scale

        return x, logits

    def forward(self, x: torch.Tensor,
                noise_scale: float | None = None
                ) -> tuple[torch.Tensor, dict]:
        """Full LLooM forward pass with logit-chain routing.

        The routing loop follows PoE's pattern:
            stem -> initial logits (in seq pool space)
            -> seq_pool.route(logits) -> topk/exit/bridge decision
            -> seq_pool.execute_hop(x, topk) -> out, next_logits
            -> x = x + out; logits = next_logits
            -> repeat until exit or budget exhausted

        Bridge crossings:
            When seq side routes to bridge -> token pool entry_router(x)
            produces logits in tok pool space -> tok pool routing loop.
            When tok side routes to bridge -> seq pool entry_router(x)
            produces logits in seq pool space -> seq pool routing loop.

        Token pool uses RCV for sample-level decisions:
            Per-token route() gives per-token exit/bridge/continue.
            RCV aggregates across tokens with sticky votes.

        Args:
            x: (B, T, D) input embeddings.
            noise_scale: Override router noise (None = use config default).

        Returns:
            output: (B, T, D) final hidden states.
            info: Dict with routing stats for monitoring.
        """
        cfg = self.config
        B, T, D = x.shape
        device = x.device
        ns = noise_scale if noise_scale is not None else cfg.router_noise

        info: dict = {}

        # --- Entry stem + initial logits ---
        x, seq_logits = self._stem_forward(x, ns)
        info['stem_logits'] = seq_logits.detach()

        # --- Tracking state ---
        seq_hops_used = torch.tensor(0, device=device)
        tok_hops_used = torch.tensor(0, device=device)
        global_hop = torch.tensor(0, device=device)
        n_bridges = torch.zeros(B, dtype=torch.long, device=device)
        # Total routing decisions: stem counts as 1, each bridge and expert hop counts as 1
        routing_decisions = torch.ones(B, dtype=torch.long, device=device)  # stem = 1

        # Per-sample state: 0=done, 1=seq, 2=tok
        side = torch.ones(B, dtype=torch.long, device=device)  # all start on seq side

        # The stem produces logits in seq space.  route() will determine
        # if the sample goes to an expert, exits, or bridges.
        # We treat stem->exit and stem->bridge as special cases of
        # the first seq routing decision.

        # Current logits for each sample (in their current pool's space)
        # For seq side: (B, seq_n_options)
        # For tok side: (B, T, tok_n_options)
        # We maintain both, only the relevant one is used per sample.
        current_seq_logits = seq_logits  # (B, seq_n_options)
        current_tok_logits = None  # will be (B, T, tok_n_options) when needed

        # Token-side RCV state
        tok_vote_state = None  # (B, T) int8

        # Stem routing stats (computed from first route call)
        _stem_stats_recorded = False

        # --- Main routing loop ---
        total_max_iters = cfg.seq_max_hops + cfg.tok_max_hops + cfg.max_bridge_crossings
        for _ in range(total_max_iters):
            any_active = side > 0
            if not any_active.any():
                break

            # ============================================================
            # Sequence side
            # ============================================================
            on_seq = side == 1
            if on_seq.any():
                # Mask bridge if tok side is maxed (bridging there would be pointless)
                seq_logits_for_route = current_seq_logits
                if tok_hops_used >= cfg.tok_max_hops:
                    seq_logits_for_route = seq_logits_for_route.clone()
                    seq_logits_for_route[:, self.seq_pool.bridge_idx] = -float('inf')

                # Route: interpret current logits
                topk_idx, topk_weights, has_exit, has_bridge, has_continue = \
                    self.seq_pool.route(seq_logits_for_route, seq_hops_used)

                # Record stem stats on first seq routing decision
                if not _stem_stats_recorded:
                    info['stem_go_seq'] = (has_continue & on_seq).float().sum() / on_seq.float().sum().clamp(min=1)
                    info['stem_go_tok'] = (has_bridge & on_seq).float().sum() / on_seq.float().sum().clamp(min=1)
                    info['stem_go_exit'] = (has_exit & on_seq).float().sum() / on_seq.float().sum().clamp(min=1)
                    _stem_stats_recorded = True

                # Handle exits: samples where rank-1 is exit
                exiting = has_exit & on_seq
                side = torch.where(exiting, torch.zeros_like(side), side)

                # Handle bridges: samples where rank-1 is bridge
                bridging = has_bridge & on_seq
                if bridging.any():
                    side = torch.where(bridging, torch.full_like(side, 2), side)
                    n_bridges = torch.where(bridging, n_bridges + 1, n_bridges)
                    routing_decisions = torch.where(bridging, routing_decisions + 1, routing_decisions)
                    # Bridge counts as a seq hop (builds exit pressure)
                    seq_hops_used = seq_hops_used + 1
                    global_hop = global_hop + 1
                    # Produce entry logits for token pool
                    x_flat_for_tok = x.reshape(B * T, D)
                    tok_entry_logits = self.tok_pool.entry_router(x_flat_for_tok)  # (B*T, tok_n_options)
                    tok_entry_logits = tok_entry_logits.reshape(B, T, self.tok_pool.n_options)
                    if ns > 0 and self.training:
                        tok_entry_logits = tok_entry_logits + torch.randn_like(tok_entry_logits) * ns
                    current_tok_logits = tok_entry_logits
                    tok_vote_state = None  # reset RCV state for new visit

                # Continue: samples where rank-1 is an expert
                continuing = has_continue & on_seq
                if continuing.any():
                    # Execute hop: experts run, outbound routers, merge
                    out, next_logits = self.seq_pool.execute_hop(
                        x, topk_idx, topk_weights, hop=global_hop)

                    # Residual add only for continuing seq samples
                    x = x + out * continuing[:, None, None].float()
                    # Update logits for next iteration
                    current_seq_logits = torch.where(
                        continuing[:, None],
                        next_logits,
                        current_seq_logits,
                    )
                    routing_decisions = torch.where(continuing, routing_decisions + 1, routing_decisions)
                    seq_hops_used = seq_hops_used + 1
                    global_hop = global_hop + 1

                # Budget check: if seq hops exhausted, force remaining seq samples to exit
                if seq_hops_used >= cfg.seq_max_hops:
                    still_on_seq = side == 1
                    side = torch.where(still_on_seq, torch.zeros_like(side), side)

            # ============================================================
            # Token side
            # ============================================================
            on_tok = side == 2
            if on_tok.any():
                if current_tok_logits is None:
                    # First entry to tok side (shouldn't happen normally, but safety)
                    x_flat = x.reshape(B * T, D)
                    current_tok_logits = self.tok_pool.entry_router(x_flat).reshape(B, T, self.tok_pool.n_options)
                    if ns > 0 and self.training:
                        current_tok_logits = current_tok_logits + torch.randn_like(current_tok_logits) * ns
                    tok_vote_state = None

                # Mask bridge if seq side is maxed
                tok_logits_for_route = current_tok_logits
                if seq_hops_used >= cfg.seq_max_hops:
                    tok_logits_for_route = tok_logits_for_route.clone()
                    tok_logits_for_route[:, :, self.tok_pool.bridge_idx] = -float('inf')

                # Per-token routing: apply biases + top-k per token
                BT_logits = tok_logits_for_route.reshape(B * T, self.tok_pool.n_options)
                BT_logits_biased = self.tok_pool.apply_biases(BT_logits, tok_hops_used)
                topk_idx_flat, topk_weights_flat, _ = self.tok_pool.select_topk(BT_logits_biased)

                topk_idx = topk_idx_flat.reshape(B, T, cfg.tok_top_k)
                topk_weights = topk_weights_flat.reshape(B, T, cfg.tok_top_k)

                # Per-token rank-1 classification
                rank1 = topk_idx[:, :, 0]  # (B, T)
                token_has_exit = rank1 == self.tok_pool.exit_idx
                token_has_bridge = rank1 == self.tok_pool.bridge_idx
                token_has_continue = rank1 < self.tok_pool.pool_size

                # RCV for sample-level decision
                do_continue, do_exit, do_bridge, tok_vote_state = \
                    TokenPool.ranked_choice_vote(
                        token_has_exit, token_has_bridge, token_has_continue,
                        vote_state=tok_vote_state,
                    )

                # Mask to only on_tok samples
                do_exit = do_exit & on_tok
                do_bridge = do_bridge & on_tok
                do_continue = do_continue & on_tok

                # Handle exits
                side = torch.where(do_exit, torch.zeros_like(side), side)

                # Handle bridges to seq side
                if do_bridge.any():
                    side = torch.where(do_bridge, torch.ones_like(side), side)
                    n_bridges = torch.where(do_bridge, n_bridges + 1, n_bridges)
                    routing_decisions = torch.where(do_bridge, routing_decisions + 1, routing_decisions)
                    # Bridge counts as a tok hop (builds exit pressure)
                    tok_hops_used = tok_hops_used + 1
                    global_hop = global_hop + 1
                    # Produce entry logits for seq pool
                    x_pooled_for_seq = x.mean(dim=1)  # (B, D)
                    seq_entry_logits = self.seq_pool.entry_router(x_pooled_for_seq)  # (B, seq_n_options)
                    if ns > 0 and self.training:
                        seq_entry_logits = seq_entry_logits + torch.randn_like(seq_entry_logits) * ns
                    current_seq_logits = torch.where(
                        do_bridge[:, None],
                        seq_entry_logits,
                        current_seq_logits,
                    )

                # Continue: execute hop
                if do_continue.any():
                    out, next_logits = self.tok_pool.execute_hop(
                        x, topk_idx, topk_weights, hop=global_hop)

                    # Residual add only for continuing tok samples
                    x = x + out * do_continue[:, None, None].float()
                    # Update tok logits
                    current_tok_logits = torch.where(
                        do_continue[:, None, None],
                        next_logits,
                        current_tok_logits,
                    )
                    routing_decisions = torch.where(do_continue, routing_decisions + 1, routing_decisions)
                    tok_hops_used = tok_hops_used + 1
                    global_hop = global_hop + 1

                # Budget check
                if tok_hops_used >= cfg.tok_max_hops:
                    still_on_tok = side == 2
                    side = torch.where(still_on_tok, torch.zeros_like(side), side)

            # Bridge crossing limit
            over_bridge_limit = n_bridges >= cfg.max_bridge_crossings
            still_active = side > 0
            side = torch.where(over_bridge_limit & still_active, torch.zeros_like(side), side)

        # --- Exit stem ---
        x = self.exit_stem(x)

        # --- Final norm ---
        x = self.final_norm(x)

        info['mean_seq_hops'] = seq_hops_used.float()
        info['mean_tok_hops'] = tok_hops_used.float()
        info['mean_global_hops'] = global_hop.float()
        info['mean_bridges'] = n_bridges.float().mean()
        info['mean_routing_decisions'] = routing_decisions.float().mean()

        return x, info
