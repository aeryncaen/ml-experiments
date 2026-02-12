"""LLooM — dual-paradigm adaptive routing model.

Top-level model combining:
- Entry/exit stems (full transformer blocks)
- Stem router (sample-level → sequence pool, token pool, or exit)
- Sequence pool (attention experts, sample-routed)
- Token pool (SwiGLU MLP experts, token-routed + RCV)
- Bridge (raw passthrough between sides)

Forward flow:
    embed → entry_stem → stem_router
    → [sequence/token pools with bridge crossings]
    → exit_stem → final_norm → head
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

        # --- Stem router: sample-level → seq pool expert / token pool / exit ---
        # Options: seq_pool_size experts + bridge-to-token + exit
        # Bias init: log(pool_size) for exit/bridge slots so that at init each
        # category (any expert, exit, bridge) has equal ~1/3 probability.
        self.stem_router = nn.Linear(config.dim, config.stem_n_options, bias=True)
        nn.init.normal_(self.stem_router.weight, std=config.dim ** -0.5)
        with torch.no_grad():
            self.stem_router.bias.zero_()
            stem_bias = math.log(config.seq_pool_size) if config.seq_pool_size > 1 else 0.0
            self.stem_router.bias[config.seq_pool_size] = stem_bias      # bridge-to-token
            self.stem_router.bias[config.seq_pool_size + 1] = stem_bias  # exit

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

    def _stem_route(self, x: torch.Tensor, noise_scale: float | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                               torch.Tensor, torch.Tensor]:
        """Stem routing decision.

        Args:
            x: (B, T, D) after entry stem.

        Returns:
            go_seq: (B,) bool — enter sequence pool.
            go_tok: (B,) bool — enter token pool.
            go_exit: (B,) bool — skip pools.
            seq_expert: (B,) int — which seq expert to start at.
            raw_logits: (B, stem_n_options) — for aux loss.
        """
        cfg = self.config
        x_pooled = x.mean(dim=1)  # (B, D)
        logits = self.stem_router(x_pooled)  # (B, stem_n_options)

        if noise_scale is not None and noise_scale > 0 and self.training:
            logits = logits + torch.randn_like(logits) * noise_scale

        # Indices: 0..seq_pool_size-1 = seq experts, seq_pool_size = bridge, seq_pool_size+1 = exit
        choice = logits.argmax(dim=-1)  # (B,)

        go_seq = choice < cfg.seq_pool_size
        go_tok = choice == cfg.seq_pool_size     # "bridge" from stem = go to token side
        go_exit = choice == cfg.seq_pool_size + 1

        seq_expert = choice.clamp(max=cfg.seq_pool_size - 1)

        return go_seq, go_tok, go_exit, seq_expert, logits

    def forward(self, x: torch.Tensor,
                noise_scale: float | None = None
                ) -> tuple[torch.Tensor, dict]:
        """Full LLooM forward pass.

        Args:
            x: (B, T, D) input embeddings.
            noise_scale: Override router noise (None = use config default).

        Returns:
            output: (B, T, D) final hidden states.
            info: Dict with routing stats for monitoring/aux loss.
        """
        cfg = self.config
        B, T, D = x.shape
        device = x.device
        ns = noise_scale if noise_scale is not None else cfg.router_noise

        info: dict = {}

        # --- Entry stem ---
        x = self.entry_stem(x)

        # --- Stem routing ---
        go_seq, go_tok, go_exit, seq_expert, stem_logits = self._stem_route(x, ns)
        info['stem_logits'] = stem_logits
        info['stem_go_seq'] = go_seq.float().mean()
        info['stem_go_tok'] = go_tok.float().mean()
        info['stem_go_exit'] = go_exit.float().mean()

        # --- Main routing loop ---
        # Track cumulative hops: per-side (for exit ramp/budget) and global (for hop embeds)
        seq_hops_used = torch.zeros(B, dtype=torch.long, device=device)
        tok_hops_used = torch.zeros(B, dtype=torch.long, device=device)
        global_hops_used = torch.zeros(B, dtype=torch.long, device=device)
        n_bridges = torch.zeros(B, dtype=torch.long, device=device)

        # Active state: which side each sample is currently on
        # 0=done, 1=seq, 2=tok  (torch.where instead of bool indexing for compile)
        side = torch.zeros(B, dtype=torch.long, device=device)
        side = torch.where(go_seq, torch.ones_like(side), side)
        side = torch.where(go_tok, torch.full_like(side, 2), side)
        # go_exit samples stay at 0 (done)

        # Per-sample state
        current_seq_expert = seq_expert
        current_tok_expert = None  # (B, T) — set on first token-side entry
        tok_vote_state = None
        tok_first_hop = torch.ones(B, dtype=torch.bool, device=device)

        # Stem → token counts as a bridge crossing
        n_bridges = torch.where(go_tok, n_bridges + 1, n_bridges)

        # Fixed iteration loop for torch.compile compatibility
        # All state mutations use torch.where (static shapes, no aten.nonzero)
        total_max_iters = cfg.seq_max_hops + cfg.tok_max_hops + cfg.max_bridge_crossings
        for _ in range(total_max_iters):
            any_active = side > 0
            if not any_active.any():
                break

            # --- Sequence side hops ---
            on_seq = side == 1
            if on_seq.any():
                # Masked max for hop counts (hops are non-negative, so masking with 0 is safe)
                # Keep as scalar tensors — Python ints cause recompilation per unique value
                seq_hop = (seq_hops_used * on_seq.long()).max()
                g_hop = (global_hops_used * on_seq.long()).max()
                x_new, still_active, do_exit, do_bridge, next_expert, _ = \
                    self.seq_pool(
                        x, on_seq, hops_used=seq_hop,
                        current_expert=current_seq_expert,
                        noise_scale=ns if self.training else 0.0,
                        global_hop=g_hop,
                    )
                x = x_new
                seq_hops_used = torch.where(on_seq, seq_hops_used + 1, seq_hops_used)
                global_hops_used = torch.where(on_seq, global_hops_used + 1, global_hops_used)
                current_seq_expert = next_expert

                # Handle exits
                exiting = do_exit & on_seq
                side = torch.where(exiting, torch.zeros_like(side), side)

                # Handle bridges to token side
                bridging = do_bridge & on_seq
                if bridging.any():
                    side = torch.where(bridging, torch.full_like(side, 2), side)
                    n_bridges = torch.where(bridging, n_bridges + 1, n_bridges)
                    tok_first_hop = tok_first_hop | bridging
                    tok_vote_state = None  # reset for new token-side visit

                # Samples that didn't exit, bridge, or continue → done
                continuing = still_active & on_seq
                fell_off = on_seq & ~exiting & ~bridging & ~continuing
                side = torch.where(fell_off, torch.zeros_like(side), side)

            # --- Token side hops ---
            on_tok = side == 2
            if on_tok.any():
                tok_hop = (tok_hops_used * on_tok.long()).max()
                g_hop = (global_hops_used * on_tok.long()).max()
                is_first = tok_first_hop.any()
                x_new, still_active, do_exit, do_bridge, new_tok_expert, \
                    tok_vote_state, _ = self.tok_pool(
                        x, on_tok, hops_used=tok_hop,
                        vote_state=tok_vote_state,
                        current_expert=current_tok_expert,
                        is_first_hop=is_first,
                        noise_scale=ns if self.training else 0.0,
                        global_hop=g_hop,
                    )
                x = x_new
                tok_hops_used = torch.where(on_tok, tok_hops_used + 1, tok_hops_used)
                global_hops_used = torch.where(on_tok, global_hops_used + 1, global_hops_used)
                current_tok_expert = new_tok_expert
                tok_first_hop = torch.zeros(B, dtype=torch.bool, device=device)

                # Handle exits
                exiting = do_exit & on_tok
                side = torch.where(exiting, torch.zeros_like(side), side)

                # Handle bridges to sequence side
                bridging = do_bridge & on_tok
                if bridging.any():
                    side = torch.where(bridging, torch.ones_like(side), side)
                    n_bridges = torch.where(bridging, n_bridges + 1, n_bridges)

                continuing = still_active & on_tok
                fell_off = on_tok & ~exiting & ~bridging & ~continuing
                side = torch.where(fell_off, torch.zeros_like(side), side)

        # --- Exit stem ---
        x = self.exit_stem(x)

        # --- Final norm ---
        x = self.final_norm(x)

        info['mean_seq_hops'] = seq_hops_used.float().mean()
        info['mean_tok_hops'] = tok_hops_used.float().mean()
        info['mean_global_hops'] = global_hops_used.float().mean()
        info['mean_bridges'] = n_bridges.float().mean()

        return x, info
