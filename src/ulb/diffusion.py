"""MaskedDiffusionPoE — PoolOfExperts for masked token diffusion (LLaDA-style).

Each expert is a denoising step that predicts tokens. Routing decides
the adaptive compute schedule. Hops refine proposals and accumulate
confidence. The exit layer makes the final unmasking decision.

Architecture:
    [prompt_tokens | masked_output_tokens] → embed → stem → routing loop → exit → logits

Each hop:
    1. Run selected experts on the embedded sequence
    2. Predict token logits for all output positions
    3. Re-embed predictions into the output segment (so next hop attends over proposals)
    4. Update accumulated confidence scores
    5. Route to next hop or exit

Mask stays fixed through all hops — hops refine proposals and confidence,
the exit/finalize layer makes the final unmasking decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import ULBConfig
from .diffuser_block import ULBDiffuserBlock
from .norm import RMSNorm
from .stack import PoolOfExperts


MASK_TOKEN_ID = -1  # sentinel; actual embedding handled separately


class MaskedDiffusionPoE(PoolOfExperts):
    """PoolOfExperts adapted for masked token diffusion.

    Processes [prompt | output] sequences where output tokens may be masked.
    Each hop predicts tokens, re-embeds proposals, and accumulates confidence.
    Mask stays fixed through hops — the exit layer makes the final decision.

    Args:
        ulb_config: ULBConfig for the diffuser blocks.
        vocab_size: Token vocabulary size.
        max_seq_len: Maximum total sequence length (prompt + output).
        pool_size: Number of experts.
        top_k: Experts selected per hop.
        max_hops: Loop bound on routing depth.
        local_window: Half-window for acausal local attention on output.
        router_mode: Exit slot density ('squared', 'single', 'half').
        router_noise: Gaussian noise on router logits during training.
        shared_fraction: Convenience — sets block, router, and hop sharing fractions.
        block_shared_fraction: Fraction of expert block output dims shared.
        router_shared_fraction: Fraction of router output dims shared.
        hop_shared_fraction: Fraction of hop embed/gate dims shared.
    """

    def __init__(self, ulb_config: ULBConfig, vocab_size: int, max_seq_len: int = 1024,
                 pool_size: int = 4, top_k: int = 2, max_hops: int | None = None,
                 local_window: int = 16,
                 router_mode: str = 'single',
                 router_noise: float = 1.0,
                 shared_fraction: float = 0.0,
                 block_shared_fraction: float | None = None,
                 router_shared_fraction: float | None = None,
                 hop_shared_fraction: float | None = None):
        dim = ulb_config.d_model
        make_layer = lambda: ULBDiffuserBlock(ulb_config, local_window=local_window)
        super().__init__(
            make_layer=make_layer,
            pool_size=pool_size,
            dim=dim,
            top_k=top_k,
            max_hops=max_hops,
            router_mode=router_mode,
            router_noise=router_noise,
            shared_fraction=shared_fraction,
            block_shared_fraction=block_shared_fraction,
            router_shared_fraction=router_shared_fraction,
            hop_shared_fraction=hop_shared_fraction,
        )

        self.ulb_config = ulb_config
        self.vocab_size = vocab_size
        self.dim = dim

        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)  # learned mask token
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Output head: hidden state → vocab logits
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self.exit_ramp_scale = 10.0

    def _set_input_len(self, n: int):
        """Propagate input_len to all expert blocks."""
        self.stem_layer.input_len = n
        self.exit_layer.input_len = n
        for expert in self.experts:
            expert.input_len = n

    def _embed_sequence(self, token_ids: torch.Tensor, mask: torch.Tensor
                        ) -> torch.Tensor:
        """Embed tokens, using mask_embed for masked positions.

        Args:
            token_ids: (B, L) token indices. Values at masked positions are ignored.
            mask: (B, L) boolean — True = masked.

        Returns:
            (B, L, D) embedded sequence.
        """
        B, L = token_ids.shape
        # Clamp masked positions to valid index for embedding lookup
        safe_ids = token_ids.clamp(min=0)
        x = self.token_embed(safe_ids)  # (B, L, D)
        # Replace masked positions with learned mask embedding
        x = torch.where(mask.unsqueeze(-1), self.mask_embed.unsqueeze(0).unsqueeze(0), x)
        # Add positional embeddings
        positions = torch.arange(L, device=x.device)
        x = x + self.pos_embed(positions)
        return x

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        """Exit layer + final norm. Returns hidden states (not logits).

        Args:
            x: (B, L, D) hidden state after routing loop.

        Returns:
            (B, L, D) final hidden states.
        """
        x = x + self.exit_layer(self.exit_norm(x))
        return self.final_norm(x)

    def forward(self, prompt_ids: torch.Tensor, output_ids: torch.Tensor,
                output_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for masked diffusion training/inference.

        Args:
            prompt_ids: (B, L_in) prompt token indices.
            output_ids: (B, L_out) output token indices (ground truth for loss,
                        or partially predicted during inference).
            output_mask: (B, L_out) boolean — True = masked position.

        Returns:
            logits: (B, L_out, vocab_size) predictions for output positions.
            confidence: (B, L_out) accumulated confidence per output position
                        (max softmax prob across hops).
        """
        B, L_in = prompt_ids.shape
        L_out = output_ids.shape[1]
        L = L_in + L_out

        # Build combined token sequence and mask
        prompt_mask = torch.zeros(B, L_in, dtype=torch.bool, device=prompt_ids.device)
        full_ids = torch.cat([prompt_ids, output_ids], dim=1)  # (B, L)
        full_mask = torch.cat([prompt_mask, output_mask], dim=1)  # (B, L)

        # Set input_len boundary for all diffuser blocks
        self._set_input_len(L_in)

        # Embed
        x = self._embed_sequence(full_ids, full_mask)  # (B, L, D)

        # Accumulated confidence per output position (max across hops)
        confidence = torch.zeros(B, L_out, device=x.device, dtype=x.dtype)

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

            # Predict tokens for all output positions
            out_hidden = x[:, L_in:]  # (B, L_out, D)
            out_logits = self.output_head(out_hidden)  # (B, L_out, vocab)

            # Track confidence: max softmax prob per position
            hop_confidence = out_logits.softmax(dim=-1).max(dim=-1).values  # (B, L_out)
            # Update running confidence (keep max across hops)
            confidence = torch.max(confidence, hop_confidence)

            # Re-embed predictions into output positions so next hop attends over proposed tokens
            pred_ids = out_logits.argmax(dim=-1)  # (B, L_out)
            new_embeds = self.token_embed(pred_ids)  # (B, L_out, D)
            positions = torch.arange(L_in, L, device=x.device)
            new_embeds = new_embeds + self.pos_embed(positions)
            # Replace all output positions with predicted embeddings
            x = torch.cat([x[:, :L_in], new_embeds], dim=1)

        # Exit layer: finalize + predict all remaining
        x = self.finalize(x)
        final_logits = self.output_head(x[:, L_in:])  # (B, L_out, vocab)

        # Accumulate final confidence
        final_confidence = final_logits.softmax(dim=-1).max(dim=-1).values  # (B, L_out)
        confidence = torch.max(confidence, final_confidence)

        self.aux_loss = total_aux
        self.last_mean_hops = non_exit_decisions / B
        if self.trace:
            self.last_trace = trace_hops

        return final_logits, confidence
