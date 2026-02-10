"""DeepMLM — Deep Hybrid Masked Language Model with learned internal diffusion.

Each layer is a learned diffusion step. Layers predict tokens at output
positions and independently decide whether to unmask or re-mask each
position via gumbel-softmax (differentiable). The noise schedule is
entirely learned — no external schedule, no fixed mask ratios.

Architecture:
    [prompt_tokens | MASK * gen_len] → embed → layer loop → final logits

Each layer:
    1. Pre-norm residual: x = x + layer(norm(x))
    2. Predict token logits at output positions via shared output head
    3. Predict per-position mask logit via learned gate (unmask vs re-mask)
    4. Gumbel-softmax the gate → soft decision g in [0, 1]
    5. x[:, P:] = g * (token_embed(pred) + pos_embed) + (1-g) * (mask_embed + pos_embed)
       Positions can be unmasked or re-masked at any layer.

Training: CE loss on final layer logits vs ground-truth output tokens.
Generation: Single forward pass, decode argmax of final logits.
"""

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class DeepMLM(nn.Module):
    """Deep hybrid MLM with learned internal diffusion.

    Each layer predicts tokens and a per-position mask/unmask gate.
    The gate is sampled via gumbel-softmax (differentiable) and controls
    whether each output position is unmasked (re-embedded with prediction)
    or re-masked. Positions can be unmasked and re-masked freely across layers.

    Args:
        make_layer: Callable that creates a single block (no arguments).
                    Block signature: forward(x) -> delta, where x is pre-normed.
        n_layers: Number of layers (each is a learned diffusion step).
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        max_seq_len: Maximum total sequence length (prompt + output).
        gumbel_tau: Temperature for gumbel-softmax (default 1.0).
    """

    def __init__(self, make_layer, n_layers: int, vocab_size: int,
                 dim: int, max_seq_len: int, gumbel_tau: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.gumbel_tau = gumbel_tau

        # Layers
        self.layers = nn.ModuleList([make_layer() for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)

        # Only the second half of layers do diffusion
        self.n_encoder = n_layers // 2
        self.n_diffusion = n_layers - self.n_encoder

        # Per-diffusion-layer mask gate: projects hidden state to 2 logits (mask, unmask)
        self.mask_gates = nn.ModuleList([
            nn.Linear(dim, 2) for _ in range(self.n_diffusion)
        ])

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)
        self.pos_embed = nn.Embedding(max_seq_len, dim)  # for output positions only

        # Output head (weight-tied to token_embed)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embed.weight

        # Stored after forward for logging
        self.aux_loss = 0.0

    def forward(self, prompt_ids: torch.Tensor,
                target_ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with learned internal diffusion.

        Args:
            prompt_ids: (B, P) prompt token indices.
            target_ids: (B, G) ground-truth output token indices.
            mask: (B, G) bool — True = masked (predict), False = given (ground truth).
                  If None, all output positions are masked.

        Returns:
            logits: (B, G, vocab_size) final layer predictions.
        """
        B, P = prompt_ids.shape
        G = target_ids.shape[1]
        device = prompt_ids.device

        # Embed prompt
        prompt_x = self.token_embed(prompt_ids)  # (B, P, D)

        # Embed output: masked positions get mask_embed, unmasked get token_embed
        output_positions = torch.arange(P, P + G, device=device)
        pos_embeds = self.pos_embed(output_positions)  # (G, D)

        if mask is None:
            # All masked
            output_x = self.mask_embed.unsqueeze(0).expand(B, G, -1) + pos_embeds
        else:
            gt_embeds = self.token_embed(target_ids) + pos_embeds  # (B, G, D)
            mask_embeds = self.mask_embed.unsqueeze(0).expand(B, G, -1) + pos_embeds
            output_x = torch.where(mask.unsqueeze(-1), mask_embeds, gt_embeds)

        x = torch.cat([prompt_x, output_x], dim=1)  # (B, P+G, D)

        aux = 0.0
        ne = self.n_encoder

        # First half: normal encoder layers (no predict/re-embed)
        for norm, layer in zip(self.norms[:ne], self.layers[:ne]):
            x = x + layer(norm(x))
            aux = aux + getattr(layer, 'aux_loss', 0.0)

        # Second half: diffusion layers with gumbel-softmax gating
        for norm, layer, gate in zip(self.norms[ne:], self.layers[ne:], self.mask_gates):
            x = x + layer(norm(x))
            aux = aux + getattr(layer, 'aux_loss', 0.0)

            # Predict tokens at output positions
            out_hidden = x[:, P:]  # (B, G, D)
            out_logits = self.output_head(out_hidden)  # (B, G, vocab)
            pred_ids = out_logits.detach().argmax(dim=-1)  # (B, G)
            pred_embeds = self.token_embed(pred_ids) + pos_embeds  # (B, G, D)

            # Mask gate: gumbel-softmax over (mask, unmask)
            gate_logits = gate(out_hidden)  # (B, G, 2)
            g = F.gumbel_softmax(gate_logits, tau=self.gumbel_tau,
                                 hard=False, dim=-1)
            g_unmask = g[:, :, 1:2]  # (B, G, 1) — probability of unmasking

            # Lerp: unmask → pred_embeds, mask → mask_embed + pos
            masked = self.mask_embed.unsqueeze(0).expand(B, G, -1) + pos_embeds
            new_output = g_unmask * pred_embeds + (1.0 - g_unmask) * masked

            x = torch.cat([x[:, :P], new_output], dim=1)

        # Final prediction
        x = self.final_norm(x)
        logits = self.output_head(x[:, P:])  # (B, G, vocab)

        self.aux_loss = aux
        return logits


class DeepMLMMoE(nn.Module):
    """Deep hybrid MLM with MoE routing and learned internal diffusion.

    Same gumbel-softmax mask/unmask gates as DeepMLM, but each layer uses
    MoE routing: route → run experts → merge → predict → gate → re-embed.

    Args:
        make_layer: Callable that creates a single expert block.
        n_layers: Number of MoE layers.
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        max_seq_len: Maximum total sequence length (prompt + output).
        n_experts: Number of experts per layer.
        top_k: Top-k expert selection per sample.
        version: MoE routing version (1 or 2).
        router_mode: 'topk' or 'relu'.
        gumbel_tau: Temperature for gumbel-softmax (default 1.0).
    """

    def __init__(self, make_layer: Callable[[], nn.Module],
                 n_layers: int, vocab_size: int, dim: int, max_seq_len: int,
                 n_experts: int = 4, top_k: int = 2,
                 version: Literal[1, 2] = 1,
                 router_mode: Literal['topk', 'relu'] = 'topk',
                 gumbel_tau: float = 1.0):
        super().__init__()
        from .stack import MoEStackedULB

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.gumbel_tau = gumbel_tau

        # MoE stacker — we'll iterate its internals manually
        self.stacker = MoEStackedULB(
            make_layer=make_layer,
            n_layers=n_layers,
            dim=dim,
            n_experts=n_experts,
            top_k=top_k,
            version=version,
            router_mode=router_mode,
        )

        # Only the second half of MoE layers do diffusion (stem is always encoder)
        self.n_encoder = n_layers // 2
        self.n_diffusion = n_layers - self.n_encoder

        # Per-diffusion-layer mask gates
        self.mask_gates = nn.ModuleList([
            nn.Linear(dim, 2) for _ in range(self.n_diffusion)
        ])

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Output head (weight-tied)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embed.weight

        self.aux_loss = 0.0

    def _gate_and_reembed(self, x, P, G, gate, pos_embeds, B):
        """Predict tokens, gate mask/unmask, re-embed output positions."""
        out_hidden = x[:, P:]  # (B, G, D)
        out_logits = self.output_head(out_hidden)  # (B, G, vocab)
        pred_ids = out_logits.detach().argmax(dim=-1)  # (B, G)
        pred_embeds = self.token_embed(pred_ids) + pos_embeds  # (B, G, D)

        gate_logits = gate(out_hidden)  # (B, G, 2)
        g = F.gumbel_softmax(gate_logits, tau=self.gumbel_tau,
                             hard=False, dim=-1)
        g_unmask = g[:, :, 1:2]  # (B, G, 1)

        masked = self.mask_embed.unsqueeze(0).expand(B, G, -1) + pos_embeds
        new_output = g_unmask * pred_embeds + (1.0 - g_unmask) * masked

        return torch.cat([x[:, :P], new_output], dim=1)

    def forward(self, prompt_ids: torch.Tensor,
                target_ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with MoE routing and learned internal diffusion.

        Args:
            mask: (B, G) bool — True = masked, False = given. None = all masked.
        """
        B, P = prompt_ids.shape
        G = target_ids.shape[1]
        T = P + G
        D = self.dim
        device = prompt_ids.device
        stk = self.stacker

        # Embed
        prompt_x = self.token_embed(prompt_ids)
        output_positions = torch.arange(P, P + G, device=device)
        pos_embeds = self.pos_embed(output_positions)  # (G, D)

        if mask is None:
            output_x = self.mask_embed.unsqueeze(0).expand(B, G, -1) + pos_embeds
        else:
            gt_embeds = self.token_embed(target_ids) + pos_embeds
            mask_embeds = self.mask_embed.unsqueeze(0).expand(B, G, -1) + pos_embeds
            output_x = torch.where(mask.unsqueeze(-1), mask_embeds, gt_embeds)

        x = torch.cat([prompt_x, output_x], dim=1)  # (B, T, D)

        total_aux = 0.0

        ne = self.n_encoder

        # Stem layer (non-routed, always encoder — no gate)
        x = x + stk.stem_layer(stk.stem_norm(x))

        layer_outputs = [x]

        if stk.version == 1:
            for l, (norm, router, experts) in enumerate(
                    zip(stk.norms, stk.routers, stk.experts)):
                h = norm(x)
                h_pool = h.mean(dim=1)
                logits = router(h_pool)

                expert_outs = torch.stack([e(h) for e in experts], dim=2)
                total_aux = total_aux + stk._collect_block_aux(experts)

                if stk.router_mode == 'relu':
                    relu_lambda = getattr(stk, f'_relu_lambda_{l}')
                    weights, route_aux = stk._relu_sample_route(logits, relu_lambda)
                    total_aux = total_aux + route_aux
                    out = (expert_outs * weights[:, None, :, None]).sum(dim=2)
                else:
                    topk_idx, topk_weights, _ = stk._topk_sample_route(logits)
                    idx_expanded = topk_idx[:, None, :, None].expand(-1, T, -1, D)
                    selected = expert_outs.gather(2, idx_expanded)
                    out = (selected * topk_weights[:, None, :, None]).sum(dim=2)

                x = x + out
                layer_outputs.append(x)

                # Only diffusion layers get gated re-embed
                if l >= ne:
                    x = self._gate_and_reembed(x, P, G, self.mask_gates[l - ne], pos_embeds, B)
                    layer_outputs[-1] = x

            # Learned layer weighting
            w = F.softmax(stk.layer_weights, dim=0)
            x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))

        elif stk.version == 2:
            route_signal = x.mean(dim=1)

            for l, (norm, experts) in enumerate(zip(stk.norms, stk.experts)):
                h = norm(x)
                logits = stk.routers[l](route_signal)

                expert_outs = torch.stack([e(h) for e in experts], dim=2)
                total_aux = total_aux + stk._collect_block_aux(experts)

                if stk.router_mode == 'relu':
                    relu_lambda = getattr(stk, f'_relu_lambda_{l}')
                    weights, route_aux = stk._relu_sample_route(logits, relu_lambda)
                    total_aux = total_aux + route_aux
                    out = (expert_outs * weights[:, None, :, None]).sum(dim=2)
                else:
                    topk_vals, topk_idx = logits.topk(stk.top_k, dim=-1)
                    idx_expanded = topk_idx[:, None, :, None].expand(-1, T, -1, D)
                    selected = expert_outs.gather(2, idx_expanded)
                    selected_pooled = selected.mean(dim=1)
                    output_scores = stk.merge_scorers[l](selected_pooled).squeeze(-1)
                    scores = output_scores + topk_vals
                    merge_weights = F.softmax(scores, dim=-1)
                    out = (selected * merge_weights[:, None, :, None]).sum(dim=2)

                x = x + out
                layer_outputs.append(x)
                route_signal = x.mean(dim=1)

                # Only diffusion layers get gated re-embed
                if l >= ne:
                    x = self._gate_and_reembed(x, P, G, self.mask_gates[l - ne], pos_embeds, B)
                    layer_outputs[-1] = x

            # Learned layer weighting
            w = F.softmax(stk.layer_weights, dim=0)
            x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))

        # Final prediction
        x = stk.final_norm(x)
        logits = self.output_head(x[:, P:])

        self.aux_loss = total_aux
        stk.aux_loss = total_aux
        return logits
