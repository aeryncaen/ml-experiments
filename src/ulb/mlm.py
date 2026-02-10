"""DeepMLM — Deep Hybrid Masked Language Model.

No noise schedule, no diffusion. Every layer predicts tokens at masked
(output) positions, accumulates logits into a secondary residual stream,
and re-embeds the current best predictions. Single-pass generation.

Architecture:
    [prompt_tokens | MASK * gen_len] → embed → layer loop → accumulated logits

Each layer:
    1. Pre-norm residual: x = x + layer(norm(x))
    2. Predict logits at output positions via shared output head
    3. Accumulate logits: accum = accum + layer_logits
    4. Re-embed: replace output hidden states with token_embed(argmax(accum)) + pos_embed
       (full replace — next layer sees current best predictions)

Training: CE loss on accumulated logits vs ground-truth output tokens.
Generation: Single forward pass, decode argmax of accumulated logits.
"""

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class DeepMLM(nn.Module):
    """Deep hybrid MLM with per-layer predict and re-embed.

    Args:
        make_layer: Callable that creates a single block (no arguments).
                    Block signature: forward(x) -> delta, where x is pre-normed.
        n_layers: Number of layers.
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        max_seq_len: Maximum total sequence length (prompt + output).
    """

    def __init__(self, make_layer, n_layers: int, vocab_size: int,
                 dim: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        # Layers
        self.layers = nn.ModuleList([make_layer() for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)

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
                target_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            prompt_ids: (B, P) prompt token indices.
            target_ids: (B, G) ground-truth output token indices.
                        During generation, pass dummy ids (zeros); the model
                        ignores them since all output positions are masked.

        Returns:
            accum: (B, G, vocab_size) accumulated softmax distribution across
                   all layers. This IS the prediction — argmax for token ids,
                   log for loss computation.
        """
        B, P = prompt_ids.shape
        G = target_ids.shape[1]
        device = prompt_ids.device

        # Embed prompt
        prompt_x = self.token_embed(prompt_ids)  # (B, P, D)

        # Embed output: mask_embed + absolute positional embeddings
        output_positions = torch.arange(P, P + G, device=device)
        output_x = self.mask_embed.unsqueeze(0).expand(B, G, -1) + self.pos_embed(output_positions)
        # (B, G, D)

        x = torch.cat([prompt_x, output_x], dim=1)  # (B, P+G, D)

        # Accumulated logits (logit residual stream)
        accum = torch.zeros(B, G, self.vocab_size, device=device)
        aux = 0.0

        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
            aux = aux + getattr(layer, 'aux_loss', 0.0)

            # Predict at output positions and accumulate logits
            out_logits = self.output_head(x[:, P:])  # (B, G, vocab)
            accum = accum + out_logits

            # Re-embed from accumulated predictions
            pred_ids = accum.detach().argmax(dim=-1)  # (B, G)
            new_embeds = self.token_embed(pred_ids) + self.pos_embed(output_positions)
            x = torch.cat([x[:, :P], new_embeds], dim=1)

        # Final layer prediction — accumulate and return
        x = self.final_norm(x)
        final_logits = self.output_head(x[:, P:])  # (B, G, vocab)
        accum = accum + final_logits

        self.aux_loss = aux
        return accum


class DeepMLMMoE(nn.Module):
    """Deep hybrid MLM with MoE routing per layer.

    Same per-layer predict/re-embed as DeepMLM, but each layer is an MoE
    layer: route → run experts → merge → predict → re-embed.

    Wraps a MoEStackedULB internally, iterating its layers manually
    instead of calling its forward().

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
    """

    def __init__(self, make_layer: Callable[[], nn.Module],
                 n_layers: int, vocab_size: int, dim: int, max_seq_len: int,
                 n_experts: int = 4, top_k: int = 2,
                 version: Literal[1, 2] = 1,
                 router_mode: Literal['topk', 'relu'] = 'topk'):
        super().__init__()
        from .stack import MoEStackedULB

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

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

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        # Output head (weight-tied)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embed.weight

        self.aux_loss = 0.0

    def forward(self, prompt_ids: torch.Tensor,
                target_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with MoE routing and per-layer predict/re-embed."""
        B, P = prompt_ids.shape
        G = target_ids.shape[1]
        T = P + G
        D = self.dim
        device = prompt_ids.device
        stk = self.stacker

        # Embed
        prompt_x = self.token_embed(prompt_ids)
        output_positions = torch.arange(P, P + G, device=device)
        output_x = self.mask_embed.unsqueeze(0).expand(B, G, -1) + self.pos_embed(output_positions)
        x = torch.cat([prompt_x, output_x], dim=1)  # (B, T, D)

        # Accumulated logits (logit residual stream)
        accum = torch.zeros(B, G, self.vocab_size, device=device)
        total_aux = 0.0

        # Stem layer (non-routed)
        x = x + stk.stem_layer(stk.stem_norm(x))

        # v1 forward with per-layer predict/re-embed
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

                # Predict at output positions and accumulate logits
                out_logits = self.output_head(x[:, P:])
                accum = accum + out_logits

                # Re-embed from accumulated predictions
                pred_ids = accum.detach().argmax(dim=-1)
                new_embeds = self.token_embed(pred_ids) + self.pos_embed(output_positions)
                x = torch.cat([x[:, :P], new_embeds], dim=1)
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

                # Predict at output positions and accumulate logits
                out_logits = self.output_head(x[:, P:])
                accum = accum + out_logits

                # Re-embed from accumulated predictions
                pred_ids = accum.detach().argmax(dim=-1)
                new_embeds = self.token_embed(pred_ids) + self.pos_embed(output_positions)
                x = torch.cat([x[:, :P], new_embeds], dim=1)
                layer_outputs[-1] = x

            # Learned layer weighting
            w = F.softmax(stk.layer_weights, dim=0)
            x = sum(w[i] * layer_outputs[i] for i in range(len(layer_outputs)))

        # Final norm + predict — accumulate and return
        x = stk.final_norm(x)
        final_logits = self.output_head(x[:, P:])
        accum = accum + final_logits

        self.aux_loss = total_aux
        stk.aux_loss = total_aux
        return accum
