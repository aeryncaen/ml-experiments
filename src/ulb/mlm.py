"""DeepMLM — Deep Hybrid Masked Language Model.

No noise schedule, no diffusion. Every layer predicts tokens at masked
(output) positions, re-embeds predictions back into the hidden state,
and tracks confidence. Single-pass generation.

Architecture:
    [prompt_tokens | MASK * gen_len] → embed → layer loop → final logits

Each layer:
    1. Pre-norm residual: x = x + layer(norm(x))
    2. Predict logits at output positions via shared output head
    3. Track confidence (max softmax prob across layers)
    4. Re-embed: replace output hidden states with token_embed(argmax) + pos_embed
       (full replace — next layer sees predicted tokens, not mask embeddings)

Training: CE loss on final layer logits vs ground-truth output tokens.
Generation: Single forward pass, decode argmax of final logits.
"""

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
                target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            prompt_ids: (B, P) prompt token indices.
            target_ids: (B, G) ground-truth output token indices.
                        During generation, pass dummy ids (zeros); the model
                        ignores them since all output positions are masked.

        Returns:
            logits: (B, G, vocab_size) final predictions for output positions.
            confidence: (B, G) max softmax confidence across all layers.
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

        confidence = torch.zeros(B, G, device=device)
        aux = 0.0

        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
            aux = aux + getattr(layer, 'aux_loss', 0.0)

            # Predict at output positions
            out_logits = self.output_head(x[:, P:])  # (B, G, vocab)

            # Track confidence
            with torch.no_grad():
                conf = F.softmax(out_logits, dim=-1).max(dim=-1).values  # (B, G)
                confidence = torch.max(confidence, conf)

            # Re-embed: full replace at output positions
            # Detach argmax — gradients flow through residual stream, not re-embedding
            pred_ids = out_logits.detach().argmax(dim=-1)  # (B, G)
            new_embeds = self.token_embed(pred_ids) + self.pos_embed(output_positions)
            x = torch.cat([x[:, :P], new_embeds], dim=1)

        # Final prediction
        x = self.final_norm(x)
        final_logits = self.output_head(x[:, P:])  # (B, G, vocab)

        with torch.no_grad():
            conf = F.softmax(final_logits, dim=-1).max(dim=-1).values
            confidence = torch.max(confidence, conf)

        self.aux_loss = aux
        return final_logits, confidence
