"""
Vanilla transformer for pipeline testing.

Standard GPT-style decoder-only transformer with:
- RMSNorm pre-norm
- SwiGLU MLP
- Grouped-query attention (GQA) or standard MHA
- RoPE positional encoding
- Tied embedding/LM-head weights

This is NOT the YAMIT architecture. It's a standard transformer used to
validate the training pipeline before swapping in MLA, composite-PIT, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 151_669  # Qwen3 base vocab
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 8  # set < n_heads for GQA
    intermediate_size: int = 1536  # SwiGLU hidden dim
    max_seq_len: int = 4096
    rope_theta: float = 50_000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def param_count_estimate(self) -> int:
        """Rough parameter count (excluding embedding if tied)."""
        embed = self.vocab_size * self.d_model
        # Attention: Q + K + V + O per layer
        n_kv = self.n_kv_heads
        attn = self.d_model * (
            self.d_model  # Q
            + n_kv * self.head_dim  # K
            + n_kv * self.head_dim  # V
            + self.d_model  # O
        )
        # SwiGLU MLP: gate + up + down
        mlp = self.d_model * self.intermediate_size * 3
        # Norms: 2 per layer + 1 final
        norms = self.d_model * (2 * self.n_layers + 1)
        total = embed + self.n_layers * (attn + mlp) + norms
        if not self.tie_embeddings:
            total += embed  # separate LM head
        return total


# ── ~130M config ──────────────────────────────────────────────────────────
#
# With Qwen3's 151k vocab, the embedding table alone is huge (~77M params
# at d_model=512). To land near 130M total:
#   embed: 151,669 * 512 = 77.7M
#   12 layers * (attn + mlp) = 12 * (1.31M + 2.36M) = 44M
#   total ≈ 122M (close enough for pipeline testing)

CONFIG_130M = TransformerConfig(
    vocab_size=151_669,
    d_model=512,
    n_layers=12,
    n_heads=8,
    n_kv_heads=8,
    intermediate_size=1536,
    max_seq_len=4096,
    rope_theta=50_000.0,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


def precompute_rope_freqs(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute complex RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding."""
    # xq, xk: (B, n_heads, T, head_dim)
    # freqs: (T, head_dim//2) complex
    B, H, T, D = xq.shape
    xq_c = torch.view_as_complex(xq.float().reshape(B, H, T, D // 2, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(B, H, T, D // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(
        self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, freqs)

        # GQA: repeat KV heads if needed.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Use PyTorch SDPA (flash attention when available).
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=(mask is None)
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(
        self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs, mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies (not a parameter, just a buffer).
        freqs = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs", freqs, persistent=False)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: (B, T) token IDs
            labels: (B, T) target token IDs for loss (optional)
            mask: attention mask (optional; None = causal)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
        """
        x = self.tok_emb(input_ids)

        for layer in self.layers:
            x = layer(x, self.freqs, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift: predict next token.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = CONFIG_130M
    print(f"Config: {cfg}")
    print(f"Estimated params: {cfg.param_count_estimate():,}")

    model = Transformer(cfg)
    print(f"Actual params: {model.param_count():,}")
    print(f"Trainable params: {model.trainable_param_count():,}")

    # Quick forward pass test.
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss = model(x, labels=x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
