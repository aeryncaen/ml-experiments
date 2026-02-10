"""Simple causal transformer — MHA + SwiGLU, GPT-2 style.

Baseline model for comparison. Nothing fancy:
- Pre-norm residual blocks
- Multi-head attention via F.scaled_dot_product_attention (flash when available)
- SwiGLU FFN
- Learned positional embeddings

Usage:
    model = CausalTransformer(vocab_size=65, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    logits = model(token_ids)  # (B, T, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: gate * silu(gate) where both come from a single linear."""

    def __init__(self, dim: int, expand: float = 8/3):
        super().__init__()
        hidden = int(dim * expand)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    """Pre-norm MHA + SwiGLU block."""

    def __init__(self, dim: int, n_heads: int, ffn_expand: float = 8/3):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ffn = SwiGLU(dim, expand=ffn_expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Attention
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D_h)
        q = q.transpose(1, 2)  # (B, H, T, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.o_proj(attn_out)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class CausalTransformer(nn.Module):
    """Simple causal language model: embed → N blocks → norm → head.

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length for positional embeddings.
        ffn_expand: SwiGLU expansion ratio (default 8/3 ≈ 2.67).
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 256, ffn_expand: float = 8/3):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_expand) for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) token indices.

        Returns:
            (B, T, vocab_size) logits.
        """
        B, T = token_ids.shape
        x = self.token_embed(token_ids) + self.pos_embed(torch.arange(T, device=token_ids.device))

        for block in self.blocks:
            x = block(x)

        return self.head(self.final_norm(x))
