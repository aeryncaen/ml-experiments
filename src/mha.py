"""MHA transformer architectures — causal and bidirectional.

CausalTransformer: MHA + SwiGLU, GPT-2 style (causal).
BidirectionalTransformer: MHA + SwiGLU, full bidirectional attention.

Usage:
    model = CausalTransformer(vocab_size=256, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    model = BidirectionalTransformer(vocab_size=256, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    logits = model(token_ids)  # (B, T, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


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


def _precompute_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE complex frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)  # (T, head_dim/2)
    return torch.polar(torch.ones_like(angles), angles)  # (T, head_dim/2) complex


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings. x is (B, H, T, D), freqs is (T, D/2) complex."""
    T = x.shape[2]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B, H, T, D/2)
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    out = torch.view_as_real(xc * freqs).flatten(-2)  # (B, H, T, D)
    return out.type_as(x)


class TransformerBlock(nn.Module):
    """Pre-norm MHA + SwiGLU block with RoPE."""

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

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Attention
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D_h)
        q = q.transpose(1, 2)  # (B, H, T, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q = _apply_rope(q, rope_freqs)
        k = _apply_rope(k, rope_freqs)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.o_proj(attn_out)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class CausalTransformer(nn.Module):
    """Simple causal language model: embed -> N blocks -> norm -> head.

    Uses RoPE for positional encoding (no learned positional embeddings).

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        ffn_expand: SwiGLU expansion ratio (default 8/3).
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 256, ffn_expand: float = 8/3):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_expand) for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        # Precompute RoPE frequencies
        head_dim = dim // n_heads
        self.register_buffer('rope_freqs', _precompute_freqs(head_dim, max_seq_len))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) token indices.

        Returns:
            (B, T, vocab_size) logits.
        """
        x = self.token_embed(token_ids)

        for block in self.blocks:
            x = block(x, self.rope_freqs)

        return self.head(self.final_norm(x))


class BidirectionalTransformerBlock(nn.Module):
    """Pre-norm MHA + SwiGLU block with RoPE. Full bidirectional attention (no causal mask)."""

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

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Attention
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        q = _apply_rope(q, rope_freqs)
        k = _apply_rope(k, rope_freqs)

        if HAS_FLASH_ATTN:
            # flash_attn_func expects (B, T, H, D) in bf16/fp16
            q = q.transpose(1, 2).to(torch.bfloat16)
            k = k.transpose(1, 2).to(torch.bfloat16)
            v = v.transpose(1, 2).to(torch.bfloat16)
            attn_out = flash_attn_func(q, k, v, causal=False)
            attn_out = attn_out.to(x.dtype).contiguous().view(B, T, D)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.o_proj(attn_out)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class BidirectionalTransformer(nn.Module):
    """Bidirectional transformer — same as CausalTransformer but without causal mask.

    For use as a LLaDA backbone where the model sees all positions.

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        ffn_expand: SwiGLU expansion ratio (default 8/3).
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 256, ffn_expand: float = 8/3):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            BidirectionalTransformerBlock(dim, n_heads, ffn_expand) for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        # Precompute RoPE frequencies
        head_dim = dim // n_heads
        self.register_buffer('rope_freqs', _precompute_freqs(head_dim, max_seq_len))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) token indices.

        Returns:
            (B, T, vocab_size) logits.
        """
        x = self.token_embed(token_ids)

        for block in self.blocks:
            x = block(x, self.rope_freqs)

        return self.head(self.final_norm(x))


# ---------------------------------------------------------------------------
# Stacker-compatible MHA layers (single-arg forward, no internal residual/norm)
# ---------------------------------------------------------------------------

class CausalMHALayer(nn.Module):
    """MHA + SwiGLU layer compatible with ULB stackers.

    Unlike TransformerBlock, this:
    - Takes a single arg (pre-normed x) — no rope_freqs arg
    - Has NO internal pre-norm or residual — the stacker handles those
    - Stores its own RoPE frequencies

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for RoPE.
        ffn_expand: SwiGLU expansion ratio.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_seq_len: int = 256,
                 ffn_expand: float = 8/3):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, expand=ffn_expand)

        head_dim = dim // n_heads
        self.register_buffer('rope_freqs', _precompute_freqs(head_dim, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is pre-normed by the stacker. Returns delta (no residual add)."""
        B, T, D = x.shape

        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        h = self.o_proj(attn_out)

        # FFN (with its own norm, since stacker only norms the outer residual)
        h = h + self.ffn(self.ffn_norm(h))
        return h


class BidirectionalMHALayer(nn.Module):
    """Bidirectional MHA + SwiGLU layer compatible with ULB stackers.

    Same as CausalMHALayer but without causal mask.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_seq_len: int = 256,
                 ffn_expand: float = 8/3):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, expand=ffn_expand)

        head_dim = dim // n_heads
        self.register_buffer('rope_freqs', _precompute_freqs(head_dim, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is pre-normed by the stacker. Returns delta (no residual add)."""
        B, T, D = x.shape

        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)

        if HAS_FLASH_ATTN:
            q = q.transpose(1, 2).to(torch.bfloat16)
            k = k.transpose(1, 2).to(torch.bfloat16)
            v = v.transpose(1, 2).to(torch.bfloat16)
            attn_out = flash_attn_func(q, k, v, causal=False)
            attn_out = attn_out.to(x.dtype).contiguous().view(B, T, D)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        h = self.o_proj(attn_out)

        h = h + self.ffn(self.ffn_norm(h))
        return h
