"""Language model architectures — causal, bidirectional, and LLaDA diffusion.

CausalTransformer: MHA + SwiGLU, GPT-2 style (causal).
CausalULB: ULB blocks, same embed/head wrapper (causal).
BidirectionalTransformer: MHA + SwiGLU, full bidirectional attention.
LLaDAModel: Masked diffusion wrapper — works with any backbone.

Usage:
    model = CausalTransformer(vocab_size=65, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    model = CausalULB(vocab_size=65, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    logits = model(token_ids)  # (B, T, vocab_size)

    # LLaDA diffusion:
    llada = LLaDAModel(backbone, vocab_size=65, dim=128)
    logits = llada(token_ids, mask)  # (B, T, vocab_size)
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
    """Simple causal language model: embed → N blocks → norm → head.

    Uses RoPE for positional encoding (no learned positional embeddings).

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        ffn_expand: SwiGLU expansion ratio (default 8/3 ≈ 2.67).
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


class CausalULB(nn.Module):
    """Causal language model using ULB blocks.

    Same embed/head structure as CausalTransformer but uses ULBBlock
    (pre-norm residual stacking) instead of MHA+SwiGLU.

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of ULB blocks.
        max_seq_len: Maximum sequence length for positional embeddings.
        paired: Use paired-head attention (default True).
        attn_mode: Attention mode — 'softmax', 'silu2', or 'blend' (default 'blend').
        inner_ratio: QKV/attention inner dim as ratio of d_model (default 1.75).
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 256,
                 paired: bool = True, attn_mode: str = 'blend',
                 inner_ratio: float = 1.75,
                 q_mix: str = 'lerp', k_lerp: bool = True):
        super().__init__()
        from .block import ULBBlock, ULBConfig
        from .norm import RMSNorm

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)

        config = ULBConfig(
            d_model=dim,
            n_heads=n_heads,
            paired=paired,
            attn_mode=attn_mode,
            inner_ratio=inner_ratio,
            q_mix=q_mix,
            k_lerp=k_lerp,
        )
        self.blocks = nn.ModuleList([ULBBlock(config) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
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
        x = self.token_embed(token_ids)

        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))

        return self.head(self.final_norm(x))


class LLaDAModel(nn.Module):
    """LLaDA masked diffusion wrapper.

    Wraps any backbone (CausalULB, BidirectionalTransformer, etc.) and adds:
    - A learned mask token embedding
    - Masking/unmasking logic for training and generation
    - Predictions over the original vocab (not the mask token)

    The backbone must have:
    - token_embed: nn.Embedding
    - vocab_size, dim, max_seq_len attributes

    LLaDA does NOT add a mask token to the vocab. Instead, the mask token
    is a learned embedding vector that replaces masked positions at the
    embedding level before passing through the backbone's layers.

    Args:
        backbone: The underlying model (we call its layers directly
                  so we can inject mask embeddings).
        vocab_size: Original vocab size (without mask token).
        dim: Model dimension.
    """

    def __init__(self, backbone: nn.Module, vocab_size: int, dim: int):
        super().__init__()
        self.backbone = backbone
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = backbone.max_seq_len

        # Learned mask token embedding
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)

    def _embed_with_mask(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Embed tokens, replacing masked positions with the mask embedding.

        Args:
            token_ids: (B, T) original token indices.
            mask: (B, T) bool — True means masked (replaced with mask_embed).

        Returns:
            (B, T, D) embeddings.
        """
        x = self.backbone.token_embed(token_ids)  # (B, T, D)
        x = torch.where(mask.unsqueeze(-1), self.mask_embed, x)
        return x

    def _run_backbone_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone's layers on pre-computed embeddings.

        Handles both TransformerBlock-based (has rope_freqs)
        and ULB-based (has norms + blocks) styles.

        Args:
            x: (B, T, D) embeddings.

        Returns:
            (B, T, vocab_size) logits.
        """
        bb = self.backbone

        if hasattr(bb, 'rope_freqs'):
            # TransformerBlock-based (Causal or Bidirectional)
            for block in bb.blocks:
                x = block(x, bb.rope_freqs)
        elif hasattr(bb, 'norms'):
            # ULB-based
            for norm, block in zip(bb.norms, bb.blocks):
                x = x + block(norm(x))
        else:
            raise ValueError(f"Unknown backbone type: {type(bb)}")

        return bb.head(bb.final_norm(x))

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for LLaDA.

        Args:
            token_ids: (B, T) ground-truth token indices (clean data x0).
            mask: (B, T) bool — True for masked positions.

        Returns:
            (B, T, vocab_size) logits — predictions for ALL positions,
            but loss should only be computed on masked positions.
        """
        x = self._embed_with_mask(token_ids, mask)
        return self._run_backbone_from_embeddings(x)
