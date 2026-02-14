"""MHA transformer architectures — causal and bidirectional.

CausalTransformer: MHA + SwiGLU, GPT-2 style (causal).
BidirectionalTransformer: MHA + SwiGLU, full bidirectional attention.

Usage:
    model = CausalTransformer(vocab_size=256, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    model = BidirectionalTransformer(vocab_size=256, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    logits = model(token_ids)  # (B, T, vocab_size)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def megatron_init_(model: nn.Module, n_layers: int, std: float = 0.02,
                   cutoff_factor: float = 2.0):
    """Full Megatron init (ModernBERT-style).

    - Input projections (qkv_proj, ffn.w_gate, ffn.w_up): trunc_normal(std)
    - Output projections (o_proj, ffn.w_down): trunc_normal(std / sqrt(2 * n_layers))
    - Embeddings (token_embed): trunc_normal(std)
    - All biases: zero

    For stacker-compatible layers (CausalMHALayer, BidirectionalMHALayer),
    pass layer_id to scale output projections per-layer.
    """
    out_std = std / math.sqrt(2.0 * n_layers)
    cutoff = cutoff_factor * std
    out_cutoff = cutoff_factor * out_std

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Identify output projections by name
            if name.endswith('.o_proj') or name.endswith('.w_down'):
                nn.init.trunc_normal_(module.weight, std=out_std,
                                      a=-out_cutoff, b=out_cutoff)
            elif name.endswith('.head'):
                # head is weight-tied to embedding, skip
                pass
            else:
                nn.init.trunc_normal_(module.weight, std=std,
                                      a=-cutoff, b=cutoff)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=std,
                                  a=-cutoff, b=cutoff)


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


class FeatureAttn(nn.Module):
    """Feature-attention: attention over feature groups within each token.

    Partitions D into G groups of (D/G) dims each.  Runs non-causal
    self-attention over the groups, applies a content-dependent sigmoid
    gate, and reshapes back to D.  No learned up/down projections — the
    grouping is a simple reshape.

    Two modes controlled by ``transpose_groups``:
      - False (default): reshape to (N, G, D/G).  G positions, each with
        a (D/G)-dim vector.  Attention is a G×G matrix — "which groups
        are relevant?"
      - True: reshape to (N, D/G, G).  D/G positions, each with a G-dim
        vector.  Attention is a (D/G)×(D/G) matrix — "which features
        are relevant?"  This is closer to true feature-attention.

    Args:
        dim: Model dimension.
        feat_expansion: Number of feature groups G.  D is partitioned into
            G groups of (D // G) features each.  (Named feat_expansion for
            backward compat; there is no expansion.)
        n_heads: Number of attention heads within feature-attention.
        transpose_groups: If True, transpose the reshape so the attention
            axis is D/G features instead of G groups.
    """

    def __init__(self, dim: int, feat_expansion: int = 4, n_heads: int = 1,
                 transpose_groups: bool = False, up_factor: int = 1):
        super().__init__()
        self.dim = dim
        self.n_groups = feat_expansion
        self.group_dim = dim // feat_expansion
        self.transpose_groups = transpose_groups
        self.up_factor = up_factor

        assert dim % feat_expansion == 0, (
            f"dim ({dim}) must be divisible by n_groups ({feat_expansion})")

        # Per-group channel expansion: group_dim -> group_dim * up_factor
        self.wide_dim = self.group_dim * up_factor

        # After grouping + optional transpose, determine attn layout
        if transpose_groups:
            self.n_pos = self.group_dim
            self.pos_dim = self.n_groups
            # Channel expansion on the pos_dim axis (G -> G * up_factor)
            self.wide_pos_dim = self.n_groups * up_factor
        else:
            self.n_pos = self.n_groups
            self.pos_dim = self.group_dim
            # Channel expansion on the pos_dim axis (group_dim -> group_dim * up_factor)
            self.wide_pos_dim = self.group_dim * up_factor

        self.n_heads = n_heads
        assert self.wide_pos_dim % n_heads == 0, (
            f"wide_pos_dim ({self.wide_pos_dim}) must be divisible by n_heads ({n_heads})")
        self.head_dim = self.wide_pos_dim // n_heads

        # Per-group channel up/down (applied per position, shared across positions)
        if up_factor > 1:
            self.chan_up = nn.Linear(self.pos_dim, self.wide_pos_dim, bias=False)
            self.chan_down = nn.Linear(self.wide_pos_dim, self.pos_dim, bias=False)
        else:
            self.chan_up = None
            self.chan_down = None

        # Feature-attention QKV at wide dim
        self.qkv_proj = nn.Linear(self.wide_pos_dim, 3 * self.wide_pos_dim, bias=False)

        # Output projection at wide dim
        self.o_proj = nn.Linear(self.wide_pos_dim, self.wide_pos_dim, bias=False)

        # Block-diagonal gate at wide dim
        self.gate_proj = nn.Parameter(
            torch.randn(self.n_pos, self.wide_pos_dim, self.wide_pos_dim)
            * (self.wide_pos_dim ** -0.5)
        )

        # Norm over full wide dim * n_pos before gated combine
        self.feat_norm = nn.RMSNorm(self.n_pos * self.wide_pos_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) input.
        Returns:
            (..., D) output.
        """
        orig_shape = x.shape[:-1]
        N = x.reshape(-1, self.dim).shape[0]
        x_flat = x.reshape(N, self.dim)

        # Partition: (N, D) -> (N, G, group_dim) then optionally transpose
        h = x_flat.view(N, self.n_groups, self.group_dim)
        if self.transpose_groups:
            h = h.transpose(1, 2).contiguous()
        # h is now (N, n_pos, pos_dim)

        # Per-group channel expansion with SiLU
        if self.chan_up is not None:
            h = F.silu(self.chan_up(h))                                 # (N, P, wide_pd)
        # h is now (N, n_pos, wide_pos_dim)

        # Feature-attention QKV
        qkv = self.qkv_proj(h)                                        # (N, P, 3*wpd)
        q, k, v = qkv.split(self.wide_pos_dim, dim=-1)

        # Reshape for multi-head: (N, n_heads, P, head_dim)
        q = q.view(N, self.n_pos, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, self.n_pos, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, self.n_pos, self.n_heads, self.head_dim).transpose(1, 2)

        # Attend over positions (non-causal)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Concat heads and output projection
        y = y.transpose(1, 2).contiguous().view(N, self.n_pos, self.wide_pos_dim)
        y = self.o_proj(y)                                             # (N, P, wpd)

        # Gate (on activated content)
        y = y.reshape(N, self.n_pos * self.wide_pos_dim)
        gate = torch.sigmoid(
            torch.einsum('npd,pde->npe', h, self.gate_proj)
        ).reshape(N, self.n_pos * self.wide_pos_dim)
        y = self.feat_norm(y) * gate

        # Per-group channel collapse
        y = y.view(N, self.n_pos, self.wide_pos_dim)
        if self.chan_down is not None:
            y = self.chan_down(y)                                       # (N, P, pos_dim)

        # Transpose back if needed, flatten
        if self.transpose_groups:
            y = y.transpose(1, 2).contiguous()
        y = y.reshape(N, self.dim)

        return y.view(*orig_shape, self.dim)


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

        # Megatron init
        megatron_init_(self, n_layers)

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


class FeatureAttnBlock(nn.Module):
    """Pre-norm sequence-attention + pre-norm feature-attention block with RoPE.

    Drop-in replacement for TransformerBlock: replaces SwiGLU FFN with
    FeatureAttn (attention over feature groups within each token).

    Args:
        dim: Model dimension.
        n_heads: Number of sequence-attention heads.
        feat_expansion: Feature expansion factor.
        feat_group_dim: Feature group size.
        feat_n_heads: Heads within feature-attention.
        feat_first: If True, feature-attention runs before sequence-attention.
            Default False (sequence-attention first, like standard transformers).
    """

    def __init__(self, dim: int, n_heads: int, feat_expansion: int = 4,
                 feat_n_heads: int = 1, feat_first: bool = False,
                 transpose_groups: bool = False, up_factor: int = 1):
        super().__init__()
        self.feat_first = feat_first

        self.attn_norm = nn.RMSNorm(dim)
        self.feat_norm = nn.RMSNorm(dim)

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ffn = FeatureAttn(dim, feat_expansion=feat_expansion,
                               n_heads=feat_n_heads,
                               transpose_groups=transpose_groups,
                               up_factor=up_factor)
        # Content-dependent gate on feature-attention delta
        self.feat_gate_proj = nn.Linear(dim, dim, bias=False)

    def _seq_attn(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.attn_norm(x)
        qkv = self.qkv_proj(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = _apply_rope(q, rope_freqs)
        k = _apply_rope(k, rope_freqs)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(attn_out)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        if self.feat_first:
            feat_delta = self.ffn(self.feat_norm(x))
            x = x + feat_delta * torch.sigmoid(self.feat_gate_proj(x))
            x = x + self._seq_attn(x, rope_freqs)
        else:
            x = x + self._seq_attn(x, rope_freqs)
            feat_delta = self.ffn(self.feat_norm(x))
            x = x + feat_delta * torch.sigmoid(self.feat_gate_proj(x))
        return x


class FeatureAttnTransformer(nn.Module):
    """Causal LM with feature-attention replacing SwiGLU.

    Identical to CausalTransformer except each block uses FeatureAttnBlock
    (sequence-attention + feature-attention) instead of TransformerBlock
    (sequence-attention + SwiGLU).

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        n_heads: Number of sequence-attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        feat_expansion: Number of feature groups (rank expansion).
        feat_n_heads: Heads within feature-attention.
        feat_first: If True, feature-attention runs before sequence-attention.
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 256,
                 feat_expansion: int = 4, feat_n_heads: int = 1,
                 feat_first: bool = False, transpose_groups: bool = False,
                 up_factor: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            FeatureAttnBlock(dim, n_heads, feat_expansion, feat_n_heads, feat_first,
                             transpose_groups=transpose_groups, up_factor=up_factor)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        # Precompute RoPE frequencies
        head_dim = dim // n_heads
        self.register_buffer('rope_freqs', _precompute_freqs(head_dim, max_seq_len))

        # Megatron init
        megatron_init_(self, n_layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(token_ids)

        for block in self.blocks:
            x = block(x, self.rope_freqs)

        return self.head(self.final_norm(x))


class DualMHA(nn.Module):
    """Two independent causal transformers whose logits are subtracted.

    Each half has its own embedding, blocks, final norm, and head (weight-tied
    to its own embedding).  Forward returns logits_a - logits_b.

    The idea: one model learns "what to predict", the other learns "what NOT
    to predict".  The difference sharpens the distribution.

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension (same for both halves).
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks per half.
        max_seq_len: Maximum sequence length for RoPE.
        ffn_expand: SwiGLU expansion ratio.
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 256, ffn_expand: float = 8/3):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # --- Half A ---
        self.embed_a = nn.Embedding(vocab_size, dim)
        self.blocks_a = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_expand) for _ in range(n_layers)
        ])
        self.norm_a = nn.RMSNorm(dim)
        self.head_a = nn.Linear(dim, vocab_size, bias=False)
        self.head_a.weight = self.embed_a.weight  # weight tying

        # --- Half B ---
        self.embed_b = nn.Embedding(vocab_size, dim)
        self.blocks_b = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_expand) for _ in range(n_layers)
        ])
        self.norm_b = nn.RMSNorm(dim)
        self.head_b = nn.Linear(dim, vocab_size, bias=False)
        self.head_b.weight = self.embed_b.weight  # weight tying

        # Shared RoPE freqs (same positional encoding for both)
        head_dim = dim // n_heads
        self.register_buffer('rope_freqs', _precompute_freqs(head_dim, max_seq_len))

        # Megatron init each half independently
        megatron_init_(self, n_layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) token indices.

        Returns:
            (B, T, vocab_size) logits_a - logits_b.
        """
        # Half A
        xa = self.embed_a(token_ids)
        for block in self.blocks_a:
            xa = block(xa, self.rope_freqs)
        logits_a = self.head_a(self.norm_a(xa))

        # Half B
        xb = self.embed_b(token_ids)
        for block in self.blocks_b:
            xb = block(xb, self.rope_freqs)
        logits_b = self.head_b(self.norm_b(xb))

        return logits_a - logits_b


class BidirectionalTransformerBlock(nn.Module):
    """Pre-norm MHA + SwiGLU block with RoPE. Supports causal or bidirectional attention."""

    def __init__(self, dim: int, n_heads: int, ffn_expand: float = 8/3,
                 is_causal: bool = True):
        super().__init__()
        self.is_causal = is_causal
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

        if HAS_FLASH_ATTN and not self.is_causal:
            # flash_attn_func expects (B, T, H, D) in bf16/fp16
            q = q.transpose(1, 2).to(torch.bfloat16)
            k = k.transpose(1, 2).to(torch.bfloat16)
            v = v.transpose(1, 2).to(torch.bfloat16)
            attn_out = flash_attn_func(q, k, v, causal=False)
            attn_out = attn_out.to(x.dtype).contiguous().view(B, T, D)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.o_proj(attn_out)

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x


class BidirectionalTransformer(nn.Module):
    """MHA + SwiGLU transformer with RoPE. Supports causal (default) or bidirectional.

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        ffn_expand: SwiGLU expansion ratio (default 8/3).
        is_causal: Whether to use causal masking (default True).
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 256, ffn_expand: float = 8/3,
                 is_causal: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            BidirectionalTransformerBlock(dim, n_heads, ffn_expand, is_causal=is_causal)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        # Precompute RoPE frequencies
        head_dim = dim // n_heads
        self.register_buffer('rope_freqs', _precompute_freqs(head_dim, max_seq_len))

        # Megatron init
        megatron_init_(self, n_layers)

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


class DiffMHALayer(nn.Module):
    """Two independent MHA+SwiGLU sub-layers whose outputs are subtracted.

    Stacker-compatible: takes pre-normed x, returns delta (no residual).
    The stacker wraps this with x = x + DiffMHALayer(norm(x)), so the
    effective update is x = x + (layer_a(norm(x)) - layer_b(norm(x))).

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for RoPE.
        ffn_expand: SwiGLU expansion ratio.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_seq_len: int = 256,
                 ffn_expand: float = 8/3):
        super().__init__()
        self.layer_a = CausalMHALayer(dim, n_heads, max_seq_len, ffn_expand)
        self.layer_b = CausalMHALayer(dim, n_heads, max_seq_len, ffn_expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_a(x) - self.layer_b(x)


class MulMHALayer(nn.Module):
    """Two independent MHA+SwiGLU sub-layers whose outputs are multiplied.

    Stacker-compatible: takes pre-normed x, returns delta (no residual).
    The stacker wraps this with x = x + MulMHALayer(norm(x)), so the
    effective update is x = x + (layer_a(norm(x)) * layer_b(norm(x))).

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for RoPE.
        ffn_expand: SwiGLU expansion ratio.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_seq_len: int = 256,
                 ffn_expand: float = 8/3):
        super().__init__()
        self.layer_a = CausalMHALayer(dim, n_heads, max_seq_len, ffn_expand)
        self.layer_b = CausalMHALayer(dim, n_heads, max_seq_len, ffn_expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_a(x) * self.layer_b(x)


class ExpandKVMHALayer(nn.Module):
    """MHA + SwiGLU where K and V are expanded along the sequence dimension.

    Q stays at T positions.  K and V are projected to kv_expand * T positions:
    each input token spawns kv_expand key-value slots.  Every query attends
    to kv_expand times as many positions.

    Stacker-compatible: takes pre-normed x, returns delta (no residual).

    RoPE: each group of kv_expand K positions shares the position index of
    the original input token (content differentiation, not positional).

    Causal mask: Q at input position i can attend to all K/V positions
    originating from input positions 0..i, i.e. K indices 0..kv_expand*(i+1)-1.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for RoPE.
        ffn_expand: SwiGLU expansion ratio.
        kv_expand: Expansion factor for K/V along the sequence dim.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_seq_len: int = 256,
                 ffn_expand: float = 8/3, kv_expand: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.kv_expand = kv_expand
        assert dim % n_heads == 0

        # Q: normal size.  K, V: kv_expand * dim each.
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_expand * dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_expand * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, expand=ffn_expand)

        head_dim = dim // n_heads
        # RoPE freqs only need max_seq_len (Q positions), K reuses via repeat
        self.register_buffer('rope_freqs', _precompute_freqs(head_dim, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh, E = self.n_heads, self.head_dim, self.kv_expand

        # Q: (B, H, T, Dh)
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)

        # K: (B, T, E*D) -> (B, T, E, H, Dh) -> (B, H, T*E, Dh)
        k = self.k_proj(x).view(B, T, E, H, Dh)
        k = k.permute(0, 3, 1, 2, 4).reshape(B, H, T * E, Dh)

        # V: same reshape as K
        v = self.v_proj(x).view(B, T, E, H, Dh)
        v = v.permute(0, 3, 1, 2, 4).reshape(B, H, T * E, Dh)

        # RoPE: Q gets positions [0..T-1].
        # K gets positions repeated E times: [0,0,..,0, 1,1,..,1, ...].
        q = _apply_rope(q, self.rope_freqs)
        # Expand rope freqs for K: repeat each position E times
        rope_k = self.rope_freqs[:T].unsqueeze(1).expand(-1, E, -1).reshape(T * E, -1)
        # _apply_rope expects (T, Dh/2) freqs and (B, H, T, Dh) input
        kc = torch.view_as_complex(k.float().reshape(B, H, T * E, -1, 2))
        rope_k = rope_k.unsqueeze(0).unsqueeze(0)  # (1, 1, T*E, Dh/2)
        k = torch.view_as_real(kc * rope_k).flatten(-2).type_as(x)

        # Causal mask: Q pos i attends to K positions from input positions 0..i.
        # K is laid out as [tok0_e0, tok0_e1, ..., tok0_eE-1, tok1_e0, ...].
        # So Q[i] can attend to K[0..E*(i+1)-1].
        # Build (T, T*E) mask.
        q_idx = torch.arange(T, device=x.device)
        k_src = torch.arange(T * E, device=x.device).div(E, rounding_mode='floor')
        causal_mask = k_src.unsqueeze(0) <= q_idx.unsqueeze(1)  # (T, T*E)
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T*E)
        # Convert to float mask for SDPA (False -> -inf)
        attn_bias = torch.zeros(1, 1, T, T * E, device=x.device, dtype=x.dtype)
        attn_bias.masked_fill_(~attn_mask, float('-inf'))

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        h = self.o_proj(attn_out)

        h = h + self.ffn(self.ffn_norm(h))
        return h


class OuterMHALayer(nn.Module):
    """Two independent MHA+SwiGLU sub-layers combined via outer product.

    Computes the outer product of the two D-dim outputs at each position,
    then projects back to D.  This decomposes to a * (b @ proj), i.e. one
    stream gates the other through a learned linear transform.

    proj is initialized to identity so at init this equals element-wise
    multiply (same as MulMHALayer), but the model can learn richer
    cross-stream interactions.

    Stacker-compatible: takes pre-normed x, returns delta (no residual).

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length for RoPE.
        ffn_expand: SwiGLU expansion ratio.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_seq_len: int = 256,
                 ffn_expand: float = 8/3):
        super().__init__()
        self.layer_a = CausalMHALayer(dim, n_heads, max_seq_len, ffn_expand)
        self.layer_b = CausalMHALayer(dim, n_heads, max_seq_len, ffn_expand)
        # Projection: (D, D) — initialized to identity so at init = element-wise mul
        self.proj = nn.Parameter(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.layer_a(x)
        b = self.layer_b(x)
        return a * (b @ self.proj)


class BidirectionalMHALayer(nn.Module):
    """MHA + SwiGLU layer compatible with ULB stackers. Supports causal or bidirectional.

    Same as CausalMHALayer but with configurable causal mask.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_seq_len: int = 256,
                 ffn_expand: float = 8/3, is_causal: bool = True):
        super().__init__()
        self.is_causal = is_causal
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

        if HAS_FLASH_ATTN and not self.is_causal:
            q = q.transpose(1, 2).to(torch.bfloat16)
            k = k.transpose(1, 2).to(torch.bfloat16)
            v = v.transpose(1, 2).to(torch.bfloat16)
            attn_out = flash_attn_func(q, k, v, causal=False)
            attn_out = attn_out.to(x.dtype).contiguous().view(B, T, D)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        h = self.o_proj(attn_out)

        h = h + self.ffn(self.ffn_norm(h))
        return h
