"""ULB language model architectures and LLaDA diffusion wrapper.

CausalULB: ULB blocks, embed/head wrapper.
LLaDAModel: Masked diffusion wrapper — works with any backbone.

Usage:
    model = CausalULB(vocab_size=256, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    logits = model(token_ids)  # (B, T, vocab_size)

    # LLaDA diffusion:
    llada = LLaDAModel(backbone, vocab_size=256, dim=128)
    logits = llada(token_ids, mask)  # (B, T, vocab_size)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 k_mix: str = 'lerp',
                 is_causal: bool = True,
                 embed_lerp: bool = False,
                 feat_attn: bool = False,
                 feat_c_h: int | None = None,
                 feat_c_w: int | None = None):
        super().__init__()
        from .block import ULBBlock, ULBConfig
        from .lerp import EmbeddingLerp
        from .norm import RMSNorm

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.embed_lerp: EmbeddingLerp | None = EmbeddingLerp(dim) if embed_lerp else None

        config = ULBConfig(
            d_model=dim,
            n_heads=n_heads,
            paired=paired,
            attn_mode=attn_mode,
            inner_ratio=inner_ratio,
            k_mix=k_mix,
            is_causal=is_causal,
            feat_attn=feat_attn,
            feat_c_h=feat_c_h,
            feat_c_w=feat_c_w,
        )
        self.blocks = nn.ModuleList([ULBBlock(config) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        # Megatron init
        from .block import ulb_megatron_init_
        ulb_megatron_init_(self, n_layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) token indices.

        Returns:
            (B, T, vocab_size) logits.
        """
        B, T = token_ids.shape
        x = self.token_embed(token_ids)
        if self.embed_lerp is not None:
            x = self.embed_lerp(x)

        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))

        return self.head(self.final_norm(x))


class CausalULB1D(nn.Module):
    """Causal language model using clean 1D ULB blocks.

    Same structure as CausalULB2D but with nn.Linear instead of 2D
    tensor projections. For ablating 2D vs 1D.

    Args:
        vocab_size: Token vocabulary size.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of blocks.
        max_seq_len: Maximum sequence length.
        use_blend: Use blend attention (default True).
        ffn_expand: Feat-attn MLP expansion ratio (default 8/3).
    """

    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12,
                 n_layers: int = 12, max_seq_len: int = 2048,
                 use_blend: bool = True, ffn_expand: float = 8/3):
        super().__init__()
        from .block import ULB1DBlock, ULB1DConfig
        from .norm import RMSNorm

        config = ULB1DConfig(
            d_model=d_model, n_heads=n_heads, ffn_expand=ffn_expand,
            is_causal=True, use_blend=use_blend,
        )
        self.d_model = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([ULB1DBlock(config) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        # Simple init — no megatron
        self._init_weights(std=0.02)

    def _init_weights(self, std: float = 0.02):
        cutoff = 2.0 * std
        for name, param in self.named_parameters():
            if param.dim() >= 2 and 'embed' not in name and 'head' not in name:
                if 'blend_gate_proj' in name:
                    pass  # already initialized in block __init__
                else:
                    nn.init.trunc_normal_(param, std=std, a=-cutoff, b=cutoff)
            elif 'embed' in name:
                nn.init.trunc_normal_(param, std=std, a=-cutoff, b=cutoff)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(token_ids)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        return self.head(self.final_norm(x))


class CausalULB2D(nn.Module):
    """Causal language model using true 2D ULB blocks.

    Tokens are embedded as flat (B, T, D) where D = n_features * desc_dim,
    reshaped to (B, T, n_features, desc_dim) for processing through
    ULB2DBlocks, then flattened back for the LM head.

    Args:
        vocab_size: Token vocabulary size.
        n_features: Number of features per token (also seq-attn heads).
        desc_dim: Descriptor dimension per feature. Must be divisible by 4 (RoPE).
        n_layers: Number of ULB2D blocks.
        max_seq_len: Maximum sequence length.
        use_blend: Use blend attention (default True).
        is_causal: Causal masking for seq-attn (default True).
    """

    def __init__(self, vocab_size: int, n_features: int = 8, desc_dim: int = 8,
                 n_layers: int = 4, max_seq_len: int = 256,
                 use_blend: bool = True,
                 is_causal: bool = True, ffn_expand: float = 8/3):
        super().__init__()
        from .block import ULB2DBlock, ULB2DConfig
        from .norm import RMSNorm

        config = ULB2DConfig(
            n_features=n_features, desc_dim=desc_dim, ffn_expand=ffn_expand,
            is_causal=is_causal, use_blend=use_blend,
        )
        dim = config.d_model
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_features = n_features
        self.desc_dim = desc_dim

        self.n_layers = n_layers
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([ULB2DBlock(config) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

        # Init
        self._init_weights(n_layers)

    def _init_weights(self, n_layers: int, std: float = 0.02):
        """Init for all 2D tensor params."""
        cutoff = 2.0 * std

        for name, param in self.named_parameters():
            if param.dim() == 4:
                if any(name.endswith(s) for s in
                         ('.w_blend',)):
                    pass  # keep zero-init
                else:
                    nn.init.trunc_normal_(param, std=std, a=-cutoff, b=cutoff)
            elif param.dim() == 2 and 'embed' in name:
                nn.init.trunc_normal_(param, std=std, a=-cutoff, b=cutoff)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) token indices.
        Returns:
            (B, T, vocab_size) logits.
        """
        B, T = token_ids.shape
        NF, DD = self.n_features, self.desc_dim

        x = self.token_embed(token_ids)  # (B, T, D)

        for norm, block in zip(self.norms, self.blocks):
            x_normed = norm(x).view(B, T, NF, DD)
            x = x + block(x_normed).reshape(B, T, -1)

        return self.head(self.final_norm(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + MLP projection.

    Maps scalar t in [0, 1] to a D-dimensional embedding that gets added
    to every token in the sequence. Same idea as DDPM / DiT.

    Args:
        dim: Output embedding dimension.
        hidden_dim: MLP hidden dimension (default 4 * dim).
        max_period: Maximum period for sinusoidal frequencies.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        hidden_dim = hidden_dim or 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0, 1]. Returns (B, dim)."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]  # (B, half)
        return torch.cat([args.cos(), args.sin()], dim=-1)  # (B, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float. Returns (B, dim)."""
        emb = self._sinusoidal_embed(t)
        return self.mlp(emb)


class LLaDAModel(nn.Module):
    """LLaDA masked diffusion wrapper.

    Wraps any backbone (CausalULB, BidirectionalTransformer, etc.) and adds:
    - A learned mask token embedding
    - Masking/unmasking logic for training and generation
    - Predictions over the original vocab (not the mask token)
    - Optional time conditioning (sinusoidal embedding of mask ratio)
    - Optional SUBS parameterization (clamp logits at unmasked positions)

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
        time_conditioning: If True, add sinusoidal time embedding to input.
        subs_parameterization: If True, clamp logits at unmasked positions
            to one-hot and set mask-token logit to -inf.
    """

    def __init__(self, backbone: nn.Module, vocab_size: int, dim: int,
                 time_conditioning: bool = False,
                 subs_parameterization: bool = False):
        super().__init__()
        self.backbone = backbone
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = backbone.max_seq_len
        self.time_conditioning = time_conditioning
        self.subs_parameterization = subs_parameterization

        # Learned mask token embedding
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)

        # Time conditioning
        self.time_embedder: TimestepEmbedder | None = None
        if time_conditioning:
            self.time_embedder = TimestepEmbedder(dim)

    def _embed_with_mask(self, token_ids: torch.Tensor, mask: torch.Tensor,
                         t: torch.Tensor | None = None) -> torch.Tensor:
        """Embed tokens, replacing masked positions with the mask embedding.

        Args:
            token_ids: (B, T) original token indices.
            mask: (B, T) bool — True means masked (replaced with mask_embed).
            t: (B,) float mask ratios in [0, 1], for time conditioning.

        Returns:
            (B, T, D) embeddings.
        """
        x = self.backbone.token_embed(token_ids)  # (B, T, D)
        x = torch.where(mask.unsqueeze(-1), self.mask_embed, x)

        if self.time_embedder is not None and t is not None:
            x = x + self.time_embedder(t)[:, None, :]  # broadcast over T

        return x

    def _run_backbone_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone's layers on pre-computed embeddings.

        Handles:
        - TransformerBlock-based (has rope_freqs): CausalTransformer, BidirectionalTransformer
        - ULB-based (has norms + blocks): CausalULB
        - StackedLM (has stacker): MoE/PoE wrapped models

        Args:
            x: (B, T, D) embeddings.

        Returns:
            (B, T, vocab_size) logits.
        """
        bb = self.backbone

        if hasattr(bb, 'stacker'):
            # StackedLM wrapping MoE/PoE stacker — stacker handles norms+residual
            x = bb.stacker(x)
            return bb.head(x)
        elif hasattr(bb, 'rope_freqs'):
            # TransformerBlock-based (Causal or Bidirectional)
            for block in bb.blocks:
                x = block(x, bb.rope_freqs)
        elif hasattr(bb, 'n_features'):
            # ULB2D-based — blocks expect (B, T, n_features, desc_dim)
            B, T, D = x.shape
            NF, DD = bb.n_features, bb.desc_dim
            for norm, block in zip(bb.norms, bb.blocks):
                x_normed = norm(x).view(B, T, NF, DD)
                x = x + block(x_normed).reshape(B, T, D)
        elif hasattr(bb, 'norms'):
            # ULB-based
            for norm, block in zip(bb.norms, bb.blocks):
                x = x + block(norm(x))
        else:
            raise ValueError(f"Unknown backbone type: {type(bb)}")

        return bb.head(bb.final_norm(x))

    def _apply_subs(self, logits: torch.Tensor, token_ids: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """SUBS parameterization: clamp logits at unmasked positions.

        For masked positions: set mask-token logit to -inf, then renormalize
        so output is a proper log-prob over the real vocab.

        For unmasked positions: force the output to be one-hot on the true token
        (logit = 0 for correct token, -inf for everything else).

        This means the model only needs to predict at masked positions.
        """
        NEG_INF = -1e6

        # For all positions: suppress any "mask token" logit
        # (We don't have a mask token in the vocab, but if the model tries
        # to predict outside the vocab, this is a safety measure.)

        # Renormalize at masked positions (already valid logits, just normalize)
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # At unmasked positions: force one-hot on ground truth
        unmasked = ~mask  # (B, T)
        if unmasked.any():
            one_hot_logits = torch.full_like(logits, NEG_INF)
            one_hot_logits.scatter_(-1, token_ids.unsqueeze(-1), 0.0)
            logits = torch.where(unmasked.unsqueeze(-1), one_hot_logits, logits)

        return logits

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor,
                t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for LLaDA.

        Args:
            token_ids: (B, T) ground-truth token indices (clean data x0).
            mask: (B, T) bool — True for masked positions.
            t: (B,) float mask ratios — used for time conditioning.
                If None and time_conditioning is enabled, mask ratio is
                computed from the mask.

        Returns:
            (B, T, vocab_size) logits — predictions for ALL positions,
            but loss should only be computed on masked positions.
        """
        # Compute t from mask if not provided and time conditioning is on
        if self.time_conditioning and t is None:
            t = mask.float().mean(dim=-1)  # (B,) actual mask ratio

        x = self._embed_with_mask(token_ids, mask, t)
        logits = self._run_backbone_from_embeddings(x)

        if self.subs_parameterization:
            logits = self._apply_subs(logits, token_ids, mask)

        return logits
