"""ULBBlock — Universal Learning Block.

Generalized from FusedGateBlock: a fused attention+MLP block that achieves
SSM-equivalent capabilities (state tracking, pattern matching) purely through
attention preprocessing, with zero sequential scan.

Architecture (forward pass):
    h_up = sub_expert_route(x, up_projs, swish_ups)  # per-token routed

    q, k, v = q_proj(h_up), k_proj(h_up), v_proj(h_up)
    q = q_norm(q) * q_bias       # per-head RMSNorm + post-norm bias
    k = k_norm(k) * k_bias       # (Mamba-3 style BC bias, init ones)

    k = causal_lerp(k, x)        # last 1/4 of head_dim blended with t-1
    q = acausal_lerp(q, x)       # last 1/2 of head_dim blended with t-1 and t+1

    dd_angles = rope.compute_dd_angles(x)
    q = rope(q, dd_angles)       # hybrid RoPE: fixed + data-dependent
    k = rope(k, dd_angles)

    y = attend(q, k, v)          # softmax, silu2, or blend

    y = attn_norm(y) * h_up      # skip-MULTIPLY (not skip-add)
    y = sub_expert_route(y, down_projs, swish_downs)  # per-token routed
    return y                      # caller does x = x + block(x)

Key design decisions (preserved from FusedGateBlock):
    - Skip-multiply, NOT skip-add
    - No internal residual — block returns delta only
    - V is left alone — no V blending
    - KV have bias, Q has no bias
    - No expansion — inner_dim = d_model
    - No conv — hurts retrieval
    - Learnable Swish (per-channel beta), not SiLU
    - Up/down projections are per-token routed sub-experts (default 4, top-2).
      Each sub-expert has its own linear + learned swish. Pointwise routing.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import LearnableSwish
from .attention import BlendAttention, silu2_attention
from .lerp import AcausalLerp, CausalLerp, QTemporalConv
from .rope import HybridRoPE, PairedRoPE


@dataclass
class ULBConfig:
    """Configuration for a ULBBlock.

    Args:
        d_model:    Model dimension.
        n_heads:    Number of attention heads. d_model must be divisible by n_heads.
        paired:     If True, use paired-head attention (interleave adjacent heads
                    at 2x sequence length). Requires n_heads to be even.
        attn_mode:  Attention mechanism: 'softmax', 'silu2', or 'blend'.
        rope_base:  Base for fixed RoPE inverse frequencies.
        k_lerp_bias:     Initial bias for K causal lerp gate (default -2.0).
        q_lerp_bias:     Initial bias for Q acausal lerp gates (default -2.0).
        blend_gate_bias: Initial bias for blend attention gate (default -1.1, ~25% silu2).
        n_sub_experts:   Number of sub-expert linear pairs for up/down projections.
                         Each sub-expert has its own linear + learned swish.
                         Routed per-token (pointwise, so no causality issue).
                         Default 4.
        sub_top_k:       How many sub-experts to activate per token (default 2).
        router_mode:     'topk' (default) or 'relu' (ReMoE-style differentiable routing).
        relu_lb:         If True (default), use load-balanced L1 regularization (Eq 10 in
                         ReMoE paper). If False, use plain L1 (Eq 9). Only relevant when
                         router_mode='relu'.
    """
    d_model: int = 128
    n_heads: int = 4
    paired: bool = True
    attn_mode: Literal['softmax', 'silu2', 'blend'] = 'blend'
    rope_base: float = 10000.0
    k_lerp_bias: float = -2.0
    q_lerp_bias: float = -2.0
    blend_gate_bias: float = -1.1
    q_mix: Literal['none', 'lerp', 'conv2', 'conv3'] = 'lerp'
    k_lerp: bool = True
    swish_mode: Literal['learnable', 'silu'] = 'learnable'
    n_sub_experts: int = 4
    sub_top_k: int = 2
    router_mode: Literal['topk', 'relu'] = 'topk'
    relu_lb: bool = False  # load-balanced L1 (Eq 10 in ReMoE paper)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.paired:
            assert self.n_heads % 2 == 0, (
                f"n_heads ({self.n_heads}) must be even for paired mode")
        head_dim = self.d_model // self.n_heads
        assert head_dim % 4 == 0, (
            f"head_dim ({head_dim}) must be divisible by 4 for hybrid RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def inner_dim(self) -> int:
        """No expansion: inner_dim = d_model."""
        return self.d_model


class ULBBlock(nn.Module):
    """Universal Learning Block.

    A fused attention+MLP block with SSM-equivalent preprocessing.
    No internal residual — caller adds residual. No internal pre-norm — caller pre-norms.

    Args:
        config: ULBConfig instance, or will be constructed from kwargs.
    """

    def __init__(self, config: ULBConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = ULBConfig(**kwargs)
        self.config = config

        d = config.d_model
        inner = config.inner_dim
        n_heads = config.n_heads
        head_dim = config.head_dim

        # --- Sub-expert routed projections ---
        # Per-token routed up/down projection pairs, each with its own
        # learned activation. Pointwise, so no causality issue.
        n_sub = config.n_sub_experts
        self.n_sub = n_sub
        self.sub_top_k = config.sub_top_k
        self.router_mode = config.router_mode
        self.relu_lb = config.relu_lb

        self.up_projs = nn.ModuleList([nn.Linear(d, inner, bias=False) for _ in range(n_sub)])
        self.down_projs = nn.ModuleList([nn.Linear(inner, d, bias=False) for _ in range(n_sub)])
        self.up_router = nn.Linear(d, n_sub, bias=False)
        self.down_router = nn.Linear(inner, n_sub, bias=False)

        # ReLU routing state (ReMoE paper: adaptive L1 sparsity regularization)
        if config.router_mode == 'relu':
            self._target_sparsity = 1.0 - config.sub_top_k / config.n_sub_experts
            self.register_buffer('_relu_lambda_up', torch.tensor(1e-8))
            self.register_buffer('_relu_lambda_down', torch.tensor(1e-8))
            self._relu_alpha = 1.2
        # aux_loss is always available; 0.0 for topk mode
        self.aux_loss = 0.0

        if config.swish_mode == 'learnable':
            self.swish_ups = nn.ModuleList([LearnableSwish(inner) for _ in range(n_sub)])
            self.swish_downs = nn.ModuleList([LearnableSwish(inner) for _ in range(n_sub)])
        elif config.swish_mode == 'silu':
            self.swish_ups = nn.ModuleList([nn.SiLU() for _ in range(n_sub)])
            self.swish_downs = nn.ModuleList([nn.SiLU() for _ in range(n_sub)])
        else:
            raise ValueError(f"Unknown swish_mode: {config.swish_mode}")

        # QKV: KV have bias, Q does not
        self.q_proj = nn.Linear(inner, inner, bias=False)
        self.k_proj = nn.Linear(inner, inner, bias=True)
        self.v_proj = nn.Linear(inner, inner, bias=True)

        # --- Post-attention norm (before skip-multiply) ---
        self.attn_norm = nn.RMSNorm(inner)

        # --- QK norm + post-norm bias (Mamba-3 style BC bias, init ones) ---
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.q_bias = nn.Parameter(torch.ones(head_dim))
        self.k_bias = nn.Parameter(torch.ones(head_dim))

        # --- Temporal lerps ---
        quarter_dim = head_dim // 4
        half_dim = head_dim // 2
        self.k_lerp: CausalLerp | None = None
        if config.k_lerp:
            self.k_lerp = CausalLerp(d, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        self.q_lerp: AcausalLerp | None = None
        self.q_conv: QTemporalConv | None = None
        if config.q_mix == 'lerp':
            self.q_lerp = AcausalLerp(d, n_heads, half_dim, init_bias=config.q_lerp_bias)
        elif config.q_mix == 'none':
            pass
        elif config.q_mix == 'conv2':
            self.q_conv = QTemporalConv(n_heads, half_dim, kernel_size=2)
        elif config.q_mix == 'conv3':
            self.q_conv = QTemporalConv(n_heads, half_dim, kernel_size=3)
        else:
            raise ValueError(f"Unknown q_mix mode: {config.q_mix}")

        # --- Hybrid RoPE ---
        self.rope = HybridRoPE(d, n_heads, head_dim, rope_base=config.rope_base)
        if config.paired:
            self.paired_rope = PairedRoPE(n_heads, head_dim, rope_base=config.rope_base)

        # --- Attention ---
        self.blend_attn: BlendAttention | None = None
        if config.attn_mode == 'blend':
            self.blend_attn = BlendAttention(
                d, n_heads, head_dim, init_bias=config.blend_gate_bias)

    def _sub_expert_forward(self, x: torch.Tensor, logits: torch.Tensor,
                            projs: nn.ModuleList, activations: nn.ModuleList,
                            relu_lambda: torch.Tensor | None = None,
                            ) -> tuple[torch.Tensor, torch.Tensor | float]:
        """Per-token routed sub-expert projection.

        Args:
            x: (B, T, D_in)
            logits: (B, T, n_sub) — pre-computed router logits
            projs: list of n_sub Linear(D_in, D_out)
            activations: list of n_sub activation modules
            relu_lambda: adaptive L1 coefficient buffer (only for relu mode)

        Returns:
            (output, aux_loss) — output is (B, T, D_out), aux_loss is scalar.
        """
        # Run all sub-experts (both modes need all outputs since n_sub is small)
        expert_outs = torch.stack(
            [act(proj(x)) for proj, act in zip(projs, activations)],
            dim=2)  # (B, T, n_sub, D_out)

        if self.router_mode == 'relu':
            return self._relu_route(logits, expert_outs, relu_lambda)
        else:
            return self._topk_route(logits, expert_outs)

    def _topk_route(self, logits: torch.Tensor, expert_outs: torch.Tensor,
                    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        """TopK + Softmax routing (original)."""
        topk_vals, topk_idx = logits.topk(self.sub_top_k, dim=-1)  # (B, T, k)
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B, T, k)
        D_out = expert_outs.shape[-1]
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D_out)  # (B, T, k, D_out)
        selected = expert_outs.gather(2, idx_expanded)  # (B, T, k, D_out)
        return (selected * topk_weights.unsqueeze(-1)).sum(dim=2), 0.0

    def _relu_route(self, logits: torch.Tensor, expert_outs: torch.Tensor,
                    relu_lambda: torch.Tensor | None,
                    ) -> tuple[torch.Tensor, torch.Tensor | float]:
        """ReLU routing with adaptive L1 sparsity regularization (ReMoE).

        Eq 5: weights = ReLU(logits)
        Eq 9/10: L_reg = mean of (f_e * weights_e) over all positions and experts
        Lambda adapted per step to hit target sparsity.
        """
        assert relu_lambda is not None, "relu_lambda buffer required for relu routing"
        weights = F.relu(logits)  # (B, T, n_sub) — non-negative, 0 = inactive

        # Weighted sum of expert outputs
        output = (expert_outs * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D_out)

        # --- Adaptive L1 regularization ---
        if self.training:
            B, T, E = weights.shape
            # Sparsity: fraction of zero entries
            with torch.no_grad():
                sparsity = (weights == 0).float().mean().item()
                # Adapt lambda: increase if too dense, decrease if too sparse
                sign = 1.0 if (self._target_sparsity - sparsity) > 0 else -1.0
                relu_lambda.mul_(self._relu_alpha ** sign)

            # L1 regularization (Eq 9 plain, Eq 10 load-balanced)
            if self.relu_lb:
                # f_e = (k/E) * T / count_e — overload ratio per expert
                with torch.no_grad():
                    active_counts = (weights > 0).float().sum(dim=(0, 1))  # (E,)
                    desired_ratio = self.sub_top_k / self.n_sub
                    f_e = desired_ratio * (B * T) / active_counts.clamp(min=1)  # (E,)
                # Weighted L1: penalize overloaded experts harder
                l_reg = (weights * f_e.unsqueeze(0).unsqueeze(0)).mean()
            else:
                # Plain L1 (Eq 9)
                l_reg = weights.mean()

            # Return differentiable loss (lambda is detached, l_reg has grad)
            return output, relu_lambda.detach() * l_reg
        else:
            return output, 0.0

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                blend_gate: torch.Tensor | None = None) -> torch.Tensor:
        """Dispatch to the configured attention mode.

        Args:
            q, k, v:    (B, H, T, D) — standard SDPA layout.
            blend_gate: (B, H, T, 1) — only used for 'blend' mode.
        """
        if self.config.attn_mode == 'silu2':
            return silu2_attention(q, k, v)
        elif self.config.attn_mode == 'blend':
            assert self.blend_attn is not None
            assert blend_gate is not None
            return self.blend_attn(q, k, v, blend_gate)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) — pre-normed input from the stacking model.

        Returns:
            (B, T, D) — the delta to add to the residual stream.
            Side effect: sets self.aux_loss (scalar tensor or 0.0).
        """
        cfg = self.config
        b, t, d = x.shape
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        # --- Up-project and activate (sub-expert routed) ---
        relu_lambda_up = getattr(self, '_relu_lambda_up', None)
        h_up, aux_up = self._sub_expert_forward(
            x, self.up_router(x), self.up_projs, self.swish_ups, relu_lambda_up)

        # --- QKV projections ---
        q = self.q_proj(h_up).view(b, t, n_heads, head_dim)
        k = self.k_proj(h_up).view(b, t, n_heads, head_dim)
        v = self.v_proj(h_up).view(b, t, n_heads, head_dim)

        # --- QK norm + post-norm bias ---
        q = self.q_norm(q) * self.q_bias
        k = self.k_norm(k) * self.k_bias

        # --- Temporal lerps (before RoPE) ---
        if self.k_lerp is not None:
            k = self.k_lerp(k, x)
        if self.q_lerp is not None:
            q = self.q_lerp(q, x)
        elif self.q_conv is not None:
            q = self.q_conv(q)

        # --- Data-dependent rotation angles (shared for Q and K) ---
        dd_angles = self.rope.compute_dd_angles(x)

        # --- Blend gate (only for blend mode) ---
        blend_gate = None
        if cfg.attn_mode == 'blend':
            assert self.blend_attn is not None
            blend_gate = self.blend_attn.compute_gate(x)  # (B, H, T, 1)

        # --- RoPE + Attention ---
        if cfg.paired:
            n2 = n_heads // 2
            # Pair adjacent heads: (B, T, H, hd) -> (B, T, H/2, 2*hd)
            q = q.view(b, t, n2, head_dim * 2)
            k = k.view(b, t, n2, head_dim * 2)
            q = self.paired_rope(q, dd_angles)
            k = self.paired_rope(k, dd_angles)
            # Interleave to 2T sequence: (B, T, H/2, 2*hd) -> (B, 2T, H/2, hd)
            q = q.view(b, t * 2, n2, head_dim)
            k = k.view(b, t * 2, n2, head_dim)
            v = v.reshape(b, t * 2, n2, head_dim)
            # Transpose to SDPA layout: (B, H, T, D)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if blend_gate is not None:
                # Paired doubles T: reshape gate (B,H,T,1) -> (B,n2,2T,1)
                bg = blend_gate.view(b, n2, 2, t, 1)
                blend_gate = bg.permute(0, 1, 3, 2, 4).reshape(b, n2, t * 2, 1)
            y = self._attend(q, k, v, blend_gate)
            # Undo paired layout: (B, H/2, 2T, hd) -> (B, T, H, hd)
            y = y.transpose(1, 2).contiguous().view(b, t, n_heads, head_dim)
        else:
            q = self.rope(q, dd_angles)
            k = self.rope(k, dd_angles)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = self._attend(q, k, v, blend_gate)
            y = y.transpose(1, 2)

        y = y.contiguous().view(b, t, cfg.inner_dim)

        # --- Skip-multiply (NOT skip-add) ---
        y = self.attn_norm(y) * h_up

        # --- Down-project with Swish (sub-expert routed) ---
        relu_lambda_down = getattr(self, '_relu_lambda_down', None)
        y, aux_down = self._sub_expert_forward(
            y, self.down_router(y), self.down_projs, self.swish_downs, relu_lambda_down)

        # Store aux_loss for caller to grab
        self.aux_loss = aux_up + aux_down

        return y
