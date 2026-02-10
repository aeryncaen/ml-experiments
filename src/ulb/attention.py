"""Attention mode implementations: softmax, silu-squared, and blend.

Three attention mechanisms that can be selected per-block:

- softmax: Standard scaled dot-product attention (causal). The baseline.
- silu2:   SiLU-squared attention — silu(QK^T/sqrt(d))^2 * causal_mask.
           Unnormalized, no softmax. Has different gradient dynamics.
           Uses Triton flash-style tiled kernels on CUDA (no T^2 materialization).
- blend:   Output-level lerp between softmax and RMSNorm'd silu2, with a
           per-position, per-head content-dependent gate. Allows the model
           to use softmax where precision matters and silu2 where smooth
           gradients help.

Blend gate initialization: weight=0, bias=-1.1 (sigmoid ~ 25% silu2).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton kernels available only on CUDA
_HAS_TRITON = False
try:
    from .triton_silu2 import silu2_attention_triton
    _HAS_TRITON = True
except ImportError:
    pass

# Global flag to force reference impl even when Triton is available.
# Set to True to bypass Triton (e.g. for short sequences where it's slower).
USE_TRITON = True


def _silu2_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference silu2 attention (materializes T^2 matrix). Used on CPU/MPS."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = (q @ k.transpose(-2, -1)) * scale
    T = logits.shape[-1]
    causal_mask = torch.tril(torch.ones(T, T, device=logits.device, dtype=logits.dtype))
    weights = F.silu(logits) ** 2 * causal_mask
    return weights @ v


def silu2_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SiLU-squared attention: silu(logits)^2, unnormalized, causal.

    Uses Triton flash-style kernels on CUDA, reference implementation elsewhere.

    Args:
        q, k, v: (B, H, T, D) — standard SDPA layout.

    Returns:
        (B, H, T, D) attention output.
    """
    if _HAS_TRITON and USE_TRITON and q.is_cuda:
        return silu2_attention_triton(q, k, v)
    return _silu2_attention_ref(q, k, v)


class BlendAttention(nn.Module):
    """Blended attention: output-level lerp between softmax and silu2.

    y = (1 - gate) * softmax_attn(q,k,v) + gate * rmsnorm(silu2_attn(q,k,v))

    The RMSNorm on the silu2 branch normalizes its output to match softmax's
    scale, preventing one branch from dominating due to magnitude alone.

    The gate is per-position, per-head, computed from the input x:
        gate = sigmoid(blend_gate_proj(x))

    Args:
        d_model:   Input dimension (for the gate projection).
        n_heads:   Number of attention heads.
        head_dim:  Dimension per head (for the silu2 RMSNorm).
        init_bias: Blend gate bias initialization (default -1.1, sigmoid ~ 25%).
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int, init_bias: float = -1.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)
        self.silu2_norm = nn.RMSNorm(head_dim)

    def compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Compute blend gate from input.

        Args:
            x: (B, T, D) input tensor.

        Returns:
            (B, H, T, 1) gate values in [0, 1].
        """
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, H)
        return gate.transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        """Blended attention forward.

        Args:
            q, k, v: (B, H, T, D) — standard SDPA layout.
            gate:    (B, H, T, 1) — blend gate from compute_gate().

        Returns:
            (B, H, T, D) blended attention output.
        """
        y_softmax = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y_silu2 = self.silu2_norm(silu2_attention(q, k, v))
        return (1 - gate) * y_softmax + gate * y_silu2
