"""ULB — Universal Learning Block.

A fused attention+MLP block that achieves SSM-equivalent capabilities
(state tracking, pattern matching) purely through attention preprocessing,
with zero sequential scan.

Usage:
    from ulb import ULBBlock, ULBConfig, StackedULB, MoEStackedULB, PoolOfExperts

    # Single block
    block = ULBBlock(ULBConfig(d_model=128, n_heads=4, paired=True, attn_mode='blend'))

    # Stacked model (pre-norm residual)
    model = StackedULB(lambda: ULBBlock(config), n_layers=2, dim=128)

    # MoE stacked model
    model = MoEStackedULB(lambda: ULBBlock(config), n_layers=2, dim=128,
                          n_experts=4, top_k=2)
"""

from .activations import LearnableSwish
from .attention import BlendAttention, silu2_attention
from .block import ULBBlock, ULBConfig
from .lerp import AcausalLerp, CausalLerp, QTemporalConv
from .norm import RMSNorm
from .rope import HybridRoPE, PairedRoPE, apply_rotary
from .stack import MoEStackedULB, PoolOfExperts, StackedULB

__all__ = [
    # Core
    "ULBBlock",
    "ULBConfig",
    # Stacking
    "StackedULB",
    "MoEStackedULB",
    "PoolOfExperts",
    # Components (for independent use)
    "RMSNorm",
    "LearnableSwish",
    "HybridRoPE",
    "PairedRoPE",
    "apply_rotary",
    "CausalLerp",
    "AcausalLerp",
    "QTemporalConv",
    "BlendAttention",
    "silu2_attention",
]
