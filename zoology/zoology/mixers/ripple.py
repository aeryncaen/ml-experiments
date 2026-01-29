"""
Ripple Attention mixer for the Zoology benchmark harness.

Wraps RippleAttention as a zoology-compatible sequence mixer.
The TransformerBlock in zoology already provides LayerNorm + MLP + residual,
so this wrapper only exposes the raw RippleAttention (multi-op attention).
"""
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from heuristic_secrets.models.ripple_attention import RippleAttention


class RippleMixer(nn.Module):
    """
    Zoology-compatible wrapper around RippleAttention.

    Zoology's TransformerBlock calls:
        sequence_mixer = config.sequence_mixer.instantiate(d_model=d_model, layer_idx=layer_idx)
    then:
        hidden_states = self.sequence_mixer(hidden_states)

    So we need __init__(d_model, layer_idx, **kwargs) and forward(x) -> x.
    """

    def __init__(
        self,
        d_model: int,
        layer_idx: int = None,
        num_heads: int = 4,
        order: str = "tele,conv,lowrank",
        max_kernel_size: int = None,
        max_seq_len: int = 8192,
        use_triton: bool = True,
        jacobi_iters: int = 1,
        siren_conv: bool = False,
        dim_divisor: int = 1,
        differential: bool = True,
        embed_residual: bool = True,
    ):
        super().__init__()
        import math
        if max_kernel_size is None:
            max_kernel_size = 16
        channels = d_model // dim_divisor
        self.d_model = d_model
        self.channels = channels
        self.layer_idx = layer_idx
        self.proj_in = nn.Linear(d_model, channels) if dim_divisor > 1 else nn.Identity()
        self.proj_out = nn.Linear(channels, d_model) if dim_divisor > 1 else nn.Identity()
        self.attn = RippleAttention(
            channels=channels,
            num_heads=num_heads,
            max_kernel_size=max_kernel_size,
            use_triton=use_triton,
            order=order,
            max_seq_len=max_seq_len,
            jacobi_iters=jacobi_iters,
            siren_conv=siren_conv,
            differential=differential,
            embed_residual=embed_residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        out, _info = self.attn(h)
        return self.proj_out(out)

    def state_size(self, sequence_length: int = 2048, **kwargs) -> int:
        return 0
