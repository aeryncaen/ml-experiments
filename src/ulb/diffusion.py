"""DiffusionPoE — PoolOfExperts subclass for single-pass diffusion.

Each expert is a denoising step, routing decides the adaptive schedule,
and the whole diffusion completes in one forward pass.

Input: noisy 2D points (B, 2).
Output: denoised x0 prediction (B, 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .stack import PoolOfExperts


class MLPExpert(nn.Module):
    """Simple MLP block for (B, D) data. Residual-friendly: returns delta."""

    def __init__(self, dim: int, expand: float = 4.0):
        super().__init__()
        hidden = int(dim * expand)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionPoE(PoolOfExperts):
    """PoolOfExperts adapted for 2D point diffusion.

    Changes from parent:
    - Input/output are (B, 2), not (B, T, D).
    - input_proj lifts 2D → d_model, output_proj maps back.
    - No sequence pooling for routers (already (B, D)).
    - Experts are MLPs, not ULBBlocks.
    """

    def __init__(self, dim: int, input_dim: int = 2, pool_size: int = 4,
                 top_k: int = 2, max_hops: int | None = None,
                 expert_expand: float = 4.0,
                 router_noise: float = 1.0, router_dropout: float = 0.0):
        make_layer = lambda: MLPExpert(dim, expand=expert_expand)
        super().__init__(
            make_layer=make_layer,
            pool_size=pool_size,
            dim=dim,
            top_k=top_k,
            max_hops=max_hops,
            router_noise=router_noise,
            router_dropout=router_dropout,
        )
        self.input_proj = nn.Linear(input_dim, dim)
        self.output_proj = nn.Linear(dim, input_dim)

        # pool_size experts + 1 exit slot
        self.n_router_options = pool_size + 1
        self.stem_router = nn.Linear(dim, self.n_router_options, bias=False)
        self.expert_routers = nn.ModuleList([
            nn.Linear(dim, self.n_router_options, bias=False) for _ in range(pool_size)
        ])

        # No exit ramp — let routing discover depth organically
        self.exit_ramp_scale = 0.0

    def stem(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project (B, 2) → (B, D), run stem layer, produce initial logits.

        Args:
            x: Noisy input points (B, 2).

        Returns:
            x: Stem-processed hidden state (B, D).
            logits: Perturbed stem router logits (B, n_router_options).
        """
        x = self.input_proj(x)  # (B, D)
        x = x + self.stem_layer(self.stem_norm(x))
        logits = self._perturb_logits(self.stem_router(x))  # no pooling needed
        return x, logits

    def execute_hop(self, x: torch.Tensor, topk_idx: torch.Tensor,
                    topk_weights: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Run selected experts on (B, D) state, merge outputs and logits.

        Same as parent but no mean(dim=1) pooling for router — already (B, D).
        """
        B = x.shape[0]
        h = self.hop_norm(x)

        active_eids = topk_idx.unique()
        active_eids = active_eids[active_eids < self.pool_size]

        hop_aux = 0.0
        expert_outs = {}
        expert_logits = {}
        for eid in active_eids.tolist():
            e_out = self.experts[eid](h)  # (B, D)
            expert_outs[eid] = e_out
            expert_logits[eid] = self.expert_routers[eid](e_out)  # no pooling

            block_aux = getattr(self.experts[eid], 'aux_loss', 0.0)
            hop_aux = hop_aux + block_aux

        # Weighted merge — (B, D) not (B, T, D), so no None,None broadcasting
        out = torch.zeros_like(x)
        next_logits = torch.zeros(B, self.n_router_options, device=x.device, dtype=x.dtype)

        for k_idx in range(self.top_k):
            w = topk_weights[:, k_idx]  # (B,)
            eids = topk_idx[:, k_idx]   # (B,)
            for eid in active_eids.tolist():
                mask = eids == eid  # (B,)
                if not mask.any():
                    continue
                out = out + (mask[:, None].float() * w[:, None]) * expert_outs[eid]
                next_logits = next_logits + (mask[:, None].float() * w[:, None]) * expert_logits[eid]

        next_logits = self._perturb_logits(next_logits)
        return out, next_logits, hop_aux

    def finalize(self, x: torch.Tensor) -> torch.Tensor:
        """Exit layer + norm + project back to input space.

        Args:
            x: Hidden state after routing loop (B, D).

        Returns:
            x0 prediction (B, 2).
        """
        x = x + self.exit_layer(self.exit_norm(x))
        x = self.final_norm(x)
        return self.output_proj(x)
