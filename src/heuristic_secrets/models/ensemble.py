import torch
import torch.nn as nn
from torch.func import stack_module_state, functional_call
from typing import Callable

from .composed import Model, ModelConfig


class Ensemble(nn.Module):
    def __init__(self, config: ModelConfig, n_members: int = 5):
        super().__init__()
        self.config = config
        self.n_members = n_members
        self.models = nn.ModuleList([Model(config) for _ in range(n_members)])
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, precomputed: torch.Tensor | None = None) -> torch.Tensor:
        params, buffers = stack_module_state(self.models)
        
        def call_single(params, buffers, x, mask, precomputed):
            return functional_call(self.models[0], (params, buffers), (x, mask, precomputed))
        
        logits = torch.vmap(call_single, in_dims=(0, 0, None, None, None), randomness='different')(
            params, buffers, x, mask, precomputed
        )
        probs = torch.sigmoid(logits)
        return probs.mean(dim=0)
    
    def forward_with_variance(self, x: torch.Tensor, mask: torch.Tensor | None = None, precomputed: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        params, buffers = stack_module_state(self.models)
        
        def call_single(params, buffers, x, mask, precomputed):
            return functional_call(self.models[0], (params, buffers), (x, mask, precomputed))
        
        logits = torch.vmap(call_single, in_dims=(0, 0, None, None, None), randomness='different')(
            params, buffers, x, mask, precomputed
        )
        probs = torch.sigmoid(logits)
        return probs.mean(dim=0), probs.var(dim=0)
