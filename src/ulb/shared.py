"""Weight sharing for expert pools.

SharedLinear splits output dimensions into shared and private slices.
Given fraction=0.4 and a (out_features, in_features) weight:
  - 40% of out_features come from a shared weight (same tensor across all experts)
  - 60% come from a private weight (unique per expert)
  - Forward: cat([shared(x), private(x)], dim=-1)

Total params go DOWN because the shared slice is stored once instead of N times.

Usage:
    share_expert_weights(expert_list, fraction=0.4)
    # Mutates experts in-place. Returns nn.ParameterDict of shared params
    # (caller must register it).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import LearnableSwish


class SharedLinear(nn.Module):
    """Linear with output dims split into shared + private slices.

    output = cat([F.linear(x, shared_weight, shared_bias),
                  F.linear(x, private_weight, private_bias)], dim=-1)

    Args:
        shared_weight: (shared_out, in_features) — same across all experts.
        shared_bias: (shared_out,) or None.
        private_weight: (private_out, in_features) — unique per expert.
        private_bias: (private_out,) or None.
    """

    def __init__(self, shared_weight: nn.Parameter, shared_bias: nn.Parameter | None,
                 private_weight: nn.Parameter, private_bias: nn.Parameter | None):
        super().__init__()
        self.shared_weight = shared_weight
        self.shared_bias = shared_bias
        self.private_weight = private_weight
        self.private_bias = private_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = F.linear(x, self.shared_weight, self.shared_bias)
        private_out = F.linear(x, self.private_weight, self.private_bias)
        return torch.cat([shared_out, private_out], dim=-1)


class SharedLearnableSwish(nn.Module):
    """LearnableSwish with beta split into shared + private slices.

    beta = cat([shared_beta, private_beta])
    output = x * sigmoid(beta * x)

    Args:
        shared_beta: (shared_dim,) — same across all experts.
        private_beta: (private_dim,) — unique per expert.
    """

    def __init__(self, shared_beta: nn.Parameter, private_beta: nn.Parameter):
        super().__init__()
        self.shared_beta = shared_beta
        self.private_beta = private_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.cat([self.shared_beta, self.private_beta])
        return x * torch.sigmoid(beta * x)


def _set_nested_attr(module: nn.Module, name: str, value: nn.Module):
    """Set a nested attribute like 'up_act' on module."""
    parts = name.split('.')
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], value)


def _get_nested_attr(module: nn.Module, name: str):
    """Get a nested attribute like 'up_act.beta' on module."""
    for part in name.split('.'):
        module = getattr(module, part)
    return module


def share_expert_weights(experts: nn.ModuleList, fraction: float
                         ) -> nn.ParameterDict:
    """Replace Linear and LearnableSwish modules in experts with shared versions.

    For each Linear(out, in), splits out_features into:
      shared_out = round(out * fraction)
      private_out = out - shared_out
    The shared slice is one parameter referenced by all experts.
    The private slice is unique per expert.

    For LearnableSwish(dim), splits dim the same way.

    Shared params are initialized from expert 0. Private params are initialized
    from each expert's original weights (the private slice portion).

    Args:
        experts: ModuleList of expert blocks (mutated in-place).
        fraction: Fraction of output dims to share (0.0 = no sharing, 1.0 = fully shared).

    Returns:
        nn.ParameterDict of shared parameters. Caller must register this
        on the parent module so they participate in optimization.
    """
    if fraction <= 0.0:
        return nn.ParameterDict()

    pool_size = len(experts)
    shared_params = nn.ParameterDict()

    # Collect all Linear and LearnableSwish submodule names from expert 0
    targets = []
    for name, mod in experts[0].named_modules():
        if isinstance(mod, nn.Linear):
            targets.append((name, 'linear'))
        elif isinstance(mod, LearnableSwish):
            targets.append((name, 'swish'))

    for name, kind in targets:
        safe_name = name.replace('.', '_')

        if kind == 'linear':
            orig0 = _get_nested_attr(experts[0], name)
            out_features = orig0.weight.shape[0]
            in_features = orig0.weight.shape[1]
            shared_out = round(out_features * fraction)
            if shared_out == 0:
                continue  # fraction too small for this layer

            # Shared slice: first shared_out rows, initialized from expert 0
            shared_w = nn.Parameter(orig0.weight.data[:shared_out].clone())
            shared_params[f'{safe_name}_weight'] = shared_w

            has_bias = orig0.bias is not None
            shared_b = None
            if has_bias:
                shared_b = nn.Parameter(orig0.bias.data[:shared_out].clone())
                shared_params[f'{safe_name}_bias'] = shared_b

            # Replace in each expert
            for i in range(pool_size):
                orig = _get_nested_attr(experts[i], name)
                priv_w = nn.Parameter(orig.weight.data[shared_out:].clone())
                priv_b = None
                if has_bias:
                    priv_b = nn.Parameter(orig.bias.data[shared_out:].clone())
                replacement = SharedLinear(shared_w, shared_b, priv_w, priv_b)
                _set_nested_attr(experts[i], name, replacement)

        elif kind == 'swish':
            orig0 = _get_nested_attr(experts[0], name)
            dim = orig0.beta.shape[0]
            shared_dim = round(dim * fraction)
            if shared_dim == 0:
                continue

            shared_beta = nn.Parameter(orig0.beta.data[:shared_dim].clone())
            shared_params[f'{safe_name}_beta'] = shared_beta

            for i in range(pool_size):
                orig = _get_nested_attr(experts[i], name)
                priv_beta = nn.Parameter(orig.beta.data[shared_dim:].clone())
                replacement = SharedLearnableSwish(shared_beta, priv_beta)
                _set_nested_attr(experts[i], name, replacement)

    return shared_params
