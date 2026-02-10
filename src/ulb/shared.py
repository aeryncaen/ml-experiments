"""Weight sharing for expert pools.

SharedLinear and SharedParam allow experts to share a fraction of their
weights via a common base parameter. Each expert has:

    W_effective = shared * alpha + private * (1 - alpha)

where alpha is the shared fraction (0.0 = fully independent, 1.0 = fully shared).

Usage:
    share_expert_weights(expert_list, fraction=0.4)
    # Mutates experts in-place, replacing nn.Linear and LearnableSwish
    # with SharedLinear and SharedParam that reference common base params.
    # Returns nn.ParameterDict of shared params (caller must register it).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import LearnableSwish


class SharedLinear(nn.Module):
    """Linear layer with shared + private weight decomposition.

    W_effective = shared_weight * alpha + private_weight * (1 - alpha)
    Same for bias if present.

    Args:
        shared_weight: Reference to the shared nn.Parameter.
        shared_bias: Reference to the shared bias nn.Parameter, or None.
        private_weight: This expert's private nn.Parameter.
        private_bias: This expert's private bias nn.Parameter, or None.
        alpha: Shared fraction (0.0 = fully private, 1.0 = fully shared).
    """

    def __init__(self, shared_weight: nn.Parameter, shared_bias: nn.Parameter | None,
                 private_weight: nn.Parameter, private_bias: nn.Parameter | None,
                 alpha: float):
        super().__init__()
        self.shared_weight = shared_weight
        self.shared_bias = shared_bias
        self.private_weight = private_weight
        self.private_bias = private_bias
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        w = self.shared_weight * a + self.private_weight * (1.0 - a)
        b = None
        if self.shared_bias is not None and self.private_bias is not None:
            b = self.shared_bias * a + self.private_bias * (1.0 - a)
        return F.linear(x, w, b)


class SharedLearnableSwish(nn.Module):
    """LearnableSwish with shared + private beta decomposition.

    beta_effective = shared_beta * alpha + private_beta * (1 - alpha)

    Args:
        shared_beta: Reference to the shared nn.Parameter.
        private_beta: This expert's private nn.Parameter.
        alpha: Shared fraction.
    """

    def __init__(self, shared_beta: nn.Parameter, private_beta: nn.Parameter,
                 alpha: float):
        super().__init__()
        self.shared_beta = shared_beta
        self.private_beta = private_beta
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        beta = self.shared_beta * a + self.private_beta * (1.0 - a)
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

    Walks each expert's submodules, finds nn.Linear and LearnableSwish instances,
    creates one shared parameter per unique name (from expert 0's weights),
    and replaces each expert's module with SharedLinear/SharedLearnableSwish.

    The original weights become the private parameters. The shared parameters
    are initialized as a copy of the mean across all experts.

    Args:
        experts: ModuleList of expert blocks (mutated in-place).
        fraction: Shared weight fraction (0.0 = no sharing, 1.0 = fully shared).

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
            # Average weights across experts for shared init
            weights = [_get_nested_attr(experts[i], name).weight.data for i in range(pool_size)]
            mean_w = torch.stack(weights).mean(dim=0)
            shared_w = nn.Parameter(mean_w.clone())
            shared_params[f'{safe_name}_weight'] = shared_w

            # Bias (may be None)
            has_bias = _get_nested_attr(experts[0], name).bias is not None
            shared_b = None
            if has_bias:
                biases = [_get_nested_attr(experts[i], name).bias.data for i in range(pool_size)]
                mean_b = torch.stack(biases).mean(dim=0)
                shared_b = nn.Parameter(mean_b.clone())
                shared_params[f'{safe_name}_bias'] = shared_b

            # Replace in each expert
            for i in range(pool_size):
                orig = _get_nested_attr(experts[i], name)
                priv_w = nn.Parameter(orig.weight.data.clone())
                priv_b = nn.Parameter(orig.bias.data.clone()) if has_bias else None
                replacement = SharedLinear(shared_w, shared_b, priv_w, priv_b, fraction)
                _set_nested_attr(experts[i], name, replacement)

        elif kind == 'swish':
            # Average betas across experts
            betas = [_get_nested_attr(experts[i], name).beta.data for i in range(pool_size)]
            mean_beta = torch.stack(betas).mean(dim=0)
            shared_beta = nn.Parameter(mean_beta.clone())
            shared_params[f'{safe_name}_beta'] = shared_beta

            # Replace in each expert
            for i in range(pool_size):
                orig = _get_nested_attr(experts[i], name)
                priv_beta = nn.Parameter(orig.beta.data.clone())
                replacement = SharedLearnableSwish(shared_beta, priv_beta, fraction)
                _set_nested_attr(experts[i], name, replacement)

    return shared_params
