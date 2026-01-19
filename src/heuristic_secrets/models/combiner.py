import torch
import torch.nn as nn

from .features import FeatureMaker


class FeatureCombiner(nn.Module):
    def __init__(
        self,
        features: nn.ModuleList,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.1,
        use_gates: bool = False,
    ):
        super().__init__()
        self.features = features
        self.use_gates = use_gates

        input_dim = sum(f.output_dim for f in features)

        if use_gates:
            self.gate_bias = nn.Parameter(torch.zeros(input_dim))
            self.gate_scale = nn.Parameter(torch.ones(input_dim))

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        precomputed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_outputs = []

        for f in self.features:
            if hasattr(f, "index") and precomputed is not None:
                out = f(precomputed, mask)
            else:
                out = f(x, mask)
            feature_outputs.append(out)

        combined = torch.cat(feature_outputs, dim=-1)

        if self.use_gates:
            gates = torch.sigmoid(combined * self.gate_scale + self.gate_bias)
            combined = combined * gates

        return self.mlp(combined)
