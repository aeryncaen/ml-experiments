import torch
import torch.nn as nn


class BinaryHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        out_dim = 1 if n_classes <= 2 else n_classes
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.n_classes <= 2:
            return out.squeeze(-1)
        return out


class MaskHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.1
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)
