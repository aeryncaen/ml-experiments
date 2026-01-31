import torch
from torch import nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_mult: int=1,
        activation: callable=F.gelu,
        return_residual: bool=False,
        **kwargs
    ):
        super().__init__()
        in_features, out_features = d_model, d_model
        hidden_features = d_model * hidden_mult
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class GLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_mult: int=1,
        activation: callable=F.gelu,
        return_residual: bool=False,
        **kwargs
    ):
        super().__init__()
        in_features, out_features = d_model, d_model
        hidden_features = d_model * hidden_mult
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.activation = activation


    def forward(self, x):
        x = self.fc1(x) * self.activation(self.fc2(x))
        y = self.fc3(x)
        return y if not self.return_residual else (y, x)


class _LearnedAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2) * 0.1)

    def forward(self, x):
        w = self.w.clamp(0.05, 1.0)
        return F.silu(x) + w[0] * F.relu(x) + w[1] * torch.tanh(x)


class LearnedGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        return_residual: bool = False,
        **kwargs
    ):
        super().__init__()
        hidden = d_model * hidden_mult
        self.return_residual = return_residual
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(d_model, hidden)
        self.fc3 = nn.Linear(hidden, d_model)
        self.activation = _LearnedAct()

    def forward(self, x):
        x = self.fc1(x) * self.activation(self.fc2(x))
        y = self.fc3(x)
        return y if not self.return_residual else (y, x)
