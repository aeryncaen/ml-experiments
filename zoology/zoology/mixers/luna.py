"""
LUNA: Linear Universal Neural Attention (Shahbazi et al., 2025).

Learned kernel feature map for linear attention.
phi(x) = (1/sqrt(M)) * [psi_l(w_i^T x)]_{i=1..M, l=1..L}
Linear attention: out = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1)
O(n * D^2) where D = M*L, linear in sequence length.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _chunked(fn, x, chunk_size):
    if x.shape[0] <= chunk_size:
        return fn(x)
    return torch.cat([fn(x[i:i+chunk_size]) for i in range(0, x.shape[0], chunk_size)], dim=0)


def _silu2(x):
    s = F.silu(x)
    return s * s


class LearnedAct(nn.Module):
    def __init__(self):
        super().__init__()
        # init equal weights: 1/3 each
        self.w = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        r = F.relu(x)
        s = F.silu(x)
        t = torch.tanh(x)
        return self.w[0] * r * r + self.w[1] * s * s + self.w[2] * t * t


_ACTS = {"relu": F.relu, "silu": F.silu, "gelu": F.gelu, "silu2": _silu2, "learned": None}


class ScalarMLP(nn.Module):
    def __init__(self, L: int = 1, hidden: int = 64, nonneg: bool = True, act: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, L)
        self.nonneg = nonneg
        if act == "learned":
            self.act = LearnedAct()
        else:
            self.act = _ACTS[act]

    def forward(self, u):  # (T, 1) -> (T, L)
        y = self.fc2(self.act(self.fc1(u)))
        if self.nonneg:
            y = F.relu(y)
        return y


class LearnableFeatureMap(nn.Module):
    def __init__(self, d: int, M: int, L: int, hidden: int = 64,
                 nonneg: bool = True, act: str = "relu", chunk: int = 1_000_000,
                 ch_rms: bool = True, ch_rms_target: float = 0.1):
        super().__init__()
        self.M = M
        self.L = L
        self.scale = 1.0 / math.sqrt(M)
        self.chunk = chunk
        self.ch_rms = ch_rms
        self.ch_rms_target = ch_rms_target

        # Task-specific projections: W_i^T x + b_i
        self.W = nn.Parameter(torch.randn(M, d) / math.sqrt(d))
        self.b = nn.Parameter(torch.zeros(M))

        # Shared channel MLP: scalar -> L channels
        self.channel_mlp = ScalarMLP(L, hidden, nonneg, act)

    def forward(self, x):  # (B, H, N, d) -> (B, H, N, M*L)
        B, H, N, d = x.shape

        # Project: (B,H,N,d) -> (B,H,N,M)
        proj = torch.einsum("md,bhnd->bhnm", self.W, x) + self.b

        # Apply channel MLP to each scalar
        u = proj.reshape(-1, 1).float()  # (B*H*N*M, 1)
        y = _chunked(self.channel_mlp, u, self.chunk)  # (B*H*N*M, L)
        y = y.to(x.dtype)
        y = y.view(B, H, N, self.M, self.L)

        if self.ch_rms:
            rms = torch.sqrt(y.pow(2).mean((0, 1, 2, 3)) + 1e-6)  # (L,)
            s = (self.ch_rms_target / (rms + 1e-6)).clamp(max=1.0)
            y = y * s.view(1, 1, 1, 1, self.L)

        return (y * self.scale).reshape(B, H, N, self.M * self.L)


class LUNA(nn.Module):
    """LUNA linear attention mixer for zoology harness.

    Args:
        d_model: model dimension
        num_heads: number of attention heads
        M: number of projection directions per head
        L: number of channel functions
        hidden: hidden dim of channel MLP
        layer_idx: unused, for zoology compat
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        M: int = 16,
        L: int = 4,
        hidden: int = 64,
        nonneg: bool = True,
        act: str = "relu",
        ch_rms: bool = True,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.D = M * L  # feature map output dim

        assert d_model % num_heads == 0

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learned feature map (shared across heads)
        self.phi = LearnableFeatureMap(
            d=self.head_dim, M=M, L=L, hidden=hidden, nonneg=nonneg, act=act, ch_rms=ch_rms,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H = self.num_heads
        d = self.head_dim
        D = self.D

        # QKV projections -> (B, H, N, d)
        Q = self.Wq(x).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.Wk(x).view(B, N, H, d).permute(0, 2, 1, 3)
        V = self.Wv(x).view(B, N, H, d).permute(0, 2, 1, 3)

        # Learned feature maps: (B, H, N, D)
        phi_Q = self.phi(Q)
        phi_K = self.phi(K)

        # Linear attention: phi(Q) @ (phi(K)^T @ V) with causal masking
        # For causal: cumulative sum formulation
        # KV = cumsum over n of phi_K[n] outer V[n]: (B, H, N, D, d)
        # But that's expensive. Use the standard causal linear attention trick:

        # Causal linear attention via cumulative sums
        # S_n = sum_{j<=n} phi_K_j^T v_j  -> (B, H, D, d), accumulated
        # z_n = sum_{j<=n} phi_K_j         -> (B, H, D), accumulated
        # out_n = phi_Q_n @ S_n / (phi_Q_n @ z_n)

        # phi_K: (B, H, N, D), V: (B, H, N, d)
        KV = torch.einsum("bhnd,bhnv->bhndv", phi_K, V)  # (B, H, N, D, d)
        KV_cum = KV.cumsum(dim=2)                          # (B, H, N, D, d)
        K_cum = phi_K.cumsum(dim=2)                        # (B, H, N, D)

        # out_n = sum_d phi_Q[n,d] * KV_cum[n,d,:] / (sum_d phi_Q[n,d] * K_cum[n,d])
        numer = torch.einsum("bhnd,bhndv->bhnv", phi_Q, KV_cum)  # (B, H, N, d)
        denom = torch.einsum("bhnd,bhnd->bhn", phi_Q, K_cum)     # (B, H, N)
        denom = denom.unsqueeze(-1).clamp(min=1e-6)

        out = numer / denom  # (B, H, N, d)

        # Reshape back
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        return self.out_proj(out)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class LUNAAttn(nn.Module):
    """LUNA linear attention → norm → residual → full causal softmax attention."""
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        M: int = 8,
        L: int = 4,
        hidden: int = 64,
        nonneg: bool = True,
        act: str = "relu",
        ch_rms: bool = True,
        dropout: float = 0.0,
        layer_idx: int = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.D = M * L
        assert d_model % num_heads == 0

        # LUNA stage
        self.luna_Wq = nn.Linear(d_model, d_model)
        self.luna_Wk = nn.Linear(d_model, d_model)
        self.luna_Wv = nn.Linear(d_model, d_model)
        self.luna_out = nn.Linear(d_model, d_model)
        self.phi = LearnableFeatureMap(
            d=self.head_dim, M=M, L=L, hidden=hidden, nonneg=nonneg, act=act, ch_rms=ch_rms,
        )
        self.luna_norm = RMSNorm(d_model)
        self.luna_gate = nn.Parameter(torch.ones(1) * 0.5)

        # Attention stage
        self.attn_Wq = nn.Linear(d_model, d_model)
        self.attn_Wk = nn.Linear(d_model, d_model)
        self.attn_Wv = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H = self.num_heads
        d = self.head_dim

        # --- LUNA stage ---
        Q = self.luna_Wq(x).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.luna_Wk(x).view(B, N, H, d).permute(0, 2, 1, 3)
        V = self.luna_Wv(x).view(B, N, H, d).permute(0, 2, 1, 3)

        phi_Q = self.phi(Q)  # (B, H, N, D)
        phi_K = self.phi(K)  # (B, H, N, D)

        KV = torch.einsum("bhnd,bhnv->bhndv", phi_K, V)  # (B, H, N, D, d)
        KV_cum = KV.cumsum(dim=2)
        K_cum = phi_K.cumsum(dim=2)

        luna_numer = torch.einsum("bhnd,bhndv->bhnv", phi_Q, KV_cum)  # (B, H, N, d)
        luna_denom = torch.einsum("bhnd,bhnd->bhn", phi_Q, K_cum).unsqueeze(-1).clamp(min=1e-6)
        luna_out = luna_numer / luna_denom  # (B, H, N, d)

        luna_out = luna_out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        luna_out = self.luna_out(luna_out)
        # norm + gated residual back to input
        g = self.luna_gate.clamp(0.1, 5.0)
        h = x + g * self.luna_norm(luna_out)

        # --- Full causal softmax attention on h ---
        Q2 = self.attn_Wq(h).view(B, N, H, d).permute(0, 2, 1, 3)
        K2 = self.attn_Wk(h).view(B, N, H, d).permute(0, 2, 1, 3)
        V2 = self.attn_Wv(h).view(B, N, H, d).permute(0, 2, 1, 3)

        scores = torch.einsum("bhnd,bhmd->bhnm", Q2, K2) * self.softmax_scale
        causal_mask = torch.triu(torch.full((N, N), -10000.0, device=x.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attn = torch.softmax(scores, dim=-1, dtype=V2.dtype)
        attn = F.dropout(attn, self.dropout_p if self.training else 0.0)

        out = torch.einsum("bhnm,bhmd->bhnd", attn, V2)  # (B, H, N, d)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.d_model)
        return self.attn_out(out)
