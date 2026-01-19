from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ByteEmbedding
from .backbone import (
    ConvBackbone,
    AttentionBackbone,
    SSMBackbone,
    SSMBackbone3,
    DeformConv1d,
    AttentionBlock,
    ContextualAttentionBlock,
    SSMBlock3,
    RMSNorm,
    SEBlock,
    MultiKernelSSMBlock,
    AdaptiveConvBiases,
)
from .features import PrecomputedFeature
from .heads import BinaryHead, MaskHead, ClassifierHead

DEBUG_NAN = True


def check_nan(t: torch.Tensor, name: str) -> None:
    if DEBUG_NAN and t.device.type == "cuda" and torch.isnan(t).any():
        print(f"NaN detected in {name}, shape={t.shape}, device={t.device}")
        print(f"  nan count: {torch.isnan(t).sum().item()}")
        print(f"  inf count: {torch.isinf(t).sum().item()}")
        raise RuntimeError(f"NaN in {name}")


@dataclass
class LayerConfig:
    embed_width: int = 32
    dropout: float = 0.1

    arch_type: Literal["conv", "attn", "ssm", "both", "conv_ssm", "unified"] = "both"

    conv_kernel_sizes: tuple[int, ...] = (3, 5, 7, 9)
    conv_groups: int = 2

    attn_depth: int = 2
    attn_heads: int = 2
    attn_ffn_mult: int = 4
    attn_window_size: int = 768
    attn_use_rope: bool = True
    num_attn_features: int = 4

    ssm_depth: int = 2
    ssm_state_size: int = 64
    ssm_n_heads: int = 4
    ssm_conv_kernel: int = 7
    ssm_kernel_sizes: tuple[int, ...] = (3, 5, 7, 9)
    ssm_expand: int = 2
    num_ssm_features: int = 4
    ssm_version: Literal[2, 3] = 3
    ssm_conv_type: Literal["none", "deform_se"] = "none"

    adaptive_conv: bool = False
    n_adaptive_branches: int = 3
    adaptive_kernel_size: int = 15
    adaptive_init_sigmas: tuple[float, ...] | None = None
    adaptive_min_sigma: float = 0.05
    adaptive_max_sigma: float = 0.5
    depthwise_conv: bool = False
    context_dim: int = 16

    num_embed_features: int = 4

    mlp_hidden_mult: int = 4
    mlp_output_dim: int = 32

    num_precomputed_features: int = 0

    num_heuristic_features: int = 4
    num_hidden_features: int = 8

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["conv_kernel_sizes"] = list(d["conv_kernel_sizes"])
        d["ssm_kernel_sizes"] = list(d["ssm_kernel_sizes"])
        if d["adaptive_init_sigmas"] is not None:
            d["adaptive_init_sigmas"] = list(d["adaptive_init_sigmas"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayerConfig":
        d = d.copy()
        if "conv_kernel_sizes" in d:
            d["conv_kernel_sizes"] = tuple(d["conv_kernel_sizes"])
        if "ssm_kernel_sizes" in d:
            d["ssm_kernel_sizes"] = tuple(d["ssm_kernel_sizes"])
        if "adaptive_init_sigmas" in d and d["adaptive_init_sigmas"] is not None:
            d["adaptive_init_sigmas"] = tuple(d["adaptive_init_sigmas"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HeadConfig:
    head_type: Literal["classifier", "mask"] = "classifier"
    n_classes: int = 2

    def to_dict(self) -> dict:
        return {"head_type": self.head_type, "n_classes": self.n_classes}

    @classmethod
    def from_dict(cls, d: dict) -> "HeadConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    n_layers: int = 1
    layer: LayerConfig | None = None
    head: HeadConfig | None = None

    def __post_init__(self):
        if self.layer is None:
            self.layer = LayerConfig()
        if self.head is None:
            self.head = HeadConfig()

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "layer": self.layer.to_dict(),
            "head": self.head.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = d.copy()
        layer_d = d.pop("layer", {})
        head_d = d.pop("head", {})
        layer = LayerConfig.from_dict(layer_d)
        head = HeadConfig.from_dict(head_d)
        return cls(
            layer=layer,
            head=head,
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )


class LearnedPooler(nn.Module):
    def __init__(self, width: int, num_queries: int = 1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, width))
        self.scale = width**-0.5

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, L, W = h.shape
        Q = self.queries.shape[0]

        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn = torch.bmm(q, h.transpose(1, 2)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        pooled = torch.bmm(attn, h)
        return pooled.reshape(B, Q * W)


class ConvBranch(nn.Module):
    def __init__(self, width: int, kernel_size: int, groups: int, dropout: float):
        super().__init__()
        self.conv = DeformConv1d(width, kernel_size, groups)
        self.norm = nn.LayerNorm(width)
        self.se = SEBlock(width)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(width, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.norm(self.conv(h))
        x = self.se(x, mask)
        x = self.dropout(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled = x.sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)
        return self.proj(pooled).squeeze(-1)


class EmbedPooler(nn.Module):
    def __init__(self, width: int, num_features: int):
        super().__init__()
        self.pooler = LearnedPooler(width, num_queries=1)
        self.proj = nn.Sequential(nn.Linear(width, num_features), nn.SiLU())

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        pooled = self.pooler(h, mask)
        return self.proj(pooled)


class AttentionGate(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        num_heads: int,
        ffn_mult: int,
        dropout: float,
        num_other_features: int,
        num_attn_features: int = 4,
    ):
        super().__init__()
        self.num_attn_features = num_attn_features
        self.backbone = AttentionBackbone(width, depth, num_heads, ffn_mult, dropout)
        self.pooler = LearnedPooler(width, num_queries=1)

        self.feature_proj = nn.Linear(width, num_attn_features)
        total_features = num_other_features + num_attn_features
        self.bias_proj = nn.Linear(width, total_features)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(h, mask)
        pooled = self.pooler(h, mask)

        attn_features = self.feature_proj(pooled)
        biases = self.bias_proj(pooled)
        return attn_features, biases


class SSMGate(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        state_size: int,
        conv_kernel: int,
        conv_groups: int,
        expand: int,
        dropout: float,
        num_other_features: int,
        num_ssm_features: int = 4,
    ):
        super().__init__()
        self.num_ssm_features = num_ssm_features
        self.backbone = SSMBackbone(
            width, depth, state_size, conv_kernel, conv_groups, expand, dropout
        )
        self.pooler = LearnedPooler(width, num_queries=1)

        self.feature_proj = nn.Linear(width, num_ssm_features)
        total_features = num_other_features + num_ssm_features
        self.bias_proj = nn.Linear(width, total_features)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(h, mask)
        pooled = self.pooler(h, mask)

        ssm_features = self.feature_proj(pooled)
        biases = self.bias_proj(pooled)
        return ssm_features, biases


class SSMGate3(nn.Module):
    """Mamba-3 style SSM gate with trapezoidal discretization and data-dependent RoPE."""

    def __init__(
        self,
        width: int,
        depth: int,
        state_size: int,
        n_heads: int,
        expand: int,
        dropout: float,
        num_other_features: int,
        num_ssm_features: int = 4,
        use_conv: bool = False,
        conv_kernel: int = 7,
        conv_groups: int = 4,
    ):
        super().__init__()
        self.num_ssm_features = num_ssm_features
        self.backbone = SSMBackbone3(
            width,
            depth,
            state_size,
            n_heads,
            expand,
            dropout,
            use_conv,
            conv_kernel,
            conv_groups,
        )
        self.pooler = LearnedPooler(width, num_queries=1)

        self.feature_proj = nn.Linear(width, num_ssm_features)
        total_features = num_other_features + num_ssm_features
        self.bias_proj = nn.Linear(width, total_features)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(h, mask)
        pooled = self.pooler(h, mask)

        ssm_features = self.feature_proj(pooled)
        biases = self.bias_proj(pooled)
        return ssm_features, biases


class HeuristicEmbed(nn.Module):
    def __init__(self, output_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


class GatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_mult: int = 4,
        output_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = input_dim * hidden_mult
        output_dim = output_dim or input_dim
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class FeatureIntegration(nn.Module):
    def __init__(
        self,
        width: int,
        feature_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.norm = RMSNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat_proj = self.proj(features).unsqueeze(1)
        return self.norm(h + self.dropout(feat_proj))


class BinaryModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.arch_type = config.arch_type
        w = config.embed_width

        self.embedding = ByteEmbedding(w, config.dropout)
        self.embed_pooler = (
            EmbedPooler(w, config.num_embed_features)
            if config.num_embed_features > 0
            else None
        )

        use_conv = config.arch_type in ("conv", "both", "conv_ssm")
        use_attn = config.arch_type in ("attn", "both")
        use_ssm = config.arch_type in ("ssm", "conv_ssm")

        self.conv_branches = None
        self.attn_gate = None
        self.ssm_gate = None

        num_embed = config.num_embed_features
        num_conv = len(config.conv_kernel_sizes) if use_conv else 0
        num_other = num_embed + num_conv + config.num_precomputed_features
        num_attn_features = config.num_attn_features if use_attn else 0
        num_ssm_features = config.num_ssm_features if use_ssm else 0

        if use_conv:
            self.conv_branches = nn.ModuleList(
                [
                    ConvBranch(w, ks, config.conv_groups, config.dropout)
                    for ks in config.conv_kernel_sizes
                ]
            )

        if use_attn:
            self.attn_gate = AttentionGate(
                w,
                config.attn_depth,
                config.attn_heads,
                config.attn_ffn_mult,
                config.dropout,
                num_other,
                num_attn_features,
            )

        if use_ssm:
            if config.ssm_version == 3:
                use_conv = config.ssm_conv_type == "deform_se"
                self.ssm_gate = SSMGate3(
                    w,
                    config.ssm_depth,
                    config.ssm_state_size,
                    config.ssm_n_heads,
                    config.ssm_expand,
                    config.dropout,
                    num_other,
                    num_ssm_features,
                    use_conv,
                    config.ssm_conv_kernel,
                    config.conv_groups,
                )
            else:
                self.ssm_gate = SSMGate(
                    w,
                    config.ssm_depth,
                    config.ssm_state_size,
                    config.ssm_conv_kernel,
                    config.conv_groups,
                    config.ssm_expand,
                    config.dropout,
                    num_other,
                    num_ssm_features,
                )

        num_features = num_other + num_attn_features + num_ssm_features
        self.mlp = GatedMLP(
            num_features, config.mlp_hidden_mult, config.mlp_output_dim, config.dropout
        )
        self.head = BinaryHead(config.mlp_output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        precomputed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.embedding(x)
        features_list = []

        if self.embed_pooler is not None:
            embed_features = self.embed_pooler(h, mask)
            features_list.append(embed_features)

        if self.conv_branches is not None:
            conv_features = torch.stack(
                [branch(h, mask) for branch in self.conv_branches], dim=-1
            )
            features_list.append(conv_features)

        if precomputed is not None:
            features_list.append(precomputed)

        gate_features_list = []
        biases = None

        if self.attn_gate is not None:
            attn_features, attn_biases = self.attn_gate(h, mask)
            gate_features_list.append(attn_features)
            biases = attn_biases

        if self.ssm_gate is not None:
            ssm_features, ssm_biases = self.ssm_gate(h, mask)
            gate_features_list.append(ssm_features)
            biases = ssm_biases if biases is None else biases + ssm_biases

        if gate_features_list and biases is not None:
            all_features = torch.cat(features_list + gate_features_list, dim=-1)
            features = all_features + biases
        else:
            features = torch.cat(features_list, dim=-1)

        return self.head(self.mlp(features))


class MaskModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        w = config.embed_width

        self.embedding = ByteEmbedding(w, config.dropout)

        if config.mask_backbone == "conv":
            self.backbone = ConvBackbone(
                w,
                config.mask_depth,
                config.mask_kernel_size,
                config.conv_groups,
                2.0,
                config.dropout,
            )
        elif config.mask_backbone == "ssm":
            if config.ssm_version == 3:
                use_conv = config.ssm_conv_type == "deform_se"
                self.backbone = SSMBackbone3(
                    w,
                    config.mask_depth,
                    config.ssm_state_size,
                    config.ssm_n_heads,
                    config.ssm_expand,
                    config.dropout,
                    use_conv,
                    config.ssm_conv_kernel,
                    config.conv_groups,
                )
            else:
                self.backbone = SSMBackbone(
                    w,
                    config.mask_depth,
                    config.ssm_state_size,
                    config.ssm_conv_kernel,
                    config.conv_groups,
                    config.ssm_expand,
                    config.dropout,
                )
        else:
            self.backbone = AttentionBackbone(
                w,
                config.mask_depth,
                config.attn_heads,
                config.attn_ffn_mult,
                config.dropout,
            )

        self.head = MaskHead(w, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        precomputed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.embedding(x)
        if isinstance(self.backbone, ConvBackbone):
            h = self.backbone(h)
        else:
            h = self.backbone(h, mask)
        return self.head(h)


class UnifiedLayer(nn.Module):
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        w = config.embed_width

        self.uses_adaptive_conv = config.adaptive_conv

        if config.adaptive_conv:
            if config.adaptive_init_sigmas is not None:
                init_sigmas = config.adaptive_init_sigmas
            else:
                n = config.n_adaptive_branches
                min_s, max_s = config.adaptive_min_sigma, config.adaptive_max_sigma
                init_sigmas = tuple(
                    min_s + (max_s - min_s) * i / (n - 1)
                    if n > 1
                    else (min_s + max_s) / 2
                    for i in range(n)
                )
            n_branches = len(init_sigmas)
            self.multi_kernel_ssm = n_branches > 1
        else:
            init_sigmas = None
            n_branches = len(config.ssm_kernel_sizes)
            self.multi_kernel_ssm = n_branches > 1

        n_envelope = n_branches if config.adaptive_conv else 0
        ctx_dim = config.context_dim if config.adaptive_conv else 0

        if config.adaptive_conv:
            self.attn_block = ContextualAttentionBlock(
                w,
                config.attn_heads,
                n_envelope,
                ctx_dim,
                config.attn_ffn_mult,
                config.dropout,
                window_size=config.attn_window_size,
                use_rope=config.attn_use_rope,
            )
            self.attn_pooler = None
        else:
            self.attn_block = AttentionBlock(
                w,
                config.attn_heads,
                config.attn_ffn_mult,
                config.dropout,
                window_size=config.attn_window_size,
                use_rope=config.attn_use_rope,
            )
            self.attn_pooler = LearnedPooler(w, num_queries=1)

        self.attn_proj = nn.Sequential(
            nn.Linear(w, config.num_attn_features), nn.SiLU()
        )

        if self.multi_kernel_ssm:
            self.ssm_block = MultiKernelSSMBlock(
                w,
                config.ssm_kernel_sizes,
                config.ssm_state_size,
                config.ssm_n_heads,
                config.ssm_expand,
                config.dropout,
                config.conv_groups,
                config.num_ssm_features,
                adaptive_conv=config.adaptive_conv,
                depthwise_conv=config.depthwise_conv,
                context_dim=ctx_dim,
                adaptive_kernel_size=config.adaptive_kernel_size,
                init_sigmas=init_sigmas,
                min_sigma=config.adaptive_min_sigma,
                max_sigma=config.adaptive_max_sigma,
                attn_window_size=config.attn_window_size,
                attn_use_rope=config.attn_use_rope,
            )
            self.ssm_pooler = None
            self.ssm_proj = None
        elif not config.adaptive_conv and config.ssm_kernel_sizes == (0,):
            self.ssm_block = SSMBlock3(
                w,
                config.ssm_state_size,
                config.ssm_n_heads,
                config.ssm_expand,
                config.dropout,
                use_conv=False,
            )
            self.ssm_pooler = LearnedPooler(w, num_queries=1)
            self.ssm_proj = nn.Sequential(
                nn.Linear(w, config.num_ssm_features), nn.SiLU()
            )
        else:
            kernel = (
                config.adaptive_kernel_size
                if config.adaptive_conv
                else (
                    config.ssm_kernel_sizes[0]
                    if config.ssm_kernel_sizes
                    else config.ssm_conv_kernel
                )
            )
            single_sigma = init_sigmas[0] if init_sigmas else 0.3
            self.ssm_block = SSMBlock3(
                w,
                config.ssm_state_size,
                config.ssm_n_heads,
                config.ssm_expand,
                config.dropout,
                use_conv=True,
                conv_kernel=kernel,
                conv_groups=config.conv_groups,
                adaptive_conv=config.adaptive_conv,
                depthwise_conv=config.depthwise_conv,
                init_sigma=single_sigma,
                min_sigma=config.adaptive_min_sigma,
                max_sigma=config.adaptive_max_sigma,
            )
            self.ssm_pooler = LearnedPooler(w, num_queries=1)
            self.ssm_proj = nn.Sequential(
                nn.Linear(w, config.num_ssm_features), nn.SiLU()
            )

        self.embed_pooler = EmbedPooler(w, config.num_embed_features)

        self.hidden_pooler = LearnedPooler(w, num_queries=1)
        self.hidden_proj = nn.Sequential(
            nn.Linear(w, config.num_hidden_features), nn.SiLU()
        )

        self.heuristic_embed = HeuristicEmbed(config.num_heuristic_features)

        self.total_features = (
            config.num_attn_features
            + config.num_ssm_features
            + config.num_embed_features
            + config.num_hidden_features
            + config.num_heuristic_features
        )
        self.feature_integration = FeatureIntegration(
            w, self.total_features, config.dropout
        )

        self._attn_pool_query = nn.Parameter(torch.randn(1, w))
        self._attn_pool_scale = w**-0.5

    def _pool_for_features(
        self, h: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        B = h.size(0)
        q = self._attn_pool_query.unsqueeze(0).expand(B, -1, -1)
        attn = torch.bmm(q, h.transpose(1, 2)) * self._attn_pool_scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, h).squeeze(1)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        precomputed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        check_nan(h, "layer_input")
        if self.uses_adaptive_conv:
            h, biases, pooler_context = self.attn_block(h, mask)
            check_nan(h, "attn_block_adaptive")
            attn_features = self.attn_proj(self._pool_for_features(h, mask))
        else:
            h = self.attn_block(h, mask)
            check_nan(h, "attn_block")
            attn_pooled = self.attn_pooler(h, mask)
            attn_features = self.attn_proj(attn_pooled)
            biases = None
            pooler_context = None

        aux_losses = {}
        if self.multi_kernel_ssm:
            h, ssm_features, aux_losses = self.ssm_block(
                h, h, mask, biases, pooler_context
            )
            check_nan(h, "multi_kernel_ssm")
        else:
            if self.uses_adaptive_conv and biases is not None:
                single_bias = AdaptiveConvBiases(
                    sigma=biases.sigma[:, 0:1].squeeze(-1)
                    if biases.sigma is not None
                    else None,
                    offset_scale=biases.offset_scale[:, 0:1].squeeze(-1)
                    if biases.offset_scale is not None
                    else None,
                    omega=biases.omega[:, 0:1].squeeze(-1)
                    if biases.omega is not None
                    else None,
                )
                h, aux_losses = self.ssm_block(h, mask, single_bias)
            else:
                h, aux_losses = self.ssm_block(h, mask)
            check_nan(h, "ssm_block")
            ssm_pooled = self.ssm_pooler(h, mask)
            ssm_features = self.ssm_proj(ssm_pooled)

        embed_features = self.embed_pooler(h, mask)

        hidden_pooled = self.hidden_pooler(h, mask)
        hidden_features = self.hidden_proj(hidden_pooled)

        if precomputed is not None:
            heuristic_features = self.heuristic_embed(precomputed)
        else:
            heuristic_features = torch.zeros(
                h.size(0),
                self.config.num_heuristic_features,
                device=h.device,
                dtype=h.dtype,
            )

        features = torch.cat(
            [
                attn_features,
                ssm_features,
                embed_features,
                hidden_features,
                heuristic_features,
            ],
            dim=-1,
        )

        h = self.feature_integration(h, features, mask)

        return h, features, aux_losses


class UnifiedModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layer_config = config.layer
        head_config = config.head
        assert layer_config is not None and head_config is not None
        w = layer_config.embed_width

        self.embedding = ByteEmbedding(w, layer_config.dropout)
        self.layers = nn.ModuleList(
            [UnifiedLayer(layer_config) for _ in range(config.n_layers)]
        )

        self.output_pooler = LearnedPooler(w, num_queries=1)

        if head_config.head_type == "classifier":
            self.head = ClassifierHead(w, head_config.n_classes)
        elif head_config.head_type == "mask":
            self.head = MaskHead(w)
        else:
            raise ValueError(f"Unknown head type: {head_config.head_type}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        precomputed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        all_aux = {}

        h = self.embedding(x)
        check_nan(h, "embedding")

        for i, layer in enumerate(self.layers):
            h, features, aux = layer(h, mask, precomputed)
            check_nan(h, f"layer{i}")
            for k, v in aux.items():
                all_aux[f"layer{i}_{k}"] = v

        pooled = self.output_pooler(h, mask)
        check_nan(pooled, "pooler")
        output = self.head(pooled)
        check_nan(output, "head")
        return output, all_aux


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._impl = UnifiedModel(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        precomputed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._impl(x, mask, precomputed)

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        mask: torch.Tensor | None = None,
        precomputed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            result = self.forward(x, mask, precomputed)
            logits = result[0]
            probs = torch.sigmoid(logits)
            return (probs >= threshold).long()
