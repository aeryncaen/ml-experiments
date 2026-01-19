from dataclasses import dataclass
import torch
import torch.nn as nn

from heuristic_secrets.bytemasker.model import ByteMaskerConfig, ByteMaskerModel
from heuristic_secrets.validator.model import ValidatorConfig, ValidatorModel
from heuristic_secrets.validator.features import FrequencyTable, char_frequency_difference


@dataclass
class JointConfig:
    masker: ByteMaskerConfig
    validator: ValidatorConfig
    prune_threshold: float = 0.5
    min_kept_bytes: int = 4

    def to_dict(self) -> dict:
        return {
            "masker": self.masker.to_dict(),
            "validator": self.validator.to_dict(),
            "prune_threshold": self.prune_threshold,
            "min_kept_bytes": self.min_kept_bytes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JointConfig":
        return cls(
            masker=ByteMaskerConfig.from_dict(d["masker"]),
            validator=ValidatorConfig.from_dict(d["validator"]),
            prune_threshold=d.get("prune_threshold", 0.5),
            min_kept_bytes=d.get("min_kept_bytes", 4),
        )

    @classmethod
    def default(cls) -> "JointConfig":
        return cls(
            masker=ByteMaskerConfig.small(),
            validator=ValidatorConfig(),
        )


class JointSecretDetector(nn.Module):
    def __init__(
        self,
        config: JointConfig,
        text_freq: FrequencyTable | None = None,
        secret_freq: FrequencyTable | None = None,
    ):
        super().__init__()
        self.config = config

        self.masker = ByteMaskerModel(config.masker)
        self.validator = ValidatorModel(config.validator)

        self.text_freq = text_freq
        self.secret_freq = secret_freq

    def set_frequency_tables(self, text_freq: FrequencyTable, secret_freq: FrequencyTable) -> None:
        self.text_freq = text_freq
        self.secret_freq = secret_freq

    def _compute_features(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B = byte_ids.shape[0]
        device = byte_ids.device
        features = torch.zeros(B, 2, device=device)

        if self.text_freq is None or self.secret_freq is None:
            return features

        for i in range(B):
            seq_len = int(lengths[i].item())
            byte_list = byte_ids[i, :seq_len].cpu().tolist()
            byte_data = bytes(byte_list)
            features[i, 0] = char_frequency_difference(byte_data, self.text_freq)
            features[i, 1] = char_frequency_difference(byte_data, self.secret_freq)

        return features

    def _prune(
        self,
        byte_ids: torch.Tensor,
        prune_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = byte_ids.shape
        device = byte_ids.device
        threshold = self.config.prune_threshold
        min_kept = self.config.min_kept_bytes

        pruned_ids_list = []
        pruned_lengths = []

        for i in range(B):
            seq_len = int(lengths[i].item())
            probs = prune_probs[i, :seq_len]
            keep_mask = probs > threshold
            keep_indices = torch.where(keep_mask)[0]

            if len(keep_indices) < min_kept:
                topk = min(min_kept, seq_len)
                keep_indices = probs.topk(topk).indices.sort().values

            pruned_ids_list.append(byte_ids[i, keep_indices])
            pruned_lengths.append(len(keep_indices))

        max_len = max(pruned_lengths)
        pruned_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, ids in enumerate(pruned_ids_list):
            pruned_ids[i, :len(ids)] = ids

        return pruned_ids, torch.tensor(pruned_lengths, device=device)

    def forward(
        self,
        byte_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, L = byte_ids.shape
        device = byte_ids.device

        if lengths is None:
            lengths = torch.full((B,), L, device=device, dtype=torch.long)

        mask_logits = self.masker(byte_ids)
        prune_probs = torch.sigmoid(mask_logits)

        pruned_ids, pruned_lengths = self._prune(byte_ids, prune_probs, lengths)

        features = self._compute_features(pruned_ids, pruned_lengths)

        cls_logits = self.validator(features, pruned_ids, pruned_lengths)

        return {
            "mask_logits": mask_logits,
            "prune_probs": prune_probs,
            "cls_logits": cls_logits,
            "pruned_lengths": pruned_lengths,
        }

    @torch.no_grad()
    def detect(
        self,
        byte_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        cls_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(byte_ids, lengths)
        cls_prob = torch.sigmoid(out["cls_logits"].squeeze(-1))
        return out["prune_probs"], cls_prob

    @classmethod
    def from_pretrained(
        cls,
        config: JointConfig,
        masker_path: str | None = None,
        validator_path: str | None = None,
        text_freq: FrequencyTable | None = None,
        secret_freq: FrequencyTable | None = None,
        device: torch.device | None = None,
    ) -> "JointSecretDetector":
        device = device or torch.device("cpu")
        model = cls(config, text_freq, secret_freq).to(device)

        if masker_path:
            ckpt = torch.load(masker_path, map_location=device, weights_only=False)
            state = ckpt.get("best_state") or ckpt["state_dict"]
            model.masker.load_state_dict(state)

        if validator_path:
            ckpt = torch.load(validator_path, map_location=device, weights_only=False)
            state = ckpt.get("best_state") or ckpt["state_dict"]
            model.validator.load_state_dict(state)

        return model
