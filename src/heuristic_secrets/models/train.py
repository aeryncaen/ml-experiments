from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, Sequence
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from .composed import Model, ModelConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MaskBatch(NamedTuple):
    bytes: torch.Tensor
    masks: torch.Tensor
    lengths: torch.Tensor


class BinaryBatch(NamedTuple):
    bytes: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor | None = None
    features: torch.Tensor | None = None
    secret_ids: list[list[str]] | None = None


Batch = MaskBatch | BinaryBatch


@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 20
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    pos_weight: float = 1.0

    use_swa: bool = True
    swa_start_pct: float = 0.4
    swa_lr_mult: float = 0.05
    swa_anneal_epochs: int = 5

    hard_example_ratio: float = 0.3
    curriculum_start_epoch: int = 2
    curriculum_end_epoch: int | None = None

    threshold_search: bool = True
    threshold_search_steps: int = 50
    threshold_optimize_for: Literal["f1", "recall"] = "f1"
    threshold_min_precision: float = 0.5

    preload_to_device: bool = False

    offset_reg_weight: float = 0.01
    entropy_reg_weight: float = 0.001

    seed: int | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Metrics:
    loss: float
    precision: float
    recall: float
    f1: float
    accuracy: float = 0.0
    fp_rate: float = 0.0
    fn_rate: float = 0.0

    def __str__(self) -> str:
        parts = [f"loss={self.loss:.4f}"]
        if self.accuracy > 0:
            parts.append(f"acc={self.accuracy:.4f}")
        if self.precision > 0 or self.recall > 0 or self.f1 > 0:
            parts.extend(
                [
                    f"P={self.precision:.4f}",
                    f"R={self.recall:.4f}",
                    f"F1={self.f1:.4f}",
                ]
            )
        if self.fp_rate > 0 or self.fn_rate > 0:
            parts.append(f"FP%={self.fp_rate * 100:.1f}")
            parts.append(f"FN%={self.fn_rate * 100:.1f}")
        return " ".join(parts)


class Trainer:
    def __init__(
        self,
        model: Model,
        train_data: Sequence[Batch],
        val_data: Sequence[Batch] | None = None,
        config: TrainConfig | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or TrainConfig()
        if self.config.seed is not None:
            set_seed(self.config.seed)

        self.device = device or get_device()
        self.model = model.to(self.device)
        self.head_type = model.config.head.head_type
        self.n_classes = model.config.head.n_classes

        self._data_on_device = self.config.preload_to_device
        if self._data_on_device:
            self.train_data = [self._move_batch_to_device(b) for b in train_data]
            self.val_data = (
                [self._move_batch_to_device(b) for b in val_data] if val_data else None
            )
        else:
            self.train_data = list(train_data)
            self.val_data = list(val_data) if val_data else None

        decay_params, no_decay_params = self._get_param_groups(model)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.lr,
        )

        self._setup_schedulers()
        self._setup_swa()

        self.best_metric = 0.0
        self.best_state: dict | None = None
        self.optimal_threshold = 0.5
        self._batch_losses: list[float] = [0.0] * len(train_data)

    def _get_param_groups(self, model: Model) -> tuple[list, list]:
        no_decay_keywords = {
            "bias",
            "LayerNorm.weight",
            "norm.weight",
            "log_sigma",
            "A_log",
            ".D",
            "b_bias",
            "c_bias",
            "pool_query",
            "pool_queries",
            "lambda_proj.bias",
            "dt_proj.bias",
        }

        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            should_decay = True
            for keyword in no_decay_keywords:
                if keyword in name:
                    should_decay = False
                    break

            if should_decay:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        return decay_params, no_decay_params

    def _setup_schedulers(self) -> None:
        cfg = self.config
        num_batches = len(self.train_data)
        self._max_batches = int(num_batches * 1.5)
        total_steps = cfg.epochs * self._max_batches
        warmup_steps = cfg.warmup_epochs * self._max_batches

        pct_start = min(warmup_steps / total_steps, 0.3) if total_steps > 0 else 0.1
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )
        self._scheduler_steps = 0
        self._scheduler_total = total_steps

    def _setup_swa(self) -> None:
        cfg = self.config
        self.swa_active = False
        self.swa_model = None
        self.swa_scheduler = None
        self.swa_start_epoch = cfg.epochs + 1

        if not cfg.use_swa:
            return

        self.swa_start_epoch = int(cfg.epochs * cfg.swa_start_pct)
        self.swa_model = AveragedModel(self.model)
        swa_lr = cfg.lr * cfg.swa_lr_mult
        for group in self.optimizer.param_groups:
            group["swa_lr"] = swa_lr
        self.swa_scheduler = SWALR(
            self.optimizer,
            swa_lr=swa_lr,
            anneal_epochs=cfg.swa_anneal_epochs,
            anneal_strategy="cos",
        )

    def set_train_data(self, train_data: Sequence[Batch]) -> None:
        self.train_data = train_data
        if len(train_data) != len(self._batch_losses):
            self._batch_losses = [0.0] * len(train_data)

    def _get_hard_ratio(self, epoch: int) -> float:
        cfg = self.config
        end = cfg.curriculum_end_epoch or (cfg.epochs // 2)
        if epoch < cfg.curriculum_start_epoch:
            return 0.0
        if epoch >= end:
            return cfg.hard_example_ratio
        progress = (epoch - cfg.curriculum_start_epoch) / (
            end - cfg.curriculum_start_epoch
        )
        return cfg.hard_example_ratio * progress

    def _compute_mask_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, L = logits.shape
        pos_mask = torch.arange(L, device=self.device).unsqueeze(0) < lengths.unsqueeze(
            1
        )
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        if self.config.pos_weight != 1.0:
            weight = torch.where(targets == 1, self.config.pos_weight, 1.0)
            bce = bce * weight
        return (bce * pos_mask).sum() / pos_mask.sum()

    def _compute_binary_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.squeeze(-1) if logits.dim() > 1 else logits
        if self.config.pos_weight != 1.0:
            weight = torch.where(labels == 1, self.config.pos_weight, 1.0)
            return nn.functional.binary_cross_entropy_with_logits(
                logits, labels, weight=weight
            )
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def _compute_multiclass_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels.long())

    def _forward_batch(
        self, batch: Batch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.head_type == "mask":
            assert isinstance(batch, MaskBatch)
            result = self.model(batch.bytes)
        else:
            assert isinstance(batch, BinaryBatch)
            mask = None
            if batch.lengths is not None:
                L = batch.bytes.shape[1]
                mask = torch.arange(L, device=self.device).unsqueeze(
                    0
                ) >= batch.lengths.unsqueeze(1)
            result = self.model(batch.bytes, mask=mask, precomputed=batch.features)

        if isinstance(result, tuple):
            return result[0], result[1]
        return result, {}

    def _compute_loss(
        self, logits: torch.Tensor, batch: Batch, aux_losses: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.head_type == "mask":
            assert isinstance(batch, MaskBatch)
            main_loss = self._compute_mask_loss(logits, batch.masks, batch.lengths)
        else:
            assert isinstance(batch, BinaryBatch)
            if self.n_classes > 2:
                main_loss = self._compute_multiclass_loss(logits, batch.labels)
            else:
                main_loss = self._compute_binary_loss(logits, batch.labels)

        if aux_losses:
            if "offset_reg" in aux_losses:
                main_loss = (
                    main_loss + self.config.offset_reg_weight * aux_losses["offset_reg"]
                )
            if "entropy_reg" in aux_losses:
                main_loss = (
                    main_loss
                    + self.config.entropy_reg_weight * aux_losses["entropy_reg"]
                )

        return main_loss

    def _move_batch_to_device(self, batch: Batch) -> Batch:
        if isinstance(batch, MaskBatch):
            return MaskBatch(
                bytes=batch.bytes.to(self.device),
                masks=batch.masks.to(self.device),
                lengths=batch.lengths.to(self.device),
            )
        else:
            return BinaryBatch(
                bytes=batch.bytes.to(self.device),
                labels=batch.labels.to(self.device),
                lengths=batch.lengths.to(self.device)
                if batch.lengths is not None
                else None,
                features=batch.features.to(self.device)
                if batch.features is not None
                else None,
            )

    def _to_device(self, batch: Batch) -> Batch:
        if self._data_on_device:
            return batch
        return self._move_batch_to_device(batch)

    def train_epoch(self, epoch: int = 0) -> Metrics:
        self.model.train()
        cfg = self.config

        if (
            self.config.use_swa
            and epoch >= self.swa_start_epoch
            and not self.swa_active
        ):
            self.swa_active = True

        batch_indices = list(range(len(self.train_data)))
        if cfg.seed is not None:
            random.seed(cfg.seed + epoch)
        random.shuffle(batch_indices)

        hard_ratio = self._get_hard_ratio(epoch)
        if epoch > 0 and hard_ratio > 0:
            n_hard = int(len(batch_indices) * hard_ratio)
            if n_hard > 0:
                hard_indices = sorted(
                    range(len(self._batch_losses)),
                    key=lambda i: self._batch_losses[i],
                    reverse=True,
                )[:n_hard]
                batch_indices = batch_indices + hard_indices
                random.shuffle(batch_indices)

        total_loss = 0.0
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_samples = 0

        desc = "Training (SWA)" if self.swa_active else "Training"
        pbar = tqdm(batch_indices, desc=desc, leave=False)

        for batch_idx in pbar:
            batch = self._to_device(self.train_data[batch_idx])

            self.optimizer.zero_grad()
            logits, aux_losses = self._forward_batch(batch)
            loss = self._compute_loss(logits, batch, aux_losses)
            loss.backward()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.optimizer.step()

            if self.swa_active and self.swa_scheduler is not None:
                self.swa_scheduler.step()
            elif self._scheduler_steps < self._scheduler_total:
                self.scheduler.step()
                self._scheduler_steps += 1

            loss_val = loss.item()
            total_loss += loss_val
            self._batch_losses[batch_idx] = loss_val

            with torch.no_grad():
                tp, fp, fn, tn, n = self._compute_confusion(logits, batch)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                total_samples += n

            sched = (
                self.swa_scheduler
                if self.swa_active and self.swa_scheduler
                else self.scheduler
            )
            lr = sched.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

        if self.swa_active and self.swa_model is not None:
            self.swa_model.update_parameters(self.model)

        return self._build_metrics(
            total_loss,
            len(batch_indices),
            total_tp,
            total_fp,
            total_fn,
            total_tn,
            total_samples,
        )

    def _compute_confusion(
        self, logits: torch.Tensor, batch: Batch, threshold: float = 0.5
    ) -> tuple[int, int, int, int, int]:
        if self.head_type == "mask":
            assert isinstance(batch, MaskBatch)
            probs = torch.sigmoid(logits)
            B, L = logits.shape
            pos_mask = torch.arange(L, device=self.device).unsqueeze(
                0
            ) < batch.lengths.unsqueeze(1)
            preds = (probs >= threshold).float()
            valid_preds = preds[pos_mask]
            valid_targets = batch.masks[pos_mask]
            n = valid_preds.numel()
            tp = ((valid_preds == 1) & (valid_targets == 1)).sum().item()
            fp = ((valid_preds == 1) & (valid_targets == 0)).sum().item()
            fn = ((valid_preds == 0) & (valid_targets == 1)).sum().item()
            tn = ((valid_preds == 0) & (valid_targets == 0)).sum().item()
        elif self.n_classes > 2:
            assert isinstance(batch, BinaryBatch)
            preds = logits.argmax(dim=-1)
            targets = batch.labels.long()
            correct = (preds == targets).sum().item()
            n = len(batch.labels)
            tp = correct
            tn = 0
            fp = n - correct
            fn = 0
        else:
            assert isinstance(batch, BinaryBatch)
            probs = torch.sigmoid(logits)
            preds = (probs.view(-1) >= threshold).float()
            valid_preds = preds
            valid_targets = batch.labels
            n = len(batch.labels)
            tp = ((valid_preds == 1) & (valid_targets == 1)).sum().item()
            fp = ((valid_preds == 1) & (valid_targets == 0)).sum().item()
            fn = ((valid_preds == 0) & (valid_targets == 1)).sum().item()
            tn = ((valid_preds == 0) & (valid_targets == 0)).sum().item()

        return int(tp), int(fp), int(fn), int(tn), int(n)

    def _build_metrics(
        self,
        total_loss: float,
        n_batches: int,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        n_samples: int,
    ) -> Metrics:
        loss = total_loss / n_batches if n_batches > 0 else 0.0

        if self.n_classes > 2:
            accuracy = tp / n_samples if n_samples > 0 else 0.0
            return Metrics(
                loss=loss,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                accuracy=accuracy,
            )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return Metrics(
            loss=loss,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
        )

    def finalize_swa(self) -> None:
        if not self.swa_active or self.swa_model is None:
            return

        def bn_loader():
            for batch in self.train_data:
                yield (
                    batch.bytes if self._data_on_device else batch.bytes.to(self.device)
                )

        update_bn(bn_loader(), self.swa_model, device=self.device)
        self.model.load_state_dict(self.swa_model.module.state_dict())

    @torch.no_grad()
    def validate(self, optimize_threshold: bool = False) -> Metrics:
        if self.val_data is None:
            raise ValueError("No validation data provided")

        self.model.eval()

        all_probs: list[torch.Tensor] = []
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        total_loss = 0.0

        for batch in self.val_data:
            batch = self._to_device(batch)
            logits, aux_losses = self._forward_batch(batch)
            loss = self._compute_loss(logits, batch, aux_losses)
            total_loss += loss.item()

            if self.head_type == "mask":
                assert isinstance(batch, MaskBatch)
                probs = torch.sigmoid(logits)
                B, L = logits.shape
                pos_mask = torch.arange(L, device=self.device).unsqueeze(
                    0
                ) < batch.lengths.unsqueeze(1)
                all_probs.append(probs[pos_mask].cpu())
                all_targets.append(batch.masks[pos_mask].cpu())
            elif self.n_classes > 2:
                assert isinstance(batch, BinaryBatch)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_targets.append(batch.labels.cpu())
            else:
                assert isinstance(batch, BinaryBatch)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.view(-1).cpu())
                all_targets.append(batch.labels.cpu())

        all_targets_t = torch.cat(all_targets)

        if self.n_classes > 2 and self.head_type != "mask":
            all_preds_t = torch.cat(all_preds)
            correct = (all_preds_t == all_targets_t.long()).sum().item()
            total = len(all_targets_t)
            accuracy = correct / total if total > 0 else 0.0
            metrics = Metrics(
                loss=total_loss / len(self.val_data),
                accuracy=accuracy,
                precision=0.0,
                recall=0.0,
                f1=0.0,
            )
        else:
            all_probs_t = torch.cat(all_probs)

            if optimize_threshold and self.config.threshold_search:
                self.optimal_threshold = self._find_optimal_threshold(
                    all_probs_t, all_targets_t
                )

            preds = (all_probs_t >= self.optimal_threshold).float()
            tp = ((preds == 1) & (all_targets_t == 1)).sum().item()
            fp = ((preds == 1) & (all_targets_t == 0)).sum().item()
            fn = ((preds == 0) & (all_targets_t == 1)).sum().item()
            tn = ((preds == 0) & (all_targets_t == 0)).sum().item()

            metrics = self._build_metrics(
                total_loss,
                len(self.val_data),
                int(tp),
                int(fp),
                int(fn),
                int(tn),
                len(all_targets_t),
            )

        target_metric = (
            metrics.accuracy
            if self.n_classes > 2
            else (
                metrics.recall
                if self.config.threshold_optimize_for == "recall"
                else metrics.f1
            )
        )
        if target_metric > self.best_metric:
            self.best_metric = target_metric
            self.best_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

        return metrics

    def _find_optimal_threshold(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> float:
        cfg = self.config
        best_score = 0.0
        best_threshold = 0.5

        for t in torch.linspace(0.1, 0.9, cfg.threshold_search_steps):
            threshold = t.item()
            preds = (probs >= threshold).float()
            tp = ((preds == 1) & (targets == 1)).sum().item()
            fp = ((preds == 1) & (targets == 0)).sum().item()
            fn = ((preds == 0) & (targets == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            if cfg.threshold_optimize_for == "recall":
                if precision >= cfg.threshold_min_precision and recall > best_score:
                    best_score = recall
                    best_threshold = threshold
            else:
                if f1 > best_score:
                    best_score = f1
                    best_threshold = threshold

        return best_threshold

    def load_best_model(self) -> None:
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def save_checkpoint(self, path: Path, extra: dict | None = None) -> None:
        data = {
            "model_config": self.model.config.to_dict(),
            "train_config": self.config.to_dict(),
            "state_dict": self.model.state_dict(),
            "best_metric": self.best_metric,
            "best_state": self.best_state,
            "optimal_threshold": self.optimal_threshold,
        }
        if extra:
            data.update(extra)
        torch.save(data, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        device: torch.device | None = None,
    ) -> tuple[Model, "Trainer", dict]:
        device = device or get_device()
        data = torch.load(path, map_location=device, weights_only=False)

        model_config = ModelConfig.from_dict(data["model_config"])
        model = Model(model_config)

        state = data.get("best_state") or data["state_dict"]
        model.load_state_dict(state)
        model = model.to(device)

        train_config = TrainConfig.from_dict(data.get("train_config", {}))
        trainer = cls(model, [], config=train_config, device=device)
        trainer.best_metric = data.get("best_metric", 0.0)
        trainer.optimal_threshold = data.get("optimal_threshold", 0.5)

        return model, trainer, data


def compute_length_bins(lengths: list[int], n_bins: int = 8) -> list[int]:
    if not lengths:
        return [512]
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    bins = []
    for i in range(1, n_bins):
        idx = int(n * i / n_bins)
        bins.append(sorted_lens[idx])
    bins.append(sorted_lens[-1])
    return sorted(set(bins))


def create_mask_batches(
    byte_tensors: list[torch.Tensor],
    mask_tensors: list[torch.Tensor],
    lengths: list[int],
    batch_size: int,
    indices: list[int] | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    show_progress: bool = True,
) -> list[MaskBatch]:
    if seed is not None:
        random.seed(seed)

    if indices is None:
        indices = list(range(len(byte_tensors)))

    length_bins = compute_length_bins([lengths[i] for i in indices])
    bins: dict[int, list[int]] = {b: [] for b in length_bins}
    bins[-1] = []

    for idx in indices:
        length = lengths[idx]
        for bin_max in length_bins:
            if length <= bin_max:
                bins[bin_max].append(idx)
                break
        else:
            bins[-1].append(idx)

    total = sum((len(b) + batch_size - 1) // batch_size for b in bins.values() if b)
    batches: list[MaskBatch] = []
    pbar = tqdm(total=total, desc="Creating batches", disable=not show_progress)

    for bin_indices in bins.values():
        if not bin_indices:
            continue
        if shuffle:
            random.shuffle(bin_indices)

        for i in range(0, len(bin_indices), batch_size):
            batch_idx = bin_indices[i : i + batch_size]
            batch_bytes = pad_sequence(
                [byte_tensors[j] for j in batch_idx], batch_first=True, padding_value=0
            )
            batch_masks = pad_sequence(
                [mask_tensors[j] for j in batch_idx],
                batch_first=True,
                padding_value=0.0,
            )
            batch_lengths = torch.tensor([lengths[j] for j in batch_idx])
            batches.append(MaskBatch(batch_bytes, batch_masks, batch_lengths))
            pbar.update(1)

    pbar.close()
    if shuffle:
        random.shuffle(batches)
    return batches


def create_binary_batches(
    byte_tensors: list[torch.Tensor],
    labels: list[float],
    lengths: list[int],
    batch_size: int,
    feature_tensors: list[torch.Tensor] | None = None,
    secret_ids: list[list[str]] | None = None,
    indices: list[int] | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    show_progress: bool = True,
) -> list[BinaryBatch]:
    if seed is not None:
        random.seed(seed)

    if indices is None:
        indices = list(range(len(byte_tensors)))

    length_bins = compute_length_bins([lengths[i] for i in indices])
    bins: dict[int, list[int]] = {b: [] for b in length_bins}
    bins[-1] = []

    for idx in indices:
        length = lengths[idx]
        for bin_max in length_bins:
            if length <= bin_max:
                bins[bin_max].append(idx)
                break
        else:
            bins[-1].append(idx)

    total = sum((len(b) + batch_size - 1) // batch_size for b in bins.values() if b)
    batches: list[BinaryBatch] = []
    pbar = tqdm(total=total, desc="Creating batches", disable=not show_progress)

    for bin_indices in bins.values():
        if not bin_indices:
            continue
        if shuffle:
            random.shuffle(bin_indices)

        for i in range(0, len(bin_indices), batch_size):
            batch_idx = bin_indices[i : i + batch_size]
            batch_bytes = pad_sequence(
                [byte_tensors[j] for j in batch_idx], batch_first=True, padding_value=0
            )
            batch_labels = torch.tensor([labels[j] for j in batch_idx])
            batch_lengths = torch.tensor([lengths[j] for j in batch_idx])

            batch_features = None
            if feature_tensors is not None:
                batch_features = torch.stack([feature_tensors[j] for j in batch_idx])

            batch_sids = None
            if secret_ids is not None:
                batch_sids = [secret_ids[j] for j in batch_idx]

            batches.append(
                BinaryBatch(
                    batch_bytes, batch_labels, batch_lengths, batch_features, batch_sids
                )
            )
            pbar.update(1)

    pbar.close()
    if shuffle:
        random.shuffle(batches)
    return batches
