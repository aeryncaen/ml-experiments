from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, Sequence
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch.nn.utils.rnn import pad_sequence
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from .composed import Model, ModelConfig


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_cuda(allow_tf32: bool = False) -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


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

    threshold_search: bool = False
    threshold_search_steps: int = 50
    threshold_optimize_for: Literal["f1", "recall", "gmean_r_f1"] = "gmean_r_f1"
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
    eval_time: float = 0.0
    eval_tokens_per_sec: float = 0.0

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
        if self.eval_time > 0:
            parts.append(f"[{self.eval_time:.2f}s {self.eval_tokens_per_sec:.0f}tok/s]")
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
        self._batch_times: list[tuple[int, float]] = []
        # (epoch, loop_idx, batch_idx, ms_per_token, mean, stddev)
        self._slow_batches: list[tuple[int, int, int, float, float, float]] = []
        self._ms_per_token_history: list[float] = []

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

    def train_epoch(self, epoch: int = 0, profile_slow: bool = False) -> Metrics:
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
        pbar = tqdm(batch_indices, desc=desc, leave=False, smoothing=0)
        
        self._batch_times = []
        self._ms_per_token_history = []

        for loop_idx, batch_idx in enumerate(pbar):
            batch_start = time.perf_counter()
            
            batch = self.train_data[batch_idx]

            if profile_slow:
                self._profile_slow_batch(batch, batch_idx)

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

            batch_time = time.perf_counter() - batch_start
            self._batch_times.append((batch_idx, batch_time))
            
            total_tokens = int(batch.lengths.sum()) if batch.lengths is not None else batch.bytes.numel()
            ms_per_token = (batch_time * 1000) / total_tokens
            self._ms_per_token_history.append(ms_per_token)
            
            if len(self._ms_per_token_history) >= 50:
                hist = self._ms_per_token_history
                mean_mpt = sum(hist) / len(hist)
                variance = sum((x - mean_mpt) ** 2 for x in hist) / len(hist)
                std_mpt = variance ** 0.5
                threshold = mean_mpt + std_mpt
                
                if ms_per_token > threshold:
                    self._slow_batches.append((epoch, loop_idx, batch_idx, ms_per_token, mean_mpt, std_mpt))
                    if profile_slow:
                        self._profile_slow_batch(batch, batch_idx)

            sched = (
                self.swa_scheduler
                if self.swa_active and self.swa_scheduler
                else self.scheduler
            )
            lr = sched.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}", t=f"{batch_time:.2f}s")

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

    def _profile_slow_batch(self, batch: "Batch", batch_idx: int) -> None:
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        
        with profile(activities=activities, record_shapes=True, with_stack=True, with_modules=True) as prof:
            self.optimizer.zero_grad()
            logits, aux = self._forward_batch(batch)
            loss = self._compute_loss(logits, batch, aux)
            loss.backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        
        sort_key = "cuda_time_total" if self.device.type == "cuda" else "cpu_time_total"
        B, L = batch.bytes.shape
        tokens = int(batch.lengths.sum()) if batch.lengths is not None else B * L
        tqdm.write(f"\n  Profiler for batch {batch_idx}: shape=({B}, {L}), tokens={tokens}")
        tqdm.write(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=30))
        
        tqdm.write("\n  Stack traces for aten::item calls:")
        events = list(prof.events() or [])
        item_events = [e for e in events if "item" in e.name.lower()]
        tqdm.write(f"  Found {len(item_events)} item events")
        for i, event in enumerate(item_events[:5]):
            tqdm.write(f"  [{i}] {event.name}: thread={event.thread}, time={event.cpu_time_total}us")
            if hasattr(event, 'stack') and event.stack:
                for frame in event.stack[:5]:
                    tqdm.write(f"      {frame}")
        
        prof.export_chrome_trace(f"/tmp/profile_batch_{batch_idx}.json")
        tqdm.write(f"\n  Full trace exported to /tmp/profile_batch_{batch_idx}.json")

    def profile_batch(self, batch_idx: int, warmup: int = 2, runs: int = 5) -> dict:
        """Profile a specific batch with torch profiler."""
        batch = self._to_device(self.train_data[batch_idx])
        
        for _ in range(warmup):
            self.optimizer.zero_grad()
            logits, aux = self._forward_batch(batch)
            loss = self._compute_loss(logits, batch, aux)
            loss.backward()
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            self.optimizer.zero_grad()
            logits, aux = self._forward_batch(batch)
            loss = self._compute_loss(logits, batch, aux)
            loss.backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        result = {
            "batch_idx": batch_idx,
            "batch_size": batch.bytes.shape[0],
            "seq_len": batch.bytes.shape[1],
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "times": times,
        }
        
        try:
            from torch.profiler import profile, ProfilerActivity
            activities = [ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            
            with profile(activities=activities, record_shapes=True) as prof:
                self.optimizer.zero_grad()
                logits, aux = self._forward_batch(batch)
                loss = self._compute_loss(logits, batch, aux)
                loss.backward()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            
            result["profile_table"] = prof.key_averages().table(
                sort_by="cpu_time_total" if self.device.type != "cuda" else "cuda_time_total",
                row_limit=20
            )
        except Exception as e:
            result["profile_error"] = str(e)
        
        return result

    def get_batch_time_stats(self) -> dict:
        """Get statistics about batch execution times from the last epoch."""
        if not self._batch_times:
            return {}
        times = [t for _, t in self._batch_times]
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        mpt_stats = {}
        if self._ms_per_token_history:
            sorted_mpt = sorted(self._ms_per_token_history)
            m = len(sorted_mpt)
            mean_mpt = sum(sorted_mpt) / m
            variance = sum((x - mean_mpt) ** 2 for x in sorted_mpt) / m
            mpt_stats = {
                "ms_per_token_mean": mean_mpt,
                "ms_per_token_std": variance ** 0.5,
                "ms_per_token_median": sorted_mpt[m // 2],
                "ms_per_token_p95": sorted_mpt[int(m * 0.95)],
            }
        
        return {
            "count": n,
            "mean": sum(times) / n,
            "median": sorted_times[n // 2],
            "p95": sorted_times[int(n * 0.95)],
            "p99": sorted_times[int(n * 0.99)],
            "min": sorted_times[0],
            "max": sorted_times[-1],
            "slow_batches": len(self._slow_batches),
            **mpt_stats,
        }

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
        total_tokens = 0

        eval_start = time.perf_counter()

        for batch in tqdm(self.val_data, desc="Validating", leave=False, smoothing=0):
            batch = self._to_device(batch)
            total_tokens += int(batch.lengths.sum()) if batch.lengths is not None else batch.bytes.numel()
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
                all_probs.append(probs[pos_mask])
                all_targets.append(batch.masks[pos_mask])
            elif self.n_classes > 2:
                assert isinstance(batch, BinaryBatch)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds)
                all_targets.append(batch.labels)
            else:
                assert isinstance(batch, BinaryBatch)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.view(-1))
                all_targets.append(batch.labels)

        all_targets_t = torch.cat(all_targets).cpu()

        if self.n_classes > 2 and self.head_type != "mask":
            all_preds_t = torch.cat(all_preds).cpu()
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
            all_probs_np = torch.cat(all_probs).cpu().numpy()
            all_targets_np = all_targets_t.numpy()

            if optimize_threshold and self.config.threshold_search:
                self.optimal_threshold = self._find_optimal_threshold_np(
                    all_probs_np, all_targets_np
                )

            preds = (all_probs_np >= self.optimal_threshold).astype(np.float32)
            tp = int(((preds == 1) & (all_targets_np == 1)).sum())
            fp = int(((preds == 1) & (all_targets_np == 0)).sum())
            fn = int(((preds == 0) & (all_targets_np == 1)).sum())
            tn = int(((preds == 0) & (all_targets_np == 0)).sum())

            metrics = self._build_metrics(
                total_loss,
                len(self.val_data),
                int(tp),
                int(fp),
                int(fn),
                int(tn),
                len(all_targets_t),
            )

        eval_time = time.perf_counter() - eval_start
        metrics.eval_time = eval_time
        metrics.eval_tokens_per_sec = total_tokens / eval_time if eval_time > 0 else 0.0

        target_metric = (
            metrics.accuracy
            if self.n_classes > 2
            else (
                metrics.recall
                if self.config.threshold_optimize_for == "recall"
                else metrics.f1
            )
        )
        if target_metric > self.best_metric + 0.001:
            self.best_metric = target_metric
            self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        return metrics

    def _find_optimal_threshold_np(
        self, probs: np.ndarray, targets: np.ndarray
    ) -> float:
        cfg = self.config
        best_score = 0.0
        best_threshold = 0.5

        thresholds = np.linspace(0.1, 0.9, cfg.threshold_search_steps)

        for threshold in thresholds:
            preds = (probs >= threshold).astype(np.float32)
            tp = ((preds == 1) & (targets == 1)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()
            fn = ((preds == 0) & (targets == 1)).sum()

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
            elif cfg.threshold_optimize_for == "gmean_r_f1":
                score = (recall * f1) ** 0.5
                if score > best_score:
                    best_score = score
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


def compute_length_bins(lengths: list[int]) -> list[int]:
    """Bins tuned for typical line-level secret detection data.
    
    Distribution has dense core at 0-70, sparse middle, then long tail clusters.
    """
    if not lengths:
        return [512]
    max_len = max(lengths)
    bins = [48, 80, 128, 288, 384, 512]
    return [b for b in bins if b <= max_len] or [max_len]


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
    pbar = tqdm(total=total, desc="Creating batches", disable=not show_progress, smoothing=0)

    for bin_indices in bins.values():
        if not bin_indices:
            continue
        bin_indices.sort(key=lambda i: lengths[i])

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
    pbar = tqdm(total=total, desc="Creating batches", disable=not show_progress, smoothing=0)

    for bin_indices in bins.values():
        if not bin_indices:
            continue
        bin_indices.sort(key=lambda i: lengths[i])

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
