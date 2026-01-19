from dataclasses import dataclass, field
import random

import torch
import torch.nn as nn
from tqdm import tqdm

from heuristic_secrets.spanfinder.model import SpanFinderModel


def get_device() -> torch.device:
    """Get best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 0.001
    weight_decay: float = 0.01
    pos_weight: float = 1000.0  # Weight for positive examples (span boundaries are rare)
    seed: int | None = None
    device: str | None = None  # None = auto-detect


@dataclass
class TrainingResult:
    final_loss: float
    loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpanFinderTrainer:
    """Trainer for SpanFinder model with support for variable-length sequences."""

    def __init__(self, model: SpanFinderModel, config: TrainingConfig):
        self.config = config

        if config.seed is not None:
            set_seed(config.seed)

        # Setup device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = get_device()
        
        self.model = model.to(self.device)

        # BCE with logits - handles sigmoid internally, more numerically stable
        # pos_weight compensates for extreme class imbalance (boundaries are ~0.04% of positions)
        pos_weight = torch.tensor([config.pos_weight, config.pos_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked BCE loss, ignoring padded positions.
        
        Args:
            logits: (batch, seq_len, 2) - model output
            targets: (batch, seq_len, 2) - target labels
            lengths: (batch,) - actual sequence lengths
        """
        batch_size, max_len, _ = logits.shape
        
        # Create mask for valid (non-padded) positions
        mask = torch.arange(max_len, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(logits)  # (batch, seq_len, 2)
        
        # Compute per-element loss
        loss = self.criterion(logits, targets)
        
        # Apply mask and average over valid positions
        masked_loss = loss * mask
        total_loss = masked_loss.sum()
        num_valid = mask.sum()
        
        return total_loss / num_valid if num_valid > 0 else total_loss

    def train(
        self,
        train_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        val_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
        verbose: bool = True,
    ) -> TrainingResult:
        """Train on pre-created bucketed batches.
        
        Args:
            train_batches: List of (byte_ids, targets, lengths) from create_bucketed_batches
            val_batches: Optional validation batches
            verbose: Show progress bars
        """
        loss_history = []
        val_loss_history = []

        epoch_iter = range(self.config.epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Epochs", unit="epoch")

        for epoch in epoch_iter:
            # Training
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle batches each epoch
            if self.config.seed is not None:
                random.seed(self.config.seed + epoch)
            epoch_batches = train_batches.copy()
            random.shuffle(epoch_batches)

            batch_iter = epoch_batches
            if verbose:
                batch_iter = tqdm(
                    epoch_batches, 
                    desc=f"Epoch {epoch+1}/{self.config.epochs}", 
                    leave=False,
                    unit="batch"
                )

            for byte_ids, targets, lengths in batch_iter:
                # Move to device
                byte_ids = byte_ids.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                self.optimizer.zero_grad()

                logits = self.model(byte_ids)
                loss = self._compute_loss(logits, targets, lengths)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                
                if verbose and hasattr(batch_iter, 'set_postfix'):
                    batch_iter.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)

            # Validation
            val_loss = None
            if val_batches:
                val_loss = self._evaluate(val_batches)
                val_loss_history.append(val_loss)

            if verbose and hasattr(epoch_iter, 'set_postfix'):
                postfix = {"train_loss": f"{avg_loss:.4f}"}
                if val_loss is not None:
                    postfix["val_loss"] = f"{val_loss:.4f}"
                epoch_iter.set_postfix(postfix)

        return TrainingResult(
            final_loss=loss_history[-1],
            loss_history=loss_history,
            val_loss_history=val_loss_history,
        )

    def _evaluate(
        self,
        batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        """Evaluate on batches, return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for byte_ids, targets, lengths in batches:
                byte_ids = byte_ids.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                logits = self.model(byte_ids)
                loss = self._compute_loss(logits, targets, lengths)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate_spanfinder(
    model: SpanFinderModel,
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    threshold: float = 0.5,
    verbose: bool = False,
    device: torch.device | None = None,
) -> dict:
    """Evaluate SpanFinder on batches, computing precision/recall for start/end predictions.
    
    Returns dict with:
        - start_precision, start_recall, start_f1
        - end_precision, end_recall, end_f1
        - loss
    """
    if device is None:
        device = get_device()
    
    model = model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    start_tp = start_fp = start_fn = 0
    end_tp = end_fp = end_fn = 0
    total_loss = 0.0
    num_samples = 0

    batch_iter = batches
    if verbose:
        batch_iter = tqdm(batches, desc="Evaluating", unit="batch")

    with torch.no_grad():
        for byte_ids, targets, lengths in batch_iter:
            byte_ids = byte_ids.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            logits = model(byte_ids)
            probs = torch.sigmoid(logits)
            
            batch_size, max_len, _ = logits.shape
            
            for i in range(batch_size):
                seq_len = lengths[i].item()
                
                # Start predictions
                start_preds = (probs[i, :seq_len, 0] >= threshold).float()
                start_labels = targets[i, :seq_len, 0]
                
                start_tp += int(((start_preds == 1) & (start_labels == 1)).sum().item())
                start_fp += int(((start_preds == 1) & (start_labels == 0)).sum().item())
                start_fn += int(((start_preds == 0) & (start_labels == 1)).sum().item())
                
                # End predictions
                end_preds = (probs[i, :seq_len, 1] >= threshold).float()
                end_labels = targets[i, :seq_len, 1]
                
                end_tp += int(((end_preds == 1) & (end_labels == 1)).sum().item())
                end_fp += int(((end_preds == 1) & (end_labels == 0)).sum().item())
                end_fn += int(((end_preds == 0) & (end_labels == 1)).sum().item())
                
                num_samples += 1
            
            # Loss
            mask = torch.arange(max_len, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).expand_as(logits)
            loss = criterion(logits, targets)
            masked_loss = (loss * mask).sum() / mask.sum()
            total_loss += masked_loss.item()

    # Compute metrics
    def safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    start_prec = safe_div(start_tp, start_tp + start_fp)
    start_rec = safe_div(start_tp, start_tp + start_fn)
    start_f1 = safe_div(2 * start_prec * start_rec, start_prec + start_rec)

    end_prec = safe_div(end_tp, end_tp + end_fp)
    end_rec = safe_div(end_tp, end_tp + end_fn)
    end_f1 = safe_div(2 * end_prec * end_rec, end_prec + end_rec)

    return {
        "start_precision": start_prec,
        "start_recall": start_rec,
        "start_f1": start_f1,
        "end_precision": end_prec,
        "end_recall": end_rec,
        "end_f1": end_f1,
        "loss": total_loss / len(batches) if batches else 0.0,
    }
