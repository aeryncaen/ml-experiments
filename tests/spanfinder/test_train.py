import pytest
import torch
from heuristic_secrets.spanfinder.train import (
    SpanFinderTrainer,
    TrainingConfig,
    TrainingResult,
)
from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig


class TestSpanFinderTrainer:
    def test_train_single_epoch(self):
        model = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))

        X = torch.randint(0, 256, (10, 100))
        y = torch.zeros(10, 100, 2)
        y[0, 20, 0] = 1.0
        y[0, 30, 1] = 1.0

        trainer = SpanFinderTrainer(
            model=model,
            config=TrainingConfig(epochs=1, batch_size=4, lr=0.01),
        )

        result = trainer.train(X, y)

        assert isinstance(result, TrainingResult)
        assert result.final_loss is not None
        assert len(result.loss_history) == 1

    def test_loss_decreases(self):
        model = SpanFinderModel(SpanFinderConfig(embed_dim=32, num_layers=2))

        X = torch.randint(0, 256, (20, 50))
        y = torch.zeros(20, 50, 2)
        y[:, 10, 0] = 1.0
        y[:, 20, 1] = 1.0

        trainer = SpanFinderTrainer(
            model=model,
            config=TrainingConfig(epochs=20, batch_size=4, lr=0.01),
        )

        result = trainer.train(X, y)

        assert result.loss_history[-1] < result.loss_history[0]
