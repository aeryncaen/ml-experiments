import torch
from heuristic_secrets.validator.train import ValidatorTrainer, TrainMetrics
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig


def create_batches(features: torch.Tensor, y: torch.Tensor, batch_size: int = 32, seq_len: int = 32):
    """Create batches in format expected by ValidatorTrainer.
    
    Returns list of (bytes, features, labels, lengths) tuples.
    """
    batches = []
    n = len(features)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_size_actual = end - i
        batch_bytes = torch.randint(0, 256, (batch_size_actual, seq_len))
        batch_features = features[i:end]
        batch_labels = y[i:end].squeeze(-1)
        batch_lengths = torch.full((batch_size_actual,), seq_len)
        batches.append((batch_bytes, batch_features, batch_labels, batch_lengths))
    return batches


class TestValidatorTrainer:
    def test_train_single_epoch(self):
        model = ValidatorModel(ValidatorConfig(width=16, depth=1, mlp_dims=(8, 4)))

        X = torch.randn(100, 6)
        y = torch.randint(0, 2, (100, 1)).float()
        train_data = create_batches(X, y, batch_size=32)

        trainer = ValidatorTrainer(
            model=model,
            train_data=train_data,
            epochs=1,
            lr=0.01,
        )

        result = trainer.train_epoch(epoch=0)

        assert isinstance(result, TrainMetrics)
        assert result.loss is not None

    def test_train_multiple_epochs(self):
        model = ValidatorModel(ValidatorConfig(width=16, depth=1, mlp_dims=(8, 4)))

        X = torch.randn(100, 6)
        y = torch.randint(0, 2, (100, 1)).float()
        train_data = create_batches(X, y, batch_size=32)

        trainer = ValidatorTrainer(
            model=model,
            train_data=train_data,
            epochs=5,
            lr=0.01,
        )

        losses = []
        for epoch in range(5):
            result = trainer.train_epoch(epoch=epoch)
            losses.append(result.loss)

        assert len(losses) == 5

    def test_loss_decreases(self):
        model = ValidatorModel(ValidatorConfig(width=32, depth=2, mlp_dims=(16, 8)))

        X_pos = torch.randn(50, 6) + 2
        X_neg = torch.randn(50, 6) - 2
        X = torch.cat([X_pos, X_neg])
        y = torch.cat([torch.ones(50, 1), torch.zeros(50, 1)])
        train_data = create_batches(X, y, batch_size=32)

        trainer = ValidatorTrainer(
            model=model,
            train_data=train_data,
            epochs=20,
            lr=0.1,
        )

        losses = []
        for epoch in range(20):
            result = trainer.train_epoch(epoch=epoch)
            losses.append(result.loss)

        assert losses[-1] < losses[0]
