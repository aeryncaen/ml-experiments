import pytest
import torch
from heuristic_secrets.validator.model import ValidatorModel, ValidatorConfig


class TestValidatorModel:
    def test_default_config(self):
        config = ValidatorConfig()
        assert config.input_dim == 6
        assert config.hidden_dims == [16, 8]

    def test_custom_config(self):
        config = ValidatorConfig(hidden_dims=[32, 16, 8])
        model = ValidatorModel(config)

        # 3 hidden (Linear+ReLU each) + output (Linear+Sigmoid) = 8 layers
        assert len(model.layers) == 8

    def test_forward_single_sample(self):
        model = ValidatorModel(ValidatorConfig())
        x = torch.randn(1, 6)

        output = model(x)

        assert output.shape == (1, 1)
        assert 0.0 <= output.item() <= 1.0

    def test_forward_batch(self):
        model = ValidatorModel(ValidatorConfig())
        x = torch.randn(32, 6)

        output = model(x)

        assert output.shape == (32, 1)
        assert (output >= 0.0).all() and (output <= 1.0).all()

    def test_tiny_config(self):
        config = ValidatorConfig(hidden_dims=[8, 4])
        model = ValidatorModel(config)
        x = torch.randn(1, 6)

        output = model(x)
        assert output.shape == (1, 1)

    def test_large_config(self):
        config = ValidatorConfig(hidden_dims=[64, 32, 16, 8])
        model = ValidatorModel(config)
        x = torch.randn(1, 6)

        output = model(x)
        assert output.shape == (1, 1)
