import pytest
import torch
from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig


class TestSpanFinderModel:
    def test_default_config(self):
        config = SpanFinderConfig()
        assert config.vocab_size == 256
        assert config.embed_dim == 64
        assert config.hidden_dim == 64
        assert config.num_layers == 3

    def test_forward_single_sequence(self):
        config = SpanFinderConfig()
        model = SpanFinderModel(config)

        x = torch.randint(0, 256, (1, 100))

        output = model(x)

        assert output.shape == (1, 100, 2)

    def test_forward_batch(self):
        config = SpanFinderConfig()
        model = SpanFinderModel(config)

        x = torch.randint(0, 256, (8, 512))

        output = model(x)

        assert output.shape == (8, 512, 2)

    def test_tiny_config(self):
        config = SpanFinderConfig(embed_dim=32, hidden_dim=32, num_layers=2)
        model = SpanFinderModel(config)

        x = torch.randint(0, 256, (1, 100))
        output = model(x)

        assert output.shape == (1, 100, 2)

    def test_output_range(self):
        config = SpanFinderConfig()
        model = SpanFinderModel(config)

        x = torch.randint(0, 256, (4, 50))
        output = model(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_medium_config(self):
        config = SpanFinderConfig(embed_dim=96, hidden_dim=96, num_layers=4)
        model = SpanFinderModel(config)

        x = torch.randint(0, 256, (1, 100))
        output = model(x)

        assert output.shape == (1, 100, 2)
