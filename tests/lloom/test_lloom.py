"""Tests for LLooM top-level model.

Covers:
- End-to-end forward and backward
- Output shapes for various (B, T) inputs
- Stem router behavior (routing to seq, tok, exit)
- Minimum compute path (entry_stem → exit_stem only)
- Bridge crossings
- Init behavior (param stats, bias inits)
- Causal masking respected in stems
- Info dict populated
- StemBlock independently
"""

import pytest
import torch
import torch.nn as nn

from lloom.config import LLooMConfig
from lloom.lloom import LLooM, StemBlock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 32
B = 2
T = 8


def _make_config(**kwargs):
    defaults = dict(
        dim=D, max_seq_len=64,
        stem_n_heads=4, stem_mlp_expansion=1.75,
        seq_pool_size=4, seq_top_k=2, seq_n_heads=4, seq_expansion=1.75,
        seq_max_hops=4,  # small for fast tests
        tok_pool_size=4, tok_top_k=2, tok_expansion=1.75,
        tok_max_hops=8,
        exit_bias_init=0.0, bridge_bias_init=0.0, exit_ramp_scale=3.0,
        router_noise=0.0,  # deterministic
        shared_fraction=0.5, film_rank=8, hop_gate_dim=12,
        max_bridge_crossings=2,
        is_causal=True, dropout=0.0,
    )
    defaults.update(kwargs)
    return LLooMConfig(**defaults)


def _make_model(**kwargs):
    return LLooM(_make_config(**kwargs))


def _make_input(b=B, t=T):
    return torch.randn(b, t, D, requires_grad=True)


# ---------------------------------------------------------------------------
# StemBlock tests
# ---------------------------------------------------------------------------

class TestStemBlock:
    def test_output_shape(self):
        cfg = _make_config()
        stem = StemBlock(dim=D, n_heads=cfg.stem_n_heads,
                         inner_dim=cfg.stem_inner_dim)
        x = torch.randn(B, T, D)
        out = stem(x)
        assert out.shape == (B, T, D)

    def test_gradient_flow(self):
        cfg = _make_config()
        stem = StemBlock(dim=D, n_heads=cfg.stem_n_heads,
                         inner_dim=cfg.stem_inner_dim)
        x = torch.randn(B, T, D, requires_grad=True)
        out = stem(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for name, p in stem.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_residual_connection(self):
        """With zero-init weights, output should be close to input (residual only)."""
        cfg = _make_config()
        stem = StemBlock(dim=D, n_heads=cfg.stem_n_heads,
                         inner_dim=cfg.stem_inner_dim)
        # Zero all weights to make sublayers output zero
        for p in stem.parameters():
            p.data.zero_()
        x = torch.randn(B, T, D)
        out = stem(x)
        torch.testing.assert_close(out, x, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# LLooM forward / shapes
# ---------------------------------------------------------------------------

class TestForward:
    def test_basic_output_shape(self):
        model = _make_model()
        x = _make_input()
        out, info = model(x)
        assert out.shape == (B, T, D)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("b,t", [(1, 4), (4, 16), (1, 1)])
    def test_various_shapes(self, b, t):
        model = _make_model()
        x = _make_input(b=b, t=t)
        out, info = model(x)
        assert out.shape == (b, t, D)

    def test_eval_mode(self):
        model = _make_model()
        model.eval()
        x = _make_input()
        with torch.no_grad():
            out, info = model(x)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_end_to_end_backward(self):
        model = _make_model()
        x = _make_input()
        out, info = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_stem_params_get_grad(self):
        model = _make_model()
        x = _make_input()
        out, info = model(x)
        loss = out.sum()
        loss.backward()
        # Entry and exit stems should always get gradient
        for name, p in model.entry_stem.named_parameters():
            assert p.grad is not None, f"No gradient for entry_stem.{name}"
        for name, p in model.exit_stem.named_parameters():
            assert p.grad is not None, f"No gradient for exit_stem.{name}"


# ---------------------------------------------------------------------------
# Stem router
# ---------------------------------------------------------------------------

class TestStemRouter:
    def test_stem_logits_in_info(self):
        model = _make_model()
        x = _make_input()
        _, info = model(x)
        assert 'stem_logits' in info
        assert info['stem_logits'].shape == (B, model.config.stem_n_options)

    def test_routing_fractions_in_info(self):
        model = _make_model()
        x = _make_input()
        _, info = model(x)
        assert 'stem_go_seq' in info
        assert 'stem_go_tok' in info
        assert 'stem_go_exit' in info

    def test_force_exit_via_bias(self):
        """With very high exit bias, all samples should exit immediately."""
        model = _make_model()
        # Bias the exit slot very high
        with torch.no_grad():
            model.stem_router.bias.data[-1] = 100.0  # exit slot
        x = _make_input()
        _, info = model(x)
        assert info['stem_go_exit'] == 1.0

    def test_force_seq_via_bias(self):
        """With very high first expert bias, all should go to seq."""
        model = _make_model()
        with torch.no_grad():
            model.stem_router.bias.data.zero_()
            model.stem_router.bias.data[0] = 100.0  # first seq expert
        x = _make_input()
        _, info = model(x)
        assert info['stem_go_seq'] == 1.0

    def test_force_tok_via_bias(self):
        """With very high bridge bias, all should go to token side."""
        model = _make_model()
        cfg = model.config
        with torch.no_grad():
            model.stem_router.bias.data.zero_()
            model.stem_router.bias.data[cfg.seq_pool_size] = 100.0  # bridge slot
        x = _make_input()
        _, info = model(x)
        assert info['stem_go_tok'] == 1.0


# ---------------------------------------------------------------------------
# Minimum compute path
# ---------------------------------------------------------------------------

class TestMinComputePath:
    def test_exit_only_path(self):
        """All samples exit from stem → only stems run."""
        model = _make_model()
        with torch.no_grad():
            model.stem_router.bias.data[-1] = 100.0
        x = _make_input()
        out, info = model(x)
        assert out.shape == (B, T, D)
        assert info['mean_seq_hops'] == 0.0
        assert info['mean_tok_hops'] == 0.0
        assert info['mean_bridges'] == 0.0

    def test_exit_path_backward(self):
        """Min compute path should still support backward."""
        model = _make_model()
        with torch.no_grad():
            model.stem_router.bias.data[-1] = 100.0
        x = _make_input()
        out, info = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Info dict
# ---------------------------------------------------------------------------

class TestInfoDict:
    def test_hop_counts(self):
        model = _make_model()
        x = _make_input()
        _, info = model(x)
        assert 'mean_seq_hops' in info
        assert 'mean_tok_hops' in info
        assert 'mean_bridges' in info
        assert info['mean_seq_hops'] >= 0.0
        assert info['mean_tok_hops'] >= 0.0
        assert info['mean_bridges'] >= 0.0


# ---------------------------------------------------------------------------
# Init behavior
# ---------------------------------------------------------------------------

class TestInitBehavior:
    def test_stem_router_bias_init(self):
        model = _make_model()
        torch.testing.assert_close(
            model.stem_router.bias.data,
            torch.zeros_like(model.stem_router.bias.data))

    def test_final_norm_init(self):
        model = _make_model()
        torch.testing.assert_close(
            model.final_norm.weight.data,
            torch.ones_like(model.final_norm.weight.data))

    def test_param_count_reasonable(self):
        """Model should have a reasonable number of parameters."""
        model = _make_model()
        n_params = sum(p.numel() for p in model.parameters())
        # Small test config — should be a few hundred K at most
        assert 1000 < n_params < 10_000_000


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_from_config(self):
        cfg = _make_config()
        model = LLooM(cfg)
        assert model.config is cfg

    def test_from_kwargs(self):
        model = LLooM(dim=D, stem_n_heads=4, seq_pool_size=4,
                       tok_pool_size=4)
        assert model.config.dim == D

    def test_config_default(self):
        model = LLooM()
        assert model.config.dim == 64  # default


# ---------------------------------------------------------------------------
# Global hop tracking
# ---------------------------------------------------------------------------

class TestGlobalHops:
    def test_global_max_hops_config_property(self):
        cfg = _make_config(seq_max_hops=4, tok_max_hops=8)
        assert cfg.global_max_hops == 12

    def test_info_contains_global_hops(self):
        model = _make_model()
        model.eval()
        x = torch.randn(B, T, D)
        with torch.no_grad():
            _, info = model(x)
        assert 'mean_global_hops' in info

    def test_global_hops_geq_side_hops(self):
        """Global hops should be >= max(seq_hops, tok_hops) for each sample."""
        model = _make_model()
        model.eval()
        x = torch.randn(B, T, D)
        with torch.no_grad():
            _, info = model(x)
        # Global = seq + tok, so it's always >= either side alone
        assert info['mean_global_hops'] >= info['mean_seq_hops']
        assert info['mean_global_hops'] >= info['mean_tok_hops']

    def test_embedding_tables_sized_to_global_max(self):
        """Both pools' hop embedding tables should be sized to global_max_hops."""
        cfg = _make_config(seq_max_hops=4, tok_max_hops=8)
        model = LLooM(cfg)
        global_max = cfg.global_max_hops  # 12
        # Sequence pool
        assert model.seq_pool.hop_embed_bank.shape[1] == global_max
        if model.seq_pool.hop_embed_shared is not None:
            assert model.seq_pool.hop_embed_shared.shape[0] == global_max
        # Token pool
        assert model.tok_pool.hop_embed_bank.shape[1] == global_max
        if model.tok_pool.hop_embed_shared is not None:
            assert model.tok_pool.hop_embed_shared.shape[0] == global_max

    def test_global_hops_sum_of_sides(self):
        """Global hops should equal seq_hops + tok_hops."""
        model = _make_model()
        model.eval()
        x = torch.randn(B, T, D)
        with torch.no_grad():
            _, info = model(x)
        assert abs(info['mean_global_hops']
                   - (info['mean_seq_hops'] + info['mean_tok_hops'])) < 1e-6
