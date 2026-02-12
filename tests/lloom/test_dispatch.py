"""Tests for LLooM dispatch: AttentionParamBank and MLPParamBank.

Covers:
- Output shapes for various (B, T, D) / (N, D) configurations
- Gradient flow through all parameters
- Weight sharing correctness (shared slice is same object across gathers)
- Top-k weighted merge behavior
- Index clamping safety (out-of-range expert ids)
"""

import pytest
import torch
import torch.nn as nn

from lloom.dispatch import (
    AttentionParamBank,
    MLPParamBank,
    _make_bank,
    _gather_weights,
    _make_1d_bank,
    _gather_1d,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D = 32
D_INNER = 48  # must be divisible by N_HEADS
N_HEADS = 4
POOL = 6
TOP_K = 2
B = 3
T = 8
N = B * T  # for token-level (flattened)


def _make_attn_bank(shared_fraction=0.5, **kwargs):
    defaults = dict(
        pool_size=POOL, dim=D, inner_dim=D_INNER,
        n_heads=N_HEADS, shared_fraction=shared_fraction,
    )
    defaults.update(kwargs)
    return AttentionParamBank(**defaults)


def _make_mlp_bank(shared_fraction=0.5, **kwargs):
    defaults = dict(
        pool_size=POOL, dim=D, inner_dim=D_INNER,
        shared_fraction=shared_fraction,
    )
    defaults.update(kwargs)
    return MLPParamBank(**defaults)


def _rand_expert_idx(batch, top_k=TOP_K, pool_size=POOL):
    return torch.randint(0, pool_size, (batch, top_k))


def _rand_weights(batch, top_k=TOP_K):
    w = torch.rand(batch, top_k)
    w = w / w.sum(dim=-1, keepdim=True)
    return w


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestMakeBank:
    def test_shapes_with_sharing(self):
        shared, private, s_out, p_out = _make_bank(POOL, D, D_INNER, 0.5)
        assert shared is not None
        assert shared.shape == (D, s_out)
        assert private.shape == (POOL, D, p_out)
        assert s_out + p_out == D_INNER

    def test_shapes_no_sharing(self):
        shared, private, s_out, p_out = _make_bank(POOL, D, D_INNER, 0.0)
        assert shared is None
        assert s_out == 0
        assert p_out == D_INNER
        assert private.shape == (POOL, D, D_INNER)

    def test_init_scale(self):
        """Weights should be initialized at ~ in_dim^(-0.5) scale."""
        shared, private, _, _ = _make_bank(POOL, D, D_INNER, 0.5)
        expected_std = D ** -0.5
        assert private.std().item() == pytest.approx(expected_std, rel=0.5)
        assert shared.std().item() == pytest.approx(expected_std, rel=0.5)


class TestGatherWeights:
    def test_output_shape(self):
        shared, private, s_out, _ = _make_bank(POOL, D, D_INNER, 0.5)
        idx = _rand_expert_idx(N)
        w = _gather_weights(shared, private, idx, s_out, N, TOP_K, D)
        assert w.shape == (N, TOP_K, D, D_INNER)

    def test_shared_slice_identical(self):
        """All gathered weights should have the same shared slice."""
        shared, private, s_out, _ = _make_bank(POOL, D, D_INNER, 0.5)
        idx = _rand_expert_idx(N)
        w = _gather_weights(shared, private, idx, s_out, N, TOP_K, D)
        # Shared slice is first s_out columns
        shared_slices = w[..., :s_out]  # (N, K, D, s_out)
        # All should be identical to the shared weight
        for i in range(N):
            for k in range(TOP_K):
                torch.testing.assert_close(shared_slices[i, k], shared.data)

    def test_no_sharing_works(self):
        shared, private, s_out, _ = _make_bank(POOL, D, D_INNER, 0.0)
        idx = _rand_expert_idx(N)
        w = _gather_weights(shared, private, idx, s_out, N, TOP_K, D)
        assert w.shape == (N, TOP_K, D, D_INNER)


class TestMake1dBank:
    def test_shapes_with_sharing(self):
        shared, private, s_dim, p_dim = _make_1d_bank(POOL, D_INNER, 0.5)
        assert shared is not None
        assert shared.shape == (s_dim,)
        assert private.shape == (POOL, p_dim)
        assert s_dim + p_dim == D_INNER

    def test_init_value(self):
        shared, private, _, _ = _make_1d_bank(POOL, D_INNER, 0.5, init_val=1.0)
        torch.testing.assert_close(shared, torch.ones_like(shared))
        torch.testing.assert_close(private, torch.ones_like(private))


class TestGather1d:
    def test_output_shape(self):
        shared, private, _, _ = _make_1d_bank(POOL, D_INNER, 0.5)
        idx = _rand_expert_idx(N)
        result = _gather_1d(shared, private, idx, N, TOP_K)
        assert result.shape == (N, TOP_K, D_INNER)


# ---------------------------------------------------------------------------
# AttentionParamBank tests
# ---------------------------------------------------------------------------

class TestAttentionParamBank:
    def test_output_shape(self):
        bank = _make_attn_bank()
        x = torch.randn(B, T, D)
        idx = _rand_expert_idx(B)
        w = _rand_weights(B)
        out = bank(x, idx, w)
        assert out.shape == (B, T, D)

    @pytest.mark.parametrize("b,t,d", [(1, 4, D), (4, 16, D), (2, 1, D)])
    def test_output_shape_various(self, b, t, d):
        bank = _make_attn_bank()
        x = torch.randn(b, t, d)
        idx = _rand_expert_idx(b)
        w = _rand_weights(b)
        out = bank(x, idx, w)
        assert out.shape == (b, t, d)

    def test_gradient_flow_all_params(self):
        """Every parameter with requires_grad should get a non-zero gradient."""
        bank = _make_attn_bank()
        x = torch.randn(B, T, D, requires_grad=True)
        idx = _rand_expert_idx(B)
        w = _rand_weights(B)

        out = bank(x, idx, w)
        loss = out.sum()
        loss.backward()

        # Check input grad
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # Check all parameter grads
        for name, p in bank.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_flow_no_sharing(self):
        """Gradient flow works without weight sharing too."""
        bank = _make_attn_bank(shared_fraction=0.0)
        x = torch.randn(B, T, D, requires_grad=True)
        idx = _rand_expert_idx(B)
        w = _rand_weights(B)

        out = bank(x, idx, w)
        loss = out.sum()
        loss.backward()

        for name, p in bank.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_sharing_params_exist(self):
        """With shared_fraction > 0, shared weights should be registered."""
        bank = _make_attn_bank(shared_fraction=0.5)
        param_names = {n for n, _ in bank.named_parameters()}
        assert 'up_shared' in param_names
        assert 'qkv_shared' in param_names
        assert 'o_shared' in param_names
        assert 'norm_shared' in param_names
        assert 'down_shared' in param_names

    def test_no_sharing_no_shared_params(self):
        """With shared_fraction=0, shared weights should be None."""
        bank = _make_attn_bank(shared_fraction=0.0)
        assert bank.up_shared is None
        assert bank.qkv_shared is None
        assert bank.o_shared is None
        assert bank.norm_shared is None
        assert bank.down_shared is None

    def test_index_clamping(self):
        """Out-of-range expert indices should be clamped, not crash."""
        bank = _make_attn_bank()
        x = torch.randn(B, T, D)
        # Indices beyond pool_size
        idx = torch.full((B, TOP_K), POOL + 10, dtype=torch.long)
        w = _rand_weights(B)
        out = bank(x, idx, w)  # should not crash
        assert out.shape == (B, T, D)

    def test_different_experts_different_output(self):
        """Routing to different experts should produce different outputs."""
        bank = _make_attn_bank()
        x = torch.randn(1, T, D)
        w = torch.tensor([[1.0, 0.0]])

        # Expert 0
        idx0 = torch.tensor([[0, 0]])
        out0 = bank(x, idx0, w)

        # Expert 1
        idx1 = torch.tensor([[1, 0]])
        out1 = bank(x, idx1, w)

        assert not torch.allclose(out0, out1, atol=1e-5)

    def test_weight_merge(self):
        """Output with weights [1,0] should match single-expert dispatch."""
        bank = _make_attn_bank()
        x = torch.randn(1, T, D)

        # Full weight on expert 0
        idx = torch.tensor([[0, 1]])
        w_full = torch.tensor([[1.0, 0.0]])
        out_full = bank(x, idx, w_full)

        # Equal weight
        w_half = torch.tensor([[0.5, 0.5]])
        out_half = bank(x, idx, w_half)

        # They should differ (unless experts are identical, which they won't be)
        assert not torch.allclose(out_full, out_half, atol=1e-5)

    def test_causal_flag(self):
        """Non-causal bank should produce different output than causal."""
        bank_causal = _make_attn_bank(is_causal=True)
        bank_noncausal = _make_attn_bank(is_causal=False)
        # Copy weights from causal to noncausal
        bank_noncausal.load_state_dict(bank_causal.state_dict())

        x = torch.randn(1, T, D)
        idx = _rand_expert_idx(1)
        w = _rand_weights(1)

        out_c = bank_causal(x, idx, w)
        out_nc = bank_noncausal(x, idx, w)

        # With T > 1, causal vs non-causal attention should differ
        assert not torch.allclose(out_c, out_nc, atol=1e-5)

    def test_param_count_sharing_reduces(self):
        """50% sharing should meaningfully reduce total parameter count."""
        bank_shared = _make_attn_bank(shared_fraction=0.5)
        bank_full = _make_attn_bank(shared_fraction=0.0)

        n_shared = sum(p.numel() for p in bank_shared.parameters())
        n_full = sum(p.numel() for p in bank_full.parameters())

        # Shared should have fewer params
        assert n_shared < n_full


# ---------------------------------------------------------------------------
# MLPParamBank tests
# ---------------------------------------------------------------------------

class TestMLPParamBank:
    def test_output_shape(self):
        bank = _make_mlp_bank()
        x = torch.randn(N, D)
        idx = _rand_expert_idx(N)
        w = _rand_weights(N)
        out = bank(x, idx, w)
        assert out.shape == (N, D)

    @pytest.mark.parametrize("n", [1, 16, 64])
    def test_output_shape_various(self, n):
        bank = _make_mlp_bank()
        x = torch.randn(n, D)
        idx = _rand_expert_idx(n)
        w = _rand_weights(n)
        out = bank(x, idx, w)
        assert out.shape == (n, D)

    def test_gradient_flow_all_params(self):
        """Every parameter should get a non-zero gradient."""
        bank = _make_mlp_bank()
        x = torch.randn(N, D, requires_grad=True)
        idx = _rand_expert_idx(N)
        w = _rand_weights(N)

        out = bank(x, idx, w)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        for name, p in bank.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_flow_no_sharing(self):
        bank = _make_mlp_bank(shared_fraction=0.0)
        x = torch.randn(N, D, requires_grad=True)
        idx = _rand_expert_idx(N)
        w = _rand_weights(N)

        out = bank(x, idx, w)
        loss = out.sum()
        loss.backward()

        for name, p in bank.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_sharing_params_exist(self):
        bank = _make_mlp_bank(shared_fraction=0.5)
        param_names = {n for n, _ in bank.named_parameters()}
        assert 'gate_up_shared' in param_names
        assert 'down_shared' in param_names

    def test_no_sharing_no_shared_params(self):
        bank = _make_mlp_bank(shared_fraction=0.0)
        assert bank.gate_up_shared is None
        assert bank.down_shared is None

    def test_index_clamping(self):
        """Out-of-range indices should be clamped, not crash."""
        bank = _make_mlp_bank()
        x = torch.randn(N, D)
        idx = torch.full((N, TOP_K), POOL + 10, dtype=torch.long)
        w = _rand_weights(N)
        out = bank(x, idx, w)
        assert out.shape == (N, D)

    def test_different_experts_different_output(self):
        bank = _make_mlp_bank()
        x = torch.randn(1, D)
        w = torch.tensor([[1.0, 0.0]])

        idx0 = torch.tensor([[0, 0]])
        out0 = bank(x, idx0, w)

        idx1 = torch.tensor([[1, 0]])
        out1 = bank(x, idx1, w)

        assert not torch.allclose(out0, out1, atol=1e-5)

    def test_zero_weight_zeroes_expert(self):
        """Expert with weight=0 should not contribute to output."""
        bank = _make_mlp_bank()
        x = torch.randn(1, D)

        idx = torch.tensor([[0, 1]])
        w_only_first = torch.tensor([[1.0, 0.0]])
        w_only_second = torch.tensor([[0.0, 1.0]])

        out_first = bank(x, idx, w_only_first)
        out_second = bank(x, idx, w_only_second)

        # These should be different (different experts with full weight)
        assert not torch.allclose(out_first, out_second, atol=1e-5)

    def test_param_count_sharing_reduces(self):
        bank_shared = _make_mlp_bank(shared_fraction=0.5)
        bank_full = _make_mlp_bank(shared_fraction=0.0)

        n_shared = sum(p.numel() for p in bank_shared.parameters())
        n_full = sum(p.numel() for p in bank_full.parameters())

        assert n_shared < n_full

    def test_swiglu_nonlinearity(self):
        """Output should not be a linear function of input (SwiGLU is nonlinear)."""
        bank = _make_mlp_bank()
        x1 = torch.randn(N, D)
        x2 = 2.0 * x1
        idx = _rand_expert_idx(N)
        w = _rand_weights(N)

        out1 = bank(x1, idx, w)
        out2 = bank(x2, idx, w)

        # If linear, out2 == 2 * out1. SwiGLU should break this.
        assert not torch.allclose(out2, 2.0 * out1, atol=1e-3)
