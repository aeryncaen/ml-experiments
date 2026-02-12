"""Tests for LLooM RoutingPool base class.

Uses a minimal concrete subclass (StubPool) to test the base class methods:
- perturb_logits: noise injection stochastic/deterministic
- apply_biases: exit ramp increases with hops, bridge bias fixed
- select_topk: weight zeroing for exit/bridge, renormalization
- classify_topk: correct expert/exit/bridge masks
- apply_hop_conditioning: content-gated hop embedding application
- get_router_logits: banked router dispatch with sharing
- Gradient flow through all base class parameters
"""

import pytest
import torch
import torch.nn as nn

from lloom.routing_pool import RoutingPool


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing base class
# ---------------------------------------------------------------------------

class StubPool(RoutingPool):
    """Minimal subclass that passes through for dispatch."""

    def dispatch(self, x, expert_idx, expert_weights, **kwargs):
        return x  # identity

    def aggregate_decisions(self, topk_idx, is_expert, is_exit, is_bridge, **kwargs):
        B = topk_idx.shape[0]
        return (torch.ones(B, dtype=torch.bool),
                torch.zeros(B, dtype=torch.bool),
                torch.zeros(B, dtype=torch.bool))

    def prepare_bridge_out(self, x):
        return x

    def accept_bridge_in(self, x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POOL = 6
D = 32
TOP_K = 2
MAX_HOPS = 8
B = 4
T = 8


def _make_pool(**kwargs):
    defaults = dict(
        pool_size=POOL, dim=D, top_k=TOP_K, max_hops=MAX_HOPS,
        exit_bias_init=0.0, bridge_bias_init=0.0, exit_ramp_scale=3.0,
        router_noise=1.0, shared_fraction=0.5, hop_gate_dim=12,
    )
    defaults.update(kwargs)
    return StubPool(**defaults)


# ---------------------------------------------------------------------------
# perturb_logits
# ---------------------------------------------------------------------------

class TestPerturbLogits:
    def test_noise_makes_stochastic(self):
        pool = _make_pool(router_noise=1.0)
        pool.train()
        logits = torch.zeros(B, pool.n_options)
        out1 = pool.perturb_logits(logits)
        out2 = pool.perturb_logits(logits)
        # With noise, two calls should give different results
        assert not torch.allclose(out1, out2)

    def test_no_noise_deterministic(self):
        pool = _make_pool(router_noise=0.0)
        pool.train()
        logits = torch.randn(B, pool.n_options)
        out1 = pool.perturb_logits(logits)
        out2 = pool.perturb_logits(logits)
        torch.testing.assert_close(out1, out2)

    def test_eval_mode_no_noise(self):
        pool = _make_pool(router_noise=1.0)
        pool.eval()
        logits = torch.randn(B, pool.n_options)
        out = pool.perturb_logits(logits)
        torch.testing.assert_close(out, logits)

    def test_noise_scale_override(self):
        pool = _make_pool(router_noise=1.0)
        pool.train()
        logits = torch.zeros(B, pool.n_options)
        out = pool.perturb_logits(logits, noise_scale=0.0)
        torch.testing.assert_close(out, logits)

    def test_output_shape(self):
        pool = _make_pool()
        pool.train()
        logits = torch.randn(B, T, pool.n_options)
        out = pool.perturb_logits(logits)
        assert out.shape == logits.shape


# ---------------------------------------------------------------------------
# apply_biases
# ---------------------------------------------------------------------------

class TestApplyBiases:
    def test_exit_ramp_increases_with_hops(self):
        pool = _make_pool(exit_bias_init=0.0, exit_ramp_scale=3.0)
        logits = torch.zeros(B, pool.n_options)

        biased_0 = pool.apply_biases(logits, hops_used=0)
        biased_mid = pool.apply_biases(logits, hops_used=MAX_HOPS // 2)
        biased_max = pool.apply_biases(logits, hops_used=MAX_HOPS)

        exit_0 = biased_0[0, pool.exit_idx].item()
        exit_mid = biased_mid[0, pool.exit_idx].item()
        exit_max = biased_max[0, pool.exit_idx].item()

        assert exit_0 < exit_mid < exit_max

    def test_exit_bias_at_zero_hops(self):
        pool = _make_pool(exit_bias_init=1.5, exit_ramp_scale=3.0)
        logits = torch.zeros(B, pool.n_options)
        biased = pool.apply_biases(logits, hops_used=0)
        # exit_bias = 1.5 + 3.0 * 0/8 = 1.5
        assert biased[0, pool.exit_idx].item() == pytest.approx(1.5)

    def test_exit_bias_at_max_hops(self):
        pool = _make_pool(exit_bias_init=1.0, exit_ramp_scale=3.0)
        logits = torch.zeros(B, pool.n_options)
        biased = pool.apply_biases(logits, hops_used=MAX_HOPS)
        # exit_bias = 1.0 + 3.0 * 8/8 = 4.0
        assert biased[0, pool.exit_idx].item() == pytest.approx(4.0)

    def test_bridge_bias_fixed(self):
        pool = _make_pool(bridge_bias_init=2.0)
        logits = torch.zeros(B, pool.n_options)

        biased_0 = pool.apply_biases(logits, hops_used=0)
        biased_max = pool.apply_biases(logits, hops_used=MAX_HOPS)

        # Bridge bias should be the same regardless of hops
        assert biased_0[0, pool.bridge_idx].item() == pytest.approx(2.0)
        assert biased_max[0, pool.bridge_idx].item() == pytest.approx(2.0)

    def test_expert_logits_unchanged(self):
        pool = _make_pool(exit_bias_init=5.0, bridge_bias_init=3.0)
        logits = torch.randn(B, pool.n_options)
        biased = pool.apply_biases(logits, hops_used=4)
        # Expert slots (0..P-1) should be unchanged
        torch.testing.assert_close(
            biased[..., :POOL], logits[..., :POOL])

    def test_does_not_mutate_input(self):
        pool = _make_pool()
        logits = torch.randn(B, pool.n_options)
        logits_copy = logits.clone()
        pool.apply_biases(logits, hops_used=4)
        torch.testing.assert_close(logits, logits_copy)


# ---------------------------------------------------------------------------
# select_topk
# ---------------------------------------------------------------------------

class TestSelectTopk:
    def test_output_shapes(self):
        pool = _make_pool()
        logits = torch.randn(B, pool.n_options)
        idx, weights, raw_weights = pool.select_topk(logits)
        assert idx.shape == (B, TOP_K)
        assert weights.shape == (B, TOP_K)
        assert raw_weights.shape == (B, TOP_K)

    def test_weights_sum_to_one_for_experts(self):
        """When all top-k are experts, renormalized weights should sum to 1."""
        pool = _make_pool()
        # Make expert logits much higher than exit/bridge
        logits = torch.randn(B, pool.n_options)
        logits[:, :POOL] += 100.0  # ensure experts dominate
        logits[:, POOL:] -= 100.0
        _, weights, _ = pool.select_topk(logits)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(B), atol=1e-5, rtol=1e-5)

    def test_exit_slots_zeroed(self):
        """When exit is in top-k, its weight should be zeroed."""
        pool = _make_pool()
        # Force exit into top-k
        logits = torch.zeros(B, pool.n_options)
        logits[:, pool.exit_idx] = 100.0  # exit dominates
        logits[:, 0] = 50.0  # one expert also high
        idx, weights, raw_weights = pool.select_topk(logits)

        for b in range(B):
            for k in range(TOP_K):
                if idx[b, k] == pool.exit_idx:
                    assert weights[b, k].item() == 0.0
                    assert raw_weights[b, k].item() > 0.0  # raw should be nonzero

    def test_bridge_slots_zeroed(self):
        """When bridge is in top-k, its weight should be zeroed."""
        pool = _make_pool()
        logits = torch.zeros(B, pool.n_options)
        logits[:, pool.bridge_idx] = 100.0
        logits[:, 0] = 50.0
        idx, weights, _ = pool.select_topk(logits)

        for b in range(B):
            for k in range(TOP_K):
                if idx[b, k] == pool.bridge_idx:
                    assert weights[b, k].item() == 0.0

    def test_all_exit_gives_zero_weights(self):
        """When ALL top-k are exit/bridge, all weights should be zero."""
        pool = _make_pool()
        logits = torch.zeros(B, pool.n_options)
        logits[:, pool.exit_idx] = 100.0
        logits[:, pool.bridge_idx] = 99.0
        logits[:, :POOL] = -100.0
        idx, weights, _ = pool.select_topk(logits)
        assert (weights == 0.0).all()

    def test_renormalization(self):
        """Expert weights should be renormalized after zeroing."""
        pool = _make_pool(top_k=3)
        logits = torch.zeros(1, pool.n_options)
        # Expert 0 and 1 are top, plus exit
        logits[0, 0] = 10.0
        logits[0, 1] = 9.0
        logits[0, pool.exit_idx] = 11.0  # highest
        idx, weights, _ = pool.select_topk(logits)
        # Weights for expert slots should sum to 1
        expert_mask = idx[0] < POOL
        assert weights[0, expert_mask].sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# classify_topk
# ---------------------------------------------------------------------------

class TestClassifyTopk:
    def test_expert_mask(self):
        pool = _make_pool()
        idx = torch.tensor([[0, 3], [1, pool.exit_idx]])
        is_expert, is_exit, is_bridge = pool.classify_topk(idx)
        assert is_expert[0].all()
        assert is_expert[1, 0] and not is_expert[1, 1]

    def test_exit_mask(self):
        pool = _make_pool()
        idx = torch.tensor([[pool.exit_idx, 0], [pool.exit_idx, pool.exit_idx]])
        _, is_exit, _ = pool.classify_topk(idx)
        assert is_exit[0, 0] and not is_exit[0, 1]
        assert is_exit[1].all()

    def test_bridge_mask(self):
        pool = _make_pool()
        idx = torch.tensor([[pool.bridge_idx, 0], [pool.bridge_idx, pool.bridge_idx]])
        _, _, is_bridge = pool.classify_topk(idx)
        assert is_bridge[0, 0] and not is_bridge[0, 1]
        assert is_bridge[1].all()

    def test_mutually_exclusive(self):
        pool = _make_pool()
        idx = torch.randint(0, pool.n_options, (B, TOP_K))
        is_expert, is_exit, is_bridge = pool.classify_topk(idx)
        # Each slot is exactly one of expert/exit/bridge
        total = is_expert.long() + is_exit.long() + is_bridge.long()
        assert (total == 1).all()


# ---------------------------------------------------------------------------
# apply_hop_conditioning
# ---------------------------------------------------------------------------

class TestApplyHopConditioning:
    def test_output_shape(self):
        pool = _make_pool()
        x = torch.randn(B, D)
        eidx = torch.randint(0, POOL, (B,))
        out = pool.apply_hop_conditioning(x, eidx, hop=0)
        assert out.shape == (B, D)

    def test_output_shape_3d(self):
        pool = _make_pool()
        x = torch.randn(B, T, D)
        eidx = torch.randint(0, POOL, (B,))
        # For 3d input with 1d expert_idx, we need matching shapes
        # In practice, sequence side passes (B, T, D) with (B,) expert_idx
        # The hop embed is (B, D), broadcast over T
        # Let's test with (B*T, D) and (B*T,) to match the flat case
        x_flat = x.view(B * T, D)
        eidx_flat = eidx.repeat_interleave(T)
        out = pool.apply_hop_conditioning(x_flat, eidx_flat, hop=0)
        assert out.shape == (B * T, D)

    def test_different_hops_different_output(self):
        pool = _make_pool()
        x = torch.randn(B, D)
        eidx = torch.randint(0, POOL, (B,))
        out0 = pool.apply_hop_conditioning(x, eidx, hop=0)
        out1 = pool.apply_hop_conditioning(x, eidx, hop=1)
        # Different hop embeddings should produce different results
        # (gate may be near zero at init, so let's set it nonzero)
        pool.hop_gate_proj.bias.data.fill_(2.0)  # open the gate
        out0 = pool.apply_hop_conditioning(x, eidx, hop=0)
        out1 = pool.apply_hop_conditioning(x, eidx, hop=1)
        assert not torch.allclose(out0, out1, atol=1e-5)

    def test_different_experts_different_output(self):
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        x = torch.randn(1, D)
        out0 = pool.apply_hop_conditioning(x, torch.tensor([0]), hop=0)
        out1 = pool.apply_hop_conditioning(x, torch.tensor([1]), hop=0)
        assert not torch.allclose(out0, out1, atol=1e-5)

    def test_gate_starts_near_half(self):
        """With zero-initialized gate proj, sigmoid(0) = 0.5."""
        pool = _make_pool()
        x = torch.randn(B, D)
        eidx = torch.randint(0, POOL, (B,))
        # Gate projection is zero-initialized, so sigmoid(0) = 0.5
        # Hop embed should be added at ~50% strength
        out = pool.apply_hop_conditioning(x, eidx, hop=0)
        # Output should differ from just hop_norm(x) but not by a huge amount
        normed = pool.hop_norm(x)
        diff = (out - normed).abs().mean()
        assert diff > 0  # should be nonzero (hop embed != 0)

    def test_hop_clamping(self):
        """Hop beyond max_hops should clamp to max_hops - 1, not crash."""
        pool = _make_pool()
        x = torch.randn(B, D)
        eidx = torch.randint(0, POOL, (B,))
        out = pool.apply_hop_conditioning(x, eidx, hop=MAX_HOPS + 10)
        assert out.shape == (B, D)

    def test_gradient_flow(self):
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        x = torch.randn(B, D, requires_grad=True)
        eidx = torch.randint(0, POOL, (B,))
        out = pool.apply_hop_conditioning(x, eidx, hop=0)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert pool.hop_embed_bank.grad is not None
        assert pool.hop_gate_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# get_router_logits
# ---------------------------------------------------------------------------

class TestGetRouterLogits:
    def test_output_shape(self):
        pool = _make_pool()
        x = torch.randn(B, D)
        eidx = torch.randint(0, POOL, (B,))
        logits = pool.get_router_logits(x, eidx)
        assert logits.shape == (B, pool.n_options)

    def test_different_experts_different_logits(self):
        pool = _make_pool()
        x = torch.randn(1, D)
        logits0 = pool.get_router_logits(x, torch.tensor([0]))
        logits1 = pool.get_router_logits(x, torch.tensor([1]))
        assert not torch.allclose(logits0, logits1, atol=1e-5)

    def test_gradient_flow(self):
        pool = _make_pool()
        x = torch.randn(B, D, requires_grad=True)
        eidx = torch.randint(0, POOL, (B,))
        logits = pool.get_router_logits(x, eidx)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert pool.router_bank.grad is not None
        if pool.router_shared is not None:
            assert pool.router_shared.grad is not None

    def test_sharing_params_exist(self):
        pool = _make_pool(shared_fraction=0.5)
        assert pool.router_shared is not None
        param_names = {n for n, _ in pool.named_parameters()}
        assert 'router_shared' in param_names

    def test_no_sharing(self):
        pool = _make_pool(shared_fraction=0.0)
        assert pool.router_shared is None


# ---------------------------------------------------------------------------
# Integration: gradient flow through full base class
# ---------------------------------------------------------------------------

class TestGradientFlowIntegration:
    def test_all_base_params_get_grad(self):
        """Test gradient flow through router + hop conditioning together.

        Uses logits.sum() as loss (not softmax) to avoid vanishing gradients
        from saturated softmax masking the actual gradient connectivity.
        """
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        pool.train()

        x = torch.randn(B, D, requires_grad=True)
        eidx = torch.arange(POOL)[:B] % POOL

        # Hop conditioning → router logits
        x_cond = pool.apply_hop_conditioning(x, eidx, hop=0)
        logits = pool.get_router_logits(x_cond, eidx)

        # Use logits.sum() directly — tests gradient connectivity without
        # softmax saturation. The select_topk grad flow is tested separately.
        loss = logits.sum()
        loss.backward()

        # Core params always in the compute path
        must_have_grad = ('router_shared', 'hop_gate_proj.weight', 'hop_gate_proj.bias')
        for name, p in pool.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                if name in must_have_grad:
                    assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"

        # router_bank: only selected expert slices get gradient
        for idx in eidx.unique():
            assert pool.router_bank.grad[idx].abs().sum() > 0, \
                f"Zero gradient for router_bank[{idx}]"

    def test_n_options_correct(self):
        pool = _make_pool()
        assert pool.n_options == POOL + 2
        assert pool.exit_idx == POOL
        assert pool.bridge_idx == POOL + 1


# ---------------------------------------------------------------------------
# Global hop embedding table sizing
# ---------------------------------------------------------------------------

class TestGlobalMaxHops:
    def test_embedding_table_uses_global_max_hops(self):
        """Hop embedding tables should be sized to global_max_hops, not max_hops."""
        local_max = 8
        global_max = 48
        pool = _make_pool(max_hops=local_max, global_max_hops=global_max)
        # Private bank: (pool_size, global_max_hops, dim_private)
        assert pool.hop_embed_bank.shape[1] == global_max
        # Shared: (global_max_hops, dim_shared)
        if pool.hop_embed_shared is not None:
            assert pool.hop_embed_shared.shape[0] == global_max

    def test_defaults_to_max_hops_when_global_not_set(self):
        """Without global_max_hops, embedding tables default to max_hops."""
        pool = _make_pool(max_hops=MAX_HOPS)
        assert pool.hop_embed_bank.shape[1] == MAX_HOPS
        assert pool.global_max_hops == MAX_HOPS

    def test_conditioning_uses_distinct_embeddings_up_to_global_max(self):
        """Each global hop index should produce a distinct embedding."""
        global_max = 20
        pool = _make_pool(max_hops=8, global_max_hops=global_max)
        pool.eval()
        x = torch.randn(1, D)
        eidx = torch.tensor([0])
        outputs = []
        with torch.no_grad():
            for h in range(global_max):
                out = pool.apply_hop_conditioning(x, eidx, hop=h)
                outputs.append(out.clone())
        # All outputs should be distinct (different hop embeddings)
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j], atol=1e-6), \
                    f"Hop {i} and {j} produced identical output"

    def test_exit_ramp_uses_local_max_hops(self):
        """apply_biases should use local max_hops for exit ramp, not global."""
        local_max = 8
        global_max = 48
        pool = _make_pool(max_hops=local_max, global_max_hops=global_max,
                          exit_ramp_scale=3.0)
        logits = torch.zeros(1, pool.n_options)
        biased = pool.apply_biases(logits, hops_used=local_max)
        # At hops_used == local_max, ramp = 3.0 * (8/8) = 3.0
        expected_exit_bias = 0.0 + 3.0 * (local_max / local_max)
        assert abs(biased[0, pool.exit_idx].item() - expected_exit_bias) < 1e-5
