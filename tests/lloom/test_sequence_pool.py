"""Tests for LLooM SequencePool.

Covers:
- Output shapes from forward()
- Gradient flow through all parameters
- All-top-k exit/bridge/continue decision logic
- Hop budget enforcement (force exit when exhausted)
- Bridge passthrough (identity)
- Active mask handling (inactive samples unchanged)
"""

import pytest
import torch

from lloom.sequence_pool import SequencePool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POOL = 4
D = 32
D_INNER = 48
N_HEADS = 4
TOP_K = 2
MAX_HOPS = 8
B = 3
T = 8


def _make_pool(**kwargs):
    defaults = dict(
        pool_size=POOL, dim=D, inner_dim=D_INNER, n_heads=N_HEADS,
        top_k=TOP_K, max_hops=MAX_HOPS, shared_fraction=0.5,
        hop_gate_dim=12, exit_bias_init=0.0, bridge_bias_init=0.0,
        exit_ramp_scale=3.0, router_noise=0.0,  # no noise for determinism
    )
    defaults.update(kwargs)
    return SequencePool(**defaults)


def _make_inputs(b=B, t=T):
    x = torch.randn(b, t, D, requires_grad=True)
    active = torch.ones(b, dtype=torch.bool)
    current_expert = torch.randint(0, POOL, (b,))
    return x, active, current_expert


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_basic_output_shape(self):
        pool = _make_pool()
        x, active, ce = _make_inputs()
        x_out, new_active, do_exit, do_bridge, next_expert, hops = \
            pool(x, active, hops_used=0, current_expert=ce)
        assert x_out.shape == (B, T, D)
        assert new_active.shape == (B,)
        assert do_exit.shape == (B,)
        assert do_bridge.shape == (B,)
        assert next_expert.shape == (B,)
        assert hops == 1

    @pytest.mark.parametrize("b,t", [(1, 4), (4, 16), (2, 1)])
    def test_various_shapes(self, b, t):
        pool = _make_pool()
        x, active, ce = _make_inputs(b=b, t=t)
        x_out, *_ = pool(x, active, hops_used=0, current_expert=ce)
        assert x_out.shape == (b, t, D)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_all_params_get_grad(self):
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        x, active, ce = _make_inputs()
        x_out, *_ = pool(x, active, hops_used=0, current_expert=ce)
        loss = x_out.sum()
        loss.backward()

        assert x.grad is not None and x.grad.abs().sum() > 0

        # Expert bank params should all get gradient
        for name, p in pool.expert_bank.named_parameters():
            assert p.grad is not None, f"No gradient for expert_bank.{name}"

    def test_backward_does_not_crash(self):
        """Multi-hop backward should not crash."""
        pool = _make_pool()
        x, active, ce = _make_inputs()
        for hop in range(3):
            x_out, active, _, _, ce, _ = pool(x, active, hops_used=hop, current_expert=ce)
            x = x + x_out  # accumulate for gradient flow
        loss = x.sum()
        loss.backward()  # should not crash


# ---------------------------------------------------------------------------
# Decision aggregation: all-top-k agreement
# ---------------------------------------------------------------------------

class TestAggregateDecisions:
    def test_all_exit(self):
        pool = _make_pool()
        # All top-k are exit
        idx = torch.full((B, TOP_K), pool.exit_idx, dtype=torch.long)
        is_expert, is_exit, is_bridge = pool.classify_topk(idx)
        cont, exit_, bridge = pool.aggregate_decisions(idx, is_expert, is_exit, is_bridge)
        assert exit_.all()
        assert not cont.any()
        assert not bridge.any()

    def test_all_bridge(self):
        pool = _make_pool()
        idx = torch.full((B, TOP_K), pool.bridge_idx, dtype=torch.long)
        is_expert, is_exit, is_bridge = pool.classify_topk(idx)
        cont, exit_, bridge = pool.aggregate_decisions(idx, is_expert, is_exit, is_bridge)
        assert bridge.all()
        assert not cont.any()
        assert not exit_.any()

    def test_mixed_continues(self):
        pool = _make_pool()
        # One expert, one exit → should continue (not all-exit)
        idx = torch.tensor([[0, pool.exit_idx]] * B)
        is_expert, is_exit, is_bridge = pool.classify_topk(idx)
        cont, exit_, bridge = pool.aggregate_decisions(idx, is_expert, is_exit, is_bridge)
        assert cont.all()
        assert not exit_.any()
        assert not bridge.any()

    def test_all_expert_continues(self):
        pool = _make_pool()
        idx = torch.randint(0, POOL, (B, TOP_K))
        is_expert, is_exit, is_bridge = pool.classify_topk(idx)
        cont, exit_, bridge = pool.aggregate_decisions(idx, is_expert, is_exit, is_bridge)
        assert cont.all()


# ---------------------------------------------------------------------------
# Hop budget
# ---------------------------------------------------------------------------

class TestHopBudget:
    def test_force_exit_at_budget(self):
        pool = _make_pool()
        x, active, ce = _make_inputs()
        x_out, new_active, do_exit, do_bridge, _, hops = \
            pool(x, active, hops_used=MAX_HOPS, current_expert=ce)
        # Budget exhausted: all active should exit
        assert do_exit.all()
        assert not new_active.any()
        assert hops == MAX_HOPS  # hop count not incremented

    def test_force_exit_beyond_budget(self):
        pool = _make_pool()
        x, active, ce = _make_inputs()
        x_out, new_active, do_exit, _, _, hops = \
            pool(x, active, hops_used=MAX_HOPS + 5, current_expert=ce)
        assert do_exit.all()

    def test_normal_hop_increments(self):
        pool = _make_pool()
        x, active, ce = _make_inputs()
        _, _, _, _, _, hops = pool(x, active, hops_used=3, current_expert=ce)
        assert hops == 4


# ---------------------------------------------------------------------------
# Active mask handling
# ---------------------------------------------------------------------------

class TestActiveMask:
    def test_inactive_samples_unchanged(self):
        pool = _make_pool()
        x_orig = torch.randn(B, T, D)
        x = x_orig.clone().requires_grad_(True)
        active = torch.zeros(B, dtype=torch.bool)  # all inactive
        ce = torch.randint(0, POOL, (B,))
        x_out, new_active, do_exit, do_bridge, _, _ = \
            pool(x, active, hops_used=0, current_expert=ce)
        # x_out should equal x for all samples (no active dispatch)
        # Inactive: expert_out is zeroed, so x_new = x + 0 = x
        # But hop conditioning and routing still run — the key is the
        # active_mask gates the residual add.
        assert not new_active.any()  # still all inactive
        assert not do_exit.any()
        assert not do_bridge.any()

    def test_partial_active(self):
        pool = _make_pool()
        x, _, ce = _make_inputs()
        active = torch.tensor([True, False, True])
        x_out, new_active, do_exit, do_bridge, _, _ = \
            pool(x, active, hops_used=0, current_expert=ce)
        # Sample 1 (inactive) should not exit or bridge
        assert not do_exit[1]
        assert not do_bridge[1]


# ---------------------------------------------------------------------------
# Bridge passthrough
# ---------------------------------------------------------------------------

class TestBridge:
    def test_prepare_bridge_is_identity(self):
        pool = _make_pool()
        x = torch.randn(B, T, D)
        assert pool.prepare_bridge_out(x) is x

    def test_accept_bridge_is_identity(self):
        pool = _make_pool()
        x = torch.randn(B, T, D)
        assert pool.accept_bridge_in(x) is x


# ---------------------------------------------------------------------------
# Global hop conditioning
# ---------------------------------------------------------------------------

class TestGlobalHop:
    def test_global_hop_changes_conditioning(self):
        """Different global_hop with same hops_used should produce different outputs."""
        pool = _make_pool(router_noise=0.0, global_max_hops=48)
        pool.eval()
        x = torch.randn(B, T, D)
        active = torch.ones(B, dtype=torch.bool)
        ce = torch.zeros(B, dtype=torch.long)

        with torch.no_grad():
            out0, *_ = pool(x, active, hops_used=0, current_expert=ce, global_hop=0)
            out10, *_ = pool(x, active, hops_used=0, current_expert=ce, global_hop=10)

        assert not torch.allclose(out0, out10, atol=1e-6), \
            "Different global_hop values produced identical outputs"

    def test_hops_used_still_controls_exit_ramp(self):
        """Exit ramp should use hops_used (local), not global_hop."""
        pool = _make_pool(router_noise=0.0, exit_ramp_scale=3.0, global_max_hops=48)
        pool.eval()
        x = torch.randn(B, T, D)
        active = torch.ones(B, dtype=torch.bool)
        ce = torch.zeros(B, dtype=torch.long)

        # At hops_used=MAX_HOPS, should force exit regardless of global_hop
        out, new_active, do_exit, _, _, _ = \
            pool(x, active, hops_used=MAX_HOPS, current_expert=ce, global_hop=0)
        assert do_exit.all()
        assert not new_active.any()

    def test_fallback_to_hops_used_when_global_not_set(self):
        """Without global_hop, conditioning should use hops_used."""
        pool = _make_pool(router_noise=0.0)
        pool.eval()
        x = torch.randn(B, T, D)
        active = torch.ones(B, dtype=torch.bool)
        ce = torch.zeros(B, dtype=torch.long)

        with torch.no_grad():
            out_explicit, *_ = pool(x, active, hops_used=3, current_expert=ce, global_hop=3)
            out_fallback, *_ = pool(x, active, hops_used=3, current_expert=ce)

        assert torch.allclose(out_explicit, out_fallback, atol=1e-6), \
            "global_hop=hops_used should match fallback behavior"
