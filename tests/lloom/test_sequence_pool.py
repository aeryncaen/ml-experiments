"""Tests for LLooM SequencePool.

Covers:
- execute_hop() output shapes
- Post-dispatch routing: outbound logits produced from expert output
- Gradient flow through all parameters (expert bank + routers + hop conditioning)
- Entry router for bridge crossings
- Bridge passthrough (identity)
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
        top_k=TOP_K, max_hops=MAX_HOPS,
        expert_shared_fraction=0.5, router_shared_fraction=0.5,
        hop_gate_dim=12, exit_bias_init=0.0, bridge_bias_init=0.0,
        exit_ramp_scale=3.0, router_noise=0.0,  # no noise for determinism
    )
    defaults.update(kwargs)
    return SequencePool(**defaults)


def _make_inputs(b=B, t=T):
    x = torch.randn(b, t, D, requires_grad=True)
    # Create topk_idx with valid expert indices
    topk_idx = torch.randint(0, POOL, (b, TOP_K))
    topk_weights = torch.full((b, TOP_K), 1.0 / TOP_K)
    return x, topk_idx, topk_weights


# ---------------------------------------------------------------------------
# execute_hop shape tests
# ---------------------------------------------------------------------------

class TestExecuteHopShapes:
    def test_basic_output_shapes(self):
        pool = _make_pool()
        x, topk_idx, topk_weights = _make_inputs()
        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        assert out.shape == (B, T, D)
        assert next_logits.shape == (B, pool.n_options)

    @pytest.mark.parametrize("b,t", [(1, 4), (4, 16), (2, 1)])
    def test_various_shapes(self, b, t):
        pool = _make_pool()
        x, topk_idx, topk_weights = _make_inputs(b=b, t=t)
        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        assert out.shape == (b, t, D)
        assert next_logits.shape == (b, pool.n_options)

    def test_exit_in_topk_still_works(self):
        """Exit/bridge indices in topk should not crash (they get clamped)."""
        pool = _make_pool()
        x = torch.randn(B, T, D)
        topk_idx = torch.tensor([[0, pool.exit_idx]] * B)
        topk_weights = torch.tensor([[1.0, 0.0]] * B)  # exit weight zeroed
        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Post-dispatch routing
# ---------------------------------------------------------------------------

class TestPostDispatchRouting:
    def test_next_logits_nonzero(self):
        """Outbound logits should be non-trivial (not all zeros)."""
        pool = _make_pool()
        x, topk_idx, topk_weights = _make_inputs()
        _, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        assert next_logits.abs().sum() > 0

    def test_next_logits_differ_across_hops(self):
        """Different hops should produce different outbound logits (hop conditioning)."""
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)  # open gate
        pool.eval()
        x = torch.randn(B, T, D)
        topk_idx = torch.zeros(B, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, TOP_K), 1.0 / TOP_K)

        with torch.no_grad():
            _, logits0 = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
            _, logits1 = pool.execute_hop(x, topk_idx, topk_weights, hop=1)

        assert not torch.allclose(logits0, logits1, atol=1e-5), \
            "Same outbound logits across hops -- hop conditioning not working"

    def test_logit_chain_flow(self):
        """execute_hop output logits can feed back into route()."""
        pool = _make_pool()
        x = torch.randn(B, T, D)
        # Initial logits from entry router
        logits = pool.entry_router(x.mean(dim=1))

        # First hop
        topk_idx, topk_weights, has_exit, has_bridge, has_continue = \
            pool.route(logits, hops_used=0)
        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        x = x + out

        # Second hop using outbound logits
        topk_idx2, topk_weights2, _, _, _ = pool.route(next_logits, hops_used=1)
        out2, next_logits2 = pool.execute_hop(x, topk_idx2, topk_weights2, hop=1)
        x = x + out2

        assert x.shape == (B, T, D)
        assert next_logits2.shape == (B, pool.n_options)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_all_params_get_grad(self):
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        x, topk_idx, topk_weights = _make_inputs()
        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        loss = out.sum() + next_logits.sum()
        loss.backward()

        assert x.grad is not None and x.grad.abs().sum() > 0

        # Expert bank params should get gradient
        for name, p in pool.expert_bank.named_parameters():
            assert p.grad is not None, f"No gradient for expert_bank.{name}"

    def test_outbound_router_gets_grad(self):
        """Router bank should get gradient through execute_hop."""
        pool = _make_pool()
        x = torch.randn(B, T, D, requires_grad=True)
        topk_idx = torch.zeros(B, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, TOP_K), 1.0 / TOP_K)
        _, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        next_logits.sum().backward()
        assert pool.router_bank.grad is not None
        assert pool.router_bank.grad.abs().sum() > 0

    def test_multi_hop_backward(self):
        """Multi-hop logit chain backward should not crash."""
        pool = _make_pool()
        x = torch.randn(B, T, D, requires_grad=True)
        logits = pool.entry_router(x.mean(dim=1))

        for hop in range(3):
            topk_idx, topk_weights, _, _, _ = pool.route(logits, hops_used=hop)
            out, logits = pool.execute_hop(x, topk_idx, topk_weights, hop=hop)
            x = x + out

        loss = x.sum()
        loss.backward()  # should not crash


# ---------------------------------------------------------------------------
# Entry router
# ---------------------------------------------------------------------------

class TestEntryRouter:
    def test_entry_router_produces_valid_logits(self):
        pool = _make_pool()
        x = torch.randn(B, D)
        logits = pool.entry_router(x)
        assert logits.shape == (B, pool.n_options)

    def test_entry_router_gradient(self):
        pool = _make_pool()
        x = torch.randn(B, D, requires_grad=True)
        logits = pool.entry_router(x)
        logits.sum().backward()
        assert x.grad is not None
        assert pool.entry_router.weight.grad is not None

    def test_entry_logits_feed_into_route(self):
        """Entry router logits should work with route()."""
        pool = _make_pool()
        x = torch.randn(B, D)
        logits = pool.entry_router(x)
        topk_idx, topk_weights, has_exit, has_bridge, has_continue = \
            pool.route(logits, hops_used=0)
        assert topk_idx.shape == (B, TOP_K)


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
# Hop conditioning through execute_hop
# ---------------------------------------------------------------------------

class TestHopConditioning:
    def test_different_hops_produce_different_outputs(self):
        """Output should differ when hop changes."""
        pool = _make_pool(router_noise=0.0)
        pool.hop_gate_proj.bias.data.fill_(2.0)
        pool.eval()
        x = torch.randn(B, T, D)
        topk_idx = torch.zeros(B, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, TOP_K), 1.0 / TOP_K)

        with torch.no_grad():
            out0, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
            out10, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=10)

        assert not torch.allclose(out0, out10, atol=1e-6), \
            "Different hops produced identical outputs"

    def test_hop_embed_gets_gradient(self):
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        x = torch.randn(B, T, D, requires_grad=True)
        topk_idx = torch.zeros(B, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, TOP_K), 1.0 / TOP_K)
        out, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        out.sum().backward()
        assert pool.hop_embed_bank.grad is not None
        assert pool.hop_embed_bank.grad.abs().max() > 0


# ---------------------------------------------------------------------------
# Global hop conditioning
# ---------------------------------------------------------------------------

class TestGlobalHop:
    def test_global_hop_changes_conditioning(self):
        """Different global hop values should produce different outputs."""
        pool = _make_pool(router_noise=0.0, global_max_hops=48)
        pool.hop_gate_proj.bias.data.fill_(2.0)
        pool.eval()
        x = torch.randn(B, T, D)
        topk_idx = torch.zeros(B, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, TOP_K), 1.0 / TOP_K)

        with torch.no_grad():
            out0, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
            out10, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=10)

        assert not torch.allclose(out0, out10, atol=1e-6), \
            "Different global_hop values produced identical outputs"
