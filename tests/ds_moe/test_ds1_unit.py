"""DS1 unit tests: shapes, gradients, bank packing, feature flags, old-code parity."""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ds_moe.model import DS1, RMSNorm, SqueezeExcite, apply_interleaved_rope, relu_squared


B, L, D = 2, 16, 32
N, R = 64, 4


def _make_ds1(**kwargs):
    defaults = dict(dim=D, state_dim=N, mimo_rank=R, n_iters=2)
    defaults.update(kwargs)
    return DS1(**defaults)


def _make_bank(ds1):
    size = DS1.bank_size(ds1.D, ds1.N, ds1.R)
    w = torch.randn(size) * 0.02
    w.requires_grad_(True)
    return w


def _make_input():
    return torch.randn(B, L, D, requires_grad=True)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(D)
        x = torch.randn(B, L, D)
        assert norm(x).shape == (B, L, D)

    def test_unit_rms(self):
        norm = RMSNorm(D)
        x = torch.randn(B, L, D)
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=0.1, rtol=0.1)

    def test_gradient_flows_through_weight(self):
        norm = RMSNorm(D)
        x = torch.randn(B, L, D)
        out = norm(x).sum()
        out.backward()
        assert norm.weight.grad is not None
        assert norm.weight.grad.abs().sum() > 0


class TestInterleaveRoPE:
    def test_output_shape(self):
        x = torch.randn(B, R, L, N)
        angles = torch.randn(B, 1, L, N // 2)
        out = apply_interleaved_rope(x, angles)
        assert out.shape == x.shape

    def test_norm_preserving(self):
        x = torch.randn(B, R, L, N)
        angles = torch.randn(B, 1, L, N // 2)
        out = apply_interleaved_rope(x, angles)
        x_norm = x.norm(dim=-1)
        out_norm = out.norm(dim=-1)
        torch.testing.assert_close(x_norm, out_norm, atol=1e-5, rtol=1e-5)

    def test_different_angles_different_output(self):
        x = torch.randn(B, R, L, N)
        a1 = torch.randn(B, 1, L, N // 2)
        a2 = a1 + 1.0
        o1 = apply_interleaved_rope(x, a1)
        o2 = apply_interleaved_rope(x, a2)
        assert not torch.allclose(o1, o2)

    def test_zero_angle_identity(self):
        x = torch.randn(B, R, L, N)
        angles = torch.zeros(B, 1, L, N // 2)
        out = apply_interleaved_rope(x, angles)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)


class TestSqueezeExcite:
    def test_output_shape(self):
        se = SqueezeExcite(D)
        x = torch.randn(B, L, D)
        assert se(x).shape == (B, L, D)

    def test_scale_bounded(self):
        se = SqueezeExcite(D)
        x = torch.randn(B, L, D)
        with torch.no_grad():
            scale = x.mean(dim=1)
            scale = torch.sigmoid(se.fc2(se.act(se.fc1(scale))))
        assert (scale >= 0).all() and (scale <= 1).all()


class TestDS1Shape:
    def test_output_shape_basic(self):
        ds1 = _make_ds1()
        w = _make_bank(ds1)
        x = _make_input()
        y = ds1(x, w)
        assert y.shape == (B, L, D)

    def test_output_shape_with_all_features(self):
        ds1 = _make_ds1(diffuse_se=True, diff_inject=True, diff_readout=True, bc_norm=True)
        w = _make_bank(ds1)
        x = _make_input()
        y = ds1(x, w)
        assert y.shape == (B, L, D)

    def test_output_shape_relu2(self):
        ds1 = _make_ds1(relu2=True)
        w = _make_bank(ds1)
        x = _make_input()
        y = ds1(x, w)
        assert y.shape == (B, L, D)

    def test_bank_size_matches_usage(self):
        ds1 = _make_ds1()
        expected = DS1.bank_size(D, N, R)
        w = torch.randn(expected)
        ds1(torch.randn(1, 4, D), w)


class TestDS1BankPacking:
    def test_bank_size_calculation(self):
        size = DS1.bank_size(D, N, R)
        expected = (D * N * R      # to_B
                    + D * N * R    # to_C
                    + D * R        # to_X
                    + D * N        # to_decay
                    + D * (N // 2) # to_theta
                    + D * 1        # to_lambda
                    + N * R * D)   # out_proj
        assert size == expected

    def test_wrong_bank_size_raises(self):
        ds1 = _make_ds1()
        w = torch.randn(10)
        with pytest.raises((AssertionError, RuntimeError)):
            ds1(torch.randn(1, 4, D), w)


class TestDS1Gradient:
    def test_gradient_flows_to_input(self):
        ds1 = _make_ds1()
        w = _make_bank(ds1)
        x = _make_input()
        y = ds1(x, w)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_flows_to_bank_weights(self):
        ds1 = _make_ds1()
        w = _make_bank(ds1)
        x = _make_input()
        y = ds1(x, w)
        y.sum().backward()
        assert w.grad is not None
        assert w.grad.abs().sum() > 0

    def test_gradient_flows_to_all_6_projections(self):
        ds1 = _make_ds1()
        size = DS1.bank_size(D, N, R)
        w = torch.randn(size, requires_grad=True)
        x = _make_input()
        y = ds1(x, w)
        y.sum().backward()

        grad = w.grad
        assert grad is not None

        idx = 0
        proj_names = ['to_B', 'to_C', 'to_X', 'to_decay', 'to_theta', 'to_lambda', 'out_proj']
        proj_sizes = [
            D * N * R, D * N * R, D * R, D * N, D * (N // 2), D * 1, N * R * D
        ]
        for name, sz in zip(proj_names, proj_sizes):
            proj_grad = grad[idx:idx + sz]
            grad_norm = proj_grad.abs().sum().item()
            assert grad_norm > 0, f"No gradient flow to {name} (grad norm = {grad_norm})"
            idx += sz

    def test_gradient_flows_to_small_params(self):
        ds1 = _make_ds1(diff_inject=True, diff_readout=True, bc_norm=True, diffuse_se=True)
        w = _make_bank(ds1)
        x = _make_input()
        y = ds1(x, w)
        y.sum().backward()

        for name, p in ds1.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert p.grad.abs().sum() > 0, f"Zero grad for {name}"

    def test_gradient_magnitude_comparable(self):
        ds1 = _make_ds1()
        w = _make_bank(ds1)
        x = _make_input()
        y = ds1(x, w)
        y.sum().backward()

        idx = 0
        proj_sizes = [D * N * R, D * N * R, D * R, D * N, D * (N // 2), D * 1, N * R * D]
        norms = []
        for sz in proj_sizes:
            norms.append(w.grad[idx:idx + sz].norm().item())
            idx += sz

        max_norm = max(norms)
        min_norm = min(norms)
        ratio = max_norm / (min_norm + 1e-10)
        assert ratio < 1000, (
            f"Gradient magnitude ratio {ratio:.0f}x across projections "
            f"(max={max_norm:.4e}, min={min_norm:.4e}). "
            f"This was the bug in old code — check norm placement."
        )


class TestDS1FeatureFlags:
    def test_diff_inject_changes_output(self):
        torch.manual_seed(42)
        ds1_off = _make_ds1(diff_inject=False)
        ds1_on = _make_ds1(diff_inject=True)
        w = torch.randn(DS1.bank_size(D, N, R))
        x = torch.randn(B, L, D)
        y_off = ds1_off(x, w)
        y_on = ds1_on(x, w)
        assert not torch.allclose(y_off, y_on, atol=1e-4)

    def test_diff_readout_changes_output(self):
        torch.manual_seed(42)
        ds1_off = _make_ds1(diff_readout=False)
        ds1_on = _make_ds1(diff_readout=True)
        w = torch.randn(DS1.bank_size(D, N, R))
        x = torch.randn(B, L, D)
        y_off = ds1_off(x, w)
        y_on = ds1_on(x, w)
        assert not torch.allclose(y_off, y_on, atol=1e-4)

    def test_bc_norm_changes_output(self):
        torch.manual_seed(42)
        ds1_off = _make_ds1(bc_norm=False)
        ds1_on = _make_ds1(bc_norm=True)
        w = torch.randn(DS1.bank_size(D, N, R))
        x = torch.randn(B, L, D)
        y_off = ds1_off(x, w)
        y_on = ds1_on(x, w)
        assert not torch.allclose(y_off, y_on, atol=1e-4)

    def test_se_changes_output(self):
        torch.manual_seed(42)
        ds1_off = _make_ds1(diffuse_se=False)
        ds1_on = _make_ds1(diffuse_se=True)
        w = torch.randn(DS1.bank_size(D, N, R))
        x = torch.randn(B, L, D)
        y_off = ds1_off(x, w)
        y_on = ds1_on(x, w)
        assert not torch.allclose(y_off, y_on, atol=1e-4)

    def test_n_iters_affects_output(self):
        torch.manual_seed(42)
        ds1_1 = _make_ds1(n_iters=1)
        ds1_3 = _make_ds1(n_iters=3)
        w = torch.randn(DS1.bank_size(D, N, R))
        x = torch.randn(B, L, D)
        y1 = ds1_1(x, w)
        y3 = ds1_3(x, w)
        assert not torch.allclose(y1, y3, atol=1e-4)


class TestDS1OldCodeParity:
    """Compare DS1 (bank pattern) against old MIMOJacobiSSM (stored weights).

    We manually copy weights from the old code's nn.Linear modules into a flat
    bank vector, then verify outputs match. This is the M1 validation milestone.
    """

    @staticmethod
    def _zero_linear_biases(old):
        """Zero all nn.Linear biases so old code matches DS1's bias-free projections.
        DS1 only has explicit B_bias/C_bias; all other projection biases are removed.
        """
        with torch.no_grad():
            for proj in [old.to_B, old.to_C, old.to_X, old.to_decay, old.to_theta, old.to_lambda, old.out_proj]:
                if proj.bias is not None:
                    proj.bias.zero_()

    @staticmethod
    def _pack_old_weights(old) -> torch.Tensor:
        """Pack old code's nn.Linear weights into DS1 bank format.
        Old uses (out, in) layout; DS1 bank uses transposed (in, out).
        """
        parts = []
        for proj in [old.to_B, old.to_C, old.to_X, old.to_decay, old.to_theta, old.to_lambda]:
            parts.append(proj.weight.data.T.reshape(-1))
        parts.append(old.out_proj.weight.data.T.reshape(-1))
        return torch.cat(parts)

    @staticmethod
    def _make_ds1_no_pos_rope(dim, state_dim, mimo_rank, n_iters, **kwargs):
        """DS1 subclass that skips positional RoPE (for old-code parity).
        Old code: angles = theta * (i+1)
        New code: angles = theta * pos * (i+1)
        """
        ds1 = DS1(dim=dim, state_dim=state_dim, mimo_rank=mimo_rank,
                  n_iters=n_iters, **kwargs)

        original_forward = ds1.forward

        def forward_no_pos(x, ssm_w):
            B_batch, L, D = x.shape
            N, R = ds1.N, ds1.R
            act = ds1.act
            w_B, w_C, w_X, w_decay, w_theta, w_lambda, w_out = ds1._unpack_weights(ssm_w)

            from ds_moe.model import apply_interleaved_rope

            B_proj = act(x @ w_B + ds1.B_bias)
            C_proj = act(x @ w_C + ds1.C_bias)
            X_r = act(x @ w_X)
            decay = torch.sigmoid(x @ w_decay)
            theta = x @ w_theta
            lam = torch.sigmoid(x @ w_lambda)

            B_base = B_proj.view(B_batch, L, N, R).permute(0, 3, 1, 2).contiguous()
            C_base = C_proj.view(B_batch, L, N, R).permute(0, 3, 1, 2).contiguous()

            if ds1.bc_norm:
                B_base = ds1.b_norm(B_base)
                C_base = ds1.c_norm(C_base)

            X_r = X_r.permute(0, 2, 1).unsqueeze(-1)
            decay = decay.unsqueeze(1)
            lam = lam.unsqueeze(1)

            H = torch.zeros(B_batch, R, L, N, device=x.device, dtype=x.dtype)
            C_rot = C_base
            prev_inject = None

            for i in range(ds1.n_iters):
                angles = theta.unsqueeze(1) * (i + 1)  # NO pos multiplication
                B_rot = apply_interleaved_rope(B_base, angles)
                C_rot = apply_interleaved_rope(C_base, angles)

                if ds1.diff_inject:
                    half_N = N // 2
                    B1, B2 = B_rot[..., :half_N], B_rot[..., half_N:]
                    inj1 = B1 * X_r
                    inj2 = B2 * X_r
                    inject = torch.cat([inj1 - ds1.inject_lambda * inj2,
                                        inj1 + ds1.inject_lambda * inj2], dim=-1)
                else:
                    inject = B_rot * X_r

                H = H.permute(0, 1, 3, 2).reshape(B_batch * R, N, L)
                H = ds1.diffuse(H)
                H = H.reshape(B_batch, R, N, L).permute(0, 1, 3, 2)

                if ds1.has_se:
                    H_flat = H.reshape(B_batch * R, L, N)
                    H_flat = ds1.diffuse_se(H_flat)
                    H = H_flat.reshape(B_batch, R, L, N)

                alpha = decay
                gamma = lam
                if prev_inject is not None:
                    beta = (1.0 - gamma) * alpha
                    H = alpha * H + beta * prev_inject + gamma * inject
                else:
                    H = alpha * H + inject
                prev_inject = inject

            if ds1.diff_readout:
                half_N = N // 2
                C1, C2 = C_rot[..., :half_N], C_rot[..., half_N:]
                H1, H2 = H[..., :half_N], H[..., half_N:]
                g1, g2 = C1 * H1, C2 * H2
                gated = torch.cat([g1 - ds1.readout_lambda * g2,
                                   g1 + ds1.readout_lambda * g2], dim=-1)
            else:
                gated = C_rot * H

            out = gated.permute(0, 2, 1, 3).reshape(B_batch, L, N * R)
            return act(out @ w_out)

        import types
        ds1.forward = types.MethodType(lambda self, x, w: forward_no_pos(x, w), ds1)
        ds1._forward_no_pos = forward_no_pos
        return ds1

    def test_output_matches_old_code(self):
        """Parity test with old MIMOJacobiSSM. Differences accounted for:
        - DS1 has no bias on projections (zeroed on old code)
        - DS1 adds positional RoPE (disabled via no-pos variant)
        """
        from heuristic_secrets.models.scatter_attention import MIMOJacobiSSM
        torch.manual_seed(123)
        old = MIMOJacobiSSM(dim=D, state_dim=N, mimo_rank=R, n_iters=2)
        self._zero_linear_biases(old)

        new = self._make_ds1_no_pos_rope(D, N, R, n_iters=2)
        with torch.no_grad():
            new.B_bias.copy_(old.B_bias)
            new.C_bias.copy_(old.C_bias)
            new.diffuse.weight.copy_(old.diffuse.weight)
            new.diffuse.bias.copy_(old.diffuse.bias)

        bank = self._pack_old_weights(old)
        torch.manual_seed(99)
        x = torch.randn(B, L, D)

        with torch.no_grad():
            y_old = old(x)
            y_new = new._forward_no_pos(x, bank)

        torch.testing.assert_close(y_old, y_new, atol=1e-4, rtol=1e-4)

    def test_output_matches_with_diff_features(self):
        from heuristic_secrets.models.scatter_attention import MIMOJacobiSSM
        torch.manual_seed(123)
        old = MIMOJacobiSSM(dim=D, state_dim=N, mimo_rank=R, n_iters=2,
                            diff_inject=True, diff_readout=True, bc_norm=True)
        self._zero_linear_biases(old)

        new = self._make_ds1_no_pos_rope(D, N, R, n_iters=2,
                                         diff_inject=True, diff_readout=True, bc_norm=True)
        with torch.no_grad():
            new.B_bias.copy_(old.B_bias)
            new.C_bias.copy_(old.C_bias)
            new.diffuse.weight.copy_(old.diffuse.weight)
            new.diffuse.bias.copy_(old.diffuse.bias)
            new.inject_lambda.copy_(old.inject_lambda)
            new.readout_lambda.copy_(old.readout_lambda)
            new.b_norm.weight.copy_(old.b_norm.weight)
            new.c_norm.weight.copy_(old.c_norm.weight)

        bank = self._pack_old_weights(old)
        torch.manual_seed(99)
        x = torch.randn(B, L, D)

        with torch.no_grad():
            y_old = old(x)
            y_new = new._forward_no_pos(x, bank)

        torch.testing.assert_close(y_old, y_new, atol=1e-4, rtol=1e-4)
