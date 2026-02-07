"""
Scan implementations for USB.

Primary path: chunked decay-masked scan (forward_scan_chunked).
Decomposes the recurrence into:
  1. Tridiagonal conv: s_t = zeta_t * kv_t + epsilon_t * kv_{t-1} + delta_t * kv_{t-2}
  2. Decay-masked matmul within chunks of size C:
     out[i] = sum_{j<=i} exp(cumA[i] - cumA[j]) * s[j]  (within chunk)
  3. Inter-chunk state passing: carry = exp(chunk_cumA) * carry + chunk_contribution

The within-chunk step is a (C,C) @ (C,D) matmul with a lower-triangular
exponential-decay mask — same structure as Mamba-2 SSD.

Full forward + backward implemented via torch.autograd.Function.
Triton kernels for CUDA, PyTorch fallback for CPU.
"""

import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _chunk_scan_fwd_kernel(
        # Pointers
        s_ptr, log_alpha_ptr, prev_states_ptr, out_ptr, cumA_out_ptr,
        # Dimensions
        chunk_size: tl.constexpr, D: tl.constexpr, nchunks,
        # Strides for s: (BNH, C, D) -- flattened
        stride_s_bnh, stride_s_c, stride_s_d,
        # Strides for log_alpha: (BNH, C)
        stride_la_bnh, stride_la_c,
        # Strides for prev_states: (BNH, D)
        stride_ps_bnh, stride_ps_d,
        # Strides for out: (BNH, C, D)
        stride_o_bnh, stride_o_c, stride_o_d,
        # Strides for cumA_out: (BNH, C) -- written only by pid_d==0
        stride_cao_bnh, stride_cao_c,
        # Block sizes
        BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
        STORE_CUMA: tl.constexpr,  # whether to write cumA to output
    ):
        """Fused intra-chunk decay-mask matmul + inter-chunk state contribution.
        
        Computes cumsum of log_alpha in-kernel (eliminates aten::cumsum hotspot).
        Optionally writes cumA to output for use in backward pass.
        """
        pid_bnh = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_c = tl.arange(0, BLOCK_C)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D

        # Load log_alpha and compute cumsum in-register
        la_base = pid_bnh * stride_la_bnh
        log_alpha = tl.load(log_alpha_ptr + la_base + offs_c * stride_la_c,
                            mask=offs_c < chunk_size, other=0.0).to(tl.float32)
        cumA = tl.cumsum(log_alpha, axis=0)

        # Optionally store cumA for backward (only one D-block needs to do this)
        if STORE_CUMA:
            if pid_d == 0:
                cao_base = pid_bnh * stride_cao_bnh
                tl.store(cumA_out_ptr + cao_base + offs_c * stride_cao_c,
                         cumA, mask=offs_c < chunk_size)

        # Decay mask: M[i,j] = exp(cumA[i] - cumA[j]) for j <= i
        decay = tl.exp(cumA[:, None] - cumA[None, :])
        causal = offs_c[:, None] >= offs_c[None, :]
        decay = tl.where(causal, decay, 0.0)

        # Load s: (C, BLOCK_D)
        s_base = pid_bnh * stride_s_bnh
        s_ptrs = s_base + offs_c[:, None] * stride_s_c + offs_d[None, :] * stride_s_d
        s_tile = tl.load(s_ptr + s_ptrs,
                         mask=(offs_c[:, None] < chunk_size) & d_mask[None, :],
                         other=0.0).to(tl.float32)

        # Intra: M @ s
        intra = tl.dot(decay.to(s_tile.dtype), s_tile)

        # Inter: prev_state * exp(cumA[i])
        ps_base = pid_bnh * stride_ps_bnh
        prev_state = tl.load(prev_states_ptr + ps_base + offs_d * stride_ps_d,
                             mask=d_mask, other=0.0).to(tl.float32)
        inter = tl.exp(cumA)[:, None] * prev_state[None, :]

        result = intra + inter

        o_base = pid_bnh * stride_o_bnh
        o_ptrs = o_base + offs_c[:, None] * stride_o_c + offs_d[None, :] * stride_o_d
        tl.store(out_ptr + o_ptrs, result,
                 mask=(offs_c[:, None] < chunk_size) & d_mask[None, :])

    @triton.jit
    def _chunk_scan_bwd_kernel(
        # Pointers
        dout_ptr, log_alpha_ptr, s_ptr, prev_states_ptr,
        ds_ptr, d_decay_mask_ptr, d_prev_states_ptr,
        # Dimensions
        chunk_size: tl.constexpr, D: tl.constexpr,
        # Strides for dout/ds: (BNH, C, D)
        stride_do_bnh, stride_do_c, stride_do_d,
        # Strides for log_alpha: (BNH, C)
        stride_la_bnh, stride_la_c,
        # Strides for s: (BNH, C, D)
        stride_s_bnh, stride_s_c, stride_s_d,
        # Strides for prev_states: (BNH, D)
        stride_ps_bnh, stride_ps_d,
        # Strides for ds: (BNH, C, D)
        stride_ds_bnh, stride_ds_c, stride_ds_d,
        # Strides for d_decay_mask: (BNH, C, C)
        stride_ddm_bnh, stride_ddm_i, stride_ddm_j,
        # Strides for d_prev_states: (BNH, D)
        stride_dps_bnh, stride_dps_d,
        # Block sizes
        BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """
        Backward through intra-chunk scan.
        Mirrors _chunked_scan_bwd steps 2-5 for one (bnh, d_block).

        Computes cumsum of log_alpha in-kernel (fused, no aten::cumsum).
        Computes:
          ds_flat = decay_mask^T @ d_intra_flat
          d_decay_mask = d_intra_flat @ s_flat^T  (partial — accumulated across D blocks)
          d_prev_states = sum_i d_inter[i] * exp(cumA[i])  per D
        """
        pid_bnh = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_c = tl.arange(0, BLOCK_C)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        c_mask = offs_c < chunk_size

        # Load log_alpha -> compute cumsum -> build decay mask
        la_base = pid_bnh * stride_la_bnh
        log_alpha = tl.load(log_alpha_ptr + la_base + offs_c * stride_la_c,
                            mask=c_mask, other=0.0).to(tl.float32)
        cumA = tl.cumsum(log_alpha, axis=0)
        decay = tl.exp(cumA[:, None] - cumA[None, :])
        causal = offs_c[:, None] >= offs_c[None, :]
        M = tl.where(causal, decay, 0.0)
        MT = tl.trans(M)

        # Load dout (= d_intra for this D-block): (C, BLOCK_D)
        do_base = pid_bnh * stride_do_bnh
        dout = tl.load(dout_ptr + do_base + offs_c[:, None] * stride_do_c + offs_d[None, :] * stride_do_d,
                       mask=c_mask[:, None] & d_mask[None, :],
                       other=0.0).to(tl.float32)

        # Load s: (C, BLOCK_D)
        s_base = pid_bnh * stride_s_bnh
        s_tile = tl.load(s_ptr + s_base + offs_c[:, None] * stride_s_c + offs_d[None, :] * stride_s_d,
                         mask=c_mask[:, None] & d_mask[None, :],
                         other=0.0).to(tl.float32)

        # ds = M^T @ dout  (step 2 reverse: d_s_flat = decay_mask^T @ d_intra_flat)
        ds = tl.dot(MT.to(dout.dtype), dout)
        ds_base = pid_bnh * stride_ds_bnh
        tl.store(ds_ptr + ds_base + offs_c[:, None] * stride_ds_c + offs_d[None, :] * stride_ds_d,
                 ds, mask=c_mask[:, None] & d_mask[None, :])

        # d_decay_mask partial = dout @ s^T  (C, C) — accumulate across D-blocks
        d_dm_partial = tl.dot(dout, tl.trans(s_tile))  # (C, C)
        ddm_base = pid_bnh * stride_ddm_bnh
        if pid_d == 0:
            tl.store(d_decay_mask_ptr + ddm_base
                     + offs_c[:, None] * stride_ddm_i + offs_c[None, :] * stride_ddm_j,
                     d_dm_partial, mask=c_mask[:, None] & c_mask[None, :])
        else:
            tl.atomic_add(d_decay_mask_ptr + ddm_base
                          + offs_c[:, None] * stride_ddm_i + offs_c[None, :] * stride_ddm_j,
                          d_dm_partial, mask=c_mask[:, None] & c_mask[None, :])

        # d_prev_states = sum_i dout[i] * exp(cumA[i])  (step 5 reverse)
        exp_cumA = tl.exp(cumA)  # (C,)
        d_prev = tl.sum(dout * exp_cumA[:, None], axis=0)  # (BLOCK_D,)
        dps_base = pid_bnh * stride_dps_bnh
        tl.store(d_prev_states_ptr + dps_base + offs_d * stride_dps_d,
                 d_prev, mask=d_mask)

    @triton.jit
    def _d_cumA_from_decay_mask_kernel(
        d_decay_mask_ptr, log_alpha_ptr, d_cumA_ptr,
        chunk_size: tl.constexpr,
        stride_ddm_bnh, stride_ddm_i, stride_ddm_j,
        stride_la_bnh, stride_la_c,
        stride_dca_bnh, stride_dca_c,
        BLOCK_C: tl.constexpr,
    ):
        """
        Recomputes decay_mask from log_alpha (fused cumsum), then:
          d_cumA[i] = sum_j (d_dm * dm)[i,j]  -  sum_j (d_dm * dm)[j,i]
        (row sum minus col sum of elementwise product)
        """
        pid_bnh = tl.program_id(0)
        offs_c = tl.arange(0, BLOCK_C)
        c_mask = offs_c < chunk_size

        ddm_base = pid_bnh * stride_ddm_bnh

        # Recompute cumA and decay_mask from log_alpha (no need to load pre-computed)
        la_base = pid_bnh * stride_la_bnh
        log_alpha = tl.load(log_alpha_ptr + la_base + offs_c * stride_la_c,
                            mask=c_mask, other=0.0).to(tl.float32)
        cumA = tl.cumsum(log_alpha, axis=0)
        decay_diff = cumA[:, None] - cumA[None, :]
        causal = offs_c[:, None] >= offs_c[None, :]
        dm = tl.where(causal, tl.exp(decay_diff), 0.0)

        # Load d_decay_mask: (C, C)
        ddm = tl.load(d_decay_mask_ptr + ddm_base
                      + offs_c[:, None] * stride_ddm_i + offs_c[None, :] * stride_ddm_j,
                      mask=c_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)

        prod = ddm * dm  # (C, C)
        d_cumA = tl.sum(prod, axis=1) - tl.sum(prod, axis=0)  # (C,)

        dca_base = pid_bnh * stride_dca_bnh
        tl.store(d_cumA_ptr + dca_base + offs_c * stride_dca_c,
                 d_cumA, mask=c_mask)

    @triton.jit
    def _state_passing_fwd_kernel(
        chunk_states_ptr, chunk_decay_ptr, init_state_ptr, out_states_ptr, final_state_ptr,
        dim, nchunks,
        stride_cs_b, stride_cs_nc, stride_cs_h, stride_cs_d,
        stride_cd_b, stride_cd_nc, stride_cd_h,
        stride_is_b, stride_is_h, stride_is_d,
        stride_os_b, stride_os_nc, stride_os_h, stride_os_d,
        stride_fs_b, stride_fs_h, stride_fs_d,
        BLOCK_D: tl.constexpr,
    ):
        """Forward state passing across chunks."""
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < dim

        is_base = pid_b * stride_is_b + pid_h * stride_is_h
        state = tl.load(init_state_ptr + is_base + offs_d * stride_is_d,
                        mask=d_mask, other=0.0).to(tl.float32)

        os_base = pid_b * stride_os_b + pid_h * stride_os_h
        tl.store(out_states_ptr + os_base + 0 * stride_os_nc + offs_d * stride_os_d,
                 state, mask=d_mask)

        for c in range(nchunks):
            cs_ptr = pid_b * stride_cs_b + c * stride_cs_nc + pid_h * stride_cs_h
            new_state = tl.load(chunk_states_ptr + cs_ptr + offs_d * stride_cs_d,
                                mask=d_mask, other=0.0).to(tl.float32)
            cd_ptr = pid_b * stride_cd_b + c * stride_cd_nc + pid_h * stride_cd_h
            decay = tl.exp(tl.load(chunk_decay_ptr + cd_ptr).to(tl.float32))

            state = decay * state + new_state

            if c < nchunks - 1:
                tl.store(out_states_ptr + os_base + (c + 1) * stride_os_nc + offs_d * stride_os_d,
                         state, mask=d_mask)
            else:
                tl.store(final_state_ptr + pid_b * stride_fs_b + pid_h * stride_fs_h + offs_d * stride_fs_d,
                         state, mask=d_mask)

    @triton.jit
    def _state_passing_bwd_kernel(
        d_prev_states_ptr, prev_states_ptr, chunk_decay_ptr,
        d_chunk_new_state_ptr, d_chunk_decay_ptr, d_init_state_ptr,
        dim, nchunks,
        # Strides for d_prev_states: (B, nchunks, H, D)
        stride_dps_b, stride_dps_nc, stride_dps_h, stride_dps_d,
        # Strides for prev_states: (B, nchunks, H, D)
        stride_ps_b, stride_ps_nc, stride_ps_h, stride_ps_d,
        # Strides for chunk_decay: (B, nchunks, H)
        stride_cd_b, stride_cd_nc, stride_cd_h,
        # Strides for d_chunk_new_state: (B, nchunks, H, D)
        stride_dcns_b, stride_dcns_nc, stride_dcns_h, stride_dcns_d,
        # Strides for d_chunk_decay: (B, nchunks, H)
        stride_dcd_b, stride_dcd_nc, stride_dcd_h,
        # Strides for d_init_state: (B, H, D)
        stride_dis_b, stride_dis_h, stride_dis_d,
        BLOCK_D: tl.constexpr,
    ):
        """
        Backward state passing: reverse sequential scan.

        Matches the Python _chunked_scan_bwd step 4:
          for c in range(nchunks-1, -1, -1):
              decay_c = exp(chunk_total_decay[c])
              d_cns[c] = d_state
              d_td[c] = (d_state * decay_c * prev_states[c]).sum(-1)
              d_state = decay_c * d_state + d_prev_states[c]
          d_init_state = d_state
        """
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < dim

        d_state = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for c_rev in range(nchunks):
            c = nchunks - 1 - c_rev

            cd_ptr = pid_b * stride_cd_b + c * stride_cd_nc + pid_h * stride_cd_h
            decay = tl.exp(tl.load(chunk_decay_ptr + cd_ptr).to(tl.float32))

            # d_cns[c] = d_state (BEFORE adding d_prev_states)
            dcns_ptr = pid_b * stride_dcns_b + c * stride_dcns_nc + pid_h * stride_dcns_h
            tl.store(d_chunk_new_state_ptr + dcns_ptr + offs_d * stride_dcns_d,
                     d_state, mask=d_mask)

            # d_td[c] = dot(d_state, decay * prev_states[c])
            ps_ptr = pid_b * stride_ps_b + c * stride_ps_nc + pid_h * stride_ps_h
            prev_s = tl.load(prev_states_ptr + ps_ptr + offs_d * stride_ps_d,
                             mask=d_mask, other=0.0).to(tl.float32)
            d_decay_scalar = tl.sum(d_state * (decay * prev_s))
            dcd_addr = d_chunk_decay_ptr + pid_b * stride_dcd_b + c * stride_dcd_nc + pid_h * stride_dcd_h
            if pid_d == 0:
                tl.store(dcd_addr, d_decay_scalar)
            else:
                tl.atomic_add(dcd_addr, d_decay_scalar)

            # d_state = decay * d_state + d_prev_states[c]
            dps_ptr = pid_b * stride_dps_b + c * stride_dps_nc + pid_h * stride_dps_h
            d_from_inter = tl.load(d_prev_states_ptr + dps_ptr + offs_d * stride_dps_d,
                                   mask=d_mask, other=0.0).to(tl.float32)
            d_state = decay * d_state + d_from_inter

        # d_init_state
        tl.store(d_init_state_ptr + pid_b * stride_dis_b + pid_h * stride_dis_h + offs_d * stride_dis_d,
                 d_state, mask=d_mask)

    @triton.jit
    def _inter_chunk_bwd_kernel(
        # Pointers
        dout_ptr, cumA_ptr, prev_states_ptr,
        d_prev_states_ptr, d_cumA_inter_ptr,
        # Dimensions
        chunk_size: tl.constexpr, D: tl.constexpr,
        # Strides for dout: (BNH, C, D)
        stride_do_bnh, stride_do_c, stride_do_d,
        # Strides for cumA: (BNH, C)
        stride_ca_bnh, stride_ca_c,
        # Strides for prev_states: (BNH, D)
        stride_ps_bnh, stride_ps_d,
        # Strides for d_prev_states: (BNH, D)
        stride_dps_bnh, stride_dps_d,
        # Strides for d_cumA_inter: (BNH, C)
        stride_dca_bnh, stride_dca_c,
        # Block sizes
        BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Fused inter-chunk backward: computes d_prev_states and d_cumA_from_inter
        without materializing (BNH, C, D)-sized intermediates.

        d_prev_states[bnh, d] = sum_c dout[bnh, c, d] * exp(cumA[bnh, c])
        d_cumA_inter[bnh, c] = sum_d dout[bnh, c, d] * prev_states[bnh, d] * exp(cumA[bnh, c])
        """
        pid_bnh = tl.program_id(0)

        offs_c = tl.arange(0, BLOCK_C)
        offs_d = tl.arange(0, BLOCK_D)
        c_mask = offs_c < chunk_size
        d_mask = offs_d < D

        # Load cumA and compute exp(cumA) — (BLOCK_C,)
        ca_base = pid_bnh * stride_ca_bnh
        cumA = tl.load(cumA_ptr + ca_base + offs_c * stride_ca_c,
                       mask=c_mask, other=0.0).to(tl.float32)
        exp_ca = tl.exp(cumA)  # (BLOCK_C,)

        # Load prev_states — (BLOCK_D,)
        ps_base = pid_bnh * stride_ps_bnh
        prev_s = tl.load(prev_states_ptr + ps_base + offs_d * stride_ps_d,
                         mask=d_mask, other=0.0).to(tl.float32)

        # Load dout — (BLOCK_C, BLOCK_D)
        do_base = pid_bnh * stride_do_bnh
        dout = tl.load(dout_ptr + do_base + offs_c[:, None] * stride_do_c + offs_d[None, :] * stride_do_d,
                       mask=c_mask[:, None] & d_mask[None, :],
                       other=0.0).to(tl.float32)

        # d_prev_states[d] = sum_c dout[c, d] * exp_ca[c]
        d_prev = tl.sum(dout * exp_ca[:, None], axis=0)  # (BLOCK_D,)
        dps_base = pid_bnh * stride_dps_bnh
        tl.store(d_prev_states_ptr + dps_base + offs_d * stride_dps_d,
                 d_prev, mask=d_mask)

        # d_cumA_inter[c] = sum_d dout[c, d] * prev_s[d] * exp_ca[c]
        #                 = exp_ca[c] * sum_d dout[c, d] * prev_s[d]
        dot_dp = tl.sum(dout * prev_s[None, :], axis=1)  # (BLOCK_C,)
        d_cumA = exp_ca * dot_dp  # (BLOCK_C,)
        dca_base = pid_bnh * stride_dca_bnh
        tl.store(d_cumA_inter_ptr + dca_base + offs_c * stride_dca_c,
                 d_cumA, mask=c_mask)


# ---------------------------------------------------------------------------
# Core logic: PyTorch (works everywhere, used on CPU and as Triton fallback)
# ---------------------------------------------------------------------------

def _tridiag_conv(kv, zeta, epsilon, delta):
    """s_t = zeta_t * kv_t + epsilon_t * kv_{t-1} + delta_t * kv_{t-2}"""
    kv_tm1 = F.pad(kv[:, :-1], (0, 0, 0, 0, 1, 0))
    kv_tm2 = F.pad(kv[:, :-2], (0, 0, 0, 0, 2, 0))
    return zeta[..., None] * kv + epsilon[..., None] * kv_tm1 + delta[..., None] * kv_tm2


def _tridiag_conv_backward(ds, kv, zeta, epsilon, delta):
    """Backward through tridiag conv."""
    B, T, H, D = kv.shape
    kv_tm1 = F.pad(kv[:, :-1], (0, 0, 0, 0, 1, 0))
    kv_tm2 = F.pad(kv[:, :-2], (0, 0, 0, 0, 2, 0))

    # d_zeta[t] = sum_d ds[t,d] * kv[t,d]
    d_zeta = (ds * kv).sum(dim=-1)
    d_epsilon = (ds * kv_tm1).sum(dim=-1)
    d_delta = (ds * kv_tm2).sum(dim=-1)

    # d_kv: kv[t] contributes to s[t] (via zeta), s[t+1] (via epsilon), s[t+2] (via delta)
    d_kv = zeta[..., None] * ds
    # From epsilon: ds[t+1] * epsilon[t+1] -> d_kv[t]
    d_kv[:, :-1] += epsilon[:, 1:, :, None] * ds[:, 1:]
    # From delta: ds[t+2] * delta[t+2] -> d_kv[t]
    d_kv[:, :-2] += delta[:, 2:, :, None] * ds[:, 2:]

    return d_kv, d_zeta, d_epsilon, d_delta


def _chunked_scan_fwd(s, alpha, init_state, chunk_size):
    """
    Chunked decay-masked scan (differentiable via autograd).

    All ops are autograd-friendly so this works on CPU without a custom backward.
    On CUDA with Triton, the ChunkedScanFn wrapper provides a fused backward.
    """
    B, T, H, D = s.shape
    device = s.device

    # Use float32 for speed in training, but preserve float64 for gradcheck
    compute_dtype = torch.float32 if s.dtype != torch.float64 else torch.float64
    s_f = s.to(compute_dtype)
    alpha_f = alpha.to(compute_dtype).clamp(min=1e-6)
    log_alpha = alpha_f.log()

    nchunks = (T + chunk_size - 1) // chunk_size
    T_padded = nchunks * chunk_size
    if T_padded > T:
        s_f = F.pad(s_f, (0, 0, 0, 0, 0, T_padded - T))
        log_alpha = F.pad(log_alpha, (0, 0, 0, T_padded - T))

    # Reshape: (B, nchunks, C, H, D) and (B, nchunks, C, H)
    s_chunks = s_f.view(B, nchunks, chunk_size, H, D)
    la_chunks = log_alpha.view(B, nchunks, chunk_size, H)
    cumA = la_chunks.cumsum(dim=2)  # (B, nchunks, C, H)
    chunk_total_decay = cumA[:, :, -1, :]  # (B, nchunks, H) log-space

    # Flatten to (BNH, C, D) for bmm
    BNH = B * nchunks * H
    cumA_flat = cumA.permute(0, 1, 3, 2).reshape(BNH, chunk_size)
    s_flat = s_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D)

    # Build decay mask and do intra-chunk matmul
    # Zero out upper-triangular BEFORE exp to avoid huge values leaking gradients
    causal = torch.tril(torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool))
    decay_diff = cumA_flat[:, :, None] - cumA_flat[:, None, :]  # (BNH, C, C)
    decay_diff = decay_diff.masked_fill(~causal, -float('inf'))
    decay_mask = decay_diff.exp()  # upper triangle → 0 via exp(-inf), grad-safe
    intra_flat = torch.bmm(decay_mask, s_flat)  # (BNH, C, D)
    intra = intra_flat.view(B, nchunks, H, chunk_size, D).permute(0, 1, 3, 2, 4)

    # chunk_new_state = intra at last position (already has correct decay)
    chunk_new_state = intra[:, :, -1, :, :]  # (B, nchunks, H, D)

    # State passing (autograd-friendly: list + stack, no in-place ops)
    prev_list = []
    state = init_state.to(compute_dtype)
    for c in range(nchunks):
        prev_list.append(state)
        state = chunk_total_decay[:, c].exp()[..., None] * state + chunk_new_state[:, c]
    prev_states = torch.stack(prev_list, dim=1)  # (B, nchunks, H, D)

    # Inter-chunk contribution
    inter = prev_states[:, :, None, :, :] * cumA[..., None].exp()

    out = intra + inter  # (B, nchunks, C, H, D)
    out_full = out.reshape(B, T_padded, H, D)[:, :T]

    return out_full


def _chunked_scan_bwd(dout, s_chunks, cumA, chunk_total_decay, chunk_new_state,
                       prev_states, init_state, nchunks, chunk_size, T, T_padded, B, H, D):
    """
    Backward pass of chunked decay-masked scan.

    Mirrors the forward step-by-step in reverse. Each step is the textbook
    reverse of the corresponding forward op.

    Forward was:
      1. decay_mask[i,j] = exp(cumA[i]-cumA[j]) * (j<=i)   -- from cumA_flat
      2. intra_flat = decay_mask @ s_flat                     -- bmm
      3. chunk_new_state = intra[:, :, -1, :, :]              -- last-row extract
      4. state passing: prev_states[c+1] = exp(td[c]) * prev_states[c] + cns[c]
      5. inter = prev_states * exp(cumA)
      6. out = intra + inter
    """
    device = dout.device
    compute_dtype = torch.float32 if dout.dtype != torch.float64 else torch.float64

    # Pad dout if needed
    dout_f = dout.to(compute_dtype)
    if T_padded > T:
        dout_f = F.pad(dout_f, (0, 0, 0, 0, 0, T_padded - T))
    dout_chunks = dout_f.view(B, nchunks, chunk_size, H, D)

    # --- Step 6 reverse: out = intra + inter => d_intra = dout, d_inter = dout ---
    d_intra = dout_chunks  # (B, nchunks, C, H, D)
    d_inter = dout_chunks  # same, since out = intra + inter

    # --- Step 5 reverse: inter[n,c,i,h,d] = prev_states[n,c,h,d] * exp(cumA[n,c,i,h]) ---
    # d_prev_states[n,c,h,d] = sum_i d_inter[n,c,i,h,d] * exp(cumA[n,c,i,h])
    exp_cumA = cumA[..., None].exp()  # (B, nchunks, C, H, 1)
    d_prev_states = (d_inter * exp_cumA).sum(dim=2)  # (B, nchunks, H, D)
    # d_cumA from inter: d_cumA[n,c,i,h] = sum_d d_inter[n,c,i,h,d] * inter[n,c,i,h,d]
    inter_vals = prev_states[:, :, None, :, :] * exp_cumA  # (B, nchunks, C, H, D)
    d_cumA_from_inter = (d_inter * inter_vals).sum(dim=-1)  # (B, nchunks, C, H)

    # --- Step 4 reverse: state passing backward ---
    # Forward: prev_states[0] = init_state
    #          prev_states[c+1] = exp(td[c]) * prev_states[c] + cns[c]  (for c=0..nchunks-1)
    # (prev_states has nchunks entries: indices 0..nchunks-1)
    # The "next state" after chunk c is: exp(td[c]) * prev_states[c] + cns[c]
    # d_prev_states[c] comes from step 5. But prev_states[c] also feeds into
    # the state passing for chunk c (as the input state). So we propagate backward:
    d_chunk_new_state = []
    d_chunk_total_decay = []
    d_state = torch.zeros(B, H, D, device=device, dtype=compute_dtype)
    for c in range(nchunks - 1, -1, -1):
        # d_state is gradient flowing back from chunks c+1..nchunks-1
        # d_prev_states[c] is gradient from the inter term of chunk c
        # Total gradient w.r.t. prev_states[c]:
        #   - from inter of chunk c: d_prev_states[:, c]
        #   - from state passing: d_state (propagated from later chunks)
        # But prev_states[c] feeds into state_passing as:
        #   next_state = exp(td[c]) * prev_states[c] + cns[c]
        # so d_prev_states[c] += exp(td[c]) * d_next_state  (already in d_state)

        # Gradient w.r.t. cns[c]: d_state (since next = ... + cns[c])
        d_cns_c = d_state + d_prev_states[:, c] * 0  # d_state only for cns
        # Wait. Let me think again.
        #
        # After chunk c, the state becomes:
        #   state_after_c = exp(td[c]) * prev_states[c] + cns[c]
        # This state_after_c IS prev_states[c+1] (if c < nchunks-1).
        # Gradient flows into state_after_c from two sources:
        #   1. It's used as prev_states[c+1] in inter for chunk c+1 => d_prev_states[:, c+1]
        #   2. It's used as input to state passing for chunk c+1 => propagated as d_state
        # So the total d(state_after_c) = d_state (accumulated from reverse pass)
        #
        # From state_after_c = exp(td[c]) * prev_states[c] + cns[c]:
        #   d_cns[c] = d(state_after_c)
        #   d_td[c] = sum_d d(state_after_c)[d] * (exp(td[c]) * prev_states[c])[d]
        #   d(prev_states[c]) from this = exp(td[c]) * d(state_after_c)
        #
        # But prev_states[c] also has gradient from inter of chunk c: d_prev_states[:, c]
        # So total d(prev_states[c]) = exp(td[c]) * d(state_after_c) + d_prev_states[:, c]
        #
        # For the next iteration (c-1), d_state = total d(prev_states[c]) = the new accumulator

        decay_c = chunk_total_decay[:, c].exp()  # (B, H)
        d_cns_c = d_state  # d(state_after_c) flows to cns[c]
        d_td_c = (d_state * (decay_c[..., None] * prev_states[:, c])).sum(dim=-1)  # (B, H)
        d_state = decay_c[..., None] * d_state + d_prev_states[:, c]

        d_chunk_new_state.append(d_cns_c)
        d_chunk_total_decay.append(d_td_c)

    d_init_state = d_state

    # Reverse the lists (we appended nchunks-1, nchunks-2, ..., 0)
    d_chunk_new_state = torch.stack(d_chunk_new_state[::-1], dim=1)  # (B, nchunks, H, D)
    d_chunk_total_decay = torch.stack(d_chunk_total_decay[::-1], dim=1)  # (B, nchunks, H)

    # --- Step 3 reverse: chunk_new_state = intra[:, :, -1, :, :] ---
    # d_intra[:, :, -1, :, :] += d_chunk_new_state
    d_intra_full = d_intra.clone()
    d_intra_full[:, :, -1, :, :] = d_intra_full[:, :, -1, :, :] + d_chunk_new_state

    # --- Step 2 reverse: intra_flat = decay_mask @ s_flat ---
    # Flatten d_intra to (BNH, C, D)
    BNH = B * nchunks * H
    d_intra_flat = d_intra_full.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D)
    cumA_flat = cumA.permute(0, 1, 3, 2).reshape(BNH, chunk_size)
    s_flat = s_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D)

    # Rebuild decay mask
    causal = torch.tril(torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool))
    decay_diff = cumA_flat[:, :, None] - cumA_flat[:, None, :]
    decay_mask = torch.where(causal, decay_diff.exp(), torch.zeros_like(decay_diff))

    # d_s_flat = decay_mask^T @ d_intra_flat
    ds_flat = torch.bmm(decay_mask.transpose(1, 2), d_intra_flat)  # (BNH, C, D)

    # d_decay_mask = d_intra_flat @ s_flat^T  (BNH, C, C)
    d_decay_mask = torch.bmm(d_intra_flat, s_flat.transpose(1, 2))

    # --- Step 1 reverse: decay_mask[i,j] = exp(cumA[i] - cumA[j]) * causal[i,j] ---
    # d(cumA[i]) from decay_mask: sum_j d_decay_mask[i,j] * decay_mask[i,j]  (positive sign)
    # d(cumA[j]) from decay_mask: -sum_i d_decay_mask[i,j] * decay_mask[i,j] (negative sign)
    d_dm_times_dm = d_decay_mask * decay_mask  # (BNH, C, C)
    d_cumA_from_intra_flat = d_dm_times_dm.sum(dim=2) - d_dm_times_dm.sum(dim=1)  # (BNH, C)
    d_cumA_from_intra = d_cumA_from_intra_flat.view(B, nchunks, H, chunk_size).permute(0, 1, 3, 2)

    # --- Combine d_cumA contributions ---
    # chunk_total_decay = cumA[:, :, -1, :], so d_cumA[:, :, -1, :] += d_chunk_total_decay
    d_cumA = d_cumA_from_intra + d_cumA_from_inter
    # Use scatter_add or just index to avoid in-place on leaf
    d_cumA_td = torch.zeros_like(d_cumA)
    d_cumA_td[:, :, -1, :] = d_chunk_total_decay
    d_cumA = d_cumA + d_cumA_td

    # d_cumA -> d_log_alpha via reverse cumsum
    d_log_alpha = d_cumA.flip(dims=[2]).cumsum(dim=2).flip(dims=[2])  # (B, nchunks, C, H)

    # Reshape ds back
    ds = ds_flat.view(B, nchunks, H, chunk_size, D).permute(0, 1, 3, 2, 4)
    ds = ds.reshape(B, T_padded, H, D)[:, :T]

    d_log_alpha = d_log_alpha.reshape(B, T_padded, H)[:, :T]

    return ds, d_log_alpha, d_init_state


# ---------------------------------------------------------------------------
# Triton launcher wrappers
# ---------------------------------------------------------------------------

def _chunked_scan_fwd_triton(s_flat, log_alpha_flat, prev_states_flat, chunk_size, D, BNH,
                              store_cumA=False):
    """Launch Triton forward kernel with fused cumsum.
    
    Takes log_alpha (not cumA) — computes prefix sum inside the kernel.
    If store_cumA=True, also writes cumA to an output buffer (for backward).
    Returns out_flat (BNH, C, D), and optionally cumA_flat (BNH, C).
    """
    out_flat = torch.empty_like(s_flat)
    cumA_flat = torch.empty(BNH, chunk_size, device=s_flat.device, dtype=torch.float32) if store_cumA else s_flat.new_empty(0)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_C = triton.next_power_of_2(chunk_size)
    n_d_blocks = (D + BLOCK_D - 1) // BLOCK_D
    grid = (BNH, n_d_blocks)
    _chunk_scan_fwd_kernel[grid](
        s_flat, log_alpha_flat, prev_states_flat, out_flat, cumA_flat,
        chunk_size, D, BNH,
        s_flat.stride(0), s_flat.stride(1), s_flat.stride(2),
        log_alpha_flat.stride(0), log_alpha_flat.stride(1),
        prev_states_flat.stride(0), prev_states_flat.stride(1),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        cumA_flat.stride(0) if store_cumA else 0,
        cumA_flat.stride(1) if store_cumA else 0,
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D,
        STORE_CUMA=store_cumA,
    )
    if store_cumA:
        return out_flat, cumA_flat
    return out_flat


def _state_passing_fwd_triton(chunk_new_state, chunk_total_decay, init_state, B, nchunks, H, D):
    """Launch Triton state passing forward. Returns prev_states (B, nchunks, H, D)."""
    prev_states = torch.empty(B, nchunks, H, D, device=init_state.device, dtype=torch.float32)
    final_state = torch.empty(B, H, D, device=init_state.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    n_d_blocks = (D + BLOCK_D - 1) // BLOCK_D
    grid = (B, n_d_blocks, H)
    _state_passing_fwd_kernel[grid](
        chunk_new_state, chunk_total_decay, init_state, prev_states, final_state,
        D, nchunks,
        chunk_new_state.stride(0), chunk_new_state.stride(1), chunk_new_state.stride(2), chunk_new_state.stride(3),
        chunk_total_decay.stride(0), chunk_total_decay.stride(1), chunk_total_decay.stride(2),
        init_state.stride(0), init_state.stride(1), init_state.stride(2),
        prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3),
        final_state.stride(0), final_state.stride(1), final_state.stride(2),
        BLOCK_D=BLOCK_D,
    )
    return prev_states


def _chunked_scan_bwd_triton(dout_flat, log_alpha_flat, s_flat, prev_states_flat,
                              chunk_size, D, BNH):
    """
    Launch Triton backward kernels for intra-chunk.
    All kernels compute cumsum from log_alpha in-kernel (no aten::cumsum).
    Returns ds_flat (BNH, C, D), d_cumA_from_intra (BNH, C), d_prev_states (BNH, D).
    """
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_C = triton.next_power_of_2(chunk_size)
    n_d_blocks = (D + BLOCK_D - 1) // BLOCK_D

    ds_flat = torch.empty_like(s_flat)
    d_decay_mask = torch.zeros(BNH, chunk_size, chunk_size, device=s_flat.device, dtype=torch.float32)
    d_prev_states_flat = torch.empty(BNH, D, device=s_flat.device, dtype=torch.float32)

    grid = (BNH, n_d_blocks)
    _chunk_scan_bwd_kernel[grid](
        dout_flat, log_alpha_flat, s_flat, prev_states_flat,
        ds_flat, d_decay_mask, d_prev_states_flat,
        chunk_size, D,
        dout_flat.stride(0), dout_flat.stride(1), dout_flat.stride(2),
        log_alpha_flat.stride(0), log_alpha_flat.stride(1),
        s_flat.stride(0), s_flat.stride(1), s_flat.stride(2),
        prev_states_flat.stride(0), prev_states_flat.stride(1),
        ds_flat.stride(0), ds_flat.stride(1), ds_flat.stride(2),
        d_decay_mask.stride(0), d_decay_mask.stride(1), d_decay_mask.stride(2),
        d_prev_states_flat.stride(0), d_prev_states_flat.stride(1),
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D,
    )

    # d_cumA from decay mask: recomputes decay_mask from log_alpha inside kernel
    d_cumA_from_intra = torch.empty(BNH, chunk_size, device=s_flat.device, dtype=torch.float32)
    grid2 = (BNH,)
    _d_cumA_from_decay_mask_kernel[grid2](
        d_decay_mask, log_alpha_flat, d_cumA_from_intra,
        chunk_size,
        d_decay_mask.stride(0), d_decay_mask.stride(1), d_decay_mask.stride(2),
        log_alpha_flat.stride(0), log_alpha_flat.stride(1),
        d_cumA_from_intra.stride(0), d_cumA_from_intra.stride(1),
        BLOCK_C=BLOCK_C,
    )

    return ds_flat, d_cumA_from_intra, d_prev_states_flat


def _state_passing_bwd_triton(d_prev_states, prev_states, chunk_total_decay, B, nchunks, H, D):
    """Launch Triton state passing backward. Returns d_cns, d_td, d_init_state."""
    d_chunk_new_state = torch.empty(B, nchunks, H, D, device=prev_states.device, dtype=torch.float32)
    d_chunk_decay = torch.zeros(B, nchunks, H, device=prev_states.device, dtype=torch.float32)
    d_init_state = torch.empty(B, H, D, device=prev_states.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    n_d_blocks = (D + BLOCK_D - 1) // BLOCK_D
    grid = (B, n_d_blocks, H)
    _state_passing_bwd_kernel[grid](
        d_prev_states, prev_states, chunk_total_decay,
        d_chunk_new_state, d_chunk_decay, d_init_state,
        D, nchunks,
        d_prev_states.stride(0), d_prev_states.stride(1), d_prev_states.stride(2), d_prev_states.stride(3),
        prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3),
        chunk_total_decay.stride(0), chunk_total_decay.stride(1), chunk_total_decay.stride(2),
        d_chunk_new_state.stride(0), d_chunk_new_state.stride(1), d_chunk_new_state.stride(2), d_chunk_new_state.stride(3),
        d_chunk_decay.stride(0), d_chunk_decay.stride(1), d_chunk_decay.stride(2),
        d_init_state.stride(0), d_init_state.stride(1), d_init_state.stride(2),
        BLOCK_D=BLOCK_D,
    )
    return d_chunk_new_state, d_chunk_decay, d_init_state


def _inter_chunk_bwd_triton(dout_flat, cumA_flat, prev_states_flat, chunk_size, D, BNH):
    """Fused inter-chunk backward. Eliminates 3 large (BNH,C,D) intermediates.
    
    Returns d_prev_states_flat (BNH, D) and d_cumA_inter_flat (BNH, C).
    """
    d_prev_states = torch.empty(BNH, D, device=dout_flat.device, dtype=torch.float32)
    d_cumA_inter = torch.empty(BNH, chunk_size, device=dout_flat.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_C = triton.next_power_of_2(chunk_size)
    grid = (BNH,)
    _inter_chunk_bwd_kernel[grid](
        dout_flat, cumA_flat, prev_states_flat,
        d_prev_states, d_cumA_inter,
        chunk_size, D,
        dout_flat.stride(0), dout_flat.stride(1), dout_flat.stride(2),
        cumA_flat.stride(0), cumA_flat.stride(1),
        prev_states_flat.stride(0), prev_states_flat.stride(1),
        d_prev_states.stride(0), d_prev_states.stride(1),
        d_cumA_inter.stride(0), d_cumA_inter.stride(1),
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D,
    )
    return d_prev_states, d_cumA_inter


# ---------------------------------------------------------------------------
# Autograd Function with Triton (CUDA) / Python (CPU) backward
# ---------------------------------------------------------------------------

class ChunkedScanFn(torch.autograd.Function):
    """
    Custom autograd for chunked scan.
    On CUDA with Triton: uses Triton kernels for forward intra-chunk + backward.
    On CPU or without Triton: uses Python _chunked_scan_bwd.
    """

    @staticmethod
    def forward(ctx, kv, alpha, delta, epsilon, zeta, init_state, chunk_size):
        B, T, H, D = kv.shape
        device = kv.device
        use_triton = HAS_TRITON and device.type == 'cuda'

        # Tridiag conv
        s = _tridiag_conv(kv, zeta, epsilon, delta)
        alpha_clamped = alpha.clamp(min=1e-6)

        s_f = s.float()
        alpha_f = alpha_clamped.float()
        log_alpha = alpha_f.log()

        nchunks = (T + chunk_size - 1) // chunk_size
        T_padded = nchunks * chunk_size
        if T_padded > T:
            s_f = F.pad(s_f, (0, 0, 0, 0, 0, T_padded - T))
            log_alpha = F.pad(log_alpha, (0, 0, 0, T_padded - T))

        s_chunks = s_f.view(B, nchunks, chunk_size, H, D)
        la_chunks = log_alpha.view(B, nchunks, chunk_size, H)

        BNH = B * nchunks * H
        la_flat = la_chunks.permute(0, 1, 3, 2).reshape(BNH, chunk_size).contiguous()
        s_flat = s_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D).contiguous()

        if use_triton:
            # Intra-chunk via Triton with fused cumsum
            # First call: store cumA (needed for chunk_total_decay + state passing + backward)
            zero_prev = torch.zeros(BNH, D, device=device, dtype=torch.float32)
            intra_flat, cumA_flat = _chunked_scan_fwd_triton(
                s_flat, la_flat, zero_prev, chunk_size, D, BNH, store_cumA=True)
            cumA = cumA_flat.view(B, nchunks, H, chunk_size).permute(0, 1, 3, 2)
            chunk_total_decay = cumA[:, :, -1, :]
        else:
            cumA = la_chunks.cumsum(dim=2)
            chunk_total_decay = cumA[:, :, -1, :]
            cumA_flat = cumA.permute(0, 1, 3, 2).reshape(BNH, chunk_size).contiguous()
            causal = torch.tril(torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool))
            decay_diff = cumA_flat[:, :, None] - cumA_flat[:, None, :]
            decay_mask = torch.where(causal, decay_diff.exp(), torch.zeros_like(decay_diff))
            intra_flat = torch.bmm(decay_mask, s_flat)

        intra = intra_flat.view(B, nchunks, H, chunk_size, D).permute(0, 1, 3, 2, 4)
        chunk_new_state = intra[:, :, -1, :, :].contiguous()

        # State passing
        if use_triton:
            prev_states = _state_passing_fwd_triton(
                chunk_new_state, chunk_total_decay, init_state.float(), B, nchunks, H, D)
        else:
            prev_states = torch.zeros(B, nchunks, H, D, device=device, dtype=torch.float32)
            state = init_state.float()
            for c in range(nchunks):
                prev_states[:, c] = state
                state = chunk_total_decay[:, c].exp()[..., None] * state + chunk_new_state[:, c]

        if use_triton:
            # Re-run fwd kernel with real prev_states (no need to store cumA again)
            prev_states_flat = prev_states.reshape(BNH, D).contiguous()
            out_flat = _chunked_scan_fwd_triton(s_flat, la_flat, prev_states_flat, chunk_size, D, BNH)
            out = out_flat.view(B, nchunks, H, chunk_size, D).permute(0, 1, 3, 2, 4)
        else:
            inter = prev_states[:, :, None, :, :] * cumA[..., None].exp()
            out = intra + inter

        out_full = out.reshape(B, T_padded, H, D)[:, :T]

        # Save cumA (from Triton fwd kernel) so backward doesn't need aten::cumsum
        ctx.save_for_backward(kv, alpha_clamped, delta, epsilon, zeta, init_state,
                              s_chunks, la_chunks, cumA, chunk_total_decay, chunk_new_state, prev_states)
        ctx.chunk_size = chunk_size
        ctx.nchunks = nchunks
        ctx.T_padded = T_padded
        ctx.T = T
        ctx.B = B
        ctx.H = H
        ctx.D = D
        ctx.use_triton = use_triton

        return out_full.to(kv.dtype)

    @staticmethod
    def backward(ctx, dout):
        kv, alpha, delta, epsilon, zeta, init_state, \
            s_chunks, la_chunks, cumA, chunk_total_decay, chunk_new_state, prev_states = ctx.saved_tensors
        B, H, D = ctx.B, ctx.H, ctx.D
        T, T_padded = ctx.T, ctx.T_padded
        nchunks, chunk_size = ctx.nchunks, ctx.chunk_size
        use_triton = ctx.use_triton

        if not use_triton:
            ds, d_log_alpha, d_init_state = _chunked_scan_bwd(
                dout, s_chunks, cumA, chunk_total_decay, chunk_new_state,
                prev_states, init_state, nchunks, chunk_size,
                T, T_padded, B, H, D)
        else:
            # === Triton backward (cumsum fused into kernels) ===
            device = dout.device
            BNH = B * nchunks * H

            # cumA saved from forward — no aten::cumsum needed

            # Pad dout
            dout_f = dout.float()
            if T_padded > T:
                dout_f = F.pad(dout_f, (0, 0, 0, 0, 0, T_padded - T))
            dout_chunks = dout_f.view(B, nchunks, chunk_size, H, D)

            # Step 6+5 reverse: fused inter-chunk backward (eliminates 3 large intermediates)
            # Flatten to (BNH, C, D) / (BNH, C) / (BNH, D) for Triton kernel
            dout_flat = dout_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D).contiguous()
            cumA_flat = cumA.permute(0, 1, 3, 2).reshape(BNH, chunk_size).contiguous()
            prev_states_flat = prev_states.reshape(BNH, D).contiguous()

            d_prev_flat, d_cumA_inter_flat = _inter_chunk_bwd_triton(
                dout_flat, cumA_flat, prev_states_flat, chunk_size, D, BNH)
            d_prev_states_from_inter = d_prev_flat.view(B, nchunks, H, D)
            d_cumA_from_inter = d_cumA_inter_flat.view(B, nchunks, H, chunk_size).permute(0, 1, 3, 2)

            # Step 4 reverse: state passing backward
            d_chunk_new_state, d_chunk_total_decay, d_init_state = _state_passing_bwd_triton(
                d_prev_states_from_inter, prev_states, chunk_total_decay, B, nchunks, H, D)

            # Step 3 reverse: d_intra[:, :, -1] += d_chunk_new_state
            d_intra = dout_chunks.clone()
            d_intra[:, :, -1, :, :] = d_intra[:, :, -1, :, :] + d_chunk_new_state

            # Step 2 reverse: Triton bwd kernels (cumsum fused — pass log_alpha, not cumA)
            d_intra_flat = d_intra.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D).contiguous()
            la_flat = la_chunks.permute(0, 1, 3, 2).reshape(BNH, chunk_size).contiguous()
            s_flat = s_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D).contiguous()
            # prev_states_flat already computed above for inter-chunk bwd

            ds_flat, d_cumA_from_intra_flat, _ = _chunked_scan_bwd_triton(
                d_intra_flat, la_flat, s_flat, prev_states_flat,
                chunk_size, D, BNH)

            # Step 1 reverse: combine d_cumA
            d_cumA_from_intra = d_cumA_from_intra_flat.view(B, nchunks, H, chunk_size).permute(0, 1, 3, 2)
            d_cumA = d_cumA_from_intra + d_cumA_from_inter
            d_cumA_td = torch.zeros_like(d_cumA)
            d_cumA_td[:, :, -1, :] = d_chunk_total_decay
            d_cumA = d_cumA + d_cumA_td

            # d_cumA -> d_log_alpha
            d_log_alpha = d_cumA.flip(dims=[2]).cumsum(dim=2).flip(dims=[2])

            ds = ds_flat.view(B, nchunks, H, chunk_size, D).permute(0, 1, 3, 2, 4)
            ds = ds.reshape(B, T_padded, H, D)[:, :T]
            d_log_alpha = d_log_alpha.reshape(B, T_padded, H)[:, :T]

        # d_alpha from d_log_alpha
        d_alpha = d_log_alpha / alpha

        # Tridiag conv backward
        d_kv, d_zeta, d_epsilon, d_delta = _tridiag_conv_backward(ds, kv, zeta, epsilon, delta)

        return d_kv, d_alpha, d_delta, d_epsilon, d_zeta, d_init_state, None


def _forward_scan_chunked_autograd(kv, alpha, delta, epsilon, zeta, init_state, chunk_size):
    """
    Chunked scan with autograd-computed backward (CPU path).
    Tridiag conv + chunked decay-masked matmul, all differentiable.
    """
    s = _tridiag_conv(kv, zeta, epsilon, delta)
    alpha_clamped = alpha.clamp(min=1e-6)
    out = _chunked_scan_fwd(s, alpha_clamped, init_state, chunk_size)
    return out.to(kv.dtype)


def forward_scan_chunked(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    epsilon: torch.Tensor,
    zeta: torch.Tensor,
    init_state: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunked decay-masked scan — primary fast path.

    Decomposes the recurrence:
        state_t = alpha_t * state_{t-1} + zeta_t * kv_t + epsilon_t * kv_{t-1} + delta_t * kv_{t-2}
    into:
        1. Tridiagonal conv (vectorized)
        2. Decay-masked matmul within chunks
        3. Inter-chunk state carry

    On CPU: autograd handles backward through differentiable PyTorch ops.
    On CUDA: ChunkedScanFn with custom backward via torch.library.custom_op
             (compile-friendly — Dynamo sees it as an opaque leaf, no graph break).
    """
    if kv.is_cuda:
        # Use torch.library.custom_op path — compile-friendly (no graph break)
        out, *_ = torch.ops.s6.chunked_scan_fwd(kv, alpha, delta, epsilon, zeta, init_state, chunk_size)
        return out
    return _forward_scan_chunked_autograd(kv, alpha, delta, epsilon, zeta, init_state, chunk_size)


# ---------------------------------------------------------------------------
# torch.library.custom_op registration for torch.compile compatibility
# ---------------------------------------------------------------------------
# Registers ChunkedScanFn forward/backward as opaque custom ops so that
# torch.compile (Dynamo) doesn't graph-break on the scan. Everything before
# and after the scan gets compiled by Inductor into fused kernels.
#
# Pattern from modded-nanogpt: @torch.library.custom_op + register_fake +
# register_autograd.
# ---------------------------------------------------------------------------

@torch.library.custom_op("s6::chunked_scan_fwd", mutates_args=())
def _chunked_scan_fwd_op(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    epsilon: torch.Tensor,
    zeta: torch.Tensor,
    init_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass of chunked scan, returning output + saved tensors for backward.
    
    Triton path: cumsum computed inside kernels (no aten::cumsum hotspot).
    """
    B, T, H, D = kv.shape
    device = kv.device
    use_triton = HAS_TRITON and device.type == 'cuda'

    s = _tridiag_conv(kv, zeta, epsilon, delta)
    alpha_clamped = alpha.clamp(min=1e-6)
    s_f = s.float()
    log_alpha = alpha_clamped.float().log()

    nchunks = (T + chunk_size - 1) // chunk_size
    T_padded = nchunks * chunk_size
    if T_padded > T:
        s_f = F.pad(s_f, (0, 0, 0, 0, 0, T_padded - T))
        log_alpha = F.pad(log_alpha, (0, 0, 0, T_padded - T))

    s_chunks = s_f.view(B, nchunks, chunk_size, H, D)
    la_chunks = log_alpha.view(B, nchunks, chunk_size, H)

    BNH = B * nchunks * H
    la_flat = la_chunks.permute(0, 1, 3, 2).reshape(BNH, chunk_size).contiguous()
    s_flat = s_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D).contiguous()

    if use_triton:
        # Fused cumsum inside Triton kernel
        zero_prev = torch.zeros(BNH, D, device=device, dtype=torch.float32)
        intra_flat, cumA_flat = _chunked_scan_fwd_triton(
            s_flat, la_flat, zero_prev, chunk_size, D, BNH, store_cumA=True)
        cumA = cumA_flat.view(B, nchunks, H, chunk_size).permute(0, 1, 3, 2)
        chunk_total_decay = cumA[:, :, -1, :]
    else:
        cumA = la_chunks.cumsum(dim=2)
        chunk_total_decay = cumA[:, :, -1, :]
        cumA_flat = cumA.permute(0, 1, 3, 2).reshape(BNH, chunk_size).contiguous()
        causal = torch.tril(torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool))
        decay_diff = cumA_flat[:, :, None] - cumA_flat[:, None, :]
        decay_mask = torch.where(causal, decay_diff.exp(), torch.zeros_like(decay_diff))
        intra_flat = torch.bmm(decay_mask, s_flat)

    intra = intra_flat.view(B, nchunks, H, chunk_size, D).permute(0, 1, 3, 2, 4)
    chunk_new_state = intra[:, :, -1, :, :].contiguous()

    if use_triton:
        prev_states = _state_passing_fwd_triton(
            chunk_new_state, chunk_total_decay, init_state.float(), B, nchunks, H, D)
    else:
        prev_states = torch.zeros(B, nchunks, H, D, device=device, dtype=torch.float32)
        state = init_state.float()
        for c in range(nchunks):
            prev_states[:, c] = state
            state = chunk_total_decay[:, c].exp()[..., None] * state + chunk_new_state[:, c]

    if use_triton:
        prev_states_flat = prev_states.reshape(BNH, D).contiguous()
        out_flat = _chunked_scan_fwd_triton(s_flat, la_flat, prev_states_flat, chunk_size, D, BNH)
        out = out_flat.view(B, nchunks, H, chunk_size, D).permute(0, 1, 3, 2, 4)
    else:
        inter = prev_states[:, :, None, :, :] * cumA[..., None].exp()
        out = intra + inter

    out_full = out.reshape(B, T_padded, H, D)[:, :T].to(kv.dtype)

    # Save cumA so backward doesn't need aten::cumsum (was 18% of CUDA time)
    # clone alpha_clamped and init_state to avoid aliasing inputs (custom_op requirement)
    # chunk_total_decay is cumA[:,:,-1,:] — derive it in backward from cumA to avoid alias
    return (out_full, alpha_clamped.clone(), s_chunks, la_chunks, cumA.contiguous(),
            chunk_new_state, prev_states, init_state.detach().clone())


@_chunked_scan_fwd_op.register_fake
def _chunked_scan_fwd_fake(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    epsilon: torch.Tensor,
    zeta: torch.Tensor,
    init_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape inference for Dynamo tracing."""
    B, T, H, D = kv.shape
    nchunks = (T + chunk_size - 1) // chunk_size

    out = kv.new_empty(B, T, H, D)
    alpha_clamped = alpha.new_empty(B, T, H)
    s_chunks = kv.new_empty(B, nchunks, chunk_size, H, D, dtype=torch.float32)
    la_chunks = alpha.new_empty(B, nchunks, chunk_size, H, dtype=torch.float32)
    cumA = alpha.new_empty(B, nchunks, chunk_size, H, dtype=torch.float32)
    chunk_new_state = kv.new_empty(B, nchunks, H, D, dtype=torch.float32)
    prev_states = kv.new_empty(B, nchunks, H, D, dtype=torch.float32)
    init_state_saved = init_state.new_empty(*init_state.shape)

    return (out, alpha_clamped, s_chunks, la_chunks, cumA,
            chunk_new_state, prev_states, init_state_saved)


@torch.library.custom_op("s6::chunked_scan_bwd", mutates_args=())
def _chunked_scan_bwd_op(
    dout: torch.Tensor,
    kv: torch.Tensor,
    alpha_clamped: torch.Tensor,
    delta: torch.Tensor,
    epsilon: torch.Tensor,
    zeta: torch.Tensor,
    init_state: torch.Tensor,
    s_chunks: torch.Tensor,
    la_chunks: torch.Tensor,
    cumA: torch.Tensor,
    chunk_new_state: torch.Tensor,
    prev_states: torch.Tensor,
    chunk_size: int,
    T_orig: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of chunked scan. cumA passed from forward (no aten::cumsum)."""
    B, T, H, D = kv.shape
    device = dout.device
    use_triton = HAS_TRITON and device.type == 'cuda'
    nchunks = s_chunks.shape[1]
    T_padded = nchunks * chunk_size

    # cumA passed from forward — derive chunk_total_decay
    chunk_total_decay = cumA[:, :, -1, :]

    if not use_triton:
        ds, d_log_alpha, d_init_state = _chunked_scan_bwd(
            dout, s_chunks, cumA, chunk_total_decay, chunk_new_state,
            prev_states, init_state, nchunks, chunk_size,
            T, T_padded, B, H, D)
    else:
        BNH = B * nchunks * H
        dout_f = dout.float()
        if T_padded > T:
            dout_f = F.pad(dout_f, (0, 0, 0, 0, 0, T_padded - T))
        dout_chunks = dout_f.view(B, nchunks, chunk_size, H, D)

        # Fused inter-chunk backward (eliminates 3 large intermediates)
        dout_flat = dout_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D).contiguous()
        cumA_flat = cumA.permute(0, 1, 3, 2).reshape(BNH, chunk_size).contiguous()
        prev_states_flat = prev_states.reshape(BNH, D).contiguous()

        d_prev_flat, d_cumA_inter_flat = _inter_chunk_bwd_triton(
            dout_flat, cumA_flat, prev_states_flat, chunk_size, D, BNH)
        d_prev_states_from_inter = d_prev_flat.view(B, nchunks, H, D)
        d_cumA_from_inter = d_cumA_inter_flat.view(B, nchunks, H, chunk_size).permute(0, 1, 3, 2)

        d_chunk_new_state, d_chunk_total_decay, d_init_state = _state_passing_bwd_triton(
            d_prev_states_from_inter, prev_states, chunk_total_decay, B, nchunks, H, D)

        d_intra = dout_chunks.clone()
        d_intra[:, :, -1, :, :] = d_intra[:, :, -1, :, :] + d_chunk_new_state

        # Triton bwd kernels: pass log_alpha (cumsum fused inside)
        d_intra_flat = d_intra.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D).contiguous()
        la_flat = la_chunks.permute(0, 1, 3, 2).reshape(BNH, chunk_size).contiguous()
        s_flat = s_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, chunk_size, D).contiguous()
        # prev_states_flat already computed above for inter-chunk bwd

        ds_flat, d_cumA_from_intra_flat, _ = _chunked_scan_bwd_triton(
            d_intra_flat, la_flat, s_flat, prev_states_flat,
            chunk_size, D, BNH)

        d_cumA_from_intra = d_cumA_from_intra_flat.view(B, nchunks, H, chunk_size).permute(0, 1, 3, 2)
        d_cumA = d_cumA_from_intra + d_cumA_from_inter
        d_cumA_td = torch.zeros_like(d_cumA)
        d_cumA_td[:, :, -1, :] = d_chunk_total_decay
        d_cumA = d_cumA + d_cumA_td

        d_log_alpha = d_cumA.flip(dims=[2]).cumsum(dim=2).flip(dims=[2])

        ds = ds_flat.view(B, nchunks, H, chunk_size, D).permute(0, 1, 3, 2, 4)
        ds = ds.reshape(B, T_padded, H, D)[:, :T]
        d_log_alpha = d_log_alpha.reshape(B, T_padded, H)[:, :T]

    d_alpha = d_log_alpha / alpha_clamped
    d_kv, d_zeta, d_epsilon, d_delta = _tridiag_conv_backward(ds, kv, zeta, epsilon, delta)

    return d_kv, d_alpha, d_delta, d_epsilon, d_zeta, d_init_state


@_chunked_scan_bwd_op.register_fake
def _chunked_scan_bwd_fake(
    dout, kv, alpha_clamped, delta, epsilon, zeta, init_state,
    s_chunks, la_chunks, cumA, chunk_new_state, prev_states,
    chunk_size, T_orig,
):
    B, T, H, D = kv.shape
    d_kv = kv.new_empty(B, T, H, D)
    d_alpha = alpha_clamped.new_empty(B, T, H)
    d_delta = delta.new_empty(B, T, H)
    d_epsilon = epsilon.new_empty(B, T, H)
    d_zeta = zeta.new_empty(B, T, H)
    d_init_state = init_state.new_empty(*init_state.shape)
    return d_kv, d_alpha, d_delta, d_epsilon, d_zeta, d_init_state


def _chunked_scan_autograd_backward(ctx, grad_out, *_saved_grads):
    kv, alpha_clamped, delta, epsilon, zeta, init_state = ctx.saved_tensors[:6]
    s_chunks, la_chunks, cumA, chunk_new_state, prev_states = ctx.saved_tensors[6:]
    chunk_size = ctx.chunk_size
    T_orig = ctx.T_orig
    d_kv, d_alpha, d_delta, d_epsilon, d_zeta, d_init_state = torch.ops.s6.chunked_scan_bwd(
        grad_out, kv, alpha_clamped, delta, epsilon, zeta, init_state,
        s_chunks, la_chunks, cumA, chunk_new_state, prev_states,
        chunk_size, T_orig,
    )
    return d_kv, d_alpha, d_delta, d_epsilon, d_zeta, d_init_state, None


def _chunked_scan_setup_context(ctx, inputs, output):
    kv, alpha, delta, epsilon, zeta, init_state, chunk_size = inputs
    out, alpha_clamped, s_chunks, la_chunks, cumA, chunk_new_state, prev_states, init_saved = output
    ctx.save_for_backward(kv, alpha_clamped, delta, epsilon, zeta, init_saved,
                          s_chunks, la_chunks, cumA, chunk_new_state, prev_states)
    ctx.chunk_size = chunk_size
    ctx.T_orig = kv.shape[1]


_chunked_scan_fwd_op.register_autograd(
    _chunked_scan_autograd_backward,
    setup_context=_chunked_scan_setup_context,
)


# ---------------------------------------------------------------------------
# Streaming reference (kept for correctness testing — DO NOT USE IN TRAINING)
# ---------------------------------------------------------------------------

def forward_scan_elementwise_streaming(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    epsilon: torch.Tensor,
    zeta: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Memory-safe elementwise streaming recurrence (REFERENCE ONLY — very slow).
    """
    b, t, h, d = kv.shape
    out_dtype = kv.dtype

    kvf = kv.float()
    af = torch.clamp(alpha.float(), min=1e-6)
    df = delta.float()
    ef = epsilon.float()
    zf = zeta.float()

    state = init_state.float()
    kv_tm1 = torch.zeros_like(state)
    kv_tm2 = torch.zeros_like(state)
    outs = []

    for i in range(t):
        kv_t = kvf[:, i]
        s_t = zf[:, i][..., None] * kv_t + ef[:, i][..., None] * kv_tm1 + df[:, i][..., None] * kv_tm2
        state = af[:, i][..., None] * state + s_t
        outs.append(state)
        kv_tm2 = kv_tm1
        kv_tm1 = kv_t

    out = torch.stack(outs, dim=1)
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# Legacy scan implementations (kept for reference)
# ---------------------------------------------------------------------------

def forward_scan_outer_readout_streaming(k, v, alpha, delta, epsilon, zeta, gate, init_state, scale):
    """Memory-safe outer-state scan with readout injection."""
    b, t, h, d = k.shape
    out_dtype = k.dtype
    kf, vf = k.float(), v.float()
    af = torch.clamp(alpha.float(), min=1e-6)
    df, ef, zf, gf = delta.float(), epsilon.float(), zeta.float(), gate.float()
    state = init_state.float()
    kv_tm1 = torch.zeros_like(state)
    kv_tm2 = torch.zeros_like(state)
    outs = []
    for i in range(t):
        k_t, v_t = kf[:, i], vf[:, i]
        kv_t = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
        s_t = zf[:, i][..., None, None] * kv_t + ef[:, i][..., None, None] * kv_tm1 + df[:, i][..., None, None] * kv_tm2
        state = af[:, i][..., None, None] * state + s_t
        read_t = torch.einsum("bhjk,bhj->bhk", state, k_t) * scale
        outs.append(v_t + gf[:, i] * read_t)
        kv_tm2, kv_tm1 = kv_tm1, kv_t
    return torch.stack(outs, dim=1).to(out_dtype)


def forward_scan(kv, alpha, delta, epsilon, zeta, init_state):
    """Forward causal scan — mask-form vectorized."""
    outer_mode = kv.ndim == 5
    if outer_mode:
        batch, seq_len, nheads, headdim, _ = kv.shape
        coef_shape = (batch, seq_len, nheads, 1, 1)
    else:
        batch, seq_len, nheads, headdim = kv.shape
        coef_shape = (batch, seq_len, nheads, 1)
    kv_tm1 = torch.cat([torch.zeros_like(kv[:, :1]), kv[:, :-1]], dim=1)
    kv_tm2 = torch.cat([torch.zeros_like(kv[:, :2]), kv[:, :-2]], dim=1)
    s = zeta.view(coef_shape) * kv + epsilon.view(coef_shape) * kv_tm1 + delta.view(coef_shape) * kv_tm2
    out_dtype = kv.dtype
    alpha_safe = torch.clamp(alpha.float(), min=1e-6)
    p = torch.clamp(torch.cumprod(alpha_safe, dim=1), min=1e-6)
    p_e = p[..., None, None] if outer_mode else p[..., None]
    s_f = s.float()
    state = p_e * torch.cumsum(s_f / p_e, dim=1) + p_e * init_state.float().unsqueeze(1)
    return state.to(out_dtype)


def backward_scan(kv, alpha, delta, epsilon, zeta, init_state):
    """Backward causal scan — mask-form vectorized."""
    outer_mode = kv.ndim == 5
    if outer_mode:
        batch, seq_len, nheads, headdim, _ = kv.shape
        coef_shape = (batch, seq_len, nheads, 1, 1)
    else:
        batch, seq_len, nheads, headdim = kv.shape
        coef_shape = (batch, seq_len, nheads, 1)
    kv_tp1 = torch.cat([kv[:, 1:], torch.zeros_like(kv[:, :1])], dim=1)
    kv_tp2 = torch.cat([kv[:, 2:], torch.zeros_like(kv[:, :2])], dim=1)
    s = zeta.view(coef_shape) * kv + epsilon.view(coef_shape) * kv_tp1 + delta.view(coef_shape) * kv_tp2
    out_dtype = kv.dtype
    alpha_rev = torch.flip(alpha.float(), dims=[1])
    s_rev = torch.flip(s, dims=[1])
    alpha_safe = torch.clamp(alpha_rev, min=1e-8)
    p = torch.clamp(torch.cumprod(alpha_safe, dim=1), min=1e-6)
    p_e = p[..., None, None] if outer_mode else p[..., None]
    s_rev_f = s_rev.float()
    state_rev = p_e * torch.cumsum(s_rev_f / p_e, dim=1) + p_e * init_state.float().unsqueeze(1)
    return torch.flip(state_rev, dims=[1]).to(out_dtype)


def centered_scan(kv, alpha, delta, epsilon, zeta, init_state):
    """Centered scan = average of forward and backward."""
    return 0.5 * (forward_scan(kv, alpha, delta, epsilon, zeta, init_state)
                  + backward_scan(kv, alpha, delta, epsilon, zeta, init_state))
