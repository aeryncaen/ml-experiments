"""
Standalone test: S4D kernel conv over L + trapezoidal scan over N.

Three implementations compared:
1. s4d_standard     — vanilla S4D (diagonal, kernel summed over n, one FFT conv)
2. s4d_perstate     — S4D with per-state FFT conv (no n-sum in kernel), then sum
                      Should be numerically identical to s4d_standard.
3. s4d_nscan        — S4D per-state FFT conv, then bidirectional scan across N, then sum
                      This couples state dimensions.
"""

import torch
import torch.nn.functional as F
import math


def make_s4d_params(H: int, N: int, dt_min=0.001, dt_max=0.1):
    """Initialize S4D parameters.
    H: number of features
    N: state dim (must be even, we store N//2 complex)
    Returns: A (H, N//2) complex, B (H, N//2) complex, C (H, N//2) complex,
             dt (H,) real, D (H,) real
    """
    Nhalf = N // 2
    # A: real part = -0.5, imag part = pi * [0, 1, ..., N//2-1]
    A_real = torch.full((H, Nhalf), -0.5)
    A_imag = -math.pi * torch.arange(Nhalf).float().unsqueeze(0).expand(H, -1)
    A = torch.complex(A_real, A_imag)  # (H, N//2)

    # B: all ones (constant init)
    B = torch.ones(H, Nhalf, dtype=torch.cfloat)

    # C: random complex
    C = torch.randn(H, Nhalf, dtype=torch.cfloat)

    # dt: log-uniform in [dt_min, dt_max]
    log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    dt = torch.exp(log_dt)

    # D: skip connection
    D = torch.randn(H)

    return A, B, C, dt, D


def s4d_standard(u, A, B, C, dt, D):
    """Standard S4D: sum kernel over n, single FFT conv.
    u: (B, H, L)
    Returns: (B, H, L)
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]

    dtA = dt.unsqueeze(-1) * A  # (H, N//2)
    # ZOH discretization correction
    CB = C * B * (torch.exp(dtA) - 1.0) / A  # (H, N//2)

    # Vandermonde: K[l] = 2*Re(Σ_n CB_n * exp(dtA_n * l))
    arange_L = torch.arange(L, device=u.device, dtype=torch.float32)
    # (H, N//2, L)
    K_complex = CB.unsqueeze(-1) * torch.exp(dtA.unsqueeze(-1) * arange_L)
    K = 2.0 * K_complex.sum(dim=1).real  # (H, L) — sum over n

    # FFT conv
    fft_len = L + L  # safe padding
    K_f = torch.fft.rfft(K, n=fft_len)        # (H, fft_len//2+1)
    u_f = torch.fft.rfft(u, n=fft_len)        # (B, H, fft_len//2+1)
    y = torch.fft.irfft(u_f * K_f, n=fft_len)[..., :L]  # (B, H, L)

    # skip connection
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def s4d_perstate(u, A, B, C, dt, D):
    """S4D with per-state FFT conv, then sum over n.
    Should be numerically identical to s4d_standard.
    u: (B, H, L)
    Returns: (B, H, L)
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]

    dtA = dt.unsqueeze(-1) * A  # (H, N//2)
    CB = C * B * (torch.exp(dtA) - 1.0) / A  # (H, N//2)

    # Per-state kernel: k_n[l] = CB_n * exp(dtA_n * l)  (complex)
    arange_L = torch.arange(L, device=u.device, dtype=torch.float32)
    K_per = CB.unsqueeze(-1) * torch.exp(dtA.unsqueeze(-1) * arange_L)  # (H, N//2, L) complex

    # Complex FFT conv: use fft (not rfft) since kernel is complex
    fft_len = L + L
    K_f = torch.fft.fft(K_per, n=fft_len)         # (H, N//2, fft_len) complex
    u_f = torch.fft.fft(u.to(torch.cfloat), n=fft_len)  # (B, H, fft_len) complex
    s_f = u_f.unsqueeze(2) * K_f.unsqueeze(0)     # (B, H, N//2, fft_len)
    s = torch.fft.ifft(s_f, n=fft_len)[..., :L]   # (B, H, N//2, L) complex

    # Sum over n with conjugate symmetry: 2*Re
    y = 2.0 * s.sum(dim=2).real  # (B, H, L)

    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def _scan_fwd_n(gates, values):
    """Sequential scan across last dim (N): h_n = gate_n * h_{n-1} + value_n.
    gates:  (..., N)
    values: (..., N)
    Returns: (..., N)
    """
    N = values.shape[-1]
    h = torch.zeros_like(values[..., 0])
    out = []
    for n in range(N):
        h = gates[..., n] * h + values[..., n]
        out.append(h)
    return torch.stack(out, dim=-1)


def _bidi_scan(gates, values):
    """Bidirectional scan across last dim."""
    fwd = _scan_fwd_n(gates, values)
    bwd = _scan_fwd_n(gates.flip(-1), values.flip(-1)).flip(-1)
    return fwd + bwd


def s4d_nscan(u, A, B, C, dt, D, scan_gates=None):
    """S4D per-state FFT conv + bidirectional trapezoidal scan across N.
    u: (B, H, L)
    scan_gates: (H, N//2) real — decay gates for the N-scan. If None, uses |exp(dtA)|.
    Returns: (B, H, L)
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]

    dtA = dt.unsqueeze(-1) * A  # (H, N//2)
    CB = C * B * (torch.exp(dtA) - 1.0) / A  # (H, N//2)

    # Per-state kernel
    arange_L = torch.arange(L, device=u.device, dtype=torch.float32)
    K_per = CB.unsqueeze(-1) * torch.exp(dtA.unsqueeze(-1) * arange_L)  # (H, N//2, L)

    # Batch FFT conv over n (complex kernel, complex FFT)
    fft_len = L + L
    K_f = torch.fft.fft(K_per, n=fft_len)
    u_f = torch.fft.fft(u.to(torch.cfloat), n=fft_len)
    s_f = u_f.unsqueeze(2) * K_f.unsqueeze(0)
    s = torch.fft.ifft(s_f, n=fft_len)[..., :L]  # (B, H, N//2, L) complex

    # Take 2*real before scan (conjugate symmetry)
    s_real = 2.0 * s.real  # (B, H, N//2, L)

    # Bidirectional trapezoidal scan across N (dim=2 here, which is N//2)
    # gates: (H, N//2) -> (1, H, N//2, 1) for broadcasting
    if scan_gates is None:
        scan_gates = torch.exp(dtA).abs()  # (H, N//2)
    g = scan_gates.unsqueeze(0).unsqueeze(-1)  # (1, H, N//2, 1)
    g = g.expand_as(s_real)  # (B, H, N//2, L)

    # scan across dim=2 (N//2) — need last dim to be scan dim
    # transpose so N is last: (B, H, L, N//2)
    s_t = s_real.permute(0, 1, 3, 2)
    g_t = g.permute(0, 1, 3, 2)
    s_scanned = _bidi_scan(g_t, s_t)  # (B, H, L, N//2)

    # Sum over n
    y = s_scanned.sum(dim=-1)  # (B, H, L)

    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def s4d_sequential_ref(u, A, B, C, dt, D):
    """Sequential reference: step-by-step recurrence. Gold standard.
    u: (B, H, L)
    Returns: (B, H, L)
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]

    dtA = dt.unsqueeze(-1) * A  # (H, N//2)
    dA = torch.exp(dtA)         # discretized A
    dB = B * (dA - 1.0) / A    # discretized B (ZOH)

    x = torch.zeros(Batch, H, Nhalf, dtype=torch.cfloat)  # state
    ys = []
    for t in range(L):
        x = dA * x + dB * u[:, :, t:t+1]  # (B, H, N//2)
        y_t = 2.0 * (C * x).sum(dim=-1).real  # (B, H)
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)  # (B, H, L)
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def s4d_nscan_sequential_ref(u, A, B, C, dt, D, scan_gates=None):
    """Sequential reference for S4D + N-scan.
    Step-by-step recurrence over L, then sequential scan over N, then sum.
    Gold standard for verifying s4d_nscan.
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]

    dtA = dt.unsqueeze(-1) * A
    dA = torch.exp(dtA)
    dB = B * (dA - 1.0) / A

    # Step 1: sequential recurrence over L, per state dim
    x = torch.zeros(Batch, H, Nhalf, dtype=torch.cfloat)
    s_all = []  # will be (L, B, H, N//2)
    for t in range(L):
        x = dA * x + dB * u[:, :, t:t+1]
        s_all.append(x.clone())
    # s_all: (B, H, N//2, L) complex
    s_complex = torch.stack(s_all, dim=-1)

    # Step 2: take 2*Re (conjugate symmetry)
    s_real = 2.0 * (C.unsqueeze(0).unsqueeze(-1) * s_complex).real  # (B, H, N//2, L)

    # Step 3: bidirectional scan across N
    if scan_gates is None:
        scan_gates = torch.exp(dtA).abs()
    g = scan_gates.unsqueeze(0).unsqueeze(-1).expand_as(s_real)  # (B, H, N//2, L)

    # transpose so N is last for scan: (B, H, L, N//2)
    s_t = s_real.permute(0, 1, 3, 2)
    g_t = g.permute(0, 1, 3, 2)
    s_scanned = _bidi_scan(g_t, s_t)  # (B, H, L, N//2)

    # Step 4: sum over N
    y = s_scanned.sum(dim=-1)  # (B, H, L)
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def s4d_bidiag_sequential_ref(u, A, B, C, dt, D, scan_gates=None):
    """Ground truth: bidiagonal state evolution over L.
    At each timestep t, state evolves as:
      1. diagonal update: x_n[t] = dA_n * x_n[t-1] + dB_n * u[t]  (standard S4D)
      2. trapezoidal scan across N: x̃_n[t] = gate_n * x̃_{n-1}[t] + x_n[t]
    The scan is INSIDE the recurrence — state coupling happens at every timestep,
    and coupled state feeds back into the next timestep.
    u: (B, H, L)
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]

    dtA = dt.unsqueeze(-1) * A
    dA = torch.exp(dtA)
    dB = B * (dA - 1.0) / A

    if scan_gates is None:
        scan_gates = torch.exp(dtA).abs()  # (H, N//2)
    g = scan_gates  # (H, N//2)

    # state after N-coupling feeds back — this is the coupled state
    x = torch.zeros(Batch, H, Nhalf, dtype=torch.cfloat)
    ys = []
    for t in range(L):
        # diagonal update (using coupled state from previous step)
        x_diag = dA * x + dB * u[:, :, t:t+1]  # (B, H, N//2)

        # trapezoidal scan across N (on real part, since readout is 2*Re)
        x_real = 2.0 * (C.unsqueeze(0) * x_diag).real  # (B, H, N//2)
        # bidirectional scan across N (dim=-1)
        x_scanned = _bidi_scan(
            g.unsqueeze(0).expand(Batch, -1, -1),  # (B, H, N//2)
            x_real
        )  # (B, H, N//2)

        y_t = x_scanned.sum(dim=-1)  # (B, H)
        ys.append(y_t)

        # feed coupled state back (keep complex state for recurrence)
        x = x_diag

    y = torch.stack(ys, dim=-1)
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def s4d_bidiag_coupled_ref(u, A, B, C, dt, D, scan_gates=None):
    """Ground truth variant: N-coupling feeds BACK into state.
    At each timestep, the N-scanned state becomes the input for the next step.
    This is what it means for the trapezoidal to truly replace the diagonal.
    u: (B, H, L)
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]

    dtA = dt.unsqueeze(-1) * A
    dA = torch.exp(dtA)
    dB = B * (dA - 1.0) / A

    if scan_gates is None:
        scan_gates = torch.exp(dtA).abs()
    g = scan_gates  # (H, N//2)

    x = torch.zeros(Batch, H, Nhalf, dtype=torch.cfloat)
    ys = []
    for t in range(L):
        # diagonal update
        x = dA * x + dB * u[:, :, t:t+1]  # (B, H, N//2)

        # N-scan on the complex state itself, then feed back
        # scan real and imag parts separately
        x_r = _bidi_scan(g.unsqueeze(0).expand(Batch, -1, -1), x.real)
        x_i = _bidi_scan(g.unsqueeze(0).expand(Batch, -1, -1), x.imag)
        x = torch.complex(x_r, x_i)  # coupled state feeds back

        y_t = 2.0 * (C.unsqueeze(0) * x).sum(dim=-1).real
        ys.append(y_t)

    y = torch.stack(ys, dim=-1)
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


if __name__ == "__main__":
    torch.manual_seed(42)

    H, N, L, Batch = 8, 16, 32, 2
    A, B, C, dt, D = make_s4d_params(H, N)
    u = torch.randn(Batch, H, L)

    # 1. Verify s4d_standard matches sequential reference
    y_ref = s4d_sequential_ref(u, A, B, C, dt, D)
    y_std = s4d_standard(u, A, B, C, dt, D)
    y_per = s4d_perstate(u, A, B, C, dt, D)

    scan_gates = torch.sigmoid(torch.randn(H, N // 2))

    # 1. Baseline: S4D correctness
    print("=== S4D baseline correctness ===")
    y_ref = s4d_sequential_ref(u, A, B, C, dt, D)
    y_std = s4d_standard(u, A, B, C, dt, D)
    y_per = s4d_perstate(u, A, B, C, dt, D)
    print(f"sequential_ref vs standard:   {(y_ref - y_std).abs().max().item():.2e}")
    print(f"standard vs perstate:         {(y_std - y_per).abs().max().item():.2e}")

    # 2. THE KEY TEST: is (diag kernel over L) + (N-scan after) equivalent to
    #    (bidiagonal state evolution at every timestep)?
    print(f"\n=== KEY: decomposed vs bidiagonal ground truth ===")
    y_nscan = s4d_nscan(u, A, B, C, dt, D, scan_gates=scan_gates)
    y_bidiag = s4d_bidiag_sequential_ref(u, A, B, C, dt, D, scan_gates=scan_gates)
    y_coupled = s4d_bidiag_coupled_ref(u, A, B, C, dt, D, scan_gates=scan_gates)
    print(f"decomposed (FFT+Nscan) vs bidiag (no feedback):  {(y_nscan - y_bidiag).abs().max().item():.2e}")
    print(f"decomposed (FFT+Nscan) vs bidiag (with feedback):{(y_nscan - y_coupled).abs().max().item():.2e}")
    print(f"bidiag (no feedback) vs bidiag (with feedback):   {(y_bidiag - y_coupled).abs().max().item():.2e}")
    print(f"")
    print(f"  decomposed norm:      {y_nscan.norm().item():.4f}")
    print(f"  bidiag no-fb norm:    {y_bidiag.norm().item():.4f}")
    print(f"  bidiag coupled norm:  {y_coupled.norm().item():.4f}")

    # 3. Autograd
    print(f"\n=== Autograd ===")
    C_g = C.detach().requires_grad_(True)
    dt_g = dt.detach().requires_grad_(True)
    sg = scan_gates.detach().requires_grad_(True)
    y_g = s4d_nscan(u, A, B, C_g, dt_g, D, scan_gates=sg)
    y_g.sum().backward()
    print(f"C grad:          {C_g.grad.norm().item():.4f}")
    print(f"dt grad:         {dt_g.grad.norm().item():.4f}")
    print(f"scan_gates grad: {sg.grad.norm().item():.4f}")
    print("OK")
