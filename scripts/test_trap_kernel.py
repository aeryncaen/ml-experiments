"""
Test: deriving and verifying the FFT kernel for a trapezoidal state update
with complex eigenvalues.

Standard S4D:
  h_t = A * h_{t-1} + B * u_t        (diagonal, A complex)
  K[l] = C * A^l * B                  (kernel is geometric series)

Trapezoidal rule (averaging current and previous input):
  h_t = A * h_{t-1} + 0.5 * B * (u_t + u_{t-1})
  This is the trapezoidal integration rule applied to the continuous SSM.

The trapezoidal kernel:
  h_0 = 0.5 * B * u_0
  h_1 = A * 0.5*B*u_0 + 0.5*B*(u_1 + u_0) = 0.5*B*((A+1)*u_0 + u_1)
  h_t = A * h_{t-1} + 0.5*B*(u_t + u_{t-1})

Unrolling:
  h_t = 0.5*B * (u_t + (A+1)*u_{t-1} + A*(A+1)*u_{t-2} + A^2*(A+1)*u_{t-3} + ...)
      = 0.5*B * (u_t + (A+1) * sum_{k=1}^{t} A^{k-1} * u_{t-k})

So K_trap[0] = 0.5*C*B
   K_trap[l] = 0.5*C*B*(A+1)*A^{l-1}  for l >= 1

This is still a geometric series! Just with a different coefficient for l=0 vs l>=1.
The kernel can be computed analytically and applied via FFT.

Let's verify this.
"""

import torch
import torch.nn.functional as F
import math


def make_params(H, N, dt_min=0.001, dt_max=0.1):
    Nhalf = N // 2
    A_real = torch.full((H, Nhalf), -0.5)
    A_imag = -math.pi * torch.arange(Nhalf).float().unsqueeze(0).expand(H, -1)
    A = torch.complex(A_real, A_imag)
    B = torch.ones(H, Nhalf, dtype=torch.cfloat)
    C = torch.randn(H, Nhalf, dtype=torch.cfloat)
    log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    dt = torch.exp(log_dt)
    D = torch.randn(H)
    return A, B, C, dt, D


def trap_sequential_ref(u, A, B, C, dt, D):
    """Sequential reference: trapezoidal integration rule.
    h_t = dA * h_{t-1} + 0.5 * dB * (u_t + u_{t-1})
    y_t = 2 * Re(C * h_t)
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]
    dtA = dt.unsqueeze(-1) * A
    dA = torch.exp(dtA)
    dB = B * (dA - 1.0) / A  # ZOH discretized B

    h = torch.zeros(Batch, H, Nhalf, dtype=torch.cfloat)
    u_prev = torch.zeros(Batch, H, 1)
    ys = []
    for t in range(L):
        u_t = u[:, :, t:t+1]
        h = dA * h + 0.5 * dB * (u_t + u_prev)
        y_t = 2.0 * (C.unsqueeze(0) * h).sum(dim=-1).real
        ys.append(y_t)
        u_prev = u_t
    y = torch.stack(ys, dim=-1)
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def trap_kernel_fft(u, A, B, C, dt, D):
    """FFT kernel version of trapezoidal integration.
    K_trap[0] = 0.5 * C * dB
    K_trap[l] = 0.5 * C * dB * (dA + 1) * dA^{l-1}  for l >= 1
    """
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]
    dtA = dt.unsqueeze(-1) * A
    dA = torch.exp(dtA)
    dB = B * (dA - 1.0) / A

    CB = C * dB  # (H, N//2)
    arange_L = torch.arange(L, device=u.device, dtype=torch.float32)

    # K[0] = 0.5 * CB
    # K[l] = 0.5 * CB * (dA + 1) * dA^{l-1} for l >= 1
    # Construct per-state kernel: (H, N//2, L) complex
    K = torch.zeros(H, Nhalf, L, dtype=torch.cfloat)
    K[:, :, 0] = 0.5 * CB
    if L > 1:
        # dA^{l-1} for l=1..L-1 -> powers 0..L-2
        powers = dA.unsqueeze(-1) ** torch.arange(L - 1, device=u.device, dtype=torch.float32)  # (H, N//2, L-1)
        K[:, :, 1:] = (0.5 * CB * (dA + 1)).unsqueeze(-1) * powers

    # Sum over N with conjugate symmetry, then FFT conv
    K_real = 2.0 * K.sum(dim=1).real  # (H, L)

    fft_len = L + L
    K_f = torch.fft.rfft(K_real, n=fft_len)
    u_f = torch.fft.rfft(u, n=fft_len)
    y = torch.fft.irfft(u_f * K_f, n=fft_len)[..., :L]

    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def s4d_standard(u, A, B, C, dt, D):
    """Standard S4D (ZOH) for comparison."""
    Batch, H, L = u.shape
    dtA = dt.unsqueeze(-1) * A
    dA = torch.exp(dtA)
    CB = C * B * (dA - 1.0) / A
    arange_L = torch.arange(L, device=u.device, dtype=torch.float32)
    K_complex = CB.unsqueeze(-1) * (dA.unsqueeze(-1) ** arange_L)
    K = 2.0 * K_complex.sum(dim=1).real
    fft_len = L + L
    K_f = torch.fft.rfft(K, n=fft_len)
    u_f = torch.fft.rfft(u, n=fft_len)
    y = torch.fft.irfft(u_f * K_f, n=fft_len)[..., :L]
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


def s4d_sequential_ref(u, A, B, C, dt, D):
    """Standard S4D sequential reference."""
    Batch, H, L = u.shape
    Nhalf = A.shape[-1]
    dtA = dt.unsqueeze(-1) * A
    dA = torch.exp(dtA)
    dB = B * (dA - 1.0) / A
    h = torch.zeros(Batch, H, Nhalf, dtype=torch.cfloat)
    ys = []
    for t in range(L):
        h = dA * h + dB * u[:, :, t:t+1]
        y_t = 2.0 * (C.unsqueeze(0) * h).sum(dim=-1).real
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)
    y = y + u * D.unsqueeze(0).unsqueeze(-1)
    return y


if __name__ == "__main__":
    torch.manual_seed(42)
    H, N, L, Batch = 8, 16, 64, 2
    A, B, C, dt, D = make_params(H, N)
    u = torch.randn(Batch, H, L)

    print("=== S4D baseline ===")
    y_s4d_seq = s4d_sequential_ref(u, A, B, C, dt, D)
    y_s4d_fft = s4d_standard(u, A, B, C, dt, D)
    print(f"S4D seq vs FFT: {(y_s4d_seq - y_s4d_fft).abs().max().item():.2e}")

    print(f"\n=== Trapezoidal kernel ===")
    y_trap_seq = trap_sequential_ref(u, A, B, C, dt, D)
    y_trap_fft = trap_kernel_fft(u, A, B, C, dt, D)
    print(f"Trap seq vs FFT: {(y_trap_seq - y_trap_fft).abs().max().item():.2e}")
    print(f"Trap seq norm:   {y_trap_seq.norm().item():.4f}")
    print(f"Trap FFT norm:   {y_trap_fft.norm().item():.4f}")

    print(f"\n=== S4D vs Trapezoidal ===")
    print(f"S4D vs Trap: {(y_s4d_fft - y_trap_fft).abs().mean().item():.4f}  (expected: nonzero)")
    print(f"S4D norm:    {y_s4d_fft.norm().item():.4f}")
    print(f"Trap norm:   {y_trap_fft.norm().item():.4f}")

    print(f"\n=== Autograd ===")
    C_g = C.detach().requires_grad_(True)
    dt_g = dt.detach().requires_grad_(True)
    y_g = trap_kernel_fft(u, A, B, C_g, dt_g, D)
    y_g.sum().backward()
    print(f"C grad:  {C_g.grad.norm().item():.4f}")
    print(f"dt grad: {dt_g.grad.norm().item():.4f}")
    print("OK")
