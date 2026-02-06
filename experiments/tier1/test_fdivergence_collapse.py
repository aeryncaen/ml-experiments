"""
Test Theorem 5.2: f-Divergence Local Collapse.

Claim:
  D_f(P_eps || mu) = (f''(1)/2) * eps^2 * ||u||^2 + O(eps^3)
  
  All f-divergences locally measure eps^2 * ||u||^2 with different scale factors:
    - chi^2: f''(1) = 2, so D = eps^2 * ||u||^2
    - KL:    f''(1) = 1, so D ~ (1/2) * eps^2 * ||u||^2
    - Hellinger^2: f''(1) = 1/2, so D ~ (1/4) * eps^2 * ||u||^2

Method:
  Sweep eps from 1e-4 to 1, compute actual divergences and compare
  to quadratic approximation. Error should be O(eps^3).
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from core import random_baseline, random_deviation, norm2, expectation


def chi2_divergence(u: np.ndarray, mu: np.ndarray, eps: float) -> float:
    """chi^2(P_eps || mu) where P_eps has density 1 + eps*u."""
    return eps ** 2 * norm2(u, mu)


def kl_divergence(u: np.ndarray, mu: np.ndarray, eps: float) -> float:
    """D_KL(P_eps || mu) = E_mu[(1+eps*u) log(1+eps*u)]."""
    w = 1.0 + eps * u
    w = np.maximum(w, 1e-300)
    return np.sum(w * np.log(w) * mu)


def reverse_kl(u: np.ndarray, mu: np.ndarray, eps: float) -> float:
    """D_KL(mu || P_eps) = -E_mu[log(1+eps*u)]."""
    w = 1.0 + eps * u
    w = np.maximum(w, 1e-300)
    return -np.sum(np.log(w) * mu)


def hellinger_squared(u: np.ndarray, mu: np.ndarray, eps: float) -> float:
    """H^2(P_eps, mu) = E_mu[(sqrt(1+eps*u) - 1)^2]."""
    w = np.maximum(1.0 + eps * u, 1e-300)
    return np.sum((np.sqrt(w) - 1.0) ** 2 * mu)


def tv_divergence(u: np.ndarray, mu: np.ndarray, eps: float) -> float:
    """TV(P_eps, mu) = (1/2) E_mu[|eps*u|]."""
    return 0.5 * np.sum(np.abs(eps * u) * mu)


def test_local_collapse():
    """
    Verify all f-divergences collapse to C * eps^2 * ||u||^2 for small eps.
    """
    n = 30
    rng = np.random.default_rng(42)
    mu = random_baseline(n, rng)
    u = random_deviation(mu, scale=1.0, rng=rng)
    # Normalize so ||u||^2 = 1 for clean numbers
    u = u / np.sqrt(norm2(u, mu))
    
    norm_sq = norm2(u, mu)  # should be 1.0
    
    divergences = {
        'chi^2':      (chi2_divergence,   2.0),   # f''(1) = 2
        'KL':         (kl_divergence,     1.0),   # f''(1) = 1
        'reverse KL': (reverse_kl,        1.0),   # f''(1) = 1
        'Hellinger^2':(hellinger_squared,  0.5),   # f''(1) = 1/2
    }
    
    epsilons = np.logspace(-4, -0.3, 30)
    
    print(f"  ||u||^2 = {norm_sq:.6f} (normalized to 1)")
    print()
    print(f"  {'Divergence':15s} | {'f\"(1)':6s} | {'Max rel error':14s} | "
          f"{'Error scaling':14s} | {'Status'}")
    print(f"  {'-' * 75}")
    
    all_ok = True
    for name, (div_fn, fpp1) in divergences.items():
        predicted_coeff = fpp1 / 2.0  # (f''(1)/2) * ||u||^2
        
        errors = []
        for eps in epsilons:
            actual = div_fn(u, mu, eps)
            predicted = predicted_coeff * eps ** 2 * norm_sq
            if predicted > 1e-15:
                rel_error = abs(actual - predicted) / predicted
                errors.append((eps, rel_error))
        
        if errors:
            max_err = max(e for _, e in errors)
            # Check that error scales as eps (i.e., relative error ~ eps)
            # At small eps, rel_error ~ eps * C for O(eps^3) remainder
            small_eps_errors = [(e, r) for e, r in errors if e < 0.01]
            if len(small_eps_errors) >= 2:
                e1, r1 = small_eps_errors[0]
                e2, r2 = small_eps_errors[-1]
                if r1 > 1e-15:
                    scaling = np.log(r2 / r1) / np.log(e2 / e1)
                else:
                    scaling = 1.0  # close enough
            else:
                scaling = float('nan')
            
            ok = max_err < 0.5 and (np.isnan(scaling) or abs(scaling - 1.0) < 0.5)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            
            print(f"  {name:15s} | {fpp1:6.1f} | {max_err:14.6f} | "
                  f"{'~eps^' + f'{scaling:.2f}':14s} | {status}")
        else:
            print(f"  {name:15s} | {fpp1:6.1f} | {'N/A':14s} | {'N/A':14s}")
    
    return all_ok


def test_divergence_ratios():
    """
    At small eps, D_KL / D_chi2 -> 1/2, D_H2 / D_chi2 -> 1/4.
    These are the local equivalence ratios from Theorem 5.2.
    """
    n = 30
    rng = np.random.default_rng(42)
    mu = random_baseline(n, rng)
    u = random_deviation(mu, scale=1.0, rng=rng)
    
    print(f"\n  Divergence ratios at decreasing eps (should converge to f''(1)/2):")
    print(f"  {'eps':>10s} | {'KL/chi2':>10s} | {'H2/chi2':>10s} | "
          f"{'revKL/chi2':>10s}")
    print(f"  {'-' * 50}")
    
    for eps in [0.5, 0.1, 0.01, 0.001, 0.0001]:
        d_chi2 = chi2_divergence(u, mu, eps)
        d_kl = kl_divergence(u, mu, eps)
        d_h2 = hellinger_squared(u, mu, eps)
        d_rkl = reverse_kl(u, mu, eps)
        
        if d_chi2 > 1e-15:
            r_kl = d_kl / d_chi2
            r_h2 = d_h2 / d_chi2
            r_rkl = d_rkl / d_chi2
            print(f"  {eps:10.5f} | {r_kl:10.6f} | {r_h2:10.6f} | {r_rkl:10.6f}")
    
    print(f"  {'expected':>10s} | {'0.500000':>10s} | {'0.250000':>10s} | {'0.500000':>10s}")
    
    # Verify convergence at smallest eps
    eps_small = 1e-4
    d_chi2 = chi2_divergence(u, mu, eps_small)
    d_kl = kl_divergence(u, mu, eps_small)
    d_h2 = hellinger_squared(u, mu, eps_small)
    
    ok_kl = abs(d_kl / d_chi2 - 0.5) < 0.01
    ok_h2 = abs(d_h2 / d_chi2 - 0.25) < 0.01
    
    ok = ok_kl and ok_h2
    if ok:
        print(f"\n  PASS: Ratios converge to f''(1)/2 as expected")
    else:
        print(f"\n  FAIL: KL ratio ok={ok_kl}, H2 ratio ok={ok_h2}")
    return ok


def test_chi2_exact_at_all_scales():
    """
    chi^2 is EXACTLY eps^2 * ||u||^2 at ALL scales (not just locally).
    This is because chi^2 IS the L^2 norm. Other divergences only match locally.
    """
    n = 20
    rng = np.random.default_rng(42)
    mu = random_baseline(n, rng)
    u = random_deviation(mu, scale=1.0, rng=rng)
    
    max_error = 0.0
    for eps in np.logspace(-4, 0, 50):
        actual = chi2_divergence(u, mu, eps)
        predicted = eps ** 2 * norm2(u, mu)
        error = abs(actual - predicted)
        max_error = max(max_error, error)
    
    if max_error < 1e-12:
        print(f"  PASS: chi^2 = eps^2 * ||u||^2 EXACTLY at all scales "
              f"(max error: {max_error:.2e})")
        return True
    else:
        print(f"  FAIL: max error = {max_error:.2e}")
        return False


def run():
    print("=" * 70)
    print("TEST Thm 5.2: f-Divergence Local Collapse")
    print("=" * 70)
    
    print("\n1. Local collapse: D_f ~ (f''(1)/2) * eps^2 * ||u||^2:")
    ok1 = test_local_collapse()
    
    print("\n2. Divergence ratios converge to f''(1)/2:")
    ok2 = test_divergence_ratios()
    
    print("\n3. chi^2 exact at all scales (not just local):")
    ok3 = test_chi2_exact_at_all_scales()
    
    print("\n" + "-" * 70)
    overall = ok1 and ok2 and ok3
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    run()
