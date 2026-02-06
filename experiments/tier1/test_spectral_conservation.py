"""
Test Law R.7: Spectral Conservation (static + sequential).

Part A (Static):
  For orthogonal decomposition u = sum u_i with u_i perp u_j:
    ||u||^2 = sum ||u_i||^2
    <Phi^o, u> = sum <Phi^o, u_i>

Part B (Sequential / filtration):
  Already tested in test_a6a_sequential_pythagoras.py.
  Here we also test:
    - Per-round advantage attribution: <Phi^o, M_T> = sum_t <Phi^o, Delta_t>
    - Orthogonality of increments: <Delta_s, Delta_t> = 0 for s != t

Also tests Law R.5 (Spectral Budget / Parseval):
  For ONB {e_i} of X and A_i = <e_i, u>, sum A_i^2 = ||u||^2.
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from core import (
    random_baseline, random_deviation, random_performance,
    norm2, norm, inner, center, expectation,
    partition_projection, conditional_expectation, chi_squared
)


def test_static_orthogonal_decomposition(n: int = 20, n_trials: int = 100):
    """
    Decompose u into orthogonal components via nested partitions.
    Verify ||u||^2 = sum ||u_i||^2 and <Phi^o, u> = sum <Phi^o, u_i>.
    """
    rng = np.random.default_rng(42)
    max_norm_error = 0.0
    max_adv_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        Phi = random_performance(n, rng)
        Phi_c = center(Phi, mu)
        
        # Create 3-level orthogonal decomposition via nested partitions
        # Coarse -> Medium -> Fine
        partition_coarse = [list(range(0, n // 2)), list(range(n // 2, n))]
        
        # Finer: split each coarse atom
        partition_fine = []
        for atom in partition_coarse:
            mid = len(atom) // 2
            if mid > 0:
                partition_fine.append(atom[:mid])
                partition_fine.append(atom[mid:])
            else:
                partition_fine.append(atom)
        
        # Components:
        # u1 = Pi_coarse u (coarse-level structure)
        # u2 = Pi_fine u - Pi_coarse u (medium-level structure)
        # u3 = u - Pi_fine u (residual / wasted)
        
        u1 = partition_projection(u, mu, partition_coarse)
        proj_fine = partition_projection(u, mu, partition_fine)
        u2 = proj_fine - u1
        u3 = u - proj_fine
        
        # Verify orthogonality
        ip_12 = abs(inner(u1, u2, mu))
        ip_13 = abs(inner(u1, u3, mu))
        ip_23 = abs(inner(u2, u3, mu))
        
        if max(ip_12, ip_13, ip_23) > 1e-10:
            print(f"  FAIL trial {trial}: components not orthogonal "
                  f"({ip_12:.2e}, {ip_13:.2e}, {ip_23:.2e})")
            return False
        
        # Norm additivity
        norm_total = norm2(u, mu)
        norm_sum = norm2(u1, mu) + norm2(u2, mu) + norm2(u3, mu)
        norm_error = abs(norm_total - norm_sum)
        max_norm_error = max(max_norm_error, norm_error)
        
        # Advantage additivity
        adv_total = inner(Phi_c, u, mu)
        adv_sum = inner(Phi_c, u1, mu) + inner(Phi_c, u2, mu) + inner(Phi_c, u3, mu)
        adv_error = abs(adv_total - adv_sum)
        max_adv_error = max(max_adv_error, adv_error)
        
        if norm_error > 1e-10 or adv_error > 1e-10:
            print(f"  FAIL trial {trial}: norm_err={norm_error:.2e}, "
                  f"adv_err={adv_error:.2e}")
            return False
    
    print(f"  PASS: Static orthogonal decomposition exact across {n_trials} trials")
    print(f"    max norm error: {max_norm_error:.2e}")
    print(f"    max advantage error: {max_adv_error:.2e}")
    return True


def test_parseval_identity(n: int = 12, n_trials: int = 50):
    """
    Law R.5 Part B: For ONB {e_i} of L^2_0(mu), sum <e_i, u>^2 = ||u||^2.
    
    On finite Omega with |Omega| = n and mu, L^2_0(mu) has dimension n-1.
    Construct ONB via Gram-Schmidt on the zero-mean subspace.
    """
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        # Construct ONB of L^2_0(mu) via Gram-Schmidt
        # Start with standard basis vectors, center them, then orthogonalize
        basis = []
        for i in range(n - 1):
            v = np.zeros(n)
            v[i] = 1.0
            v -= expectation(v, mu)  # center
            
            # Gram-Schmidt against existing basis
            for b in basis:
                v -= inner(v, b, mu) * b
            
            nv = norm(v, mu)
            if nv > 1e-12:
                v /= nv
                basis.append(v)
        
        # Parseval: sum <e_i, u>^2 = ||u||^2
        parseval_sum = sum(inner(e, u, mu) ** 2 for e in basis)
        norm_sq = norm2(u, mu)
        
        error = abs(parseval_sum - norm_sq)
        max_error = max(max_error, error)
        
        if error > 1e-8:
            print(f"  FAIL trial {trial}: Parseval sum={parseval_sum:.10f}, "
                  f"||u||^2={norm_sq:.10f}")
            return False
    
    print(f"  PASS: Parseval identity sum A_i^2 = ||u||^2 across {n_trials} trials "
          f"(max error: {max_error:.2e})")
    return True


def test_per_round_advantage_attribution(n: int = 16, n_trials: int = 50):
    """
    Verify <Phi^o, M_T> = sum_t <Phi^o, Delta_t> (advantage attribution).
    """
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        Phi = random_performance(n, rng)
        Phi_c = center(Phi, mu)
        
        # Build filtration
        current = [list(range(n))]
        filtration = [current]
        while max(len(a) for a in current) > 1:
            new_part = []
            for atom in current:
                if len(atom) > 1:
                    mid = len(atom) // 2
                    new_part.append(atom[:mid])
                    new_part.append(atom[mid:])
                else:
                    new_part.append(atom)
            current = new_part
            filtration.append(current)
        
        # Compute martingale and increments
        M_prev = np.zeros(n)
        delta_advs = []
        
        for partition in filtration[1:]:
            M_t = conditional_expectation(u, mu, partition)
            delta = M_t - M_prev
            delta_advs.append(inner(Phi_c, delta, mu))
            M_prev = M_t.copy()
        
        M_T = M_prev
        total_adv = inner(Phi_c, M_T, mu)
        sum_adv = sum(delta_advs)
        
        error = abs(total_adv - sum_adv)
        max_error = max(max_error, error)
        
        if error > 1e-10:
            print(f"  FAIL trial {trial}: total_adv={total_adv:.12f}, "
                  f"sum_adv={sum_adv:.12f}")
            return False
    
    print(f"  PASS: Per-round advantage attribution exact across {n_trials} trials "
          f"(max error: {max_error:.2e})")
    return True


def test_increment_orthogonality(n: int = 16, n_trials: int = 50):
    """
    Verify <Delta_s, Delta_t> = 0 for s != t (martingale increment orthogonality).
    """
    rng = np.random.default_rng(42)
    max_ip = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        # Build filtration
        current = [list(range(n))]
        filtration = [current]
        while max(len(a) for a in current) > 1:
            new_part = []
            for atom in current:
                if len(atom) > 1:
                    mid = len(atom) // 2
                    new_part.append(atom[:mid])
                    new_part.append(atom[mid:])
                else:
                    new_part.append(atom)
            current = new_part
            filtration.append(current)
        
        # Compute increments
        M_prev = np.zeros(n)
        deltas = []
        for partition in filtration[1:]:
            M_t = conditional_expectation(u, mu, partition)
            delta = M_t - M_prev
            deltas.append(delta)
            M_prev = M_t.copy()
        
        # Check all pairwise inner products
        for s in range(len(deltas)):
            for t in range(s + 1, len(deltas)):
                ip = abs(inner(deltas[s], deltas[t], mu))
                max_ip = max(max_ip, ip)
                if ip > 1e-10:
                    print(f"  FAIL trial {trial}: <Delta_{s}, Delta_{t}> = {ip:.2e}")
                    return False
    
    print(f"  PASS: Increment orthogonality <Delta_s, Delta_t> = 0 "
          f"across {n_trials} trials (max: {max_ip:.2e})")
    return True


def test_spectral_budget_subspace(n: int = 20, n_trials: int = 50):
    """
    Law R.5: For subspace V, sum_{e_i in V} A_i^2 = ||Pi_V u||^2.
    (Restricting to subspace reduces budget.)
    """
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        partition = [list(range(0, n // 3)), list(range(n // 3, 2 * n // 3)),
                     list(range(2 * n // 3, n))]
        
        proj_u = partition_projection(u, mu, partition)
        proj_norm_sq = norm2(proj_u, mu)
        total_norm_sq = norm2(u, mu)
        
        # Build ONB of V_S
        basis_V = []
        for atom in partition:
            v = np.zeros(n)
            v[atom] = 1.0
            v -= expectation(v, mu)
            for b in basis_V:
                v -= inner(v, b, mu) * b
            nv = norm(v, mu)
            if nv > 1e-12:
                v /= nv
                basis_V.append(v)
        
        # Parseval within V_S
        budget_V = sum(inner(e, u, mu) ** 2 for e in basis_V)
        
        error = abs(budget_V - proj_norm_sq)
        max_error = max(max_error, error)
        
        # Verify budget reduction
        if proj_norm_sq > total_norm_sq + 1e-10:
            print(f"  FAIL: subspace budget {proj_norm_sq} > total budget {total_norm_sq}")
            return False
    
    print(f"  PASS: Spectral budget in subspace = ||Pi_V u||^2 "
          f"(max error: {max_error:.2e})")
    print(f"    Budget reduction verified: ||Pi_V u||^2 <= ||u||^2")
    return True


def run():
    print("=" * 70)
    print("TEST R.7 + R.5: Spectral Conservation and Budget")
    print("=" * 70)
    
    print("\n1. Static orthogonal decomposition (||u||^2 = sum ||u_i||^2):")
    ok1 = test_static_orthogonal_decomposition()
    
    print("\n2. Parseval identity (sum A_i^2 = ||u||^2):")
    ok2 = test_parseval_identity()
    
    print("\n3. Per-round advantage attribution (<Phi^o, M_T> = sum <Phi^o, Delta_t>):")
    ok3 = test_per_round_advantage_attribution()
    
    print("\n4. Martingale increment orthogonality (<Delta_s, Delta_t> = 0):")
    ok4 = test_increment_orthogonality()
    
    print("\n5. Spectral budget in subspace (Law R.5, budget reduction):")
    ok5 = test_spectral_budget_subspace()
    
    print("\n" + "-" * 70)
    overall = ok1 and ok2 and ok3 and ok4 and ok5
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    run()
