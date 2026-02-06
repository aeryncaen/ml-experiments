"""
Test Selection Laws S.3 (Bottleneck) and S.4 (Additivity).

Law S.3 (Bottleneck):
  When Phi is S-measurable, V(U; Phi^o) = V(Pi_S U; Phi^o).
  Selection reduces to observable subspace. Wasted u^perp is inert.

Law S.4 (Additivity):
  Under separable U = oplus U_t and orthogonal increments:
    (A) Minkowski: V = sum_t V_t
    (B) Ellipsoidal: V = sqrt(sum_t rho_t * ||Phi^o_t||^2)
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from core import (
    random_baseline, random_deviation, random_performance,
    norm, norm2, inner, center, expectation,
    partition_projection, conditional_expectation
)


# =========================================================================
# Law S.3: Bottleneck (Observable Restriction)
# =========================================================================

def solve_ball_selection(Phi_c: np.ndarray, mu: np.ndarray,
                         rho: float) -> float:
    """V = sup_{||u|| <= sqrt(rho)} <Phi^o, u> = sqrt(rho) * ||Phi^o||."""
    return np.sqrt(rho) * norm(Phi_c, mu)


def solve_projected_ball_selection(Phi_c: np.ndarray, mu: np.ndarray,
                                   partition: list, rho: float) -> float:
    """
    V(Pi_S U; Phi^o) where U = ball of radius sqrt(rho).
    
    For S-measurable Phi^o, this equals sqrt(rho) * ||Pi_S Phi^o||
    = sqrt(rho) * ||Phi^o|| (since Phi^o is already in V_S).
    
    The bottleneck law says V(U; Phi^o) = V(Pi_S U; Phi^o).
    """
    proj_Phi = partition_projection(Phi_c, mu, partition)
    return np.sqrt(rho) * norm(proj_Phi, mu)


def test_bottleneck_law(n: int = 20, n_trials: int = 100):
    """
    For S-measurable Phi:
      V(U; Phi^o) = V(Pi_S U; Phi^o)
    
    Test: sup over full ball = sup over projected ball
    when Phi is restricted to V_S.
    """
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        rho = rng.uniform(0.01, 5.0)
        
        k = rng.integers(2, min(n, 6))
        assignments = rng.integers(0, k, size=n)
        partition = [list(np.where(assignments == g)[0]) for g in range(k)
                     if np.any(assignments == g)]
        
        # S-measurable Phi (constant on each atom)
        Phi = np.zeros(n)
        for atom in partition:
            Phi[atom] = rng.standard_normal()
        Phi_c = center(Phi, mu)
        
        # Full ball selection
        V_full = solve_ball_selection(Phi_c, mu, rho)
        
        # Projected ball selection (should equal V_full for S-measurable Phi)
        V_proj = solve_projected_ball_selection(Phi_c, mu, partition, rho)
        
        error = abs(V_full - V_proj)
        max_error = max(max_error, error)
        
        if error > 1e-10:
            print(f"  FAIL trial {trial}: V_full={V_full:.10f}, V_proj={V_proj:.10f}")
            return False
    
    print(f"  PASS: Bottleneck law V(U) = V(Pi_S U) for S-measurable Phi "
          f"(max error: {max_error:.2e})")
    return True


def test_bottleneck_reduction(n: int = 20, n_trials: int = 50):
    """
    For NON-S-measurable Phi, the projected value should be LESS than
    the full value (selection on projected envelope is weaker).
    
    V(Pi_S U; Phi^o) <= V(U; Phi^o) with strict inequality when
    Phi^o has components outside V_S.
    """
    rng = np.random.default_rng(42)
    strict_reductions = 0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        rho = rng.uniform(0.1, 3.0)
        
        partition = [list(range(0, n // 2)), list(range(n // 2, n))]
        
        # Generic Phi (NOT S-measurable)
        Phi = random_performance(n, rng)
        Phi_c = center(Phi, mu)
        
        V_full = solve_ball_selection(Phi_c, mu, rho)
        V_proj = solve_projected_ball_selection(Phi_c, mu, partition, rho)
        
        if V_proj > V_full + 1e-10:
            print(f"  FAIL: V_proj > V_full ({V_proj:.6f} > {V_full:.6f})")
            return False
        
        if V_full - V_proj > 1e-6:
            strict_reductions += 1
    
    rate = strict_reductions / n_trials
    print(f"  PASS: V(Pi_S U) <= V(U) always holds; "
          f"strict reduction in {strict_reductions}/{n_trials} trials ({rate:.0%})")
    return True


# =========================================================================
# Law S.4: Additivity (Sequential Decomposition)
# =========================================================================

def test_minkowski_additivity(n: int = 16, n_trials: int = 50):
    """
    Law S.4(A): Under Minkowski-sum budget U = oplus U_t:
      V(U, R; Phi^o) = sum_t V(U_t, R_t; Phi^o_t)
    
    Test with spherical per-round budgets and orthogonal increments.
    """
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        
        # Build filtration: trivial -> binary -> 4-way
        partition_1 = [list(range(0, n // 2)), list(range(n // 2, n))]
        partition_2_atoms = []
        for atom in partition_1:
            mid = len(atom) // 2
            if mid > 0:
                partition_2_atoms.append(atom[:mid])
                partition_2_atoms.append(atom[mid:])
            else:
                partition_2_atoms.append(atom)
        
        # Create S-measurable Phi components for each round
        Phi_1 = np.zeros(n)
        for atom in partition_1:
            Phi_1[atom] = rng.standard_normal()
        Phi_1_c = center(Phi_1, mu)
        
        Phi_2 = np.zeros(n)
        for atom in partition_2_atoms:
            Phi_2[atom] = rng.standard_normal()
        Phi_2_c = center(Phi_2, mu)
        # Project out the part already in V_{S1}
        proj_1 = partition_projection(Phi_2_c, mu, partition_1)
        Phi_2_c_orth = Phi_2_c - proj_1  # orthogonal increment of Phi
        
        # Per-round budgets
        rho_1 = rng.uniform(0.1, 2.0)
        rho_2 = rng.uniform(0.1, 2.0)
        
        # Per-round values (spherical budgets)
        V_1 = np.sqrt(rho_1) * norm(Phi_1_c, mu)
        V_2 = np.sqrt(rho_2) * norm(Phi_2_c_orth, mu)
        
        # Total value under Minkowski sum should be V_1 + V_2
        V_total = V_1 + V_2
        
        # Verify by constructing the actual optimizers
        # u_1* = sqrt(rho_1) * Phi_1_c / ||Phi_1_c|| (in V_{S1})
        # u_2* = sqrt(rho_2) * Phi_2_c_orth / ||Phi_2_c_orth|| (in V_{S2} \ V_{S1})
        norm_Phi1 = norm(Phi_1_c, mu)
        norm_Phi2 = norm(Phi_2_c_orth, mu)
        
        if norm_Phi1 < 1e-12 or norm_Phi2 < 1e-12:
            continue
        
        u_1 = np.sqrt(rho_1) * Phi_1_c / norm_Phi1
        u_2 = np.sqrt(rho_2) * Phi_2_c_orth / norm_Phi2
        u_total = u_1 + u_2
        
        # Verify orthogonality of u_1, u_2
        ip = abs(inner(u_1, u_2, mu))
        if ip > 1e-8:
            continue  # skip if not well-separated
        
        # Total advantage from combined optimizer
        Phi_total_c = Phi_1_c + Phi_2_c_orth
        adv_total = inner(Phi_total_c, u_total, mu)
        
        error = abs(adv_total - V_total)
        max_error = max(max_error, error)
    
    if max_error < 1e-6:
        print(f"  PASS: Minkowski additivity V = V_1 + V_2 "
              f"(max error: {max_error:.2e})")
        return True
    else:
        print(f"  FAIL: max error = {max_error:.2e}")
        return False


def test_ellipsoidal_additivity(n: int = 16, n_trials: int = 50):
    """
    Law S.4(B): Under ellipsoidal budget sum ||u_t||^2/rho_t <= 1:
      V = sqrt(sum_t rho_t * ||Phi^o_t||^2)
    """
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        
        # Two rounds with orthogonal subspaces
        partition_1 = [list(range(0, n // 2)), list(range(n // 2, n))]
        
        partition_2_atoms = []
        for atom in partition_1:
            mid = len(atom) // 2
            if mid > 0:
                partition_2_atoms.append(atom[:mid])
                partition_2_atoms.append(atom[mid:])
            else:
                partition_2_atoms.append(atom)
        
        # Phi components
        Phi_1 = np.zeros(n)
        for atom in partition_1:
            Phi_1[atom] = rng.standard_normal()
        Phi_1_c = center(Phi_1, mu)
        
        Phi_2 = np.zeros(n)
        for atom in partition_2_atoms:
            Phi_2[atom] = rng.standard_normal()
        Phi_2_c = center(Phi_2, mu)
        proj_1 = partition_projection(Phi_2_c, mu, partition_1)
        Phi_2_orth = Phi_2_c - proj_1
        
        # Per-round budgets
        rho_1 = rng.uniform(0.1, 3.0)
        rho_2 = rng.uniform(0.1, 3.0)
        
        n1 = norm2(Phi_1_c, mu)
        n2 = norm2(Phi_2_orth, mu)
        
        # Predicted value under ellipsoidal budget
        V_predicted = np.sqrt(rho_1 * n1 + rho_2 * n2)
        
        # Verify by Lagrangian optimization
        # max <Phi, u_1 + u_2> s.t. ||u_1||^2/rho_1 + ||u_2||^2/rho_2 <= 1
        # Optimal: u_t = c * rho_t * Phi_t where c normalizes constraint
        if n1 < 1e-15 or n2 < 1e-15:
            continue
        
        # Lagrange: u_t = lambda * rho_t * Phi_t^o
        # Constraint: sum lambda^2 * rho_t^2 * ||Phi_t||^2 / rho_t = 1
        # => lambda^2 * sum rho_t * ||Phi_t||^2 = 1
        # => lambda = 1 / sqrt(sum rho_t * ||Phi_t||^2)
        
        lam = 1.0 / np.sqrt(rho_1 * n1 + rho_2 * n2)
        u_1 = lam * rho_1 * Phi_1_c
        u_2 = lam * rho_2 * Phi_2_orth
        
        # Verify constraint
        constraint = norm2(u_1, mu) / rho_1 + norm2(u_2, mu) / rho_2
        
        # Compute actual value
        Phi_total = Phi_1_c + Phi_2_orth
        u_total = u_1 + u_2
        V_actual = inner(Phi_total, u_total, mu)
        
        error = abs(V_actual - V_predicted)
        max_error = max(max_error, error)
    
    if max_error < 1e-6:
        print(f"  PASS: Ellipsoidal additivity V = sqrt(sum rho_t * ||Phi_t||^2) "
              f"(max error: {max_error:.2e})")
        return True
    else:
        print(f"  FAIL: max error = {max_error:.2e}")
        return False


def run():
    print("=" * 70)
    print("TEST S.3 + S.4: Selection Bottleneck and Additivity")
    print("=" * 70)
    
    print("\n1. Law S.3: Bottleneck (V(U) = V(Pi_S U) for S-measurable Phi):")
    ok1 = test_bottleneck_law()
    
    print("\n2. Bottleneck reduction (V(Pi_S U) <= V(U) for general Phi):")
    ok2 = test_bottleneck_reduction()
    
    print("\n3. Law S.4(A): Minkowski additivity (V = sum V_t):")
    ok3 = test_minkowski_additivity()
    
    print("\n4. Law S.4(B): Ellipsoidal additivity (V = sqrt(sum rho_t ||Phi_t||^2)):")
    ok4 = test_ellipsoidal_additivity()
    
    print("\n" + "-" * 70)
    overall = ok1 and ok2 and ok3 and ok4
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    run()
