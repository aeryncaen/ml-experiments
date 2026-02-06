"""
Core L^2(mu) primitives for geometric optimization verification.

All operations on FINITE discrete (Omega, mu). Functions are vectors in R^n,
inner product is <f,g> = sum_i f(i)*g(i)*mu(i), norm is ||f||^2 = <f,f>.

Conventions:
  - mu: probability vector (sums to 1), the baseline measure
  - w = dP/dmu: density ratio (Radon-Nikodym derivative)
  - u = w - 1: zero-mean deviation, E_mu[u] = 0
  - Phi: performance functional in L^2(mu)
  - Phi_c: centered performance, Phi - E_mu[Phi]
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Inner product and norms
# ---------------------------------------------------------------------------

def inner(f: np.ndarray, g: np.ndarray, mu: np.ndarray) -> float:
    """L^2(mu) inner product: <f,g> = E_mu[fg] = sum f*g*mu."""
    return np.sum(f * g * mu)


def norm2(f: np.ndarray, mu: np.ndarray) -> float:
    """Squared L^2(mu) norm: ||f||^2 = E_mu[f^2]."""
    return inner(f, f, mu)


def norm(f: np.ndarray, mu: np.ndarray) -> float:
    """L^2(mu) norm."""
    return np.sqrt(max(0.0, norm2(f, mu)))


def lp_norm(f: np.ndarray, mu: np.ndarray, p: float) -> float:
    """L^p(mu) norm: (E_mu[|f|^p])^{1/p}."""
    return np.sum(np.abs(f) ** p * mu) ** (1.0 / p)


# ---------------------------------------------------------------------------
# Centering and deviation
# ---------------------------------------------------------------------------

def expectation(f: np.ndarray, mu: np.ndarray) -> float:
    """E_mu[f] = sum f*mu."""
    return np.sum(f * mu)


def center(f: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Phi^o := Phi - E_mu[Phi]."""
    return f - expectation(f, mu)


def deviation(w: np.ndarray) -> np.ndarray:
    """u = w - 1 where w = dP/dmu."""
    return w - 1.0


def chi_squared(u: np.ndarray, mu: np.ndarray) -> float:
    """chi^2(P||mu) = E_mu[u^2] = ||u||^2."""
    return norm2(u, mu)


# ---------------------------------------------------------------------------
# Conditional expectation / projection onto sigma-algebra subspace
# ---------------------------------------------------------------------------

def partition_projection(f: np.ndarray, mu: np.ndarray,
                         partition: list[list[int]]) -> np.ndarray:
    """
    Orthogonal projection Pi_S f = E_mu[f|S] - E_mu[f]
    where S generates a partition of Omega.
    
    partition: list of lists, each inner list is indices forming one atom.
    Returns the zero-mean projected function.
    """
    E_f = expectation(f, mu)
    result = np.zeros_like(f)
    for atom in partition:
        atom = np.array(atom)
        mu_atom = np.sum(mu[atom])
        if mu_atom > 1e-15:
            cond_exp = np.sum(f[atom] * mu[atom]) / mu_atom
            result[atom] = cond_exp - E_f
    return result


def conditional_expectation(f: np.ndarray, mu: np.ndarray,
                            partition: list[list[int]]) -> np.ndarray:
    """
    E_mu[f|S](omega) for each omega, where S generates the partition.
    NOT centered (returns the raw conditional expectation).
    """
    result = np.zeros_like(f)
    for atom in partition:
        atom = np.array(atom)
        mu_atom = np.sum(mu[atom])
        if mu_atom > 1e-15:
            cond_exp = np.sum(f[atom] * mu[atom]) / mu_atom
            result[atom] = cond_exp
    return result


# ---------------------------------------------------------------------------
# Observable divergence
# ---------------------------------------------------------------------------

def observable_chi_squared(u: np.ndarray, mu: np.ndarray,
                           partition: list[list[int]]) -> float:
    """chi^2(P_S || mu_S) = ||Pi_S u||^2."""
    proj = partition_projection(u, mu, partition)
    return norm2(proj, mu)


# ---------------------------------------------------------------------------
# Advantage
# ---------------------------------------------------------------------------

def advantage(Phi: np.ndarray, u: np.ndarray, mu: np.ndarray) -> float:
    """E_P[Phi] - E_mu[Phi] = <Phi^o, u>."""
    Phi_c = center(Phi, mu)
    return inner(Phi_c, u, mu)


def advantage_direct(Phi: np.ndarray, w: np.ndarray, mu: np.ndarray) -> float:
    """E_P[Phi] - E_mu[Phi] computed directly as E_mu[Phi*u]."""
    u = deviation(w)
    return np.sum(Phi * u * mu)


# ---------------------------------------------------------------------------
# Alignment coefficient
# ---------------------------------------------------------------------------

def alignment(Phi_c: np.ndarray, proj_u: np.ndarray, mu: np.ndarray) -> float:
    """
    rho = <Phi^o, Pi_S u> / (||Phi^o|| * ||Pi_S u||)
    Returns NaN if either norm is zero.
    """
    n1 = norm(Phi_c, mu)
    n2 = norm(proj_u, mu)
    if n1 < 1e-15 or n2 < 1e-15:
        return float('nan')
    return inner(Phi_c, proj_u, mu) / (n1 * n2)


# ---------------------------------------------------------------------------
# Filtration tools (for sequential tests)
# ---------------------------------------------------------------------------

def filtration_increments(u: np.ndarray, mu: np.ndarray,
                          filtration: list[list[list[int]]]) -> list[np.ndarray]:
    """
    Given a filtration (sequence of increasingly fine partitions),
    compute martingale increments Delta_t = M_t - M_{t-1}.
    
    filtration: [partition_0, partition_1, ..., partition_T]
      where partition_0 is trivial (all of Omega), partition_T is finest.
    
    Returns list of Delta_t arrays (length T).
    """
    n = len(u)
    # M_0 = E_mu[u | G_0] = E_mu[u] = 0 for trivial sigma-algebra
    M_prev = np.zeros(n)
    deltas = []
    for t, partition in enumerate(filtration):
        M_t = conditional_expectation(u, mu, partition)
        delta = M_t - M_prev
        deltas.append(delta)
        M_prev = M_t.copy()
    return deltas


# ---------------------------------------------------------------------------
# Utility: random problem generation
# ---------------------------------------------------------------------------

def random_baseline(n: int, rng: np.random.Generator = None) -> np.ndarray:
    """Random probability vector (Dirichlet-uniform)."""
    if rng is None:
        rng = np.random.default_rng()
    mu = rng.dirichlet(np.ones(n))
    return mu


def random_deviation(mu: np.ndarray, scale: float = 1.0,
                     rng: np.random.Generator = None) -> np.ndarray:
    """
    Random zero-mean deviation u with E_mu[u]=0 and ||u||_2 ~ scale.
    Ensures w = 1+u >= 0 (valid density).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(mu)
    # Raw random, then center
    raw = rng.standard_normal(n)
    raw -= expectation(raw, mu)  # E_mu[raw] = 0
    # Scale
    cur_norm = norm(raw, mu)
    if cur_norm < 1e-15:
        return np.zeros(n)
    u = raw * (scale / cur_norm)
    # Ensure w = 1+u >= 0: clamp then re-center
    w = np.maximum(1.0 + u, 1e-10)
    w /= expectation(w, mu)  # renormalize
    u = w - 1.0
    return u


def random_performance(n: int, rng: np.random.Generator = None) -> np.ndarray:
    """Random performance functional."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(n)
