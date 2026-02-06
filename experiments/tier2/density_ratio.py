"""
Density ratio estimation via uLSIF (unconstrained Least-Squares 
Importance Fitting) and kernel logistic regression.

This is the foundation for all Tier 2 experiments. Everything in the 
geometric framework depends on estimating u = dP/dmu - 1 from samples.

References:
  Kanamori, Hido, Sugiyama (2009). A least-squares approach to 
  density ratio estimation. JMLR 10:1391-1445.
"""

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def rbf_kernel_matrix(X, Y, sigma):
    """Gaussian RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))."""
    D = cdist(X, Y, 'sqeuclidean')
    return np.exp(-D / (2.0 * sigma ** 2))


def median_heuristic(X, Y):
    """Median heuristic for RBF bandwidth selection."""
    XY = np.vstack([X, Y])
    D = cdist(XY, XY, 'sqeuclidean')
    # Median of nonzero pairwise distances
    upper = D[np.triu_indices_from(D, k=1)]
    return np.sqrt(np.median(upper) / 2.0)


# ---------------------------------------------------------------------------
# uLSIF: Unconstrained Least-Squares Importance Fitting
# ---------------------------------------------------------------------------

class ULSIF:
    """
    Estimates density ratio w(x) = dP/dmu(x) via least-squares.
    
    Minimizes: (1/2) E_mu[w(x)^2] - E_P[w(x)]
    
    Using kernel basis: w(x) = sum_l alpha_l * k(x, c_l)
    
    Parameters:
        n_centers: number of kernel centers (sampled from combined data)
        sigma: RBF bandwidth (None = median heuristic)
        lam: regularization strength (None = cross-validated)
        lam_candidates: grid for CV
    """
    
    def __init__(self, n_centers=200, sigma=None, lam=None,
                 lam_candidates=None):
        self.n_centers = n_centers
        self.sigma = sigma
        self.lam = lam
        self.lam_candidates = lam_candidates or np.logspace(-4, 1, 12)
        
        self.alpha_ = None
        self.centers_ = None
        self.sigma_ = None
        self.lam_ = None
    
    def fit(self, X_P, X_mu):
        """
        Fit density ratio w = dP/dmu.
        
        Args:
            X_P: samples from P, shape (n_P, d)
            X_mu: samples from mu, shape (n_mu, d)
        """
        X_P = np.atleast_2d(X_P)
        X_mu = np.atleast_2d(X_mu)
        n_P, d = X_P.shape
        n_mu = X_mu.shape[0]
        
        # Select centers from combined data
        XY = np.vstack([X_P, X_mu])
        rng = np.random.default_rng(42)
        n_c = min(self.n_centers, len(XY))
        idx = rng.choice(len(XY), n_c, replace=False)
        self.centers_ = XY[idx]
        
        # Bandwidth
        if self.sigma is not None:
            self.sigma_ = self.sigma
        else:
            self.sigma_ = median_heuristic(X_P, X_mu)
        
        # Kernel matrices
        K_mu = rbf_kernel_matrix(X_mu, self.centers_, self.sigma_)  # (n_mu, n_c)
        K_P = rbf_kernel_matrix(X_P, self.centers_, self.sigma_)    # (n_P, n_c)
        
        # H_hat = (1/n_mu) * K_mu^T K_mu
        H_hat = K_mu.T @ K_mu / n_mu
        # h_hat = (1/n_P) * K_P^T 1
        h_hat = K_P.mean(axis=0)
        
        # Cross-validate lambda if not specified
        if self.lam is not None:
            self.lam_ = self.lam
        else:
            self.lam_ = self._cross_validate(K_mu, K_P, H_hat, h_hat,
                                              n_mu, n_P)
        
        # Solve: alpha = (H_hat + lam * I)^{-1} h_hat
        A = H_hat + self.lam_ * np.eye(n_c)
        self.alpha_ = np.linalg.solve(A, h_hat)
        
        return self
    
    def _cross_validate(self, K_mu, K_P, H_hat, h_hat, n_mu, n_P,
                        n_folds=5):
        """LOOCV-approximate cross-validation for lambda selection."""
        best_lam = self.lam_candidates[0]
        best_score = np.inf
        n_c = H_hat.shape[0]
        
        for lam in self.lam_candidates:
            A = H_hat + lam * np.eye(n_c)
            try:
                alpha = np.linalg.solve(A, h_hat)
            except np.linalg.LinAlgError:
                continue
            
            # Score: (1/2) * alpha^T H_hat alpha - h_hat^T alpha + lam/2 * ||alpha||^2
            # (penalized objective; lower is better)
            score = 0.5 * alpha @ H_hat @ alpha - h_hat @ alpha
            
            if score < best_score:
                best_score = score
                best_lam = lam
        
        return best_lam
    
    def predict(self, X):
        """Estimate w(x) = dP/dmu(x) at new points."""
        X = np.atleast_2d(X)
        K = rbf_kernel_matrix(X, self.centers_, self.sigma_)
        w = K @ self.alpha_
        return w
    
    def predict_deviation(self, X):
        """Estimate u(x) = w(x) - 1."""
        return self.predict(X) - 1.0


# ---------------------------------------------------------------------------
# chi^2 estimation from density ratios
# ---------------------------------------------------------------------------

def estimate_chi2(X_P, X_mu, w_hat_on_mu=None, model=None):
    """
    Estimate chi^2(P||mu) = E_mu[(w-1)^2] = E_mu[w^2] - 1.
    
    Can use either pre-computed w values on mu samples, or a fitted model.
    """
    if w_hat_on_mu is not None:
        u_hat = w_hat_on_mu - 1.0
        return np.mean(u_hat ** 2)
    elif model is not None:
        w_hat = model.predict(X_mu)
        u_hat = w_hat - 1.0
        return np.mean(u_hat ** 2)
    else:
        raise ValueError("Provide either w_hat_on_mu or model")


def estimate_observable_chi2(X_P, X_mu, S_P, S_mu, n_bins=20):
    """
    Estimate chi^2(P_S||mu_S) from transcript-level density ratios.
    
    For continuous S, bin S and compute histogram-based density ratio.
    For discrete S, compute exact conditional counts.
    
    Args:
        X_P, X_mu: full samples
        S_P: transcript values for P samples, shape (n_P,) or (n_P, d_S)
        S_mu: transcript values for mu samples, shape (n_mu,) or (n_mu, d_S)
        n_bins: number of bins per dimension for continuous S
    """
    S_P = np.atleast_1d(S_P)
    S_mu = np.atleast_1d(S_mu)
    
    if S_P.ndim == 1:
        S_P = S_P.reshape(-1, 1)
        S_mu = S_mu.reshape(-1, 1)
    
    n_P = len(S_P)
    n_mu = len(S_mu)
    d_S = S_P.shape[1]
    
    # For 1D or low-dim S, use histogram approach
    if d_S <= 3:
        # Compute bin edges from combined data
        S_all = np.vstack([S_P, S_mu])
        
        # Multi-dimensional binning
        bin_edges = []
        for dim in range(d_S):
            lo = np.min(S_all[:, dim])
            hi = np.max(S_all[:, dim])
            margin = (hi - lo) * 0.01 + 1e-10
            edges = np.linspace(lo - margin, hi + margin, n_bins + 1)
            bin_edges.append(edges)
        
        # Assign bins
        def assign_bins(S, edges_list):
            bins = np.zeros(len(S), dtype=int)
            strides = [1]
            for dim in range(d_S - 1, 0, -1):
                strides.insert(0, strides[0] * n_bins)
            for dim in range(d_S):
                b = np.digitize(S[:, dim], edges_list[dim]) - 1
                b = np.clip(b, 0, n_bins - 1)
                bins += b * strides[dim]
            return bins
        
        bins_P = assign_bins(S_P, bin_edges)
        bins_mu = assign_bins(S_mu, bin_edges)
        
        # Count per bin
        total_bins = n_bins ** d_S
        counts_P = np.bincount(bins_P, minlength=total_bins).astype(float)
        counts_mu = np.bincount(bins_mu, minlength=total_bins).astype(float)
        
        # Density ratio per bin: w_s = (counts_P/n_P) / (counts_mu/n_mu)
        # chi^2(P_S||mu_S) = sum over s: mu_S(s) * (w_s - 1)^2
        #                   = sum over s: (counts_mu[s]/n_mu) * ((counts_P[s]/n_P)/(counts_mu[s]/n_mu) - 1)^2
        mask = counts_mu > 0
        w_s = np.zeros(total_bins)
        w_s[mask] = (counts_P[mask] / n_P) / (counts_mu[mask] / n_mu)
        mu_s = counts_mu / n_mu
        
        chi2_S = np.sum(mu_s[mask] * (w_s[mask] - 1.0) ** 2)
        return chi2_S
    else:
        # For high-dim S, fit a separate density ratio model on S
        model_S = ULSIF(n_centers=min(100, n_P + n_mu))
        model_S.fit(S_P, S_mu)
        w_S = model_S.predict(S_mu)
        return np.mean((w_S - 1.0) ** 2)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_chi2(X_P, X_mu, n_bootstrap=200, confidence=0.95,
                   n_centers=200):
    """
    Bootstrap confidence interval for chi^2(P||mu).
    
    Returns: (point_estimate, ci_low, ci_high)
    """
    rng = np.random.default_rng(42)
    
    # Point estimate
    model = ULSIF(n_centers=n_centers)
    model.fit(X_P, X_mu)
    chi2_point = estimate_chi2(X_P, X_mu, model=model)
    
    # Bootstrap
    chi2_boots = []
    n_P = len(X_P)
    n_mu = len(X_mu)
    
    for b in range(n_bootstrap):
        idx_P = rng.choice(n_P, n_P, replace=True)
        idx_mu = rng.choice(n_mu, n_mu, replace=True)
        
        m = ULSIF(n_centers=min(n_centers, n_P))
        m.fit(X_P[idx_P], X_mu[idx_mu])
        chi2_b = estimate_chi2(X_P[idx_P], X_mu[idx_mu], model=m)
        chi2_boots.append(chi2_b)
    
    chi2_boots = np.array(chi2_boots)
    alpha = (1 - confidence) / 2
    ci_low = np.quantile(chi2_boots, alpha)
    ci_high = np.quantile(chi2_boots, 1 - alpha)
    
    return chi2_point, ci_low, ci_high


# ---------------------------------------------------------------------------
# Full geometric diagnostic from samples
# ---------------------------------------------------------------------------

def geometric_diagnostic(X_P, X_mu, Phi, S_fn=None, n_centers=200,
                         n_bins=20):
    """
    Complete geometric diagnostic from samples.
    
    Args:
        X_P: samples from P, shape (n_P, d)
        X_mu: samples from mu, shape (n_mu, d)
        Phi: performance functional, callable Phi(X) -> array
        S_fn: transcript function, callable S(X) -> array (optional)
        n_centers: uLSIF centers
        n_bins: bins for observable chi^2
    
    Returns dict with:
        chi2: total chi^2(P||mu)
        chi2_S: observable chi^2(P_S||mu_S) (if S_fn provided)
        waste: chi2 - chi2_S (if S_fn provided)
        eta: capture efficiency chi2_S / chi2
        advantage: E_P[Phi] - E_mu[Phi]
        norm_Phi: ||Phi^o||_2
        norm_proj_u: ||Pi_S u||_2 (approx)
        rho: alignment coefficient (if S_fn provided)
    """
    X_P = np.atleast_2d(X_P)
    X_mu = np.atleast_2d(X_mu)
    
    # Fit density ratio
    model = ULSIF(n_centers=n_centers)
    model.fit(X_P, X_mu)
    
    # Evaluate
    w_on_mu = model.predict(X_mu)
    u_on_mu = w_on_mu - 1.0
    
    w_on_P = model.predict(X_P)
    
    # chi^2(P||mu) = E_mu[u^2]
    chi2 = np.mean(u_on_mu ** 2)
    
    # Performance
    Phi_mu = Phi(X_mu)
    Phi_P = Phi(X_P)
    E_mu_Phi = np.mean(Phi_mu)
    E_P_Phi = np.mean(Phi_P)
    advantage = E_P_Phi - E_mu_Phi
    
    # ||Phi^o||_2 = sqrt(Var_mu[Phi])
    norm_Phi = np.std(Phi_mu)
    
    result = {
        'chi2': chi2,
        'advantage': advantage,
        'norm_Phi': norm_Phi,
        'sqrt_chi2': np.sqrt(chi2),
    }
    
    if S_fn is not None:
        S_P = S_fn(X_P)
        S_mu = S_fn(X_mu)
        
        chi2_S = estimate_observable_chi2(X_P, X_mu, S_P, S_mu, n_bins)
        waste = chi2 - chi2_S
        eta = chi2_S / max(chi2, 1e-15)
        
        # Alignment: advantage / (||Phi^o|| * sqrt(chi2_S))
        bound = norm_Phi * np.sqrt(chi2_S)
        rho = advantage / max(bound, 1e-15)
        rho = np.clip(rho, -1.0, 1.0)
        
        result.update({
            'chi2_S': chi2_S,
            'waste': waste,
            'eta': eta,
            'rho': rho,
            'bound': bound,
        })
    
    return result
