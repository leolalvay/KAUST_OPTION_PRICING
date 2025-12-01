# This file is based on DM_ML_OT_comb.py and ML_OTmap_comb.py from Amelie's work.
# Refactored for publication-quality code with memory-efficient OT-enhanced MLMC.

"""
Fast Multi-Level Optimal Transport (FML_optimal_transport)
==========================================================

Implements Gaussian-Brenier optimal transport maps for variance reduction
in the Multi-Level Monte Carlo framework. The key insight is that for
log-normal distributions (which GBM produces), the optimal coupling between
fine and coarse discretisations can be computed in closed form.

The Brenier map T: R^d → R^d that pushes forward N(μ_f, C_f) to N(μ_c, C_c)
while minimising the transport cost is given by:

    T(x) = μ_c + A (x - μ_f)

where A = C_f^{-1/2} (C_f^{1/2} C_c C_f^{1/2})^{1/2} C_f^{-1/2}

This is the financial mathematics analogue of adiabatic transformations
in phase space - we're finding the "smoothest" way to transform fine-scale
realisations to their coarse-scale counterparts.

Classes
-------
GaussianBrenierMap : Implements the optimal transport map for Gaussian distributions

Functions
---------
identity_map : Identity function for l=0
logpaths_maps : Compute OT maps from pilot simulations
mlmc_level_OT : MLMC level estimator with OT coupling
make_c_OT : Compute coefficients via OT-enhanced telescoping sum
"""

import numpy as np
import math
from numpy.polynomial.legendre import legvander, legval

from FML_utils import (
    scalings_l0,
    tot_degree_poly,
    mlmc_level,
    make_b_bar
)


# =============================================================================
# Gaussian-Brenier Optimal Transport
# =============================================================================

class GaussianBrenierMap:
    """
    Optimal transport map between two Gaussian distributions.
    
    Implements Brenier's theorem for Gaussian measures: the unique
    optimal transport map (minimising L² Wasserstein distance) from
    N(μ_f, C_f) to N(μ_c, C_c) is an affine map T(x) = μ_c + A(x - μ_f).
    
    The matrix A is computed via the formula:
        A = C_f^{-1/2} (C_f^{1/2} C_c C_f^{1/2})^{1/2} C_f^{-1/2}
    
    We use eigendecomposition for numerical stability, avoiding
    explicit matrix square roots.
    
    Parameters
    ----------
    mu_f : ndarray, shape (d,)
        Mean of the fine (source) distribution.
    C_f : ndarray, shape (d, d)
        Covariance of the fine distribution.
    mu_c : ndarray, shape (d,)
        Mean of the coarse (target) distribution.
    C_c : ndarray, shape (d, d)
        Covariance of the coarse distribution.
        
    Attributes
    ----------
    A : ndarray, shape (d, d)
        The linear transformation matrix.
    """
    
    def __init__(self, mu_f, C_f, mu_c, C_c):
        self.mu_f = np.asarray(mu_f)
        self.mu_c = np.asarray(mu_c)
        self.C_f = np.asarray(C_f)
        self.C_c = np.asarray(C_c)

        # Eigendecomposition of C_f for numerical stability
        eig_f, U_f = np.linalg.eigh(self.C_f)
        sqrt_C_f = U_f @ np.diag(np.sqrt(eig_f)) @ U_f.T
        invsqrt_C_f = U_f @ np.diag(1.0 / np.sqrt(eig_f)) @ U_f.T

        # Middle matrix M = C_f^{1/2} C_c C_f^{1/2}
        M = sqrt_C_f @ self.C_c @ sqrt_C_f

        # Eigendecomposition of M
        eig_M, U_M = np.linalg.eigh(M)
        sqrt_M = U_M @ np.diag(np.sqrt(eig_M)) @ U_M.T

        # Brenier map: A = C_f^{-1/2} M^{1/2} C_f^{-1/2}
        self.A = invsqrt_C_f @ sqrt_M @ invsqrt_C_f

    def map(self, x):
        """Apply the optimal transport map."""
        x = np.asarray(x)
        return self.mu_c + (x - self.mu_f) @ self.A.T

    def __call__(self, x):
        """Callable interface for the map."""
        return self.map(x)


def identity_map(x):
    """Identity map for time n=0 (no transport needed)."""
    return x


def logpaths_maps(x0, T, h0, l, r, cov_mat, vol, M_samp=20, batch_size=5):
    """
    Compute optimal transport maps from pilot simulations.
    
    Runs a small pilot simulation to estimate the mean and covariance
    of log(X_f) and log(X_c) at each coarse timestep. These statistics
    are then used to construct Gaussian-Brenier maps.
    
    Working in log-space is crucial: GBM produces log-normal distributions,
    so log(X) is Gaussian, making the Brenier theory directly applicable.
    
    Parameters
    ----------
    x0 : ndarray
        Initial asset prices.
    T : float
        Time to maturity.
    h0 : float
        Base time step.
    l : int
        MLMC level (must be >= 1 for fine/coarse pair).
    r : float
        Risk-free rate.
    cov_mat : ndarray
        Correlation matrix.
    vol : ndarray
        Volatility vector.
    M_samp : int, optional
        Number of pilot paths (default 20).
    batch_size : int, optional
        Batch size for processing (default 5).
        
    Returns
    -------
    maps : list of callable
        List of length N_c, where maps[n] transforms log(X_f[n]) to log(X_c[n]).
        maps[0] is the identity.
        
    Raises
    ------
    ValueError
        If l < 1 (need fine/coarse pair for OT).
    """
    # Validation
    if l < 0:
        raise ValueError(f"Level l must be non-negative, got l={l}")
    if l == 0:
        raise ValueError("Level l must be >= 1 to form a fine/coarse pair.")

    x0 = np.asarray(x0).reshape(-1)
    vol = np.asarray(vol).reshape(-1)
    d = len(vol)
    if x0.shape[0] != d:
        raise ValueError("x0 and vol must have same dimension d.")
    cov_mat = np.asarray(cov_mat)
    if cov_mat.shape != (d, d):
        raise ValueError("cov_mat must be (d, d).")

    # Time discretisation
    hl_f = h0 * 2 ** (-l)
    N_f = int(round(T / hl_f))
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    hl_c = 2 * hl_f
    N_c = N_f // 2
    Gchol = np.linalg.cholesky(cov_mat)

    # Accumulators for log-space statistics at coarse times
    sum_f = np.zeros((N_c, d))
    sum_c = np.zeros((N_c, d))
    sum2_f = np.zeros((N_c, d, d))
    sum2_c = np.zeros((N_c, d, d))
    count = 0

    for m0 in range(0, M_samp, batch_size):
        m1 = min(M_samp, m0 + batch_size)
        B = m1 - m0
        Xf = np.tile(x0, (B, 1))
        Xc = Xf.copy()

        # Initial state (n=0)
        logf = np.log(Xf)
        logc = np.log(Xc)
        sum_f[0] += logf.sum(axis=0)
        sum_c[0] += logc.sum(axis=0)
        sum2_f[0] += logf.T @ logf
        sum2_c[0] += logc.T @ logc

        Z = np.random.randn(B, N_f, d)
        for n in range(1, N_c):
            # Two fine steps
            Z0 = Z[:, 2 * (n - 1), :]
            dW0 = (Z0 @ Gchol.T) * math.sqrt(hl_f)
            Xf = Xf + r * Xf * hl_f + (Xf * vol) * dW0

            Z1 = Z[:, 2 * (n - 1) + 1, :]
            dW1 = (Z1 @ Gchol.T) * math.sqrt(hl_f)
            Xf = Xf + r * Xf * hl_f + (Xf * vol) * dW1

            # One coarse step
            dWc = dW0 + dW1
            Xc = Xc + r * Xc * hl_c + (Xc * vol) * dWc

            # Accumulate log-space statistics
            logf = np.log(Xf)
            logc = np.log(Xc)
            sum_f[n] += logf.sum(axis=0)
            sum_c[n] += logc.sum(axis=0)
            sum2_f[n] += logf.T @ logf
            sum2_c[n] += logc.T @ logc

        count += B

    # Compute means and covariances
    mu_f = sum_f / count
    mu_c = sum_c / count
    C_f = np.empty((N_c, d, d))
    C_c = np.empty((N_c, d, d))
    for n in range(N_c):
        C_f[n] = (sum2_f[n] / count) - np.outer(mu_f[n], mu_f[n])
        C_c[n] = (sum2_c[n] / count) - np.outer(mu_c[n], mu_c[n])

    # Construct maps
    maps = [identity_map]
    for n in range(1, N_c):
        maps.append(GaussianBrenierMap(mu_f[n], C_f[n], mu_c[n], C_c[n]))
    return maps


# =============================================================================
# OT-Enhanced MLMC Level Estimator
# =============================================================================

def mlmc_level_OT(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, s_min0, s_max0,
                  C=80, batch_size_ML=50, batch_size_maps=10):
    """
    Compute MLMC level-l coefficients with optimal transport coupling.
    
    For l >= 1, uses Gaussian-Brenier maps to couple fine and coarse paths
    instead of the standard Brownian bridge coupling. This achieves
    provably optimal variance reduction for the telescoping differences.
    
    The key difference from mlmc_level():
    - Standard: X_c evolves with shared Brownian increments dW1 + dW2
    - OT: X_c = exp(T_n(log(X_f))) where T_n is the Brenier map
    
    Parameters
    ----------
    x0 : ndarray
        Initial asset prices.
    T : float
        Time to maturity.
    h0 : float
        Base time step.
    l : int
        MLMC level (must be >= 1).
    r : float
        Risk-free rate.
    cov_mat : ndarray
        Correlation matrix.
    vol : ndarray
        Volatility vector.
    max_deg : int
        Maximum polynomial degree.
    P1 : ndarray
        Basket weights.
    s_min0, s_max0 : float
        Basket scaling bounds.
    C : int, optional
        Base sample size factor.
    batch_size_ML : int, optional
        Batch size for MLMC paths.
    batch_size_maps : int, optional
        Batch size for computing OT maps.
        
    Returns
    -------
    c_padded : ndarray
        Coefficient vector, zero-padded to dim_max length.
    """
    # Compute OT maps from pilot simulation
    maps = logpaths_maps(x0, T, h0, l, r, cov_mat, vol, 
                         M_samp=50, batch_size=batch_size_maps)

    # Time discretisation
    hl_f = h0 * 2 ** (-l)
    N_f = int(round(T / hl_f))
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    hl_c = 2 * hl_f
    N_c = N_f // 2

    # Polynomial basis
    l_V = max_deg - l
    print(f"Level {l} (OT): polynomial degree l_V = {l_V}")
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))

    M_l = max(C, int(C * dimV ** 2))

    G_chol = np.linalg.cholesky(cov_mat)
    d = len(vol)

    # Time Legendre basis
    idx = np.arange(N_c)
    t_scal = 2.0 * (idx / max(N_c - 1, 1)) - 1.0
    deg_t = max(i for i, _ in pairs)
    VT = legvander(t_scal, deg_t)
    norm_t = np.sqrt((2 * np.arange(deg_t + 1) + 1) / 2)
    VT *= norm_t[None, :]

    # Space Legendre normalisation
    deg_s = max(j for _, j in pairs)
    norm_s = np.sqrt((2 * np.arange(deg_s + 1) + 1) / 2)[None, :]

    # Accumulated normal equations
    G = np.zeros((dimV, dimV))
    g = np.zeros((dimV, 1))

    for m0 in range(0, M_l, batch_size_ML):
        m1 = min(M_l, m0 + batch_size_ML)
        B = m1 - m0

        # Initial state
        X0 = np.tile(x0.flatten(), (B, 1))
        trow0 = VT[0, :]
        s0 = X0 @ P1
        s0_scaled = 2.0 * (s0 - s_min0) / (s_max0 - s_min0) - 1.0
        VS0 = legvander(s0_scaled, deg_s) * norm_s

        D0 = np.empty((B, dimV), dtype=float)
        for p, (i_t, i_s) in enumerate(pairs):
            D0[:, p] = trow0[i_t] * VS0[:, i_s]
        psi0 = np.zeros((B, 1), dtype=float)

        G += D0.T @ D0
        g += D0.T @ psi0

        # Generate Brownian increments
        Z = np.random.randn(B, N_f, d)

        X_f = X0.copy()
        for n in range(1, N_c):
            # Two fine steps
            Z_n1 = Z[:, 2 * (n - 1), :]
            sigma = X_f * vol
            dW1 = (Z_n1 @ G_chol.T) * math.sqrt(hl_f)
            X_f = X_f + r * X_f * hl_f + sigma * dW1

            Z_n2 = Z[:, 2 * n - 1, :]
            sigma = X_f * vol
            dW2 = (Z_n2 @ G_chol.T) * math.sqrt(hl_f)
            X_f = X_f + r * X_f * hl_f + sigma * dW2

            # OT coupling: map fine to coarse in log-space
            logX_f = np.log(X_f)
            logX_c = maps[n](logX_f)
            X_c = np.exp(logX_c)

            # Build design matrix row block
            D_n = np.empty((B, dimV), dtype=float)
            trow = VT[n, :]
            s = X_f @ P1
            s_scaled = 2.0 * (s - s_min0) / (s_max0 - s_min0) - 1.0
            VS = legvander(s_scaled, deg_s) * norm_s

            for p, (i1, i2) in enumerate(pairs):
                D_n[:, p] = trow[i1] * VS[:, i2]

            # Telescoping difference
            sigma_f = X_f * vol
            b_f = ((sigma_f @ cov_mat) * sigma_f).sum(axis=1) / (d ** 2)
            sigma_c = X_c * vol
            b_c = ((sigma_c @ cov_mat) * sigma_c).sum(axis=1) / (d ** 2)
            psi_n = (b_f - b_c).reshape(-1, 1)

            G += D_n.T @ D_n
            g += D_n.T @ psi_n

    # Solve via Cholesky
    G_reg = 0.5 * (G + G.T)
    lam = np.linalg.eigvalsh(G_reg)
    cond_approx = np.sqrt(lam[-1] / max(lam[0], 1e-300))
    print(f"  Condition number ≈ {cond_approx:.2e}")

    L = np.linalg.cholesky(G_reg)
    y = np.linalg.solve(L, g)
    c = np.linalg.solve(L.T, y).ravel()

    c_padded = np.zeros(dim_max)
    c_padded[:c.shape[0]] = c
    return c_padded


def make_c_OT(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0,
              C=80, batch_size_ML=50, batch_size_maps=10):
    """
    Compute full coefficient vector via OT-enhanced telescoping sum.
    
    Uses standard MLMC for level 0 (no OT needed), then OT-enhanced
    estimators for levels 1 through max_deg.
    
    Parameters
    ----------
    x0 : ndarray
        Initial asset prices.
    T : float
        Time to maturity.
    h0 : float
        Base time step.
    r : float
        Risk-free rate.
    cov_mat : ndarray
        Correlation matrix.
    vol : ndarray
        Volatility vector.
    max_deg : int
        Maximum polynomial degree.
    P1 : ndarray
        Basket weights.
    s_min0, s_max0 : float
        Basket scaling bounds.
    C : int, optional
        Base sample size factor.
    batch_size_ML : int, optional
        Batch size for MLMC paths.
    batch_size_maps : int, optional
        Batch size for computing OT maps.
        
    Returns
    -------
    c : ndarray
        Full coefficient vector for the volatility surface.
    """
    # Level 0: standard MLMC (no coarse, so no OT)
    c_0 = mlmc_level(x0, T, h0, 0, r, cov_mat, vol, max_deg, P1, 
                     s_min0, s_max0, C, batch_size=batch_size_ML)
    
    # Levels 1 to max_deg: OT-enhanced
    c_higher = sum(
        mlmc_level_OT(x0, T, h0, l, r, cov_mat, vol, max_deg, P1,
                      s_min0, s_max0, C, batch_size_ML, batch_size_maps)
        for l in range(1, max_deg + 1)
    )
    
    return c_0 + c_higher


# =============================================================================
# Main Block for Testing
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Basket parameters
    d = 3
    P1 = np.ones(d) / d
    r = 0.05
    x0 = np.linspace(225, 275, num=d)[:, np.newaxis]
    vol = np.array([0.2, 0.15, 0.1])
    cov_mat = np.array([[1.0, 0.8, 0.3],
                        [0.8, 1.0, 0.1],
                        [0.3, 0.1, 1.0]])
    T = 1.0
    h0 = 0.01
    max_degrees = [2, 3]

    for max_deg in max_degrees:
        print(f"\n{'='*60}")
        print(f"Running OT-MLMC with max_deg = {max_deg}")
        print(f"{'='*60}")

        dt = h0 * 2 ** (-max_deg)
        N_t = int(T / dt)
        t = np.linspace(0, N_t * dt, N_t)
        pairs = tot_degree_poly(max_deg)

        s_min0, s_max0, basket0 = scalings_l0(
            x0, T, h0, r, cov_mat, vol, max_deg, P1, M_0=10000, return_basket=True
        )
        print(f"Basket range: [{s_min0:.2f}, {s_max0:.2f}]")

        c = make_c_OT(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0)
        b_bar = make_b_bar(c, pairs, s_min0, s_max0, T, max_deg)

        s_min = np.percentile(basket0, 1, axis=0)
        s_max = np.percentile(basket0, 99, axis=0)
        del basket0

        K, L = 50, 150
        t_1 = np.linspace(0, t[-1], K)
        idx = np.searchsorted(t, t_1)

        TT = np.zeros((L, K))
        SS = np.zeros((L, K))
        for j, ti in enumerate(idx):
            TT[:, j] = t_1[j]
            SS[:, j] = np.linspace(s_min[ti], s_max[ti], L)

        bbar = b_bar(TT, SS)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(TT, SS, bbar, rcount=40, ccount=40,
                          color=f'C{max_deg}', linewidth=0.5)
        ax.set_title(f'OT-MLMC Volatility Surface (max degree = {max_deg})')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Basket $S$')
        ax.set_zlabel(r'$\bar{b}(t,S)$')
        plt.savefig(f"plots/ML/VolSurf_OT_maxdeg{max_deg}.pdf")
        print(f"Saved: plots/ML/VolSurf_OT_maxdeg{max_deg}.pdf")

    plt.show()
