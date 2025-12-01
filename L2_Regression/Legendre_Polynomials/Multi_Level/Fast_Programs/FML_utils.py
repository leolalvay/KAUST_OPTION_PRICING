# This file is based on DM_ML_allcomb.py, DM_utils_ML.py, and DM_ML.py from Amelie's work.
# Refactored for publication-quality code with memory-efficient accumulated normal equations.

"""
Fast Multi-Level Utilities (FML_utils)
======================================

Core utilities for the memory-efficient MLMC implementation using accumulated
normal equations. Instead of storing the full design matrix D (which scales as
M × N × dimV), we accumulate G = D'D and g = D'ψ on-the-fly, reducing memory
from O(MN·dimV) to O(dimV²).

This is analogous to computing ⟨ψ|H|ψ⟩ in quantum mechanics without storing
the full Hamiltonian matrix - we only need the expectation values.

Functions
---------
GBM_paths : Generate correlated Geometric Brownian Motion paths
scalings_l0 : Pilot run to determine basket scaling parameters
tot_degree_poly : Generate total-degree polynomial index pairs
mlmc_level : Compute MLMC level-l coefficients using accumulated normal equations
make_c : Compute full coefficient vector via telescoping sum
make_b_bar : Construct the projected volatility surface function
"""

import numpy as np
import math
from numpy.polynomial.legendre import legvander, legval


# =============================================================================
# Path Generation
# =============================================================================

def GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t):
    """
    Generate correlated Geometric Brownian Motion paths via Euler-Maruyama.
    
    Uses Cholesky decomposition of the correlation matrix to generate
    correlated Brownian increments: dW = Z @ G.T where G G.T = cov_mat.
    
    Parameters
    ----------
    x0 : ndarray, shape (d,) or (d, 1)
        Initial asset prices.
    r : float
        Risk-free interest rate.
    vol : ndarray, shape (d,)
        Volatility vector for each asset.
    cov_mat : ndarray, shape (d, d)
        Correlation matrix (not covariance - volatilities applied separately).
    dt : float
        Time step size.
    N_t : int
        Number of time steps.
    M_t : int
        Number of Monte Carlo paths.
        
    Returns
    -------
    paths : ndarray, shape (M_t, N_t, d)
        Simulated asset price paths.
    """
    G = np.linalg.cholesky(cov_mat)
    sqrtdt = math.sqrt(dt)
    d = len(vol)
    X = np.tile(x0.flatten(), (M_t, 1))
    paths = np.empty((M_t, N_t, d))
    paths[:, 0, :] = X
    for n in range(1, N_t):
        Z = np.random.randn(M_t, d)
        sigma = X * vol
        dW = Z @ G.T
        X = X + r * X * dt + sigma * dW * sqrtdt
        paths[:, n, :] = X
    return paths


# =============================================================================
# Scaling and Basis Functions
# =============================================================================

def scalings_l0(x0, T, h0, r, cov_mat, vol, max_deg, P1, M_0=10000, return_basket=False):
    """
    Pilot run to determine basket scaling parameters for Legendre normalisation.
    
    Runs a coarse simulation to estimate the range [s_min, s_max] of the
    projected basket process. These bounds are used to scale the basket
    values to [-1, 1] for Legendre polynomial evaluation.
    
    Parameters
    ----------
    x0 : ndarray
        Initial asset prices.
    T : float
        Time to maturity.
    h0 : float
        Base time step (coarsest level).
    r : float
        Risk-free rate.
    cov_mat : ndarray
        Correlation matrix.
    vol : ndarray
        Volatility vector.
    max_deg : int
        Maximum polynomial degree (determines finest discretisation).
    P1 : ndarray
        Basket weights (should sum to 1 for normalisation).
    M_0 : int, optional
        Number of pilot paths (default 10000).
    return_basket : bool, optional
        If True, also return the pilot basket paths.
        
    Returns
    -------
    s_min : float
        Lower bound (0.01 percentile).
    s_max : float
        Upper bound (99.99 percentile).
    basket : ndarray, optional
        Pilot basket paths if return_basket=True.
    """
    dt = h0 * 2 ** (-max_deg)
    N_t = int(T / dt)
    basket0 = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_0).dot(P1)
    p1, p99 = np.percentile(basket0.flatten(), [0.01, 99.99])

    if return_basket:
        return p1, p99, basket0
    else:
        return p1, p99


def tot_degree_poly(maxdeg=3):
    """
    Generate index pairs for total-degree polynomial basis.
    
    Returns all pairs (i, j) such that i + j <= maxdeg, representing
    the polynomial basis {t^i * s^j : i + j <= maxdeg}.
    
    Parameters
    ----------
    maxdeg : int
        Maximum total degree.
        
    Returns
    -------
    pairs : list of tuple
        Index pairs [(i, j), ...] with i + j <= maxdeg.
        
    Example
    -------
    >>> tot_degree_poly(2)
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
    """
    return [(i, j) for i in range(maxdeg + 1) for j in range(maxdeg + 1) if (i + j <= maxdeg)]


# =============================================================================
# Accumulated Normal Equations MLMC
# =============================================================================

def mlmc_level(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, s_min0, s_max0, 
               C=80, batch_size=50):
    """
    Compute MLMC level-l coefficients using accumulated normal equations.
    
    This is the memory-efficient core of the fast MLMC implementation.
    Instead of building the full design matrix D ∈ R^{MN × dimV}, we
    accumulate the normal equations incrementally:
    
        G += D_n.T @ D_n    (Gram matrix)
        g += D_n.T @ psi_n  (moment vector)
    
    Then solve G c = g via Cholesky decomposition.
    
    The polynomial degree decreases with level: l_V = max_deg - l,
    implementing the multi-resolution principle where coarse levels
    capture global features and fine levels capture local corrections.
    
    Parameters
    ----------
    x0 : ndarray
        Initial asset prices.
    T : float
        Time to maturity.
    h0 : float
        Base time step.
    l : int
        MLMC level (0 = coarsest).
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
        Basket scaling bounds from pilot run.
    C : int, optional
        Base sample size scaling factor (default 80).
    batch_size : int, optional
        Batch size for memory-efficient processing (default 50).
        
    Returns
    -------
    c_padded : ndarray
        Coefficient vector, zero-padded to dim_max length.
    """
    # Time step sizes for fine and coarse grids
    hl_f = h0 * 2 ** (-l)
    N_f = int(round(T / hl_f))
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    hl_c = 2 * hl_f
    N_c = N_f // 2

    # Polynomial basis dimension decreases with level
    l_V = max_deg - l
    print(f"Level {l}: polynomial degree l_V = {l_V}")
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))

    # Sample size scales with basis dimension squared
    M_l = max(C, int(C * dimV ** 2))

    # Cholesky factor for correlated increments
    G_chol = np.linalg.cholesky(cov_mat)
    d = len(vol)

    # Precompute time Legendre basis
    if l == 0:
        idx = np.arange(N_f)
        t_scal = 2.0 * (idx / max(N_f - 1, 1)) - 1.0
    else:
        idx = np.arange(N_c)
        t_scal = 2.0 * (idx / max(N_c - 1, 1)) - 1.0

    deg_t = max(i for i, _ in pairs)
    VT = legvander(t_scal, deg_t)
    norm_t = np.sqrt((2 * np.arange(deg_t + 1) + 1) / 2)
    VT *= norm_t[None, :]

    # Space Legendre normalisation
    deg_s = max(j for _, j in pairs)
    norm_s = np.sqrt((2 * np.arange(deg_s + 1) + 1) / 2)[None, :]

    # Accumulated normal equation components
    G = np.zeros((dimV, dimV))  # Gram matrix D'D
    g = np.zeros((dimV, 1))     # Moment vector D'ψ

    for m0 in range(0, M_l, batch_size):
        m1 = min(M_l, m0 + batch_size)
        B = m1 - m0

        # Initial state
        X0 = np.tile(x0.flatten(), (B, 1))
        trow0 = VT[0, :]
        s0 = X0 @ P1
        s0_scaled = 2.0 * (s0 - s_min0) / (s_max0 - s_min0) - 1.0
        VS0 = legvander(s0_scaled, deg_s) * norm_s

        # Build D0 row block
        D0 = np.empty((B, dimV), dtype=float)
        for p, (i_t, i_s) in enumerate(pairs):
            D0[:, p] = trow0[i_t] * VS0[:, i_s]

        # Diffusion coefficient b = (σ @ Σ @ σ) / d²
        sigma_f0 = X0 * vol
        b_f0 = ((sigma_f0 @ cov_mat) * sigma_f0).sum(axis=1) / (d ** 2)

        if l == 0:
            psi0 = b_f0.reshape(-1, 1)
        else:
            psi0 = np.zeros((B, 1), dtype=float)

        # Accumulate normal equations
        G += D0.T @ D0
        g += D0.T @ psi0

        # Generate all Brownian increments for this batch
        Z = np.random.randn(B, N_f, d)

        if l == 0:
            # Level 0: single-level (no coarse correction)
            X_f = X0.copy()
            for n in range(1, N_f):
                Z_n = Z[:, n, :]
                sigma = X_f * vol
                dW = (Z_n @ G_chol.T) * math.sqrt(hl_f)
                X_f = X_f + r * X_f * hl_f + sigma * dW

                # Build design matrix row block
                D_n = np.empty((B, dimV), dtype=float)
                trow = VT[n, :]
                s = X_f @ P1
                s_scaled = 2.0 * (s - s_min0) / (s_max0 - s_min0) - 1.0
                VS = legvander(s_scaled, deg_s) * norm_s

                for p, (i1, i2) in enumerate(pairs):
                    D_n[:, p] = trow[i1] * VS[:, i2]

                sigma_f = X_f * vol
                b_f = ((sigma_f @ cov_mat) * sigma_f).sum(axis=1) / (d ** 2)
                psi_n = b_f.reshape(-1, 1)

                G += D_n.T @ D_n
                g += D_n.T @ psi_n
        else:
            # Level l > 0: fine-coarse difference (telescoping)
            X_f = X0.copy()
            X_c = X0.copy()
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

                # One coarse step (shared Brownian increment)
                dWc = dW1 + dW2
                sigma = X_c * vol
                X_c = X_c + r * X_c * hl_c + sigma * dWc

                # Build design matrix row block (at fine grid points)
                D_n = np.empty((B, dimV), dtype=float)
                trow = VT[n, :]
                s = X_f @ P1
                s_scaled = 2.0 * (s - s_min0) / (s_max0 - s_min0) - 1.0
                VS = legvander(s_scaled, deg_s) * norm_s

                for p, (i1, i2) in enumerate(pairs):
                    D_n[:, p] = trow[i1] * VS[:, i2]

                # Telescoping difference: b_f - b_c
                sigma_f = X_f * vol
                b_f = ((sigma_f @ cov_mat) * sigma_f).sum(axis=1) / (d ** 2)
                sigma_c = X_c * vol
                b_c = ((sigma_c @ cov_mat) * sigma_c).sum(axis=1) / (d ** 2)
                psi_n = (b_f - b_c).reshape(-1, 1)

                G += D_n.T @ D_n
                g += D_n.T @ psi_n

    # Solve normal equations via Cholesky
    G_reg = 0.5 * (G + G.T)  # Ensure symmetry
    lam = np.linalg.eigvalsh(G_reg)
    cond_approx = np.sqrt(lam[-1] / max(lam[0], 1e-300))
    print(f"  Condition number ≈ {cond_approx:.2e}")

    L = np.linalg.cholesky(G_reg)
    y = np.linalg.solve(L, g)
    c = np.linalg.solve(L.T, y).ravel()

    # Zero-pad to maximum dimension
    c_padded = np.zeros(dim_max)
    c_padded[:c.shape[0]] = c
    return c_padded


def make_c(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0, 
           C=80, batch_size=50):
    """
    Compute full coefficient vector via MLMC telescoping sum.
    
    Implements c = Σ_{l=0}^{max_deg} c_l where each c_l is computed
    using accumulated normal equations at level l.
    
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
        Maximum polynomial degree (also number of MLMC levels - 1).
    P1 : ndarray
        Basket weights.
    s_min0, s_max0 : float
        Basket scaling bounds.
    C : int, optional
        Base sample size factor.
    batch_size : int, optional
        Batch size for processing.
        
    Returns
    -------
    c : ndarray
        Full coefficient vector for the volatility surface.
    """
    return sum(
        mlmc_level(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, 
                   s_min0, s_max0, C, batch_size)
        for l in range(max_deg + 1)
    )


# =============================================================================
# Volatility Surface Construction
# =============================================================================

def make_b_bar(c, pairs, s_min, s_max, T, max_deg):
    """
    Construct the projected volatility surface function.
    
    Returns a callable b̄(t, S) that evaluates the fitted local volatility
    at any (time, basket) point using the Legendre expansion:
    
        b̄²(t, S) = Σ_p c_p · P_{i_p}(t̃) · P_{j_p}(S̃)
    
    where t̃, S̃ are scaled to [-1, 1].
    
    Parameters
    ----------
    c : ndarray
        Coefficient vector from make_c().
    pairs : list of tuple
        Polynomial index pairs from tot_degree_poly().
    s_min, s_max : float
        Basket scaling bounds.
    T : float
        Time to maturity.
    max_deg : int
        Maximum polynomial degree.
        
    Returns
    -------
    b_bar : callable
        Function b̄(t, S) returning the projected volatility.
    """
    def b_bar(t, S):
        t_c = 2 * t / T - 1
        s_c = 2 * (S - s_min) / (s_max - s_min) - 1
        h = 0.0
        norm = np.sqrt((2 * np.arange(max_deg + 1) + 1) / 2)
        for p, (i1, i2) in enumerate(pairs):
            P_t = legval(t_c, [0] * i1 + [1]) * norm[i1]
            P_s = legval(s_c, [0] * i2 + [1]) * norm[i2]
            h += c[p] * P_t * P_s

        # Warning for negative values (indicates extrapolation issues)
        if np.any(h < 0):
            idx = np.unravel_index(np.argmin(h), h.shape)
            bad_t = np.array(t)[idx] if np.ndim(t) > 0 else t
            bad_S = np.array(S)[idx] if np.ndim(S) > 0 else S
            bad_h = h[idx]
            print(f"Warning: Negative h = {bad_h:.4e} at t = {bad_t:.4f}, S = {bad_S:.4f}")

        return np.sqrt(np.maximum(h, 0.0))  # Enforce non-negativity

    return b_bar


# =============================================================================
# Main Block for Testing
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Basket parameters
    d = 3
    P1 = np.ones(d) / d  # Equal-weighted basket
    r = 0.05
    x0 = np.linspace(225, 275, num=d)[:, np.newaxis]
    vol = np.array([0.2, 0.15, 0.1])
    cov_mat = np.array([[1.0, 0.8, 0.3],
                        [0.8, 1.0, 0.1],
                        [0.3, 0.1, 1.0]])
    T = 1.0
    h0 = 0.01
    max_degrees = [3]

    for max_deg in max_degrees:
        print(f"\n{'='*60}")
        print(f"Running MLMC with max_deg = {max_deg}")
        print(f"{'='*60}")

        dt = h0 * 2 ** (-max_deg)
        N_t = int(T / dt)
        t = np.linspace(0, N_t * dt, N_t)
        pairs = tot_degree_poly(max_deg)

        # Pilot run for scaling
        s_min0, s_max0, basket0 = scalings_l0(
            x0, T, h0, r, cov_mat, vol, max_deg, P1, M_0=10000, return_basket=True
        )
        print(f"Basket range: [{s_min0:.2f}, {s_max0:.2f}]")

        # Compute coefficients
        c = make_c(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0)
        b_bar = make_b_bar(c, pairs, s_min0, s_max0, T, max_deg)

        # Visualisation bounds
        s_min = np.percentile(basket0, 1, axis=0)
        s_max = np.percentile(basket0, 99, axis=0)
        del basket0

        # Create surface plot
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
        ax.plot_surface(TT, SS, bbar, cmap='viridis', rcount=40, ccount=40)
        ax.set_title(f'Projected Volatility Surface (max degree = {max_deg})')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Basket $S$')
        ax.set_zlabel(r'$\bar{b}(t,S)$')
        plt.savefig(f"plots/ML/VolSurf_maxdeg{max_deg}.pdf")
        print(f"Saved: plots/ML/VolSurf_maxdeg{max_deg}.pdf")

    plt.show()
