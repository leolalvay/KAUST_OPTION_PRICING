# This file is based on DM_SL_comb.py from Amelie's work.
# Refactored for publication-quality code with memory-efficient single-level MC.

"""
Fast Single-Level Monte Carlo (FML_single_level)
=================================================

Single-level Monte Carlo baseline for comparison with MLMC methods.
Uses the same accumulated normal equations approach as FML_utils,
but runs only at the finest discretisation level without the
telescoping sum decomposition.

This serves as the reference implementation: MLMC should match
single-level results while achieving lower computational cost
through variance reduction via the telescoping structure.

Functions
---------
single_level : Compute coefficients via single-level MC with accumulated normal equations
"""

import numpy as np
import math
from numpy.polynomial.legendre import legvander, legval

from FML_utils import (
    scalings_l0,
    tot_degree_poly,
    make_b_bar
)


def single_level(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0,
                 C=80, batch_size=50):
    """
    Compute volatility surface coefficients via single-level Monte Carlo.
    
    Uses the same accumulated normal equations approach as mlmc_level(),
    but runs at the finest discretisation (h = h0 * 2^{-max_deg}) and
    fits all polynomial degrees simultaneously.
    
    This is the reference implementation - MLMC should produce equivalent
    results with reduced variance per unit computational cost.
    
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
        Basket scaling bounds from pilot run.
    C : int, optional
        Base sample size scaling factor (default 80).
    batch_size : int, optional
        Batch size for memory-efficient processing (default 50).
        
    Returns
    -------
    c : ndarray
        Coefficient vector for the volatility surface.
    """
    # Finest discretisation
    hl_f = h0 * 2 ** (-max_deg)
    N_f = int(round(T / hl_f))
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f

    # Full polynomial basis
    pairs = tot_degree_poly(max_deg)
    dimV = len(pairs)

    # Sample size scales with basis dimension
    M_l = max(C, int(C * dimV ** 2))

    # Cholesky factor for correlated increments
    G_chol = np.linalg.cholesky(cov_mat)
    d = len(vol)

    # Time Legendre basis
    t = np.arange(N_f) * hl_f
    t_scal = 2 * t / T - 1
    deg_t = max(i for i, _ in pairs)
    VT = legvander(t_scal, deg_t)
    norm_t = np.sqrt((2 * np.arange(deg_t + 1) + 1) / 2)
    VT *= norm_t[None, :]

    # Space Legendre normalisation
    deg_s = max(j for _, j in pairs)
    norm_s = np.sqrt((2 * np.arange(deg_s + 1) + 1) / 2)[None, :]

    # Accumulated normal equations
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

        D0 = np.empty((B, dimV), dtype=float)
        for p, (i_t, i_s) in enumerate(pairs):
            D0[:, p] = trow0[i_t] * VS0[:, i_s]

        # Diffusion coefficient at t=0
        sigma_f0 = X0 * vol
        b_f0 = ((sigma_f0 @ cov_mat) * sigma_f0).sum(axis=1) / (d ** 2)
        psi0 = b_f0.reshape(-1, 1)

        G += D0.T @ D0
        g += D0.T @ psi0

        # Generate Brownian increments
        Z = np.random.randn(B, N_f, d)

        X_f = X0.copy()
        for n in range(1, N_f):
            Z_n = Z[:, n, :]
            sigma = X_f * vol
            dW1 = (Z_n @ G_chol.T) * math.sqrt(hl_f)
            X_f = X_f + r * X_f * hl_f + sigma * dW1

            # Build design matrix row block
            D_n = np.empty((B, dimV), dtype=float)
            trow = VT[n, :]
            s = X_f @ P1
            s_scaled = 2.0 * (s - s_min0) / (s_max0 - s_min0) - 1.0
            VS = legvander(s_scaled, deg_s) * norm_s

            for p, (i1, i2) in enumerate(pairs):
                D_n[:, p] = trow[i1] * VS[:, i2]

            # Diffusion coefficient
            sigma_f = X_f * vol
            b_f = ((sigma_f @ cov_mat) * sigma_f).sum(axis=1) / (d ** 2)
            psi_n = b_f.reshape(-1, 1)

            G += D_n.T @ D_n
            g += D_n.T @ psi_n

    # Solve via Cholesky
    G_reg = 0.5 * (G + G.T)
    lam = np.linalg.eigvalsh(G_reg)
    cond_approx = np.sqrt(lam[-1] / max(lam[0], 1e-300))
    print(f"Single-Level: Condition number ≈ {cond_approx:.2e}")

    L = np.linalg.cholesky(G_reg)
    y = np.linalg.solve(L, g)
    c = np.linalg.solve(L.T, y).ravel()

    c_padded = np.zeros(dimV)
    c_padded[:c.shape[0]] = c
    return c_padded


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
    max_degrees = [1, 2, 3]

    for max_deg in max_degrees:
        print(f"\n{'='*60}")
        print(f"Running Single-Level with max_deg = {max_deg}")
        print(f"{'='*60}")

        dt = h0 * 2 ** (-max_deg)
        N_t = int(T / dt)
        t = np.linspace(0, N_t * dt, N_t)
        pairs = tot_degree_poly(max_deg)

        s_min0, s_max0, basket0 = scalings_l0(
            x0, T, h0, r, cov_mat, vol, max_deg, P1, M_0=10000, return_basket=True
        )
        print(f"Basket range: [{s_min0:.2f}, {s_max0:.2f}]")

        c_SL = single_level(x0, T, h0, r, cov_mat, vol, max_deg, P1,
                            s_min0, s_max0, C=80, batch_size=50)
        b_bar = make_b_bar(c_SL, pairs, s_min0, s_max0, T, max_deg)

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
        ax.plot_surface(TT, SS, bbar, cmap='viridis', rcount=40, ccount=40)
        ax.set_title(f'Single-Level Volatility Surface (max degree = {max_deg})')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Basket $S$')
        ax.set_zlabel(r'$\bar{b}(t,S)$')
        plt.savefig(f"plots/SL/VolSurf_maxdeg{max_deg}.pdf")
        print(f"Saved: plots/SL/VolSurf_maxdeg{max_deg}.pdf")

    plt.show()
