# This file is based on DM_ML_comb3.py from Amelie's work.
# Refactored for publication-quality code with hierarchical QR decomposition.

"""
Fast Multi-Level with Hierarchical QR (FML_hierarchical_qr)
===========================================================

An alternative MLMC implementation using three-level hierarchical QR
decomposition instead of accumulated normal equations. This approach
has superior numerical stability since QR decomposition doesn't square
the condition number like forming D'D does.

The hierarchy works like a renormalisation group flow:
1. Per-timestep: Q_n, R_n = qr(D_n)
2. Per-batch: Stack R's, compute Q_b, R_b = qr(R_stack)  
3. Final: Q_f, R_f = qr(R_big), then solve R_f c = g_f

This is analogous to the tensor network contraction strategy in quantum
many-body physics - we compress information at each scale before
combining with the next level.

Functions
---------
mlmc_level_qr : MLMC level estimator with hierarchical QR
make_c_qr : Compute coefficients via QR-based telescoping sum
"""

import numpy as np
import math
from numpy.polynomial.legendre import legvander, legval

from FML_utils import (
    scalings_l0,
    tot_degree_poly,
    make_b_bar
)


def mlmc_level_qr(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, s_min0, s_max0,
                  C=80, batch_size=50):
    """
    Compute MLMC level-l coefficients using hierarchical QR decomposition.
    
    Instead of accumulating normal equations G = D'D which squares the
    condition number, this uses a three-level QR hierarchy:
    
    1. Per timestep: Factor each D_n block
    2. Per batch: Combine timestep factors
    3. Final: Combine batch factors and solve
    
    This maintains numerical stability for ill-conditioned problems.
    
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
        Basket scaling bounds.
    C : int, optional
        Base sample size factor (default 80).
    batch_size : int, optional
        Batch size for processing (default 50).
        
    Returns
    -------
    c_padded : ndarray
        Coefficient vector, zero-padded to dim_max length.
    """
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
    print(f"Level {l} (QR): polynomial degree l_V = {l_V}")
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))

    M_l = max(C, int(C * dimV ** 2))

    G_chol = np.linalg.cholesky(cov_mat)
    d = len(vol)

    # Time Legendre basis (at coarse timesteps)
    t = np.arange(N_c) * hl_c
    t_scal = 2 * t / T - 1
    deg_t = max(i for i, _ in pairs)
    VT = legvander(t_scal, deg_t)
    norm_t = np.sqrt((2 * np.arange(deg_t + 1) + 1) / 2)
    VT *= norm_t[None, :]

    # Space Legendre normalisation
    deg_s = max(j for _, j in pairs)
    norm_s = np.sqrt((2 * np.arange(deg_s + 1) + 1) / 2)[None, :]

    # Batch-level QR factors
    R_batches = []
    g_batches = []

    for m0 in range(0, M_l, batch_size):
        m1 = min(M_l, m0 + batch_size)
        B = m1 - m0

        # Per-timestep QR factors for this batch
        R_parts = []
        g_parts = []

        # Initial state
        X0 = np.tile(x0.flatten(), (B, 1))
        trow0 = VT[0, :]
        s0 = X0 @ P1
        s0_scaled = 2.0 * (s0 - s_min0) / (s_max0 - s_min0) - 1.0
        VS0 = legvander(s0_scaled, deg_s) * norm_s

        D0 = np.empty((B, dimV), dtype=float)
        for p, (i_t, i_s) in enumerate(pairs):
            D0[:, p] = trow0[i_t] * VS0[:, i_s]

        sigma_f0 = X0 * vol
        b_f0 = ((sigma_f0 @ cov_mat) * sigma_f0).sum(axis=1) / (d ** 2)
        if l > 0:
            psi0 = np.zeros((B, 1), dtype=float)
        else:
            psi0 = b_f0.reshape(-1, 1)

        # QR for initial timestep
        Q0, R0 = np.linalg.qr(D0, mode='reduced')  # Q0: (B, dimV), R0: (dimV, dimV)
        g0 = Q0.T @ psi0                           # (dimV, 1)
        R_parts.append(R0)
        g_parts.append(g0)

        # Generate Brownian increments
        Z = np.random.randn(B, N_f, d)

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

            # One coarse step (for l > 0)
            if l > 0:
                dWc = dW1 + dW2
                sigma = X_c * vol
                X_c = X_c + r * X_c * hl_c + sigma * dWc

            # Build design matrix block
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
            if l > 0:
                sigma_c = X_c * vol
                b_c = ((sigma_c @ cov_mat) * sigma_c).sum(axis=1) / (d ** 2)
            else:
                b_c = 0.0
            psi_n = (b_f - b_c).reshape(-1, 1)

            # QR for this timestep
            Qn, Rn = np.linalg.qr(D_n, mode='reduced')  # (B, dimV), (dimV, dimV)
            gn = Qn.T @ psi_n                           # (dimV, 1)
            R_parts.append(Rn)
            g_parts.append(gn)

        # Combine timestep factors for this batch (second level QR)
        R_stack = np.vstack(R_parts)  # (N_c * dimV, dimV)
        g_stack = np.vstack(g_parts)  # (N_c * dimV, 1)
        Qb, Rb = np.linalg.qr(R_stack, mode='reduced')  # Rb: (dimV, dimV)
        gb = Qb.T @ g_stack                              # (dimV, 1)

        R_batches.append(Rb)
        g_batches.append(gb)

    # Final QR combination across batches (third level)
    R_big = np.vstack(R_batches)  # (num_batches * dimV, dimV)
    g_big = np.vstack(g_batches)  # (num_batches * dimV, 1)
    Qf, Rf = np.linalg.qr(R_big, mode='reduced')  # Rf: (dimV, dimV)
    gf = Qf.T @ g_big                              # (dimV, 1)

    # Solve Rf c = gf
    c = np.linalg.solve(Rf, gf).ravel()

    # Report condition number
    cond = np.linalg.cond(Rf)
    print(f"  Final R condition number: {cond:.2e}")

    c_padded = np.zeros(dim_max)
    c_padded[:c.shape[0]] = c
    return c_padded


def make_c_qr(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0,
              C=80, batch_size=50):
    """
    Compute full coefficient vector via QR-based telescoping sum.
    
    Implements c = Σ_{l=0}^{max_deg} c_l using hierarchical QR
    decomposition at each level.
    
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
    batch_size : int, optional
        Batch size for processing.
        
    Returns
    -------
    c : ndarray
        Full coefficient vector for the volatility surface.
    """
    return sum(
        mlmc_level_qr(x0, T, h0, l, r, cov_mat, vol, max_deg, P1,
                      s_min0, s_max0, C, batch_size)
        for l in range(max_deg + 1)
    )


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
    max_degrees = [3, 5, 7]

    for max_deg in max_degrees:
        print(f"\n{'='*60}")
        print(f"Running Hierarchical QR MLMC with max_deg = {max_deg}")
        print(f"{'='*60}")

        dt = h0 * 2 ** (-max_deg)
        N_t = int(T / dt)
        t = np.linspace(0, N_t * dt, N_t)
        pairs = tot_degree_poly(max_deg)

        s_min0, s_max0, basket0 = scalings_l0(
            x0, T, h0, r, cov_mat, vol, max_deg, P1, M_0=10000, return_basket=True
        )
        print(f"Basket range: [{s_min0:.2f}, {s_max0:.2f}]")

        c = make_c_qr(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0)
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
        ax.plot_surface(TT, SS, bbar, cmap='viridis', rcount=40, ccount=40)
        ax.set_title(f'Hierarchical QR (max degree = {max_deg})')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('Basket $S$')
        ax.set_zlabel(r'$\bar{b}(t,S)$')
        plt.savefig(f"plots/ML/VolSurf_QR_maxdeg{max_deg}.pdf")
        print(f"Saved: plots/ML/VolSurf_QR_maxdeg{max_deg}.pdf")

    plt.show()
