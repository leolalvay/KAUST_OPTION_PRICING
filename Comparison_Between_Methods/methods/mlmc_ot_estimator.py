"""
MLMC with Optimal Transport for Volatility Surface Estimation
==============================================================

Production-grade implementation consolidating the Fast Multi-Level (FML)
framework with Gaussian-Brenier optimal transport coupling.

This module provides everything needed to estimate projected volatility
surfaces for American basket option pricing using MLMC + OT methods.

Key Features
------------
- Memory-efficient accumulated normal equations: O(dimV²) instead of O(MN×dimV)
- Multi-resolution polynomial degrees: coarse levels capture structure, fine levels refine
- Gaussian-Brenier optimal transport: provably optimal coupling for variance reduction
- Cholesky solver: efficient solution of the normal equations

Mathematical Background
-----------------------
For a d-dimensional basket with weights P₁, Gyöngy's Lemma projects the dynamics
onto a 1D process with volatility b(t,S). We estimate b² via polynomial regression:

    b²(t,S) ≈ Σₚ cₚ Pᵢ₁(t/T) Pᵢ₂((S-S_min)/(S_max-S_min))

where Pₖ are orthonormalised Legendre polynomials.

MLMC uses the telescoping sum identity:
    E[Y_L] = E[Y₀] + Σₗ₌₁ᴸ E[Yₗ - Yₗ₋₁]

with decreasing variance for the differences when paths are coupled.

Optimal transport provides the theoretically optimal coupling by computing
Gaussian-Brenier maps in log-space (since GBM produces log-normal distributions).

References
----------
- Bayer, Häppölä, Tempone (2017): Markovian projection for American basket options
- Giles (2008, 2015): Multi-level Monte Carlo methods
- Brenier (1991): Optimal transport for Gaussian measures

Author: Wadoud Charbak (KAUST Internship)
Based on: Amelie's FML implementation, refactored for Comparison framework
"""

import numpy as np
import math
import time
from typing import Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
from numpy.polynomial.legendre import legvander

from .common import VolatilitySurfaceResult


# =============================================================================
# SECTION 1: Path Generation
# =============================================================================

def GBM_paths(x0: np.ndarray, r: float, vol: np.ndarray, cov_mat: np.ndarray,
              dt: float, N_t: int, M_t: int) -> np.ndarray:
    """
    Generate correlated Geometric Brownian Motion paths via Euler-Maruyama.
    
    Uses Cholesky decomposition of the correlation matrix to generate
    correlated Brownian increments: dW = Z @ G.T where G G.T = cov_mat.
    
    Parameters
    ----------
    x0 : ndarray, shape (d,)
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
        
    Notes
    -----
    The GBM SDE is: dX_i = r X_i dt + σ_i X_i dW_i
    with correlated Brownian motions: E[dW_i dW_j] = ρ_ij dt
    """
    G = np.linalg.cholesky(cov_mat)
    sqrtdt = math.sqrt(dt)
    d = len(vol)
    
    X = np.tile(np.asarray(x0).flatten(), (M_t, 1))
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
# SECTION 2: Polynomial Basis
# =============================================================================

def tot_degree_poly(maxdeg: int = 3) -> list:
    """
    Generate index pairs for total-degree polynomial basis.
    
    Returns all pairs (i, j) such that i + j <= maxdeg, representing
    the polynomial basis {t^i × s^j : i + j <= maxdeg}.
    
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
    return [(i, j) for i in range(maxdeg + 1) 
            for j in range(maxdeg + 1) if (i + j <= maxdeg)]


# =============================================================================
# SECTION 3: Domain Estimation
# =============================================================================

def estimate_domain(x0: np.ndarray, T: float, h0: float, r: float,
                    cov_mat: np.ndarray, vol: np.ndarray, max_deg: int,
                    P1: np.ndarray, M_pilot: int = 10000) -> Tuple[float, float]:
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
    M_pilot : int, optional
        Number of pilot paths (default 10000).
        
    Returns
    -------
    s_min : float
        Lower bound (0.01 percentile).
    s_max : float
        Upper bound (99.99 percentile).
    """
    dt = h0 * 2 ** (-max_deg)
    N_t = int(round(T / dt))
    
    paths = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_pilot)
    basket = paths @ P1
    
    s_min, s_max = np.percentile(basket.flatten(), [0.01, 99.99])
    
    return float(s_min), float(s_max)


# =============================================================================
# SECTION 4: Gaussian-Brenier Optimal Transport
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
        
    Notes
    -----
    Physics analogy: This is like constructing a canonical transformation
    between two phase space distributions that minimises the action.
    The Brenier map is the unique gradient of a convex function achieving
    this optimal transport.
    """
    
    def __init__(self, mu_f: np.ndarray, C_f: np.ndarray,
                 mu_c: np.ndarray, C_c: np.ndarray):
        self.mu_f = np.asarray(mu_f)
        self.mu_c = np.asarray(mu_c)
        self.C_f = np.asarray(C_f)
        self.C_c = np.asarray(C_c)
        
        # Regularise covariances for numerical stability
        eps = 1e-10
        C_f_reg = self.C_f + eps * np.eye(len(mu_f))
        C_c_reg = self.C_c + eps * np.eye(len(mu_c))
        
        # Eigendecomposition of C_f for numerical stability
        eig_f, U_f = np.linalg.eigh(C_f_reg)
        eig_f = np.maximum(eig_f, eps)  # Ensure positive
        sqrt_C_f = U_f @ np.diag(np.sqrt(eig_f)) @ U_f.T
        invsqrt_C_f = U_f @ np.diag(1.0 / np.sqrt(eig_f)) @ U_f.T
        
        # Middle matrix M = C_f^{1/2} C_c C_f^{1/2}
        M = sqrt_C_f @ C_c_reg @ sqrt_C_f
        
        # Eigendecomposition of M
        eig_M, U_M = np.linalg.eigh(M)
        eig_M = np.maximum(eig_M, eps)  # Ensure positive
        sqrt_M = U_M @ np.diag(np.sqrt(eig_M)) @ U_M.T
        
        # Brenier map: A = C_f^{-1/2} M^{1/2} C_f^{-1/2}
        self.A = invsqrt_C_f @ sqrt_M @ invsqrt_C_f
    
    def map(self, x: np.ndarray) -> np.ndarray:
        """Apply the optimal transport map."""
        x = np.asarray(x)
        return self.mu_c + (x - self.mu_f) @ self.A.T
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Callable interface for the map."""
        return self.map(x)


def identity_map(x: np.ndarray) -> np.ndarray:
    """Identity map for time n=0 (no transport needed)."""
    return x


def compute_ot_maps(x0: np.ndarray, T: float, h0: float, level: int,
                    r: float, cov_mat: np.ndarray, vol: np.ndarray,
                    M_pilot: int = 500, batch_size: int = 50) -> list:
    """
    Compute optimal transport maps from pilot simulations.
    
    Runs a pilot simulation to estimate the mean and covariance of
    log(X_f) and log(X_c) at each coarse timestep. These statistics
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
    level : int
        MLMC level (must be >= 1 for fine/coarse pair).
    r : float
        Risk-free rate.
    cov_mat : ndarray
        Correlation matrix.
    vol : ndarray
        Volatility vector.
    M_pilot : int, optional
        Number of pilot paths (default 500).
    batch_size : int, optional
        Batch size for processing (default 50).
        
    Returns
    -------
    maps : list of callable
        List of length N_c, where maps[n] transforms log(X_f[n]) to log(X_c[n]).
        maps[0] is the identity.
        
    Raises
    ------
    ValueError
        If level < 1 (need fine/coarse pair for OT).
    """
    if level < 1:
        raise ValueError(f"Level must be >= 1 for OT coupling, got {level}")
    
    x0 = np.asarray(x0).flatten()
    vol = np.asarray(vol).flatten()
    d = len(vol)
    
    # Time discretisation
    hl_f = h0 * 2 ** (-level)
    N_f = int(round(T / hl_f))
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    hl_c = 2 * hl_f
    N_c = N_f // 2
    
    G_chol = np.linalg.cholesky(cov_mat)
    
    # Accumulators for log-space statistics at coarse times
    sum_f = np.zeros((N_c, d))
    sum_c = np.zeros((N_c, d))
    sum2_f = np.zeros((N_c, d, d))
    sum2_c = np.zeros((N_c, d, d))
    count = 0
    
    for m0 in range(0, M_pilot, batch_size):
        m1 = min(M_pilot, m0 + batch_size)
        B = m1 - m0
        
        X_f = np.tile(x0, (B, 1))
        X_c = X_f.copy()
        
        # Initial state (n=0)
        log_f = np.log(X_f)
        log_c = np.log(X_c)
        sum_f[0] += log_f.sum(axis=0)
        sum_c[0] += log_c.sum(axis=0)
        sum2_f[0] += log_f.T @ log_f
        sum2_c[0] += log_c.T @ log_c
        
        Z = np.random.randn(B, N_f, d)
        
        for n in range(1, N_c):
            # Two fine steps
            Z0 = Z[:, 2 * (n - 1), :]
            dW0 = (Z0 @ G_chol.T) * math.sqrt(hl_f)
            X_f = X_f + r * X_f * hl_f + (X_f * vol) * dW0
            
            Z1 = Z[:, 2 * (n - 1) + 1, :]
            dW1 = (Z1 @ G_chol.T) * math.sqrt(hl_f)
            X_f = X_f + r * X_f * hl_f + (X_f * vol) * dW1
            
            # One coarse step (using combined increment)
            dW_c = dW0 + dW1
            X_c = X_c + r * X_c * hl_c + (X_c * vol) * dW_c
            
            # Accumulate log-space statistics
            log_f = np.log(X_f)
            log_c = np.log(X_c)
            sum_f[n] += log_f.sum(axis=0)
            sum_c[n] += log_c.sum(axis=0)
            sum2_f[n] += log_f.T @ log_f
            sum2_c[n] += log_c.T @ log_c
        
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
# SECTION 5: MLMC Level Estimators (Accumulated Normal Equations)
# =============================================================================

def mlmc_level(x0: np.ndarray, T: float, h0: float, level: int,
               r: float, cov_mat: np.ndarray, vol: np.ndarray,
               max_deg: int, P1: np.ndarray, s_min: float, s_max: float,
               C: int = 80, batch_size: int = 50,
               verbose: bool = False) -> np.ndarray:
    """
    Compute MLMC level-l coefficients using accumulated normal equations.
    
    This is the memory-efficient core of the fast MLMC implementation.
    Instead of building the full design matrix D ∈ R^{MN × dimV}, we
    accumulate the normal equations incrementally:
    
        G += D_n.T @ D_n    (Gram matrix)
        g += D_n.T @ psi_n  (moment vector)
    
    Then solve G c = g via Cholesky decomposition.
    
    The polynomial degree decreases with level: l_V = max_deg - level,
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
    level : int
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
    s_min, s_max : float
        Basket scaling bounds from pilot run.
    C : int, optional
        Base sample size scaling factor (default 80).
    batch_size : int, optional
        Batch size for memory-efficient processing (default 50).
    verbose : bool, optional
        Print progress information (default False).
        
    Returns
    -------
    c_padded : ndarray
        Coefficient vector, zero-padded to dim_max length.
    """
    x0 = np.asarray(x0).flatten()
    vol = np.asarray(vol).flatten()
    d = len(vol)
    
    # Multi-resolution: polynomial degree decreases with level
    l_V = max_deg - level
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))
    
    # Sample size scales with basis dimension squared
    M_l = max(C, int(C * dimV ** 2))
    n_batches = int(np.ceil(M_l / batch_size))
    
    # Time discretisation
    hl_f = h0 * 2 ** (-level)
    N_f = int(round(T / hl_f))
    if level > 0 and N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    
    if level == 0:
        hl_c = hl_f
        N_c = N_f
    else:
        hl_c = 2 * hl_f
        N_c = N_f // 2
    
    G_chol = np.linalg.cholesky(cov_mat)
    
    # Precompute Legendre values for time grid (normalised to [-1, 1])
    t_grid = np.arange(N_c) * hl_c
    t_scaled = 2.0 * t_grid / T - 1.0
    deg_t = l_V
    deg_s = l_V
    norm_t = np.sqrt(2 * np.arange(deg_t + 1) + 1)
    norm_s = np.sqrt(2 * np.arange(deg_s + 1) + 1)
    VT = legvander(t_scaled, deg_t) * norm_t
    
    # Accumulators for normal equations
    G = np.zeros((dimV, dimV))
    g = np.zeros((dimV, 1))
    
    if verbose:
        print(f"  Level {level}: deg={l_V}, dimV={dimV}, M={M_l}, batches={n_batches}")
    
    for b_idx in range(n_batches):
        B = min(batch_size, M_l - b_idx * batch_size)
        if B <= 0:
            break
        
        X0 = np.tile(x0, (B, 1))
        
        if level == 0:
            # Level 0: no coarse paths, just fit b² directly
            X_f = X0.copy()
            Z = np.random.randn(B, N_f, d)
            
            for n in range(N_f):
                if n > 0:
                    Z_n = Z[:, n - 1, :]
                    sigma = X_f * vol
                    dW = (Z_n @ G_chol.T) * math.sqrt(hl_f)
                    X_f = X_f + r * X_f * hl_f + sigma * dW
                
                # Build design matrix row block
                D_n = np.empty((B, dimV), dtype=float)
                trow = VT[n, :]
                s = X_f @ P1
                s_scaled = np.clip(2.0 * (s - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
                VS = legvander(s_scaled, deg_s) * norm_s
                
                for p, (i1, i2) in enumerate(pairs):
                    D_n[:, p] = trow[i1] * VS[:, i2]
                
                # Instantaneous variance (target for regression)
                sigma_f = X_f * vol
                b_sq = ((sigma_f @ cov_mat) * sigma_f).sum(axis=1) / (d ** 2)
                psi_n = b_sq.reshape(-1, 1)
                
                G += D_n.T @ D_n
                g += D_n.T @ psi_n
        
        else:
            # Level > 0: compute telescoping difference b²_fine - b²_coarse
            X_f = X0.copy()
            X_c = X0.copy()
            Z = np.random.randn(B, N_f, d)
            
            # Initial timestep
            trow0 = VT[0, :]
            s0 = X0 @ P1
            s0_scaled = np.clip(2.0 * (s0 - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
            VS0 = legvander(s0_scaled, deg_s) * norm_s
            
            D0 = np.empty((B, dimV), dtype=float)
            for p, (i_t, i_s) in enumerate(pairs):
                D0[:, p] = trow0[i_t] * VS0[:, i_s]
            psi0 = np.zeros((B, 1), dtype=float)  # No difference at t=0
            
            G += D0.T @ D0
            g += D0.T @ psi0
            
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
                dW_c = dW1 + dW2
                sigma = X_c * vol
                X_c = X_c + r * X_c * hl_c + sigma * dW_c
                
                # Build design matrix row block
                D_n = np.empty((B, dimV), dtype=float)
                trow = VT[n, :]
                s = X_f @ P1
                s_scaled = np.clip(2.0 * (s - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
                VS = legvander(s_scaled, deg_s) * norm_s
                
                for p, (i1, i2) in enumerate(pairs):
                    D_n[:, p] = trow[i1] * VS[:, i2]
                
                # Telescoping difference: b²_fine - b²_coarse
                sigma_f = X_f * vol
                b_f = ((sigma_f @ cov_mat) * sigma_f).sum(axis=1) / (d ** 2)
                sigma_c = X_c * vol
                b_c = ((sigma_c @ cov_mat) * sigma_c).sum(axis=1) / (d ** 2)
                psi_n = (b_f - b_c).reshape(-1, 1)
                
                G += D_n.T @ D_n
                g += D_n.T @ psi_n
    
    # Solve normal equations via Cholesky
    G_reg = 0.5 * (G + G.T)  # Ensure symmetry
    
    # Check condition number
    lam = np.linalg.eigvalsh(G_reg)
    lam_min = max(lam[0], 1e-300)
    cond_approx = np.sqrt(lam[-1] / lam_min)
    
    if verbose:
        print(f"    Condition number ≈ {cond_approx:.2e}")
    
    # Add regularisation if ill-conditioned
    if cond_approx > 1e12:
        reg = 1e-10 * lam[-1]
        G_reg += reg * np.eye(dimV)
        if verbose:
            print(f"    Added regularisation: {reg:.2e}")
    
    L = np.linalg.cholesky(G_reg)
    y = np.linalg.solve(L, g)
    c = np.linalg.solve(L.T, y).ravel()
    
    # Zero-pad to maximum dimension
    c_padded = np.zeros(dim_max)
    c_padded[:len(c)] = c
    
    return c_padded


def mlmc_level_ot(x0: np.ndarray, T: float, h0: float, level: int,
                  r: float, cov_mat: np.ndarray, vol: np.ndarray,
                  max_deg: int, P1: np.ndarray, s_min: float, s_max: float,
                  C: int = 80, batch_size: int = 50,
                  M_pilot: int = 500, verbose: bool = False) -> np.ndarray:
    """
    Compute MLMC level-l coefficients with optimal transport coupling.
    
    Instead of using standard Brownian coupling for coarse paths, this
    uses Gaussian-Brenier maps to create optimally coupled paths in
    log-space. This provides theoretically optimal variance reduction.
    
    Parameters
    ----------
    [Same as mlmc_level, plus:]
    M_pilot : int, optional
        Number of pilot paths for OT map estimation (default 500).
        
    Returns
    -------
    c_padded : ndarray
        Coefficient vector, zero-padded to dim_max length.
        
    Raises
    ------
    ValueError
        If level < 1 (OT requires fine/coarse pairing).
    """
    if level < 1:
        raise ValueError(f"Level must be >= 1 for OT coupling, got {level}")
    
    x0 = np.asarray(x0).flatten()
    vol = np.asarray(vol).flatten()
    d = len(vol)
    
    # Multi-resolution: polynomial degree decreases with level
    l_V = max_deg - level
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))
    
    # Sample size scales with basis dimension squared
    M_l = max(C, int(C * dimV ** 2))
    n_batches = int(np.ceil(M_l / batch_size))
    
    # Time discretisation
    hl_f = h0 * 2 ** (-level)
    N_f = int(round(T / hl_f))
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    hl_c = 2 * hl_f
    N_c = N_f // 2
    
    G_chol = np.linalg.cholesky(cov_mat)
    
    # Compute OT maps from pilot simulation
    if verbose:
        print(f"  Level {level}: Computing OT maps from {M_pilot} pilot paths...")
    
    maps = compute_ot_maps(x0, T, h0, level, r, cov_mat, vol, 
                           M_pilot=M_pilot, batch_size=min(50, M_pilot))
    
    # Precompute Legendre values for time grid
    t_grid = np.arange(N_c) * hl_c
    t_scaled = 2.0 * t_grid / T - 1.0
    deg_t = l_V
    deg_s = l_V
    norm_t = np.sqrt(2 * np.arange(deg_t + 1) + 1)
    norm_s = np.sqrt(2 * np.arange(deg_s + 1) + 1)
    VT = legvander(t_scaled, deg_t) * norm_t
    
    # Accumulators for normal equations
    G = np.zeros((dimV, dimV))
    g = np.zeros((dimV, 1))
    
    if verbose:
        print(f"  Level {level}: deg={l_V}, dimV={dimV}, M={M_l}, batches={n_batches}")
    
    for b_idx in range(n_batches):
        B = min(batch_size, M_l - b_idx * batch_size)
        if B <= 0:
            break
        
        X0 = np.tile(x0, (B, 1))
        X_f = X0.copy()
        
        # Initial timestep
        trow0 = VT[0, :]
        s0 = X0 @ P1
        s0_scaled = np.clip(2.0 * (s0 - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
        VS0 = legvander(s0_scaled, deg_s) * norm_s
        
        D0 = np.empty((B, dimV), dtype=float)
        for p, (i_t, i_s) in enumerate(pairs):
            D0[:, p] = trow0[i_t] * VS0[:, i_s]
        psi0 = np.zeros((B, 1), dtype=float)
        
        G += D0.T @ D0
        g += D0.T @ psi0
        
        Z = np.random.randn(B, N_f, d)
        
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
            log_X_f = np.log(X_f)
            log_X_c = maps[n](log_X_f)
            X_c = np.exp(log_X_c)
            
            # Build design matrix row block
            D_n = np.empty((B, dimV), dtype=float)
            trow = VT[n, :]
            s = X_f @ P1
            s_scaled = np.clip(2.0 * (s - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
            VS = legvander(s_scaled, deg_s) * norm_s
            
            for p, (i1, i2) in enumerate(pairs):
                D_n[:, p] = trow[i1] * VS[:, i2]
            
            # Telescoping difference: b²_fine - b²_coarse
            sigma_f = X_f * vol
            b_f = ((sigma_f @ cov_mat) * sigma_f).sum(axis=1) / (d ** 2)
            sigma_c = X_c * vol
            b_c = ((sigma_c @ cov_mat) * sigma_c).sum(axis=1) / (d ** 2)
            psi_n = (b_f - b_c).reshape(-1, 1)
            
            G += D_n.T @ D_n
            g += D_n.T @ psi_n
    
    # Solve normal equations via Cholesky
    G_reg = 0.5 * (G + G.T)
    
    lam = np.linalg.eigvalsh(G_reg)
    lam_min = max(lam[0], 1e-300)
    cond_approx = np.sqrt(lam[-1] / lam_min)
    
    if verbose:
        print(f"    Condition number ≈ {cond_approx:.2e}")
    
    if cond_approx > 1e12:
        reg = 1e-10 * lam[-1]
        G_reg += reg * np.eye(dimV)
    
    L = np.linalg.cholesky(G_reg)
    y = np.linalg.solve(L, g)
    c = np.linalg.solve(L.T, y).ravel()
    
    c_padded = np.zeros(dim_max)
    c_padded[:len(c)] = c
    
    return c_padded


# =============================================================================
# SECTION 6: Telescoping Sum Aggregation
# =============================================================================

def make_c(x0: np.ndarray, T: float, h0: float, r: float,
           cov_mat: np.ndarray, vol: np.ndarray, max_deg: int,
           P1: np.ndarray, s_min: float, s_max: float,
           C: int = 80, batch_size: int = 50,
           verbose: bool = False) -> np.ndarray:
    """
    Compute full coefficient vector via MLMC telescoping sum (standard coupling).
    
    Implements c = Σ_{l=0}^{max_deg} c_l where each c_l is computed
    using accumulated normal equations at level l with Brownian coupling.
    
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
    s_min, s_max : float
        Basket scaling bounds.
    C : int, optional
        Base sample size factor.
    batch_size : int, optional
        Batch size for processing.
    verbose : bool, optional
        Print progress information.
        
    Returns
    -------
    c : ndarray
        Full coefficient vector for the volatility surface.
    """
    if verbose:
        print(f"\nMLMC Coefficient Estimation (max_deg={max_deg}, coupling=Brownian)")
        print("=" * 60)
    
    c_total = np.zeros(len(tot_degree_poly(max_deg)))
    
    for level in range(max_deg + 1):
        c_l = mlmc_level(x0, T, h0, level, r, cov_mat, vol, max_deg,
                         P1, s_min, s_max, C, batch_size, verbose)
        c_total += c_l
    
    if verbose:
        print("=" * 60)
        print("MLMC Aggregation Complete\n")
    
    return c_total


def make_c_ot(x0: np.ndarray, T: float, h0: float, r: float,
              cov_mat: np.ndarray, vol: np.ndarray, max_deg: int,
              P1: np.ndarray, s_min: float, s_max: float,
              C: int = 80, batch_size: int = 50,
              M_pilot: int = 500, verbose: bool = False) -> np.ndarray:
    """
    Compute full coefficient vector via OT-enhanced MLMC telescoping sum.
    
    Uses standard MLMC for level 0 (no OT needed), then OT-enhanced
    estimators for levels 1 through max_deg.
    
    Parameters
    ----------
    [Same as make_c, plus:]
    M_pilot : int, optional
        Number of pilot paths for OT map estimation (default 500).
        
    Returns
    -------
    c : ndarray
        Full coefficient vector for the volatility surface.
    """
    if verbose:
        print(f"\nMLMC+OT Coefficient Estimation (max_deg={max_deg})")
        print("=" * 60)
    
    c_total = np.zeros(len(tot_degree_poly(max_deg)))
    
    # Level 0: standard MLMC (no OT needed)
    c_0 = mlmc_level(x0, T, h0, 0, r, cov_mat, vol, max_deg,
                     P1, s_min, s_max, C, batch_size, verbose)
    c_total += c_0
    
    # Levels 1+: OT-enhanced MLMC
    for level in range(1, max_deg + 1):
        c_l = mlmc_level_ot(x0, T, h0, level, r, cov_mat, vol, max_deg,
                            P1, s_min, s_max, C, batch_size, M_pilot, verbose)
        c_total += c_l
    
    if verbose:
        print("=" * 60)
        print("MLMC+OT Aggregation Complete\n")
    
    return c_total


# =============================================================================
# SECTION 7: Surface Construction
# =============================================================================

def make_b_bar(c: np.ndarray, pairs: list, s_min: float, s_max: float,
               T: float, max_deg: int) -> Callable[[float, float], float]:
    """
    Construct the projected volatility surface function b̄(t, S).
    
    Creates a callable that evaluates the polynomial approximation:
        b̄(t, S) = sqrt(Σₚ cₚ Pᵢ₁(2t/T - 1) Pᵢ₂(2(S-s_min)/(s_max-s_min) - 1))
    
    where Pₖ are orthonormalised Legendre polynomials.
    
    Parameters
    ----------
    c : ndarray
        Coefficient vector from MLMC.
    pairs : list of tuple
        Polynomial index pairs [(i, j), ...].
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
    norm_t = np.sqrt(2 * np.arange(max_deg + 1) + 1)
    norm_s = np.sqrt(2 * np.arange(max_deg + 1) + 1)
    
    def b_bar(t: float, S: float) -> float:
        """Evaluate projected volatility at (t, S)."""
        t_scaled = 2.0 * t / T - 1.0
        S_scaled = np.clip(2.0 * (S - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
        
        VT = legvander(np.array([t_scaled]), max_deg)[0] * norm_t
        VS = legvander(np.array([S_scaled]), max_deg)[0] * norm_s
        
        b_sq = 0.0
        for p, (i1, i2) in enumerate(pairs):
            if p < len(c):
                b_sq += c[p] * VT[i1] * VS[i2]
        
        return np.sqrt(max(b_sq, 0.0))
    
    return b_bar


def make_b_squared(c: np.ndarray, pairs: list, s_min: float, s_max: float,
                   T: float, max_deg: int) -> Callable[[float, float], float]:
    """
    Construct the projected volatility squared surface function b̄²(t, S).
    
    Same as make_b_bar but returns b² directly without the square root.
    """
    norm_t = np.sqrt(2 * np.arange(max_deg + 1) + 1)
    norm_s = np.sqrt(2 * np.arange(max_deg + 1) + 1)
    
    def b_squared(t: float, S: float) -> float:
        """Evaluate projected volatility squared at (t, S)."""
        t_scaled = 2.0 * t / T - 1.0
        S_scaled = np.clip(2.0 * (S - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
        
        VT = legvander(np.array([t_scaled]), max_deg)[0] * norm_t
        VS = legvander(np.array([S_scaled]), max_deg)[0] * norm_s
        
        b_sq = 0.0
        for p, (i1, i2) in enumerate(pairs):
            if p < len(c):
                b_sq += c[p] * VT[i1] * VS[i2]
        
        return b_sq
    
    return b_squared


# =============================================================================
# SECTION 8: Main Interface
# =============================================================================

def estimate_volatility_mlmc(
    params,  # ProblemParameters dataclass
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    random_seed: int = 42,
    use_ot: bool = True,
    C: int = 80,
    batch_size: int = 50,
    M_pilot_domain: int = 10000,
    M_pilot_ot: int = 500,
    verbose: bool = False
) -> VolatilitySurfaceResult:
    """
    Estimate projected volatility surface using MLMC with optional OT coupling.
    
    This is the main interface for the comparison framework. It handles:
    1. Domain estimation via pilot simulation
    2. MLMC coefficient estimation (with or without OT)
    3. Surface construction and grid evaluation
    4. Packaging results in VolatilitySurfaceResult format
    
    Parameters
    ----------
    params : ProblemParameters
        Problem configuration (from config.py).
    t_grid : np.ndarray
        Time points for surface evaluation.
    s_grid : np.ndarray
        Basket values for surface evaluation.
    random_seed : int, optional
        Random seed for reproducibility (default 42).
    use_ot : bool, optional
        If True, use Gaussian-Brenier OT coupling (default True).
        If False, use standard Brownian coupling.
    C : int, optional
        Base sample size scaling factor (default 80).
    batch_size : int, optional
        Batch size for memory-efficient processing (default 50).
    M_pilot_domain : int, optional
        Number of pilot paths for domain estimation (default 10000).
    M_pilot_ot : int, optional
        Number of pilot paths for OT map estimation (default 500).
    verbose : bool, optional
        Print progress information (default False).
        
    Returns
    -------
    VolatilitySurfaceResult
        Standardised result object containing:
        - b_surface: callable b(t, S)
        - b_squared_surface: callable b²(t, S)
        - Grid values for plotting
        - Timing and metadata
    """
    np.random.seed(random_seed)
    start_time = time.perf_counter()
    
    method_name = "MLMC+OT" if use_ot else "MLMC"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{method_name} Volatility Surface Estimation")
        print(f"{'='*60}")
        print(f"  Assets: {params.d}, max_degree: {params.max_degree}")
        print(f"  OT coupling: {use_ot}")
    
    # Step 1: Estimate domain bounds via pilot run
    if verbose:
        print(f"\nStep 1: Domain estimation ({M_pilot_domain} pilot paths)...")
    
    s_min, s_max = estimate_domain(
        x0=params.x0,
        T=params.T,
        h0=params.h0,
        r=params.r,
        cov_mat=params.corr_matrix,
        vol=params.sigma,
        max_deg=params.max_degree,
        P1=params.P1,
        M_pilot=M_pilot_domain
    )
    
    # Add padding to domain bounds
    pad = (s_max - s_min) * 0.05
    s_min -= pad
    s_max += pad
    
    if verbose:
        print(f"    Domain: [{s_min:.1f}, {s_max:.1f}]")
    
    # Step 2: MLMC coefficient estimation
    if verbose:
        print(f"\nStep 2: {method_name} coefficient estimation...")
    
    if use_ot:
        coefficients = make_c_ot(
            x0=params.x0,
            T=params.T,
            h0=params.h0,
            r=params.r,
            cov_mat=params.corr_matrix,
            vol=params.sigma,
            max_deg=params.max_degree,
            P1=params.P1,
            s_min=s_min,
            s_max=s_max,
            C=C,
            batch_size=batch_size,
            M_pilot=M_pilot_ot,
            verbose=verbose
        )
    else:
        coefficients = make_c(
            x0=params.x0,
            T=params.T,
            h0=params.h0,
            r=params.r,
            cov_mat=params.corr_matrix,
            vol=params.sigma,
            max_deg=params.max_degree,
            P1=params.P1,
            s_min=s_min,
            s_max=s_max,
            C=C,
            batch_size=batch_size,
            verbose=verbose
        )
    
    if verbose:
        print(f"    Coefficients: {len(coefficients)} terms")
    
    # Step 3: Construct volatility surface
    pairs = tot_degree_poly(params.max_degree)
    b_surface = make_b_bar(coefficients, pairs, s_min, s_max, 
                           params.T, params.max_degree)
    b_squared_surface = make_b_squared(coefficients, pairs, s_min, s_max,
                                        params.T, params.max_degree)
    
    # Step 4: Evaluate on grid
    if verbose:
        print(f"\nStep 3: Evaluating on {len(t_grid)}×{len(s_grid)} grid...")
    
    b_squared_values = np.zeros((len(t_grid), len(s_grid)))
    for i, t in enumerate(t_grid):
        for j, s in enumerate(s_grid):
            b_squared_values[i, j] = b_squared_surface(t, s)
    
    computation_time = time.perf_counter() - start_time
    
    if verbose:
        print(f"\nCompleted in {computation_time:.2f}s")
        print(f"{'='*60}\n")
    
    return VolatilitySurfaceResult(
        b_surface=b_surface,
        b_squared_surface=b_squared_surface,
        t_grid=t_grid,
        s_grid=s_grid,
        b_squared_values=b_squared_values,
        method_name=method_name,
        computation_time=computation_time,
        parameters={
            "use_ot": use_ot,
            "max_degree": params.max_degree,
            "h0": params.h0,
            "C": C,
            "M_pilot_domain": M_pilot_domain,
            "M_pilot_ot": M_pilot_ot if use_ot else None,
            "random_seed": random_seed,
        },
        coefficients=coefficients,
        n_samples=None,  # Could compute total samples if needed
        mlmc_levels=params.max_degree + 1,
        domain_bounds=(s_min, s_max)
    )


def estimate_volatility_mlmc_multiple_runs(
    params,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    n_runs: int = 20,
    base_seed: int = 42,
    use_ot: bool = True,
    C: int = 80,
    batch_size: int = 50,
    M_pilot_domain: int = 10000,
    M_pilot_ot: int = 500,
    verbose: bool = True
) -> Tuple[VolatilitySurfaceResult, np.ndarray, np.ndarray]:
    """
    Run MLMC estimation multiple times for convergence study.

    Parameters
    ----------
    params : ProblemParameters
        Problem configuration.
    t_grid : np.ndarray
        Time grid.
    s_grid : np.ndarray
        Basket value grid.
    n_runs : int
        Number of independent runs.
    base_seed : int
        Base random seed (each run uses base_seed + i).
    use_ot : bool
        If True, use Gaussian-Brenier OT coupling (default True).
    C : int, optional
        Base sample size scaling factor (default 80).
    batch_size : int, optional
        Batch size for memory-efficient processing (default 50).
    M_pilot_domain : int, optional
        Number of pilot paths for domain estimation (default 10000).
    M_pilot_ot : int, optional
        Number of pilot paths for OT map estimation (default 500).
    verbose : bool
        Print progress.

    Returns
    -------
    mean_result : VolatilitySurfaceResult
        Result with mean b² surface.
    all_b_squared : np.ndarray
        All b² surfaces, shape (n_runs, len(t_grid), len(s_grid)).
    all_times : np.ndarray
        Computation times for each run.
    """
    if verbose:
        method_name = "MLMC+OT" if use_ot else "MLMC"
        print(f"Running {n_runs} {method_name} iterations for convergence study...")

    all_b_squared = []
    all_times = []

    for i in range(n_runs):
        seed = base_seed + i
        if verbose and (i + 1) % 5 == 0:
            print(f"  Run {i + 1}/{n_runs}...")

        result = estimate_volatility_mlmc(
            params, t_grid, s_grid,
            random_seed=seed,
            use_ot=use_ot,
            C=C,
            batch_size=batch_size,
            M_pilot_domain=M_pilot_domain,
            M_pilot_ot=M_pilot_ot,
            verbose=False
        )
        all_b_squared.append(result.b_squared_values)
        all_times.append(result.computation_time)

    all_b_squared = np.array(all_b_squared)
    all_times = np.array(all_times)

    # Compute mean surface
    mean_b_squared = np.mean(all_b_squared, axis=0)

    # Create interpolated surface for mean
    from scipy.interpolate import RectBivariateSpline
    interp = RectBivariateSpline(t_grid, s_grid, mean_b_squared)

    def mean_b_squared_surface(t: float, s: float) -> float:
        return float(interp(t, s))

    def mean_b_surface(t: float, s: float) -> float:
        return np.sqrt(max(mean_b_squared_surface(t, s), 0))

    method_name = "MLMC+OT (mean)" if use_ot else "MLMC (mean)"
    mean_result = VolatilitySurfaceResult(
        b_surface=mean_b_surface,
        b_squared_surface=mean_b_squared_surface,
        t_grid=t_grid,
        s_grid=s_grid,
        b_squared_values=mean_b_squared,
        method_name=method_name,
        computation_time=np.sum(all_times),
        parameters={
            "n_runs": n_runs,
            "base_seed": base_seed,
            "use_ot": use_ot,
            "max_degree": params.max_degree,
            "C": C,
            "M_pilot_domain": M_pilot_domain,
            "M_pilot_ot": M_pilot_ot if use_ot else None,
        },
        coefficients=None,
        n_samples=n_runs,
        mlmc_levels=params.max_degree + 1,
        domain_bounds=result.domain_bounds
    )

    if verbose:
        print(f"  Total time: {np.sum(all_times):.1f}s")
        print(f"  Mean time per run: {np.mean(all_times):.2f}s")
        std_b_squared = np.std(all_b_squared, axis=0)
        print(f"  Std of b²: [{np.nanmin(std_b_squared):.2f}, {np.nanmax(std_b_squared):.2f}]")

    return mean_result, all_b_squared, all_times
