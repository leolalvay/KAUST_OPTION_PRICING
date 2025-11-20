# This file is based on DM_ML.py from Amelie's work.

"""
Multi-Level Monte Carlo for Volatility Surface Estimation

Implements MLMC hierarchy to estimate polynomial coefficients for projected
volatility surfaces b(t,S) from basket option paths. Uses telescoping sums
across refinement levels to reduce computational cost.
"""

import numpy as np
import math
from basket_simulation import (
    generate_polynomial_basis_pairs,
    construct_regression_system,
    fit_volatility_coefficients
)


def estimate_basket_domain(S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, N_pilot=10000):
    """
    Pilot run to estimate spatial domain bounds for basket values.
    
    Simulates basket paths at finest timestep and extracts percentiles to
    determine [S_min, S_max] for spatial scaling in polynomial regression.
    
    Parameters
    ----------
    S0 : np.ndarray, shape (d,) or (d, 1)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep
    r : float
        Risk-free rate
    cov_mat : np.ndarray, shape (d, d)
        Correlation matrix
    vol : np.ndarray, shape (d,)
        Asset volatilities
    max_degree : int
        Maximum MLMC level (determines finest timestep)
    basket_weights : np.ndarray, shape (d,)
        Basket aggregation weights
    N_pilot : int, default 10000
        Number of pilot paths to simulate
        
    Returns
    -------
    S_min : float
        1st percentile of basket values
    S_max : float
        99th percentile of basket values
    basket_paths : np.ndarray, shape (N_pilot, N_steps)
        Basket value paths for diagnostics
        
    Notes
    -----
    - Uses finest timestep dt = h0 * 2^(-max_degree)
    - Percentiles chosen to capture most of the distribution while excluding outliers
    """
    dt = h0 * 2 ** (-max_degree)
    N_steps = int(T / dt)
    
    # Simulate paths at finest resolution
    chol_correlation = np.linalg.cholesky(cov_mat)
    sqrt_dt = math.sqrt(dt)
    d = len(vol)
    
    asset_prices = np.tile(S0.flatten(), (N_pilot, 1))
    paths = np.empty((N_pilot, N_steps, d))
    paths[:, 0, :] = asset_prices
    
    for n in range(1, N_steps):
        brownian_increments = np.random.randn(N_pilot, d)
        sigma = asset_prices * vol
        dW = brownian_increments @ chol_correlation.T
        asset_prices = asset_prices + r * asset_prices * dt + sigma * dW * sqrt_dt
        paths[:, n, :] = asset_prices
    
    # Aggregate to basket
    basket_paths = paths @ basket_weights
    S_min, S_max = np.percentile(basket_paths.flatten(), [1, 99])
    
    return S_min, S_max, basket_paths


def generate_coupled_paths(S0, r, vol, cov_mat, dt_fine, N_fine, N_paths, level):
    """
    Generate coupled fine and coarse paths for MLMC level estimation.
    
    Critical for variance reduction: coarse paths reuse the same Brownian
    increments as fine paths (summing pairs of increments). This ensures
    Y_l = P_fine - P_coarse has much lower variance than P_fine alone.
    
    Parameters
    ----------
    S0 : np.ndarray, shape (d,) or (d, 1)
        Initial asset prices
    r : float
        Risk-free rate
    vol : np.ndarray, shape (d,)
        Asset volatilities
    cov_mat : np.ndarray, shape (d, d)
        Correlation matrix
    dt_fine : float
        Fine timestep
    N_fine : int
        Number of fine timesteps
    N_paths : int
        Number of paths to generate
    level : int
        MLMC level (0 = coarsest, higher = finer)
        
    Returns
    -------
    paths_fine : np.ndarray, shape (N_paths, N_fine, d)
        Fine resolution paths
    paths_coarse : np.ndarray, shape (N_paths, N_coarse, d)
        Coarse resolution paths (coupled)
        
    Notes
    -----
    - For level 0: coarse paths are zero (no coarser level exists)
    - For level > 0: dt_coarse = 2 * dt_fine, N_coarse = N_fine // 2
    - Coarse increments = sum of consecutive pairs of fine increments
    """
    chol_correlation = np.linalg.cholesky(cov_mat)
    d = len(vol)
    dt_coarse = 2 * dt_fine
    N_coarse = N_fine // 2
    
    # Initialise paths
    paths_fine = np.empty((N_paths, N_fine, d))
    paths_coarse = np.empty((N_paths, N_coarse, d))
    
    initial_prices = np.tile(S0.flatten(), (N_paths, 1))
    paths_fine[:, 0, :] = initial_prices
    paths_coarse[:, 0, :] = initial_prices
    
    # Generate all Brownian increments upfront
    Z_fine = np.random.randn(N_paths, N_fine, d)
    
    # Simulate fine paths
    asset_prices = initial_prices.copy()
    for n in range(1, N_fine):
        Z_n = Z_fine[:, n-1, :]
        sigma = asset_prices * vol
        dW = (Z_n @ chol_correlation.T) * math.sqrt(dt_fine)
        asset_prices = asset_prices + r * asset_prices * dt_fine + sigma * dW
        paths_fine[:, n, :] = asset_prices
    
    # Simulate coarse paths (coupled)
    if level > 0:
        # Sum consecutive pairs of fine increments
        Z_coarse = Z_fine.reshape(N_paths, N_coarse, 2, d).sum(axis=2)
        
        asset_prices = initial_prices.copy()
        for n in range(1, N_coarse):
            Z_n = Z_coarse[:, n-1, :]
            sigma = asset_prices * vol
            dW = (Z_n @ chol_correlation.T) * math.sqrt(dt_coarse)
            asset_prices = asset_prices + r * asset_prices * dt_coarse + sigma * dW
            paths_coarse[:, n, :] = asset_prices
    else:
        # Level 0: no coarser level exists
        paths_coarse = np.zeros((N_paths, N_coarse, d))
    
    return paths_fine, paths_coarse


def estimate_coefficients_at_level(
    S0, T, h0, level, r, cov_mat, vol, max_degree, basket_weights, S_min, S_max
):
    """
    Estimate polynomial coefficients at a single MLMC level.
    
    Implements the MLMC telescoping: estimates c_l such that
        E[Y_L] = sum_{l=0}^L E[Y_l - Y_{l-1}]
    where Y_l represents the volatility difference on level l.
    
    Key features:
    - Polynomial degree reduces with level: l_V = max_degree - level
    - Sample size scales with basis dimension: M_l ~ C * dim(V)²
    - Paths are coupled for variance reduction
    
    Parameters
    ----------
    S0 : np.ndarray, shape (d,)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep
    level : int
        MLMC level (0 = coarsest)
    r : float
        Risk-free rate
    cov_mat : np.ndarray, shape (d, d)
        Correlation matrix
    vol : np.ndarray, shape (d,)
        Asset volatilities
    max_degree : int
        Maximum polynomial degree at level 0
    basket_weights : np.ndarray, shape (d,)
        Basket aggregation weights
    S_min, S_max : float
        Spatial domain bounds from pilot run
        
    Returns
    -------
    c_padded : np.ndarray, shape (n_basis_max,)
        Coefficient vector, padded to maximum basis size. Non-zero entries
        correspond to the reduced basis at this level.
        
    Notes
    -----
    - Level-dependent timestep: dt_l = h0 * 2^(-level)
    - Polynomial degree: l_V = max_degree - level (fewer basis functions at finer levels)
    - Sample size: M_l = max(C, C * dim(V)²) where C=80
    - Prints condition number and smallest singular values for diagnostics
    """
    d = len(vol)
    
    # Level-dependent timestep
    dt_fine = h0 * 2 ** (-level)
    dt_coarse = 2 * dt_fine
    N_fine = int(T / dt_fine)
    N_coarse = int(T / dt_coarse)
    
    # Reduced polynomial degree at this level
    degree_at_level = max_degree - level
    print(f"Level {level}: polynomial degree = {degree_at_level}")
    
    basis_pairs = generate_polynomial_basis_pairs(degree_at_level)
    n_basis = len(basis_pairs)
    n_basis_max = len(generate_polynomial_basis_pairs(max_degree))
    
    # Sample size scaling with basis dimension
    C = 80
    N_paths = max(C, int(C * n_basis**2))
    
    # Generate coupled paths
    paths_fine, paths_coarse = generate_coupled_paths(
        S0, r, vol, cov_mat, dt_fine, N_fine, N_paths, level
    )
    
    # Construct regression system
    D, psi = construct_regression_system(
        paths_fine, paths_coarse, basket_weights, basis_pairs,
        cov_mat, vol, S_min, S_max, T
    )
    
    # Diagnostics for numerical stability
    cond_num = np.linalg.cond(D)
    singular_values = np.linalg.svd(D, compute_uv=False)
    print(f"  Condition number: {cond_num:.2e}")
    print(f"  Smallest 5 singular values: {singular_values[-5:]}")
    
    # Fit coefficients
    c = fit_volatility_coefficients(D, psi)
    
    # Pad to maximum basis size (other levels may have more basis functions)
    c_padded = np.zeros(n_basis_max)
    c_padded[:c.shape[0]] = c
    
    return c_padded


def aggregate_mlmc_coefficients(
    S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, S_min, S_max
):
    """
    Aggregate coefficients across all MLMC levels via telescoping sum.
    
    Computes:
        c_total = sum_{l=0}^{max_degree} c_l
    
    where c_l are the level-specific corrections estimated by
    estimate_coefficients_at_level().
    
    Parameters
    ----------
    S0 : np.ndarray, shape (d,)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep
    r : float
        Risk-free rate
    cov_mat : np.ndarray, shape (d, d)
        Correlation matrix
    vol : np.ndarray, shape (d,)
        Asset volatilities
    max_degree : int
        Maximum MLMC level
    basket_weights : np.ndarray, shape (d,)
        Basket aggregation weights
    S_min, S_max : float
        Spatial domain bounds
        
    Returns
    -------
    c_total : np.ndarray, shape (n_basis_max,)
        Aggregated polynomial coefficients for volatility surface
        
    Notes
    -----
    - Iterates through levels 0, 1, ..., max_degree
    - Each level uses fewer basis functions (degree reduction)
    - Total cost ~ O(ε^(-2)) vs O(ε^(-3)) for standard MC
    """
    print(f"\n{'='*60}")
    print(f"MLMC Coefficient Estimation (max_degree={max_degree})")
    print(f"{'='*60}\n")
    
    c_total = sum(
        estimate_coefficients_at_level(
            S0, T, h0, level, r, cov_mat, vol, max_degree, 
            basket_weights, S_min, S_max
        )
        for level in range(max_degree + 1)
    )
    
    print(f"\n{'='*60}")
    print("MLMC Aggregation Complete")
    print(f"{'='*60}\n")
    
    return c_total
