# This file is based on DM_ML.py from Amelie's work.
# JAX version with selective JIT optimization and PRNG

"""
Multi-Level Monte Carlo for Volatility Surface Estimation (JAX Version)

Implements MLMC hierarchy to estimate polynomial coefficients for projected
volatility surfaces b(t,S) from basket option paths. Uses telescoping sums
across refinement levels to reduce computational cost.
"""

import jax.numpy as jnp
import jax
from jax import jit, random
from JAX_basket_simulation import (
    generate_polynomial_basis_pairs,
    construct_regression_system,
    fit_volatility_coefficients,
    simulate_gbm_paths
)


def estimate_basket_domain(S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, N_pilot, key):
    """
    Pilot run to estimate spatial domain bounds for basket values.

    Simulates basket paths at finest timestep and extracts percentiles to
    determine [S_min, S_max] for spatial scaling in polynomial regression.

    Parameters
    ----------
    S0 : jnp.ndarray, shape (d,) or (d, 1)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep
    r : float
        Risk-free rate
    cov_mat : jnp.ndarray, shape (d, d)
        Correlation matrix
    vol : jnp.ndarray, shape (d,)
        Asset volatilities
    max_degree : int
        Maximum MLMC level (determines finest timestep)
    basket_weights : jnp.ndarray, shape (d,)
        Basket aggregation weights
    N_pilot : int
        Number of pilot paths to simulate
    key : jax.random.PRNGKey
        Random key for JAX PRNG

    Returns
    -------
    S_min : float
        1st percentile of basket values
    S_max : float
        99th percentile of basket values
    basket_paths : jnp.ndarray, shape (N_pilot, N_steps)
        Basket value paths for diagnostics

    Notes
    -----
    - Uses finest timestep dt = h0 * 2^(-max_degree)
    - Percentiles chosen to capture most of the distribution while excluding outliers
    """
    dt = h0 * 2 ** (-max_degree)
    N_steps = int(T / dt)

    # Generate random increments
    d = len(vol)
    Z_random = random.normal(key, shape=(N_pilot, N_steps - 1, d))

    # Simulate paths at finest resolution
    paths = simulate_gbm_paths(S0, r, vol, cov_mat, dt, N_steps, N_pilot, Z_random)

    # Aggregate to basket
    basket_paths = paths @ basket_weights
    S_min, S_max = jnp.percentile(basket_paths.flatten(), jnp.array([1, 99]))

    return float(S_min), float(S_max), basket_paths


@jit
def generate_coupled_paths_core(S0, r, vol, chol_correlation, dt_fine, dt_coarse, N_fine, N_coarse, N_paths, level, Z_fine):
    """
    Core JIT-compiled function for generating coupled paths.

    Parameters
    ----------
    S0 : jnp.ndarray, shape (d,)
        Initial asset prices
    r : float
        Risk-free rate
    vol : jnp.ndarray, shape (d,)
        Asset volatilities
    chol_correlation : jnp.ndarray, shape (d, d)
        Cholesky factor of correlation matrix
    dt_fine : float
        Fine timestep
    dt_coarse : float
        Coarse timestep
    N_fine : int
        Number of fine timesteps
    N_coarse : int
        Number of coarse timesteps
    N_paths : int
        Number of paths
    level : int
        MLMC level
    Z_fine : jnp.ndarray, shape (N_paths, N_fine, d)
        Pre-generated random normals

    Returns
    -------
    paths_fine : jnp.ndarray, shape (N_paths, N_fine, d)
        Fine resolution paths
    paths_coarse : jnp.ndarray, shape (N_paths, N_coarse, d)
        Coarse resolution paths (coupled)
    """
    d = len(vol)
    sqrt_dt_fine = jnp.sqrt(dt_fine)
    sqrt_dt_coarse = jnp.sqrt(dt_coarse)

    # Get dimensions from Z_fine.shape
    N_paths = Z_fine.shape[0]
    N_fine_steps = Z_fine.shape[1]

    initial_prices = jnp.broadcast_to(S0.flatten()[None, :], (N_paths, d))
    initial_step = initial_prices[:, None, :]  # Shape: (N_paths, 1, d)

    # Simulate fine paths using scan
    def scan_fn_fine(asset_prices, n):
        Z_n = Z_fine[:, n, :]
        sigma = asset_prices * vol
        dW = (Z_n @ chol_correlation.T) * sqrt_dt_fine
        asset_prices = asset_prices + r * asset_prices * dt_fine + sigma * dW
        return asset_prices, asset_prices[:, None, :]

    _, fine_steps = jax.lax.scan(scan_fn_fine, initial_prices, jnp.arange(N_fine_steps))
    paths_fine = jnp.concatenate([initial_step, fine_steps.reshape(N_paths, -1, d)], axis=1)

    # Simulate coarse paths (coupled) if level > 0
    def simulate_coarse():
        # Sum consecutive pairs of fine increments
        N_coarse_steps = N_fine_steps // 2
        Z_coarse = Z_fine.reshape(N_paths, N_coarse_steps, 2, d).sum(axis=2)

        def scan_fn_coarse(asset_prices, n):
            Z_n = Z_coarse[:, n, :]
            sigma = asset_prices * vol
            dW = (Z_n @ chol_correlation.T) * sqrt_dt_coarse
            asset_prices = asset_prices + r * asset_prices * dt_coarse + sigma * dW
            return asset_prices, asset_prices[:, None, :]

        _, coarse_steps = jax.lax.scan(scan_fn_coarse, initial_prices, jnp.arange(N_coarse_steps))
        return jnp.concatenate([initial_step, coarse_steps.reshape(N_paths, -1, d)], axis=1)

    # Level 0: no coarser level exists
    paths_coarse = jax.lax.cond(
        level > 0,
        lambda: simulate_coarse(),
        lambda: jnp.zeros((N_paths, (N_fine_steps // 2) + 1, d)),
    )

    return paths_fine, paths_coarse


def generate_coupled_paths(S0, r, vol, cov_mat, dt_fine, N_fine, N_paths, level, key):
    """
    Generate coupled fine and coarse paths for MLMC level estimation.

    Critical for variance reduction: coarse paths reuse the same Brownian
    increments as fine paths (summing pairs of increments). This ensures
    Y_l = P_fine - P_coarse has much lower variance than P_fine alone.

    Parameters
    ----------
    S0 : jnp.ndarray, shape (d,) or (d, 1)
        Initial asset prices
    r : float
        Risk-free rate
    vol : jnp.ndarray, shape (d,)
        Asset volatilities
    cov_mat : jnp.ndarray, shape (d, d)
        Correlation matrix
    dt_fine : float
        Fine timestep
    N_fine : int
        Number of fine timesteps
    N_paths : int
        Number of paths to generate
    level : int
        MLMC level (0 = coarsest, higher = finer)
    key : jax.random.PRNGKey
        Random key for JAX PRNG

    Returns
    -------
    paths_fine : jnp.ndarray, shape (N_paths, N_fine, d)
        Fine resolution paths
    paths_coarse : jnp.ndarray, shape (N_paths, N_coarse, d)
        Coarse resolution paths (coupled)

    Notes
    -----
    - For level 0: coarse paths are zero (no coarser level exists)
    - For level > 0: dt_coarse = 2 * dt_fine, N_coarse = N_fine // 2
    - Coarse increments = sum of consecutive pairs of fine increments
    """
    chol_correlation = jnp.linalg.cholesky(cov_mat)
    d = len(vol)
    dt_coarse = 2 * dt_fine
    N_coarse = N_fine // 2

    # Generate all Brownian increments upfront
    Z_fine = random.normal(key, shape=(N_paths, N_fine, d))

    paths_fine, paths_coarse = generate_coupled_paths_core(
        S0, r, vol, chol_correlation, dt_fine, dt_coarse, N_fine, N_coarse, N_paths, level, Z_fine
    )

    return paths_fine, paths_coarse


def estimate_coefficients_at_level(
    S0, T, h0, level, r, cov_mat, vol, max_degree, basket_weights, S_min, S_max, key
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
    S0 : jnp.ndarray, shape (d,)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep
    level : int
        MLMC level (0 = coarsest)
    r : float
        Risk-free rate
    cov_mat : jnp.ndarray, shape (d, d)
        Correlation matrix
    vol : jnp.ndarray, shape (d,)
        Asset volatilities
    max_degree : int
        Maximum polynomial degree at level 0
    basket_weights : jnp.ndarray, shape (d,)
        Basket aggregation weights
    S_min, S_max : float
        Spatial domain bounds from pilot run
    key : jax.random.PRNGKey
        Random key for JAX PRNG

    Returns
    -------
    c_padded : jnp.ndarray, shape (n_basis_max,)
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
        S0, r, vol, cov_mat, dt_fine, N_fine, N_paths, level, key
    )

    # Construct regression system
    D, psi = construct_regression_system(
        paths_fine, paths_coarse, basket_weights, basis_pairs,
        cov_mat, vol, S_min, S_max, T
    )

    # Diagnostics for numerical stability
    cond_num = jnp.linalg.cond(D)
    singular_values = jnp.linalg.svd(D, compute_uv=False)
    print(f"  Condition number: {cond_num:.2e}")
    print(f"  Smallest 5 singular values: {singular_values[-5:]}")

    # Fit coefficients
    c = fit_volatility_coefficients(D, psi)

    # Pad to maximum basis size (other levels may have more basis functions)
    c_padded = jnp.zeros(n_basis_max)
    c_padded = c_padded.at[:c.shape[0]].set(c)

    return c_padded


def aggregate_mlmc_coefficients(
    S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, S_min, S_max, key
):
    """
    Aggregate coefficients across all MLMC levels via telescoping sum.

    Computes:
        c_total = sum_{l=0}^{max_degree} c_l

    where c_l are the level-specific corrections estimated by
    estimate_coefficients_at_level().

    Parameters
    ----------
    S0 : jnp.ndarray, shape (d,)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep
    r : float
        Risk-free rate
    cov_mat : jnp.ndarray, shape (d, d)
        Correlation matrix
    vol : jnp.ndarray, shape (d,)
        Asset volatilities
    max_degree : int
        Maximum MLMC level
    basket_weights : jnp.ndarray, shape (d,)
        Basket aggregation weights
    S_min, S_max : float
        Spatial domain bounds
    key : jax.random.PRNGKey
        Random key for JAX PRNG

    Returns
    -------
    c_total : jnp.ndarray, shape (n_basis_max,)
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

    c_total = jnp.zeros(len(generate_polynomial_basis_pairs(max_degree)))

    for level in range(max_degree + 1):
        # Split key for each level
        key, subkey = random.split(key)
        c_level = estimate_coefficients_at_level(
            S0, T, h0, level, r, cov_mat, vol, max_degree,
            basket_weights, S_min, S_max, subkey
        )
        c_total = c_total + c_level

    print(f"\n{'='*60}")
    print("MLMC Aggregation Complete")
    print(f"{'='*60}\n")

    return c_total
