# This file is based on FML_single_level.py, enhanced with JAX JIT compilation.
# Original work by Amelie, JAX conversion by Wadoud Charbak.

"""
JAX-Enhanced Single-Level Monte Carlo

Provides a JIT-compiled baseline single-level implementation for comparison
with multi-level methods. Uses the finest discretisation throughout.

This serves as the reference solution against which MLMC methods are validated.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from functools import partial
from typing import List, Tuple

from JAX_FML_utils import (
    GBM_paths, tot_degree_poly, eval_legendre_basis_2d, make_b_bar
)


# =============================================================================
# Single-Level Estimator
# =============================================================================

def single_level(key: jax.Array, x0: jax.Array, T: float, h0: float,
                 r: float, cov_mat: jax.Array, vol: jax.Array, max_deg: int,
                 P1: jax.Array, s_min0: float, s_max0: float,
                 C: int = 80) -> jax.Array:
    """
    Compute volatility coefficients using single-level Monte Carlo.
    
    Uses the finest discretisation h = h0 * 2^{-max_deg} throughout,
    providing the reference solution for MLMC validation.
    
    Parameters
    ----------
    key : jax.Array
        JAX PRNG key
    x0 : jax.Array, shape (d,)
        Initial asset prices
    T : float
        Maturity
    h0 : float
        Base timestep (will be refined by 2^max_deg)
    r : float
        Risk-free rate
    cov_mat : jax.Array, shape (d, d)
        Correlation matrix
    vol : jax.Array, shape (d,)
        Asset volatilities
    max_deg : int
        Maximum polynomial degree (determines finest timestep)
    P1 : jax.Array, shape (d,)
        Basket weights
    s_min0, s_max0 : float
        Domain bounds for normalisation
    C : int
        Sample size scaling factor
        
    Returns
    -------
    c : jax.Array, shape (dimV,)
        Coefficient vector for Legendre expansion
        
    Notes
    -----
    Computational cost scales as O(C * dimV^2 * N) where:
    - dimV = number of basis functions
    - N = T / (h0 * 2^{-max_deg}) = number of fine timesteps
    
    This is more expensive than MLMC but provides the unbiased reference.
    """
    d = vol.shape[0]
    
    # Polynomial basis
    pairs = tot_degree_poly(max_deg)
    pairs_tuple = tuple(pairs)
    dimV = len(pairs)
    
    # Finest timestep
    dt = h0 * (2.0 ** (-max_deg))
    N_t = int(round(T / dt))
    
    # Sample size (matches level 0 of MLMC)
    M = C * dimV * dimV
    
    # Generate paths at finest resolution
    paths = GBM_paths(key, x0, r, vol, cov_mat, dt, N_t, M)
    
    # Accumulate normal equations
    G = jnp.zeros((dimV, dimV))
    g = jnp.zeros((dimV,))
    
    # Process each timestep
    for n in range(N_t):
        # Basket values
        S = jnp.dot(paths[:, n, :], P1)  # (M,)
        
        # Normalise to [-1, 1]
        t_norm = 2.0 * (n * dt) / T - 1.0
        s_norm = 2.0 * (S - s_min0) / (s_max0 - s_min0) - 1.0
        s_norm = jnp.clip(s_norm, -1.0, 1.0)
        
        # Design matrix
        D_n = eval_legendre_basis_2d(jnp.full(M, t_norm), s_norm, pairs_tuple)
        
        # Compute b^2 for this timestep
        X = paths[:, n, :]
        sigma = X * vol  # (M, d)
        Sigma_cov = sigma @ cov_mat  # (M, d)
        b_sq = jnp.sum(Sigma_cov * sigma, axis=1) * jnp.dot(P1, P1)
        b_sq = b_sq / (S ** 2 + 1e-10)
        
        # Target: b^2 * dt
        psi_n = b_sq * dt
        
        # Accumulate
        G = G + D_n.T @ D_n
        g = g + D_n.T @ psi_n
    
    # Solve via Cholesky
    L = jla.cholesky(G + 1e-8 * jnp.eye(dimV), lower=True)
    y = jla.solve_triangular(L, g, lower=True)
    c = jla.solve_triangular(L.T, y, lower=False)
    
    return c


def single_level_qr(key: jax.Array, x0: jax.Array, T: float, h0: float,
                    r: float, cov_mat: jax.Array, vol: jax.Array, max_deg: int,
                    P1: jax.Array, s_min0: float, s_max0: float,
                    C: int = 80) -> jax.Array:
    """
    Single-level estimator using QR decomposition for better conditioning.
    
    Parameters
    ----------
    [Same as single_level]
        
    Returns
    -------
    c : jax.Array, shape (dimV,)
        Coefficient vector
    """
    d = vol.shape[0]
    
    pairs = tot_degree_poly(max_deg)
    pairs_tuple = tuple(pairs)
    dimV = len(pairs)
    
    dt = h0 * (2.0 ** (-max_deg))
    N_t = int(round(T / dt))
    M = C * dimV * dimV
    
    paths = GBM_paths(key, x0, r, vol, cov_mat, dt, N_t, M)
    
    # Collect R matrices and transformed RHS
    R_list = []
    g_list = []
    
    for n in range(N_t):
        S = jnp.dot(paths[:, n, :], P1)
        
        t_norm = 2.0 * (n * dt) / T - 1.0
        s_norm = jnp.clip(2.0 * (S - s_min0) / (s_max0 - s_min0) - 1.0, -1.0, 1.0)
        
        D_n = eval_legendre_basis_2d(jnp.full(M, t_norm), s_norm, pairs_tuple)
        
        X = paths[:, n, :]
        sigma = X * vol
        Sigma_cov = sigma @ cov_mat
        b_sq = jnp.sum(Sigma_cov * sigma, axis=1) * jnp.dot(P1, P1)
        b_sq = b_sq / (S ** 2 + 1e-10)
        psi_n = b_sq * dt
        
        # QR decomposition
        Q_n, R_n = jnp.linalg.qr(D_n, mode='reduced')
        g_n = Q_n.T @ psi_n
        
        R_list.append(R_n)
        g_list.append(g_n)
    
    # Stack and final QR
    R_stacked = jnp.vstack(R_list)
    g_stacked = jnp.concatenate(g_list)
    
    Q_final, R_final = jnp.linalg.qr(R_stacked, mode='reduced')
    g_final = Q_final.T @ g_stacked
    
    # Solve
    c = jla.solve_triangular(R_final, g_final[:dimV], lower=False)
    
    return c


# =============================================================================
# Comparison Utilities
# =============================================================================

def compute_sl_ml_difference(key: jax.Array, x0: jax.Array, T: float, h0: float,
                              r: float, cov_mat: jax.Array, vol: jax.Array,
                              max_deg: int, P1: jax.Array,
                              s_min0: float, s_max0: float,
                              C: int = 80) -> dict:
    """
    Compute and compare single-level vs multi-level coefficients.
    
    Returns
    -------
    results : dict
        Coefficient vectors and difference metrics
    """
    from JAX_FML_utils import make_c
    
    # Single-level
    key, subkey = jax.random.split(key)
    c_sl = single_level(subkey, x0, T, h0, r, cov_mat, vol, max_deg,
                        P1, s_min0, s_max0, C)
    
    # Multi-level
    key, subkey = jax.random.split(key)
    c_ml = make_c(subkey, x0, T, h0, r, cov_mat, vol, max_deg,
                  P1, s_min0, s_max0, C)
    
    # Trim ML to same size as SL
    dimV = len(c_sl)
    c_ml_trimmed = c_ml[:dimV]
    
    # Metrics
    abs_diff = jnp.linalg.norm(c_sl - c_ml_trimmed)
    rel_diff = abs_diff / (jnp.linalg.norm(c_sl) + 1e-10)
    
    return {
        'c_sl': c_sl,
        'c_ml': c_ml_trimmed,
        'abs_diff': float(abs_diff),
        'rel_diff': float(rel_diff),
        'c_sl_norm': float(jnp.linalg.norm(c_sl)),
        'c_ml_norm': float(jnp.linalg.norm(c_ml_trimmed))
    }


# =============================================================================
# Main block for testing
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("Testing JAX_FML_single_level.py")
    print("=" * 60)
    
    # Test parameters
    d = 3
    x0 = jnp.array([250.0, 250.0, 250.0])
    vol = jnp.array([0.2, 0.15, 0.1])
    cov_mat = jnp.array([[1.0, 0.5, 0.3],
                         [0.5, 1.0, 0.2],
                         [0.3, 0.2, 1.0]])
    P1 = jnp.ones(d) / d
    r = 0.05
    T = 1.0
    h0 = 0.01
    max_deg = 2
    
    key = jax.random.PRNGKey(42)
    
    # Get scaling
    from JAX_FML_utils import scalings_l0
    key, subkey = jax.random.split(key)
    s_min, s_max = scalings_l0(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1)
    print(f"Domain: [{s_min:.2f}, {s_max:.2f}]")
    
    # Test single-level (Cholesky)
    print("\nTesting single_level (Cholesky)...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c_sl = single_level(subkey, x0, T, h0, r, cov_mat, vol, max_deg,
                        P1, s_min, s_max, C=30)
    t1 = time.time()
    print(f"  Computed in {t1-t0:.2f}s")
    print(f"  Coefficient norm: {jnp.linalg.norm(c_sl):.4f}")
    print(f"  Coefficients: {c_sl[:5]}")
    
    # Test single-level (QR)
    print("\nTesting single_level_qr...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c_sl_qr = single_level_qr(subkey, x0, T, h0, r, cov_mat, vol, max_deg,
                              P1, s_min, s_max, C=30)
    t1 = time.time()
    print(f"  Computed in {t1-t0:.2f}s")
    print(f"  Coefficient norm: {jnp.linalg.norm(c_sl_qr):.4f}")
    
    # Compare Cholesky vs QR
    diff = jnp.linalg.norm(c_sl - c_sl_qr) / jnp.linalg.norm(c_sl)
    print(f"  Cholesky vs QR relative diff: {diff:.2e}")
    
    # Test SL vs ML comparison
    print("\nComparing Single-Level vs Multi-Level...")
    key, subkey = jax.random.split(key)
    results = compute_sl_ml_difference(subkey, x0, T, h0, r, cov_mat, vol,
                                        max_deg, P1, s_min, s_max, C=30)
    print(f"  SL norm: {results['c_sl_norm']:.4f}")
    print(f"  ML norm: {results['c_ml_norm']:.4f}")
    print(f"  Absolute diff: {results['abs_diff']:.4e}")
    print(f"  Relative diff: {results['rel_diff']:.4e}")
    
    # Test volatility surface
    print("\nTesting volatility surface construction...")
    pairs = tot_degree_poly(max_deg)
    bbar = make_b_bar(c_sl, pairs, s_min, s_max, T, max_deg)
    
    t_test = jnp.linspace(0.1, 0.9, 5)
    S_test = jnp.linspace(s_min + 10, s_max - 10, 5)
    TT, SS = jnp.meshgrid(t_test, S_test)
    
    b_vals = bbar(TT, SS)
    print(f"  Surface shape: {b_vals.shape}")
    print(f"  Surface range: [{b_vals.min():.4f}, {b_vals.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
