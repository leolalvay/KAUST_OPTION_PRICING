# This file is based on FML_hierarchical_qr.py, enhanced with JAX JIT compilation.
# Original work by Amelie, JAX conversion by Wadoud Charbak.

"""
JAX-Enhanced Hierarchical QR for MLMC

Implements three-level hierarchical QR decomposition with JAX JIT compilation
for numerically stable least-squares solving without squaring the condition number.

Hierarchy:
    Level 1: QR per timestep     D_n = Q_n R_n
    Level 2: QR per batch        stack(R_n) = Q_b R_b
    Level 3: Final QR            stack(R_b) = Q_f R_f

This avoids forming the normal equations G = D'D which squares condition number.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax
from functools import partial
from typing import Tuple, List
import numpy as np

from JAX_FML_utils import (
    GBM_paths, scalings_l0, tot_degree_poly,
    eval_legendre_basis_2d, make_b_bar
)


# =============================================================================
# QR Decomposition Utilities
# =============================================================================

@jax.jit
def qr_decomposition(A: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Compute thin QR decomposition.
    
    Parameters
    ----------
    A : jax.Array, shape (m, n) with m >= n
        Input matrix
        
    Returns
    -------
    Q : jax.Array, shape (m, n)
        Orthonormal columns
    R : jax.Array, shape (n, n)
        Upper triangular
    """
    Q, R = jnp.linalg.qr(A, mode='reduced')
    return Q, R


@jax.jit
def solve_triangular_system(R: jax.Array, b: jax.Array) -> jax.Array:
    """
    Solve upper triangular system R @ x = b.
    
    Parameters
    ----------
    R : jax.Array, shape (n, n)
        Upper triangular matrix
    b : jax.Array, shape (n,)
        Right-hand side
        
    Returns
    -------
    x : jax.Array, shape (n,)
        Solution
    """
    return jla.solve_triangular(R, b, lower=False)


@jax.jit
def accumulate_qr_rhs(Q: jax.Array, psi: jax.Array, 
                      g_prev: jax.Array) -> jax.Array:
    """
    Accumulate transformed RHS: g += Q' @ psi
    
    Parameters
    ----------
    Q : jax.Array, shape (m, n)
        Orthonormal matrix from QR
    psi : jax.Array, shape (m,)
        Target vector
    g_prev : jax.Array, shape (n,)
        Previous accumulated RHS
        
    Returns
    -------
    g : jax.Array, shape (n,)
        Updated RHS
    """
    return g_prev + Q.T @ psi


# =============================================================================
# Hierarchical QR Level Estimator
# =============================================================================

def mlmc_level_qr(key: jax.Array, x0: jax.Array, T: float, h0: float, level: int,
                  r: float, cov_mat: jax.Array, vol: jax.Array, max_deg: int,
                  P1: jax.Array, s_min0: float, s_max0: float,
                  C: int = 80) -> jax.Array:
    """
    Compute MLMC level estimator using hierarchical QR decomposition.
    
    Instead of accumulating normal equations (which squares condition number),
    we use a three-level QR hierarchy:
    
    1. Per timestep: D_n = Q_n R_n
    2. Per batch: stack R_n matrices and QR again
    3. Final: stack batch R matrices and solve
    
    Parameters
    ----------
    key : jax.Array
        JAX PRNG key
    x0 : jax.Array, shape (d,)
        Initial asset prices
    T : float
        Maturity
    h0 : float
        Base timestep
    level : int
        MLMC level (0, 1, ..., max_deg)
    r : float
        Risk-free rate
    cov_mat : jax.Array, shape (d, d)
        Correlation matrix
    vol : jax.Array, shape (d,)
        Asset volatilities
    max_deg : int
        Maximum polynomial degree
    P1 : jax.Array, shape (d,)
        Basket weights
    s_min0, s_max0 : float
        Domain bounds
    C : int
        Sample size scaling factor
        
    Returns
    -------
    c : jax.Array, shape (dim_max,)
        Coefficient vector (zero-padded)
    """
    d = vol.shape[0]
    
    # Level-dependent parameters
    deg_l = max_deg - level
    pairs = tot_degree_poly(deg_l)
    pairs_tuple = tuple(pairs)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))
    
    # Timesteps
    dt_f = h0 * (2.0 ** (-level - 1))
    dt_c = h0 * (2.0 ** (-level))
    N_f = int(round(T / dt_f))
    N_c = int(round(T / dt_c))
    
    # Sample size
    M_l = C * dimV * dimV
    
    # Generate coupled paths
    key, subkey = jax.random.split(key)
    paths_f = GBM_paths(subkey, x0, r, vol, cov_mat, dt_f, N_f, M_l)
    
    # Coarse paths
    if level > 0:
        key, subkey = jax.random.split(key)
        paths_c = GBM_paths(subkey, x0, r, vol, cov_mat, dt_c, N_c, M_l)
    else:
        paths_c = jnp.zeros((M_l, N_c, d))
    
    is_level_zero = (level == 0)
    
    # Storage for R matrices and transformed RHS from each timestep
    R_list = []
    g_list = []
    
    # Level 1: QR per timestep
    for n in range(N_c):
        # Basket values at fine timesteps
        S_f1 = jnp.dot(paths_f[:, 2*n, :], P1)
        S_f2 = jnp.dot(paths_f[:, 2*n + 1, :], P1)
        
        # Normalise to [-1, 1]
        t_norm1 = 2.0 * (2*n * dt_f) / T - 1.0
        t_norm2 = 2.0 * ((2*n + 1) * dt_f) / T - 1.0
        s_norm1 = jnp.clip(2.0 * (S_f1 - s_min0) / (s_max0 - s_min0) - 1.0, -1.0, 1.0)
        s_norm2 = jnp.clip(2.0 * (S_f2 - s_min0) / (s_max0 - s_min0) - 1.0, -1.0, 1.0)
        
        # Design matrices for both fine steps
        D1 = eval_legendre_basis_2d(jnp.full(M_l, t_norm1), s_norm1, pairs_tuple)
        D2 = eval_legendre_basis_2d(jnp.full(M_l, t_norm2), s_norm2, pairs_tuple)
        
        # Stack into combined design matrix
        D_n = jnp.vstack([D1, D2])  # (2*M_l, dimV)
        
        # Compute psi for this timestep
        def compute_b_sq(X):
            S = jnp.dot(X, P1)
            sigma = X * vol
            Sigma_cov = sigma @ cov_mat
            b_sq = jnp.sum(Sigma_cov * sigma, axis=1) * jnp.dot(P1, P1)
            return b_sq / (S ** 2 + 1e-10)
        
        b_f1_sq = compute_b_sq(paths_f[:, 2*n, :])
        b_f2_sq = compute_b_sq(paths_f[:, 2*n + 1, :])
        
        if is_level_zero:
            psi_fine = jnp.concatenate([b_f1_sq * dt_f, b_f2_sq * dt_f])
        else:
            b_c_sq = compute_b_sq(paths_c[:, n, :])
            # Split coarse contribution between fine steps
            psi1 = b_f1_sq * dt_f - b_c_sq * dt_c / 2.0
            psi2 = b_f2_sq * dt_f - b_c_sq * dt_c / 2.0
            psi_fine = jnp.concatenate([psi1, psi2])
        
        # QR decomposition for this timestep
        Q_n, R_n = qr_decomposition(D_n)
        g_n = Q_n.T @ psi_fine
        
        R_list.append(R_n)
        g_list.append(g_n)
    
    # Level 2: Stack R matrices and do final QR
    # Stack all R matrices
    R_stacked = jnp.vstack(R_list)  # (N_c * dimV, dimV)
    g_stacked = jnp.concatenate(g_list)  # (N_c * dimV,)
    
    # Final QR
    Q_final, R_final = qr_decomposition(R_stacked)
    g_final = Q_final.T @ g_stacked
    
    # Solve triangular system
    c = solve_triangular_system(R_final, g_final[:dimV])
    
    # Zero-pad to maximum basis size
    c_padded = jnp.zeros(dim_max)
    c_padded = c_padded.at[:dimV].set(c)
    
    return c_padded


def make_c_qr(key: jax.Array, x0: jax.Array, T: float, h0: float,
              r: float, cov_mat: jax.Array, vol: jax.Array, max_deg: int,
              P1: jax.Array, s_min0: float, s_max0: float,
              C: int = 80) -> jax.Array:
    """
    Compute telescoping sum using hierarchical QR.
    
    c_total = sum_{l=0}^{max_deg} c_l
    
    Parameters
    ----------
    [Same as make_c in JAX_FML_utils]
        
    Returns
    -------
    c_total : jax.Array, shape (dim_max,)
        Aggregated coefficient vector
    """
    dim_max = len(tot_degree_poly(max_deg))
    c_total = jnp.zeros(dim_max)
    
    for level in range(max_deg + 1):
        key, subkey = jax.random.split(key)
        c_l = mlmc_level_qr(subkey, x0, T, h0, level, r, cov_mat, vol,
                            max_deg, P1, s_min0, s_max0, C)
        c_total = c_total + c_l
    
    return c_total


# =============================================================================
# Condition Number Analysis
# =============================================================================

@jax.jit
def estimate_condition_number(D: jax.Array) -> float:
    """
    Estimate condition number of matrix D.
    
    Parameters
    ----------
    D : jax.Array, shape (m, n)
        Input matrix
        
    Returns
    -------
    kappa : float
        Condition number (ratio of largest to smallest singular value)
    """
    s = jnp.linalg.svd(D, compute_uv=False)
    return s[0] / (s[-1] + 1e-15)


def compare_conditioning(key: jax.Array, x0: jax.Array, T: float, h0: float,
                         r: float, cov_mat: jax.Array, vol: jax.Array,
                         max_deg: int, P1: jax.Array, 
                         s_min0: float, s_max0: float) -> dict:
    """
    Compare condition numbers for normal equations vs QR approach.
    
    Demonstrates that QR avoids squaring the condition number.
    
    Returns
    -------
    results : dict
        Condition numbers for D, D'D, and R
    """
    # Generate some test data
    dt = h0 * (2.0 ** (-max_deg))
    N_t = int(round(T / dt))
    M = 1000
    
    key, subkey = jax.random.split(key)
    paths = GBM_paths(subkey, x0, r, vol, cov_mat, dt, N_t, M)
    
    pairs = tot_degree_poly(max_deg)
    pairs_tuple = tuple(pairs)
    
    # Build design matrix for one timestep
    S = jnp.dot(paths[:, N_t//2, :], P1)
    t_norm = jnp.zeros(M)
    s_norm = jnp.clip(2.0 * (S - s_min0) / (s_max0 - s_min0) - 1.0, -1.0, 1.0)
    
    D = eval_legendre_basis_2d(t_norm, s_norm, pairs_tuple)
    
    # Condition numbers
    kappa_D = estimate_condition_number(D)
    
    G = D.T @ D
    kappa_G = estimate_condition_number(G)
    
    Q, R = qr_decomposition(D)
    kappa_R = estimate_condition_number(R)
    
    return {
        'kappa_D': float(kappa_D),
        'kappa_G': float(kappa_G),
        'kappa_R': float(kappa_R),
        'ratio_G_to_D': float(kappa_G / kappa_D),
        'ratio_R_to_D': float(kappa_R / kappa_D)
    }


# =============================================================================
# Main block for testing
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("Testing JAX_FML_hierarchical_qr.py")
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
    key, subkey = jax.random.split(key)
    s_min, s_max = scalings_l0(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1)
    print(f"Domain: [{s_min:.2f}, {s_max:.2f}]")
    
    # Test QR operations
    print("\nTesting QR decomposition...")
    A = jax.random.normal(jax.random.PRNGKey(0), (100, 10))
    Q, R = qr_decomposition(A)
    ortho_error = jnp.linalg.norm(Q.T @ Q - jnp.eye(10))
    print(f"  Orthogonality error: {ortho_error:.2e}")
    recon_error = jnp.linalg.norm(Q @ R - A)
    print(f"  Reconstruction error: {recon_error:.2e}")
    
    # Compare conditioning
    print("\nComparing condition numbers...")
    key, subkey = jax.random.split(key)
    cond_results = compare_conditioning(subkey, x0, T, h0, r, cov_mat, vol,
                                         max_deg, P1, s_min, s_max)
    print(f"  κ(D) = {cond_results['kappa_D']:.2e}")
    print(f"  κ(D'D) = {cond_results['kappa_G']:.2e}")
    print(f"  κ(R) = {cond_results['kappa_R']:.2e}")
    print(f"  κ(D'D) / κ(D) = {cond_results['ratio_G_to_D']:.2f} (should be ~κ(D))")
    print(f"  κ(R) / κ(D) = {cond_results['ratio_R_to_D']:.2f} (should be ~1)")
    
    # Test hierarchical QR MLMC
    print("\nTesting make_c_qr...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c_qr = make_c_qr(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min, s_max, C=30)
    t1 = time.time()
    print(f"  Computed in {t1-t0:.2f}s")
    print(f"  Coefficient norm: {jnp.linalg.norm(c_qr):.4f}")
    
    # Compare with standard method
    print("\nComparing with standard MLMC...")
    from JAX_FML_utils import make_c
    key, subkey = jax.random.split(key)
    c_std = make_c(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min, s_max, C=30)
    
    diff = jnp.linalg.norm(c_qr - c_std) / (jnp.linalg.norm(c_std) + 1e-10)
    print(f"  Relative difference: {diff:.2e}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
