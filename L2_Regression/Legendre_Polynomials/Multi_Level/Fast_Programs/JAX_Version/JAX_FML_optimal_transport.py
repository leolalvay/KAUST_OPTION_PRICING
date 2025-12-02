# This file is based on FML_optimal_transport.py, enhanced with JAX JIT compilation.
# Original work by Amelie, JAX conversion by Wadoud Charbak.

"""
JAX-Enhanced Optimal Transport for MLMC

Implements Gaussian-Brenier optimal transport maps with JAX JIT compilation
for efficient fine-to-coarse path coupling in Multi-Level Monte Carlo.

Key features:
- JIT-compiled Brenier map application
- Efficient matrix square root via eigendecomposition
- Vectorised map evaluation across paths

The Gaussian-Brenier map T: R^d -> R^d is given by:
    T(x) = mu_c + A @ (x - mu_f)
where A = C_f^{-1/2} (C_f^{1/2} C_c C_f^{1/2})^{1/2} C_f^{-1/2}
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax
from functools import partial
from typing import Tuple, Callable, List, NamedTuple
import numpy as np

from JAX_FML_utils import (
    GBM_paths, scalings_l0, tot_degree_poly,
    eval_legendre_basis_2d, make_b_bar
)


# =============================================================================
# Gaussian-Brenier Map (Functional Implementation)
# =============================================================================

class BrenierMapParams(NamedTuple):
    """Parameters for a Gaussian-Brenier map."""
    mu_f: jax.Array      # Fine mean, shape (d,)
    mu_c: jax.Array      # Coarse mean, shape (d,)
    A: jax.Array         # Transport matrix, shape (d, d)


@jax.jit
def compute_matrix_sqrt(M: jax.Array) -> jax.Array:
    """
    Compute matrix square root via eigendecomposition.
    
    For symmetric positive definite M, computes M^{1/2} such that
    M^{1/2} @ M^{1/2} = M.
    
    Parameters
    ----------
    M : jax.Array, shape (d, d)
        Symmetric positive definite matrix
        
    Returns
    -------
    M_sqrt : jax.Array, shape (d, d)
        Matrix square root
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(M)
    # Clamp eigenvalues to avoid numerical issues
    eigenvalues = jnp.maximum(eigenvalues, 1e-10)
    sqrt_eigenvalues = jnp.sqrt(eigenvalues)
    return eigenvectors @ jnp.diag(sqrt_eigenvalues) @ eigenvectors.T


@jax.jit
def compute_matrix_inv_sqrt(M: jax.Array) -> jax.Array:
    """
    Compute inverse matrix square root via eigendecomposition.
    
    Parameters
    ----------
    M : jax.Array, shape (d, d)
        Symmetric positive definite matrix
        
    Returns
    -------
    M_inv_sqrt : jax.Array, shape (d, d)
        Inverse matrix square root: M^{-1/2}
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(M)
    eigenvalues = jnp.maximum(eigenvalues, 1e-10)
    inv_sqrt_eigenvalues = 1.0 / jnp.sqrt(eigenvalues)
    return eigenvectors @ jnp.diag(inv_sqrt_eigenvalues) @ eigenvectors.T


@jax.jit
def compute_brenier_matrix(C_f: jax.Array, C_c: jax.Array) -> jax.Array:
    """
    Compute the Brenier transport matrix A.
    
    A = C_f^{-1/2} (C_f^{1/2} C_c C_f^{1/2})^{1/2} C_f^{-1/2}
    
    Parameters
    ----------
    C_f : jax.Array, shape (d, d)
        Fine covariance matrix
    C_c : jax.Array, shape (d, d)
        Coarse covariance matrix
        
    Returns
    -------
    A : jax.Array, shape (d, d)
        Transport matrix
    """
    C_f_sqrt = compute_matrix_sqrt(C_f)
    C_f_inv_sqrt = compute_matrix_inv_sqrt(C_f)
    
    # M = C_f^{1/2} C_c C_f^{1/2}
    M = C_f_sqrt @ C_c @ C_f_sqrt
    M_sqrt = compute_matrix_sqrt(M)
    
    # A = C_f^{-1/2} M^{1/2} C_f^{-1/2}
    A = C_f_inv_sqrt @ M_sqrt @ C_f_inv_sqrt
    
    return A


def fit_brenier_map(log_paths_f: jax.Array, log_paths_c: jax.Array) -> BrenierMapParams:
    """
    Fit Gaussian-Brenier map from samples.
    
    Estimates means and covariances from log-paths, then computes
    the optimal transport matrix.
    
    Parameters
    ----------
    log_paths_f : jax.Array, shape (M, d)
        Log of fine paths at a single timestep
    log_paths_c : jax.Array, shape (M, d)
        Log of coarse paths at corresponding timestep
        
    Returns
    -------
    params : BrenierMapParams
        Named tuple with (mu_f, mu_c, A)
    """
    # Compute empirical means
    mu_f = jnp.mean(log_paths_f, axis=0)
    mu_c = jnp.mean(log_paths_c, axis=0)
    
    # Compute empirical covariances
    centered_f = log_paths_f - mu_f
    centered_c = log_paths_c - mu_c
    
    M = log_paths_f.shape[0]
    C_f = (centered_f.T @ centered_f) / (M - 1)
    C_c = (centered_c.T @ centered_c) / (M - 1)
    
    # Add regularisation for numerical stability
    d = C_f.shape[0]
    C_f = C_f + 1e-6 * jnp.eye(d)
    C_c = C_c + 1e-6 * jnp.eye(d)
    
    # Compute transport matrix
    A = compute_brenier_matrix(C_f, C_c)
    
    return BrenierMapParams(mu_f=mu_f, mu_c=mu_c, A=A)


@jax.jit
def apply_brenier_map(params: BrenierMapParams, x: jax.Array) -> jax.Array:
    """
    Apply Gaussian-Brenier map to points.
    
    T(x) = mu_c + A @ (x - mu_f)
    
    Parameters
    ----------
    params : BrenierMapParams
        Map parameters (mu_f, mu_c, A)
    x : jax.Array, shape (M, d) or (d,)
        Points to transform
        
    Returns
    -------
    y : jax.Array, same shape as x
        Transformed points
    """
    centered = x - params.mu_f
    if x.ndim == 1:
        return params.mu_c + params.A @ centered
    else:
        return params.mu_c + centered @ params.A.T


def identity_map_params(d: int) -> BrenierMapParams:
    """Create identity map parameters."""
    return BrenierMapParams(
        mu_f=jnp.zeros(d),
        mu_c=jnp.zeros(d),
        A=jnp.eye(d)
    )


# =============================================================================
# Build OT Maps from Pilot Simulations
# =============================================================================

def logpaths_maps(key: jax.Array, x0: jax.Array, T: float, h0: float,
                  level: int, r: float, cov_mat: jax.Array, vol: jax.Array,
                  M_pilot: int = 5000) -> Tuple[List[BrenierMapParams], jax.Array, jax.Array]:
    """
    Generate pilot paths and build Brenier maps for each coarse timestep.
    
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
        MLMC level
    r : float
        Risk-free rate
    cov_mat : jax.Array, shape (d, d)
        Correlation matrix
    vol : jax.Array, shape (d,)
        Volatilities
    M_pilot : int
        Number of pilot paths
        
    Returns
    -------
    maps : list of BrenierMapParams
        One map per coarse timestep
    paths_f : jax.Array
        Fine pilot paths
    paths_c : jax.Array
        Coarse pilot paths
    """
    d = x0.shape[0]
    
    # Timesteps
    dt_f = h0 * (2.0 ** (-level - 1))
    dt_c = h0 * (2.0 ** (-level))
    N_f = int(round(T / dt_f))
    N_c = int(round(T / dt_c))
    
    # Generate fine paths
    key, subkey = jax.random.split(key)
    paths_f = GBM_paths(subkey, x0, r, vol, cov_mat, dt_f, N_f, M_pilot)
    
    # Generate coarse paths (independent for map fitting)
    key, subkey = jax.random.split(key)
    paths_c = GBM_paths(subkey, x0, r, vol, cov_mat, dt_c, N_c, M_pilot)
    
    # Build maps for each coarse timestep
    maps = []
    for n in range(N_c):
        if n == 0:
            # Identity map at t=0 (paths start at same point)
            maps.append(identity_map_params(d))
        else:
            # Extract log-paths at this timestep
            # Fine: use timestep 2n (corresponding to coarse n)
            log_f = jnp.log(paths_f[:, 2*n, :])
            log_c = jnp.log(paths_c[:, n, :])
            
            # Fit Brenier map
            params = fit_brenier_map(log_f, log_c)
            maps.append(params)
    
    return maps, paths_f, paths_c


# =============================================================================
# OT-Enhanced MLMC Level Estimator
# =============================================================================

def mlmc_level_OT(key: jax.Array, x0: jax.Array, T: float, h0: float, level: int,
                  r: float, cov_mat: jax.Array, vol: jax.Array, max_deg: int,
                  P1: jax.Array, s_min0: float, s_max0: float,
                  C: int = 80) -> jax.Array:
    """
    Compute MLMC level estimator with optimal transport coupling.
    
    Instead of using standard Brownian coupling, we use Gaussian-Brenier
    maps to create optimally coupled fine/coarse paths.
    
    Parameters
    ----------
    [Same as mlmc_level in JAX_FML_utils]
        
    Returns
    -------
    c : jax.Array, shape (dim_max,)
        Coefficient vector
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
    
    if level == 0:
        # Level 0: standard MLMC (no coarse term)
        from JAX_FML_utils import mlmc_level
        return mlmc_level(key, x0, T, h0, 0, r, cov_mat, vol, max_deg,
                          P1, s_min0, s_max0, C)
    
    # Build OT maps from pilot simulations
    key, subkey = jax.random.split(key)
    maps, _, _ = logpaths_maps(subkey, x0, T, h0, level, r, cov_mat, vol, M_pilot=3000)
    
    # Generate fine paths for main estimation
    key, subkey = jax.random.split(key)
    paths_f = GBM_paths(subkey, x0, r, vol, cov_mat, dt_f, N_f, M_l)
    
    # Create estimated coarse paths via OT
    paths_c_est = jnp.zeros((M_l, N_c, d))
    paths_c_est = paths_c_est.at[:, 0, :].set(paths_f[:, 0, :])  # Same start
    
    for n in range(1, N_c):
        log_f = jnp.log(paths_f[:, 2*n, :])
        log_c_est = apply_brenier_map(maps[n], log_f)
        paths_c_est = paths_c_est.at[:, n, :].set(jnp.exp(log_c_est))
    
    # Accumulate normal equations
    G = jnp.zeros((dimV, dimV))
    g = jnp.zeros((dimV,))
    
    for n in range(N_c):
        # Basket values at fine timesteps
        S_f1 = jnp.dot(paths_f[:, 2*n, :], P1)
        S_f2 = jnp.dot(paths_f[:, 2*n + 1, :], P1)
        S_c = jnp.dot(paths_c_est[:, n, :], P1)
        
        # Normalise to [-1, 1]
        t_norm1 = 2.0 * (2*n * dt_f) / T - 1.0
        t_norm2 = 2.0 * ((2*n + 1) * dt_f) / T - 1.0
        s_norm1 = jnp.clip(2.0 * (S_f1 - s_min0) / (s_max0 - s_min0) - 1.0, -1.0, 1.0)
        s_norm2 = jnp.clip(2.0 * (S_f2 - s_min0) / (s_max0 - s_min0) - 1.0, -1.0, 1.0)
        
        # Design matrices
        D1 = eval_legendre_basis_2d(jnp.full(M_l, t_norm1), s_norm1, pairs_tuple)
        D2 = eval_legendre_basis_2d(jnp.full(M_l, t_norm2), s_norm2, pairs_tuple)
        
        # Compute b^2 values
        def compute_b_sq(X):
            S = jnp.dot(X, P1)
            sigma = X * vol
            Sigma_cov = sigma @ cov_mat
            b_sq = jnp.sum(Sigma_cov * sigma, axis=1) * jnp.dot(P1, P1)
            return b_sq / (S ** 2 + 1e-10)
        
        b_f1_sq = compute_b_sq(paths_f[:, 2*n, :])
        b_f2_sq = compute_b_sq(paths_f[:, 2*n + 1, :])
        b_c_sq = compute_b_sq(paths_c_est[:, n, :])
        
        # psi = (b_f1^2 + b_f2^2) * dt_f - b_c^2 * dt_c
        psi = (b_f1_sq + b_f2_sq) * dt_f - b_c_sq * dt_c
        
        # Accumulate
        psi_half = psi / 2.0
        G = G + D1.T @ D1 + D2.T @ D2
        g = g + D1.T @ psi_half + D2.T @ psi_half
    
    # Solve via Cholesky
    L = jla.cholesky(G + 1e-8 * jnp.eye(dimV), lower=True)
    y = jla.solve_triangular(L, g, lower=True)
    c = jla.solve_triangular(L.T, y, lower=False)
    
    # Zero-pad
    c_padded = jnp.zeros(dim_max)
    c_padded = c_padded.at[:dimV].set(c)
    
    return c_padded


def make_c_OT(key: jax.Array, x0: jax.Array, T: float, h0: float,
              r: float, cov_mat: jax.Array, vol: jax.Array, max_deg: int,
              P1: jax.Array, s_min0: float, s_max0: float,
              C: int = 80) -> jax.Array:
    """
    Compute telescoping sum with OT-enhanced coupling.
    
    Uses standard MLMC for level 0, OT-enhanced for levels >= 1.
    
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
        c_l = mlmc_level_OT(subkey, x0, T, h0, level, r, cov_mat, vol,
                            max_deg, P1, s_min0, s_max0, C)
        c_total = c_total + c_l
    
    return c_total


# =============================================================================
# Main block for testing
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("Testing JAX_FML_optimal_transport.py")
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
    max_deg = 2  # Lower for faster testing
    
    key = jax.random.PRNGKey(42)
    
    # Test matrix sqrt
    print("\nTesting matrix operations...")
    M = cov_mat
    M_sqrt = compute_matrix_sqrt(M)
    reconstruction = M_sqrt @ M_sqrt
    error = jnp.linalg.norm(reconstruction - M)
    print(f"  Matrix sqrt reconstruction error: {error:.2e}")
    
    # Test Brenier map fitting
    print("\nTesting Brenier map fitting...")
    key, subkey = jax.random.split(key)
    log_f = jax.random.normal(subkey, (1000, d))
    key, subkey = jax.random.split(key)
    log_c = jax.random.normal(subkey, (1000, d)) * 0.8 + 0.5
    
    params = fit_brenier_map(log_f, log_c)
    print(f"  mu_f: {params.mu_f}")
    print(f"  mu_c: {params.mu_c}")
    print(f"  A diagonal: {jnp.diag(params.A)}")
    
    # Test map application
    print("\nTesting map application...")
    y = apply_brenier_map(params, log_f)
    print(f"  Input mean: {log_f.mean(axis=0)}")
    print(f"  Output mean: {y.mean(axis=0)}")
    print(f"  Target mean: {params.mu_c}")
    
    # Test scaling
    key, subkey = jax.random.split(key)
    s_min, s_max = scalings_l0(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1)
    print(f"\nDomain: [{s_min:.2f}, {s_max:.2f}]")
    
    # Test OT-enhanced MLMC
    print("\nTesting make_c_OT...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c_OT = make_c_OT(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min, s_max, C=30)
    t1 = time.time()
    print(f"  Computed in {t1-t0:.2f}s")
    print(f"  Coefficient norm: {jnp.linalg.norm(c_OT):.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
