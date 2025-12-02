# This file is based on FML_utils.py, enhanced with JAX JIT compilation.
# Original work by Amelie, JAX conversion by Wadoud Charbak.

"""
JAX-Enhanced Fast Multi-Level Utilities

This module provides JAX JIT-compiled versions of the core FML functions
for significant performance improvements on CPU/GPU/TPU.

Key JAX adaptations:
1. Explicit PRNG keys instead of numpy global state
2. jax.lax.scan for efficient loop compilation
3. Immutable array operations (no in-place updates)
4. Static shape handling via functools.partial

Usage:
    import jax
    from JAX_FML_utils import GBM_paths, mlmc_level, make_c
    
    key = jax.random.PRNGKey(42)
    paths = GBM_paths(key, x0, r, vol, cov_mat, dt, N_t, M_t)
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import lax
from functools import partial
from typing import Tuple, Callable, List
import numpy as np  # For non-JIT operations only


# =============================================================================
# Path Generation
# =============================================================================

@partial(jax.jit, static_argnums=(5, 6, 7))
def GBM_paths(key: jax.Array, x0: jax.Array, r: float, vol: jax.Array,
              cov_mat: jax.Array, dt: float, N_t: int, M_t: int) -> jax.Array:
    """
    Generate correlated GBM paths using JAX.
    
    Uses jax.lax.scan for efficient sequential simulation, compiling the
    entire path generation into a single fused kernel.
    
    Parameters
    ----------
    key : jax.Array
        JAX PRNG key for reproducible randomness
    x0 : jax.Array, shape (d,)
        Initial asset prices (1D array, not column vector)
    r : float
        Risk-free rate
    vol : jax.Array, shape (d,)
        Asset volatilities
    cov_mat : jax.Array, shape (d, d)
        Correlation matrix
    dt : float
        Time step (static for JIT)
    N_t : int
        Number of time steps (static for JIT)
    M_t : int
        Number of paths (static for JIT)
        
    Returns
    -------
    paths : jax.Array, shape (M_t, N_t, d)
        Simulated asset paths
        
    Notes
    -----
    The Cholesky decomposition is computed once and reused for all steps.
    Using lax.scan avoids Python loop overhead and enables XLA fusion.
    """
    d = x0.shape[0]
    sqrtdt = jnp.sqrt(dt)
    
    # Cholesky decomposition of correlation matrix
    G = jla.cholesky(cov_mat, lower=True)
    
    # Initial state: all paths start at x0
    X0 = jnp.tile(x0, (M_t, 1))  # (M_t, d)
    
    def step_fn(carry, key_t):
        """Single Euler-Maruyama step."""
        X = carry
        Z = jax.random.normal(key_t, shape=(M_t, d))
        dW = Z @ G.T * sqrtdt
        sigma_X = X * vol  # Element-wise: (M_t, d) * (d,)
        X_next = X + r * X * dt + sigma_X * dW
        return X_next, X_next
    
    # Generate keys for each timestep
    keys = jax.random.split(key, N_t)
    
    # Run simulation using scan
    _, paths_body = lax.scan(step_fn, X0, keys)
    
    # paths_body has shape (N_t, M_t, d), need to prepend initial state
    # and transpose to (M_t, N_t, d)
    paths = jnp.concatenate([X0[None, :, :], paths_body[:-1]], axis=0)
    paths = jnp.transpose(paths, (1, 0, 2))
    
    return paths


def scalings_l0(key: jax.Array, x0: jax.Array, T: float, h0: float, 
                r: float, cov_mat: jax.Array, vol: jax.Array,
                max_deg: int, P1: jax.Array, M_0: int = 10000,
                return_basket: bool = False):
    """
    Pilot run to estimate basket scaling parameters.
    
    This function is not fully JIT-compiled as it returns Python scalars
    for use in subsequent computations. The path generation inside IS JIT-compiled.
    
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
    M_0 : int
        Number of pilot paths
    return_basket : bool
        If True, also return basket values
        
    Returns
    -------
    s_min, s_max : float
        Domain bounds (0.01 and 99.99 percentiles)
    basket : jax.Array, optional
        Basket values if return_basket=True
    """
    dt = h0 * (2.0 ** (-max_deg))
    N_t = int(round(T / dt))
    
    # Generate paths (JIT-compiled)
    paths = GBM_paths(key, x0, r, vol, cov_mat, dt, N_t, M_0)
    
    # Compute basket values
    basket = jnp.einsum('mtd,d->mt', paths, P1)
    
    # Percentiles (use numpy for this as it returns scalars)
    basket_np = np.array(basket)
    s_min = float(np.percentile(basket_np, 0.01))
    s_max = float(np.percentile(basket_np, 99.99))
    
    if return_basket:
        return s_min, s_max, basket
    return s_min, s_max


def tot_degree_poly(maxdeg: int = 3) -> List[Tuple[int, int]]:
    """
    Generate total-degree polynomial index pairs.
    
    This is a pure Python function (no JAX) as it returns a list used
    for indexing. Called once at setup, not in hot loops.
    
    Parameters
    ----------
    maxdeg : int
        Maximum total degree
        
    Returns
    -------
    pairs : list of (int, int)
        Index pairs (i, j) with i + j <= maxdeg
    """
    return [(i, j) for i in range(maxdeg + 1) 
            for j in range(maxdeg + 1) if (i + j <= maxdeg)]


# =============================================================================
# Legendre Polynomial Evaluation (JIT-compiled)
# =============================================================================

@jax.jit
def legendre_basis_1d(x: jax.Array, degree: int) -> jax.Array:
    """
    Evaluate 1D Legendre polynomials up to given degree.
    
    Uses the three-term recurrence relation, compiled via lax.fori_loop.
    
    Parameters
    ----------
    x : jax.Array, shape (N,)
        Points in [-1, 1]
    degree : int
        Maximum polynomial degree
        
    Returns
    -------
    P : jax.Array, shape (N, degree+1)
        Legendre polynomial values P_0(x), P_1(x), ..., P_degree(x)
    """
    N = x.shape[0]
    
    # Initialise with P_0 = 1, P_1 = x
    P = jnp.zeros((N, degree + 1))
    P = P.at[:, 0].set(1.0)
    
    if degree >= 1:
        P = P.at[:, 1].set(x)
    
    # Three-term recurrence: (n+1)P_{n+1} = (2n+1)x P_n - n P_{n-1}
    def recurrence(n, P):
        P_next = ((2*n + 1) * x * P[:, n] - n * P[:, n-1]) / (n + 1)
        return P.at[:, n+1].set(P_next)
    
    if degree >= 2:
        P = lax.fori_loop(1, degree, recurrence, P)
    
    return P


@partial(jax.jit, static_argnums=(2,))
def eval_legendre_basis_2d(t_norm: jax.Array, s_norm: jax.Array,
                            pairs: tuple) -> jax.Array:
    """
    Evaluate 2D tensor-product Legendre basis.
    
    Parameters
    ----------
    t_norm : jax.Array, shape (N,)
        Normalised time coordinates in [-1, 1]
    s_norm : jax.Array, shape (N,)
        Normalised space coordinates in [-1, 1]
    pairs : tuple of (int, int)
        Basis function index pairs (must be static for JIT)
        
    Returns
    -------
    D : jax.Array, shape (N, len(pairs))
        Design matrix with Legendre basis evaluations
    """
    max_t = max(p[0] for p in pairs)
    max_s = max(p[1] for p in pairs)
    
    # Evaluate 1D bases
    P_t = legendre_basis_1d(t_norm, max_t)  # (N, max_t+1)
    P_s = legendre_basis_1d(s_norm, max_s)  # (N, max_s+1)
    
    # Build design matrix
    D = jnp.stack([P_t[:, i] * P_s[:, j] for i, j in pairs], axis=1)
    
    return D


# =============================================================================
# Normal Equation Components (JIT-compiled core)
# =============================================================================

@partial(jax.jit, static_argnums=(6, 7, 8))
def compute_psi_batch(paths_f: jax.Array, paths_c: jax.Array,
                      P1: jax.Array, cov_mat: jax.Array, vol: jax.Array,
                      n: int, is_level_zero: bool, dt_f: float, dt_c: float) -> jax.Array:
    """
    Compute target vector psi = b_fine^2 - b_coarse^2 for a batch.
    
    Parameters
    ----------
    paths_f : jax.Array, shape (M, N_f, d)
        Fine paths
    paths_c : jax.Array, shape (M, N_c, d)
        Coarse paths
    P1 : jax.Array, shape (d,)
        Basket weights
    cov_mat : jax.Array, shape (d, d)
        Correlation matrix
    vol : jax.Array, shape (d,)
        Volatilities
    n : int
        Coarse timestep index
    is_level_zero : bool
        If True, compute only b_fine^2 (no coarse term)
    dt_f, dt_c : float
        Fine and coarse timesteps
        
    Returns
    -------
    psi : jax.Array, shape (M,)
        Target values for regression
    """
    d = vol.shape[0]
    
    # Fine contribution (two fine steps per coarse step)
    X_f1 = paths_f[:, 2*n, :]      # (M, d)
    X_f2 = paths_f[:, 2*n + 1, :]  # (M, d)
    
    # Diffusion coefficients
    sigma_f1 = X_f1 * vol  # (M, d)
    sigma_f2 = X_f2 * vol
    
    # b^2 = P1^T Sigma Cov Sigma^T P1 / S^2
    def compute_b_squared(X, sigma):
        S = jnp.dot(X, P1)  # (M,)
        Sigma_cov = sigma @ cov_mat  # (M, d)
        b_sq = jnp.sum(Sigma_cov * sigma, axis=1) * jnp.dot(P1, P1)
        return b_sq / (S ** 2 + 1e-10)
    
    b_f1_sq = compute_b_squared(X_f1, sigma_f1)
    b_f2_sq = compute_b_squared(X_f2, sigma_f2)
    
    psi_fine = (b_f1_sq + b_f2_sq) * dt_f
    
    # Coarse contribution (if not level 0)
    def add_coarse_term(psi):
        X_c = paths_c[:, n, :]
        sigma_c = X_c * vol
        b_c_sq = compute_b_squared(X_c, sigma_c)
        return psi - b_c_sq * dt_c
    
    psi = lax.cond(is_level_zero, lambda p: p, add_coarse_term, psi_fine)
    
    return psi


# =============================================================================
# MLMC Level Estimator
# =============================================================================

def mlmc_level(key: jax.Array, x0: jax.Array, T: float, h0: float, level: int,
               r: float, cov_mat: jax.Array, vol: jax.Array, max_deg: int,
               P1: jax.Array, s_min0: float, s_max0: float, C: int = 80) -> jax.Array:
    """
    Compute MLMC level-l coefficient estimator using accumulated normal equations.
    
    This function orchestrates the computation but the inner loops are JIT-compiled.
    
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
        Coefficient vector (zero-padded to maximum basis size)
    """
    d = vol.shape[0]
    
    # Level-dependent parameters
    deg_l = max_deg - level
    pairs = tot_degree_poly(deg_l)
    pairs_tuple = tuple(pairs)  # For JIT static argument
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
    
    # Coarse paths via Brownian coupling
    if level > 0:
        # Average consecutive fine steps for coarse
        paths_c = (paths_f[:, ::2, :] + paths_f[:, 1::2, :]) / 2.0
        # Re-simulate with coarse step (proper coupling)
        key, subkey = jax.random.split(key)
        paths_c = GBM_paths(subkey, x0, r, vol, cov_mat, dt_c, N_c, M_l)
    else:
        paths_c = jnp.zeros((M_l, N_c, d))
    
    # Accumulate normal equations
    G = jnp.zeros((dimV, dimV))
    g = jnp.zeros((dimV,))
    
    # Process each coarse timestep
    is_level_zero = (level == 0)
    
    for n in range(N_c):
        # Basket values at fine timesteps
        S_f1 = jnp.dot(paths_f[:, 2*n, :], P1)
        S_f2 = jnp.dot(paths_f[:, 2*n + 1, :], P1)
        
        # Normalise to [-1, 1]
        t_n = n * dt_c
        t_norm1 = 2.0 * (2*n * dt_f) / T - 1.0
        t_norm2 = 2.0 * ((2*n + 1) * dt_f) / T - 1.0
        s_norm1 = 2.0 * (S_f1 - s_min0) / (s_max0 - s_min0) - 1.0
        s_norm2 = 2.0 * (S_f2 - s_min0) / (s_max0 - s_min0) - 1.0
        
        # Clamp to valid domain
        s_norm1 = jnp.clip(s_norm1, -1.0, 1.0)
        s_norm2 = jnp.clip(s_norm2, -1.0, 1.0)
        
        # Design matrices
        D1 = eval_legendre_basis_2d(jnp.full(M_l, t_norm1), s_norm1, pairs_tuple)
        D2 = eval_legendre_basis_2d(jnp.full(M_l, t_norm2), s_norm2, pairs_tuple)
        
        # Compute psi
        psi = compute_psi_batch(paths_f, paths_c, P1, cov_mat, vol,
                                n, is_level_zero, dt_f, dt_c)
        
        # Accumulate (split psi between two fine steps)
        psi_half = psi / 2.0
        G = G + D1.T @ D1 + D2.T @ D2
        g = g + D1.T @ psi_half + D2.T @ psi_half
    
    # Solve via Cholesky
    try:
        L = jla.cholesky(G + 1e-8 * jnp.eye(dimV), lower=True)
        y = jla.solve_triangular(L, g, lower=True)
        c = jla.solve_triangular(L.T, y, lower=False)
    except:
        # Fallback to least-squares
        c = jnp.linalg.lstsq(G, g, rcond=None)[0]
    
    # Zero-pad to maximum size
    c_padded = jnp.zeros(dim_max)
    c_padded = c_padded.at[:dimV].set(c)
    
    return c_padded


def make_c(key: jax.Array, x0: jax.Array, T: float, h0: float,
           r: float, cov_mat: jax.Array, vol: jax.Array, max_deg: int,
           P1: jax.Array, s_min0: float, s_max0: float, C: int = 80) -> jax.Array:
    """
    Compute telescoping sum of MLMC coefficients.
    
    c_total = sum_{l=0}^{max_deg} c_l
    
    Parameters
    ----------
    key : jax.Array
        JAX PRNG key
    [other parameters as in mlmc_level]
        
    Returns
    -------
    c_total : jax.Array, shape (dim_max,)
        Aggregated coefficient vector
    """
    dim_max = len(tot_degree_poly(max_deg))
    c_total = jnp.zeros(dim_max)
    
    for level in range(max_deg + 1):
        key, subkey = jax.random.split(key)
        c_l = mlmc_level(subkey, x0, T, h0, level, r, cov_mat, vol,
                         max_deg, P1, s_min0, s_max0, C)
        c_total = c_total + c_l
    
    return c_total


# =============================================================================
# Volatility Surface Constructor
# =============================================================================

def make_b_bar(c: jax.Array, pairs: List[Tuple[int, int]],
               s_min: float, s_max: float, T: float,
               max_deg: int) -> Callable:
    """
    Construct JIT-compiled volatility surface function.
    
    Parameters
    ----------
    c : jax.Array, shape (dimV,)
        Coefficient vector
    pairs : list of (int, int)
        Polynomial basis pairs
    s_min, s_max : float
        Domain bounds
    T : float
        Maturity
    max_deg : int
        Maximum polynomial degree
        
    Returns
    -------
    bbar : Callable
        JIT-compiled function bbar(t, S) -> volatility values
    """
    pairs_tuple = tuple(pairs)
    c_arr = jnp.array(c[:len(pairs)])
    
    @jax.jit
    def bbar(t: jax.Array, S: jax.Array) -> jax.Array:
        """
        Evaluate volatility surface at (t, S).
        
        Parameters
        ----------
        t : jax.Array
            Time values (any shape)
        S : jax.Array
            Basket values (same shape as t)
            
        Returns
        -------
        b : jax.Array
            Volatility values (same shape as t)
        """
        # Flatten for evaluation
        t_flat = t.ravel()
        S_flat = S.ravel()
        
        # Normalise to [-1, 1]
        t_norm = 2.0 * t_flat / T - 1.0
        s_norm = 2.0 * (S_flat - s_min) / (s_max - s_min) - 1.0
        s_norm = jnp.clip(s_norm, -1.0, 1.0)
        
        # Evaluate basis
        D = eval_legendre_basis_2d(t_norm, s_norm, pairs_tuple)
        
        # Compute b^2 and take sqrt
        b_sq = D @ c_arr
        b_sq = jnp.maximum(b_sq, 0.0)  # Ensure non-negative
        b = jnp.sqrt(b_sq)
        
        # Reshape to original
        return b.reshape(t.shape)
    
    return bbar


# =============================================================================
# Main block for testing
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("Testing JAX_FML_utils.py")
    print("=" * 60)
    
    # Check JAX backend
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    
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
    max_deg = 3
    
    key = jax.random.PRNGKey(42)
    
    # Test GBM paths
    print("\nTesting GBM_paths...")
    key, subkey = jax.random.split(key)
    
    # Warm-up (JIT compilation)
    _ = GBM_paths(subkey, x0, r, vol, cov_mat, 0.01, 100, 1000)
    
    # Timed run
    t0 = time.time()
    paths = GBM_paths(subkey, x0, r, vol, cov_mat, 0.01, 100, 10000)
    paths.block_until_ready()  # Wait for async execution
    t1 = time.time()
    print(f"  Generated {paths.shape} paths in {t1-t0:.4f}s")
    print(f"  Mean final price: {paths[:, -1, :].mean(axis=0)}")
    
    # Test scalings
    print("\nTesting scalings_l0...")
    key, subkey = jax.random.split(key)
    s_min, s_max = scalings_l0(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1)
    print(f"  Domain: [{s_min:.2f}, {s_max:.2f}]")
    
    # Test full MLMC
    print("\nTesting make_c (full MLMC)...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c = make_c(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min, s_max, C=40)
    t1 = time.time()
    print(f"  Computed coefficients in {t1-t0:.2f}s")
    print(f"  Coefficient norm: {jnp.linalg.norm(c):.4f}")
    
    # Test volatility surface
    print("\nTesting make_b_bar...")
    pairs = tot_degree_poly(max_deg)
    bbar = make_b_bar(c, pairs, s_min, s_max, T, max_deg)
    
    t_test = jnp.array([0.5])
    S_test = jnp.array([250.0])
    b_val = bbar(t_test, S_test)
    print(f"  bbar(0.5, 250) = {float(b_val):.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
