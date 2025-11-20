# This file is based on DM_utils_ML.py from Amelie's work.

"""
Basket Option Simulation and Volatility Surface Estimation

Provides tools for:
1. Simulating correlated GBM paths for multi-asset baskets
2. Constructing polynomial regression systems for volatility fitting
3. Building projected volatility surfaces b(t,S) from estimated coefficients
"""

import numpy as np
import math
from numpy.polynomial.legendre import legvander, legval


def simulate_gbm_paths(S0, r, vol, cov_mat, dt, N_steps, N_paths):
    """
    Simulate correlated Geometric Brownian Motion paths for basket components.
    
    Uses Cholesky decomposition to generate correlated Brownian increments:
        dS_i = r S_i dt + vol_i S_i dW_i
    where dW are correlated with covariance matrix cov_mat.
    
    Parameters
    ----------
    S0 : np.ndarray, shape (d,) or (d, 1)
        Initial asset prices for d basket components
    r : float
        Risk-free interest rate
    vol : np.ndarray, shape (d,)
        Volatilities for each asset
    cov_mat : np.ndarray, shape (d, d)
        Correlation matrix (must be positive definite)
    dt : float
        Timestep size
    N_steps : int
        Number of timesteps (excluding initial point)
    N_paths : int
        Number of Monte Carlo paths to simulate
        
    Returns
    -------
    paths : np.ndarray, shape (N_paths, N_steps, d)
        Simulated asset price paths. paths[m, n, i] is the price of asset i
        at timestep n along path m.
        
    Notes
    -----
    - Uses Euler-Maruyama discretisation with Cholesky factorisation
    - Time complexity: O(N_paths * N_steps * d²) for Cholesky multiply
    """
    chol_correlation = np.linalg.cholesky(cov_mat)
    sqrt_dt = math.sqrt(dt)
    d = len(vol)
    
    # Initialise all paths at S0
    asset_prices = np.tile(S0.flatten(), (N_paths, 1))
    paths = np.empty((N_paths, N_steps, d))
    paths[:, 0, :] = asset_prices
    
    # Euler-Maruyama timestepping
    for n in range(1, N_steps):
        brownian_increments = np.random.randn(N_paths, d)
        sigma = asset_prices * vol  # Diagonal volatility scaling
        dW = brownian_increments @ chol_correlation.T  # Correlated increments
        
        asset_prices = asset_prices + r * asset_prices * dt + sigma * dW * sqrt_dt
        paths[:, n, :] = asset_prices
    
    return paths


def generate_polynomial_basis_pairs(max_degree=3):
    """
    Generate polynomial basis pairs (i, j) for total degree constraint.
    
    Returns all pairs (i, j) where i + j ≤ max_degree, used to construct
    tensor product Legendre polynomials P_i(t) * P_j(S).
    
    Parameters
    ----------
    max_degree : int, default 3
        Maximum total degree for polynomial basis
        
    Returns
    -------
    pairs : list of tuple
        List of (i, j) pairs satisfying i + j ≤ max_degree
        
    Examples
    --------
    >>> generate_polynomial_basis_pairs(max_degree=2)
    [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    
    Notes
    -----
    For max_degree=d, returns (d+1)(d+2)/2 basis functions.
    """
    return [
        (i, j) 
        for i in range(max_degree + 1) 
        for j in range(max_degree + 1) 
        if (i + j <= max_degree)
    ]


def construct_regression_system(
    paths_fine, 
    paths_coarse, 
    basket_weights, 
    basis_pairs, 
    cov_mat, 
    vol, 
    S_min, 
    S_max, 
    T
):
    """
    Construct normal equation components D and psi for volatility regression.
    
    Builds the system D @ c = psi where:
    - D: Design matrix of Legendre polynomials evaluated on paths
    - psi: Target vector of volatility differences (b_fine² - b_coarse²)
    
    The volatility b² for a basket is computed as:
        b² = (1/d²) * sum_{i,j} sigma_i sigma_j rho_{ij}
    where sigma_i = vol_i * S_i and rho_{ij} from cov_mat.
    
    Parameters
    ----------
    paths_fine : np.ndarray, shape (N_paths, N_fine, d)
        Fine timestep paths
    paths_coarse : np.ndarray, shape (N_paths, N_coarse, d)
        Coarse timestep paths (coupled with fine)
    basket_weights : np.ndarray, shape (d,)
        Weights for basket aggregation
    basis_pairs : list of tuple
        Polynomial basis pairs from generate_polynomial_basis_pairs()
    cov_mat : np.ndarray, shape (d, d)
        Asset correlation matrix
    vol : np.ndarray, shape (d,)
        Asset volatilities
    S_min, S_max : float
        Domain bounds for spatial scaling to [-1, 1]
    T : float
        Maturity time for temporal scaling to [-1, 1]
        
    Returns
    -------
    D : np.ndarray, shape (N_paths * N_coarse, n_basis)
        Design matrix with Legendre polynomials
    psi : np.ndarray, shape (N_paths * N_coarse, 1)
        Target volatility differences
        
    Notes
    -----
    - Fine paths are subsampled to coarse timesteps (every other point)
    - Legendre polynomials are orthonormalised on [-1, 1]
    - Scaling ensures numerical stability of regression
    """
    N_paths, N_coarse, d = paths_coarse.shape
    n_basis = len(basis_pairs)
    
    print(f"Basis pairs: {basis_pairs}")
    
    # Construct design matrix using fine paths at coarse time intervals
    t = np.linspace(0, T, N_coarse)
    t_scaled = 2 * t / T - 1  # Map [0, T] → [-1, 1]
    t_vals = np.tile(t_scaled, N_paths)
    
    # Subsample fine paths to coarse times (every 2nd timestep)
    # Note: We need exactly N_coarse timesteps, matching paths_coarse
    paths_fine_subsampled = paths_fine[:, ::2, :]
    
    # Ensure we have exactly N_coarse timesteps (handle off-by-one from ::2)
    if paths_fine_subsampled.shape[1] > N_coarse:
        paths_fine_subsampled = paths_fine_subsampled[:, :N_coarse, :]
    
    basket_vals = paths_fine_subsampled @ basket_weights  # Weighted basket
    S_scaled = 2 * (basket_vals.flatten() - S_min) / (S_max - S_min) - 1
    
    # Build Vandermonde matrices for Legendre polynomials
    max_degree = max(i for i, _ in basis_pairs)
    V_time = legvander(t_vals, max_degree)
    V_space = legvander(S_scaled, max_degree)
    
    # Orthonormalisation weights for Legendre polynomials on [-1, 1]
    norm_factors = np.sqrt((2 * np.arange(max_degree + 1) + 1) / 2)
    V_time *= norm_factors[None, :]
    V_space *= norm_factors[None, :]
    
    # Construct tensor product basis
    D = np.empty((N_paths * N_coarse, n_basis))
    for p, (i1, i2) in enumerate(basis_pairs):
        D[:, p] = V_time[:, i1] * V_space[:, i2]
    
    # Construct target vector psi = b_fine² - b_coarse²
    psi = np.empty((N_paths * N_coarse, 1))
    for m in range(N_paths):
        for n in range(N_coarse):
            idx = m * N_coarse + n
            
            asset_prices_coarse = paths_coarse[m, n]
            asset_prices_fine = paths_fine_subsampled[m, n]
            
            # Compute basket prices
            basket_price_coarse = asset_prices_coarse @ basket_weights
            basket_price_fine = asset_prices_fine @ basket_weights
            
            # Compute local volatility: sigma_diag @ cov_mat @ sigma_diag
            sigma_coarse = np.diag(vol * asset_prices_coarse)
            sigma_fine = np.diag(vol * asset_prices_fine)
            
            # Basket absolute variance (in currency units squared)
            var_abs_fine = (sigma_fine @ cov_mat @ sigma_fine).sum() / d**2
            var_abs_coarse = (sigma_coarse @ cov_mat @ sigma_coarse).sum() / d**2
            
            # Convert to relative volatility (percentage) by dividing by basket price
            # Handle level 0 case where coarse basket might be zero
            if basket_price_fine > 0:
                b_fine_sq = var_abs_fine / (basket_price_fine ** 2)
            else:
                b_fine_sq = 0.0
                
            if basket_price_coarse > 0:
                b_coarse_sq = var_abs_coarse / (basket_price_coarse ** 2)
            else:
                b_coarse_sq = 0.0
            
            psi[idx] = b_fine_sq - b_coarse_sq
    
    return D, psi


def fit_volatility_coefficients(D, psi):
    """
    Solve the normal equations D @ c = psi using QR decomposition.
    
    More numerically stable than normal equations (D.T @ D) @ c = D.T @ psi.
    
    Parameters
    ----------
    D : np.ndarray, shape (n_samples, n_basis)
        Design matrix from construct_regression_system()
    psi : np.ndarray, shape (n_samples, 1) or (n_samples,)
        Target vector (volatility differences)
        
    Returns
    -------
    c : np.ndarray, shape (n_basis,)
        Fitted polynomial coefficients
        
    Notes
    -----
    Uses reduced QR: D = QR where Q.T @ Q = I and R is upper triangular.
    Then c = R⁻¹ @ Q.T @ psi.
    """
    Q, R = np.linalg.qr(D, mode='reduced')
    alpha = Q.T @ psi
    c = np.linalg.solve(R, alpha).ravel()
    return c


def construct_volatility_surface(c, basis_pairs, S_min, S_max, T, max_degree):
    """
    Construct the projected volatility surface b(t, S) from fitted coefficients.
    
    The surface is represented as:
        b²(t, S) = sum_{p} c_p * P_{i1}(t) * P_{i2}(S)
    where P are orthonormalised Legendre polynomials.
    
    Parameters
    ----------
    c : np.ndarray, shape (n_basis,)
        Polynomial coefficients from fit_volatility_coefficients()
    basis_pairs : list of tuple
        Polynomial basis pairs (i, j)
    S_min, S_max : float
        Spatial domain bounds for scaling
    T : float
        Maturity time for temporal scaling
    max_degree : int
        Maximum polynomial degree (for normalisation array)
        
    Returns
    -------
    b_surface : callable
        Function b(t, S) that evaluates volatility surface.
        Accepts scalars or arrays for t and S.
        
    Notes
    -----
    - Input (t, S) are mapped to [-1, 1] before polynomial evaluation
    - Returns sqrt(h) where h = sum of polynomial basis functions
    - Warns if h becomes negative (indicates poor fit or extrapolation)
    """
    def b_surface(t, S):
        # Scale to [-1, 1]
        t_scaled = 2 * t / T - 1
        S_scaled = 2 * (S - S_min) / (S_max - S_min) - 1
        
        # Evaluate polynomial expansion
        h = 0.0
        norm_factors = np.sqrt((2 * np.arange(max_degree + 1) + 1) / 2)
        
        for p, (i1, i2) in enumerate(basis_pairs):
            P_time = legval(t_scaled, [0] * i1 + [1]) * norm_factors[i1]
            P_space = legval(S_scaled, [0] * i2 + [1]) * norm_factors[i2]
            h += c[p] * P_time * P_space
        
        # Check for negative variance (numerical issue or poor fit)
        if np.any(h < 0):
            idx = np.unravel_index(np.argmin(h), h.shape)
            bad_t = np.array(t)[idx] if np.ndim(t) > 0 else t
            bad_S = np.array(S)[idx] if np.ndim(S) > 0 else S
            bad_h = h[idx]
            print(f"⚠️  Negative variance h² = {bad_h:.6e} at t={bad_t:.3f}, S={bad_S:.2f}")
        
        return np.sqrt(np.maximum(h, 0.0))  # Clamp to avoid NaN
    
    return b_surface
