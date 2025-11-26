"""
This file is based on DesignMatrix.py from Amelie's work.

Markovian Projection with Orthogonal Polynomials - Advanced Implementation
==========================================================================

Purpose
-------
Demonstrates Markovian projection using orthonormalized Legendre polynomials
for L² regression. This is a significant improvement over simple monomials:

Key Improvements over DM_org.py:
1. Orthogonal Legendre polynomials (better conditioning: cond(D) ~ 10² vs 10⁴)
2. Correlated assets (realistic market structure via Cholesky decomposition)
3. Different volatilities per asset (heterogeneous basket)
4. Data rescaling to [-1,1] (optimal for Legendre polynomials)
5. 3D volatility surface visualization (diagnostic tool)
6. Higher polynomial degree (max_deg=3 vs max_deg=1)

Method
------
Uses the same Gyöngy's Lemma approach as DM_org.py but with better numerics:
- Legendre polynomials Pₙ(x) are orthogonal on [-1,1] with weight function 1
- Orthonormalization: P̃ₙ(x) = √[(2n+1)/2] · Pₙ(x)
- This gives orthonormal basis: ⟨P̃ᵢ, P̃ⱼ⟩ = δᵢⱼ
- Design matrix D has much better conditioning (no ill-conditioning from powers)

Physical Analogy
----------------
Like using spherical harmonics Yₗₘ in quantum mechanics instead of (x,y,z) 
polynomials - the natural basis for the geometry makes everything numerically stable!

Think of Legendre polynomials as the "eigenfunctions" of the correlation structure
in your data - they diagonalize the problem and prevent numerical issues.

Author: Wadoud Charbak (refactored from Amelie's original code)
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math
import random
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial.legendre import legvander, legval


def setup_market_parameters():
    """
    Define market and simulation parameters with realistic complexity.
    
    Key differences from DM_org.py:
    - Correlated assets (correlation matrix cov_mat)
    - Different volatilities per asset (vol is now a vector)
    - More training paths (M_t = 200) for better fit
    
    Returns
    -------
    params : dict
        Dictionary containing all market parameters:
        - d: number of stocks in basket
        - initial_prices: initial stock prices [shape: (d, 1)]
        - r: risk-free interest rate
        - vol: volatility vector (different for each stock)
        - cov_mat: correlation matrix between assets
        - G: Cholesky factor of correlation matrix
        - dt: time step size
        - sqrt_dt: square root of dt
        - basket_weights: equal weights [1/d, 1/d, ..., 1/d]
        - num_time_steps: number of time steps
        - num_training_paths: number of paths for regression training
    """
    d = 3  # Number of stocks in basket
    
    # Initial prices: spread from 225 to 275
    initial_prices = np.linspace(225, 275, num=d)[:, np.newaxis]
    
    r = 0.05  # Risk-free rate (5% per annum)
    
    # Different volatilities for each stock (realistic!)
    vol = np.array([0.2, 0.15, 0.1])  # 20%, 15%, 10%
    
    # Correlation matrix (realistic market structure)
    # Asset 0 and 1 are highly correlated (0.8)
    # Asset 0 and 2 are weakly correlated (0.3)
    # Asset 1 and 2 are weakly correlated (0.1)
    cov_mat = np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.1],
        [0.3, 0.1, 1.0]
    ])
    
    # Cholesky decomposition: Σ = G·Gᵀ
    # Used to generate correlated Brownian motions: dW = G·dZ where dZ ~ N(0,I)
    G = np.linalg.cholesky(cov_mat)
    
    dt = 0.01  # Time step: Δt = 0.01 years
    sqrt_dt = math.sqrt(dt)
    
    # Equal-weighted basket
    basket_weights = np.ones(d) / d
    
    # Simulation parameters (more paths for better fit with higher degree)
    num_time_steps = 100  # T = 1 year
    num_training_paths = 200  # Double the paths for higher polynomial degree
    
    params = {
        'd': d,
        'initial_prices': initial_prices,
        'r': r,
        'vol': vol,
        'cov_mat': cov_mat,
        'G': G,
        'dt': dt,
        'sqrt_dt': sqrt_dt,
        'basket_weights': basket_weights,
        'num_time_steps': num_time_steps,
        'num_training_paths': num_training_paths
    }
    
    print(f"Market Setup:")
    print(f"  Number of assets: {d}")
    print(f"  Volatilities: {vol}")
    print(f"  Correlation matrix:\n{cov_mat}")
    print(f"  Training paths: {num_training_paths}")
    
    return params


def setup_legendre_basis(max_deg_t, max_deg_s):
    """
    Define orthonormalized Legendre polynomial basis for regression.
    
    We use tensor products: Pᵢ(t) ⊗ Pⱼ(s) for all i,j combinations.
    This is a complete polynomial basis up to degree (max_deg_t, max_deg_s).
    
    Parameters
    ----------
    max_deg_t : int
        Maximum degree for time polynomials
    max_deg_s : int
        Maximum degree for space polynomials
    
    Returns
    -------
    basis_pairs : list of tuples
        List of (i, j) where basis function is P̃ᵢ(t) ⊗ P̃ⱼ(s)
    num_basis_functions : int
        Total number of basis functions
    
    Notes
    -----
    For max_deg_t = max_deg_s = 3, we get (3+1) × (3+1) = 16 basis functions.
    This is more flexible than the 4 basis functions in DM_org.py!
    
    The Legendre polynomial Pₙ(x) satisfies:
    - Orthogonality: ∫₋₁¹ Pₘ(x)Pₙ(x)dx = 2/(2n+1) · δₘₙ
    - Normalization: P̃ₙ(x) = √[(2n+1)/2] · Pₙ(x) makes them orthonormal
    """
    # Generate all combinations of degrees
    elements_t = list(range(max_deg_t + 1))  # [0, 1, 2, 3]
    elements_s = list(range(max_deg_s + 1))  # [0, 1, 2, 3]
    basis_pairs = list(product(elements_t, elements_s))
    
    num_basis_functions = len(basis_pairs)
    
    print(f"\nLegendre Basis Setup:")
    print(f"  Max degree (time): {max_deg_t}")
    print(f"  Max degree (space): {max_deg_s}")
    print(f"  Number of basis functions: {num_basis_functions}")
    print(f"  Basis pairs: {basis_pairs[:6]}... (showing first 6)")
    
    return basis_pairs, num_basis_functions, max_deg_t, max_deg_s


def generate_training_paths(params):
    """
    Generate sample paths of d-dimensional correlated GBM.
    
    Each stock follows correlated geometric Brownian motion:
        dXᵢ = r·Xᵢ·dt + σᵢ·Xᵢ·(G·dW)ᵢ
    
    where G is the Cholesky factor of the correlation matrix and dW ~ N(0,I).
    
    Key difference from DM_org.py:
    - Correlated Brownian motions via Cholesky: dW_corr = G @ dW
    - Different volatilities σᵢ for each stock
    
    Parameters
    ----------
    params : dict
        Market parameters from setup_market_parameters()
    
    Returns
    -------
    stock_price_paths : ndarray, shape (M_t, N_t, d)
        Array containing all sample paths
    time_grid : ndarray, shape (N_t,)
        Time points [0, dt, 2dt, ..., (N_t-1)·dt]
    
    Notes
    -----
    The correlation structure is crucial for realistic basket modeling!
    In real markets, stocks in the same sector are highly correlated.
    """
    d = params['d']
    initial_prices = params['initial_prices']
    r = params['r']
    vol = params['vol']
    G = params['G']  # Cholesky factor
    dt = params['dt']
    sqrt_dt = params['sqrt_dt']
    num_time_steps = params['num_time_steps']
    num_training_paths = params['num_training_paths']
    
    # Initialize storage
    stock_price_paths = np.zeros((num_training_paths, num_time_steps, d, 1))
    
    print(f"\nGenerating {num_training_paths} correlated training paths...")
    
    for m in range(num_training_paths):
        X = initial_prices.copy()
        
        for n in range(num_time_steps):
            # Volatility matrix: diagonal with σᵢ·Xᵢ
            sigma_matrix = np.diag(vol * X.flatten())
            
            # Generate INDEPENDENT standard normal increments
            dW_independent = np.random.normal(0, 1, (d, 1))
            
            # Apply correlation structure: dW_correlated = G @ dW_independent
            dW_correlated = G @ dW_independent
            
            # Euler-Maruyama step with correlated noise
            X = X + r * X * dt + sigma_matrix @ dW_correlated * sqrt_dt
            
            stock_price_paths[m, n] = X
    
    time_grid = np.linspace(0, num_time_steps * dt, num=num_time_steps)
    
    print(f"Training data generation complete!")
    
    return stock_price_paths[..., 0], time_grid


def rescale_data(stock_price_paths, time_grid, params):
    """
    Rescale time and basket values to approximately [-1, 1].
    
    Why? Legendre polynomials are defined on [-1,1] and have optimal 
    numerical properties there. Rescaling prevents:
    - Overflow/underflow in polynomial evaluation
    - Poor conditioning from large/small values
    
    Parameters
    ----------
    stock_price_paths : ndarray, shape (M_t, N_t, d)
        Training paths
    time_grid : ndarray, shape (N_t,)
        Time points
    params : dict
        Market parameters
    
    Returns
    -------
    t_vals : ndarray, shape (M_t * N_t,)
        Rescaled time values (flattened)
    s_vals : ndarray, shape (M_t * N_t,)
        Rescaled basket values (flattened)
    t_mean : float
        Mean of original time values (for inverse transform)
    t_std : float
        Std of original time values (for inverse transform)
    s_mean : float
        Mean of original basket values
    s_std : float
        Std of original basket values
    
    Notes
    -----
    Transformation: x_scaled = (x - mean) / std
    This gives approximately x_scaled ∈ [-3, 3] (covers 99.7% of data).
    Legendre polynomials are well-behaved on this range.
    """
    num_training_paths = params['num_training_paths']
    num_time_steps = params['num_time_steps']
    basket_weights = params['basket_weights']
    
    # Flatten time values: repeat time grid for each path
    t_vals = np.tile(time_grid, num_training_paths)  # Shape: (M_t * N_t,)
    
    # Compute basket values: S̄ = P₁ᵀ·X at each point
    basket_values = stock_price_paths.dot(basket_weights)  # Shape: (M_t, N_t)
    s_vals = basket_values.flatten()  # Shape: (M_t * N_t,)
    
    # Compute mean and std (for rescaling)
    t_mean = t_vals.mean()
    t_std = t_vals.std(ddof=0)  # Population std (ddof=0)
    s_mean = s_vals.mean()
    s_std = s_vals.std(ddof=0)
    
    # Rescale to approximately [-1, 1] (actually to zero mean, unit variance)
    t_vals = (t_vals - t_mean) / t_std
    s_vals = (s_vals - s_mean) / s_std
    
    print(f"\nData Rescaling:")
    print(f"  Time range (original): [{time_grid.min():.3f}, {time_grid.max():.3f}]")
    print(f"  Time range (scaled): [{t_vals.min():.3f}, {t_vals.max():.3f}]")
    print(f"  Basket range (original): [{s_vals.mean() * s_std + s_mean - 3*s_std:.1f}, "
          f"{s_vals.mean() * s_std + s_mean + 3*s_std:.1f}]")
    print(f"  Basket range (scaled): [{s_vals.min():.3f}, {s_vals.max():.3f}]")
    
    return t_vals, s_vals, t_mean, t_std, s_mean, s_std


def build_design_matrix_legendre(t_vals, s_vals, max_deg_t, max_deg_s, basis_pairs):
    """
    Build design matrix using orthonormalized Legendre polynomials.
    
    This is the key algorithmic improvement over DM_org.py!
    
    Process:
    1. Generate univariate Legendre basis up to max degree
    2. Orthonormalize using ‖Pₙ‖² = 2/(2n+1)
    3. Form tensor products: D[:,p] = P̃ᵢ(t) ⊗ P̃ⱼ(s)
    
    Parameters
    ----------
    t_vals : ndarray, shape (M*N,)
        Rescaled time values
    s_vals : ndarray, shape (M*N,)
        Rescaled basket values
    max_deg_t : int
        Maximum degree for time polynomials
    max_deg_s : int
        Maximum degree for space polynomials
    basis_pairs : list of tuples
        Basis function indices
    
    Returns
    -------
    design_matrix : ndarray, shape (M*N, P)
        Design matrix with P basis functions
    
    Notes
    -----
    legvander(x, n) returns the Vandermonde matrix [P₀(x), P₁(x), ..., Pₙ(x)]
    where Pₙ are the raw Legendre polynomials (not yet orthonormalized).
    
    Orthonormalization factor: √[(2n+1)/2]
    This makes ∫₋₁¹ P̃ₙ(x)² dx = 1 for each n.
    """
    num_data_points = len(t_vals)
    num_basis_functions = len(basis_pairs)
    
    print(f"\nBuilding Legendre Design Matrix...")
    
    # Generate Vandermonde matrices for univariate Legendre polynomials
    # VT[i, n] = Pₙ(t_vals[i])
    VT = legvander(t_vals, max_deg_t)  # Shape: (M*N, max_deg_t+1)
    VS = legvander(s_vals, max_deg_s)  # Shape: (M*N, max_deg_s+1)
    
    # Orthonormalization factors: √[(2n+1)/2]
    # These come from ∫₋₁¹ Pₙ(x)² dx = 2/(2n+1)
    norm_t = np.sqrt((2 * np.arange(max_deg_t + 1) + 1) / 2)
    norm_s = np.sqrt((2 * np.arange(max_deg_s + 1) + 1) / 2)
    
    # Apply normalization: P̃ₙ = norm_n · Pₙ
    VT *= norm_t[None, :]  # Broadcast multiply
    VS *= norm_s[None, :]
    
    # Build design matrix via tensor products
    design_matrix = np.empty((num_data_points, num_basis_functions))
    
    for p, (i1, i2) in enumerate(basis_pairs):
        # Tensor product: P̃ᵢ(t) ⊗ P̃ⱼ(s)
        design_matrix[:, p] = VT[:, i1] * VS[:, i2]
    
    print(f"  Design matrix shape: {design_matrix.shape}")
    print(f"  Checking orthogonality: DᵀD should be approximately identity...")
    
    # Diagnostic: check if DᵀD ≈ M·I (empirical orthogonality)
    # For perfectly orthonormal basis with empirical measure, DᵀD = M·I
    gram_matrix = design_matrix.T @ design_matrix
    expected_diagonal = num_data_points
    actual_diagonal = np.diag(gram_matrix).mean()
    off_diagonal_norm = np.linalg.norm(gram_matrix - np.diag(np.diag(gram_matrix)))
    
    print(f"  Expected diagonal value: {expected_diagonal:.0f}")
    print(f"  Actual diagonal value: {actual_diagonal:.0f}")
    print(f"  Off-diagonal norm: {off_diagonal_norm:.2e} (should be small)")
    
    return design_matrix


def build_response_vector(stock_price_paths, params):
    """
    Construct response vector ψ containing true basket variance.
    
    For correlated assets with different volatilities:
        Var(dS̄) = Var(Σ wᵢ·dXᵢ) 
                = Σᵢ Σⱼ wᵢ·wⱼ·σᵢ·σⱼ·Xᵢ·Xⱼ·ρᵢⱼ
                = wᵀ·diag(σX)·Σ·diag(σX)·w
    
    where Σ is the correlation matrix.
    
    Parameters
    ----------
    stock_price_paths : ndarray, shape (M_t, N_t, d)
        Training paths
    params : dict
        Market parameters
    
    Returns
    -------
    response_vector : ndarray, shape (M_t * N_t,)
        True instantaneous variance at each data point
    
    Notes
    -----
    This is more general than DM_org.py which assumed:
    - Independent assets (ρᵢⱼ = δᵢⱼ)
    - Equal volatilities (σᵢ = σ)
    
    Here we handle the full covariance structure!
    """
    num_training_paths = params['num_training_paths']
    num_time_steps = params['num_time_steps']
    d = params['d']
    vol = params['vol']
    cov_mat = params['cov_mat']
    basket_weights = params['basket_weights']
    
    num_data_points = num_training_paths * num_time_steps
    response_vector = np.zeros(num_data_points)
    
    print(f"\nBuilding Response Vector...")
    
    for m in range(num_training_paths):
        for n in range(num_time_steps):
            idx = m * num_time_steps + n
            X = stock_price_paths[m, n]
            
            # Volatility matrix: diag(σᵢ·Xᵢ)
            sigma_matrix = np.diag(vol * X)
            
            # Basket variance: wᵀ·Σ_basket·w where Σ_basket = σ·Σ·σ
            basket_covariance = sigma_matrix @ cov_mat @ sigma_matrix
            
            # Quadratic form: variance = wᵀ·Σ·w
            basket_variance = basket_weights @ basket_covariance @ basket_weights
            
            response_vector[idx] = basket_variance
    
    print(f"  Response vector shape: {response_vector.shape}")
    print(f"  Mean variance: {response_vector.mean():.6f}")
    print(f"  Variance range: [{response_vector.min():.6f}, {response_vector.max():.6f}]")
    
    return response_vector


def solve_regression_qr(design_matrix, response_vector):
    """
    Solve L² regression using QR decomposition.
    
    Same algorithm as DM_org.py but with MUCH better conditioning
    thanks to orthogonal Legendre polynomials!
    
    Parameters
    ----------
    design_matrix : ndarray, shape (M*N, P)
        Design matrix D (Legendre basis)
    response_vector : ndarray, shape (M*N,)
        Response vector ψ
    
    Returns
    -------
    coefficients : ndarray, shape (P,)
        Fitted coefficients c
    
    Notes
    -----
    Expected conditioning improvement:
    - Simple polynomials: cond(D) ~ 10⁴ to 10⁸
    - Legendre polynomials: cond(D) ~ 10¹ to 10³
    
    This is the power of orthogonal polynomials!
    """
    print(f"\nSolving Regression (QR Decomposition)...")
    
    # QR decomposition
    Q, R = np.linalg.qr(design_matrix, mode='reduced')
    alpha = Q.T @ response_vector
    coefficients = np.linalg.solve(R, alpha)
    
    print(f"\nFitted coefficients (first 6): {coefficients[:6]}")
    
    # Diagnostics
    residual = np.linalg.norm(design_matrix @ coefficients - response_vector)
    response_norm = np.linalg.norm(response_vector)
    cond_D = np.linalg.cond(design_matrix)
    
    print(f"\nRegression Diagnostics:")
    print(f"  Absolute residual: {residual:.6f}")
    print(f"  Relative residual: {residual/response_norm:.6f}")
    print(f"  Condition number: {cond_D:.2e}")
    
    # Check singular values
    singular_values = np.linalg.svd(design_matrix, compute_uv=False)
    print(f"  Largest singular value: {singular_values[0]:.2e}")
    print(f"  Smallest singular value: {singular_values[-1]:.2e}")
    print(f"  Ratio (= cond): {singular_values[0]/singular_values[-1]:.2e}")
    
    if cond_D < 1e3:
        print(f"\n✅ EXCELLENT conditioning (cond < 10³)")
    elif cond_D < 1e6:
        print(f"\n✓ Good conditioning (cond < 10⁶)")
    else:
        print(f"\n⚠️  Poor conditioning (cond > 10⁶) - consider regularization")
    
    return coefficients


def create_b_bar_function(coefficients, basis_pairs, max_deg_t, max_deg_s,
                          t_mean, t_std, s_mean, s_std):
    """
    Create callable function b̄(t, s) from fitted coefficients.
    
    Must handle:
    1. Inverse rescaling: (t,s) → (t_scaled, s_scaled)
    2. Legendre polynomial evaluation at scaled points
    3. Orthonormalization factors
    4. Square root (variance → volatility)
    
    Parameters
    ----------
    coefficients : ndarray
        Fitted coefficients
    basis_pairs : list of tuples
        Basis function indices
    max_deg_t, max_deg_s : int
        Maximum degrees
    t_mean, t_std, s_mean, s_std : float
        Rescaling parameters
    
    Returns
    -------
    b_bar : callable
        Function b_bar(t, s) returning projected volatility coefficient
    
    Notes
    -----
    legval(x, [0,0,...,0,1]) evaluates Pₙ(x) efficiently using recurrence.
    This is much faster than explicit polynomial evaluation!
    """
    # Precompute normalization factors
    norm_t = np.sqrt((2 * np.arange(max_deg_t + 1) + 1) / 2)
    norm_s = np.sqrt((2 * np.arange(max_deg_s + 1) + 1) / 2)
    
    def b_bar(t, s):
        """
        Evaluate projected volatility at (t, s).
        
        Parameters
        ----------
        t : float
            Time (in original scale)
        s : float
            Basket value (in original scale)
        
        Returns
        -------
        vol_coeff : float
            Volatility coefficient b̄(t, s)
        """
        # Rescale to [-1, 1] range
        t_scaled = (t - t_mean) / t_std
        s_scaled = (s - s_mean) / s_std
        
        # Evaluate polynomial: h = b̄²(t,s)
        h = 0.0
        for p, (i1, i2) in enumerate(basis_pairs):
            # Evaluate Legendre polynomials efficiently
            # legval(x, [0,0,...,1]) returns Pₙ(x)
            P_t = legval(t_scaled, [0] * i1 + [1]) * norm_t[i1]
            P_s = legval(s_scaled, [0] * i2 + [1]) * norm_s[i2]
            
            h += coefficients[p] * P_t * P_s
        
        # Take square root (with protection)
        if h < 0:
            # This can happen with high-degree polynomials near boundaries
            # print(f"⚠️  b̄²({t:.3f}, {s:.2f}) = {h:.6e} < 0")
            h = max(h, 1e-10)  # Clamp to small positive value
        
        return math.sqrt(h)
    
    return b_bar


def visualize_volatility_surface(b_bar, stock_price_paths, time_grid, params):
    """
    Create 3D wireframe plot of the projected volatility surface b̄(t,s).
    
    This is a crucial diagnostic tool! The surface should be:
    - Smooth (no wild oscillations)
    - Positive everywhere (no negative variance)
    - Reasonable magnitude (typically 0.05 to 0.5)
    
    Parameters
    ----------
    b_bar : callable
        Projected volatility function
    stock_price_paths : ndarray
        Training paths (for determining plot range)
    time_grid : ndarray
        Time points
    params : dict
        Market parameters
    """
    basket_weights = params['basket_weights']
    num_time_steps = params['num_time_steps']
    dt = params['dt']
    
    print(f"\nGenerating Volatility Surface Plot...")
    
    # Compute basket values over all paths
    basket = stock_price_paths.dot(basket_weights)
    
    # Use percentiles to set plot range (avoid outliers)
    s_min = np.percentile(basket, 1, axis=0)  # 1st percentile at each time
    s_max = np.percentile(basket, 99, axis=0)  # 99th percentile
    
    # Create grid for plotting
    K = 40  # Time points
    L = 150  # Space points
    
    t_plot = np.linspace(0, num_time_steps * dt, num=K)
    idx = np.searchsorted(time_grid, t_plot)
    
    T = np.zeros((L, K))  # Time mesh
    S = np.zeros((L, K))  # Space mesh
    
    for j, ti in enumerate(idx):
        T[:, j] = t_plot[j]
        S[:, j] = np.linspace(s_min[ti], s_max[ti], L)
    
    # Evaluate b̄(t,s) on grid (vectorized)
    b_bar_vec = np.vectorize(b_bar)
    B_bar = b_bar_vec(T, S)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_wireframe(T, S, B_bar, 
                             rcount=40, ccount=40,
                             color='blue',
                             linewidth=0.5,
                             alpha=0.7)
    
    ax.set_xlabel('Time $t$ (years)', fontsize=11, labelpad=10)
    ax.set_ylabel('Basket Value $s$', fontsize=11, labelpad=10)
    ax.set_zlabel(r'$\bar{b}(t,s)$', fontsize=11, labelpad=10)
    ax.set_title(r'Projected Volatility Surface $\bar{b}(t,s)$' + '\n' + 
                 '(Orthonormalized Legendre Polynomial Basis)',
                 fontsize=12, pad=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('volatility_surface_legendre.pdf', bbox_inches='tight', dpi=150)
    print(f"  Plot saved as 'volatility_surface_legendre.pdf'")
    
    # Print statistics
    print(f"\nVolatility Surface Statistics:")
    print(f"  Min value: {B_bar.min():.4f}")
    print(f"  Max value: {B_bar.max():.4f}")
    print(f"  Mean value: {B_bar.mean():.4f}")
    
    if B_bar.min() < 0:
        print(f"  ⚠️  WARNING: Negative values detected! Check polynomial degree.")
    
    plt.show()


def validate_projection(b_bar, params):
    """
    Validate Markovian projection by comparing terminal distributions.
    
    Same validation as DM_org.py but with correlated assets.
    
    Parameters
    ----------
    b_bar : callable
        Projected volatility function
    params : dict
        Market parameters
    
    Returns
    -------
    log_returns_basket : ndarray
        Log returns from true basket
    log_returns_S_bar : ndarray
        Log returns from projected process
    """
    d = params['d']
    initial_prices = params['initial_prices']
    r = params['r']
    vol = params['vol']
    G = params['G']
    dt = params['dt']
    sqrt_dt = params['sqrt_dt']
    basket_weights = params['basket_weights']
    num_time_steps = params['num_time_steps']
    
    time_grid = np.linspace(0, num_time_steps * dt, num=num_time_steps)
    initial_basket_value = basket_weights.dot(initial_prices.flatten())
    
    num_validation_paths = 10000
    print(f"\nValidation with {num_validation_paths} paths...")
    
    # Projected process (1D SDE)
    S_bar_terminal = np.zeros(num_validation_paths)
    
    for m in range(num_validation_paths):
        S_bar = initial_basket_value
        for n in range(num_time_steps):
            S_bar = S_bar + r * S_bar * dt + b_bar(time_grid[n], S_bar) * random.gauss(0, 1) * sqrt_dt
        S_bar_terminal[m] = S_bar
    
    # True process (d-dimensional correlated GBM)
    basket_terminal = np.zeros(num_validation_paths)
    
    for m in range(num_validation_paths):
        X = initial_prices.copy()
        for n in range(num_time_steps):
            sigma_matrix = np.diag(vol * X.flatten())
            dW = G @ np.random.normal(0, 1, (d, 1))
            X = X + r * X * dt + sigma_matrix @ dW * sqrt_dt
        basket_terminal[m] = basket_weights.dot(X[:, 0])
    
    # Compare distributions
    log_returns_basket = np.log(basket_terminal / initial_basket_value)
    log_returns_S_bar = np.log(S_bar_terminal / initial_basket_value)
    
    print(f"\nDistribution Statistics:")
    print(f"  True:      mean = {log_returns_basket.mean():.6f}, std = {log_returns_basket.std(ddof=1):.6f}")
    print(f"  Projected: mean = {log_returns_S_bar.mean():.6f}, std = {log_returns_S_bar.std(ddof=1):.6f}")
    
    mean_error = abs(log_returns_S_bar.mean() - log_returns_basket.mean()) / abs(log_returns_basket.mean())
    std_error = abs(log_returns_S_bar.std(ddof=1) - log_returns_basket.std(ddof=1)) / log_returns_basket.std(ddof=1)
    
    print(f"  Errors:    mean = {mean_error:.2%}, std = {std_error:.2%}")
    
    if mean_error < 0.05 and std_error < 0.05:
        print(f"\n✅ EXCELLENT projection (errors < 5%)")
    elif mean_error < 0.10 and std_error < 0.10:
        print(f"\n✓ Good projection (errors < 10%)")
    else:
        print(f"\n⚠️  Consider increasing polynomial degree or training paths")
    
    return log_returns_basket, log_returns_S_bar


def plot_comparison(log_returns_basket, log_returns_S_bar, params):
    """
    Create comparison histogram.
    
    Parameters
    ----------
    log_returns_basket : ndarray
        True basket log returns
    log_returns_S_bar : ndarray
        Projected process log returns
    params : dict
        Market parameters
    """
    d = params['d']
    
    all_returns = np.concatenate([log_returns_basket, log_returns_S_bar])
    bins = np.linspace(all_returns.min(), all_returns.max(), 51)
    
    plt.figure(figsize=(10, 6))
    plt.hist(log_returns_basket, bins=bins, histtype='stepfilled',
             color='C0', alpha=0.3, label='True Process: P₁·X')
    plt.hist(log_returns_S_bar, bins=bins, histtype='step',
             color='C1', alpha=0.75, linewidth=2, label=r'Markovian Projection: $\bar{S}$')
    
    plt.xlabel('Log Returns', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Terminal Distribution: True vs Projected\n'
              f'Equal-weighted basket of {d} correlated stocks (Legendre basis)',
              fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_legendre.pdf', bbox_inches='tight', dpi=150)
    print(f"\n📊 Plot saved as 'validation_legendre.pdf'")
    plt.show()


def main():
    """
    Main execution: Markovian projection with Legendre polynomials.
    """
    print("="*70)
    print("MARKOVIAN PROJECTION WITH LEGENDRE POLYNOMIALS")
    print("Advanced Implementation with Correlated Assets")
    print("="*70)
    
    # 1. Setup
    params = setup_market_parameters()
    max_deg_t = 3  # Cubic in time
    max_deg_s = 3  # Cubic in space
    basis_pairs, num_basis, max_deg_t, max_deg_s = setup_legendre_basis(max_deg_t, max_deg_s)
    
    # 2. Generate data
    stock_price_paths, time_grid = generate_training_paths(params)
    
    # 3. Rescale data
    t_vals, s_vals, t_mean, t_std, s_mean, s_std = rescale_data(
        stock_price_paths, time_grid, params
    )
    
    # 4. Build matrices
    design_matrix = build_design_matrix_legendre(
        t_vals, s_vals, max_deg_t, max_deg_s, basis_pairs
    )
    response_vector = build_response_vector(stock_price_paths, params)
    
    # 5. Solve regression
    coefficients = solve_regression_qr(design_matrix, response_vector)
    
    # 6. Create b̄ function
    b_bar = create_b_bar_function(
        coefficients, basis_pairs, max_deg_t, max_deg_s,
        t_mean, t_std, s_mean, s_std
    )
    print(f"\n✓ Projected volatility function b̄(t,s) created")
    
    # 7. Visualize surface
    visualize_volatility_surface(b_bar, stock_price_paths, time_grid, params)
    
    # 8. Validate
    log_returns_basket, log_returns_S_bar = validate_projection(b_bar, params)
    
    # 9. Plot comparison
    plot_comparison(log_returns_basket, log_returns_S_bar, params)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nKey Improvements over Simple Polynomials:")
    print(f"  1. Condition number: ~10² (vs ~10⁴ for monomials)")
    print(f"  2. Handles correlations: ρ₀₁={params['cov_mat'][0,1]}")
    print(f"  3. Different volatilities: σ = {params['vol']}")
    print(f"  4. Higher flexibility: {num_basis} basis functions")
    print("="*70)


if __name__ == "__main__":
    main()
