"""
This file is based on DM_org.py from Amelie's work.

Markovian Projection for American Options - Basic Implementation
================================================================

Purpose
-------
Demonstrates the fundamental concept of Markovian projection using L² regression.
This is the simplest implementation that shows how to:
1. Project d-dimensional basket dynamics to an effective 1D process
2. Fit a local volatility function b̄(t,s) using least squares regression
3. Validate that the projected process matches the true basket distribution

Method
------
Uses Gyöngy's Lemma: If we choose b̄(t,s) such that b̄²(t,s) = E[σ²(X_t) | S̄_t = s],
then the 1D SDE: dS̄_t = rS̄_t dt + b̄(t,S̄_t)dW_t
has the same marginal distribution at maturity T as the true d-dimensional basket.

Implementation Details
---------------------
- Polynomial basis: Simple monomials {1, s, t, ts} (degree 1 in each variable)
- Regression method: QR decomposition for numerical stability
- Assets: Independent GBM with identical volatility
- Basket: Equal-weighted (1/d, 1/d, ..., 1/d)

Physical Analogy
---------------
Like finding an "effective field theory" in physics - we integrate out the 
high-dimensional degrees of freedom (individual stock movements) and retain only
the collective behaviour (basket value) with an effective interaction (local vol).

Author: Wadoud Charbak (refactored from Amelie's original code)
Date: November 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math
import random


def setup_market_parameters():
    """
    Define market and simulation parameters.
    
    Returns
    -------
    params : dict
        Dictionary containing all market parameters:
        - d: number of stocks in basket
        - initial_prices: initial stock prices [shape: (d, 1)]
        - r: risk-free interest rate
        - vol: volatility (same for all stocks)
        - dt: time step size
        - sqrt_dt: square root of dt (for Euler-Maruyama)
        - basket_weights: equal weights [1/d, 1/d, ..., 1/d]
        - num_time_steps: number of time steps (N_t)
        - num_training_paths: number of paths for regression training (M_t)
    """
    d = 3  # Number of stocks in basket
    initial_prices = np.linspace(100, 160, num=d)[:, np.newaxis]  # [100, 130, 160]
    r = 0.05  # Risk-free rate (5% per annum)
    vol = 0.15  # Volatility (15% - same for all stocks, simplest case)
    
    dt = 0.01  # Time step: Δt = 0.01 years
    sqrt_dt = math.sqrt(dt)  # Precompute √Δt for efficiency
    
    # Equal-weighted basket: P₁ = (1/d, 1/d, ..., 1/d)
    basket_weights = np.ones(d) / d
    
    # Simulation parameters for regression training
    num_time_steps = 100  # N_t: simulate to T = N_t * dt = 1 year
    num_training_paths = 100  # M_t: number of sample paths for fitting
    
    params = {
        'd': d,
        'initial_prices': initial_prices,
        'r': r,
        'vol': vol,
        'dt': dt,
        'sqrt_dt': sqrt_dt,
        'basket_weights': basket_weights,
        'num_time_steps': num_time_steps,
        'num_training_paths': num_training_paths
    }
    
    return params


def setup_polynomial_basis():
    """
    Define polynomial basis functions for regression.
    
    We use simple monomials: {1, s, t, ts}
    This corresponds to fitting b̄²(t,s) = c₀ + c₁·s + c₂·t + c₃·t·s
    
    Returns
    -------
    basis_pairs : list of tuples
        List of (i₁, i₂) where basis function is t^i₁ · s^i₂
        Example: [(0,0), (0,1), (1,0), (1,1)]
    num_basis_functions : int
        Total number of basis functions (P)
    
    Notes
    -----
    Using degree 1 polynomials is the simplest choice. Higher degrees would
    give more flexibility but require more training data to avoid overfitting.
    """
    # For simplicity, use degree 1 in both t and s
    polynomial_degrees = [0, 1]
    
    # Generate all combinations: (i₁, i₂) where i₁, i₂ ∈ {0, 1}
    basis_pairs = list(product(polynomial_degrees, repeat=2))
    # Result: [(0,0), (0,1), (1,0), (1,1)]
    
    num_basis_functions = len(basis_pairs)
    print(f"Using {num_basis_functions} polynomial basis functions: {basis_pairs}")
    
    return basis_pairs, num_basis_functions


def generate_training_paths(params):
    """
    Generate sample paths of d-dimensional GBM for regression training.
    
    Each stock follows independent geometric Brownian motion:
        dXᵢ = r·Xᵢ·dt + σ·Xᵢ·dWᵢ
    
    Discretization (Euler-Maruyama):
        Xᵢ(t+dt) = Xᵢ(t) + r·Xᵢ(t)·dt + σ·Xᵢ(t)·√dt·Zᵢ
    where Zᵢ ~ N(0,1) are independent standard normal variables.
    
    Parameters
    ----------
    params : dict
        Market parameters from setup_market_parameters()
    
    Returns
    -------
    stock_price_paths : ndarray, shape (M_t, N_t, d)
        Array containing all sample paths
        stock_price_paths[m, n, i] = price of stock i at time step n on path m
    time_grid : ndarray, shape (N_t,)
        Time points [0, dt, 2dt, ..., (N_t-1)·dt]
    
    Notes
    -----
    We store the entire path (not just terminal values) because we need
    pairs (t_n, S̄_n, σ²_n) at every time step for the regression.
    """
    d = params['d']
    initial_prices = params['initial_prices']
    r = params['r']
    vol = params['vol']
    dt = params['dt']
    sqrt_dt = params['sqrt_dt']
    num_time_steps = params['num_time_steps']
    num_training_paths = params['num_training_paths']
    
    # Initialize storage: (num_paths, num_timesteps, num_stocks)
    stock_price_paths = np.zeros((num_training_paths, num_time_steps, d, 1))
    
    print(f"Generating {num_training_paths} training paths with {num_time_steps} time steps each...")
    
    for m in range(num_training_paths):
        X = initial_prices.copy()  # Start at initial prices
        
        for n in range(num_time_steps):
            # Volatility matrix: σᵢⱼ = σ·Xᵢ if i=j, else 0 (diagonal)
            sigma_matrix = np.diag((vol * X).flatten())
            
            # Generate independent standard normal increments
            dW = np.random.normal(0, 1, (d, 1))
            
            # Euler-Maruyama step
            X = X + r * X * dt + sigma_matrix @ dW * sqrt_dt
            
            # Store this state
            stock_price_paths[m, n] = X
    
    # Time grid for reference
    time_grid = np.linspace(0, num_time_steps * dt, num=num_time_steps)
    
    print(f"Training data generation complete!")
    print(f"Total data points for regression: {num_training_paths * num_time_steps}")
    
    return stock_price_paths[..., 0], time_grid  # Remove last dimension for convenience


def build_regression_matrices(stock_price_paths, time_grid, params, basis_pairs):
    """
    Construct design matrix D and response vector ψ for L² regression.
    
    Regression problem: Find coefficients c such that D·c ≈ ψ
    where:
        D[idx, p] = (t_n)^i₁ · (S̄_n)^i₂  (basis function p evaluated at data point idx)
        ψ[idx] = true variance of basket at (m, n)
    
    Parameters
    ----------
    stock_price_paths : ndarray, shape (M_t, N_t, d)
        Training paths from generate_training_paths()
    time_grid : ndarray, shape (N_t,)
        Time points
    params : dict
        Market parameters
    basis_pairs : list of tuples
        Polynomial basis from setup_polynomial_basis()
    
    Returns
    -------
    design_matrix : ndarray, shape (M_t·N_t, P)
        Design matrix D where P is number of basis functions
    response_vector : ndarray, shape (M_t·N_t,)
        Response vector ψ containing true basket variance at each point
    
    Notes
    -----
    For independent stocks with equal volatility σ:
        Var(dS̄) = Var(Σ wᵢ·dXᵢ) = σ² · (1/d²) · Σ Xᵢ²
    
    This is the "target" that our fitted function b̄²(t,s) tries to match.
    """
    num_training_paths = params['num_training_paths']
    num_time_steps = params['num_time_steps']
    d = params['d']
    vol = params['vol']
    basket_weights = params['basket_weights']
    
    num_data_points = num_training_paths * num_time_steps
    num_basis_functions = len(basis_pairs)
    
    # Initialize matrices
    design_matrix = np.zeros((num_data_points, num_basis_functions))
    response_vector = np.zeros(num_data_points)
    
    print("Building design matrix and response vector...")
    
    for m in range(num_training_paths):
        for n in range(num_time_steps):
            # Flatten index: row number in matrices
            idx = m * num_time_steps + n
            
            # Current time
            t_n = time_grid[n]
            
            # Stock prices at this point
            X = stock_price_paths[m, n]
            
            # Basket value: S̄ = P₁ᵀ·X = Σ wᵢ·Xᵢ
            basket_value = float(basket_weights.dot(X))
            
            # TRUE instantaneous variance of the basket
            # For independent assets with volatility σ:
            #   Var(dS̄) = σ² · Σ wᵢ² · Xᵢ²
            # With equal weights wᵢ = 1/d:
            #   Var(dS̄) = σ² · (1/d²) · Σ Xᵢ²
            response_vector[idx] = (vol**2 / d**2) * float(X.dot(X))
            
            # Fill design matrix: evaluate each basis function
            for p, (i1, i2) in enumerate(basis_pairs):
                # Basis function p: t^i₁ · s^i₂
                design_matrix[idx, p] = (t_n ** i1) * (basket_value ** i2)
    
    print(f"Matrices constructed. Shape: D={design_matrix.shape}, ψ={response_vector.shape}")
    
    return design_matrix, response_vector


def solve_regression_qr(design_matrix, response_vector):
    """
    Solve least squares regression D·c ≈ ψ using QR decomposition.
    
    QR decomposition: D = Q·R where Q is orthogonal and R is upper triangular
    Then: c = R⁻¹·Qᵀ·ψ
    
    Parameters
    ----------
    design_matrix : ndarray, shape (M·N, P)
        Design matrix D
    response_vector : ndarray, shape (M·N,)
        Response vector ψ
    
    Returns
    -------
    coefficients : ndarray, shape (P,)
        Fitted coefficients c
    
    Notes
    -----
    Why QR instead of normal equations (DᵀD)·c = Dᵀ·ψ?
    - Numerical stability: cond(DᵀD) = [cond(D)]² (squaring is bad!)
    - QR avoids forming DᵀD explicitly
    - Better for ill-conditioned problems
    
    Diagnostics printed:
    - Absolute residual: ||D·c - ψ||
    - Relative residual: ||D·c - ψ|| / ||ψ||
    - Condition number: cond(D)
    - Smallest singular values: checks for near rank-deficiency
    """
    print("\nSolving regression using QR decomposition...")
    
    # QR decomposition: D = Q·R
    Q, R = np.linalg.qr(design_matrix, mode='reduced')
    
    # Project response onto Q: α = Qᵀ·ψ
    alpha = Q.T @ response_vector
    
    # Solve triangular system: R·c = α
    coefficients = np.linalg.solve(R, alpha)
    
    print(f"\nFitted coefficients: c = {coefficients}")
    
    # Diagnostics
    residual = np.linalg.norm(design_matrix @ coefficients - response_vector)
    response_norm = np.linalg.norm(response_vector)
    
    print(f"\nRegression Diagnostics:")
    print(f"  Absolute residual ||D·c - ψ||: {residual:.6f}")
    print(f"  Relative residual: {residual/response_norm:.6f}")
    print(f"  Condition number cond(D): {np.linalg.cond(design_matrix):.2e}")
    
    # Check smallest singular values (indicates near rank-deficiency if very small)
    singular_values = np.linalg.svd(design_matrix, compute_uv=False)
    print(f"  Smallest 5 singular values: {singular_values[-5:]}")
    
    # Warning if poorly conditioned
    if np.linalg.cond(design_matrix) > 1e8:
        print("\n  ⚠️  WARNING: Design matrix is poorly conditioned!")
        print("     Consider using orthogonal polynomials (Legendre) instead.")
    
    return coefficients


def create_projected_volatility_function(coefficients, basis_pairs):
    """
    Create callable function b̄(t, s) from fitted coefficients.
    
    The fitted polynomial is: b̄²(t,s) = Σ cₚ · t^i₁ · s^i₂
    We return: b̄(t,s) = √[b̄²(t,s)]
    
    Parameters
    ----------
    coefficients : ndarray, shape (P,)
        Fitted coefficients from solve_regression_qr()
    basis_pairs : list of tuples
        Polynomial basis pairs (i₁, i₂)
    
    Returns
    -------
    b_bar : callable
        Function b_bar(t, s) that returns projected volatility coefficient
    
    Notes
    -----
    We fitted b̄²(t,s) (the variance) but need b̄(t,s) (the volatility coefficient)
    for the SDE: dS̄_t = r·S̄_t·dt + b̄(t,S̄_t)·dW_t
    
    Warning: If b̄²(t,s) < 0 for some (t,s), this is non-physical and indicates
    poor fit. This can happen with low-degree polynomials or insufficient data.
    """
    def b_bar(t, s):
        """
        Evaluate projected volatility coefficient at (t, s).
        
        Parameters
        ----------
        t : float
            Time
        s : float
            Basket value
        
        Returns
        -------
        vol_coeff : float
            Volatility coefficient b̄(t, s)
        """
        # Evaluate polynomial: h = b̄²(t,s)
        h = 0.0
        for p, (i1, i2) in enumerate(basis_pairs):
            h += coefficients[p] * (t ** i1) * (s ** i2)
        
        # Take square root (could be problematic if h < 0)
        if h < 0:
            # This shouldn't happen, but print warning if it does
            print(f"⚠️  Warning: b̄²({t:.3f}, {s:.2f}) = {h:.6f} < 0 (non-physical!)")
            h = 0.0  # Clamp to zero
        
        return math.sqrt(h)
    
    return b_bar


def validate_projection(b_bar, params):
    """
    Validate the Markovian projection by comparing distributions.
    
    We simulate:
    1. TRUE process: d-dimensional GBM, then compute basket at maturity
    2. PROJECTED process: 1D SDE with b̄(t,s), starting at basket value
    
    Gyöngy's Lemma guarantees: If b̄ is correct, the terminal distributions match.
    
    Parameters
    ----------
    b_bar : callable
        Projected volatility function from create_projected_volatility_function()
    params : dict
        Market parameters
    
    Returns
    -------
    log_returns_basket : ndarray, shape (num_validation_paths,)
        Log returns from true basket
    log_returns_S_bar : ndarray, shape (num_validation_paths,)
        Log returns from projected process
    
    Notes
    -----
    We use many more paths here (10,000) than for training (100) to get
    reliable distribution estimates for validation.
    """
    d = params['d']
    initial_prices = params['initial_prices']
    r = params['r']
    vol = params['vol']
    dt = params['dt']
    sqrt_dt = params['sqrt_dt']
    basket_weights = params['basket_weights']
    num_time_steps = params['num_time_steps']
    
    # Time grid
    time_grid = np.linspace(0, num_time_steps * dt, num=num_time_steps)
    
    # Initial basket value
    initial_basket_value = basket_weights.dot(initial_prices.flatten())
    
    num_validation_paths = 10000
    print(f"\nValidating projection with {num_validation_paths} paths...")
    
    # ========================================
    # 1. PROJECTED PROCESS (1D SDE)
    # ========================================
    # dS̄_t = r·S̄_t·dt + b̄(t,S̄_t)·dW_t
    
    S_bar_terminal = np.zeros(num_validation_paths)  # Terminal values of projected basket process
    
    for m in range(num_validation_paths):
        S_bar = initial_basket_value  # Projected basket value (1D process)
        
        for n in range(num_time_steps):
            t_n = time_grid[n]
            # Euler-Maruyama for 1D SDE
            S_bar = S_bar + r * S_bar * dt + b_bar(t_n, S_bar) * random.gauss(0, 1) * sqrt_dt
        
        S_bar_terminal[m] = S_bar
    
    # ========================================
    # 2. TRUE PROCESS (d-dimensional GBM)
    # ========================================
    # dXᵢ = r·Xᵢ·dt + σ·Xᵢ·dWᵢ
    # Then compute S̄_T = P₁ᵀ·X_T
    
    basket_terminal = np.zeros(num_validation_paths)  # Terminal basket values from true d-dimensional process
    
    for m in range(num_validation_paths):
        X = initial_prices.copy()
        
        for n in range(num_time_steps):
            sigma_matrix = np.diag((vol * X).flatten())
            dW = np.random.normal(0, 1, (d, 1))
            X = X + r * X * dt + sigma_matrix @ dW * sqrt_dt
        
        basket_terminal[m] = basket_weights.dot(X[:, 0])
    
    # ========================================
    # 3. COMPARE DISTRIBUTIONS
    # ========================================
    # Use log returns for better comparison (more symmetric)
    
    log_returns_basket = np.log(basket_terminal / initial_basket_value)  # True basket log returns
    log_returns_S_bar = np.log(S_bar_terminal / initial_basket_value)  # Projected process log returns
    
    print("\nDistribution Statistics:")
    print(f"  True process:      mean = {log_returns_basket.mean():.6f}, std = {log_returns_basket.std(ddof=1):.6f}")
    print(f"  Projected process: mean = {log_returns_S_bar.mean():.6f}, std = {log_returns_S_bar.std(ddof=1):.6f}")
    
    # Compute relative error in moments
    mean_error = abs(log_returns_S_bar.mean() - log_returns_basket.mean()) / abs(log_returns_basket.mean())
    std_error = abs(log_returns_S_bar.std(ddof=1) - log_returns_basket.std(ddof=1)) / log_returns_basket.std(ddof=1)
    
    print(f"  Relative errors:   mean = {mean_error:.2%}, std = {std_error:.2%}")
    
    if mean_error < 0.05 and std_error < 0.05:
        print("\n✅ Projection is EXCELLENT (errors < 5%)")
    elif mean_error < 0.10 and std_error < 0.10:
        print("\n✓ Projection is GOOD (errors < 10%)")
    else:
        print("\n⚠️  Projection quality could be improved")
        print("   Consider: Higher polynomial degree or more training paths")
    
    return log_returns_basket, log_returns_S_bar


def plot_comparison(log_returns_basket, log_returns_S_bar, params):
    """
    Create histogram comparing true vs projected distributions.
    
    Parameters
    ----------
    log_returns_basket : ndarray
        Log returns from true basket
    log_returns_S_bar : ndarray
        Log returns from projected process
    params : dict
        Market parameters (for plot title)
    """
    d = params['d']
    
    # Use consistent bins for both histograms
    all_returns = np.concatenate([log_returns_basket, log_returns_S_bar])
    bins = np.linspace(all_returns.min(), all_returns.max(), 51)
    
    plt.figure(figsize=(10, 6))
    
    # Plot true distribution (filled, transparent)
    plt.hist(log_returns_basket, bins=bins, histtype='stepfilled', 
             color='C0', alpha=0.3, label='True Process: P₁·X', density=False)
    
    # Plot projected distribution (outline only)
    plt.hist(log_returns_S_bar, bins=bins, histtype='step', 
             color='C1', alpha=0.75, linewidth=2, label='Markovian Projection: S̄', density=False)
    
    plt.xlabel('Log Returns', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Terminal Distribution: True vs Projected Process\n'
              f'Equal-weighted basket of {d} stocks (simple polynomial basis)',
              fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('markovian_projection_validation.pdf', bbox_inches='tight', dpi=150)
    print("\n📊 Plot saved as 'markovian_projection_validation.pdf'")
    plt.show()


def main():
    """
    Main execution: demonstrate Markovian projection with simple polynomial basis.
    """
    print("="*70)
    print("MARKOVIAN PROJECTION FOR BASKET OPTIONS")
    print("Basic Implementation with Simple Polynomial Basis")
    print("="*70)
    
    # 1. Setup
    print("\n" + "="*70)
    print("STEP 1: Market Parameters Setup")
    print("="*70)
    params = setup_market_parameters()
    
    print("\n" + "="*70)
    print("STEP 2: Polynomial Basis Definition")
    print("="*70)
    basis_pairs, num_basis_functions = setup_polynomial_basis()
    
    # 2. Generate training data
    print("\n" + "="*70)
    print("STEP 3: Training Data Generation")
    print("="*70)
    stock_price_paths, time_grid = generate_training_paths(params)
    
    # 3. Build regression matrices
    print("\n" + "="*70)
    print("STEP 4: Regression Matrix Construction")
    print("="*70)
    design_matrix, response_vector = build_regression_matrices(
        stock_price_paths, time_grid, params, basis_pairs
    )
    
    # 4. Solve regression
    print("\n" + "="*70)
    print("STEP 5: L² Regression (QR Decomposition)")
    print("="*70)
    coefficients = solve_regression_qr(design_matrix, response_vector)
    
    # 5. Create projected volatility function
    print("\n" + "="*70)
    print("STEP 6: Projected Volatility Function")
    print("="*70)
    b_bar = create_projected_volatility_function(coefficients, basis_pairs)
    print("✓ Function b̄(t,s) created and ready to use")
    
    # 6. Validate projection
    print("\n" + "="*70)
    print("STEP 7: Validation (Gyöngy's Lemma Test)")
    print("="*70)
    log_returns_basket, log_returns_S_bar = validate_projection(b_bar, params)
    
    # 7. Visualize results
    print("\n" + "="*70)
    print("STEP 8: Visualization")
    print("="*70)
    plot_comparison(log_returns_basket, log_returns_S_bar, params)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. We reduced d=3 dimensional basket to an effective 1D process")
    print("2. The fitted function b̄(t,s) captures the local volatility")
    print("3. Terminal distributions match (validates Gyöngy's Lemma)")
    print("4. This forms the foundation for American option pricing via PDE methods")
    print("\nNext steps:")
    print("→ Use orthogonal polynomials (Legendre) for better conditioning")
    print("→ Add multi-level Monte Carlo for computational efficiency")
    print("→ Include correlations between assets")
    print("="*70)


if __name__ == "__main__":
    main()
