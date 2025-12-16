"""
Wrapper for MLMC Volatility Estimation Method

This module provides a unified interface to the MLMC (Multi-Level Monte Carlo)
volatility estimation from the PDE/ module.

The MLMC method:
1. Runs pilot simulations to estimate the basket domain [S_min, S_max]
2. Uses hierarchical Monte Carlo with polynomial regression
3. Returns coefficients for a volatility surface b(t, S)

Mathematical Background
-----------------------
MLMC exploits the telescoping sum identity:
    E[Y_L] = E[Y_0] + sum_{l=1}^{L} E[Y_l - Y_{l-1}]

where Y_l is the volatility estimate at level l with timestep h_l = h_0 * 2^(-l).

The variance of (Y_l - Y_{l-1}) decreases with l (due to path coupling),
while the cost per sample increases. Optimal allocation gives O(ε^{-2})
complexity vs O(ε^{-3}) for standard MC.

Author: Wadoud (KAUST Internship)
Reference: Bayer, Häppölä, Tempone (2017) Section 3
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import time
import numpy as np
from typing import Tuple, Optional

from .common import VolatilitySurfaceResult

# Try to import the actual MLMC code
_MLMC_AVAILABLE = False
try:
    # Import basket_simulation first and add to sys.modules
    # This is needed because mlmc_volatility_estimation uses a relative import
    import PDE.basket_simulation
    sys.modules['basket_simulation'] = PDE.basket_simulation

    # Now we can safely import mlmc_volatility_estimation
    from PDE.mlmc_volatility_estimation import (
        aggregate_mlmc_coefficients,
        estimate_basket_domain,
    )
    from PDE.basket_simulation import (
        construct_volatility_surface,
        generate_polynomial_basis_pairs,
    )
    _MLMC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MLMC modules: {e}")
    print("  Using demonstration fallback implementation.")


def _fallback_volatility_surface(params, t_grid, s_grid, random_seed):
    """
    Fallback demonstration: generate synthetic b² surface.
    
    This mimics the expected behaviour for testing the comparison framework
    when the actual PDE code is not available.
    
    The synthetic surface approximates:
        b²(t, s) ≈ (σ_basket * s)²
    with some time and spatial variation.
    """
    np.random.seed(random_seed)
    
    S0 = params.S0
    sigma_eff = np.sqrt(
        params.P1 @ np.diag(params.sigma**2) @ params.corr_matrix @ np.diag(params.sigma**2).T @ params.P1
    )
    
    # Base volatility squared
    T_mesh, S_mesh = np.meshgrid(t_grid, s_grid, indexing='ij')
    
    # b² = (weighted volatility * basket value)² with some structure
    # Add time dependence (volatility smile effect)
    time_factor = 1 + 0.1 * np.sqrt(T_mesh)  # Slight increase with time
    
    # Add spatial dependence (skew)
    moneyness = S_mesh / S0
    skew_factor = 1 + 0.05 * (1 - moneyness)  # Higher vol for lower basket values
    
    # Base calculation
    b_squared = (sigma_eff ** 2) * (S_mesh ** 2) * time_factor * skew_factor
    
    # Add small random noise to simulate estimation uncertainty
    noise = 1 + 0.02 * np.random.randn(*b_squared.shape)
    b_squared = b_squared * np.maximum(noise, 0.9)
    
    return b_squared


def estimate_volatility_mlmc(
    params,  # ProblemParameters
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    random_seed: int = 42,
    verbose: bool = True
) -> VolatilitySurfaceResult:
    """
    Estimate projected volatility surface using MLMC + polynomial regression.
    
    Parameters
    ----------
    params : ProblemParameters
        Problem configuration containing:
        - x0: Initial asset prices
        - T: Maturity
        - r: Risk-free rate
        - sigma: Asset volatilities
        - corr_matrix: Correlation matrix
        - P1: Basket weights
        - max_degree: MLMC polynomial degree
        - h0: Coarsest timestep
    t_grid : np.ndarray
        Time points for evaluation.
    s_grid : np.ndarray
        Basket values for evaluation.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    VolatilitySurfaceResult
        Standardised result object containing:
        - b_surface: callable b(t, s)
        - b_squared_surface: callable b²(t, s)
        - Grid values for plotting
        - Timing and metadata
        
    Notes
    -----
    The MLMC method returns polynomial coefficients c such that:
        b(t, S) = sqrt(sum_p c_p * P_{i1}(t/T) * P_{i2}((S-S_min)/(S_max-S_min)))
    where P are orthonormalised Legendre polynomials.
    """
    np.random.seed(random_seed)
    start_time = time.perf_counter()
    
    if verbose:
        print(f"MLMC Volatility Estimation")
        print(f"  Assets: {params.d}, max_degree: {params.max_degree}")
    
    # Use actual MLMC if available, otherwise fallback
    if _MLMC_AVAILABLE:
        # Step 1: Estimate domain bounds via pilot run
        if verbose:
            print("  Step 1: Pilot run for domain estimation...")
        
        S_min, S_max, _ = estimate_basket_domain(
            S0=params.x0,
            T=params.T,
            h0=params.h0,
            r=params.r,
            cov_mat=params.corr_matrix,
            vol=params.sigma,
            max_degree=params.max_degree,
            basket_weights=params.P1,
            N_pilot=10000
        )
        
        if verbose:
            print(f"    Domain: [{S_min:.1f}, {S_max:.1f}]")
        
        # Step 2: Run MLMC coefficient estimation
        if verbose:
            print("  Step 2: MLMC coefficient estimation...")
        
        coefficients = aggregate_mlmc_coefficients(
            S0=params.x0,
            T=params.T,
            h0=params.h0,
            r=params.r,
            cov_mat=params.corr_matrix,
            vol=params.sigma,
            max_degree=params.max_degree,
            basket_weights=params.P1,
            S_min=S_min,
            S_max=S_max
        )
        
        if verbose:
            print(f"    Coefficients computed: {len(coefficients)}")
        
        # Step 3: Construct callable volatility surface
        basis_pairs = generate_polynomial_basis_pairs(params.max_degree)
        b_surface_func = construct_volatility_surface(
            coefficients, basis_pairs, S_min, S_max, params.T, params.max_degree
        )
        
        # Step 4: Evaluate on grid
        if verbose:
            print("  Step 3: Evaluating on grid...")
        
        b_squared_values = np.zeros((len(t_grid), len(s_grid)))
        for i, t in enumerate(t_grid):
            for j, s in enumerate(s_grid):
                b_val = b_surface_func(t, s)
                b_squared_values[i, j] = b_val ** 2
        
        # Create b² callable
        def b_squared_surface(t: float, s: float) -> float:
            return b_surface_func(t, s) ** 2
        
        domain_bounds = (S_min, S_max)
        
    else:
        # Fallback demonstration
        if verbose:
            print("  Using fallback demonstration...")
        
        b_squared_values = _fallback_volatility_surface(
            params, t_grid, s_grid, random_seed
        )
        
        # Estimate domain from s_grid
        S_min, S_max = s_grid.min(), s_grid.max()
        domain_bounds = (S_min, S_max)
        coefficients = None
        
        # Create interpolated surface functions
        from scipy.interpolate import RectBivariateSpline
        interp = RectBivariateSpline(t_grid, s_grid, b_squared_values)
        
        def b_squared_surface(t: float, s: float) -> float:
            return float(interp(t, s))
        
        def b_surface_func(t: float, s: float) -> float:
            return np.sqrt(max(b_squared_surface(t, s), 0))
    
    computation_time = time.perf_counter() - start_time
    
    if verbose:
        print(f"  Completed in {computation_time:.2f}s")
        print(f"  b² range: [{np.nanmin(b_squared_values):.1f}, {np.nanmax(b_squared_values):.1f}]")
    
    return VolatilitySurfaceResult(
        b_surface=b_surface_func,
        b_squared_surface=b_squared_surface,
        t_grid=t_grid,
        s_grid=s_grid,
        b_squared_values=b_squared_values,
        method_name="MLMC",
        computation_time=computation_time,
        parameters={
            "max_degree": params.max_degree,
            "h0": params.h0,
            "random_seed": random_seed,
            "mlmc_available": _MLMC_AVAILABLE,
        },
        coefficients=coefficients,
        n_samples=None,  # Could track in aggregate_mlmc_coefficients
        mlmc_levels=params.max_degree + 1,
        domain_bounds=domain_bounds
    )


def estimate_volatility_mlmc_multiple_runs(
    params,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    n_runs: int = 20,
    base_seed: int = 42,
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
        print(f"Running {n_runs} MLMC iterations for convergence study...")
    
    all_b_squared = []
    all_times = []
    
    for i in range(n_runs):
        seed = base_seed + i
        if verbose and (i + 1) % 5 == 0:
            print(f"  Run {i + 1}/{n_runs}...")
        
        result = estimate_volatility_mlmc(
            params, t_grid, s_grid, 
            random_seed=seed, 
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
    
    mean_result = VolatilitySurfaceResult(
        b_surface=mean_b_surface,
        b_squared_surface=mean_b_squared_surface,
        t_grid=t_grid,
        s_grid=s_grid,
        b_squared_values=mean_b_squared,
        method_name="MLMC (mean)",
        computation_time=np.sum(all_times),
        parameters={
            "n_runs": n_runs,
            "base_seed": base_seed,
            "max_degree": params.max_degree,
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


if __name__ == "__main__":
    # Test the wrapper
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DEFAULT_PARAMS
    
    print("Testing MLMC wrapper...")
    
    params = DEFAULT_PARAMS.copy()
    params.max_degree = 2  # Faster for testing
    
    t_grid = np.linspace(0.02, params.T, 10)
    s_grid = np.linspace(250, 350, 20)
    
    result = estimate_volatility_mlmc(params, t_grid, s_grid, verbose=True)
    
    print("\n" + result.summary())
