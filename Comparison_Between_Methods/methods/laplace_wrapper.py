"""
Wrapper for Laplace Approximation Volatility Estimation

This module provides a unified interface to the Laplace approximation method
from the Laplace_Replication/ module.

The Laplace method (Paper Equation 41):
1. For each (t, s) point, finds the mode of the posterior distribution
2. Uses second-order Taylor expansion around the mode
3. Computes b²(t, s) as a ratio of Gaussian integrals

Mathematical Background
-----------------------
The Laplace approximation computes:

    b²(t, s) = ∫ exp(f(z)) dz / ∫ exp(f̃(z)) dz

where:
- f(z) is the log of (density × basket_variance) restricted to basket=s
- f̃(z) is the log of density restricted to basket=s
- z ∈ R^{d-1} parametrises the constraint hyperplane

The integrals are approximated as:
    ∫ exp(f(z)) dz ≈ exp(f(z*)) × √((2π)^{d-1} / |det(-H_f)|)

where z* is the mode and H_f is the Hessian at the mode.

Author: Wadoud (KAUST Internship)
Reference: Bayer, Häppölä, Tempone (2017) Section 3.1.1, Equation 41
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import time
import warnings
import numpy as np
from typing import Optional

from .common import VolatilitySurfaceResult

# Try to import the actual Laplace code
_LAPLACE_AVAILABLE = False
try:
    from Laplace_Replication.laplace_volatility import (
        compute_volatility_surface as laplace_compute_surface,
        laplace_approximation_volatility_squared,
    )
    _LAPLACE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Laplace modules: {e}")
    print("  Using demonstration fallback implementation.")


def _fallback_laplace_surface(params, t_grid, s_grid):
    """
    Fallback demonstration: generate synthetic b² surface.
    
    This mimics the expected Laplace output for testing the comparison
    framework when the actual code is not available.
    
    The synthetic surface approximates the expected b² behaviour:
    - Quadratic dependence on basket value (GBM characteristic)
    - Time-dependent skew
    - Correlation effects
    """
    S0 = params.S0
    
    # Effective basket volatility (simplified calculation)
    # Full calculation involves correlation structure
    sigma_vec = params.sigma * params.P1
    corr = params.corr_matrix
    sigma_eff_sq = sigma_vec @ corr @ sigma_vec
    
    T_mesh, S_mesh = np.meshgrid(t_grid, s_grid, indexing='ij')
    
    # Base: b² ≈ σ²_eff * S²
    b_squared = sigma_eff_sq * (S_mesh ** 2)
    
    # Add time evolution (volatility structure)
    # Near t=0, volatility is exact; increases with time
    time_factor = 1 + 0.15 * np.sqrt(T_mesh / params.T)
    
    # Add skew (higher vol for lower basket values - typical for equity baskets)
    moneyness = S_mesh / S0
    skew = 1 + 0.08 * (1 - moneyness)
    
    b_squared = b_squared * time_factor * skew
    
    return b_squared


def estimate_volatility_laplace(
    params,  # ProblemParameters
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    verbose: bool = True
) -> VolatilitySurfaceResult:
    """
    Estimate projected volatility surface using Laplace approximation.
    
    This is a deterministic method (no Monte Carlo), so no random seed is needed.
    
    Parameters
    ----------
    params : ProblemParameters
        Problem configuration containing:
        - x0: Initial asset prices
        - T: Maturity
        - r: Risk-free rate
        - sigma: Asset volatilities
        - corr_matrix: Correlation matrix (need Cholesky factor)
        - P1: Basket weights
    t_grid : np.ndarray
        Time points for evaluation. Should avoid t=0 (use t≥0.01).
    s_grid : np.ndarray
        Basket values for evaluation.
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
    The Laplace method directly computes b²(t, s). The paper's Figure 1a
    shows b² values in the range ~700-2500 for the 3D test case.
    
    Warning: Near t=0, the Laplace approximation can be unstable.
    Recommend using t_grid starting at t ≥ 0.01.
    """
    start_time = time.perf_counter()
    
    if verbose:
        print(f"Laplace Approximation Volatility Estimation")
        print(f"  Assets: {params.d}")
        print(f"  Grid: {len(t_grid)} × {len(s_grid)}")
    
    # Check for t near zero
    if np.any(t_grid < 0.005):
        warnings.warn(
            "Laplace approximation unstable near t=0. "
            "Consider starting t_grid at t ≥ 0.01."
        )
    
    # Use actual Laplace if available, otherwise fallback
    if _LAPLACE_AVAILABLE:
        if verbose:
            print("  Computing b² surface via Laplace approximation...")
        
        b_squared_values = laplace_compute_surface(
            t_grid=t_grid,
            s_grid=s_grid,
            P1=params.P1,
            x0=params.x0,
            r=params.r,
            sigma=params.sigma,
            corr_chol=params.corr_chol
        )
        
    else:
        # Fallback demonstration
        if verbose:
            print("  Using fallback demonstration...")
        
        b_squared_values = _fallback_laplace_surface(params, t_grid, s_grid)
    
    # Handle NaN values for interpolation
    b_squared_clean = b_squared_values.copy()
    nan_count = np.sum(np.isnan(b_squared_clean))
    
    if nan_count > 0 and verbose:
        print(f"  Warning: {nan_count} NaN values in surface ({100*nan_count/b_squared_clean.size:.1f}%)")
    
    # Replace NaN with interpolated/extrapolated values
    for i in range(len(t_grid)):
        slice_data = b_squared_clean[i, :]
        valid = np.isfinite(slice_data)
        if np.sum(valid) > 1 and np.sum(~valid) > 0:
            from scipy.interpolate import interp1d
            interp = interp1d(
                s_grid[valid], slice_data[valid],
                kind='linear', fill_value='extrapolate'
            )
            b_squared_clean[i, ~valid] = interp(s_grid[~valid])
    
    # Create interpolated callable for b²
    from scipy.interpolate import RectBivariateSpline
    
    # Ensure clean surface for interpolation
    b_sq_for_interp = np.nan_to_num(b_squared_clean, nan=np.nanmean(b_squared_clean))
    b_sq_for_interp = np.maximum(b_sq_for_interp, 1.0)  # Floor at 1.0
    
    interp = RectBivariateSpline(t_grid, s_grid, b_sq_for_interp)
    
    def b_squared_surface(t: float, s: float) -> float:
        """Evaluate b²(t, s) via interpolation."""
        # Clamp to grid bounds
        t_clamped = np.clip(t, t_grid.min(), t_grid.max())
        s_clamped = np.clip(s, s_grid.min(), s_grid.max())
        return max(float(interp(t_clamped, s_clamped)), 0.0)
    
    def b_surface(t: float, s: float) -> float:
        """Evaluate b(t, s) as sqrt(b²)."""
        return np.sqrt(b_squared_surface(t, s))
    
    computation_time = time.perf_counter() - start_time
    
    if verbose:
        print(f"  Completed in {computation_time:.2f}s")
        print(f"  b² range: [{np.nanmin(b_squared_values):.1f}, {np.nanmax(b_squared_values):.1f}]")
        if _LAPLACE_AVAILABLE:
            print(f"  (Paper Figure 1a shows ~700-2500 for 3D case)")
    
    return VolatilitySurfaceResult(
        b_surface=b_surface,
        b_squared_surface=b_squared_surface,
        t_grid=t_grid,
        s_grid=s_grid,
        b_squared_values=b_squared_values,
        method_name="Laplace",
        computation_time=computation_time,
        parameters={
            "method": "Laplace approximation (Paper Eq. 41)",
            "laplace_available": _LAPLACE_AVAILABLE,
        },
        coefficients=None,
        n_samples=None,
        mlmc_levels=None,
        domain_bounds=(s_grid.min(), s_grid.max())
    )


def estimate_volatility_laplace_pointwise(
    s: float,
    t: float,
    params,
    verbose: bool = False
) -> float:
    """
    Compute b²(t, s) at a single point using Laplace approximation.
    
    Parameters
    ----------
    s : float
        Basket value.
    t : float
        Time.
    params : ProblemParameters
        Problem configuration.
    verbose : bool
        Print debug information.
        
    Returns
    -------
    float
        Projected volatility squared, b²(t, s).
    """
    if not _LAPLACE_AVAILABLE:
        # Fallback: simplified calculation
        sigma_vec = params.sigma * params.P1
        sigma_eff_sq = sigma_vec @ params.corr_matrix @ sigma_vec
        return sigma_eff_sq * s ** 2
    
    return laplace_approximation_volatility_squared(
        s=s,
        t=t,
        P1=params.P1,
        x0=params.x0,
        r=params.r,
        sigma=params.sigma,
        corr_chol=params.corr_chol
    )


if __name__ == "__main__":
    # Test the wrapper
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DEFAULT_PARAMS
    
    print("Testing Laplace wrapper...")
    
    params = DEFAULT_PARAMS
    
    # Avoid t=0 for Laplace
    t_grid = np.linspace(0.02, params.T, 15)
    s_grid = np.linspace(250, 350, 25)
    
    result = estimate_volatility_laplace(params, t_grid, s_grid, verbose=True)
    
    print("\n" + result.summary())
    
    # Test single point
    print("\nSingle point evaluation:")
    b_sq = estimate_volatility_laplace_pointwise(300.0, 0.25, params)
    print(f"  b²(t=0.25, s=300) = {b_sq:.1f}")
