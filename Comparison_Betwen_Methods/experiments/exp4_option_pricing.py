"""
Experiment 4: Option Pricing Comparison

Use volatility surfaces from both methods to price American basket options
via PDE solving, then compare the resulting option prices.

This experiment:
1. Computes volatility surfaces using MLMC and Laplace
2. Solves the projected 1D American option PDE with each surface
3. Compares option prices at t=0 for various spot values
4. Studies the impact of volatility surface differences on prices

Key Question: Do differences in volatility surfaces propagate
significantly into option price differences?

Expected Outputs
----------------
- results/figures/exp4_price_comparison.png
- results/figures/exp4_price_difference.png
- results/tables/exp4_option_prices.md

Author: Wadoud (KAUST Internship)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_script_dir = Path(__file__).resolve().parent
_comparison_dir = _script_dir.parent
sys.path.insert(0, str(_comparison_dir))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DEFAULT_PARAMS
from methods import (
    estimate_volatility_mlmc,
    estimate_volatility_laplace,
)


def solve_american_pde(b_squared_surface, params, S_grid, t_grid):
    """
    Solve 1D American option PDE using the projected volatility surface.
    
    Uses backward Euler with early exercise constraint.
    
    Parameters
    ----------
    b_squared_surface : callable
        Function b²(t, S) giving squared diffusion coefficient
    params : ProblemParameters
        Option parameters
    S_grid : np.ndarray
        Spatial grid
    t_grid : np.ndarray
        Time grid (increasing from 0 to T)
    
    Returns
    -------
    V : np.ndarray
        Option values, shape (len(t_grid), len(S_grid))
    """
    N_t = len(t_grid)
    N_s = len(S_grid)
    
    dt = t_grid[1] - t_grid[0] if N_t > 1 else params.T / 50
    dS = S_grid[1] - S_grid[0]
    
    # Terminal payoff
    if params.option_type == "put":
        payoff = np.maximum(params.K - S_grid, 0.0)
    else:
        payoff = np.maximum(S_grid - params.K, 0.0)
    
    # Value array (time runs backwards from T to 0)
    V = np.zeros((N_t, N_s))
    V[-1, :] = payoff
    
    # March backwards in time
    for n in range(N_t - 2, -1, -1):
        t = t_grid[n]
        
        # Get b² values at this time
        b2_vals = np.array([
            b_squared_surface(t, S) if callable(b_squared_surface) 
            else np.interp(S, S_grid, b_squared_surface[n, :])
            for S in S_grid
        ])
        b2_vals = np.clip(b2_vals, 1e-6, None)  # Ensure positive
        
        # Build tridiagonal system (Crank-Nicolson / implicit)
        alpha = 0.5 * b2_vals * dt / (dS ** 2)
        beta = params.r * S_grid * dt / (2 * dS)
        
        # Coefficient matrices for implicit scheme
        # V[n] = A * V[n+1] (approximately)
        a = -alpha + beta  # Lower diagonal
        b = 1 + 2 * alpha + params.r * dt  # Main diagonal
        c = -alpha - beta  # Upper diagonal
        
        # Solve tridiagonal system
        V_next = V[n + 1, :].copy()
        
        # Thomas algorithm for tridiagonal solve
        n_pts = N_s
        c_prime = np.zeros(n_pts)
        d_prime = np.zeros(n_pts)
        
        # Forward sweep
        c_prime[0] = c[0] / b[0] if abs(b[0]) > 1e-12 else 0
        d_prime[0] = V_next[0] / b[0] if abs(b[0]) > 1e-12 else 0
        
        for i in range(1, n_pts):
            denom = b[i] - a[i] * c_prime[i - 1]
            if abs(denom) < 1e-12:
                denom = 1e-12
            c_prime[i] = c[i] / denom
            d_prime[i] = (V_next[i] - a[i] * d_prime[i - 1]) / denom
        
        # Back substitution
        V[n, -1] = d_prime[-1]
        for i in range(n_pts - 2, -1, -1):
            V[n, i] = d_prime[i] - c_prime[i] * V[n, i + 1]
        
        # Apply early exercise constraint
        V[n, :] = np.maximum(V[n, :], payoff)
    
    return V


def run_experiment(
    params=None,
    n_spot_points: int = 50,
    save_results: bool = True,
    show_plots: bool = False,
    verbose: bool = True
):
    """
    Run Experiment 4: Option pricing comparison.
    
    Parameters
    ----------
    params : ProblemParameters, optional
        Problem parameters.
    n_spot_points : int
        Number of spot values for price curve.
    save_results : bool
        Whether to save figures and tables.
    show_plots : bool
        Whether to display plots interactively.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    dict
        Dictionary containing:
        - S_values: Spot values tested
        - prices_mlmc: Option prices using MLMC surface
        - prices_laplace: Option prices using Laplace surface
        - price_diff: Absolute differences
        - price_diff_pct: Percentage differences
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Setup paths
    results_dir = _comparison_dir / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    
    for d in [results_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("EXPERIMENT 4: Option Pricing Comparison")
        print("=" * 70)
        print()
        print(f"d = {params.d} assets, S0 = {params.S0}, K = {params.K}")
        print(f"Option type: {params.option_type}")
        print()
    
    # Grid for volatility surface estimation
    t_vol_grid = np.linspace(0.02, params.T, 20)
    s_vol_grid = np.linspace(params.S0 * 0.5, params.S0 * 1.5, 40)
    
    # PDE grid (finer for pricing)
    t_pde_grid = np.linspace(0, params.T, params.N_t)
    S_pde_grid = np.linspace(params.S0 * 0.5, params.S0 * 1.5, params.N_s)
    
    # Step 1: Compute volatility surfaces
    if verbose:
        print("-" * 50)
        print("Step 1: Computing volatility surfaces...")
        print("-" * 50)
    
    result_mlmc = estimate_volatility_mlmc(
        params, t_vol_grid, s_vol_grid,
        verbose=verbose
    )
    
    result_laplace = estimate_volatility_laplace(
        params, t_vol_grid, s_vol_grid,
        verbose=verbose
    )
    
    if verbose:
        print(f"\nMLMC time: {result_mlmc.computation_time:.2f}s")
        print(f"Laplace time: {result_laplace.computation_time:.2f}s")
    
    # Step 2: Solve PDEs
    if verbose:
        print()
        print("-" * 50)
        print("Step 2: Solving American option PDEs...")
        print("-" * 50)
    
    # PDE with MLMC surface
    if verbose:
        print("  Solving PDE with MLMC volatility surface...")
    V_mlmc = solve_american_pde(
        result_mlmc.b_squared_surface, params, S_pde_grid, t_pde_grid
    )
    
    # PDE with Laplace surface
    if verbose:
        print("  Solving PDE with Laplace volatility surface...")
    V_laplace = solve_american_pde(
        result_laplace.b_squared_surface, params, S_pde_grid, t_pde_grid
    )
    
    # Step 3: Extract prices at t=0
    if verbose:
        print()
        print("-" * 50)
        print("Step 3: Extracting option prices at t=0...")
        print("-" * 50)
    
    prices_mlmc = V_mlmc[0, :]
    prices_laplace = V_laplace[0, :]
    
    # Compute differences
    price_diff = np.abs(prices_mlmc - prices_laplace)
    price_diff_pct = np.zeros_like(price_diff)
    nonzero = prices_laplace > 0.01  # Avoid division by tiny numbers
    price_diff_pct[nonzero] = 100 * price_diff[nonzero] / prices_laplace[nonzero]
    
    if verbose:
        atm_idx = np.argmin(np.abs(S_pde_grid - params.K))
        print(f"\nAt-the-money (S ≈ K = {params.K}):")
        print(f"  MLMC price:    {prices_mlmc[atm_idx]:.4f}")
        print(f"  Laplace price: {prices_laplace[atm_idx]:.4f}")
        print(f"  Difference:    {price_diff[atm_idx]:.4f} ({price_diff_pct[atm_idx]:.2f}%)")
        
        s0_idx = np.argmin(np.abs(S_pde_grid - params.S0))
        print(f"\nAt initial basket (S0 = {params.S0}):")
        print(f"  MLMC price:    {prices_mlmc[s0_idx]:.4f}")
        print(f"  Laplace price: {prices_laplace[s0_idx]:.4f}")
        print(f"  Difference:    {price_diff[s0_idx]:.4f} ({price_diff_pct[s0_idx]:.2f}%)")
    
    # Generate visualisations
    if save_results or show_plots:
        if verbose:
            print()
            print("-" * 50)
            print("Generating visualisations...")
            print("-" * 50)
        
        # 1. Price comparison
        fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Option prices
        ax1a.plot(S_pde_grid, prices_mlmc, 'b-', linewidth=2, label='MLMC')
        ax1a.plot(S_pde_grid, prices_laplace, 'r--', linewidth=2, label='Laplace')
        
        # Add payoff for reference
        if params.option_type == "put":
            payoff = np.maximum(params.K - S_pde_grid, 0)
        else:
            payoff = np.maximum(S_pde_grid - params.K, 0)
        ax1a.plot(S_pde_grid, payoff, 'k:', alpha=0.5, linewidth=1, label='Intrinsic')
        
        ax1a.axvline(params.S0, color='gray', linestyle='--', alpha=0.5, label=f'$S_0$ = {params.S0}')
        ax1a.axvline(params.K, color='gray', linestyle=':', alpha=0.5, label=f'K = {params.K}')
        
        ax1a.set_xlabel('Basket Value (S)', fontsize=12)
        ax1a.set_ylabel('Option Price', fontsize=12)
        ax1a.set_title(f'American {params.option_type.capitalize()} Prices at t=0', fontsize=13)
        ax1a.legend(fontsize=10)
        ax1a.grid(True, alpha=0.3)
        
        # Price comparison scatter
        ax1b.scatter(prices_laplace, prices_mlmc, alpha=0.6, s=20)
        max_price = max(np.max(prices_mlmc), np.max(prices_laplace))
        ax1b.plot([0, max_price], [0, max_price], 'k--', alpha=0.5, label='Perfect agreement')
        ax1b.set_xlabel('Laplace Price', fontsize=12)
        ax1b.set_ylabel('MLMC Price', fontsize=12)
        ax1b.set_title('Price Comparison', fontsize=13)
        ax1b.legend(fontsize=10)
        ax1b.grid(True, alpha=0.3)
        ax1b.set_aspect('equal')
        
        fig1.tight_layout()
        
        if save_results:
            fig1.savefig(figures_dir / "exp4_price_comparison.png", dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig1)
        
        # 2. Price differences
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute difference
        ax2a.plot(S_pde_grid, price_diff, 'b-', linewidth=2)
        ax2a.axvline(params.S0, color='gray', linestyle='--', alpha=0.5)
        ax2a.axvline(params.K, color='gray', linestyle=':', alpha=0.5)
        ax2a.set_xlabel('Basket Value (S)', fontsize=12)
        ax2a.set_ylabel('Absolute Difference', fontsize=12)
        ax2a.set_title('Price Difference: |MLMC - Laplace|', fontsize=13)
        ax2a.grid(True, alpha=0.3)
        
        # Percentage difference
        ax2b.plot(S_pde_grid[nonzero], price_diff_pct[nonzero], 'r-', linewidth=2)
        ax2b.axvline(params.S0, color='gray', linestyle='--', alpha=0.5)
        ax2b.axvline(params.K, color='gray', linestyle=':', alpha=0.5)
        ax2b.set_xlabel('Basket Value (S)', fontsize=12)
        ax2b.set_ylabel('Percentage Difference (%)', fontsize=12)
        ax2b.set_title('Relative Price Difference', fontsize=13)
        ax2b.grid(True, alpha=0.3)
        ax2b.set_ylim(0, min(20, np.max(price_diff_pct[nonzero]) * 1.1))
        
        fig2.tight_layout()
        
        if save_results:
            fig2.savefig(figures_dir / "exp4_price_difference.png", dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig2)
        
        # 3. Time evolution of prices
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Select a few spot values
        spot_indices = [
            np.argmin(np.abs(S_pde_grid - params.S0 * 0.8)),
            np.argmin(np.abs(S_pde_grid - params.S0)),
            np.argmin(np.abs(S_pde_grid - params.K)),
            np.argmin(np.abs(S_pde_grid - params.S0 * 1.2)),
        ]
        spot_labels = [f'S={S_pde_grid[i]:.0f}' for i in spot_indices]
        colors = ['blue', 'green', 'orange', 'red']
        
        for idx, (si, label, col) in enumerate(zip(spot_indices, spot_labels, colors)):
            ax3a.plot(t_pde_grid, V_mlmc[:, si], '-', color=col, linewidth=2, label=f'MLMC {label}')
            ax3a.plot(t_pde_grid, V_laplace[:, si], '--', color=col, linewidth=1.5, alpha=0.7)
        
        ax3a.set_xlabel('Time (t)', fontsize=12)
        ax3a.set_ylabel('Option Value', fontsize=12)
        ax3a.set_title('Option Value vs Time (solid=MLMC, dashed=Laplace)', fontsize=13)
        ax3a.legend(fontsize=9)
        ax3a.grid(True, alpha=0.3)
        
        # Value differences over time
        for idx, (si, label, col) in enumerate(zip(spot_indices, spot_labels, colors)):
            diff_t = V_mlmc[:, si] - V_laplace[:, si]
            ax3b.plot(t_pde_grid, diff_t, '-', color=col, linewidth=2, label=label)
        
        ax3b.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3b.set_xlabel('Time (t)', fontsize=12)
        ax3b.set_ylabel('MLMC - Laplace', fontsize=12)
        ax3b.set_title('Price Difference vs Time', fontsize=13)
        ax3b.legend(fontsize=10)
        ax3b.grid(True, alpha=0.3)
        
        fig3.tight_layout()
        
        if save_results:
            fig3.savefig(figures_dir / "exp4_price_evolution.png", dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig3)
    
    # Save results table
    if save_results:
        with open(tables_dir / "exp4_option_prices.md", 'w') as f:
            f.write("# Experiment 4: Option Pricing Results\n\n")
            f.write("## Parameters\n\n")
            f.write(f"- Dimension: d = {params.d}\n")
            f.write(f"- Strike: K = {params.K}\n")
            f.write(f"- Initial basket: S0 = {params.S0}\n")
            f.write(f"- Maturity: T = {params.T}\n")
            f.write(f"- Option type: {params.option_type}\n\n")
            
            f.write("## Price Comparison at Key Spot Values\n\n")
            f.write("| S | MLMC Price | Laplace Price | Difference | Diff (%) |\n")
            f.write("|---|------------|---------------|------------|----------|\n")
            
            key_spots = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
            for mult in key_spots:
                target_S = params.S0 * mult
                idx = np.argmin(np.abs(S_pde_grid - target_S))
                S = S_pde_grid[idx]
                p_mlmc = prices_mlmc[idx]
                p_lap = prices_laplace[idx]
                diff = price_diff[idx]
                diff_p = price_diff_pct[idx] if nonzero[idx] else np.nan
                f.write(f"| {S:.1f} | {p_mlmc:.4f} | {p_lap:.4f} | {diff:.4f} | ")
                if np.isfinite(diff_p):
                    f.write(f"{diff_p:.2f} |\n")
                else:
                    f.write("N/A |\n")
            
            f.write("\n## Statistics\n\n")
            f.write(f"- Mean absolute difference: {np.mean(price_diff):.4f}\n")
            f.write(f"- Max absolute difference: {np.max(price_diff):.4f}\n")
            f.write(f"- Mean percentage difference: {np.mean(price_diff_pct[nonzero]):.2f}%\n")
        
        if verbose:
            print(f"\nResults saved to:")
            print(f"  {figures_dir / 'exp4_*.png'}")
            print(f"  {tables_dir / 'exp4_option_prices.md'}")
    
    if verbose:
        print()
        print("=" * 70)
        print("EXPERIMENT 4 COMPLETE")
        print("=" * 70)
    
    return {
        "S_values": S_pde_grid,
        "prices_mlmc": prices_mlmc,
        "prices_laplace": prices_laplace,
        "price_diff": price_diff,
        "price_diff_pct": price_diff_pct,
        "V_mlmc": V_mlmc,
        "V_laplace": V_laplace,
        "t_grid": t_pde_grid,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 4: Option Pricing Comparison"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display plots interactively"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    results = run_experiment(
        save_results=not args.no_save,
        show_plots=args.show,
        verbose=not args.quiet
    )
