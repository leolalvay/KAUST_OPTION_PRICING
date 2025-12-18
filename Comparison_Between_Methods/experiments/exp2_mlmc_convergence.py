"""
Experiment 2: MLMC Convergence Study

Verify that MLMC estimates converge as the number of runs increases,
and study the variance reduction properties.

This experiment:
1. Runs Laplace once as deterministic reference
2. Runs MLMC multiple times with different seeds
3. Computes mean and standard deviation of MLMC estimates
4. Plots convergence of L2 error with averaging
5. Validates 1/sqrt(n) convergence rate

Expected Outputs
----------------
- results/figures/exp2_convergence_curves.png
- results/figures/exp2_confidence_bands.png
- results/tables/exp2_convergence_stats.md

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
    estimate_volatility_laplace,
    compute_accuracy_metrics,
)
from methods.mlmc_wrapper import estimate_volatility_mlmc_multiple_runs
from visualisation.convergence_plots import (
    plot_mlmc_convergence,
    plot_confidence_bands,
)
from PDE.mlmc_volatility_estimation import estimate_basket_domain


def run_experiment(
    params=None,
    n_runs: int = 20,
    save_results: bool = True,
    show_plots: bool = False,
    verbose: bool = True
):
    """
    Run Experiment 2: MLMC convergence study.
    
    Parameters
    ----------
    params : ProblemParameters, optional
        Problem parameters.
    n_runs : int
        Number of MLMC runs for convergence study.
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
        - result_laplace: Laplace VolatilitySurfaceResult
        - result_mlmc_mean: Mean MLMC result
        - all_b_squared: All MLMC surfaces (n_runs, n_t, n_s)
        - all_times: Computation times per run
        - convergence_stats: Statistics dict
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
        print("EXPERIMENT 2: MLMC Convergence Study")
        print("=" * 70)
        print()
        print(f"Number of MLMC runs: {n_runs}")
        print()

    # Step 1: Run pilot to get MLMC domain bounds
    if verbose:
        print("-" * 50)
        print("Estimating MLMC domain via pilot run...")
        print("-" * 50)

    np.random.seed(params.random_seed)
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

    # Step 2: Define grids WITHIN the domain (with safety margin)
    margin = 0.02  # 2% inset from boundaries
    S_range = S_max - S_min

    t_grid = np.linspace(0.02, params.T, 15)
    s_grid = np.linspace(
        S_min + margin * S_range,  # Slightly above S_min
        S_max - margin * S_range,  # Slightly below S_max
        25
    )

    if verbose:
        print(f"Domain from pilot: [{S_min:.1f}, {S_max:.1f}]")
        print(f"Evaluation grid: [{s_grid.min():.1f}, {s_grid.max():.1f}]")
        print()
    
    # Run Laplace (deterministic reference)
    if verbose:
        print("-" * 50)
        print("Running Laplace approximation (reference)...")
        print("-" * 50)
    
    result_laplace = estimate_volatility_laplace(
        params, t_grid, s_grid,
        verbose=verbose
    )
    
    # Run MLMC multiple times
    if verbose:
        print()
        print("-" * 50)
        print(f"Running {n_runs} MLMC iterations...")
        print("-" * 50)
    
    result_mlmc_mean, all_b_squared, all_times = estimate_volatility_mlmc_multiple_runs(
        params, t_grid, s_grid,
        n_runs=n_runs,
        base_seed=params.random_seed,
        verbose=verbose
    )
    
    # Compute statistics
    if verbose:
        print()
        print("-" * 50)
        print("Computing convergence statistics...")
        print("-" * 50)
    
    # L2 errors for running averages
    l2_errors_running = []
    for n in range(1, n_runs + 1):
        running_mean = np.mean(all_b_squared[:n], axis=0)
        diff = running_mean - result_laplace.b_squared_values
        valid = np.isfinite(diff) & np.isfinite(result_laplace.b_squared_values)
        if np.any(valid):
            norm = np.sqrt(np.mean(result_laplace.b_squared_values[valid] ** 2))
            l2 = np.sqrt(np.mean(diff[valid] ** 2)) / norm if norm > 0 else np.inf
        else:
            l2 = np.inf
        l2_errors_running.append(l2)
    
    # Individual run errors
    l2_errors_individual = []
    for i in range(n_runs):
        diff = all_b_squared[i] - result_laplace.b_squared_values
        valid = np.isfinite(diff) & np.isfinite(result_laplace.b_squared_values)
        if np.any(valid):
            norm = np.sqrt(np.mean(result_laplace.b_squared_values[valid] ** 2))
            l2 = np.sqrt(np.mean(diff[valid] ** 2)) / norm if norm > 0 else np.inf
        else:
            l2 = np.inf
        l2_errors_individual.append(l2)
    
    convergence_stats = {
        "n_runs": n_runs,
        "mean_time_per_run": np.mean(all_times),
        "std_time_per_run": np.std(all_times),
        "total_time": np.sum(all_times),
        "mean_l2_error": np.mean(l2_errors_individual),
        "std_l2_error": np.std(l2_errors_individual),
        "final_running_l2": l2_errors_running[-1],
        "l2_errors_running": l2_errors_running,
        "l2_errors_individual": l2_errors_individual,
    }
    
    if verbose:
        print()
        print("Convergence Statistics:")
        print(f"  Mean L2 error (individual): {convergence_stats['mean_l2_error']:.4f}")
        print(f"  Std L2 error:               {convergence_stats['std_l2_error']:.4f}")
        print(f"  Final running avg L2:       {convergence_stats['final_running_l2']:.4f}")
        print(f"  Mean time per run:          {convergence_stats['mean_time_per_run']:.2f}s")
        print(f"  Total computation time:     {convergence_stats['total_time']:.1f}s")
    
    # Generate visualisations
    if save_results or show_plots:
        if verbose:
            print()
            print("-" * 50)
            print("Generating visualisations...")
            print("-" * 50)
        
        # 1. Main convergence plot
        fig1 = plot_mlmc_convergence(
            all_b_squared, result_laplace,
            t_grid, s_grid,
            save_path=str(figures_dir / "exp2_convergence_curves.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig1)
        
        # 2. Confidence bands at S0
        fig2 = plot_confidence_bands(
            result_mlmc_mean, all_b_squared, result_laplace,
            s_value=params.S0,
            save_path=str(figures_dir / "exp2_confidence_bands.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig2)
        
        # 3. Convergence rate analysis plot
        fig3, ax = plt.subplots(figsize=(10, 6))
        
        n_vals = np.arange(1, n_runs + 1)
        ax.semilogy(n_vals, l2_errors_running, 'b-o', markersize=5, label=r'Running Average $L^2$ Error')

        # Add 1/sqrt(n) reference
        ref = l2_errors_running[0] / np.sqrt(n_vals)
        ax.semilogy(n_vals, ref, 'r--', alpha=0.7, label=r'Reference: $\propto 1/\sqrt{n}$')

        ax.set_xlabel('Number of Runs', fontsize=12)
        ax.set_ylabel(r'$L^2$ Relative Error', fontsize=12)
        ax.set_title('MLMC Convergence Rate', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        if save_results:
            fig3.savefig(figures_dir / "exp2_convergence_rate.png", dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig3)
    
    # Save statistics to markdown
    if save_results:
        with open(tables_dir / "exp2_convergence_stats.md", 'w') as f:
            f.write("# Experiment 2: MLMC Convergence Statistics\n\n")
            f.write("## Summary\n\n")
            f.write("| Statistic | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| Number of runs | {n_runs} |\n")
            f.write(f"| Mean L2 error (individual) | {convergence_stats['mean_l2_error']:.6f} |\n")
            f.write(f"| Std L2 error | {convergence_stats['std_l2_error']:.6f} |\n")
            f.write(f"| Final running avg L2 | {convergence_stats['final_running_l2']:.6f} |\n")
            f.write(f"| Mean time per run | {convergence_stats['mean_time_per_run']:.2f} s |\n")
            f.write(f"| Total computation time | {convergence_stats['total_time']:.1f} s |\n")
            f.write("\n")
            f.write("## Running Average L2 Errors\n\n")
            f.write("| n | L2 Error |\n")
            f.write("|---|----------|\n")
            for i, err in enumerate(l2_errors_running):
                if i < 5 or i >= n_runs - 3 or i % 5 == 0:
                    f.write(f"| {i+1} | {err:.6f} |\n")
        
        if verbose:
            print(f"\nResults saved to:")
            print(f"  {figures_dir / 'exp2_*.png'}")
            print(f"  {tables_dir / 'exp2_convergence_stats.md'}")
    
    if verbose:
        print()
        print("=" * 70)
        print("EXPERIMENT 2 COMPLETE")
        print("=" * 70)
    
    return {
        "result_laplace": result_laplace,
        "result_mlmc_mean": result_mlmc_mean,
        "all_b_squared": all_b_squared,
        "all_times": all_times,
        "convergence_stats": convergence_stats,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 2: MLMC Convergence Study"
    )
    parser.add_argument(
        "-n", "--n-runs", type=int, default=20,
        help="Number of MLMC runs (default: 20)"
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
        n_runs=args.n_runs,
        save_results=not args.no_save,
        show_plots=args.show,
        verbose=not args.quiet
    )
