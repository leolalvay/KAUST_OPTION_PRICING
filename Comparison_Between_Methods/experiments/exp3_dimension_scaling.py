"""
Experiment 3: Dimension Scaling Study

Investigate how MLMC and Laplace approximation methods scale with the
number of assets d in the basket option.

This experiment:
1. Runs both methods for d = 2, 3, 5, 10 assets
2. Compares computation times
3. Measures accuracy as dimension increases
4. Studies the "curse of dimensionality" mitigation

Key Question: Does Markovian projection successfully reduce the
d-dimensional problem to 1D regardless of d?

Expected Outputs
----------------
- results/figures/exp3_time_vs_dimension.png
- results/figures/exp3_error_vs_dimension.png
- results/tables/exp3_dimension_scaling.md

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

from config import (
    DEFAULT_PARAMS,
    create_2d_params,
    create_5d_params,
    create_10d_params,
)
from methods.mlmc_ot_estimator import estimate_volatility_mlmc, estimate_domain
from methods.laplace_wrapper import estimate_volatility_laplace
from methods.common import compute_accuracy_metrics


def run_experiment(
    dimensions: list = None,
    save_results: bool = True,
    show_plots: bool = False,
    verbose: bool = True
):
    """
    Run Experiment 3: Dimension scaling study.
    
    Parameters
    ----------
    dimensions : list of int, optional
        Dimensions to test. Default: [2, 3, 5, 10]
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
        - dimensions: List of tested dimensions
        - mlmc_times: MLMC computation times per dimension
        - laplace_times: Laplace computation times per dimension
        - l2_errors: L2 errors between methods
        - all_results: Full results for each dimension
    """
    if dimensions is None:
        dimensions = [2, 3, 5, 10]
    
    # Setup paths
    results_dir = _comparison_dir / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    
    for d in [results_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("EXPERIMENT 3: Dimension Scaling Study")
        print("=" * 70)
        print()
        print(f"Testing dimensions: {dimensions}")
        print()
    
    # Create parameter sets for each dimension
    param_creators = {
        2: create_2d_params,
        3: lambda: DEFAULT_PARAMS.copy(),  # Default is 3D
        5: create_5d_params,
        10: create_10d_params,
    }
    
    # Storage for results
    mlmc_times = []
    laplace_times = []
    l2_errors = []
    linf_errors = []
    all_results = {}
    
    for dim in dimensions:
        if verbose:
            print("-" * 50)
            print(f"Testing d = {dim} assets")
            print("-" * 50)
        
        # Get parameters for this dimension
        if dim in param_creators:
            params = param_creators[dim]()
        else:
            if verbose:
                print(f"  Warning: No predefined parameters for d={dim}, skipping")
            continue
        
        if verbose:
            print(f"  S0 = {params.S0:.1f}, K = {params.K:.1f}")
            print(f"  σ = {params.sigma}")

        # Run pilot to get MLMC domain bounds
        if verbose:
            print(f"  Estimating MLMC domain via pilot run...")

        np.random.seed(params.random_seed)
        S_min, S_max = estimate_domain(
            x0=params.x0,
            T=params.T,
            h0=params.h0,
            r=params.r,
            cov_mat=params.corr_matrix,
            vol=params.sigma,
            max_deg=params.max_degree,
            P1=params.P1,
            M_pilot=10000
        )

        # Define grids WITHIN the domain (with safety margin)
        margin = 0.02  # 2% inset from boundaries
        S_range = S_max - S_min

        t_grid = np.linspace(0.02, params.T, 15)
        s_grid = np.linspace(
            S_min + margin * S_range,  # Slightly above S_min
            S_max - margin * S_range,  # Slightly below S_max
            25
        )

        if verbose:
            print(f"  Domain from pilot: [{S_min:.1f}, {S_max:.1f}]")
            print(f"  Evaluation grid: [{s_grid.min():.1f}, {s_grid.max():.1f}]")
        
        # Run MLMC
        if verbose:
            print(f"  Running MLMC...")
        
        result_mlmc = estimate_volatility_mlmc(
            params, t_grid, s_grid,
            use_ot=True,
            verbose=False
        )
        mlmc_times.append(result_mlmc.computation_time)
        
        if verbose:
            print(f"    Time: {result_mlmc.computation_time:.2f}s")
        
        # Run Laplace
        if verbose:
            print(f"  Running Laplace approximation...")
        
        result_laplace = estimate_volatility_laplace(
            params, t_grid, s_grid,
            verbose=False
        )
        laplace_times.append(result_laplace.computation_time)
        
        if verbose:
            print(f"    Time: {result_laplace.computation_time:.2f}s")
        
        # Compute accuracy metrics
        metrics = compute_accuracy_metrics(
            result_mlmc.b_squared_values,
            result_laplace.b_squared_values
        )
        l2_errors.append(metrics['l2_relative_error'])
        linf_errors.append(metrics['linf_relative_error'])
        
        if verbose:
            print(f"  L2 relative error: {metrics['l2_relative_error']:.4f}")
            print(f"  L∞ relative error: {metrics['linf_relative_error']:.4f}")
        
        # Store full results
        all_results[dim] = {
            'params': params,
            'result_mlmc': result_mlmc,
            'result_laplace': result_laplace,
            'metrics': metrics,
        }
    
    # Generate visualisations
    if save_results or show_plots:
        if verbose:
            print()
            print("-" * 50)
            print("Generating visualisations...")
            print("-" * 50)
        
        # 1. Time vs Dimension
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(dimensions))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, mlmc_times, width, label='MLMC', color='steelblue')
        bars2 = ax1.bar(x_pos + width/2, laplace_times, width, label='Laplace', color='darkorange')
        
        ax1.set_xlabel('Number of Assets (d)', fontsize=12)
        ax1.set_ylabel('Computation Time (s)', fontsize=12)
        ax1.set_title('Computation Time vs Basket Dimension', fontsize=13)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([str(d) for d in dimensions])
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}s',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}s',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        fig1.tight_layout()
        
        if save_results:
            fig1.savefig(figures_dir / "exp3_time_vs_dimension.png", dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig1)
        
        # 2. Error vs Dimension
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
        
        # L2 errors
        ax2a.plot(dimensions, l2_errors, 'bo-', markersize=10, linewidth=2)
        ax2a.set_xlabel('Number of Assets (d)', fontsize=12)
        ax2a.set_ylabel(r'$L^2$ Relative Error', fontsize=12)
        ax2a.set_title(r'$L^2$ Error vs Dimension', fontsize=13)
        ax2a.grid(True, alpha=0.3)
        ax2a.set_xticks(dimensions)
        
        # L-infinity errors
        ax2b.plot(dimensions, linf_errors, 'rs-', markersize=10, linewidth=2)
        ax2b.set_xlabel('Number of Assets (d)', fontsize=12)
        ax2b.set_ylabel(r'$L^\infty$ Relative Error', fontsize=12)
        ax2b.set_title(r'$L^\infty$ Error vs Dimension', fontsize=13)
        ax2b.grid(True, alpha=0.3)
        ax2b.set_xticks(dimensions)
        
        fig2.tight_layout()
        
        if save_results:
            fig2.savefig(figures_dir / "exp3_error_vs_dimension.png", dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig2)
        
        # 3. Combined scaling plot (log scale)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        ax3.semilogy(dimensions, mlmc_times, 'bo-', markersize=10, linewidth=2, label='MLMC Time')
        ax3.semilogy(dimensions, laplace_times, 'rs-', markersize=10, linewidth=2, label='Laplace Time')
        
        # Add theoretical O(d) reference
        if len(dimensions) > 1:
            ref_scale = mlmc_times[0] / dimensions[0]
            ref_line = [ref_scale * d for d in dimensions]
            ax3.semilogy(dimensions, ref_line, 'k--', alpha=0.5, label=r'$\mathcal{O}(d)$ reference')
        
        ax3.set_xlabel('Number of Assets (d)', fontsize=12)
        ax3.set_ylabel('Computation Time (s) [log scale]', fontsize=12)
        ax3.set_title('Scaling Behaviour (Log Scale)', fontsize=13)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_xticks(dimensions)
        ax3.set_xticklabels([str(d) for d in dimensions])
        
        fig3.tight_layout()
        
        if save_results:
            fig3.savefig(figures_dir / "exp3_scaling_log.png", dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig3)
    
    # Save results table
    if save_results:
        with open(tables_dir / "exp3_dimension_scaling.md", 'w') as f:
            f.write("# Experiment 3: Dimension Scaling Results\n\n")
            f.write("## Summary\n\n")
            f.write("| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |\n")
            f.write("|---|---------------|------------------|----------|----------|\n")
            for i, dim in enumerate(dimensions):
                f.write(f"| {dim} | {mlmc_times[i]:.2f} | {laplace_times[i]:.2f} | ")
                f.write(f"{l2_errors[i]:.4f} | {linf_errors[i]:.4f} |\n")
            
            f.write("\n## Key Observations\n\n")
            f.write("- Markovian projection reduces d-dimensional problem to 1D PDE\n")
            f.write("- Main computational cost scales with number of Monte Carlo paths\n")
            f.write("- Laplace approximation remains purely analytical\n")
            f.write("- Both methods avoid exponential curse of dimensionality\n")
        
        if verbose:
            print(f"\nResults saved to:")
            print(f"  {figures_dir / 'exp3_*.png'}")
            print(f"  {tables_dir / 'exp3_dimension_scaling.md'}")
    
    if verbose:
        print()
        print("=" * 70)
        print("EXPERIMENT 3 COMPLETE")
        print("=" * 70)
    
    return {
        "dimensions": dimensions,
        "mlmc_times": mlmc_times,
        "laplace_times": laplace_times,
        "l2_errors": l2_errors,
        "linf_errors": linf_errors,
        "all_results": all_results,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 3: Dimension Scaling Study"
    )
    parser.add_argument(
        "-d", "--dimensions", type=int, nargs='+', default=[2, 3, 5, 10],
        help="Dimensions to test (default: 2 3 5 10)"
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
        dimensions=args.dimensions,
        save_results=not args.no_save,
        show_plots=args.show,
        verbose=not args.quiet
    )
