"""
Experiment 1: Direct Surface Comparison

Compare the projected volatility surfaces b²(t, S) from MLMC and Laplace
methods on the paper's 3D Black-Scholes test case (Equation 56).

This experiment:
1. Runs both methods with identical parameters
2. Computes accuracy metrics
3. Generates comparison visualisations
4. Saves results to files

Expected Outputs
----------------
- results/figures/exp1_surface_comparison.png
- results/figures/exp1_difference_heatmap.png
- results/figures/exp1_summary_panel.png
- results/tables/exp1_metrics.md

Author: Wadoud (KAUST Internship)
Reference: Bayer, Häppölä, Tempone (2017) Section 3.5.2
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
_script_dir = Path(__file__).resolve().parent
_comparison_dir = _script_dir.parent
sys.path.insert(0, str(_comparison_dir))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt

# Import from our modules
from config import DEFAULT_PARAMS
from methods.mlmc_ot_estimator import estimate_volatility_mlmc, estimate_domain
from methods.laplace_wrapper import estimate_volatility_laplace
from methods.common import (
    compute_accuracy_metrics,
    format_metrics_table,
    save_metrics_markdown,
)
from visualisation import (
    plot_surface_comparison,
    plot_difference_heatmap,
)
from visualisation.error_plots import plot_summary_panel, plot_scatter_comparison


def run_experiment(
    params=None,
    save_results: bool = True,
    show_plots: bool = False,
    verbose: bool = True
):
    """
    Run Experiment 1: Direct surface comparison.
    
    Parameters
    ----------
    params : ProblemParameters, optional
        Problem parameters. Uses DEFAULT_PARAMS if not provided.
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
        - result_mlmc: MLMC VolatilitySurfaceResult
        - result_laplace: Laplace VolatilitySurfaceResult
        - metrics: Accuracy metrics dictionary
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
        print("EXPERIMENT 1: Direct Surface Comparison")
        print("=" * 70)
        print()
        print(params)
        print()

    # Step 1: Run pilot to get MLMC domain bounds
    if verbose:
        print("-" * 50)
        print("Estimating MLMC domain via pilot run...")
        print("-" * 50)

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
        print(f"Evaluation grid: {len(t_grid)} × {len(s_grid)}")
        print(f"  t ∈ [{t_grid.min():.3f}, {t_grid.max():.3f}]")
        print(f"  s ∈ [{s_grid.min():.1f}, {s_grid.max():.1f}]")
        print()
    
    # Run MLMC estimation
    if verbose:
        print("-" * 50)
        print("Running MLMC estimation...")
        print("-" * 50)
    
    result_mlmc = estimate_volatility_mlmc(
        params, t_grid, s_grid,
        random_seed=params.random_seed,
        use_ot=True,
        verbose=verbose
    )
    
    if verbose:
        print()
        print("-" * 50)
        print("Running Laplace approximation...")
        print("-" * 50)
    
    # Run Laplace approximation
    result_laplace = estimate_volatility_laplace(
        params, t_grid, s_grid,
        verbose=verbose
    )
    
    # Compute accuracy metrics
    if verbose:
        print()
        print("-" * 50)
        print("Computing accuracy metrics...")
        print("-" * 50)
    
    metrics = compute_accuracy_metrics(
        result_mlmc.b_squared_values,
        result_laplace.b_squared_values
    )
    
    if verbose:
        print()
        print(format_metrics_table(metrics))
    
    # Generate visualisations
    if save_results or show_plots:
        if verbose:
            print()
            print("-" * 50)
            print("Generating visualisations...")
            print("-" * 50)
        
        # 1. Side-by-side surface comparison
        fig1 = plot_surface_comparison(
            result_mlmc, result_laplace,
            save_path=str(figures_dir / "exp1_surface_comparison.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig1)
        
        # 2. Difference heatmap
        fig2 = plot_difference_heatmap(
            result_mlmc, result_laplace,
            save_path=str(figures_dir / "exp1_difference_heatmap.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig2)
        
        # 3. Summary panel
        fig3 = plot_summary_panel(
            result_mlmc, result_laplace, metrics,
            save_path=str(figures_dir / "exp1_summary_panel.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig3)
        
        # 4. Scatter comparison
        fig4 = plot_scatter_comparison(
            result_mlmc, result_laplace, metrics,
            save_path=str(figures_dir / "exp1_scatter.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig4)
    
    # Save metrics to markdown
    if save_results:
        save_metrics_markdown(
            metrics,
            str(tables_dir / "exp1_metrics.md"),
            params_summary=str(params)
        )
        
        if verbose:
            print(f"\nResults saved to:")
            print(f"  {figures_dir / 'exp1_*.png'}")
            print(f"  {tables_dir / 'exp1_metrics.md'}")
    
    if verbose:
        print()
        print("=" * 70)
        print("EXPERIMENT 1 COMPLETE")
        print("=" * 70)
    
    return {
        "result_mlmc": result_mlmc,
        "result_laplace": result_laplace,
        "metrics": metrics,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 1: Direct Surface Comparison"
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
