"""
Experiment 2: MLMC Convergence and Uncertainty Analysis

Comprehensive MLMC diagnostic study with three separate analyses:

1. MLMC Self-Convergence (PRIMARY):
   - Level corrections m_ℓ = ||b²_ℓ - b²_{ℓ-1}||
   - Weak convergence rate α
   - Variance decay rate β
   - Richardson extrapolation bias estimate

2. Statistical Uncertainty:
   - Across-run standard error (std / √n)
   - Variance Reduction Factor (VRF)
   - Correlation and kurtosis checks

3. Method Agreement (SECONDARY):
   - MLMC vs Laplace comparison
   - NOTE: Neither method is ground truth - this is disagreement, not error

Expected Outputs
----------------
- results/figures/exp2_mlmc_diagnostics.png (Giles-style panel)
- results/figures/exp2_additional_diagnostics.png (VRF, kurtosis, GCI)
- results/figures/exp2_statistical_uncertainty.png
- results/figures/exp2_method_agreement.png
- results/tables/exp2_mlmc_diagnostics.md

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
from methods.mlmc_ot_estimator import (
    estimate_volatility_mlmc,
    estimate_volatility_mlmc_multiple_runs,
    estimate_domain,
    make_c_ot,
    tot_degree_poly,
    make_b_bar
)
from methods.laplace_wrapper import estimate_volatility_laplace
from methods.common import (
    compute_method_agreement,
    compute_mlmc_level_diagnostics,
    compute_across_run_uncertainty,
    format_metrics_table,
)
from visualisation.diagnostic_plots import (
    plot_mlmc_convergence_diagnostics,
    plot_additional_diagnostics,
    plot_statistical_uncertainty,
    plot_method_agreement,
    plot_l2_disagreement_vs_runs,
)


def run_experiment(
    params=None,
    n_runs: int = 20,
    save_results: bool = True,
    show_plots: bool = False,
    verbose: bool = True
):
    """
    Run Experiment 2: MLMC Convergence and Uncertainty Analysis.

    This experiment performs three types of analysis:

    1. MLMC Self-Convergence Analysis (PRIMARY):
       - Computes level-by-level diagnostics using LevelStats
       - Estimates convergence rates α and β
       - Provides Richardson extrapolation bias estimate

    2. Statistical Uncertainty Analysis:
       - Runs MLMC multiple times to estimate variance
       - Computes standard error following 1/√n

    3. Method Agreement Analysis (SECONDARY):
       - Compares MLMC to Laplace
       - NOTE: Neither is ground truth, this measures disagreement

    Parameters
    ----------
    params : ProblemParameters, optional
        Problem parameters.
    n_runs : int
        Number of MLMC runs for statistical uncertainty study.
    save_results : bool
        Whether to save figures and tables.
    show_plots : bool
        Whether to display plots interactively.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        Dictionary containing all results and diagnostics.
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
        print("EXPERIMENT 2: MLMC Convergence and Uncertainty Analysis")
        print("=" * 70)
        print()
        print("This experiment has THREE separate analyses:")
        print("  1. MLMC Self-Convergence (level diagnostics)")
        print("  2. Statistical Uncertainty (across-run variance)")
        print("  3. Method Agreement (MLMC vs Laplace - neither is ground truth)")
        print()
        print(f"Number of MLMC runs for uncertainty analysis: {n_runs}")
        print()

    # Step 1: Estimate domain
    if verbose:
        print("-" * 50)
        print("Step 1: Estimating domain via pilot run...")
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

    # Define evaluation grid
    margin = 0.02
    S_range = S_max - S_min
    t_grid = np.linspace(0.02, params.T, 15)
    s_grid = np.linspace(
        S_min + margin * S_range,
        S_max - margin * S_range,
        25
    )

    if verbose:
        print(f"Domain from pilot: [{S_min:.1f}, {S_max:.1f}]")
        print(f"Evaluation grid: [{s_grid.min():.1f}, {s_grid.max():.1f}]")
        print()

    # =========================================================================
    # ANALYSIS 1: MLMC Self-Convergence (Level Diagnostics)
    # =========================================================================
    if verbose:
        print("-" * 50)
        print("Analysis 1: MLMC Self-Convergence (Level Diagnostics)")
        print("-" * 50)
        print("Computing MLMC with return_stats=True to get level statistics...")

    np.random.seed(params.random_seed)

    # Run make_c_ot with return_stats=True to get level statistics
    c, level_stats = make_c_ot(
        x0=params.x0,
        T=params.T,
        h0=params.h0,
        r=params.r,
        cov_mat=params.corr_matrix,
        vol=params.sigma,
        max_deg=params.max_degree,
        P1=params.P1,
        s_min=S_min,
        s_max=S_max,
        C=80,
        batch_size=50,
        verbose=verbose,
        return_stats=True
    )

    # Compute level diagnostics
    level_diagnostics = compute_mlmc_level_diagnostics(level_stats)

    if verbose:
        print()
        print("Level Diagnostics:")
        print(f"  Weak convergence rate α = {level_diagnostics['alpha']:.2f}")
        print(f"  Variance decay rate β = {level_diagnostics['beta']:.2f}")
        print(f"  Richardson bias estimate = {level_diagnostics['richardson_bias']:.2e}")
        print(f"  Convergence quality: {level_diagnostics['convergence_quality'].upper()}")
        if level_diagnostics['warnings']:
            print("  Warnings:")
            for w in level_diagnostics['warnings'][:5]:
                print(f"    - {w}")
        print()

    # =========================================================================
    # ANALYSIS 2: Statistical Uncertainty (Multiple Runs)
    # =========================================================================
    if verbose:
        print("-" * 50)
        print("Analysis 2: Statistical Uncertainty (Multiple Runs)")
        print("-" * 50)
        print(f"Running {n_runs} independent MLMC iterations...")

    result_mlmc_mean, all_b_squared, all_times = estimate_volatility_mlmc_multiple_runs(
        params, t_grid, s_grid,
        n_runs=n_runs,
        base_seed=params.random_seed,
        use_ot=True,
        verbose=verbose
    )

    if verbose:
        print()
        print("-" * 50)
        print("Running Laplace approximation (for method agreement)...")
        print("-" * 50)

    result_laplace = estimate_volatility_laplace(
        params, t_grid, s_grid,
        verbose=verbose
    )

    # Compute across-run uncertainty
    uncertainty_stats = compute_across_run_uncertainty(
        all_b_squared,
        reference_surface=result_laplace.b_squared_values
    )

    if verbose:
        print()
        print("Statistical Uncertainty:")
        print(f"  Number of runs: {uncertainty_stats['n_runs']}")
        print(f"  Mean relative std: {uncertainty_stats['relative_std']:.4f} ({uncertainty_stats['relative_std']*100:.2f}%)")
        print(f"  Mean relative SE:  {uncertainty_stats['relative_se']:.4f} ({uncertainty_stats['relative_se']*100:.2f}%)")
        print(f"  Mean time per run: {np.mean(all_times):.2f}s")
        print()

    # =========================================================================
    # ANALYSIS 3: Method Agreement (MLMC vs Laplace)
    # =========================================================================
    if verbose:
        print("-" * 50)
        print("Analysis 3: Method Agreement (MLMC vs Laplace)")
        print("NOTE: Neither method is ground truth!")
        print("-" * 50)

    agreement_metrics = compute_method_agreement(
        result_mlmc_mean.b_squared_values,
        result_laplace.b_squared_values
    )

    if verbose:
        print()
        print(format_metrics_table(agreement_metrics))
        print()

    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    if save_results or show_plots:
        if verbose:
            print("-" * 50)
            print("Generating visualisations...")
            print("-" * 50)

        # 1. MLMC Convergence Diagnostics (Giles-style)
        fig1 = plot_mlmc_convergence_diagnostics(
            level_diagnostics,
            save_path=str(figures_dir / "exp2_mlmc_diagnostics.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig1)

        # 2. Additional Diagnostics (VRF, kurtosis)
        fig2 = plot_additional_diagnostics(
            level_diagnostics,
            save_path=str(figures_dir / "exp2_additional_diagnostics.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig2)

        # 3. Statistical Uncertainty
        fig3 = plot_statistical_uncertainty(
            uncertainty_stats,
            t_grid, s_grid,
            save_path=str(figures_dir / "exp2_statistical_uncertainty.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig3)

        # 4. Method Agreement
        fig4 = plot_method_agreement(
            result_mlmc_mean.b_squared_values,
            result_laplace.b_squared_values,
            t_grid, s_grid,
            metrics=agreement_metrics,
            save_path=str(figures_dir / "exp2_method_agreement.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig4)

        # 5. L2 Disagreement vs runs
        fig5 = plot_l2_disagreement_vs_runs(
            uncertainty_stats,
            save_path=str(figures_dir / "exp2_disagreement_vs_runs.png") if save_results else None,
            show=show_plots
        )
        plt.close(fig5)

    # =========================================================================
    # Save Results to Markdown
    # =========================================================================
    if save_results:
        with open(tables_dir / "exp2_mlmc_diagnostics.md", 'w') as f:
            f.write("# Experiment 2: MLMC Diagnostics and Uncertainty Analysis\n\n")

            f.write("## 1. MLMC Self-Convergence (Level Diagnostics)\n\n")
            f.write("These diagnostics assess MLMC internal convergence.\n\n")
            f.write("| Metric | Value | Status |\n")
            f.write("|--------|-------|--------|\n")
            alpha = level_diagnostics['alpha']
            beta = level_diagnostics['beta']
            f.write(f"| Weak convergence rate α | {alpha:.3f} | {'OK' if not np.isnan(alpha) and alpha >= 0.5 else 'WARN'} |\n")
            f.write(f"| Variance decay rate β | {beta:.3f} | {'OK' if not np.isnan(beta) and beta >= 1.0 else 'WARN'} |\n")
            f.write(f"| Richardson bias | {level_diagnostics['richardson_bias']:.2e} | - |\n")
            f.write(f"| Convergence quality | {level_diagnostics['convergence_quality']} | - |\n")
            f.write("\n")

            f.write("### Per-Level Statistics\n\n")
            f.write("| Level | Variance | Correlation | VRF | Kurtosis |\n")
            f.write("|-------|----------|-------------|-----|----------|\n")
            for l in sorted(level_stats.keys()):
                stats = level_stats[l]
                f.write(f"| {l} | {stats.variance:.2e} | {stats.correlation:.3f} | "
                        f"{stats.variance_reduction_factor:.1f} | {stats.kurtosis:.1f} |\n")
            f.write("\n")

            if level_diagnostics['warnings']:
                f.write("### Warnings\n\n")
                for w in level_diagnostics['warnings']:
                    f.write(f"- {w}\n")
                f.write("\n")

            f.write("## 2. Statistical Uncertainty (Across-Run)\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Number of runs | {uncertainty_stats['n_runs']} |\n")
            f.write(f"| Mean relative std | {uncertainty_stats['relative_std']*100:.2f}% |\n")
            f.write(f"| Mean relative SE | {uncertainty_stats['relative_se']*100:.2f}% |\n")
            f.write(f"| Mean time per run | {np.mean(all_times):.2f} s |\n")
            f.write(f"| Total time | {np.sum(all_times):.1f} s |\n")
            f.write("\n")

            f.write("## 3. Method Agreement (MLMC vs Laplace)\n\n")
            f.write("**IMPORTANT**: Neither method is ground truth. These metrics\n")
            f.write("measure disagreement, not error.\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| L² disagreement | {agreement_metrics['l2_disagreement']:.6f} |\n")
            f.write(f"| L∞ disagreement | {agreement_metrics['linf_disagreement']:.4f} |\n")
            f.write(f"| Mean relative diff | {agreement_metrics['mean_relative_difference']*100:.2f}% |\n")
            f.write(f"| Correlation | {agreement_metrics['correlation']:.4f} |\n")
            f.write(f"| Bias (MLMC - Laplace) | {agreement_metrics['bias']:.4f} |\n")
            f.write("\n")

        if verbose:
            print(f"\nResults saved to:")
            print(f"  {figures_dir / 'exp2_*.png'}")
            print(f"  {tables_dir / 'exp2_mlmc_diagnostics.md'}")

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
        "level_stats": level_stats,
        "level_diagnostics": level_diagnostics,
        "uncertainty_stats": uncertainty_stats,
        "agreement_metrics": agreement_metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment 2: MLMC Convergence and Uncertainty Analysis"
    )
    parser.add_argument(
        "-n", "--n-runs", type=int, default=20,
        help="Number of MLMC runs for uncertainty analysis (default: 20)"
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
