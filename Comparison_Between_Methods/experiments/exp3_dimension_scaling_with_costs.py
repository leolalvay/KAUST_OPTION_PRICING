"""
Experiment 3: Dimension Scaling Study (with Hardware-Independent Cost Metrics)

Investigate how MLMC and Laplace approximation methods scale with the
number of assets d in the basket option.

This experiment:
1. Runs both methods for d = 2, 3, 5, 10 assets
2. Compares computation times AND hardware-independent cost metrics
3. Measures accuracy as dimension increases
4. Studies the "curse of dimensionality" mitigation

Cost Metrics Collected
----------------------
- wall_time: Traditional timing (for reference)
- cpu_time: CPU time excluding sleep/IO (more consistent)
- function_calls: Total Python function calls (hardware-independent)
- peak_memory: Memory usage (hardware-independent)

Expected Outputs
----------------
- results/figures/exp3_time_vs_dimension.png
- results/figures/exp3_cost_metrics.png (hardware-independent metrics)
- results/figures/exp3_disagreement_vs_dimension.png
- results/tables/exp3_dimension_scaling.md

Author: Wadoud (KAUST Internship)
"""

import sys
from pathlib import Path
from contextlib import contextmanager

# Add parent directory to path for imports
# This allows running from Comparison_Between_Methods/ directory
_script_dir = Path(__file__).resolve().parent
_comparison_dir = _script_dir.parent
sys.path.insert(0, str(_comparison_dir))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from our modules
from config import (
    DEFAULT_PARAMS,
    create_2d_params,
    create_5d_params,
    create_10d_params,
    create_nd_params,
)
from methods.mlmc_ot_estimator import estimate_volatility_mlmc, estimate_domain
from methods.laplace_wrapper import estimate_volatility_laplace
from methods.common import compute_method_agreement

# Import cost metrics utilities
from methods.cost_metrics import (
    CostMetrics,
    CostProfiler,
    single_threaded,
    print_comparison_table,
)


# =============================================================================
# Configuration for Benchmarking
# =============================================================================

# Add this to your config.py if you want global control:
# BENCHMARK_CONFIG = {
#     'disable_threading': False,  # Set True for reproducible benchmarks
#     'collect_cost_metrics': True,
# }

BENCHMARK_CONFIG = {
    'disable_threading': False,  # Toggle this for fair comparison
    'collect_cost_metrics': True,
}


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    dimensions: list = None,
    save_results: bool = True,
    show_plots: bool = False,
    verbose: bool = True,
    disable_threading: bool = None,
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
    disable_threading : bool, optional
        Override BENCHMARK_CONFIG setting for threading.
        If True, disables BLAS parallelisation for fair comparison.
        
    Returns
    -------
    dict
        Dictionary containing all results including cost metrics.
    """
    if dimensions is None:
        dimensions = [2, 3, 5, 10]
    
    # Use config or override
    use_single_thread = (
        disable_threading 
        if disable_threading is not None 
        else BENCHMARK_CONFIG.get('disable_threading', False)
    )

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
        print(f"Threading disabled: {use_single_thread}")
        print()
    
    # Parameter creators
    param_creators = {
        2: create_2d_params,
        3: lambda: DEFAULT_PARAMS.copy(),
        5: create_5d_params,
        10: create_10d_params,
    }

    # Storage for results
    mlmc_metrics_list = []
    laplace_metrics_list = []
    l2_disagreements = []
    linf_disagreements = []
    all_results = {}

    # Choose context manager based on threading setting
    if use_single_thread:
        thread_ctx = single_threaded
    else:
        @contextmanager
        def thread_ctx():
            yield

    for dim in dimensions:
        if verbose:
            print("-" * 50)
            print(f"Testing d = {dim} assets")
            print("-" * 50)

        # Get parameters
        if dim in param_creators:
            params = param_creators[dim]()
        else:
            if verbose:
                print(f"  Using generic parameters for d={dim}")
            params = create_nd_params(dim)
        
        if verbose:
            print(f"  S0 = {params.S0:.1f}, K = {params.K:.1f}")
            print(f"  σ = {params.sigma}")

        # Pilot run for domain
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

        # Define grids
        margin = 0.02
        S_range = S_max - S_min
        t_grid = np.linspace(0.02, params.T, 15)
        s_grid = np.linspace(
            S_min + margin * S_range,
            S_max - margin * S_range,
            25
        )

        if verbose:
            print(f"  Domain: [{S_min:.1f}, {S_max:.1f}]")
            print(f"  Grid: [{s_grid.min():.1f}, {s_grid.max():.1f}]")
        
        # =====================================================================
        # Run MLMC with cost profiling
        # =====================================================================
        if verbose:
            print(f"  Running MLMC...")
        
        with thread_ctx():
            with CostProfiler("MLMC") as mlmc_profiler:
                result_mlmc = estimate_volatility_mlmc(
                    params, t_grid, s_grid,
                    use_ot=True,
                    verbose=False
                )
        
        mlmc_metrics = mlmc_profiler.metrics
        mlmc_metrics_list.append(mlmc_metrics)
        
        if verbose:
            print(f"    Wall time:      {mlmc_metrics.wall_time:.2f}s")
            print(f"    CPU time:       {mlmc_metrics.cpu_time:.2f}s")
            print(f"    Function calls: {mlmc_metrics.total_calls:,}")
            print(f"    Peak memory:    {mlmc_metrics.peak_memory_mb:.1f} MB")
        
        # =====================================================================
        # Run Laplace with cost profiling
        # =====================================================================
        if verbose:
            print(f"  Running Laplace approximation...")
        
        with thread_ctx():
            with CostProfiler("Laplace") as laplace_profiler:
                result_laplace = estimate_volatility_laplace(
                    params, t_grid, s_grid,
                    verbose=False
                )
        
        laplace_metrics = laplace_profiler.metrics
        laplace_metrics_list.append(laplace_metrics)
        
        if verbose:
            print(f"    Wall time:      {laplace_metrics.wall_time:.2f}s")
            print(f"    CPU time:       {laplace_metrics.cpu_time:.2f}s")
            print(f"    Function calls: {laplace_metrics.total_calls:,}")
            print(f"    Peak memory:    {laplace_metrics.peak_memory_mb:.1f} MB")
        
        # Compute agreement metrics
        metrics = compute_method_agreement(
            result_mlmc.b_squared_values,
            result_laplace.b_squared_values
        )
        l2_disagreements.append(metrics['l2_disagreement'])
        linf_disagreements.append(metrics['linf_disagreement'])

        if verbose:
            print(f"  L² disagreement: {metrics['l2_disagreement']:.4f}")
        
        # Store full results
        all_results[dim] = {
            'params': params,
            'result_mlmc': result_mlmc,
            'result_laplace': result_laplace,
            'mlmc_metrics': mlmc_metrics,
            'laplace_metrics': laplace_metrics,
            'agreement_metrics': metrics,
        }
    
    # =========================================================================
    # Generate Visualisations
    # =========================================================================
    if save_results or show_plots:
        if verbose:
            print()
            print("-" * 50)
            print("Generating visualisations...")
            print("-" * 50)
        
        # Extract data for plotting
        mlmc_wall = [m.wall_time for m in mlmc_metrics_list]
        mlmc_cpu = [m.cpu_time for m in mlmc_metrics_list]
        mlmc_calls = [m.total_calls for m in mlmc_metrics_list]
        mlmc_mem = [m.peak_memory_mb for m in mlmc_metrics_list]
        
        laplace_wall = [m.wall_time for m in laplace_metrics_list]
        laplace_cpu = [m.cpu_time for m in laplace_metrics_list]
        laplace_calls = [m.total_calls for m in laplace_metrics_list]
        laplace_mem = [m.peak_memory_mb for m in laplace_metrics_list]
        
        x_pos = np.arange(len(dimensions))
        width = 0.35
        
        # -----------------------------------------------------------------
        # 1. Wall Time vs Dimension (original plot)
        # -----------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        bars1 = ax1.bar(x_pos - width/2, mlmc_wall, width, 
                       label='MLMC', color='steelblue')
        bars2 = ax1.bar(x_pos + width/2, laplace_wall, width, 
                       label='Laplace', color='darkorange')
        
        ax1.set_xlabel('Number of Assets (d)', fontsize=12)
        ax1.set_ylabel('Wall-Clock Time (s)', fontsize=12)
        ax1.set_title('Wall-Clock Time vs Dimension', fontsize=13)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([str(d) for d in dimensions])
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
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
            fig1.savefig(figures_dir / "exp3_time_vs_dimension.png", 
                        dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig1)
        
        # -----------------------------------------------------------------
        # 2. Hardware-Independent Cost Metrics (NEW)
        # -----------------------------------------------------------------
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # CPU Time (more consistent than wall time)
        ax = axes[0, 0]
        ax.bar(x_pos - width/2, mlmc_cpu, width, label='MLMC', color='steelblue')
        ax.bar(x_pos + width/2, laplace_cpu, width, label='Laplace', color='darkorange')
        ax.set_xlabel('Number of Assets (d)')
        ax.set_ylabel('CPU Time (s)')
        ax.set_title('CPU Time vs Dimension\n(Excludes sleep/IO, more consistent)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Function Calls (hardware-independent)
        ax = axes[0, 1]
        ax.bar(x_pos - width/2, mlmc_calls, width, label='MLMC', color='steelblue')
        ax.bar(x_pos + width/2, laplace_calls, width, label='Laplace', color='darkorange')
        ax.set_xlabel('Number of Assets (d)')
        ax.set_ylabel('Function Calls')
        ax.set_title('Python Function Calls vs Dimension\n(Hardware-independent!)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Peak Memory (hardware-independent)
        ax = axes[1, 0]
        ax.bar(x_pos - width/2, mlmc_mem, width, label='MLMC', color='steelblue')
        ax.bar(x_pos + width/2, laplace_mem, width, label='Laplace', color='darkorange')
        ax.set_xlabel('Number of Assets (d)')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Peak Memory Usage vs Dimension\n(Hardware-independent)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Scaling comparison (log scale)
        ax = axes[1, 1]
        ax.semilogy(dimensions, mlmc_cpu, 'bo-', markersize=10, linewidth=2, 
                   label='MLMC (CPU time)')
        ax.semilogy(dimensions, laplace_cpu, 'rs-', markersize=10, linewidth=2, 
                   label='Laplace (CPU time)')
        
        # Add O(d) reference line
        if len(dimensions) > 1 and mlmc_cpu[0] > 0:
            ref_scale = mlmc_cpu[0] / dimensions[0]
            ref_line = [ref_scale * d for d in dimensions]
            ax.semilogy(dimensions, ref_line, 'k--', alpha=0.5, 
                       label=r'$\mathcal{O}(d)$ reference')
        
        ax.set_xlabel('Number of Assets (d)')
        ax.set_ylabel('CPU Time (s) [log scale]')
        ax.set_title('Scaling Behaviour (Log Scale)')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xticks(dimensions)
        ax.set_xticklabels([str(d) for d in dimensions])
        
        fig2.suptitle('Hardware-Independent Cost Metrics', fontsize=14, fontweight='bold')
        fig2.tight_layout()
        
        if save_results:
            fig2.savefig(figures_dir / "exp3_cost_metrics.png", 
                        dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig2)
        
        # -----------------------------------------------------------------
        # 3. Disagreement vs Dimension
        # -----------------------------------------------------------------
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

        ax3a.plot(dimensions, l2_disagreements, 'bo-', markersize=10, linewidth=2)
        ax3a.set_xlabel('Number of Assets (d)', fontsize=12)
        ax3a.set_ylabel(r'$L^2$ Disagreement', fontsize=12)
        ax3a.set_title(r'$L^2$ Disagreement vs Dimension', fontsize=13)
        ax3a.grid(True, alpha=0.3)
        ax3a.set_xticks(dimensions)

        ax3b.plot(dimensions, linf_disagreements, 'rs-', markersize=10, linewidth=2)
        ax3b.set_xlabel('Number of Assets (d)', fontsize=12)
        ax3b.set_ylabel(r'$L^\infty$ Disagreement', fontsize=12)
        ax3b.set_title(r'$L^\infty$ Disagreement vs Dimension', fontsize=13)
        ax3b.grid(True, alpha=0.3)
        ax3b.set_xticks(dimensions)
        
        fig3.tight_layout()
        
        if save_results:
            fig3.savefig(figures_dir / "exp3_disagreement_vs_dimension.png", 
                        dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig3)
    
    # =========================================================================
    # Save Results Table (Enhanced with Cost Metrics)
    # =========================================================================
    if save_results:
        with open(tables_dir / "exp3_dimension_scaling.md", 'w') as f:
            f.write("# Experiment 3: Dimension Scaling Results\n\n")
            f.write("**NOTE**: Disagreement metrics measure difference between MLMC and Laplace.\n")
            f.write("Neither method is ground truth.\n\n")
            
            f.write(f"**Threading disabled**: {use_single_thread}\n\n")
            
            f.write("## Wall-Clock Time Summary\n\n")
            f.write("| d | MLMC Time (s) | Laplace Time (s) | ")
            f.write("L² Disagreement | L∞ Disagreement |\n")
            f.write("|---|---------------|------------------|")
            f.write("-----------------|------------------|\n")
            for i, dim in enumerate(dimensions):
                f.write(f"| {dim} | {mlmc_wall[i]:.2f} | {laplace_wall[i]:.2f} | ")
                f.write(f"{l2_disagreements[i]:.4f} | {linf_disagreements[i]:.4f} |\n")
            
            f.write("\n## Hardware-Independent Cost Metrics\n\n")
            f.write("| d | MLMC CPU (s) | Laplace CPU (s) | ")
            f.write("MLMC Calls | Laplace Calls | MLMC Mem (MB) | Laplace Mem (MB) |\n")
            f.write("|---|--------------|-----------------|")
            f.write("-----------|---------------|---------------|------------------|\n")
            for i, dim in enumerate(dimensions):
                f.write(f"| {dim} | {mlmc_cpu[i]:.2f} | {laplace_cpu[i]:.2f} | ")
                f.write(f"{mlmc_calls[i]:,} | {laplace_calls[i]:,} | ")
                f.write(f"{mlmc_mem[i]:.1f} | {laplace_mem[i]:.1f} |\n")
            
            f.write("\n## Key Observations\n\n")
            f.write("- **Function calls**: Hardware-independent metric ")
            f.write("(same results on any computer)\n")
            f.write("- **CPU time**: More consistent than wall time ")
            f.write("(excludes sleep/IO)\n")
            f.write("- **Peak memory**: Shows memory complexity scaling\n")
            f.write("- Markovian projection reduces d-dimensional problem to 1D PDE\n")
        
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
        "mlmc_metrics": mlmc_metrics_list,
        "laplace_metrics": laplace_metrics_list,
        "l2_disagreements": l2_disagreements,
        "linf_disagreements": linf_disagreements,
        "all_results": all_results,
        # Legacy keys for backward compatibility
        "mlmc_times": [m.wall_time for m in mlmc_metrics_list],
        "laplace_times": [m.wall_time for m in laplace_metrics_list],
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
    parser.add_argument(
        "--single-thread", action="store_true",
        help="Disable BLAS parallelisation for fair comparison"
    )
    
    args = parser.parse_args()
    
    results = run_experiment(
        dimensions=args.dimensions,
        save_results=not args.no_save,
        show_plots=args.show,
        verbose=not args.quiet,
        disable_threading=args.single_thread,
    )
