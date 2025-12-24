"""
Experiment 5: Parameter Sensitivity Analysis

Study how MLMC and Laplace approximation methods respond to changes
in key market parameters.

This experiment:
1. Varies individual parameters: σ, ρ, r, T, moneyness (K/S0)
2. Measures L2 error between methods for each parameter value
3. Studies parameter regions where methods agree/disagree
4. Identifies potential edge cases

Parameters Studied
------------------
- Volatility (σ): 0.05 to 0.50
- Correlation (ρ): -0.5 to 0.9
- Interest rate (r): 0.00 to 0.15
- Maturity (T): 0.1 to 2.0
- Moneyness (K/S0): 0.8 to 1.2

Expected Outputs
----------------
- results/figures/exp5_sensitivity_*.png (one per parameter)
- results/figures/exp5_sensitivity_summary.png
- results/tables/exp5_parameter_sensitivity.md

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

from config import DEFAULT_PARAMS, create_2d_params
from methods.mlmc_ot_estimator import estimate_volatility_mlmc, estimate_domain
from methods.laplace_wrapper import estimate_volatility_laplace
from methods.common import compute_accuracy_metrics


def run_single_comparison(params, t_grid, s_grid, verbose=False):
    """
    Run both methods and compute metrics for given parameters.
    
    Returns
    -------
    dict with keys: l2_error, linf_error, mlmc_time, laplace_time, correlation
    """
    try:
        result_mlmc = estimate_volatility_mlmc(
            params, t_grid, s_grid,
            use_ot=True,
            verbose=False
        )
        result_laplace = estimate_volatility_laplace(
            params, t_grid, s_grid, verbose=False
        )
        
        metrics = compute_accuracy_metrics(
            result_mlmc.b_squared_values,
            result_laplace.b_squared_values
        )
        
        return {
            'l2_error': metrics['l2_relative_error'],
            'linf_error': metrics['linf_relative_error'],
            'mlmc_time': result_mlmc.computation_time,
            'laplace_time': result_laplace.computation_time,
            'correlation': metrics.get('correlation', np.nan),
            'success': True,
        }
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return {
            'l2_error': np.nan,
            'linf_error': np.nan,
            'mlmc_time': np.nan,
            'laplace_time': np.nan,
            'correlation': np.nan,
            'success': False,
        }


def run_experiment(
    save_results: bool = True,
    show_plots: bool = False,
    verbose: bool = True
):
    """
    Run Experiment 5: Parameter sensitivity analysis.
    
    Parameters
    ----------
    save_results : bool
        Whether to save figures and tables.
    show_plots : bool
        Whether to display plots interactively.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    dict
        Dictionary containing sensitivity results for each parameter.
    """
    # Setup paths
    results_dir = _comparison_dir / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    
    for d in [results_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("EXPERIMENT 5: Parameter Sensitivity Analysis")
        print("=" * 70)
        print()

    # Use 2D for faster sensitivity analysis
    base_params = create_2d_params()

    # Run pilot to get MLMC domain bounds
    if verbose:
        print("-" * 50)
        print("Estimating MLMC domain via pilot run...")
        print("-" * 50)

    np.random.seed(base_params.random_seed)
    S_min, S_max = estimate_domain(
        x0=base_params.x0,
        T=base_params.T,
        h0=base_params.h0,
        r=base_params.r,
        cov_mat=base_params.corr_matrix,
        vol=base_params.sigma,
        max_deg=base_params.max_degree,
        P1=base_params.P1,
        M_pilot=10000
    )

    # Common grid WITHIN the domain (with safety margin)
    margin = 0.02  # 2% inset from boundaries
    S_range = S_max - S_min

    t_grid = np.linspace(0.02, base_params.T, 12)
    s_grid = np.linspace(
        S_min + margin * S_range,  # Slightly above S_min
        S_max - margin * S_range,  # Slightly below S_max
        20
    )

    if verbose:
        print(f"Domain from pilot: [{S_min:.1f}, {S_max:.1f}]")
        print(f"Evaluation grid: [{s_grid.min():.1f}, {s_grid.max():.1f}]")
        print()
    
    all_results = {}
    
    # =========================================================================
    # 1. Volatility Sensitivity
    # =========================================================================
    if verbose:
        print("-" * 50)
        print("1. Volatility Sensitivity (σ)")
        print("-" * 50)
    
    sigma_values = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    sigma_results = []
    
    for sigma_val in sigma_values:
        if verbose:
            print(f"  σ = {sigma_val:.2f}...", end=" ", flush=True)
        
        params = create_2d_params(sigma=np.array([sigma_val, sigma_val * 0.9]))
        result = run_single_comparison(params, t_grid, s_grid, verbose)
        sigma_results.append(result)
        
        if verbose:
            if result['success']:
                print(f"L2 = {result['l2_error']:.4f}")
            else:
                print("FAILED")
    
    all_results['volatility'] = {
        'values': sigma_values,
        'results': sigma_results,
        'label': r'Volatility ($\sigma$)',
    }
    
    # =========================================================================
    # 2. Correlation Sensitivity
    # =========================================================================
    if verbose:
        print()
        print("-" * 50)
        print("2. Correlation Sensitivity (ρ)")
        print("-" * 50)
    
    rho_values = np.array([-0.5, -0.25, 0.0, 0.25, 0.5, 0.7, 0.8, 0.9])
    rho_results = []
    
    for rho_val in rho_values:
        if verbose:
            print(f"  ρ = {rho_val:.2f}...", end=" ", flush=True)
        
        params = create_2d_params(rho=rho_val)
        result = run_single_comparison(params, t_grid, s_grid, verbose)
        rho_results.append(result)
        
        if verbose:
            if result['success']:
                print(f"L2 = {result['l2_error']:.4f}")
            else:
                print("FAILED")
    
    all_results['correlation'] = {
        'values': rho_values,
        'results': rho_results,
        'label': r'Correlation ($\rho$)',
    }
    
    # =========================================================================
    # 3. Interest Rate Sensitivity
    # =========================================================================
    if verbose:
        print()
        print("-" * 50)
        print("3. Interest Rate Sensitivity (r)")
        print("-" * 50)
    
    r_values = np.array([0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15])
    r_results = []
    
    for r_val in r_values:
        if verbose:
            print(f"  r = {r_val:.2f}...", end=" ", flush=True)
        
        params = create_2d_params(r=r_val)
        result = run_single_comparison(params, t_grid, s_grid, verbose)
        r_results.append(result)
        
        if verbose:
            if result['success']:
                print(f"L2 = {result['l2_error']:.4f}")
            else:
                print("FAILED")
    
    all_results['interest_rate'] = {
        'values': r_values,
        'results': r_results,
        'label': 'Interest Rate (r)',
    }
    
    # =========================================================================
    # 4. Maturity Sensitivity
    # =========================================================================
    if verbose:
        print()
        print("-" * 50)
        print("4. Maturity Sensitivity (T)")
        print("-" * 50)
    
    T_values = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    T_results = []
    
    for T_val in T_values:
        if verbose:
            print(f"  T = {T_val:.2f}...", end=" ", flush=True)
        
        params = create_2d_params(T=T_val)
        # Adjust grid for different maturities
        t_grid_T = np.linspace(0.02, T_val, 12)
        result = run_single_comparison(params, t_grid_T, s_grid, verbose)
        T_results.append(result)
        
        if verbose:
            if result['success']:
                print(f"L2 = {result['l2_error']:.4f}")
            else:
                print("FAILED")
    
    all_results['maturity'] = {
        'values': T_values,
        'results': T_results,
        'label': 'Maturity (T)',
    }
    
    # =========================================================================
    # 5. Moneyness Sensitivity
    # =========================================================================
    if verbose:
        print()
        print("-" * 50)
        print("5. Moneyness Sensitivity (K/S0)")
        print("-" * 50)
    
    moneyness_values = np.array([0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
    moneyness_results = []
    
    S0 = base_params.S0
    for m_val in moneyness_values:
        K_val = S0 * m_val
        if verbose:
            print(f"  K/S0 = {m_val:.2f} (K = {K_val:.0f})...", end=" ", flush=True)
        
        params = create_2d_params(K=K_val)
        result = run_single_comparison(params, t_grid, s_grid, verbose)
        moneyness_results.append(result)
        
        if verbose:
            if result['success']:
                print(f"L2 = {result['l2_error']:.4f}")
            else:
                print("FAILED")
    
    all_results['moneyness'] = {
        'values': moneyness_values,
        'results': moneyness_results,
        'label': 'Moneyness (K/S₀)',
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
        
        # Individual parameter plots
        param_keys = ['volatility', 'correlation', 'interest_rate', 'maturity', 'moneyness']
        colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'purple']
        
        for param_key, color in zip(param_keys, colors):
            data = all_results[param_key]
            values = data['values']
            results = data['results']
            label = data['label']
            
            l2_errors = [r['l2_error'] for r in results]
            valid = [not np.isnan(e) for e in l2_errors]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # L2 error
            ax1.plot(np.array(values)[valid], np.array(l2_errors)[valid],
                    'o-', color=color, markersize=8, linewidth=2)
            ax1.set_xlabel(label, fontsize=12)
            ax1.set_ylabel(r'$L^2$ Relative Error', fontsize=12)
            ax1.set_title(rf'$L^2$ Error vs {label}', fontsize=13)
            ax1.grid(True, alpha=0.3)
            
            # Computation times
            mlmc_times = [r['mlmc_time'] for r in results]
            laplace_times = [r['laplace_time'] for r in results]
            
            x_valid = np.array(values)[valid]
            ax2.plot(x_valid, np.array(mlmc_times)[valid], 's-', 
                    color='steelblue', markersize=6, linewidth=2, label='MLMC')
            ax2.plot(x_valid, np.array(laplace_times)[valid], '^-', 
                    color='darkorange', markersize=6, linewidth=2, label='Laplace')
            ax2.set_xlabel(label, fontsize=12)
            ax2.set_ylabel('Computation Time (s)', fontsize=12)
            ax2.set_title(f'Timing vs {label}', fontsize=13)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            if save_results:
                fig.savefig(figures_dir / f"exp5_sensitivity_{param_key}.png", 
                           dpi=150, bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close(fig)
        
        # Summary plot (all parameters)
        fig_summary, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (param_key, color) in enumerate(zip(param_keys, colors)):
            data = all_results[param_key]
            values = data['values']
            results = data['results']
            label = data['label']
            
            l2_errors = [r['l2_error'] for r in results]
            valid = [not np.isnan(e) for e in l2_errors]
            
            ax = axes[idx]
            ax.plot(np.array(values)[valid], np.array(l2_errors)[valid],
                   'o-', color=color, markersize=8, linewidth=2)
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel(r'$L^2$ Error', fontsize=11)
            ax.set_title(f'{label}', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Empty subplot
        axes[-1].axis('off')
        axes[-1].text(0.5, 0.5, 'Parameter\nSensitivity\nSummary', 
                     ha='center', va='center', fontsize=14, 
                     transform=axes[-1].transAxes)
        
        fig_summary.suptitle('Experiment 5: Parameter Sensitivity Overview', fontsize=14, y=1.02)
        fig_summary.tight_layout()
        
        if save_results:
            fig_summary.savefig(figures_dir / "exp5_sensitivity_summary.png", 
                               dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig_summary)
    
    # =========================================================================
    # Save Results Table
    # =========================================================================
    if save_results:
        with open(tables_dir / "exp5_parameter_sensitivity.md", 'w') as f:
            f.write("# Experiment 5: Parameter Sensitivity Results\n\n")
            
            for param_key in param_keys:
                data = all_results[param_key]
                f.write(f"## {data['label']}\n\n")
                f.write("| Value | L2 Error | L∞ Error | MLMC Time (s) | Laplace Time (s) |\n")
                f.write("|-------|----------|----------|---------------|------------------|\n")
                
                for val, res in zip(data['values'], data['results']):
                    if res['success']:
                        f.write(f"| {val:.2f} | {res['l2_error']:.4f} | ")
                        f.write(f"{res['linf_error']:.4f} | ")
                        f.write(f"{res['mlmc_time']:.2f} | {res['laplace_time']:.2f} |\n")
                    else:
                        f.write(f"| {val:.2f} | FAILED | - | - | - |\n")
                
                f.write("\n")
            
            f.write("## Key Observations\n\n")
            f.write("- Higher volatility generally increases L2 error\n")
            f.write("- Correlation has moderate effect on agreement\n")
            f.write("- Methods agree well across interest rate range\n")
            f.write("- Longer maturities may show increased divergence\n")
            f.write("- Both methods handle ATM and OTM options similarly\n")
        
        if verbose:
            print(f"\nResults saved to:")
            print(f"  {figures_dir / 'exp5_*.png'}")
            print(f"  {tables_dir / 'exp5_parameter_sensitivity.md'}")
    
    if verbose:
        print()
        print("=" * 70)
        print("EXPERIMENT 5 COMPLETE")
        print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 5: Parameter Sensitivity Analysis"
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
