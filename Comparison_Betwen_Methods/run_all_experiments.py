#!/usr/bin/env python3
"""
Master Script: Run All Comparison Framework Experiments

This script runs all experiments comparing MLMC and Laplace approximation
methods for American basket option volatility estimation.

Experiments
-----------
1. Surface Comparison: Direct b²(t,s) comparison
2. MLMC Convergence: Convergence study with multiple runs
3. Dimension Scaling: Performance vs number of assets
4. Option Pricing: End-to-end PDE comparison
5. Parameter Sensitivity: Vary σ, ρ, K, T, r

Usage
-----
    python run_all_experiments.py           # Run all experiments
    python run_all_experiments.py --exp 1 2 # Run specific experiments
    python run_all_experiments.py --quick   # Quick mode (fewer runs)

Author: Wadoud (KAUST Internship)
"""

import sys
import time
import argparse
from pathlib import Path

# Setup paths
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from config import DEFAULT_PARAMS


def run_experiment_1(params=None, verbose=True):
    """Run Experiment 1: Surface Comparison"""
    from experiments.exp1_surface_comparison import run_experiment
    return run_experiment(params, save_results=True, show_plots=False, verbose=verbose)


def run_experiment_2(params=None, n_runs=20, verbose=True):
    """Run Experiment 2: MLMC Convergence"""
    from experiments.exp2_mlmc_convergence import run_experiment
    return run_experiment(params, n_runs=n_runs, save_results=True, show_plots=False, verbose=verbose)


def run_experiment_3(dimensions=None, verbose=True):
    """Run Experiment 3: Dimension Scaling"""
    from experiments.exp3_dimension_scaling import run_experiment
    return run_experiment(dimensions=dimensions, save_results=True, show_plots=False, verbose=verbose)


def run_experiment_4(params=None, verbose=True):
    """Run Experiment 4: Option Pricing"""
    from experiments.exp4_option_pricing import run_experiment
    return run_experiment(params, save_results=True, show_plots=False, verbose=verbose)


def run_experiment_5(verbose=True):
    """Run Experiment 5: Parameter Sensitivity"""
    from experiments.exp5_parameter_sensitivity import run_experiment
    return run_experiment(save_results=True, show_plots=False, verbose=verbose)


def run_all(
    experiments: list = None,
    quick_mode: bool = False,
    verbose: bool = True
):
    """
    Run all (or selected) experiments.
    
    Parameters
    ----------
    experiments : list of int, optional
        Experiment numbers to run. Default: all available.
    quick_mode : bool
        Use reduced parameters for faster execution.
    verbose : bool
        Print progress information.
        
    Returns
    -------
    dict
        Results from all experiments.
    """
    available_experiments = {
        1: ("Surface Comparison", run_experiment_1),
        2: ("MLMC Convergence", run_experiment_2),
        3: ("Dimension Scaling", run_experiment_3),
        4: ("Option Pricing", run_experiment_4),
        5: ("Parameter Sensitivity", run_experiment_5),
    }
    
    if experiments is None:
        experiments = list(available_experiments.keys())
    
    # Validate experiment numbers
    invalid = [exp for exp in experiments if exp not in available_experiments]
    for exp in invalid:
        print(f"Warning: Experiment {exp} not implemented. Skipping.")
        experiments = [e for e in experiments if e != exp]
    
    if verbose:
        print("=" * 70)
        print("COMPARISON FRAMEWORK: Running Experiments")
        print("=" * 70)
        print()
        print("Experiments to run:")
        for exp in experiments:
            print(f"  {exp}. {available_experiments[exp][0]}")
        print()
        if quick_mode:
            print("Quick mode: Using reduced parameters")
            print()
    
    # Setup parameters
    params = DEFAULT_PARAMS.copy()
    if quick_mode:
        params.max_degree = 2  # Reduce MLMC levels
        params.N_t = 30
        params.N_s = 50
    
    results = {}
    total_start = time.time()
    
    for exp_num in experiments:
        name, runner = available_experiments[exp_num]
        
        if verbose:
            print()
            print("#" * 70)
            print(f"# EXPERIMENT {exp_num}: {name}")
            print("#" * 70)
            print()
        
        exp_start = time.time()
        
        try:
            if exp_num == 1:
                results[exp_num] = runner(params, verbose=verbose)
            elif exp_num == 2:
                # Convergence study has special parameters
                n_runs = 5 if quick_mode else 20
                results[exp_num] = runner(params, n_runs=n_runs, verbose=verbose)
            elif exp_num == 3:
                # Dimension scaling - use fewer dimensions in quick mode
                dimensions = [2, 3] if quick_mode else [2, 3, 5, 10]
                results[exp_num] = runner(dimensions=dimensions, verbose=verbose)
            elif exp_num == 4:
                results[exp_num] = runner(params, verbose=verbose)
            elif exp_num == 5:
                results[exp_num] = runner(verbose=verbose)
            else:
                results[exp_num] = runner(verbose=verbose)
            
            exp_time = time.time() - exp_start
            
            if verbose:
                print(f"\nExperiment {exp_num} completed in {exp_time:.1f}s")
            
        except Exception as e:
            print(f"\nError in Experiment {exp_num}: {e}")
            import traceback
            traceback.print_exc()
            results[exp_num] = {"error": str(e)}
    
    total_time = time.time() - total_start
    
    if verbose:
        print()
        print("=" * 70)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 70)
        print(f"\nTotal time: {total_time:.1f}s")
        print(f"\nResults saved to: {_script_dir / 'results'}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Comparison Framework Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_experiments.py           # Run all experiments
  python run_all_experiments.py --exp 1   # Run only experiment 1
  python run_all_experiments.py --quick   # Quick mode (fewer runs)
        """
    )
    
    parser.add_argument(
        "--exp", type=int, nargs="+",
        help="Experiment numbers to run (default: all)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode with reduced parameters"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    results = run_all(
        experiments=args.exp,
        quick_mode=args.quick,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    main()
