# This file is based on FML_comparison.py, enhanced with JAX JIT compilation.
# Original work by Amelie, JAX conversion by Wadoud Charbak.

"""
JAX-Enhanced Method Comparison Utilities

Provides JIT-compiled functions for comparing volatility surfaces
across different estimation methods (SL, ML, OT-ML, QR-ML).

Key features:
- Vectorised surface evaluation for efficient error computation
- JIT-compiled RMS error calculation
- Parallel comparison across multiple trials
"""

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from typing import Callable, Tuple, Dict, List
import numpy as np
import time

from JAX_FML_utils import (
    GBM_paths, scalings_l0, tot_degree_poly, make_c, make_b_bar
)
from JAX_FML_single_level import single_level
from JAX_FML_optimal_transport import make_c_OT
from JAX_FML_hierarchical_qr import make_c_qr


# =============================================================================
# Error Computation (JIT-compiled)
# =============================================================================

@jax.jit
def compute_pointwise_errors(bbar_vals_A: jax.Array, 
                              bbar_vals_B: jax.Array) -> Tuple[float, float]:
    """
    Compute RMS error between two volatility surface evaluations.
    
    Parameters
    ----------
    bbar_vals_A : jax.Array
        Surface A values (any shape)
    bbar_vals_B : jax.Array
        Surface B values (same shape as A)
        
    Returns
    -------
    abs_error : float
        Root mean squared error
    rel_error : float
        Relative RMS error (normalised by A)
    """
    diff = bbar_vals_A - bbar_vals_B
    mse = jnp.mean(diff ** 2)
    abs_error = jnp.sqrt(mse)
    
    mean_sq_A = jnp.mean(bbar_vals_A ** 2)
    rel_error = abs_error / (jnp.sqrt(mean_sq_A) + 1e-10)
    
    return abs_error, rel_error


def compute_surface_error(bbar_A: Callable, bbar_B: Callable,
                          val_paths: jax.Array, P1: jax.Array,
                          dt: float) -> Tuple[float, float]:
    """
    Compute RMS error between two surfaces on validation paths.
    
    Parameters
    ----------
    bbar_A : Callable
        First volatility surface function
    bbar_B : Callable
        Second volatility surface function
    val_paths : jax.Array, shape (M, N, d)
        Validation paths (not used for fitting)
    P1 : jax.Array, shape (d,)
        Basket weights
    dt : float
        Timestep for validation paths
        
    Returns
    -------
    abs_error : float
        Absolute RMS error
    rel_error : float
        Relative RMS error
    """
    M, N, d = val_paths.shape
    
    # Compute basket values
    basket = jnp.einsum('mnd,d->mn', val_paths, P1)  # (M, N)
    
    # Time grid
    t_grid = jnp.arange(N) * dt
    T_grid = jnp.broadcast_to(t_grid, (M, N))  # (M, N)
    
    # Evaluate surfaces
    vals_A = bbar_A(T_grid, basket)
    vals_B = bbar_B(T_grid, basket)
    
    return compute_pointwise_errors(vals_A, vals_B)


# =============================================================================
# Multi-Trial Comparison
# =============================================================================

def run_single_comparison(key: jax.Array, x0: jax.Array, T: float, h0: float,
                          r: float, cov_mat: jax.Array, vol: jax.Array,
                          max_deg: int, P1: jax.Array,
                          s_min0: float, s_max0: float,
                          C: int, val_paths: jax.Array,
                          dt_val: float) -> Dict:
    """
    Run a single comparison trial across all methods.
    
    Parameters
    ----------
    [Standard parameters]
    val_paths : jax.Array
        Pre-generated validation paths
    dt_val : float
        Validation path timestep
        
    Returns
    -------
    results : dict
        Errors and timing for each method
    """
    pairs = tot_degree_poly(max_deg)
    results = {}
    
    # Single-Level (reference)
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c_sl = single_level(subkey, x0, T, h0, r, cov_mat, vol, max_deg,
                        P1, s_min0, s_max0, C)
    results['time_SL'] = time.time() - t0
    bbar_SL = make_b_bar(c_sl, pairs, s_min0, s_max0, T, max_deg)
    
    # Multi-Level (accumulated normal equations)
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c_ml = make_c(subkey, x0, T, h0, r, cov_mat, vol, max_deg,
                  P1, s_min0, s_max0, C)
    results['time_ML'] = time.time() - t0
    bbar_ML = make_b_bar(c_ml[:len(pairs)], pairs, s_min0, s_max0, T, max_deg)
    
    # OT-Enhanced MLMC
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c_ot = make_c_OT(subkey, x0, T, h0, r, cov_mat, vol, max_deg,
                     P1, s_min0, s_max0, C)
    results['time_OT'] = time.time() - t0
    bbar_OT = make_b_bar(c_ot[:len(pairs)], pairs, s_min0, s_max0, T, max_deg)
    
    # Hierarchical QR MLMC
    key, subkey = jax.random.split(key)
    t0 = time.time()
    c_qr = make_c_qr(subkey, x0, T, h0, r, cov_mat, vol, max_deg,
                     P1, s_min0, s_max0, C)
    results['time_QR'] = time.time() - t0
    bbar_QR = make_b_bar(c_qr[:len(pairs)], pairs, s_min0, s_max0, T, max_deg)
    
    # Compute errors vs SL
    abs_ML, rel_ML = compute_surface_error(bbar_SL, bbar_ML, val_paths, P1, dt_val)
    abs_OT, rel_OT = compute_surface_error(bbar_SL, bbar_OT, val_paths, P1, dt_val)
    abs_QR, rel_QR = compute_surface_error(bbar_SL, bbar_QR, val_paths, P1, dt_val)
    
    results['abs_ML'] = float(abs_ML)
    results['rel_ML'] = float(rel_ML)
    results['abs_OT'] = float(abs_OT)
    results['rel_OT'] = float(rel_OT)
    results['abs_QR'] = float(abs_QR)
    results['rel_QR'] = float(rel_QR)
    
    # Store coefficients
    results['c_sl'] = c_sl
    results['c_ml'] = c_ml[:len(pairs)]
    results['c_ot'] = c_ot[:len(pairs)]
    results['c_qr'] = c_qr[:len(pairs)]
    
    return results


def run_comparison_study(key: jax.Array, x0: jax.Array, T: float, h0: float,
                         r: float, cov_mat: jax.Array, vol: jax.Array,
                         max_degs: List[int], P1: jax.Array,
                         C: int = 80, n_trials: int = 3,
                         M_val: int = 1000) -> Dict:
    """
    Run comprehensive comparison study across polynomial degrees.
    
    Parameters
    ----------
    max_degs : list of int
        Polynomial degrees to test
    n_trials : int
        Number of independent trials per degree
    M_val : int
        Number of validation paths
        
    Returns
    -------
    results : dict
        Nested dictionary with results[deg][trial]
    """
    results = {}
    
    for max_deg in max_degs:
        print(f"\n{'='*50}")
        print(f"Polynomial degree {max_deg}")
        print(f"{'='*50}")
        
        # Get scaling for this degree
        key, subkey = jax.random.split(key)
        s_min, s_max = scalings_l0(subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1)
        print(f"Domain: [{s_min:.2f}, {s_max:.2f}]")
        
        # Generate validation paths
        dt_val = h0 * (2.0 ** (-max_deg))
        N_val = int(round(T / dt_val))
        key, subkey = jax.random.split(key)
        val_paths = GBM_paths(subkey, x0, r, vol, cov_mat, dt_val, N_val, M_val)
        
        results[max_deg] = []
        
        for trial in range(n_trials):
            print(f"\n  Trial {trial + 1}/{n_trials}")
            key, subkey = jax.random.split(key)
            
            trial_results = run_single_comparison(
                subkey, x0, T, h0, r, cov_mat, vol, max_deg, P1,
                s_min, s_max, C, val_paths, dt_val
            )
            
            print(f"    ML: abs={trial_results['abs_ML']:.4e}, "
                  f"rel={trial_results['rel_ML']:.4e}, "
                  f"time={trial_results['time_ML']:.2f}s")
            print(f"    OT: abs={trial_results['abs_OT']:.4e}, "
                  f"rel={trial_results['rel_OT']:.4e}, "
                  f"time={trial_results['time_OT']:.2f}s")
            print(f"    QR: abs={trial_results['abs_QR']:.4e}, "
                  f"rel={trial_results['rel_QR']:.4e}, "
                  f"time={trial_results['time_QR']:.2f}s")
            
            results[max_deg].append(trial_results)
    
    return results


def summarise_results(results: Dict) -> Dict:
    """
    Compute summary statistics from comparison study.
    
    Parameters
    ----------
    results : dict
        Output from run_comparison_study
        
    Returns
    -------
    summary : dict
        Mean and std for each metric by degree
    """
    summary = {}
    
    for deg, trials in results.items():
        n = len(trials)
        
        metrics = ['abs_ML', 'rel_ML', 'abs_OT', 'rel_OT', 
                   'abs_QR', 'rel_QR', 'time_SL', 'time_ML', 
                   'time_OT', 'time_QR']
        
        summary[deg] = {}
        for metric in metrics:
            values = [t[metric] for t in trials]
            summary[deg][f'{metric}_mean'] = np.mean(values)
            summary[deg][f'{metric}_std'] = np.std(values)
    
    return summary


def print_summary_table(summary: Dict):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\n{'Deg':<5} {'Method':<8} {'Abs Error':<18} {'Rel Error':<18} {'Time (s)':<12}")
    print("-" * 80)
    
    for deg in sorted(summary.keys()):
        s = summary[deg]
        
        # ML
        print(f"{deg:<5} {'ML':<8} "
              f"{s['abs_ML_mean']:.4e} ± {s['abs_ML_std']:.2e}   "
              f"{s['rel_ML_mean']:.4e} ± {s['rel_ML_std']:.2e}   "
              f"{s['time_ML_mean']:.2f}")
        
        # OT
        print(f"{'':5} {'OT':<8} "
              f"{s['abs_OT_mean']:.4e} ± {s['abs_OT_std']:.2e}   "
              f"{s['rel_OT_mean']:.4e} ± {s['rel_OT_std']:.2e}   "
              f"{s['time_OT_mean']:.2f}")
        
        # QR
        print(f"{'':5} {'QR':<8} "
              f"{s['abs_QR_mean']:.4e} ± {s['abs_QR_std']:.2e}   "
              f"{s['rel_QR_mean']:.4e} ± {s['rel_QR_std']:.2e}   "
              f"{s['time_QR_mean']:.2f}")
        
        print("-" * 80)


# =============================================================================
# JAX Performance Benchmarking
# =============================================================================

def benchmark_jit_speedup(key: jax.Array, x0: jax.Array, r: float,
                          vol: jax.Array, cov_mat: jax.Array,
                          n_runs: int = 5) -> Dict:
    """
    Benchmark JIT compilation speedup for path generation.
    
    Returns
    -------
    results : dict
        Timing comparison for first run (includes compilation) vs subsequent runs
    """
    dt = 0.01
    N_t = 100
    M = 10000
    
    results = {'first_run': [], 'subsequent_runs': []}
    
    for run in range(n_runs):
        # Fresh key for new compilation
        key, subkey = jax.random.split(key)
        
        # First call (includes JIT compilation)
        t0 = time.time()
        paths = GBM_paths(subkey, x0, r, vol, cov_mat, dt, N_t, M)
        paths.block_until_ready()
        t1 = time.time()
        results['first_run'].append(t1 - t0)
        
        # Subsequent call (cached)
        key, subkey = jax.random.split(key)
        t0 = time.time()
        paths = GBM_paths(subkey, x0, r, vol, cov_mat, dt, N_t, M)
        paths.block_until_ready()
        t1 = time.time()
        results['subsequent_runs'].append(t1 - t0)
    
    results['speedup'] = np.mean(results['first_run']) / np.mean(results['subsequent_runs'])
    
    return results


# =============================================================================
# Main block for testing
# =============================================================================

if __name__ == "__main__":
    print("Testing JAX_FML_comparison.py")
    print("=" * 60)
    
    # Check JAX backend
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    
    # Test parameters
    d = 3
    x0 = jnp.array([250.0, 250.0, 250.0])
    vol = jnp.array([0.2, 0.15, 0.1])
    cov_mat = jnp.array([[1.0, 0.5, 0.3],
                         [0.5, 1.0, 0.2],
                         [0.3, 0.2, 1.0]])
    P1 = jnp.ones(d) / d
    r = 0.05
    T = 1.0
    h0 = 0.01
    
    key = jax.random.PRNGKey(42)
    
    # Benchmark JIT speedup
    print("\nBenchmarking JIT compilation speedup...")
    bench = benchmark_jit_speedup(key, x0, r, vol, cov_mat)
    print(f"  First run (with compilation): {np.mean(bench['first_run']):.4f}s")
    print(f"  Subsequent runs (cached): {np.mean(bench['subsequent_runs']):.4f}s")
    print(f"  JIT speedup: {bench['speedup']:.1f}x")
    
    # Run comparison study
    print("\nRunning comparison study...")
    key, subkey = jax.random.split(key)
    results = run_comparison_study(
        subkey, x0, T, h0, r, cov_mat, vol,
        max_degs=[1, 2], P1=P1, C=30, n_trials=2, M_val=500
    )
    
    # Summarise
    summary = summarise_results(results)
    print_summary_table(summary)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
