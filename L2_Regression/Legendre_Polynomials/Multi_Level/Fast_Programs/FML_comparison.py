# This file is based on singlemulticomp.py from Amelie's work.
# Refactored for publication-quality validation of SL vs ML vs OT-MLMC methods.

"""
Fast Multi-Level Method Comparison (FML_comparison)
===================================================

Validation framework for comparing Single-Level, Multi-Level, and
Optimal-Transport-enhanced MLMC methods. Generates error plots
showing how well each method approximates the volatility surface.

The key metric is the RMS error between volatility surfaces:

    error² = (1/MN) Σ_{m,n} [b̄_A(t_n, S_m,n) - b̄_B(t_n, S_m,n)]²

where the comparison is performed on fresh validation paths not
used in fitting.

Functions
---------
compute_surface_error : Compute RMS error between two volatility surfaces
run_comparison : Full comparison study across polynomial degrees
"""

import numpy as np
import matplotlib.pyplot as plt

from FML_utils import (
    GBM_paths,
    scalings_l0,
    tot_degree_poly,
    make_b_bar,
    make_c
)

from FML_single_level import single_level

from FML_optimal_transport import make_c_OT


def compute_surface_error(bbar_ref, bbar_test, paths, P1, dt):
    """
    Compute RMS error between two volatility surfaces on validation paths.
    
    Evaluates both surfaces on the projected basket values from fresh
    Monte Carlo paths and computes the root mean square difference.
    
    Parameters
    ----------
    bbar_ref : callable
        Reference volatility surface (e.g., single-level).
    bbar_test : callable
        Test volatility surface (e.g., MLMC or OT-MLMC).
    paths : ndarray, shape (M, N, d)
        Validation asset price paths.
    P1 : ndarray, shape (d,)
        Basket weights.
    dt : float
        Time step size.
        
    Returns
    -------
    abs_error : float
        Absolute RMS error: sqrt(mean((b̄_ref - b̄_test)²)).
    rel_error : float
        Relative RMS error: abs_error / sqrt(mean(b̄_ref²)).
    """
    M, N, d = paths.shape
    t = np.arange(N) * dt
    proj = paths @ P1  # Projected basket values, shape (M, N)
    T = np.broadcast_to(t, (M, N))

    # Evaluate surfaces on all (time, basket) points
    ref_vals = bbar_ref(T, proj)
    test_vals = bbar_test(T, proj)

    # Compute RMS errors
    squared_diff = (ref_vals - test_vals) ** 2
    mean_sq_error = squared_diff.mean()
    abs_error = np.sqrt(mean_sq_error)

    # Relative error normalised by reference surface magnitude
    ref_magnitude = np.sqrt(np.mean(ref_vals ** 2))
    rel_error = abs_error / ref_magnitude if ref_magnitude > 0 else np.inf

    return abs_error, rel_error


def run_comparison(x0, T, h0, r, cov_mat, vol, P1, max_degs, trials=5,
                   C=80, M_val=800):
    """
    Run full comparison study across polynomial degrees and methods.
    
    For each polynomial degree, generates multiple independent realisations
    of SL, ML, and OT-ML volatility surfaces, then computes errors against
    the single-level reference on fresh validation paths.
    
    Parameters
    ----------
    x0 : ndarray
        Initial asset prices.
    T : float
        Time to maturity.
    h0 : float
        Base time step.
    r : float
        Risk-free rate.
    cov_mat : ndarray
        Correlation matrix.
    vol : ndarray
        Volatility vector.
    P1 : ndarray
        Basket weights.
    max_degs : list of int
        Polynomial degrees to test.
    trials : int, optional
        Number of independent trials per degree (default 5).
    C : int, optional
        Base sample size factor (default 80).
    M_val : int, optional
        Number of validation paths (default 800).
        
    Returns
    -------
    results : dict
        Dictionary containing error arrays:
        - 'abs_ML': Absolute errors for ML vs SL
        - 'rel_ML': Relative errors for ML vs SL
        - 'abs_OT': Absolute errors for OT-ML vs SL
        - 'rel_OT': Relative errors for OT-ML vs SL
        Each array has shape (len(max_degs), trials).
    """
    d = len(vol)

    # Get scaling parameters (use max degree = 3 as reference)
    s_mint, s_maxt = scalings_l0(x0, T, h0, r, cov_mat, vol, 3, P1, M_0=10000)
    pad = (s_maxt - s_mint) * 0.05
    s_min0 = s_mint - pad
    s_max0 = s_maxt + pad

    # Generate validation paths at finest resolution
    dt_val = h0 * 2 ** (-3)  # Use degree 3 resolution
    N_val = int(round(T / dt_val))
    val_paths = GBM_paths(x0, r, vol, cov_mat, dt_val, N_val, M_val)

    # Storage for results
    abs_ML = np.zeros((len(max_degs), trials))
    rel_ML = np.zeros((len(max_degs), trials))
    abs_OT = np.zeros((len(max_degs), trials))
    rel_OT = np.zeros((len(max_degs), trials))

    for j, max_deg in enumerate(max_degs):
        print(f"\n{'='*60}")
        print(f"Polynomial degree {max_deg}")
        print(f"{'='*60}")

        pairs = tot_degree_poly(max_deg)

        for k in range(trials):
            print(f"\n  Trial {k+1}/{trials}")

            # Single-Level (reference)
            print("    Computing Single-Level...")
            c_SL = single_level(x0, T, h0, r, cov_mat, vol, max_deg, P1,
                               s_min0, s_max0, C=C, batch_size=50)
            bbar_SL = make_b_bar(c_SL, pairs, s_min0, s_max0, T, max_deg)

            # Multi-Level (standard)
            print("    Computing Multi-Level...")
            c_ML = make_c(x0, T, h0, r, cov_mat, vol, max_deg, P1,
                         s_min0, s_max0, C=C, batch_size=50)
            bbar_ML = make_b_bar(c_ML, pairs, s_min0, s_max0, T, max_deg)

            # Multi-Level with Optimal Transport
            print("    Computing OT-Multi-Level...")
            c_OT = make_c_OT(x0, T, h0, r, cov_mat, vol, max_deg, P1,
                            s_min0, s_max0, C=C, batch_size_ML=50, batch_size_maps=10)
            bbar_OT = make_b_bar(c_OT, pairs, s_min0, s_max0, T, max_deg)

            # Compute errors
            abs_err_ML, rel_err_ML = compute_surface_error(bbar_SL, bbar_ML,
                                                           val_paths, P1, dt_val)
            abs_err_OT, rel_err_OT = compute_surface_error(bbar_SL, bbar_OT,
                                                           val_paths, P1, dt_val)

            abs_ML[j, k] = abs_err_ML
            rel_ML[j, k] = rel_err_ML
            abs_OT[j, k] = abs_err_OT
            rel_OT[j, k] = rel_err_OT

            print(f"    ML error: abs={abs_err_ML:.4e}, rel={rel_err_ML:.4e}")
            print(f"    OT error: abs={abs_err_OT:.4e}, rel={rel_err_OT:.4e}")

    return {
        'max_degs': max_degs,
        'abs_ML': abs_ML,
        'rel_ML': rel_ML,
        'abs_OT': abs_OT,
        'rel_OT': rel_OT
    }


def plot_comparison(results, save_path="plots/SingleVsMultiError.pdf"):
    """
    Generate comparison plots from run_comparison results.
    
    Parameters
    ----------
    results : dict
        Output from run_comparison().
    save_path : str, optional
        Path to save the figure.
    """
    max_degs = results['max_degs']
    abs_ML = results['abs_ML']
    rel_ML = results['rel_ML']
    abs_OT = results['abs_OT']
    rel_OT = results['rel_OT']
    trials = abs_ML.shape[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute Error
    ax1.errorbar(max_degs, abs_ML.mean(axis=1),
                yerr=abs_ML.std(axis=1, ddof=1),
                fmt='-d', capsize=4, color='blue',
                markerfacecolor='blue', label='ML')
    for i, deg in enumerate(max_degs):
        ax1.scatter([deg] * trials, abs_ML[i, :], marker='o',
                   color='blue', alpha=0.5, s=20)

    ax1.errorbar(max_degs, abs_OT.mean(axis=1),
                yerr=abs_OT.std(axis=1, ddof=1),
                fmt='-d', capsize=4, color='green',
                markerfacecolor='green', label='ML + Optimal Transport')
    for i, deg in enumerate(max_degs):
        ax1.scatter([deg] * trials, abs_OT[i, :], marker='o',
                   color='green', alpha=0.5, s=20)

    ax1.set_xlabel("Maximum total degree L")
    ax1.set_ylabel("Absolute RMS error")
    ax1.set_title("Absolute RMS Error: Single-Level vs Multi-Level")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Relative Error
    ax2.errorbar(max_degs, rel_ML.mean(axis=1),
                yerr=rel_ML.std(axis=1, ddof=1),
                fmt='-d', capsize=4, color='blue',
                markerfacecolor='blue', label='ML')
    for i, deg in enumerate(max_degs):
        ax2.scatter([deg] * trials, rel_ML[i, :], marker='o',
                   color='blue', alpha=0.5, s=20)

    ax2.errorbar(max_degs, rel_OT.mean(axis=1),
                yerr=rel_OT.std(axis=1, ddof=1),
                fmt='-d', capsize=4, color='green',
                markerfacecolor='green', label='ML + Optimal Transport')
    for i, deg in enumerate(max_degs):
        ax2.scatter([deg] * trials, rel_OT[i, :], marker='o',
                   color='green', alpha=0.5, s=20)

    ax2.set_xlabel("Maximum total degree L")
    ax2.set_ylabel("Relative RMS error")
    ax2.set_title("Relative RMS Error: Single-Level vs Multi-Level")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved: {save_path}")


# =============================================================================
# Main Block
# =============================================================================

if __name__ == "__main__":
    # Basket parameters
    d = 3
    P1 = np.ones(d) / d
    r = 0.05
    x0 = np.linspace(225, 275, num=d)[:, np.newaxis]
    vol = np.array([0.2, 0.15, 0.1])
    cov_mat = np.array([[1.0, 0.8, 0.3],
                        [0.8, 1.0, 0.1],
                        [0.3, 0.1, 1.0]])
    T = 1.0
    h0 = 0.01

    # Run comparison
    max_degs = [1, 2, 3]
    trials = 5

    print("Running Method Comparison Study")
    print("================================")
    print(f"Polynomial degrees: {max_degs}")
    print(f"Trials per degree: {trials}")

    results = run_comparison(x0, T, h0, r, cov_mat, vol, P1,
                            max_degs=max_degs, trials=trials)

    # Generate plots
    plot_comparison(results)
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    for i, deg in enumerate(max_degs):
        print(f"\nDegree {deg}:")
        print(f"  ML  - Abs: {results['abs_ML'][i].mean():.4e} ± {results['abs_ML'][i].std():.4e}")
        print(f"  ML  - Rel: {results['rel_ML'][i].mean():.4e} ± {results['rel_ML'][i].std():.4e}")
        print(f"  OT  - Abs: {results['abs_OT'][i].mean():.4e} ± {results['abs_OT'][i].std():.4e}")
        print(f"  OT  - Rel: {results['rel_OT'][i].mean():.4e} ± {results['rel_OT'][i].std():.4e}")
