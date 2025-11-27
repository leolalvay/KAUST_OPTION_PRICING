"""
This file is based on DM_SL_WE.py from Amelie's work.

Weak error analysis: quantifies option pricing accuracy as a function of
sample size M and polynomial degree. Demonstrates Monte Carlo convergence
rate (M^(-1/2)) and regression error trade-offs.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from SL_legendre_utilities import (
    GBM_paths,
    scalings_l0,
    tot_degree_poly,
    normaleq_components_SL,
    fit_local_vol,
    make_b_bar
)


def compute_weak_error(b_bar, x0, P1, vol, cov_mat, r, dt, N_t, M_samples, 
                       trials=10):
    """
    Compute weak error for option pricing across multiple sample sizes.
    
    Parameters
    ----------
    b_bar : callable
        Fitted local volatility function from make_b_bar
    x0 : ndarray, shape (d, 1)
        Initial asset values
    P1 : ndarray, shape (d,)
        Basket weights
    vol : ndarray, shape (d,)
        Asset volatilities
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    r : float
        Risk-free rate
    dt : float
        Time step size
    N_t : int
        Number of time steps
    M_samples : list of int
        Sample sizes to test (e.g., [2000, 4000, 8000, ...])
    trials : int, optional
        Number of independent trials per sample size (default: 10)
    
    Returns
    -------
    weak_errors : ndarray, shape (len(M_samples), trials)
        Relative weak errors for each sample size and trial
    
    Notes
    -----
    Weak error measures option pricing accuracy:
        ε_weak = |E[g(X_T)] - E[g(S̄_T)]| / |E[g(X_T)]|
    
    where g(·) = max(· - K, 0) is the call option payoff.
    
    Expected behaviour: ε_weak ≈ ε_reg + C/√M
    - ε_reg: regression error (depends on polynomial degree)
    - C/√M: Monte Carlo sampling error
    """
    d = len(P1)
    sqrtdt = math.sqrt(dt)
    t = np.linspace(0, dt * N_t, num=N_t)
    G = np.linalg.cholesky(cov_mat)  # Cholesky factor for correlations
    K = float(P1.dot(x0.flatten()))  # Strike = initial basket value (ATM)
    
    weak_errors = np.zeros((len(M_samples), trials))
    
    for i, M in enumerate(M_samples):
        print(f"  Processing M = {M:>6d} with {trials} trials...", end=" ")
        
        # ====================================================================
        # Projected Process Simulation (1D Markovian SDE)
        # ====================================================================
        
        Z_proj = np.random.randn(trials, M, N_t)  # Pre-generate randomness
        S_proj = np.full((trials, M), K, dtype=float)  # Initialize at K
        
        # Evolve: dS̄ = r*S̄*dt + b̄(t,S̄)*dW
        for n in range(N_t):
            S_proj = S_proj + r * S_proj * dt + \
                     b_bar(t[n], S_proj) * Z_proj[:, :, n] * sqrtdt
        
        # Call option payoff
        price_proj = np.mean(np.maximum(S_proj - K, 0.0), axis=1)
        
        # ====================================================================
        # True Process Simulation (d-dimensional GBM)
        # ====================================================================
        
        Z_true = np.random.randn(trials, M, N_t, d)  # Pre-generate randomness
        X_true = np.broadcast_to(x0.flatten(), (trials, M, d)).copy()
        
        # Evolve: dX = r*X*dt + diag(vol*X)*dW
        for n in range(N_t):
            sigma = vol[None, None, :] * X_true  # Multiplicative volatility
            dW = Z_true[:, :, n, :] @ G.T  # Correlated Brownian increments
            X_true = X_true + r * X_true * dt + sigma * dW * sqrtdt
        
        # Project and compute payoff
        basket_true = X_true.dot(P1)
        price_true = np.mean(np.maximum(basket_true - K, 0.0), axis=1)
        
        # ====================================================================
        # Compute Relative Weak Error
        # ====================================================================
        
        weak_errors[i, :] = np.abs(price_true - price_proj) / np.abs(price_true)
        
        mean_error = weak_errors[i, :].mean()
        std_error = weak_errors[i, :].std(ddof=1)
        print(f"mean = {mean_error:.4f}, std = {std_error:.4f}")
    
    return weak_errors


def plot_convergence_study(M_samples, weak_errors_dict, max_degrees, 
                           save_path=None, show_plot=True):
    """
    Create log-log convergence plot for multiple polynomial degrees.
    
    Parameters
    ----------
    M_samples : list of int
        Sample sizes tested
    weak_errors_dict : dict
        Dictionary mapping polynomial degree to weak_errors array
        Each array has shape (len(M_samples), trials)
    max_degrees : list of int
        Polynomial degrees tested
    save_path : str, optional
        If provided, save figure to this path (default: None)
    show_plot : bool, optional
        Whether to display the plot (default: True)
    
    Notes
    -----
    On log-log scale, Monte Carlo error appears as a line with slope -0.5.
    The vertical offset indicates regression error for each polynomial degree.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, deg in enumerate(max_degrees):
        W = weak_errors_dict[deg]
        mean = np.nanmean(W, axis=1)
        std = np.nanstd(W, axis=1, ddof=1)
        
        # Plot mean with error bars
        ax.errorbar(M_samples, mean, yerr=std, 
                   fmt='-d', color=f'C{i}', 
                   capsize=4, markersize=6,
                   label=f"Max degree = {deg}",
                   linewidth=2, alpha=0.8)
        
        # Plot individual trial points
        for j, M in enumerate(M_samples):
            ax.scatter([M] * len(W[j, :]), W[j, :], 
                      alpha=0.3, s=20, color=f'C{i}')
    
    # Reference line: M^(-1/2) slope
    M_ref = np.array([M_samples[0], M_samples[-1]])
    error_ref = 0.1 * (M_ref / M_samples[0]) ** (-0.5)  # Arbitrary scaling
    ax.plot(M_ref, error_ref, 'k--', linewidth=1.5, alpha=0.5, 
            label=r'Reference: $M^{-1/2}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample Paths $M$', fontsize=12)
    ax.set_ylabel(r'Weak error $\varepsilon_{\mathrm{weak}} = '
                  r'|u - \bar{u}| / |u|$', fontsize=12)
    ax.set_title('Weak Error Convergence: Monte Carlo vs Regression Error', 
                 fontsize=13)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def export_results_csv(M_samples, weak_errors_dict, max_degrees, 
                       filename="weak_error_results.csv"):
    """
    Export weak error results to CSV for further analysis.
    
    Parameters
    ----------
    M_samples : list of int
        Sample sizes tested
    weak_errors_dict : dict
        Dictionary mapping polynomial degree to weak_errors array
    max_degrees : list of int
        Polynomial degrees tested
    filename : str, optional
        Output filename (default: "weak_error_results.csv")
    """
    with open(filename, 'w') as f:
        # Header
        f.write("Degree,M,Trial,WeakError\n")
        
        # Data
        for deg in max_degrees:
            W = weak_errors_dict[deg]
            for i, M in enumerate(M_samples):
                for trial in range(W.shape[1]):
                    f.write(f"{deg},{M},{trial},{W[i, trial]:.6e}\n")
    
    print(f"Results exported to {filename}")


if __name__ == "__main__":
    print("=" * 70)
    print("Weak Error Analysis: Monte Carlo Convergence Study")
    print("=" * 70)
    
    # ========================================================================
    # Model Parameters
    # ========================================================================
    
    # Basket configuration
    d = 3  # Number of assets
    P1 = np.ones(d) / d  # Equal-weighted basket
    
    # Market parameters
    r = 0.05  # Risk-free rate
    x0 = np.linspace(225, 275, num=d)[:, np.newaxis]  # Initial asset values
    vol = np.array([0.2, 0.15, 0.1])  # Asset volatilities
    cov_mat = np.array([[1.0, 0.8, 0.3],
                        [0.8, 1.0, 0.1],
                        [0.3, 0.1, 1.0]])  # Correlation matrix
    
    # Time discretisation
    T = 1.0  # Time horizon
    dt = 0.005  # Time step
    N_t = int(T / dt)  # Number of time steps
    M_t = 400  # Number of training paths for regression
    
    t = np.linspace(0, T, N_t)
    
    # ========================================================================
    # Convergence Study Parameters
    # ========================================================================
    
    max_deg_test = [3, 2, 1, 0]  # Polynomial degrees to compare
    M_samp = [2000, 4000, 8000, 16000, 32000, 64000, 100000]  # Sample sizes
    trials = 10  # Independent trials for statistical confidence
    
    print(f"\nTest configuration:")
    print(f"  Polynomial degrees: {max_deg_test}")
    print(f"  Sample sizes: {M_samp}")
    print(f"  Trials per sample size: {trials}")
    print(f"  Training paths for regression: {M_t}")
    
    # ========================================================================
    # Run Convergence Study
    # ========================================================================
    
    weak_errors_dict = {}
    
    for deg_idx, max_deg in enumerate(max_deg_test):
        print(f"\n{'=' * 70}")
        print(f"Processing polynomial degree = {max_deg} "
              f"({deg_idx + 1}/{len(max_deg_test)})")
        print(f"{'=' * 70}")
        
        # Generate basis functions
        pairs = tot_degree_poly(max_deg)
        print(f"Number of basis functions: {len(pairs)}")
        
        # Pilot run for domain scaling
        s_min0, s_max0, basket0 = scalings_l0(x0, T, dt, r, cov_mat, vol, P1, 
                                              M_0=10000)
        
        # Generate training paths
        paths = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t)
        
        # Build regression system
        D, psi = normaleq_components_SL(paths, P1, pairs, cov_mat, vol, 
                                        s_min0, s_max0, T)
        
        # Solve for coefficients
        c = fit_local_vol(D, psi)
        
        # Quality metrics
        psi_fit = D @ c.reshape(-1, 1)
        residual = np.linalg.norm(psi - psi_fit) / np.linalg.norm(psi)
        cond_D = np.linalg.cond(D)
        
        print(f"Regression quality:")
        print(f"  Condition number: {cond_D:.2e}")
        print(f"  Relative residual: {residual * 100:.2f}%")
        
        # Create callable volatility function
        b_bar = make_b_bar(c, pairs, s_min0, s_max0, T, max_deg)
        
        # Compute weak errors
        print(f"\nComputing weak errors:")
        weak_errors = compute_weak_error(b_bar, x0, P1, vol, cov_mat, r, 
                                        dt, N_t, M_samp, trials=trials)
        
        weak_errors_dict[max_deg] = weak_errors
        
        # Summary statistics
        mean_errors = weak_errors.mean(axis=1)
        print(f"\nSummary for degree {max_deg}:")
        print(f"  Minimum mean error: {mean_errors.min():.4f} "
              f"(at M={M_samp[mean_errors.argmin()]})")
        print(f"  Maximum mean error: {mean_errors.max():.4f} "
              f"(at M={M_samp[mean_errors.argmax()]})")
    
    # ========================================================================
    # Generate Plots and Export Results
    # ========================================================================
    
    print(f"\n{'=' * 70}")
    print("Creating convergence plot...")
    print(f"{'=' * 70}")
    
    fig, ax = plot_convergence_study(M_samp, weak_errors_dict, max_deg_test,
                                     save_path="WeakErrors_deg.pdf",
                                     show_plot=True)
    
    print("\nExporting results to CSV...")
    export_results_csv(M_samp, weak_errors_dict, max_deg_test,
                      filename="weak_error_results.csv")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"{'=' * 70}")
    
    print("\nKey findings:")
    for deg in max_deg_test:
        W = weak_errors_dict[deg]
        mean_at_largest_M = W[-1, :].mean()
        std_at_largest_M = W[-1, :].std(ddof=1)
        print(f"  Degree {deg}: ε_weak = {mean_at_largest_M:.4f} ± "
              f"{std_at_largest_M:.4f} (at M={M_samp[-1]})")
    
    print(f"\nOutputs:")
    print(f"  Plot: WeakErrors_deg.pdf")
    print(f"  Data: weak_error_results.csv")
    print(f"{'=' * 70}")
