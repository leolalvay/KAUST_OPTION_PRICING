"""
This file is based on DM_SL_logreturns.py from Amelie's work.

Validates Gyöngy's Lemma by comparing log return distributions between
the true high-dimensional process and the Markovian projection. If the
projection is accurate, the terminal distributions should match.
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def validate_log_returns(b_bar, x0, P1, vol, cov_mat, r, dt, N_t, M=10000, 
                        save_path=None, show_plot=True):
    """
    Compare log return distributions between true and projected processes.
    
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
    M : int, optional
        Number of Monte Carlo paths (default: 10000)
    save_path : str, optional
        If provided, save figure to this path (default: None)
    show_plot : bool, optional
        Whether to display the plot (default: True)
    
    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'mean_true': mean log return of true process
        - 'std_true': standard deviation of true process
        - 'mean_proj': mean log return of projected process
        - 'std_proj': standard deviation of projected process
        - 'mean_error': absolute difference in means
        - 'std_error': absolute difference in standard deviations
    
    Notes
    -----
    Gyöngy's Lemma guarantees that marginal distributions match, not paths.
    This function validates that the terminal distributions are consistent.
    """
    d = len(P1)
    sqrtdt = math.sqrt(dt)
    t = np.linspace(0, dt * N_t, num=N_t)
    G = np.linalg.cholesky(cov_mat)  # Cholesky factor for correlations
    S0 = P1.dot(x0.flatten())  # Initial basket value
    
    # ========================================================================
    # Simulate Projected Process (1D Markovian SDE)
    # ========================================================================
    
    S_bar = np.full(M, S0, dtype=float)  # Initialize at basket value
    Z_proj = np.random.randn(M, N_t)  # Pre-generate all random numbers
    
    for n in range(N_t):
        # Evolve: dS_bar = r*S_bar*dt + b_bar(t,S_bar)*dW
        S_bar = S_bar + r * S_bar * dt + b_bar(t[n], S_bar) * Z_proj[:, n] * sqrtdt
    
    S_bar_final = S_bar
    
    # ========================================================================
    # Simulate True Process (d-dimensional GBM, then project)
    # ========================================================================
    
    X = np.tile(x0.flatten(), (M, 1))  # Initialize all paths at x0
    Z_true = np.random.randn(M, N_t, d)  # Pre-generate all random numbers
    
    for n in range(N_t):
        # Evolve: dX = r*X*dt + diag(vol*X)*dW
        sigma = X * vol  # Multiplicative volatility
        dW = Z_true[:, n, :] @ G.T  # Correlated Brownian increments
        X = X + r * X * dt + sigma * dW * sqrtdt
    
    basket_final = X.dot(P1)  # Project at the end
    
    # ========================================================================
    # Compute Log Returns
    # ========================================================================
    
    returns_true = np.log(basket_final / S0)
    returns_proj = np.log(S_bar_final / S0)
    
    # ========================================================================
    # Statistical Comparison
    # ========================================================================
    
    mean_true = returns_true.mean()
    std_true = returns_true.std(ddof=0)  # Population std (Monte Carlo)
    mean_proj = returns_proj.mean()
    std_proj = returns_proj.std(ddof=0)
    
    mean_error = abs(mean_true - mean_proj)
    std_error = abs(std_true - std_proj)
    
    stats = {
        'mean_true': mean_true,
        'std_true': std_true,
        'mean_proj': mean_proj,
        'std_proj': std_proj,
        'mean_error': mean_error,
        'std_error': std_error
    }
    
    # Print summary
    print(f"True process:      mean = {mean_true:.6f}, std = {std_true:.6f}")
    print(f"Projected process: mean = {mean_proj:.6f}, std = {std_proj:.6f}")
    print(f"Absolute errors:   Δmean = {mean_error:.6f}, Δstd = {std_error:.6f}")
    print(f"Relative errors:   Δmean = {mean_error/abs(mean_true)*100:.2f}%, "
          f"Δstd = {std_error/std_true*100:.2f}%")
    
    # ========================================================================
    # Histogram Comparison
    # ========================================================================
    
    all_returns = np.concatenate([returns_true, returns_proj])
    bins = np.linspace(all_returns.min(), all_returns.max(), 51)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # True process: filled histogram
    ax.hist(returns_true, bins=bins, 
            histtype='stepfilled', 
            color='C0', alpha=0.3, 
            label='True Process: $P_1 \\cdot X_T$')
    
    # Projected process: outlined histogram
    ax.hist(returns_proj, bins=bins, 
            histtype='step', 
            color='C1', alpha=0.75, linewidth=2,
            label=r'Markovian Projection: $\bar{S}_T$')
    
    ax.set_xlabel('Log returns: $\\log(S_T / S_0)$')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution Validation: {d} assets, '
                f'$dt={dt}$, $N_t={N_t}$, $M={M}$')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return stats


if __name__ == "__main__":
    # ========================================================================
    # Import dependencies and set up parameters
    # ========================================================================
    
    from SL_legendre_utilities import (
        GBM_paths,
        scalings_l0,
        tot_degree_poly,
        normaleq_components_SL,
        fit_local_vol,
        make_b_bar
    )
    
    print("=" * 60)
    print("Distribution Validation: Gyöngy's Lemma")
    print("=" * 60)
    
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
    M_t = 400  # Number of training paths
    
    # Validation parameters
    M_val = 50000  # Large sample for accurate distribution comparison
    maxdeg = 5  # Polynomial degree
    
    # ========================================================================
    # Fit Local Volatility
    # ========================================================================
    
    print(f"\nFitting local volatility (maxdeg={maxdeg})...")
    
    # Generate basis functions
    pairs = tot_degree_poly(maxdeg)
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
    
    # Create callable volatility function
    b_bar = make_b_bar(c, pairs, s_min0, s_max0, T, maxdeg)
    
    # Quality metrics
    psi_fit = D @ c.reshape(-1, 1)
    residual = np.linalg.norm(psi - psi_fit) / np.linalg.norm(psi)
    cond_D = np.linalg.cond(D)
    
    print(f"Condition number: {cond_D:.2e}")
    print(f"Relative residual: {residual * 100:.2f}%")
    
    # ========================================================================
    # Validate Distribution Matching
    # ========================================================================
    
    print(f"\n{'=' * 60}")
    print(f"Validating with M={M_val} paths...")
    print(f"{'=' * 60}\n")
    
    stats = validate_log_returns(
        b_bar, x0, P1, vol, cov_mat, r, dt, N_t,
        M=M_val,
        save_path="distribution_validation.pdf",
        show_plot=True
    )
    
    # ========================================================================
    # Interpretation
    # ========================================================================
    
    print(f"\n{'=' * 60}")
    print("Interpretation:")
    print(f"{'=' * 60}")
    
    if stats['mean_error'] / abs(stats['mean_true']) < 0.01:
        print("✓ Mean matching: EXCELLENT (< 1% error)")
    elif stats['mean_error'] / abs(stats['mean_true']) < 0.05:
        print("✓ Mean matching: GOOD (< 5% error)")
    else:
        print("✗ Mean matching: NEEDS IMPROVEMENT (> 5% error)")
    
    if stats['std_error'] / stats['std_true'] < 0.01:
        print("✓ Std matching: EXCELLENT (< 1% error)")
    elif stats['std_error'] / stats['std_true'] < 0.05:
        print("✓ Std matching: GOOD (< 5% error)")
    else:
        print("✗ Std matching: NEEDS IMPROVEMENT (> 5% error)")
    
    print(f"\n{'=' * 60}")
    print("Validation complete!")
    print("Saved: distribution_validation.pdf")
    print(f"{'=' * 60}")
