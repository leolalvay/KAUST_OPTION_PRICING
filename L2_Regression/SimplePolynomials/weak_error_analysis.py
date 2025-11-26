"""
This file is based on DM_tests2.py from Amelie's work.

Weak Error Analysis - Vectorised Implementation
===============================================

Performs convergence analysis of Markovian projection for option pricing.
Measures weak error (pricing accuracy) vs number of validation samples.
Vectorised implementation for computational efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math
import random
import pandas as pd


def GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t):
    """
    Generate correlated GBM paths (vectorised).
    
    Parameters
    ----------
    x0 : ndarray, shape (d, 1)
        Initial asset prices
    r : float
        Risk-free rate
    vol : ndarray, shape (d,)
        Volatilities
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    dt : float
        Time step size
    N_t : int
        Number of time steps
    M_t : int
        Number of paths
    
    Returns
    -------
    paths : ndarray, shape (M_t, N_t, d)
        Asset price paths
    """
    G = np.linalg.cholesky(cov_mat)
    sqrtdt = math.sqrt(dt)
    d = len(vol)
    
    X = np.tile(x0.flatten(), (M_t, 1))
    paths = np.empty((M_t, N_t, d))
    
    for n in range(N_t):
        Z = np.random.randn(M_t, d)
        sigma = X * vol
        dW = Z @ G.T
        X = X + r * X * dt + sigma * dW * sqrtdt
        paths[:, n, :] = X
    
    return paths


def normaleq_components(paths, P1, pairs, dt, cov_mat, vol):
    """
    Build design matrix D and response vector psi.
    
    Parameters
    ----------
    paths : ndarray, shape (M_t, N_t, d)
        Asset paths
    P1 : ndarray
        Basket weights
    pairs : list of tuples
        Polynomial basis pairs
    dt : float
        Time step
    cov_mat : ndarray
        Correlation matrix
    vol : ndarray
        Volatilities
    
    Returns
    -------
    D : ndarray
        Design matrix
    psi : ndarray
        Response vector
    scalers : tuple
        Rescaling parameters
    """
    M_t, N_t, d = paths.shape
    t = np.linspace(0, N_t * dt, N_t)
    
    basket = paths.dot(P1)
    t_vals = np.tile(t, M_t)
    s_vals = basket.flatten()
    t_mean, t_std = t_vals.mean(), t_vals.std()
    s_mean, s_std = s_vals.mean(), s_vals.std()
    
    P = len(pairs)
    D = np.zeros((M_t * N_t, P))
    psi = np.zeros((M_t * N_t, 1))
    
    for m in range(M_t):
        for n in range(N_t):
            idx = m * N_t + n
            t_c = (t[n] - t_mean) / t_std
            s_c = (basket[m, n] - s_mean) / s_std
            
            X = paths[m, n]
            sigma_matrix = np.diag(vol * X)
            psi[idx] = (sigma_matrix @ cov_mat @ sigma_matrix).sum() / d**2
            
            for p, (i1, i2) in enumerate(pairs):
                D[idx, p] = t_c**i1 * s_c**i2
    
    return D, psi, (t_mean, t_std, s_mean, s_std)


def fit_local_vol(D, psi):
    """
    Fit coefficients using QR decomposition (silent version).
    
    Parameters
    ----------
    D : ndarray
        Design matrix
    psi : ndarray
        Response vector
    
    Returns
    -------
    c : ndarray
        Fitted coefficients
    """
    Q, R = np.linalg.qr(D, mode='reduced')
    alpha = Q.T @ psi
    c = np.linalg.solve(R, alpha).ravel()
    return c


def make_b_bar(c, pairs, t_mean, t_std, s_mean, s_std):
    """
    Create projected volatility function (vectorised-compatible).
    
    Parameters
    ----------
    c : ndarray
        Coefficients
    pairs : list of tuples
        Basis pairs
    t_mean, t_std : float
        Time rescaling
    s_mean, s_std : float
        Space rescaling
    
    Returns
    -------
    b_bar : callable
        Projected volatility function
    """
    def b_bar(t, S):
        """Evaluate b_bar(t, S) with numpy sqrt for vectorisation."""
        t_c = (t - t_mean) / t_std
        s_c = (S - s_mean) / s_std
        h = 0.0
        for p, (i1, i2) in enumerate(pairs):
            h += c[p] * (t_c**i1) * (s_c**i2)
        return np.sqrt(h)  # Use np.sqrt for array compatibility
    
    return b_bar


def plot_localvol(b_bar, paths, P1, t, K=40, L=150):
    """
    Plot 3D volatility surface.
    
    Parameters
    ----------
    b_bar : callable
        Projected volatility
    paths : ndarray
        Asset paths
    P1 : ndarray
        Basket weights
    t : ndarray
        Time grid
    K : int
        Time points for plot
    L : int
        Space points for plot
    
    Returns
    -------
    domain : tuple
        (s_min, s_max) for domain reference
    """
    basket = paths.dot(P1)
    s_min = np.percentile(basket, 1, axis=0)
    s_max = np.percentile(basket, 99, axis=0)
    
    t_plot = np.linspace(0, t[-1], K)
    idx = np.searchsorted(t, t_plot)
    
    T = np.zeros((L, K))
    S = np.zeros((L, K))
    
    for j, ti in enumerate(idx):
        T[:, j] = t_plot[j]
        S[:, j] = np.linspace(s_min[ti], s_max[ti], L)
    
    b_bar_vec = np.vectorize(b_bar)
    B_bar = b_bar_vec(T, S)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(T, S, B_bar, rcount=40, ccount=40,
                     color='blue', linewidth=0.5)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Basket $S$')
    ax.set_zlabel(r'$\bar{b}(t,S)$')
    plt.show()
    plt.close()
    
    return (s_min, s_max)


def plot_weakerror(b_bar, x0, P1, vol, cov_mat, r, dt, N_t, M_samples):
    """
    Plot weak error vs number of validation samples (vectorised).
    
    Weak error measures option pricing accuracy:
    |E[payoff_true] - E[payoff_projected]| / |E[payoff_true]|
    
    Parameters
    ----------
    b_bar : callable
        Projected volatility function
    x0 : ndarray
        Initial prices
    P1 : ndarray
        Basket weights
    vol : ndarray
        Volatilities
    cov_mat : ndarray
        Correlation matrix
    r : float
        Risk-free rate
    dt : float
        Time step
    N_t : int
        Number of time steps
    M_samples : list of int
        Sample sizes to test
    """
    d = len(P1)
    sqrtdt = math.sqrt(dt)
    t = np.linspace(0, dt * N_t, num=N_t)
    G = np.linalg.cholesky(cov_mat)
    
    trials = 10  # Number of independent trials per sample size
    weak_errors = np.zeros((len(M_samples), trials))
    K = float(P1.dot(x0.flatten()))  # Strike = initial basket value
    
    for i, M in enumerate(M_samples):
        # Pre-generate random numbers for all trials
        Z_proj = np.random.randn(trials, M, N_t)  # Projected process noise
        S_proj = np.full((trials, M), K, dtype=float)  # Initialize at strike
        
        # Vectorised simulation of projected process
        for n in range(N_t):
            S_proj = S_proj + r * S_proj * dt + b_bar(t[n], S_proj) * Z_proj[:, :, n] * sqrtdt
        
        # Compute projected call prices
        price_proj = np.mean(np.maximum(S_proj - K, 0.0), axis=1)
        
        # Pre-generate random numbers for true process
        Z_true = np.random.randn(trials, M, N_t, d)
        X_true = np.broadcast_to(x0.flatten(), (trials, M, d)).copy()
        
        # Vectorised simulation of true process
        for n in range(N_t):
            sigma = vol[None, None, :] * X_true  # Broadcasting volatilities
            dW = Z_true[:, :, n, :] @ G.T
            X_true = X_true + r * X_true * dt + sigma * dW * sqrtdt
        
        # Compute true call prices
        price_true = np.mean(np.maximum(X_true.dot(P1) - K, 0.0), axis=1)
        
        # Compute relative weak errors
        weak_errors[i, :] = np.abs(price_true - price_proj) / np.abs(price_true)
        print(f"Completed {trials} trials for M = {M}")
    
    # Compute statistics
    we_mean = weak_errors.mean(axis=1)
    we_std = weak_errors.std(axis=1, ddof=1)
    
    # Save to CSV for further analysis
    df = pd.DataFrame({
        "M": np.repeat(M_samples, trials),
        "weak_error": weak_errors.flatten()
    })
    df["mean"] = np.repeat(we_mean, trials)
    df["std"] = np.repeat(we_std, trials)
    df.to_csv("weak_errors_by_M.csv", index=False)
    print("\nResults saved to 'weak_errors_by_M.csv'")

    
    # Plot results
    for i, M in enumerate(M_samples):
        plt.scatter([M] * trials, weak_errors[i, :], alpha=0.6, color='C1')

    plt.errorbar(M_samples, we_mean, yerr=we_std, fmt='-o', 
                color='blue', capsize=4, label='Mean ± Std')
    plt.xscale('log')

    # Custom x-axis formatting: show as powers of 2 with ×10³
    ax = plt.gca()
    ax.set_xticks(M_samples)
    ax.set_xticklabels(['2', '4', '8', '16', '32', '64'])
    plt.xlabel(r'Sample Paths $M$ ($\times 10^3$)')

    plt.ylabel(r'Weak error $|u_{E}-\bar{u}_{E}| / |u_{E}|$')
    plt.title(r'Weak Error vs Sample Size (fixed regression basis)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("weak_error_analysis.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Basket parameters
    d = 3
    P1 = np.ones(d) / d
    
    # Polynomial basis: {1, s, t} (reduced from production for faster testing)
    pairs = [(0, 0), (0, 1), (1, 0)]
    P = len(pairs)
    print(f"Number of basis functions: {P}")
    print(f"Basis pairs: {pairs}")
    
    # Market parameters
    r = 0.05
    x0 = np.linspace(225, 275, num=d)[:, np.newaxis]
    vol = np.array([0.2, 0.15, 0.1])
    cov_mat = np.array([[1.0, 0.8, 0.3],
                        [0.8, 1.0, 0.1],
                        [0.3, 0.1, 1.0]])
    
    # Simulation parameters
    dt = 0.005
    N_t = 200  # T = 1.0 year
    M_t = 400  # Training paths (more than production for better fit)
    
    # Test sample sizes (powers of 2 for clean log scale)
    M_samples = [2000, 4000, 8000, 16000, 32000, 64000]
    
    t = np.linspace(0, N_t * dt, num=N_t)
    
    # Main workflow
    print("\n1. Generating training paths...")
    paths = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t)
    
    print("\n2. Building regression matrices...")
    D, psi, scalers = normaleq_components(paths, P1, pairs, dt, cov_mat, vol)
    
    print("\n3. Fitting local volatility...")
    c = fit_local_vol(D, psi)
    print(f"Fitted coefficients: {c.round(3)}")
    
    print("\n4. Creating b_bar function...")
    b_bar = make_b_bar(c, pairs, *scalers)
    
    print("\n5. Plotting volatility surface...")
    domain = plot_localvol(b_bar, paths, P1, t, K=40, L=150)
    
    print("\n6. Running weak error analysis...")
    print("This will take several minutes (vectorised implementation)...")
    plot_weakerror(b_bar, x0, P1, vol, cov_mat, r, dt, N_t, M_samples)
