"""
This file is based on DM_simplepoly.py from Amelie's work.

Single-Level Markovian Projection - Production Implementation
=============================================================

Implements Markovian projection using simple polynomial basis for L² regression.
Generates correlated GBM paths, fits local volatility surface, and validates projection.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math
import random
from mpl_toolkits.mplot3d import Axes3D


def GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t):
    """
    Generate correlated GBM paths (vectorised implementation).
    
    Parameters
    ----------
    x0 : ndarray, shape (d, 1)
        Initial asset prices
    r : float
        Risk-free rate
    vol : ndarray, shape (d,)
        Volatility for each asset
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
        Simulated asset paths
    """
    G = np.linalg.cholesky(cov_mat)  # Cholesky factor for correlations
    sqrtdt = math.sqrt(dt)
    d = len(vol)
    
    # Initialize all paths at x0
    X = np.tile(x0.flatten(), (M_t, 1))  # Shape: (M_t, d)
    paths = np.empty((M_t, N_t, d))
    paths[:, 0, :] = X
    
    # Vectorised simulation over all paths simultaneously
    for n in range(1, N_t):
        Z = np.random.randn(M_t, d)  # Independent standard normals
        sigma = X * vol  # Element-wise volatility scaling
        dW = Z @ G.T  # Apply correlation structure
        X = X + r * X * dt + sigma * dW * sqrtdt  # Euler-Maruyama step
        paths[:, n, :] = X
    
    return paths


def normaleq_components(paths, P1, pairs, dt, cov_mat, vol):
    """
    Build design matrix D and response vector psi for regression.
    
    Parameters
    ----------
    paths : ndarray, shape (M_t, N_t, d)
        Asset price paths
    P1 : ndarray, shape (d,)
        Basket weights
    pairs : list of tuples
        Polynomial basis pairs (i_t, i_s)
    dt : float
        Time step size
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    vol : ndarray, shape (d,)
        Volatilities
    
    Returns
    -------
    D : ndarray, shape (M_t*N_t, P)
        Design matrix
    psi : ndarray, shape (M_t*N_t, 1)
        Response vector (instantaneous basket variance)
    scalers : tuple
        (t_mean, t_std, s_mean, s_std) for data rescaling
    """
    M_t, N_t, d = paths.shape
    t = np.linspace(0, N_t * dt, N_t)  # Time grid
    
    # Compute basket values: S_bar = P1 · X
    basket = paths.dot(P1)  # Shape: (M_t, N_t)
    
    # Flatten and compute statistics for rescaling
    t_vals = np.tile(t, M_t)  # Repeat time grid for each path
    s_vals = basket.flatten()  # Flatten basket values
    t_mean, t_std = t_vals.mean(), t_vals.std()
    s_mean, s_std = s_vals.mean(), s_vals.std()
    
    # Initialize matrices
    P = len(pairs)
    D = np.zeros((M_t * N_t, P))  # Design matrix
    psi = np.zeros((M_t * N_t, 1))  # Response vector
    
    # Build matrices point by point
    for m in range(M_t):
        for n in range(N_t):
            idx = m * N_t + n
            
            # Rescale to approximately [-1, 1]
            t_c = (t[n] - t_mean) / t_std
            s_c = (basket[m, n] - s_mean) / s_std
            
            # Instantaneous basket variance: w^T Sigma_basket w
            X = paths[m, n]
            sigma_matrix = np.diag(vol * X)  # Diagonal volatility matrix
            basket_cov = sigma_matrix @ cov_mat @ sigma_matrix
            psi[idx] = basket_cov.sum() / d**2  # Equal-weighted basket variance
            
            # Design matrix: polynomial basis evaluated at (t_c, s_c)
            for p, (i1, i2) in enumerate(pairs):
                D[idx, p] = t_c**i1 * s_c**i2
    
    return D, psi, (t_mean, t_std, s_mean, s_std)


def fit_local_vol(D, psi):
    """
    Fit local volatility coefficients using QR decomposition.
    
    Parameters
    ----------
    D : ndarray, shape (M*N, P)
        Design matrix
    psi : ndarray, shape (M*N, 1)
        Response vector
    
    Returns
    -------
    c : ndarray, shape (P,)
        Fitted coefficients
    """
    # QR decomposition for numerical stability
    Q, R = np.linalg.qr(D, mode='reduced')
    alpha = Q.T @ psi
    c = np.linalg.solve(R, alpha).ravel()
    
    # Print diagnostics
    residual = np.linalg.norm(D @ c - psi.ravel())
    psi_norm = np.linalg.norm(psi)
    singular_vals = np.linalg.svd(D, compute_uv=False)
    
    print("Absolute residual ||Dc-psi||:", residual)
    print("Relative residual:", residual / psi_norm)
    print("cond(D):", np.linalg.cond(D))
    print("Smallest singular values:", singular_vals[-5:])
    print("c =", c.round(3))
    
    return c


def make_b_bar(c, pairs, t_mean, t_std, s_mean, s_std):
    """
    Create callable projected volatility function b_bar(t, S).
    
    Parameters
    ----------
    c : ndarray, shape (P,)
        Fitted coefficients
    pairs : list of tuples
        Polynomial basis pairs
    t_mean, t_std : float
        Time rescaling parameters
    s_mean, s_std : float
        Space rescaling parameters
    
    Returns
    -------
    b_bar : callable
        Function b_bar(t, S) returning projected volatility coefficient
    """
    def b_bar(t, S):
        """Evaluate b_bar^2(t, S) = sum_p c_p * t^i1 * s^i2, then take sqrt."""
        # Rescale inputs
        t_c = (t - t_mean) / t_std
        s_c = (S - s_mean) / s_std
        
        # Evaluate polynomial
        h = 0.0
        for p, (i1, i2) in enumerate(pairs):
            h += c[p] * (t_c**i1) * (s_c**i2)
        
        return math.sqrt(h)
    
    return b_bar


def plot_localvol(b_bar, paths, P1, t, K=40, L=150):
    """
    Plot 3D wireframe of projected volatility surface.
    
    Parameters
    ----------
    b_bar : callable
        Projected volatility function
    paths : ndarray, shape (M_t, N_t, d)
        Asset paths (for domain determination)
    P1 : ndarray, shape (d,)
        Basket weights
    t : ndarray
        Time grid
    K : int
        Number of time points for plotting
    L : int
        Number of space points for plotting
    """
    basket = paths.dot(P1)
    
    # Determine plot domain using percentiles
    s_min = np.percentile(basket, 1, axis=0)  # 1st percentile at each time
    s_max = np.percentile(basket, 99, axis=0)  # 99th percentile
    
    # Create plotting grid
    t_plot = np.linspace(0, t[-1], K)
    idx = np.searchsorted(t, t_plot)
    
    T = np.zeros((L, K))  # Time mesh
    S = np.zeros((L, K))  # Space mesh
    
    for j, ti in enumerate(idx):
        T[:, j] = t_plot[j]
        S[:, j] = np.linspace(s_min[ti], s_max[ti], L)
    
    # Evaluate b_bar on grid
    b_bar_vec = np.vectorize(b_bar)
    B_bar = b_bar_vec(T, S)
    
    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(T, S, B_bar, rcount=40, ccount=40,
                     color='blue', linewidth=0.5)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Basket $S$')
    ax.set_zlabel(r'$\bar{b}(t,S)$')
    if save_plot:
        plt.savefig("local_volatility_surface.png", dpi=300)
    plt.show()
    plt.close()


def plot_logreturns(b_bar, x0, P1, vol, cov_mat, r, dt, N_t, M=10000, save_plot = False):
    """
    Validate projection by comparing log returns distributions.
    
    Parameters
    ----------
    b_bar : callable
        Projected volatility function
    x0 : ndarray
        Initial asset prices
    P1 : ndarray
        Basket weights
    vol : ndarray
        Volatilities
    cov_mat : ndarray
        Correlation matrix
    r : float
        Risk-free rate
    dt : float
        Time step size
    N_t : int
        Number of time steps
    M : int
        Number of validation paths
    """
    d = len(P1)
    sqrtdt = math.sqrt(dt)
    t = np.linspace(0, dt * N_t, num=N_t)
    G = np.linalg.cholesky(cov_mat)
    initial_basket = P1.dot(x0.flatten())
    
    # Simulate projected 1D process
    S_bar_terminal = np.zeros(M)
    for m in range(M):
        S_bar = initial_basket
        for n in range(N_t):
            S_bar = S_bar + r * S_bar * dt + b_bar(t[n], S_bar) * random.gauss(0, 1) * sqrtdt
        S_bar_terminal[m] = S_bar
    
    # Simulate true d-dimensional process
    basket_terminal = np.zeros(M)
    for m in range(M):
        X = x0.copy()
        for n in range(N_t):
            sigma_matrix = np.diag(vol * X.flatten())
            dW = G @ np.random.normal(0, 1, (d, 1))
            X = X + r * X * dt + sigma_matrix @ dW * sqrtdt
        basket_terminal[m] = P1.dot(X[:, 0])
    
    # Compute log returns
    log_returns_basket = np.log(basket_terminal / initial_basket)
    log_returns_S_bar = np.log(S_bar_terminal / initial_basket)
    all_returns = np.concatenate([log_returns_basket, log_returns_S_bar])
    bins = np.linspace(all_returns.min(), all_returns.max(), 51)
    
    # Print statistics
    print("Mean, std of true process:", log_returns_basket.mean(), log_returns_basket.std(ddof=0))
    print("Mean, std of projected process:", log_returns_S_bar.mean(), log_returns_S_bar.std(ddof=0))
    
    # Plot comparison
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.hist(log_returns_basket, bins=bins, histtype='stepfilled', 
            color='C0', alpha=0.3, label='True Process: P1·X')
    ax.hist(log_returns_S_bar, bins=bins, alpha=0.75, histtype='step', 
            color='C1', label=r'Markovian Projection: $\bar{S}$')
    ax.set_xlabel('Log returns')
    ax.set_ylabel('Count')
    ax.set_title(f"Equal-weight basket of {d} stocks: dt={dt}, N_t={N_t}, M={M}")
    ax.legend()

    if save_plot:
        plt.savefig("log_returns_comparison.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    # Basket parameters
    d = 3  # Number of assets
    P1 = np.ones(d) / d  # Equal-weighted basket
    
    # Polynomial basis: {1, s, s^2, t}
    pairs = [(0, 0), (0, 1), (0, 2), (1, 0)]
    P = len(pairs)
    print(f"Number of basis functions: {P}")
    print(f"Basis pairs: {pairs}")
    
    # Market parameters
    r = 0.05  # Risk-free rate
    x0 = np.linspace(225, 275, num=d)[:, np.newaxis]  # Initial prices
    vol = np.array([0.2, 0.15, 0.1])  # Volatilities
    cov_mat = np.array([[1.0, 0.8, 0.3],
                        [0.8, 1.0, 0.1],
                        [0.3, 0.1, 1.0]])  # Correlation matrix
    
    # Simulation parameters
    dt = 0.005  # Time step
    N_t = 200  # Number of time steps (T = N_t * dt = 1.0)
    M_t = 200  # Number of training paths
    
    t = np.linspace(0, N_t * dt, num=N_t)
    
    # Main workflow
    print("\n1. Generating training paths...")
    paths = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t)
    
    print("\n2. Building regression matrices...")
    D, psi, scalers = normaleq_components(paths, P1, pairs, dt, cov_mat, vol)
    
    print("\n3. Fitting local volatility...")
    c = fit_local_vol(D, psi)
    
    print("\n4. Creating b_bar function...")
    b_bar = make_b_bar(c, pairs, *scalers)
    
    save_plot = False # Set to True to save plots as PNG files

    print("\n5. Plotting volatility surface...")
    plot_localvol(b_bar, paths, P1, t, K=40, L=150)
    
    print("\n6. Validating projection...")
    plot_logreturns(b_bar, x0, P1, vol, cov_mat, r, dt, N_t, M=10000, save_plot=save_plot)
