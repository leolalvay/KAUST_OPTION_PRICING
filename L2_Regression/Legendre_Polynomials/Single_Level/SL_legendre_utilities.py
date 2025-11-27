"""
This file is based on DM_SL_utils.py from Amelie's work.

Core utilities for single-level Markovian projection using orthonormalized
Legendre polynomial basis functions. Provides path generation, domain scaling,
regression system construction, and local volatility fitting.
"""

import numpy as np
import math
from numpy.polynomial.legendre import legvander, legval


def GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t):
    """
    Generate vectorized geometric Brownian motion paths for all assets.
    
    Parameters
    ----------
    x0 : ndarray, shape (d, 1)
        Initial asset values
    r : float
        Risk-free interest rate
    vol : ndarray, shape (d,)
        Volatilities for each asset
    cov_mat : ndarray, shape (d, d)
        Correlation matrix for asset returns
    dt : float
        Time step size
    N_t : int
        Number of time steps
    M_t : int
        Number of paths to generate
    
    Returns
    -------
    paths : ndarray, shape (M_t, N_t, d)
        Simulated asset paths with time as middle dimension
    
    Notes
    -----
    Uses Cholesky decomposition for correlated Brownian motion.
    Fully vectorized for computational efficiency.
    """
    G = np.linalg.cholesky(cov_mat)  # Cholesky factor for correlations
    sqrtdt = math.sqrt(dt)
    d = len(vol)
    
    # Initialize all paths at x0
    X = np.tile(x0.flatten(), (M_t, 1))
    paths = np.empty((M_t, N_t, d))
    paths[:, 0, :] = X
    
    # Evolve all paths simultaneously
    for n in range(1, N_t):
        Z = np.random.randn(M_t, d)  # Independent Gaussian draws
        sigma = X * vol  # Multiplicative volatility (geometric Brownian motion)
        dW = Z @ G.T  # Correlated Brownian increments
        X = X + r * X * dt + sigma * dW * sqrtdt
        paths[:, n, :] = X
    
    return paths


def scalings_l0(x0, T, dt, r, cov_mat, vol, P1, M_0=10000):
    """
    Pilot run to determine domain scaling parameters for Legendre basis.
    
    Parameters
    ----------
    x0 : ndarray, shape (d, 1)
        Initial asset values
    T : float
        Time horizon
    dt : float
        Time step size
    r : float
        Risk-free rate
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    vol : ndarray, shape (d,)
        Asset volatilities
    P1 : ndarray, shape (d,)
        Basket weights (typically equal-weighted)
    M_0 : int, optional
        Number of pilot paths (default: 10000)
    
    Returns
    -------
    p1 : float
        1st percentile of basket values (lower bound)
    p99 : float
        99th percentile of basket values (upper bound)
    basket0 : ndarray, shape (M_0, N_t)
        Basket value paths from pilot run
    
    Notes
    -----
    The percentiles define the domain [s_min, s_max] which is then
    scaled to [-1, 1] for Legendre polynomial evaluation.
    """
    N_t = int(T / dt)
    basket0 = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_0).dot(P1)
    p1, p99 = np.percentile(basket0.flatten(), [0.01, 99.99])
    return p1, p99, basket0


def tot_degree_poly(maxdeg=3):
    """
    Generate basis function index pairs with total degree constraint.
    
    Parameters
    ----------
    maxdeg : int, optional
        Maximum total polynomial degree (default: 3)
    
    Returns
    -------
    pairs : list of tuples
        List of (i, j) pairs where i + j <= maxdeg
    
    Notes
    -----
    For maxdeg=3: returns 10 pairs
    For maxdeg=2: returns 6 pairs
    Total degree constraint reduces number of basis functions vs full tensor product.
    """
    return [(i, j) for i in range(maxdeg + 1) 
            for j in range(maxdeg + 1) 
            if (i + j <= maxdeg)]


def normaleq_components_SL(paths, P1, pairs, cov_mat, vol, s_min, s_max, T):
    """
    Build design matrix D and target vector psi for L2 regression.
    
    Parameters
    ----------
    paths : ndarray, shape (M_t, N_t, d)
        Asset paths from GBM_paths
    P1 : ndarray, shape (d,)
        Basket weights
    pairs : list of tuples
        Basis function indices from tot_degree_poly
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    vol : ndarray, shape (d,)
        Asset volatilities
    s_min : float
        Lower bound for basket scaling
    s_max : float
        Upper bound for basket scaling
    T : float
        Time horizon
    
    Returns
    -------
    D : ndarray, shape (M_t * N_t, P)
        Design matrix with orthonormalized Legendre basis
    psi : ndarray, shape (M_t * N_t, 1)
        Target values (squared local volatility)
    
    Notes
    -----
    Scales time to [-1, 1]: t_scal = 2*t/T - 1
    Scales basket to [-1, 1]: s_scal = 2*(s - s_min)/(s_max - s_min) - 1
    Uses orthonormalized Legendre polynomials with norm sqrt((2*i+1)/2)
    """
    M_t, N_t, d = paths.shape
    P = len(pairs)
    
    # Time grid and scaling
    t = np.linspace(0, T, N_t)
    t_scal = 2 * t / T - 1  # Scale to [-1, 1]
    t_vals = np.tile(t_scal, M_t)
    
    # Basket values and scaling
    s_vals = paths.dot(P1).flatten()
    s_vals = 2 * (s_vals - s_min) / (s_max - s_min) - 1  # Scale to [-1, 1]
    
    # Build Vandermonde matrices for Legendre polynomials
    degree = max(i for i, _ in pairs)
    VT = legvander(t_vals, degree)
    VS = legvander(s_vals, degree)
    
    # Apply orthonormalization: tilde{P}_i = sqrt((2*i+1)/2) * P_i
    norm = np.sqrt((2 * np.arange(degree + 1) + 1) / 2)
    VT *= norm[None, :]
    VS *= norm[None, :]
    
    # Form design matrix as tensor products
    D = np.empty((M_t * N_t, P))
    for p, (i1, i2) in enumerate(pairs):
        D[:, p] = VT[:, i1] * VS[:, i2]
    
    # Compute target: squared local volatility
    psi = np.empty((M_t * N_t, 1))
    for m in range(M_t):
        for n in range(N_t):
            idx = m * N_t + n
            X = paths[m, n]
            sigma = np.diag(vol * X)  # Diagonal volatility matrix
            psi[idx] = (sigma @ cov_mat @ sigma).sum() / d**2  # Equal-weight basket
    
    return D, psi


def fit_local_vol(D, psi):
    """
    Solve for coefficients using QR decomposition.
    
    Parameters
    ----------
    D : ndarray, shape (M_t * N_t, P)
        Design matrix from normaleq_components_SL
    psi : ndarray, shape (M_t * N_t, 1)
        Target vector
    
    Returns
    -------
    c : ndarray, shape (P,)
        Fitted coefficients for Legendre expansion
    
    Notes
    -----
    Uses QR decomposition for numerical stability.
    Solves: R*c = Q^T*psi where D = Q*R
    """
    Q, R = np.linalg.qr(D, mode='reduced')
    alpha = Q.T @ psi
    c = np.linalg.solve(R, alpha).ravel()
    return c


def make_b_bar(c, pairs, s_min, s_max, T, max_deg):
    """
    Create callable local volatility function from fitted coefficients.
    
    Parameters
    ----------
    c : ndarray, shape (P,)
        Fitted coefficients from fit_local_vol
    pairs : list of tuples
        Basis function indices
    s_min : float
        Lower bound for basket scaling
    s_max : float
        Upper bound for basket scaling
    T : float
        Time horizon
    max_deg : int
        Maximum polynomial degree
    
    Returns
    -------
    b_bar : callable
        Function b_bar(t, S) that evaluates local volatility
        Can handle scalar or array inputs for t and S
    
    Notes
    -----
    Returned function automatically scales inputs to [-1, 1],
    evaluates Legendre expansion, and returns sqrt(result).
    Fits b_bar^2 but returns b_bar.
    """
    def b_bar(t, S):
        # Scale inputs to [-1, 1]
        t_c = 2 * t / T - 1
        s_c = 2 * (S - s_min) / (s_max - s_min) - 1
        
        # Evaluate Legendre expansion
        h = 0.0
        norm = np.sqrt((2 * np.arange(max_deg + 1) + 1) / 2)
        for p, (i1, i2) in enumerate(pairs):
            P_t = legval(t_c, [0] * i1 + [1]) * norm[i1]  # Normalized Legendre in time
            P_s = legval(s_c, [0] * i2 + [1]) * norm[i2]  # Normalized Legendre in space
            h += c[p] * P_t * P_s
        
        return np.sqrt(h)  # Return b_bar, not b_bar^2
    
    return b_bar
