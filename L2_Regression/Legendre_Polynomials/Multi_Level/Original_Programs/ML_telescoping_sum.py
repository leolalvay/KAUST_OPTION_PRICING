# This file is based on DM_ML.py from Amelie's work.

"""
Multi-Level Monte Carlo via Telescoping Sum

Implements the standard MLMC hierarchy for local volatility estimation using
the telescoping sum decomposition:

    c_total = Σ_{l=0}^L c_l

where c_l are level-specific corrections estimated with coupled fine/coarse
paths for variance reduction.

Key features:
- Level-dependent timesteps: dt_l = h0 * 2^(-l)
- Level-dependent polynomial degrees: deg_l = max_deg - l
- Adaptive sample sizes: M_l ∝ (number of basis functions)²
- Coupled path generation for variance reduction
"""

import numpy as np
import math
from ML_level_utilities import (
    tot_degree_poly,
    normaleq_components_ML,
    fit_local_vol
)


def scalings_l0(x0, T, h0, r, cov_mat, vol, max_deg, P1, M_0=10000, 
                return_basket=False):
    """
    Pilot run to determine basket domain scaling parameters [s_min, s_max].
    
    Simulates paths at the finest timestep (level L = max_deg) to capture
    the full range of basket values, then extracts percentiles for domain
    bounds used in Legendre polynomial scaling.
    
    Parameters
    ----------
    x0 : ndarray, shape (d, 1)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep (level 0)
    r : float
        Risk-free rate
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    vol : ndarray, shape (d,)
        Asset volatilities
    max_deg : int
        Maximum MLMC level (determines finest timestep)
    P1 : ndarray, shape (d,)
        Basket weights
    M_0 : int, optional
        Number of pilot paths (default: 10000)
    return_basket : bool, optional
        If True, also return basket paths for plotting (default: False)
        
    Returns
    -------
    p1 : float
        1st percentile of basket values (lower bound)
    p99 : float
        99th percentile of basket values (upper bound)
    basket0 : ndarray, shape (M_0, N_t), optional
        Basket paths (only if return_basket=True)
        
    Notes
    -----
    - Uses finest timestep dt = h0 * 2^(-max_deg) to capture full dynamics
    - Percentiles chosen to exclude outliers whilst covering main distribution
    - Basket paths can be reused for domain visualisation
    
    Physics analogy: Like running a pilot experiment to determine the range
    of your detector before doing the full measurement campaign.
    """
    from ML_level_utilities import GBM_paths
    
    # Finest timestep for pilot run
    dt = h0 * 2 ** (-max_deg)
    N_t = int(T / dt)
    
    # Generate pilot paths
    basket0 = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_0).dot(P1)
    
    # Extract domain bounds (exclude extreme outliers)
    p1, p99 = np.percentile(basket0.flatten(), [0.01, 99.99])
    
    if return_basket:
        return p1, p99, basket0
    else:
        return p1, p99


def mlmc_l(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, s_min0, s_max0, C=80):
    """
    Estimate level-l correction coefficients using coupled fine/coarse paths.
    
    This is the core MLMC function. It generates coupled paths at timesteps
    h_fine and h_coarse = 2*h_fine, computes the volatility difference
    b_fine² - b_coarse², and fits polynomial coefficients.
    
    The key MLMC innovation: coupling paths (same random numbers) ensures
    Var[Y_l - Y_{l-1}] << Var[Y_l], giving massive variance reduction.
    
    Parameters
    ----------
    x0 : ndarray, shape (d, 1)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep (level 0)
    l : int
        MLMC level (0, 1, 2, ..., max_deg)
    r : float
        Risk-free rate
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    vol : ndarray, shape (d,)
        Asset volatilities
    max_deg : int
        Maximum polynomial degree at level 0
    P1 : ndarray, shape (d,)
        Basket weights
    s_min0, s_max0 : float
        Domain bounds from pilot run
    C : int, optional
        Sample size scaling factor (default: 80)
        
    Returns
    -------
    c_padded : ndarray, shape (n_basis_max,)
        Coefficient vector, padded to maximum basis size (for summing levels)
        
    Notes
    -----
    Level-dependent features:
    - Timestep: h_l = h0 * 2^(-l)
    - Polynomial degree: deg_l = max_deg - l (fewer basis at fine levels)
    - Sample size: M_l = max(C, C * dim(V)²) where dim(V) = #basis functions
    
    For level 0: paths_c are zeros (no coarser level exists).
    
    Coupling strategy:
    - Fine paths: use random increments Z directly
    - Coarse paths: sum consecutive pairs: Z_coarse[i] = Z[2i] + Z[2i+1]
    
    Physics analogy: Like computing perturbative corrections in QFT where
    each level adds smaller and smaller corrections to the ground state.
    """
    G = np.linalg.cholesky(cov_mat)
    d = len(vol)
    
    # Level-dependent timesteps
    hl_f = h0 * 2 ** (-l)
    N_f = int(round(T / hl_f))
    
    # Ensure even number of steps for clean pairing
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    
    hl_c = 2 * hl_f
    N_c = N_f // 2
    
    # Level-dependent polynomial degree (coarse levels get more basis functions)
    l_V = max_deg - l
    print(f"Level {l}: polynomial degree = {l_V}")
    
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))
    
    # Sample size scales with basis dimension
    M_l = max(C, int(C * dimV**2))
    
    # Initialise path arrays
    paths_f = np.empty((M_l, N_f, d))
    paths_c = np.empty((M_l, N_c, d))
    
    X0 = np.tile(x0.flatten(), (M_l, 1))
    paths_f[:, 0, :] = X0
    paths_c[:, 0, :] = X0
    
    # Generate random increments (shared for coupling!)
    Z = np.random.randn(M_l, N_f, d)
    
    # ========================================================================
    # Fine paths (timestep h_fine)
    # ========================================================================
    X = X0.copy()
    for n in range(1, N_f):
        Z_n = Z[:, n-1, :]
        sigma = X * vol  # Diagonal volatility scaling
        dW = (Z_n @ G.T) * math.sqrt(hl_f)  # Correlated increments
        X = X + r * X * hl_f + sigma * dW
        paths_f[:, n, :] = X
    
    # ========================================================================
    # Coarse paths (timestep h_coarse = 2*h_fine)
    # ========================================================================
    if l > 0:
        # COUPLING: Sum consecutive pairs of fine increments
        Z_c = Z.reshape(M_l, N_c, 2, d).sum(axis=2)
        
        X = X0.copy()
        for n in range(1, N_c):
            Z_n = Z_c[:, n-1, :]
            sigma = X * vol
            dW = (Z_n @ G.T) * math.sqrt(hl_f)  # Note: sqrt(hl_f), not sqrt(hl_c)!
            X = X + r * X * hl_c + sigma * dW
            paths_c[:, n, :] = X
    else:
        # Level 0: no coarser level exists
        paths_c = np.zeros((M_l, N_c, d))
    
    # ========================================================================
    # Construct regression system and solve
    # ========================================================================
    D, psi = normaleq_components_ML(paths_f, paths_c, P1, pairs, cov_mat, vol,
                                     s_min0, s_max0, T)
    
    # Numerical diagnostics
    cond_num = np.linalg.cond(D)
    singular_values = np.linalg.svd(D, compute_uv=False)
    print(f"  Condition number: {cond_num:.2e}")
    print(f"  Smallest 5 singular values: {singular_values[-5:]}")
    
    # Solve for coefficients
    c = fit_local_vol(D, psi)

    pairs_max = tot_degree_poly(max_deg)
    index_map = {pair: idx for idx, pair in enumerate(pairs_max)}

    c_padded = np.zeros(len(pairs_max))
    for k, pair in enumerate(pairs):
        c_padded[index_map[pair]] = c[k]
    
    # Pad to maximum basis size (so all levels can be summed)
    #c_padded = np.zeros(dim_max)
    #c_padded[:c.shape[0]] = c
    
    return c_padded


def make_c(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0, C=80):
    """
    Aggregate coefficients across all MLMC levels via telescoping sum.
    
    Computes:
        c_total = Σ_{l=0}^{max_deg} c_l
    
    where c_l are the level-specific corrections from mlmc_l().
    
    Parameters
    ----------
    x0 : ndarray, shape (d, 1)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep
    r : float
        Risk-free rate
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    vol : ndarray, shape (d,)
        Asset volatilities
    max_deg : int
        Maximum polynomial degree (also number of levels - 1)
    P1 : ndarray, shape (d,)
        Basket weights
    s_min0, s_max0 : float
        Domain bounds from pilot run
    C : int, optional
        Sample size scaling factor (default: 80)
        
    Returns
    -------
    c_total : ndarray, shape (n_basis_max,)
        Aggregated coefficient vector
        
    Notes
    -----
    Number of levels: L = max_deg + 1 (levels 0, 1, ..., max_deg)
    
    Total samples used: Σ M_l where M_l ∝ (max_deg - l + 1)²
    For max_deg = 3:
        - Level 0: ~1280 samples (deg 3: 10 basis)
        - Level 1: ~720 samples (deg 2: 6 basis)
        - Level 2: ~320 samples (deg 1: 3 basis)
        - Level 3: ~80 samples (deg 0: 1 basis)
        Total: ~2400 samples vs ~6400 for single-level
    
    Physics analogy: Like summing perturbative corrections in quantum field
    theory: ground state + first-order + second-order + ...
    """
    return sum(mlmc_l(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, s_min0, 
                      s_max0, C) 
               for l in range(max_deg + 1))
