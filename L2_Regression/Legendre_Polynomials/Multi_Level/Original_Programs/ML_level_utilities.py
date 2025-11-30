# This file is based on DM_utils_ML.py from Amelie's work.

"""
Multi-Level Utilities for Markovian Projection

Provides Multi-Level specific functions for:
1. Regression system construction (fine/coarse differences)
2. Coefficient solving with column normalisation
3. Volatility surface construction with validation

Key difference from Single-Level: computes b_fine² - b_coarse² for MLMC
telescoping sum instead of just b².
"""

import numpy as np
from numpy.polynomial.legendre import legval

# Import from Single-Level (these are identical)
import sys
sys.path.append('../Single_Level')
from SL_legendre_utilities import GBM_paths, tot_degree_poly


def normaleq_components_ML(paths_f, paths_c, P1, pairs, cov_mat, vol, s_min, 
                           s_max, T, pathsf_are_reduced=False):
    """
    Construct normal equation components D and ψ for Multi-Level regression.
    
    Builds the system D @ c = ψ where:
    - D: Design matrix of Legendre polynomials evaluated on paths
    - ψ: Target vector of volatility DIFFERENCES (b_fine² - b_coarse²)
    
    This is the key difference from Single-Level: we compute the difference
    between fine and coarse volatilities for the MLMC telescoping sum.
    
    Parameters
    ----------
    paths_f : ndarray, shape (M_t, N_f, d)
        Fine timestep paths (or already reduced if pathsf_are_reduced=True)
    paths_c : ndarray, shape (M_t, N_c, d)
        Coarse timestep paths (coupled with fine paths)
    P1 : ndarray, shape (d,)
        Basket weights
    pairs : list of tuple
        Polynomial basis pairs from tot_degree_poly()
    cov_mat : ndarray, shape (d, d)
        Asset correlation matrix
    vol : ndarray, shape (d,)
        Asset volatilities
    s_min, s_max : float
        Domain bounds for spatial scaling to [-1, 1]
    T : float
        Maturity time for temporal scaling to [-1, 1]
    pathsf_are_reduced : bool, optional
        If True, paths_f are already sampled at coarse time intervals
        (default: False, will subsample every other point)
        
    Returns
    -------
    D : ndarray, shape (M_t * N_c, P)
        Design matrix with Legendre polynomials
    psi : ndarray, shape (M_t * N_c, 1)
        Target volatility differences b_fine² - b_coarse²
        
    Notes
    -----
    - Fine paths are subsampled to coarse timesteps (every other point)
    - Legendre polynomials are orthonormalised on [-1, 1]
    - Basket volatility: b² = (1/d²) * Σᵢⱼ σᵢ σⱼ Xᵢ Xⱼ ρᵢⱼ
    - For level l=0, paths_c should be zeros (no coarser level exists)
    """
    M_t, N_c, d = paths_c.shape
    P = len(pairs)
    print(f"Basis pairs: {pairs}")
    
    # Construct design matrix using fine paths at coarse time intervals
    t = np.linspace(0, T, N_c)
    t_scal = 2 * t / T - 1  # Map [0, T] → [-1, 1]
    t_vals = np.tile(t_scal, M_t)
    
    # Subsample fine paths to coarse time intervals if needed
    if not pathsf_are_reduced:
        reducedpaths_f = paths_f[:, ::2, :]  # Every other timestep
    else:
        reducedpaths_f = paths_f
    
    # Compute basket values and scale to [-1, 1]
    s_vals = reducedpaths_f.dot(P1).flatten()
    s_vals = 2 * (s_vals - s_min) / (s_max - s_min) - 1
    
    # Build Vandermonde matrices for Legendre polynomials
    degree = max(i for i, _ in pairs)
    from numpy.polynomial.legendre import legvander
    
    VT = legvander(t_vals, degree)  # Time polynomials
    VS = legvander(s_vals, degree)  # Space polynomials
    
    # Orthonormalisation factor: sqrt((2n+1)/2)
    norm = np.sqrt((2 * np.arange(degree + 1) + 1) / 2)
    VT *= norm[None, :]
    VS *= norm[None, :]
    
    # Construct tensor product design matrix
    D = np.empty((M_t * N_c, P))
    for p, (i1, i2) in enumerate(pairs):
        D[:, p] = VT[:, i1] * VS[:, i2]
    
    # Construct ψ: volatility differences b_fine² - b_coarse²
    psi = np.empty((M_t * N_c, 1))
    for m in range(M_t):
        for n in range(N_c):
            idx = m * N_c + n
            
            # Coarse path volatility
            X_c = paths_c[m, n]
            sigma_c = np.diag(vol * X_c)
            b_c = (sigma_c @ cov_mat @ sigma_c).sum() / d**2
            
            # Fine path volatility (at coarse timestep)
            X_f = reducedpaths_f[m, n]
            sigma_f = np.diag(vol * X_f)
            b_f = (sigma_f @ cov_mat @ sigma_f).sum() / d**2
            
            # Difference for MLMC telescoping sum
            psi[idx] = b_f - b_c
    
    return D, psi


def fit_local_vol(D, psi):
    """
    Solve normal equations D @ c = ψ using QR decomposition with column scaling.
    
    Applies column normalisation before QR to improve numerical stability when
    columns of D have very different magnitudes. This is particularly important
    for high polynomial degrees.
    
    Parameters
    ----------
    D : ndarray, shape (M*N, P)
        Design matrix
    psi : ndarray, shape (M*N, 1)
        Target vector
        
    Returns
    -------
    c : ndarray, shape (P,)
        Coefficient vector
        
    Notes
    -----
    Algorithm:
    1. Normalise columns: D_scaled = D @ diag(1/||D[:,j]||)
    2. QR decomposition: D_scaled = Q @ R
    3. Solve: R @ c_scaled = Q^T @ ψ
    4. Rescale: c = c_scaled / column_norms
    
    This avoids squaring the condition number (unlike normal equations)
    whilst handling badly scaled columns.
    """
    # Column normalisation for stability
    invcolnorm = 1.0 / np.linalg.norm(D, axis=0)
    D *= invcolnorm[None, :]
    
    # QR decomposition (Householder reflections)
    Q, R = np.linalg.qr(D, mode='reduced')
    alpha = Q.T @ psi
    c_s = np.linalg.solve(R, alpha).ravel()
    
    # Rescale coefficients
    c = c_s * invcolnorm
    
    return c


def make_b_bar(c, pairs, s_min, s_max, T, max_deg):
    """
    Construct callable local volatility function b̄(t, S) from coefficients.
    
    Returns a function that evaluates:
        b̄(t, S) = sqrt(Σₚ cₚ P̃ᵢ₁(τ(t)) P̃ᵢ₂(ξ(S)))
    
    where:
    - τ(t) = 2t/T - 1 maps [0, T] → [-1, 1]
    - ξ(S) = 2(S - s_min)/(s_max - s_min) - 1 maps [s_min, s_max] → [-1, 1]
    - P̃ are orthonormalised Legendre polynomials
    
    Parameters
    ----------
    c : ndarray, shape (P,)
        Fitted coefficients
    pairs : list of tuple
        Polynomial basis pairs
    s_min, s_max : float
        Spatial domain bounds
    T : float
        Maturity time
    max_deg : int
        Maximum polynomial degree (for normalisation)
        
    Returns
    -------
    b_bar : callable
        Function b_bar(t, S) that accepts scalar or array inputs
        
    Notes
    -----
    - Checks for negative h = Σ cₚ Pₚ before taking sqrt
    - Prints warning with location if negative values found
    - Returns sqrt(h), which may be NaN if h < 0
    
    Physics analogy: Like constructing effective potential from expansion
    coefficients in quantum mechanics.
    """
    def b_bar(t, S):
        # Scale to [-1, 1]
        t_c = 2 * t / T - 1
        s_c = 2 * (S - s_min) / (s_max - s_min) - 1
        
        # Evaluate polynomial expansion
        h = 0.0
        norm = np.sqrt((2 * np.arange(max_deg + 1) + 1) / 2)
        
        for p, (i1, i2) in enumerate(pairs):
            # Evaluate Legendre polynomials
            P_t = legval(t_c, [0] * i1 + [1]) * norm[i1]
            P_s = legval(s_c, [0] * i2 + [1]) * norm[i2]
            h += c[p] * P_t * P_s
        
        # Check for negative values (volatility must be positive!)
        if np.any(h < 0):
            # Find first offending index
            idx = np.unravel_index(np.argmin(h), h.shape)
            bad_t = np.array(t)[idx] if np.ndim(t) > 0 else t
            bad_S = np.array(S)[idx] if np.ndim(S) > 0 else S
            bad_h = h[idx]
            print(f"Warning: Negative h = {bad_h:.6e} at t = {bad_t:.4f}, "
                  f"S = {bad_S:.4f}")
        
        return np.sqrt(h)
    
    return b_bar
