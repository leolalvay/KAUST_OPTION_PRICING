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
import math
from numpy.polynomial.legendre import legval, legvander


# Note: GBM_paths and tot_degree_poly are identical to Single-Level versions.
# In production, import from SL_legendre_utilities.py
# For standalone testing, we define minimal versions here:

def GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t):
    """Generate GBM paths - minimal version for testing."""
    G = np.linalg.cholesky(cov_mat)
    sqrtdt = math.sqrt(dt)
    d = len(vol)
    X = np.tile(x0.flatten(), (M_t, 1))
    paths = np.empty((M_t, N_t, d))
    paths[:, 0, :] = X
    for n in range(1, N_t):
        Z = np.random.randn(M_t, d)
        sigma = X * vol
        dW = Z @ G.T
        X = X + r * X * dt + sigma * dW * sqrtdt
        paths[:, n, :] = X
    return paths


def tot_degree_poly(maxdeg=3):
    """Generate polynomial basis pairs - minimal version for testing."""
    return [(i, j) for i in range(maxdeg + 1) for j in range(maxdeg + 1) 
            if (i + j <= maxdeg)]


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
        Fine paths (timestep h_fine)
    paths_c : ndarray, shape (M_t, N_c, d)
        Coarse paths (timestep h_coarse = 2*h_fine)
    P1 : ndarray, shape (d,)
        Basket weights
    pairs : list of tuple
        Polynomial basis pairs (i1, i2) with i1 + i2 <= max_deg
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    vol : ndarray, shape (d,)
        Asset volatilities
    s_min, s_max : float
        Domain bounds for Legendre scaling
    T : float
        Maturity time
    pathsf_are_reduced : bool, optional
        If True, paths_f are already subsampled at coarse timesteps
        (default: False)
        
    Returns
    -------
    D : ndarray, shape (M_t * N_c, P)
        Design matrix
    psi : ndarray, shape (M_t * N_c, 1)
        Target vector (b_fine² - b_coarse²)
        
    Notes
    -----
    The design matrix is evaluated at coarse timesteps (N_c points) using
    fine path values subsampled at those times. This ensures both fine and
    coarse volatilities are computed at the same spatial locations.
    
    Physics analogy: Like computing perturbative corrections where you
    evaluate both the full and approximate Hamiltonians at the same points.
    """
    M_t, N_f, d = paths_f.shape
    N_c = paths_c.shape[1]
    P = len(pairs)
    
    # Subsample fine paths at coarse timesteps (every 2nd point)
    if pathsf_are_reduced:
        reducedpaths_f = paths_f
    else:
        reducedpaths_f = paths_f[:, ::2, :]
    
    # Time grid at coarse resolution
    t = np.linspace(0, T, N_c)
    
    # Basket values for scaling
    basket_f = reducedpaths_f.dot(P1)
    
    # Scale to [-1, 1] for Legendre polynomials
    t_vals = np.tile(t, M_t)
    s_vals = basket_f.flatten()
    
    t_c = 2 * t_vals / T - 1
    s_c = 2 * (s_vals - s_min) / (s_max - s_min) - 1
    
    # Compute Legendre basis (Vandermonde-style)
    degree = max(max(p) for p in pairs)
    VT = legvander(t_c, degree)
    VS = legvander(s_c, degree)
    
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
    This version includes two safety mechanisms to prevent NaN values:
    
    1. Domain Clamping (Fix 1): Values of S outside [s_min, s_max] are
       clamped to the boundary. This prevents polynomial extrapolation
       which can cause wild oscillations and negative values.
       
    2. Non-Negative Enforcement (Fix 2): After computing h = Σ cₚ Pₚ,
       negative values are floored to zero before taking sqrt. This
       handles cases where MLMC coefficient cancellation creates
       negative regions even within the fitted domain.
    
    Physics analogy: Like an EFT with a validity cutoff. Outside the
    fitted domain, we use boundary values rather than extrapolating
    into a regime where the effective description breaks down.
    """
    def b_bar(t, S):
        # ====================================================================
        # FIX 1: Domain Clamping
        # Prevent extrapolation outside the fitted region [s_min, s_max].
        # Legendre polynomials are orthogonal on [-1, 1] and can explode
        # outside this domain, leading to negative h values.
        # ====================================================================
        S_clamped = np.clip(S, s_min, s_max)
        
        # Scale to [-1, 1]
        t_c = 2 * t / T - 1
        s_c = 2 * (S_clamped - s_min) / (s_max - s_min) - 1
        
        # Evaluate polynomial expansion
        h = 0.0
        norm = np.sqrt((2 * np.arange(max_deg + 1) + 1) / 2)
        
        for p, (i1, i2) in enumerate(pairs):
            # Evaluate Legendre polynomials
            P_t = legval(t_c, [0] * i1 + [1]) * norm[i1]
            P_s = legval(s_c, [0] * i2 + [1]) * norm[i2]
            h += c[p] * P_t * P_s
        
        # ====================================================================
        # FIX 2: Non-Negative Enforcement
        # MLMC telescoping sums can create negative regions even within the
        # domain due to coefficient cancellation. Floor h to zero to prevent
        # sqrt of negative numbers.
        # ====================================================================
        h_array = np.atleast_1d(h)
        n_negative = np.sum(h_array < 0)
        
        if n_negative > 0:
            neg_fraction = n_negative / h_array.size
            # Only warn if a significant fraction is negative (> 1%)
            if neg_fraction > 0.01:
                min_h = np.min(h_array)
                print(f"Warning: {neg_fraction*100:.1f}% of h values negative "
                      f"(min = {min_h:.4e}), clamping to zero")
        
        # Floor at zero to prevent NaN from sqrt
        h_safe = np.maximum(h, 0.0)
        
        return np.sqrt(h_safe)
    
    return b_bar


# ============================================================================
# Testing / Example Usage
# ============================================================================

if __name__ == "__main__":
    print("ML_level_utilities.py - Multi-Level utilities for Markovian projection")
    print("=" * 70)
    
    # Quick sanity check of the fixes
    print("\nTesting make_b_bar with domain clamping and non-negative enforcement...")
    
    # Create a simple test case
    pairs = tot_degree_poly(2)  # 6 basis functions
    c = np.array([1.0, 0.1, -0.05, 0.02, -0.01, 0.005])  # Example coefficients
    s_min, s_max = 200.0, 300.0
    T = 1.0
    max_deg = 2
    
    b_bar = make_b_bar(c, pairs, s_min, s_max, T, max_deg)
    
    # Test within domain
    S_in = np.array([220.0, 250.0, 280.0])
    b_in = b_bar(0.5, S_in)
    print(f"  Within domain S = {S_in}: b_bar = {b_in}")
    
    # Test outside domain (should be clamped)
    S_out = np.array([150.0, 350.0])
    b_out = b_bar(0.5, S_out)
    print(f"  Outside domain S = {S_out}: b_bar = {b_out} (clamped)")
    
    # Verify no NaN values
    assert not np.any(np.isnan(b_in)), "NaN detected within domain!"
    assert not np.any(np.isnan(b_out)), "NaN detected outside domain!"
    
    print("\n✅ All tests passed - no NaN values produced")
    print("=" * 70)
