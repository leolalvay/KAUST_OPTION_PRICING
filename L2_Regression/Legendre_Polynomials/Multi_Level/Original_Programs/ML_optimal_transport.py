# This file is based on DM_ML_OT.py and ML_BS_OTmap.py from Amelie's work.

"""
Multi-Level Monte Carlo with Optimal Transport Variance Reduction

Implements MLMC using Gaussian-Brenier maps for optimal coupling between
fine and coarse paths. This reduces variance beyond standard MLMC by using
optimal transport theory instead of just sharing random numbers.

Key innovation: Instead of sampling coarse paths directly, we TRANSFORM
fine paths to estimated coarse paths using Brenier maps in log-space.

Theory: Brenier's theorem guarantees existence of optimal transport map
between Gaussians with explicit formula via eigendecomposition.
"""

import numpy as np
import math
from ML_level_utilities import tot_degree_poly, normaleq_components_ML, fit_local_vol
from ML_telescoping_sum import scalings_l0, mlmc_l


# ============================================================================
# PART 1: Gaussian-Brenier Map (Optimal Transport)
# ============================================================================

class GaussianBrenierMap:
    """
    Optimal transport map between two Gaussian distributions.
    
    Implements the Brenier map T: X_f ↦ X_c that minimises:
        E[||X_f - T(X_f)||²]
    
    subject to T mapping N(μ_f, C_f) to N(μ_c, C_c).
    
    For Gaussians, the optimal map has explicit form:
        T(x) = μ_c + A(x - μ_f)
    
    where A = C_f^{-1/2} (C_f^{1/2} C_c C_f^{1/2})^{1/2} C_f^{-1/2}
    
    Attributes
    ----------
    mu_f, mu_c : ndarray
        Means of fine and coarse distributions
    C_f, C_c : ndarray
        Covariances of fine and coarse distributions
    A : ndarray
        Linear transformation matrix
        
    Notes
    -----
    Eigendecomposition approach:
    1. C_f = U_f Λ_f U_f^T → C_f^{1/2} = U_f Λ_f^{1/2} U_f^T
    2. M = C_f^{1/2} C_c C_f^{1/2} (symmetric, positive definite)
    3. M = U_M Λ_M U_M^T → M^{1/2} = U_M Λ_M^{1/2} U_M^T
    4. A = C_f^{-1/2} M^{1/2} C_f^{-1/2}
    
    Physics analogy: Like adiabatically transforming one quantum state to
    another whilst preserving the occupation number distribution.
    """
    
    def __init__(self, mu_f, C_f, mu_c, C_c):
        """
        Construct optimal transport map between Gaussians.
        
        Parameters
        ----------
        mu_f : ndarray, shape (d,)
            Mean of fine distribution
        C_f : ndarray, shape (d, d)
            Covariance of fine distribution
        mu_c : ndarray, shape (d,)
            Mean of coarse distribution
        C_c : ndarray, shape (d, d)
            Covariance of coarse distribution
        """
        self.mu_f = np.asarray(mu_f)
        self.mu_c = np.asarray(mu_c)
        self.C_f = np.asarray(C_f)
        self.C_c = np.asarray(C_c)
        
        # Compute C_f^{1/2} and C_f^{-1/2} via eigendecomposition
        eig_f, U_f = np.linalg.eigh(self.C_f)
        sqrt_C_f = U_f @ np.diag(np.sqrt(eig_f)) @ U_f.T
        invsqrt_C_f = U_f @ np.diag(1.0 / np.sqrt(eig_f)) @ U_f.T
        
        # Compute M = C_f^{1/2} C_c C_f^{1/2}
        M = sqrt_C_f @ self.C_c @ sqrt_C_f
        
        # Compute M^{1/2} via eigendecomposition
        eig_M, U_M = np.linalg.eigh(M)
        sqrt_M = U_M @ np.diag(np.sqrt(eig_M)) @ U_M.T
        
        # Final transformation matrix
        self.A = invsqrt_C_f @ sqrt_M @ invsqrt_C_f
    
    def map(self, x):
        """
        Apply Brenier map to transform fine samples to coarse.
        
        Parameters
        ----------
        x : ndarray, shape (..., d)
            Fine samples (can be any shape ending in d)
            
        Returns
        -------
        y : ndarray, shape (..., d)
            Transformed samples (estimated coarse)
            
        Notes
        -----
        T(x) = μ_c + A(x - μ_f)
        
        This preserves mean and covariance:
            E[T(X_f)] = μ_c
            Cov(T(X_f)) = C_c
        """
        x = np.asarray(x)
        return self.mu_c + (x - self.mu_f) @ self.A.T
    
    def __call__(self, x):
        """Convenience: allows map to be called as a function."""
        return self.map(x)


def identity_map(x):
    """
    Identity map for level 0 (no transformation needed).
    
    Parameters
    ----------
    x : ndarray
        Input samples
        
    Returns
    -------
    x : ndarray
        Same samples (no transformation)
        
    Notes
    -----
    At time t=0, fine and coarse paths are identical (both start at X₀),
    so no transport map is needed.
    """
    return x


# ============================================================================
# PART 2: Path Generation with OT Maps
# ============================================================================

def logpaths_maps(x0, T, h0, l, r, cov_mat, vol, max_deg, 
                  return_redpathsf=False, return_pathsc=False, C=80):
    """
    Generate coupled fine/coarse paths and compute Gaussian-Brenier maps.
    
    This function does THREE things:
    1. Generates fine paths (timestep h_fine = h0 * 2^(-l))
    2. Generates coarse paths (timestep h_coarse = 2*h_fine) using SAME random numbers
    3. Computes optimal transport maps in LOG-SPACE at each timestep
    
    Why log-space? GBM is log-normal, so log(X) is Gaussian, allowing us
    to use Gaussian-Brenier maps.
    
    Parameters
    ----------
    x0 : ndarray, shape (d,) or (d, 1)
        Initial asset prices
    T : float
        Maturity time
    h0 : float
        Coarsest timestep
    l : int
        MLMC level (must be >= 1 for OT)
    r : float
        Risk-free rate
    cov_mat : ndarray, shape (d, d)
        Correlation matrix
    vol : ndarray, shape (d,)
        Asset volatilities
    max_deg : int
        Maximum polynomial degree
    return_redpathsf : bool, optional
        If True, return reduced fine paths (default: False)
    return_pathsc : bool, optional
        If True, return coarse paths (default: False)
    C : int, optional
        Sample size scaling factor (default: 80)
        
    Returns
    -------
    maps : list of GaussianBrenierMap
        OT maps for each coarse timestep (maps[0] is identity)
    reducedpaths_f : ndarray, optional
        Fine paths at coarse time intervals (if return_redpathsf=True)
    paths_c : ndarray, optional
        Coarse paths (if return_pathsc=True)
        
    Raises
    ------
    ValueError
        If l < 1 (OT requires fine/coarse pairing)
        
    Notes
    -----
    Map construction:
    1. Compute empirical means: μ_f[n] = mean(log(paths_f[:, n, :]))
    2. Compute empirical covariances: C_f[n] = cov(log(paths_f[:, n, :]))
    3. Construct Brenier map: maps[n] = GaussianBrenierMap(μ_f[n], C_f[n], μ_c[n], C_c[n])
    
    Physics analogy: Like constructing a sequence of canonical transformations
    that map the fine phase space to the coarse phase space at each time slice.
    """
    # Validation
    if l < 0:
        raise ValueError(f"Level l must be non-negative, got l={l}")
    if l == 0:
        raise ValueError("Level l must be >= 1 to form a fine/coarse pair for OT.")
    
    x0 = np.asarray(x0).reshape(-1)
    vol = np.asarray(vol).reshape(-1)
    d = len(vol)
    
    if x0.shape[0] != d:
        raise ValueError("x0 and vol must have same dimension d.")
    
    cov_mat = np.asarray(cov_mat)
    if cov_mat.shape != (d, d):
        raise ValueError("cov_mat must be (d, d).")
    
    # Timestep setup
    hl_f = h0 * 2 ** (-l)
    N_f = int(round(T / hl_f))
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    
    hl_c = 2 * hl_f
    N_c = N_f // 2
    
    G = np.linalg.cholesky(cov_mat)
    
    # Sample size determination
    l_V = max_deg - l
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    M_l = max(C, int(C * dimV**2))
    
    # Initialise path arrays
    paths_f = np.empty((M_l, N_f, d))
    paths_c = np.empty((M_l, N_c, d))
    
    X0 = np.tile(x0, (M_l, 1))
    paths_f[:, 0, :] = X0
    paths_c[:, 0, :] = X0
    
    # Generate random increments (shared for coupling!)
    Z = np.random.randn(M_l, N_f, d)
    
    # ========================================================================
    # Fine paths
    # ========================================================================
    X = X0.copy()
    for n in range(1, N_f):
        Z_n = Z[:, n-1, :]
        sigma = X * vol
        dW = (Z_n @ G.T) * math.sqrt(hl_f)
        X = X + r * X * hl_f + sigma * dW
        paths_f[:, n, :] = X
    
    # ========================================================================
    # Coarse paths
    # ========================================================================
    Z_c = Z.reshape(M_l, N_c, 2, d).sum(axis=2)
    X = X0.copy()
    for n in range(1, N_c):
        Z_n = Z_c[:, n-1, :]
        sigma = X * vol
        dW = (Z_n @ G.T) * math.sqrt(hl_f)
        X = X + r * X * hl_c + sigma * dW
        paths_c[:, n, :] = X
    
    # ========================================================================
    # Extract fine paths at coarse time intervals
    # ========================================================================
    reducedpaths_f = paths_f[:, ::2, :]  # Every other timestep
    
    # ========================================================================
    # Transform to log-space for Gaussian assumption
    # ========================================================================
    logpaths_c = np.log(paths_c)
    logpaths_f = np.log(reducedpaths_f)
    
    # ========================================================================
    # Compute empirical means and covariances
    # ========================================================================
    mu_c = np.mean(logpaths_c, axis=0)  # Shape: (N_c, d)
    mu_f = np.mean(logpaths_f, axis=0)  # Shape: (N_c, d)
    
    # Covariances at each timestep
    C_c = np.array([np.cov(logpaths_c[:, n, :].T, bias=False) 
                    for n in range(N_c)])  # Shape: (N_c, d, d)
    C_f = np.array([np.cov(logpaths_f[:, n, :].T, bias=False) 
                    for n in range(N_c)])  # Shape: (N_c, d, d)
    
    # ========================================================================
    # Construct Brenier maps at each timestep
    # ========================================================================
    maps = [identity_map]  # Time 0: no transformation needed
    
    for n in range(1, N_c):
        maps.append(GaussianBrenierMap(mu_f[n], C_f[n], mu_c[n], C_c[n]))
    
    # ========================================================================
    # Return based on flags
    # ========================================================================
    out = (maps,)
    if return_redpathsf:
        out += (reducedpaths_f,)
    if return_pathsc:
        out += (paths_c,)
    
    if len(out) == 1:
        return out[0]
    elif len(out) == 2:
        return out[0], out[1]
    else:
        return out[0], out[1], out[2]


def apply_maps(paths_f, maps, input_logpaths=False):
    """
    Apply Gaussian-Brenier maps to transform fine paths to estimated coarse.
    
    Parameters
    ----------
    paths_f : ndarray, shape (M, N_c, d)
        Fine paths at coarse time intervals (or log-paths if input_logpaths=True)
    maps : list of callable
        OT maps from logpaths_maps()
    input_logpaths : bool, optional
        If True, paths_f are already in log-space (default: False)
        
    Returns
    -------
    est_paths_c : ndarray, shape (M, N_c, d)
        Estimated coarse paths (in original space, not log)
        
    Notes
    -----
    Algorithm:
    1. Transform to log-space (if needed)
    2. Apply map at each timestep: log(X_c)[n] = map[n](log(X_f)[n])
    3. Transform back: X_c = exp(log(X_c))
    
    This gives estimated coarse paths that:
    - Have same marginal distributions as true coarse paths
    - Are optimally coupled to fine paths (minimal L² distance)
    """
    if not input_logpaths:
        logpaths_f = np.log(paths_f)
    else:
        logpaths_f = paths_f
    
    # Apply maps at each timestep
    logpaths_c = np.empty_like(logpaths_f)
    for k in range(len(maps)):
        logpaths_c[:, k, :] = maps[k](logpaths_f[:, k, :])
    
    # Transform back to original space
    return np.exp(logpaths_c)


def validate_maps(logpaths_f, logpaths_c, maps, tol_mean=1e-2, tol_var=1e-2, 
                  verbose=True):
    """
    Validate that Brenier maps preserve mean and covariance.
    
    Checks that T(X_f) has same mean and covariance as X_c at each timestep.
    This is a diagnostic to ensure maps were computed correctly.
    
    Parameters
    ----------
    logpaths_f : ndarray, shape (M, N_c, d)
        Fine paths in log-space
    logpaths_c : ndarray, shape (M, N_c, d)
        True coarse paths in log-space
    maps : list of callable
        OT maps to validate
    tol_mean : float, optional
        Tolerance for mean difference (default: 1e-2)
    tol_var : float, optional
        Tolerance for covariance Frobenius norm difference (default: 1e-2)
    verbose : bool, optional
        If True, print detailed results (default: True)
        
    Returns
    -------
    all_pass : bool
        True if all timesteps pass tolerance checks
        
    Notes
    -----
    For each timestep n:
    1. Apply map: Y = map[n](X_f[:, n, :])
    2. Check: ||mean(Y) - mean(X_c[:, n, :])||_∞ < tol_mean
    3. Check: ||cov(Y) - cov(X_c[:, n, :])||_F < tol_var
    """
    all_pass = True
    
    for k in range(len(maps)):
        # Apply map to fine paths
        mapped_paths_f = maps[k](logpaths_f[:, k, :])
        
        # Compute statistics
        mu_mapped = np.mean(mapped_paths_f, axis=0)
        mu_c = np.mean(logpaths_c[:, k, :], axis=0)
        mean_diff = np.max(np.abs(mu_mapped - mu_c))
        
        cov_mapped = np.cov(mapped_paths_f.T, bias=False)
        cov_c = np.cov(logpaths_c[:, k, :].T, bias=False)
        cov_diff = np.linalg.norm(cov_mapped - cov_c, ord='fro')
        
        # Check tolerances
        passed = (mean_diff < tol_mean) and (cov_diff < tol_var)
        if not passed:
            all_pass = False
        
        if verbose:
            print(f"Timestep {k}: mean_diff={mean_diff:.3e}, "
                  f"cov_diff={cov_diff:.3e}, pass={passed}")
    
    return all_pass


# ============================================================================
# PART 3: MLMC with Optimal Transport
# ============================================================================

def mlmc_l_OT(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, s_min0, s_max0, C=80):
    """
    Estimate level-l correction using Optimal Transport coupling.
    
    Instead of sampling coarse paths directly, this function:
    1. Generates fine paths
    2. Computes OT maps from fine to coarse distributions
    3. Transforms fine paths to ESTIMATED coarse paths using maps
    4. Computes volatility difference using estimated coarse paths
    
    This gives better variance reduction than standard coupling because
    the fine/coarse correlation is maximised by the OT map.
    
    Parameters
    ----------
    Same as mlmc_l() from ML_telescoping_sum.py
        
    Returns
    -------
    c_padded : ndarray, shape (n_basis_max,)
        Coefficient vector, padded to maximum basis size
        
    Notes
    -----
    Variance improvement: Var[OT] < Var[standard coupling]
    
    The OT map ensures that fine and coarse paths are "as close as possible"
    in the sense of minimising E[||X_f - X_c||²], leading to smaller
    variance in the difference b_f² - b_c².
    
    Physics analogy: Like using optimal control theory to find the path
    between two quantum states that minimises action.
    """
    l_V = max_deg - l
    pairs = tot_degree_poly(l_V)
    dim_max = len(tot_degree_poly(max_deg))
    
    # Generate coupled paths and OT maps
    maps, reducedpaths_f = logpaths_maps(x0, T, h0, l, r, cov_mat, vol, max_deg,
                                         return_redpathsf=True, 
                                         return_pathsc=False, C=C)
    
    # Transform fine paths to estimated coarse paths using OT maps
    est_paths_c = apply_maps(reducedpaths_f, maps, input_logpaths=False)
    
    # Construct regression system (using estimated coarse paths!)
    D, psi = normaleq_components_ML(reducedpaths_f, est_paths_c, P1, pairs, 
                                     cov_mat, vol, s_min0, s_max0, T, 
                                     pathsf_red=True)
    
    # Solve for coefficients
    c = fit_local_vol(D, psi)
    
    # Pad to maximum basis size
    c_padded = np.zeros(dim_max)
    c_padded[:c.shape[0]] = c
    
    return c_padded


def make_c_OT(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0, C=80):
    """
    Aggregate coefficients using OT for levels 1 to L, standard for level 0.
    
    Hybrid approach:
    - Level 0: Standard MLMC (no coarser level for OT)
    - Levels 1-L: Optimal Transport coupling
    
    Computes:
        c_total = c_0 + Σ_{l=1}^L c_l^{OT}
    
    Parameters
    ----------
    Same as make_c() from ML_telescoping_sum.py
        
    Returns
    -------
    c_total : ndarray, shape (n_basis_max,)
        Aggregated coefficient vector
        
    Notes
    -----
    Why hybrid? Level 0 has no coarser level to couple with, so we use
    standard MLMC. Levels 1+ benefit from OT variance reduction.
    
    Computational cost: Slightly higher than standard MLMC due to OT map
    computation, but variance reduction usually outweighs this.
    """
    # Level 0: Standard MLMC
    c_0 = mlmc_l(x0, T, h0, 0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0, C)
    
    # Levels 1 to L: Optimal Transport
    c_1_L = sum(mlmc_l_OT(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, 
                          s_min0, s_max0, C) 
                for l in range(1, max_deg + 1))
    
    return c_0 + c_1_L
