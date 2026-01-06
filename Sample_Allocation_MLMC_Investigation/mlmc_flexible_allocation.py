"""
Modified MLMC Functions with Flexible Sample Allocation
========================================================

This module provides modified versions of the MLMC functions that accept
different sample allocation formulas for M_l.

Allocation Formulas Available:
- 'quadratic': M_l = C * dim(V)^2        (current, conservative)
- 'log':       M_l = C * dim(V) * log(dim(V)+1)  (Cohen-Migliorati optimal)
- 'linear':    M_l = C * dim(V)          (minimum for uniqueness)
- 'sqrt':      M_l = C * dim(V)^1.5      (intermediate)
- callable:    Custom function f(dimV, C) -> int

STANDALONE VERSION: All imports are from local files in the same folder.

Author: Wadoud Bouslama (KAUST Internship)
Based on: mlmc_ot_estimator.py
"""

import numpy as np
import math
import time
from typing import Tuple, Callable, Optional, Dict, Any, Union
from dataclasses import dataclass
from numpy.polynomial.legendre import legvander
from scipy import stats

# LOCAL IMPORTS (same folder)
from mlmc_ot_estimator import (
    tot_degree_poly,
    GBM_paths,
    estimate_domain,
    compute_ot_maps,
    make_b_bar,
    make_b_squared,
    LevelStats
)
from common import VolatilitySurfaceResult


# =============================================================================
# SECTION 1: Sample Allocation Formulas
# =============================================================================

def allocation_quadratic(dimV: int, C: int) -> int:
    """Current implementation: M_l = C * dim(V)^2"""
    return max(C, int(C * dimV ** 2))


def allocation_log(dimV: int, C: int) -> int:
    """Cohen-Migliorati optimal: M_l = C * dim(V) * log(dim(V) + 1)"""
    return max(C, int(C * dimV * np.log(dimV + 1)))


def allocation_linear(dimV: int, C: int) -> int:
    """Minimum for uniqueness: M_l = C * dim(V)"""
    return max(C, int(C * dimV))


def allocation_sqrt(dimV: int, C: int) -> int:
    """Intermediate: M_l = C * dim(V)^1.5"""
    return max(C, int(C * dimV ** 1.5))


def allocation_log_squared(dimV: int, C: int) -> int:
    """Alternative: M_l = C * dim(V) * log(dim(V) + 1)^2"""
    return max(C, int(C * dimV * np.log(dimV + 1) ** 2))


# Registry of named formulas
ALLOCATION_FORMULAS = {
    'quadratic': allocation_quadratic,
    'log': allocation_log,
    'linear': allocation_linear,
    'sqrt': allocation_sqrt,
    'log_squared': allocation_log_squared,
}


def get_allocation_function(formula: Union[str, Callable]) -> Callable:
    """
    Get allocation function from name or return callable directly.
    
    Parameters
    ----------
    formula : str or callable
        Either a string name ('quadratic', 'log', 'linear', 'sqrt')
        or a callable with signature f(dimV: int, C: int) -> int
    
    Returns
    -------
    callable
        The allocation function
    """
    if callable(formula):
        return formula
    elif formula in ALLOCATION_FORMULAS:
        return ALLOCATION_FORMULAS[formula]
    else:
        raise ValueError(f"Unknown formula: {formula}. "
                        f"Available: {list(ALLOCATION_FORMULAS.keys())}")


def compute_samples_for_formula(max_deg: int, C: int, formula: Union[str, Callable]) -> Dict[int, int]:
    """
    Compute M_l for each level given the allocation formula.
    
    Parameters
    ----------
    max_deg : int
        Maximum polynomial degree
    C : int
        Base constant
    formula : str or callable
        Allocation formula
    
    Returns
    -------
    Dict[int, int]
        Mapping from level to sample count
    """
    alloc_fn = get_allocation_function(formula)
    samples = {}
    for level in range(max_deg + 1):
        l_V = max_deg - level  # Polynomial degree at this level
        dimV = len(tot_degree_poly(l_V))
        samples[level] = alloc_fn(dimV, C)
    return samples


# =============================================================================
# SECTION 2: Modified MLMC Level Functions
# =============================================================================

def mlmc_level_flexible(
    x0: np.ndarray, T: float, h0: float, level: int,
    r: float, cov_mat: np.ndarray, vol: np.ndarray,
    max_deg: int, P1: np.ndarray, s_min: float, s_max: float,
    C: int = 80, batch_size: int = 50, verbose: bool = False,
    return_stats: bool = False,
    formula: Union[str, Callable] = 'quadratic'
) -> Union[np.ndarray, Tuple[np.ndarray, LevelStats]]:
    """
    Compute MLMC level-l coefficients with flexible sample allocation.
    
    This is a modified version of mlmc_level that accepts different
    allocation formulas for M_l.
    
    Parameters
    ----------
    [Same as mlmc_level, plus:]
    formula : str or callable
        Sample allocation formula. Options:
        - 'quadratic': M_l = C * dim(V)^2 (default)
        - 'log': M_l = C * dim(V) * log(dim(V)+1)
        - 'linear': M_l = C * dim(V)
        - 'sqrt': M_l = C * dim(V)^1.5
        - callable: Custom function f(dimV, C) -> int
    
    Returns
    -------
    c_padded : ndarray
        Coefficient vector, zero-padded to dim_max length.
    stats : LevelStats (only if return_stats=True)
        Level statistics including variance, correlation, kurtosis.
    """
    x0 = np.asarray(x0).flatten()
    vol = np.asarray(vol).flatten()
    d = len(vol)
    
    # Multi-resolution: polynomial degree decreases with level
    l_V = max_deg - level
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))
    
    # FLEXIBLE SAMPLE ALLOCATION
    alloc_fn = get_allocation_function(formula)
    M_l = alloc_fn(dimV, C)
    n_batches = int(np.ceil(M_l / batch_size))
    
    # Time discretisation
    hl_f = h0 * 2 ** (-level)
    N_f = int(round(T / hl_f))
    if level > 0 and N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    
    if level == 0:
        hl_c = hl_f
        N_c = N_f
    else:
        hl_c = 2 * hl_f
        N_c = N_f // 2
    
    G_chol = np.linalg.cholesky(cov_mat)
    
    # Precompute Legendre values for time grid (normalised to [-1, 1])
    t_grid = np.arange(N_c) * hl_c
    t_scaled = 2.0 * t_grid / T - 1.0
    deg_t = l_V
    deg_s = l_V
    norm_t = np.sqrt(2 * np.arange(deg_t + 1) + 1)
    norm_s = np.sqrt(2 * np.arange(deg_s + 1) + 1)
    VT = legvander(t_scaled, deg_t) * norm_t

    # Accumulators for normal equations
    G = np.zeros((dimV, dimV))
    g = np.zeros((dimV, 1))

    # Accumulators for variance statistics
    if return_stats:
        n_samples_total = 0
        sum_psi = 0.0
        sum_psi2 = 0.0
        sum_bf = 0.0
        sum_bc = 0.0
        sum_bf2 = 0.0
        sum_bc2 = 0.0
        sum_bf_bc = 0.0
        all_psi = []

    if verbose:
        print(f"  Level {level}: deg={l_V}, dimV={dimV}, M={M_l}, batches={n_batches}")

    for b_idx in range(n_batches):
        B = min(batch_size, M_l - b_idx * batch_size)
        if B <= 0:
            break

        X0 = np.tile(x0, (B, 1))

        if level == 0:
            # Level 0: no coarse paths, just fit b² directly
            X_f = X0.copy()
            Z = np.random.randn(B, N_f, d)
            
            for n in range(N_f):
                if n > 0:
                    Z_n = Z[:, n - 1, :]
                    sigma = X_f * vol
                    dW = (Z_n @ G_chol.T) * math.sqrt(hl_f)
                    X_f = X_f + r * X_f * hl_f + sigma * dW
                
                # Build design matrix row block
                D_n = np.empty((B, dimV), dtype=float)
                trow = VT[n, :]
                s = X_f @ P1
                s_scaled = np.clip(2.0 * (s - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
                VS = legvander(s_scaled, deg_s) * norm_s
                
                for p, (i1, i2) in enumerate(pairs):
                    D_n[:, p] = trow[i1] * VS[:, i2]
                
                # Instantaneous variance
                w_sigma_f = X_f * vol * P1
                b_sq = ((w_sigma_f @ cov_mat) * w_sigma_f).sum(axis=1)
                psi_n = b_sq.reshape(-1, 1)

                if return_stats:
                    n_samples_total += B
                    sum_psi += np.sum(b_sq)
                    sum_psi2 += np.sum(b_sq ** 2)
                    sum_bf += np.sum(b_sq)
                    sum_bf2 += np.sum(b_sq ** 2)
                    sum_bc += np.sum(b_sq)
                    sum_bc2 += np.sum(b_sq ** 2)
                    sum_bf_bc += np.sum(b_sq ** 2)
                    all_psi.extend(b_sq.tolist())

                G += D_n.T @ D_n
                g += D_n.T @ psi_n
        
        else:
            # Level > 0: compute telescoping difference
            X_f = X0.copy()
            X_c = X0.copy()
            Z = np.random.randn(B, N_f, d)
            
            # Initial timestep
            trow0 = VT[0, :]
            s0 = X0 @ P1
            s0_scaled = np.clip(2.0 * (s0 - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
            VS0 = legvander(s0_scaled, deg_s) * norm_s
            
            D0 = np.empty((B, dimV), dtype=float)
            for p, (i_t, i_s) in enumerate(pairs):
                D0[:, p] = trow0[i_t] * VS0[:, i_s]
            psi0 = np.zeros((B, 1), dtype=float)
            
            G += D0.T @ D0
            g += D0.T @ psi0
            
            for n in range(1, N_c):
                # Two fine steps
                Z_n1 = Z[:, 2 * (n - 1), :]
                sigma = X_f * vol
                dW1 = (Z_n1 @ G_chol.T) * math.sqrt(hl_f)
                X_f = X_f + r * X_f * hl_f + sigma * dW1
                
                Z_n2 = Z[:, 2 * n - 1, :]
                sigma = X_f * vol
                dW2 = (Z_n2 @ G_chol.T) * math.sqrt(hl_f)
                X_f = X_f + r * X_f * hl_f + sigma * dW2
                
                # One coarse step (shared Brownian increment)
                dW_c = dW1 + dW2
                sigma = X_c * vol
                X_c = X_c + r * X_c * hl_c + sigma * dW_c
                
                # Build design matrix
                D_n = np.empty((B, dimV), dtype=float)
                trow = VT[n, :]
                s = X_f @ P1
                s_scaled = np.clip(2.0 * (s - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
                VS = legvander(s_scaled, deg_s) * norm_s
                
                for p, (i1, i2) in enumerate(pairs):
                    D_n[:, p] = trow[i1] * VS[:, i2]
                
                # Telescoping difference
                w_sigma_f = X_f * vol * P1
                b_f = ((w_sigma_f @ cov_mat) * w_sigma_f).sum(axis=1)
                w_sigma_c = X_c * vol * P1
                b_c = ((w_sigma_c @ cov_mat) * w_sigma_c).sum(axis=1)
                psi_n = (b_f - b_c).reshape(-1, 1)

                if return_stats:
                    diff = b_f - b_c
                    n_samples_total += B
                    sum_psi += np.sum(diff)
                    sum_psi2 += np.sum(diff ** 2)
                    sum_bf += np.sum(b_f)
                    sum_bc += np.sum(b_c)
                    sum_bf2 += np.sum(b_f ** 2)
                    sum_bc2 += np.sum(b_c ** 2)
                    sum_bf_bc += np.sum(b_f * b_c)
                    all_psi.extend(diff.tolist())

                G += D_n.T @ D_n
                g += D_n.T @ psi_n

    # Solve normal equations via Cholesky
    G_reg = 0.5 * (G + G.T)
    
    # Check condition number
    lam = np.linalg.eigvalsh(G_reg)
    lam_min = max(lam[0], 1e-300)
    cond_approx = np.sqrt(lam[-1] / lam_min)
    
    if verbose:
        print(f"    Condition number ≈ {cond_approx:.2e}")
    
    # Add regularisation if ill-conditioned
    if cond_approx > 1e12:
        reg = 1e-10 * lam[-1]
        G_reg += reg * np.eye(dimV)
        if verbose:
            print(f"    Added regularisation: {reg:.2e}")
    
    L = np.linalg.cholesky(G_reg)
    y = np.linalg.solve(L, g)
    c = np.linalg.solve(L.T, y).ravel()

    # Zero-pad to maximum dimension
    c_padded = np.zeros(dim_max)
    c_padded[:len(c)] = c

    if not return_stats:
        return c_padded

    # Compute level statistics
    n = n_samples_total
    if n > 1:
        mean_psi = sum_psi / n
        var_psi = (sum_psi2 / n - mean_psi ** 2) * n / (n - 1)

        mean_bf = sum_bf / n
        mean_bc = sum_bc / n
        var_bf = (sum_bf2 / n - mean_bf ** 2) * n / (n - 1)
        var_bc = (sum_bc2 / n - mean_bc ** 2) * n / (n - 1)

        if level == 0:
            corr = 1.0
        else:
            cov_bf_bc = (sum_bf_bc / n - mean_bf * mean_bc) * n / (n - 1)
            std_bf = np.sqrt(max(var_bf, 1e-15))
            std_bc = np.sqrt(max(var_bc, 1e-15))
            if std_bf > 1e-10 and std_bc > 1e-10:
                corr = cov_bf_bc / (std_bf * std_bc)
                corr = np.clip(corr, -1.0, 1.0)
            else:
                corr = 1.0

        all_psi_arr = np.array(all_psi)
        if len(all_psi_arr) > 3:
            try:
                kurt = stats.kurtosis(all_psi_arr, fisher=True)
            except:
                kurt = 0.0
        else:
            kurt = 0.0
    else:
        mean_psi = sum_psi if n > 0 else 0.0
        var_psi = 0.0
        var_bf = 0.0
        var_bc = 0.0
        corr = 1.0
        kurt = 0.0

    level_stats = LevelStats(
        level=level,
        coefficients=c_padded,
        variance=max(var_psi, 0.0),
        mean_correction=mean_psi,
        correlation=corr,
        kurtosis=kurt,
        n_samples=n,
        var_fine=max(var_bf, 0.0),
        var_coarse=max(var_bc, 0.0)
    )

    return c_padded, level_stats


def mlmc_level_ot_flexible(
    x0: np.ndarray, T: float, h0: float, level: int,
    r: float, cov_mat: np.ndarray, vol: np.ndarray,
    max_deg: int, P1: np.ndarray, s_min: float, s_max: float,
    C: int = 80, batch_size: int = 50,
    M_pilot: int = 500, verbose: bool = False,
    return_stats: bool = False,
    formula: Union[str, Callable] = 'quadratic'
) -> Union[np.ndarray, Tuple[np.ndarray, LevelStats]]:
    """
    Compute MLMC level-l coefficients with OT coupling and flexible allocation.
    
    Parameters
    ----------
    [Same as mlmc_level_ot, plus:]
    formula : str or callable
        Sample allocation formula (see mlmc_level_flexible for options).
    
    Returns
    -------
    c_padded : ndarray
        Coefficient vector, zero-padded to dim_max length.
    stats : LevelStats (only if return_stats=True)
        Level statistics including variance, correlation, kurtosis.
    """
    if level < 1:
        raise ValueError(f"Level must be >= 1 for OT coupling, got {level}")
    
    x0 = np.asarray(x0).flatten()
    vol = np.asarray(vol).flatten()
    d = len(vol)
    
    # Multi-resolution: polynomial degree decreases with level
    l_V = max_deg - level
    pairs = tot_degree_poly(l_V)
    dimV = len(pairs)
    dim_max = len(tot_degree_poly(max_deg))
    
    # FLEXIBLE SAMPLE ALLOCATION
    alloc_fn = get_allocation_function(formula)
    M_l = alloc_fn(dimV, C)
    n_batches = int(np.ceil(M_l / batch_size))
    
    # Time discretisation
    hl_f = h0 * 2 ** (-level)
    N_f = int(round(T / hl_f))
    if N_f % 2 == 1:
        N_f += 1
    hl_f = T / N_f
    hl_c = 2 * hl_f
    N_c = N_f // 2
    
    G_chol = np.linalg.cholesky(cov_mat)
    
    # Compute OT maps from pilot simulation
    if verbose:
        print(f"  Level {level}: Computing OT maps from {M_pilot} pilot paths...")
    
    maps = compute_ot_maps(x0, T, h0, level, r, cov_mat, vol, 
                           M_pilot=M_pilot, batch_size=min(50, M_pilot))
    
    # Precompute Legendre values for time grid
    t_grid = np.arange(N_c) * hl_c
    t_scaled = 2.0 * t_grid / T - 1.0
    deg_t = l_V
    deg_s = l_V
    norm_t = np.sqrt(2 * np.arange(deg_t + 1) + 1)
    norm_s = np.sqrt(2 * np.arange(deg_s + 1) + 1)
    VT = legvander(t_scaled, deg_t) * norm_t

    # Accumulators for normal equations
    G = np.zeros((dimV, dimV))
    g = np.zeros((dimV, 1))

    # Accumulators for variance statistics
    if return_stats:
        n_samples_total = 0
        sum_psi = 0.0
        sum_psi2 = 0.0
        sum_bf = 0.0
        sum_bc = 0.0
        sum_bf2 = 0.0
        sum_bc2 = 0.0
        sum_bf_bc = 0.0
        all_psi = []

    if verbose:
        print(f"  Level {level}: deg={l_V}, dimV={dimV}, M={M_l}, batches={n_batches}")

    for b_idx in range(n_batches):
        B = min(batch_size, M_l - b_idx * batch_size)
        if B <= 0:
            break

        X0 = np.tile(x0, (B, 1))
        X_f = X0.copy()

        # Initial timestep
        trow0 = VT[0, :]
        s0 = X0 @ P1
        s0_scaled = np.clip(2.0 * (s0 - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
        VS0 = legvander(s0_scaled, deg_s) * norm_s
        
        D0 = np.empty((B, dimV), dtype=float)
        for p, (i_t, i_s) in enumerate(pairs):
            D0[:, p] = trow0[i_t] * VS0[:, i_s]
        psi0 = np.zeros((B, 1), dtype=float)
        
        G += D0.T @ D0
        g += D0.T @ psi0
        
        Z = np.random.randn(B, N_f, d)
        
        for n in range(1, N_c):
            # Two fine steps
            Z_n1 = Z[:, 2 * (n - 1), :]
            sigma = X_f * vol
            dW1 = (Z_n1 @ G_chol.T) * math.sqrt(hl_f)
            X_f = X_f + r * X_f * hl_f + sigma * dW1
            
            Z_n2 = Z[:, 2 * n - 1, :]
            sigma = X_f * vol
            dW2 = (Z_n2 @ G_chol.T) * math.sqrt(hl_f)
            X_f = X_f + r * X_f * hl_f + sigma * dW2
            
            # OT coupling: map fine to coarse in log-space
            log_X_f = np.log(X_f)
            log_X_c = maps[n](log_X_f)
            X_c = np.exp(log_X_c)
            
            # Build design matrix row block
            D_n = np.empty((B, dimV), dtype=float)
            trow = VT[n, :]
            s = X_f @ P1
            s_scaled = np.clip(2.0 * (s - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
            VS = legvander(s_scaled, deg_s) * norm_s
            
            for p, (i1, i2) in enumerate(pairs):
                D_n[:, p] = trow[i1] * VS[:, i2]
            
            # Telescoping difference with OT coupling
            w_sigma_f = X_f * vol * P1
            b_f = ((w_sigma_f @ cov_mat) * w_sigma_f).sum(axis=1)
            w_sigma_c = X_c * vol * P1
            b_c = ((w_sigma_c @ cov_mat) * w_sigma_c).sum(axis=1)
            psi_n = (b_f - b_c).reshape(-1, 1)

            if return_stats:
                diff = b_f - b_c
                n_samples_total += B
                sum_psi += np.sum(diff)
                sum_psi2 += np.sum(diff ** 2)
                sum_bf += np.sum(b_f)
                sum_bc += np.sum(b_c)
                sum_bf2 += np.sum(b_f ** 2)
                sum_bc2 += np.sum(b_c ** 2)
                sum_bf_bc += np.sum(b_f * b_c)
                all_psi.extend(diff.tolist())

            G += D_n.T @ D_n
            g += D_n.T @ psi_n

    # Solve normal equations via Cholesky
    G_reg = 0.5 * (G + G.T)
    
    lam = np.linalg.eigvalsh(G_reg)
    lam_min = max(lam[0], 1e-300)
    cond_approx = np.sqrt(lam[-1] / lam_min)
    
    if verbose:
        print(f"    Condition number ≈ {cond_approx:.2e}")
    
    if cond_approx > 1e12:
        reg = 1e-10 * lam[-1]
        G_reg += reg * np.eye(dimV)
        if verbose:
            print(f"    Added regularisation: {reg:.2e}")
    
    L = np.linalg.cholesky(G_reg)
    y = np.linalg.solve(L, g)
    c = np.linalg.solve(L.T, y).ravel()

    c_padded = np.zeros(dim_max)
    c_padded[:len(c)] = c

    if not return_stats:
        return c_padded

    # Compute level statistics
    n = n_samples_total
    if n > 1:
        mean_psi = sum_psi / n
        var_psi = (sum_psi2 / n - mean_psi ** 2) * n / (n - 1)

        mean_bf = sum_bf / n
        mean_bc = sum_bc / n
        var_bf = (sum_bf2 / n - mean_bf ** 2) * n / (n - 1)
        var_bc = (sum_bc2 / n - mean_bc ** 2) * n / (n - 1)

        cov_bf_bc = (sum_bf_bc / n - mean_bf * mean_bc) * n / (n - 1)
        std_bf = np.sqrt(max(var_bf, 1e-15))
        std_bc = np.sqrt(max(var_bc, 1e-15))
        if std_bf > 1e-10 and std_bc > 1e-10:
            corr = cov_bf_bc / (std_bf * std_bc)
            corr = np.clip(corr, -1.0, 1.0)
        else:
            corr = 1.0

        all_psi_arr = np.array(all_psi)
        if len(all_psi_arr) > 3:
            try:
                kurt = stats.kurtosis(all_psi_arr, fisher=True)
            except:
                kurt = 0.0
        else:
            kurt = 0.0
    else:
        mean_psi = sum_psi if n > 0 else 0.0
        var_psi = 0.0
        var_bf = 0.0
        var_bc = 0.0
        corr = 1.0
        kurt = 0.0

    level_stats = LevelStats(
        level=level,
        coefficients=c_padded,
        variance=max(var_psi, 0.0),
        mean_correction=mean_psi,
        correlation=corr,
        kurtosis=kurt,
        n_samples=n,
        var_fine=max(var_bf, 0.0),
        var_coarse=max(var_bc, 0.0)
    )

    return c_padded, level_stats


# =============================================================================
# SECTION 3: Telescoping Sum Aggregation with Flexible Allocation
# =============================================================================

def make_c_flexible(
    x0: np.ndarray, T: float, h0: float, r: float,
    cov_mat: np.ndarray, vol: np.ndarray, max_deg: int,
    P1: np.ndarray, s_min: float, s_max: float,
    C: int = 80, batch_size: int = 50,
    verbose: bool = False, return_stats: bool = False,
    formula: Union[str, Callable] = 'quadratic'
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, LevelStats]]]:
    """
    Compute full coefficient vector via MLMC with flexible allocation.
    
    Parameters
    ----------
    [Same as make_c, plus:]
    formula : str or callable
        Sample allocation formula.
    
    Returns
    -------
    c : ndarray
        Full coefficient vector for the volatility surface.
    level_stats : Dict[int, LevelStats] (only if return_stats=True)
        Statistics for each MLMC level.
    """
    if verbose:
        print(f"\nMLMC Coefficient Estimation (max_deg={max_deg}, formula={formula})")
        print("=" * 60)

    c_total = np.zeros(len(tot_degree_poly(max_deg)))
    level_stats_dict = {} if return_stats else None

    for level in range(max_deg + 1):
        if return_stats:
            c_l, stats = mlmc_level_flexible(
                x0, T, h0, level, r, cov_mat, vol, max_deg,
                P1, s_min, s_max, C, batch_size, verbose,
                return_stats=True, formula=formula
            )
            level_stats_dict[level] = stats
        else:
            c_l = mlmc_level_flexible(
                x0, T, h0, level, r, cov_mat, vol, max_deg,
                P1, s_min, s_max, C, batch_size, verbose,
                formula=formula
            )
        c_total += c_l

    if verbose:
        print("=" * 60)
        print("MLMC Aggregation Complete\n")

    if return_stats:
        return c_total, level_stats_dict
    return c_total


def make_c_ot_flexible(
    x0: np.ndarray, T: float, h0: float, r: float,
    cov_mat: np.ndarray, vol: np.ndarray, max_deg: int,
    P1: np.ndarray, s_min: float, s_max: float,
    C: int = 80, batch_size: int = 50,
    M_pilot: int = 500, verbose: bool = False,
    return_stats: bool = False,
    formula: Union[str, Callable] = 'quadratic'
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, LevelStats]]]:
    """
    Compute full coefficient vector via OT-enhanced MLMC with flexible allocation.
    
    Parameters
    ----------
    [Same as make_c_ot, plus:]
    formula : str or callable
        Sample allocation formula.
    
    Returns
    -------
    c : ndarray
        Full coefficient vector for the volatility surface.
    level_stats : Dict[int, LevelStats] (only if return_stats=True)
        Statistics for each MLMC level.
    """
    if verbose:
        print(f"\nMLMC+OT Coefficient Estimation (max_deg={max_deg}, formula={formula})")
        print("=" * 60)

    c_total = np.zeros(len(tot_degree_poly(max_deg)))
    level_stats_dict = {} if return_stats else None

    # Level 0: standard MLMC (no OT needed)
    if return_stats:
        c_0, stats_0 = mlmc_level_flexible(
            x0, T, h0, 0, r, cov_mat, vol, max_deg,
            P1, s_min, s_max, C, batch_size, verbose,
            return_stats=True, formula=formula
        )
        level_stats_dict[0] = stats_0
    else:
        c_0 = mlmc_level_flexible(
            x0, T, h0, 0, r, cov_mat, vol, max_deg,
            P1, s_min, s_max, C, batch_size, verbose,
            formula=formula
        )
    c_total += c_0

    # Levels 1+: OT-enhanced MLMC
    for level in range(1, max_deg + 1):
        if return_stats:
            c_l, stats = mlmc_level_ot_flexible(
                x0, T, h0, level, r, cov_mat, vol, max_deg,
                P1, s_min, s_max, C, batch_size, M_pilot, verbose,
                return_stats=True, formula=formula
            )
            level_stats_dict[level] = stats
        else:
            c_l = mlmc_level_ot_flexible(
                x0, T, h0, level, r, cov_mat, vol, max_deg,
                P1, s_min, s_max, C, batch_size, M_pilot, verbose,
                formula=formula
            )
        c_total += c_l

    if verbose:
        print("=" * 60)
        print("MLMC+OT Aggregation Complete\n")

    if return_stats:
        return c_total, level_stats_dict
    return c_total


# =============================================================================
# SECTION 4: High-Level Estimation Function
# =============================================================================

def estimate_volatility_mlmc_flexible(
    params,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    random_seed: int = 42,
    use_ot: bool = True,
    C: int = 80,
    batch_size: int = 50,
    M_pilot_domain: int = 10000,
    M_pilot_ot: int = 500,
    verbose: bool = True,
    formula: Union[str, Callable] = 'quadratic'
) -> VolatilitySurfaceResult:
    """
    Estimate volatility surface using MLMC with flexible sample allocation.
    
    This is the main entry point for comparing different allocation formulas.
    
    Parameters
    ----------
    params : ProblemParameters
        Problem configuration.
    t_grid : np.ndarray
        Time grid for evaluation.
    s_grid : np.ndarray
        Basket value grid for evaluation.
    random_seed : int
        Random seed for reproducibility.
    use_ot : bool
        If True, use Gaussian-Brenier OT coupling (default True).
    C : int
        Base sample size scaling factor (default 80).
    batch_size : int
        Batch size for memory-efficient processing (default 50).
    M_pilot_domain : int
        Number of pilot paths for domain estimation (default 10000).
    M_pilot_ot : int
        Number of pilot paths for OT map estimation (default 500).
    verbose : bool
        Print progress information.
    formula : str or callable
        Sample allocation formula. Options:
        - 'quadratic': M_l = C * dim(V)^2 (default, conservative)
        - 'log': M_l = C * dim(V) * log(dim(V)+1) (Cohen-Migliorati)
        - 'linear': M_l = C * dim(V) (minimum)
        - 'sqrt': M_l = C * dim(V)^1.5 (intermediate)
        - callable: Custom function f(dimV, C) -> int
    
    Returns
    -------
    VolatilitySurfaceResult
        Results including b²(t,S) surface, computation time, metadata.
    """
    np.random.seed(random_seed)
    start_time = time.perf_counter()
    
    method_name = f"MLMC+OT ({formula})" if use_ot else f"MLMC ({formula})"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{method_name} Volatility Surface Estimation")
        print(f"{'='*60}")
        print(f"  Assets: {params.d}, max_degree: {params.max_degree}")
        print(f"  OT coupling: {use_ot}")
        print(f"  Allocation formula: {formula}")
        
        # Show sample counts
        samples = compute_samples_for_formula(params.max_degree, C, formula)
        total = sum(samples.values())
        print(f"  Sample counts: {samples}")
        print(f"  Total samples: {total}")
    
    # Step 1: Estimate domain bounds via pilot run
    if verbose:
        print(f"\nStep 1: Domain estimation ({M_pilot_domain} pilot paths)...")
    
    s_min, s_max = estimate_domain(
        x0=params.x0,
        T=params.T,
        h0=params.h0,
        r=params.r,
        cov_mat=params.corr_matrix,
        vol=params.sigma,
        max_deg=params.max_degree,
        P1=params.P1,
        M_pilot=M_pilot_domain
    )
    
    # Add padding to domain bounds
    pad = (s_max - s_min) * 0.05
    s_min -= pad
    s_max += pad
    
    if verbose:
        print(f"    Domain: [{s_min:.1f}, {s_max:.1f}]")
    
    # Step 2: MLMC coefficient estimation
    if verbose:
        print(f"\nStep 2: {method_name} coefficient estimation...")
    
    if use_ot:
        coefficients = make_c_ot_flexible(
            x0=params.x0,
            T=params.T,
            h0=params.h0,
            r=params.r,
            cov_mat=params.corr_matrix,
            vol=params.sigma,
            max_deg=params.max_degree,
            P1=params.P1,
            s_min=s_min,
            s_max=s_max,
            C=C,
            batch_size=batch_size,
            M_pilot=M_pilot_ot,
            verbose=verbose,
            formula=formula
        )
    else:
        coefficients = make_c_flexible(
            x0=params.x0,
            T=params.T,
            h0=params.h0,
            r=params.r,
            cov_mat=params.corr_matrix,
            vol=params.sigma,
            max_deg=params.max_degree,
            P1=params.P1,
            s_min=s_min,
            s_max=s_max,
            C=C,
            batch_size=batch_size,
            verbose=verbose,
            formula=formula
        )
    
    if verbose:
        print(f"    Coefficients: {len(coefficients)} terms")
    
    # Step 3: Construct volatility surface
    pairs = tot_degree_poly(params.max_degree)
    b_surface = make_b_bar(coefficients, pairs, s_min, s_max, 
                           params.T, params.max_degree)
    b_squared_surface = make_b_squared(coefficients, pairs, s_min, s_max,
                                        params.T, params.max_degree)
    
    # Step 4: Evaluate on grid
    if verbose:
        print(f"\nStep 3: Evaluating on {len(t_grid)}×{len(s_grid)} grid...")
    
    b_squared_values = np.zeros((len(t_grid), len(s_grid)))
    for i, t in enumerate(t_grid):
        for j, s in enumerate(s_grid):
            b_squared_values[i, j] = b_squared_surface(t, s)
    
    computation_time = time.perf_counter() - start_time
    
    if verbose:
        print(f"\nCompleted in {computation_time:.2f}s")
        print(f"{'='*60}\n")
    
    # Compute actual samples used
    samples_used = compute_samples_for_formula(params.max_degree, C, formula)
    total_samples = sum(samples_used.values())
    
    return VolatilitySurfaceResult(
        b_surface=b_surface,
        b_squared_surface=b_squared_surface,
        t_grid=t_grid,
        s_grid=s_grid,
        b_squared_values=b_squared_values,
        method_name=method_name,
        computation_time=computation_time,
        parameters={
            "use_ot": use_ot,
            "max_degree": params.max_degree,
            "h0": params.h0,
            "C": C,
            "formula": str(formula),
            "samples_per_level": samples_used,
            "total_samples": total_samples,
            "M_pilot_domain": M_pilot_domain,
            "M_pilot_ot": M_pilot_ot if use_ot else None,
            "random_seed": random_seed,
        },
        coefficients=coefficients,
        n_samples=total_samples,
        mlmc_levels=params.max_degree + 1,
        domain_bounds=(s_min, s_max)
    )
