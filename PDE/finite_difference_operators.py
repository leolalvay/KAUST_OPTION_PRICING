# This file is based on utils.py from Amelie's work.

"""
Finite Difference Operators for American Option PDE Solver

Implements discretisation schemes for the pricing PDE:
    ∂U/∂t + (1/2)b²∂²U/∂S² + rS∂U/∂S - rU = 0

Using central differences in space with backward Euler in time.
"""

import numpy as np


def apply_pde_operator(
    U_next: np.ndarray,
    b: np.ndarray,
    S_grid: np.ndarray,
    r: float,
    dS: float
) -> np.ndarray:
    """
    Apply the discretised PDE operator L to the value function.
    
    Computes L(U) = (1/2)b²∂²U/∂S² + rS∂U/∂S - rU using central differences.
    
    The finite difference stencil at interior point i is:
        L(U_i) = A_i * U_{i-1} - B_i * U_i + C_i * U_{i+1}
    
    where:
        A_i = b²/(2ΔS²) + rS_i/(2ΔS)   (backward contribution)
        B_i = r + b²/ΔS²                (diagonal term)
        C_i = b²/(2ΔS²) - rS_i/(2ΔS)   (forward contribution)
    
    Parameters
    ----------
    U_next : np.ndarray, shape (N_S,)
        Value function at next time step (t + Δt)
    b : np.ndarray, shape (N_S,)
        Projected volatility at current time step on spatial grid
    S_grid : np.ndarray, shape (N_S,)
        Spatial grid points for asset price
    r : float
        Risk-free interest rate
    dS : float
        Spatial grid spacing
        
    Returns
    -------
    L_U : np.ndarray, shape (N_S,)
        Result of operator application. Boundary points (0, -1) are set to zero
        as they are handled separately via boundary conditions.
        
    Notes
    -----
    - This discretisation is consistent with backward Euler timestepping
    - The operator is applied to U at t_{n+1} to solve for U at t_n
    - Stability is unconditional for backward Euler with diffusion
    """
    N_S = len(S_grid)
    L_U = np.zeros_like(U_next)
    
    # Precompute grid-dependent coefficients
    b_squared = b ** 2
    dS_squared = dS ** 2
    
    # Three-point stencil coefficients
    A = (b_squared / (2 * dS_squared)) + (r * S_grid) / (2 * dS)
    B = r + (b_squared / dS_squared)
    C = (b_squared / (2 * dS_squared)) - (r * S_grid) / (2 * dS)
    
    # Apply stencil at interior points (boundaries handled externally)
    L_U[1:-1] = (
        A[1:-1] * U_next[0:-2] -    # Backward difference contribution
        B[1:-1] * U_next[1:-1] +     # Diagonal term
        C[1:-1] * U_next[2:]         # Forward difference contribution
    )
    
    return L_U


def compute_payoff(S: np.ndarray, K: float, option_type: str = "put") -> np.ndarray:
    """
    Compute option payoff at maturity or for early exercise constraint.
    
    For American options, the payoff serves dual purpose:
        1. Terminal condition: U(T, S) = g(S)
        2. Early exercise constraint: U(t, S) ≥ g(S) for all t < T
    
    Parameters
    ----------
    S : np.ndarray
        Asset price(s) at which to evaluate payoff. Can be scalar or array.
    K : float
        Strike price
    option_type : str, default "put"
        Either "put" or "call"
        
    Returns
    -------
    payoff : np.ndarray
        Payoff value(s), same shape as S
        
    Examples
    --------
    >>> S = np.array([90, 100, 110])
    >>> compute_payoff(S, K=100, option_type="put")
    array([10.,  0.,  0.])
    
    >>> compute_payoff(S, K=100, option_type="call")
    array([ 0.,  0., 10.])
    """
    if option_type == "put":
        return np.maximum(K - S, 0.0)
    elif option_type == "call":
        return np.maximum(S - K, 0.0)
    else:
        raise ValueError(
            f"Invalid option_type '{option_type}'. Must be 'put' or 'call'."
        )
