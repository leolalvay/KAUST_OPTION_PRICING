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
    dS: float,
    dt: float = None
) -> np.ndarray:
    """
    Apply implicit backward Euler step for the Black-Scholes PDE.

    Solves (I - dt*L) U^n = U^{n+1} for U^n using Thomas algorithm,
    where L is the spatial operator:
        L(U) = (1/2)b²S²∂²U/∂S² + rS∂U/∂S - rU

    The finite difference stencil coefficients are:
        A_i = b²S²/(2ΔS²) + rS_i/(2ΔS)   (lower diagonal)
        B_i = r + b²S²/ΔS²                (main diagonal)
        C_i = b²S²/(2ΔS²) - rS_i/(2ΔS)   (upper diagonal)

    Note: These coefficients are sometimes denoted α, β, γ in numerical analysis literature.

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
    dt : float
        Time step size (required for implicit solve)

    Returns
    -------
    U_current : np.ndarray, shape (N_S,)
        Solution at current time step. Boundary points preserve input values.

    Notes
    -----
    - Uses Thomas algorithm (O(N)) for tridiagonal solve
    - Unconditionally stable for any dt
    - Boundary conditions are Dirichlet (fixed at input values)
    """
    # N_S = len(S_grid)

    # Precompute grid-dependent coefficients
    # Diffusion coefficient: (1/2) * b² * S² from Black-Scholes PDE
    diffusion_coeff = (b ** 2) * (S_grid ** 2)
    dS_squared = dS ** 2

    # Three-point stencil coefficients for L
    A = (diffusion_coeff / (2 * dS_squared)) + (r * S_grid) / (2 * dS)  # lower
    B = r + (diffusion_coeff / dS_squared)                                # main
    C = (diffusion_coeff / (2 * dS_squared)) - (r * S_grid) / (2 * dS)  # upper

    # Build tridiagonal system (I - dt*L) for interior points
    # L has: +A on lower, -B on main, +C on upper
    # So (I - dt*L) has: -dt*A on lower, 1+dt*B on main, -dt*C on upper
    # n_interior = N_S - 2

    # Diagonals for interior points (indices 1 to N_S-2)
    lower = -dt * A[2:-1]      # coefficients for U_{i-1}, length n_interior-1
    main = 1 + dt * B[1:-1]     # coefficients for U_i, length n_interior
    upper = -dt * C[1:-2]      # coefficients for U_{i+1}, length n_interior-1

    # Right-hand side: U_next at interior points
    rhs = U_next[1:-1].copy()

    # Add boundary contributions to RHS
    # At i=1: need to add dt*A[1]*U_next[0] (boundary term moves to RHS)
    rhs[0] += dt * A[1] * U_next[0]
    # At i=N_S-2: need to add dt*C[N_S-2]*U_next[N_S-1]
    rhs[-1] += dt * C[-2] * U_next[-1]

    # Thomas algorithm (tridiagonal solver)
    U_interior = thomas_algorithm(lower, main, upper, rhs)

    # Assemble full solution
    U_current = U_next.copy()
    U_current[1:-1] = U_interior

    return U_current


def thomas_algorithm(lower, main, upper, rhs):
    """
    Solve tridiagonal system using Thomas algorithm.

    Solves Ax = d where A is tridiagonal with:
    - lower: sub-diagonal (length n-1)
    - main: main diagonal (length n)
    - upper: super-diagonal (length n-1)
    - rhs: right-hand side (length n)

    Returns x (length n).
    """
    n = len(main)

    # Forward elimination
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    c_prime[0] = upper[0] / main[0]
    d_prime[0] = rhs[0] / main[0]

    for i in range(1, n - 1):
        denom = main[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    # Last row
    denom = main[n - 1] - lower[n - 2] * c_prime[n - 2]
    d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / denom

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = d_prime[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


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
