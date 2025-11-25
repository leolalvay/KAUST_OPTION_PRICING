# This file is based on utils.py from Amelie's work.
# JAX version with JIT optimization

"""
Finite Difference Operators for American Option PDE Solver (JAX Version)

Implements discretisation schemes for the pricing PDE:
    ∂U/∂t + (1/2)b²S²∂²U/∂S² + rS∂U/∂S - rU = 0

Using central differences in space with implicit backward Euler in time.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax


@jit
def thomas_algorithm(lower: jnp.ndarray, main: jnp.ndarray,
                     upper: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """
    Solve tridiagonal system using Thomas algorithm (JAX JIT-compatible).

    Solves Ax = d where A is tridiagonal with:
    - lower: sub-diagonal (length n-1)
    - main: main diagonal (length n)
    - upper: super-diagonal (length n-1)
    - rhs: right-hand side (length n)

    Returns x (length n).

    Note: Uses lax.fori_loop for JIT compatibility instead of Python loops.
    """
    n = len(main)

    # Forward elimination using lax.scan
    def forward_step(carry, i):
        c_prime_prev, d_prime_prev = carry
        # For i >= 1: compute new c_prime and d_prime
        denom = main[i] - lower[i - 1] * c_prime_prev
        c_prime_new = lax.cond(
            i < n - 1,
            lambda: upper[i] / denom,
            lambda: 0.0  # Not used for last element
        )
        d_prime_new = (rhs[i] - lower[i - 1] * d_prime_prev) / denom
        return (c_prime_new, d_prime_new), (c_prime_new, d_prime_new)

    # Initial values for i=0
    c_prime_0 = upper[0] / main[0]
    d_prime_0 = rhs[0] / main[0]

    # Run forward elimination for i = 1 to n-1
    _, (c_primes, d_primes) = lax.scan(
        forward_step,
        (c_prime_0, d_prime_0),
        jnp.arange(1, n)
    )

    # Combine initial with scanned values
    c_prime_all = jnp.concatenate([jnp.array([c_prime_0]), c_primes])
    d_prime_all = jnp.concatenate([jnp.array([d_prime_0]), d_primes])

    # Back substitution using lax.scan (reverse order)
    def backward_step(x_next, i):
        # i goes from n-2 down to 0
        idx = n - 2 - i
        x_current = d_prime_all[idx] - c_prime_all[idx] * x_next
        return x_current, x_current

    # Start with last element
    x_last = d_prime_all[n - 1]

    # Run backward substitution
    _, x_rest = lax.scan(backward_step, x_last, jnp.arange(n - 1))

    # Combine: x_rest is in reverse order (from x[n-2] to x[0])
    x = jnp.concatenate([x_rest[::-1], jnp.array([x_last])])

    return x


@jit
def apply_pde_operator(
    U_next: jnp.ndarray,
    b: jnp.ndarray,
    S_grid: jnp.ndarray,
    r: float,
    dS: float,
    dt: float
) -> jnp.ndarray:
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
    U_next : jnp.ndarray, shape (N_S,)
        Value function at next time step (t + Δt)
    b : jnp.ndarray, shape (N_S,)
        Projected volatility at current time step on spatial grid
    S_grid : jnp.ndarray, shape (N_S,)
        Spatial grid points for asset price
    r : float
        Risk-free interest rate
    dS : float
        Spatial grid spacing
    dt : float
        Time step size (required for implicit solve)

    Returns
    -------
    U_current : jnp.ndarray, shape (N_S,)
        Solution at current time step. Boundary points preserve input values.

    Notes
    -----
    - Uses Thomas algorithm (O(N)) for tridiagonal solve
    - Unconditionally stable for any dt
    - Boundary conditions are Dirichlet (fixed at input values)
    - JIT compiled for performance
    """
    N_S = len(S_grid)

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

    # Diagonals for interior points (indices 1 to N_S-2)
    lower = -dt * A[2:-1]      # coefficients for U_{i-1}, length n_interior-1
    main = 1 + dt * B[1:-1]     # coefficients for U_i, length n_interior
    upper = -dt * C[1:-2]      # coefficients for U_{i+1}, length n_interior-1

    # Right-hand side: U_next at interior points
    rhs = U_next[1:-1]

    # Add boundary contributions to RHS
    # At i=1: need to add dt*A[1]*U_next[0] (boundary term moves to RHS)
    # At i=N_S-2: need to add dt*C[N_S-2]*U_next[N_S-1]
    rhs = rhs.at[0].add(dt * A[1] * U_next[0])
    rhs = rhs.at[-1].add(dt * C[-2] * U_next[-1])

    # Thomas algorithm (tridiagonal solver)
    U_interior = thomas_algorithm(lower, main, upper, rhs)

    # Assemble full solution (preserve boundary values)
    U_current = U_next.at[1:-1].set(U_interior)

    return U_current


def compute_payoff(S: jnp.ndarray, K: float, option_type: str = "put") -> jnp.ndarray:
    """
    Compute option payoff at maturity or for early exercise constraint.

    For American options, the payoff serves dual purpose:
        1. Terminal condition: U(T, S) = g(S)
        2. Early exercise constraint: U(t, S) ≥ g(S) for all t < T

    Parameters
    ----------
    S : jnp.ndarray
        Asset price(s) at which to evaluate payoff. Can be scalar or array.
    K : float
        Strike price
    option_type : str, default "put"
        Either "put" or "call"

    Returns
    -------
    payoff : jnp.ndarray
        Payoff value(s), same shape as S

    Examples
    --------
    >>> S = jnp.array([90, 100, 110])
    >>> compute_payoff(S, K=100, option_type="put")
    array([10.,  0.,  0.])

    >>> compute_payoff(S, K=100, option_type="call")
    array([ 0.,  0., 10.])
    """
    if option_type == "put":
        return jnp.maximum(K - S, 0.0)
    elif option_type == "call":
        return jnp.maximum(S - K, 0.0)
    else:
        raise ValueError(
            f"Invalid option_type '{option_type}'. Must be 'put' or 'call'."
        )
