# This file is based on pdesolver.py from Amelie's work.
# JAX version with plotting conversions

"""
American Option PDE Solver using Implicit Backward Euler (JAX Version)

Solves the pricing PDE with early exercise constraint:
    ∂U/∂t + (1/2)b²S²∂²U/∂S² + rS∂U/∂S - rU = 0
    U(t, S) ≥ g(S)  for all t (early exercise)

Uses implicit backward Euler timestepping with projected volatility b(t, S)
from Markovian projection. The implicit scheme is unconditionally stable.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from JAX_finite_difference_operators import apply_pde_operator, compute_payoff


def solve_american_option(
    T, S_min, S_max, N_timesteps, N_spatial, r, K,
    option_type, volatility_surface, plot=False
):
    """
    Solve American option pricing PDE using backward Euler with early exercise.

    The backward Euler scheme at each timestep:
        U^n = U^{n+1} + dt * L(U^{n+1})
    then enforce early exercise:
        U^n = max(U^n, g(S))

    Parameters
    ----------
    T : float
        Maturity time
    S_min, S_max : float
        Spatial domain for asset price
    N_timesteps : int
        Number of timesteps (excluding initial time)
    N_spatial : int
        Number of spatial grid points
    r : float
        Risk-free interest rate
    K : float
        Strike price
    option_type : str
        Either 'put' or 'call'
    volatility_surface : callable
        Function b(t, S) returning volatility at (t, S).
        Should accept arrays and return arrays of same shape.
    plot : bool, default False
        Whether to generate 3D plots of solution and volatility

    Returns
    -------
    t_grid : jnp.ndarray, shape (N_timesteps + 1,)
        Time grid points from 0 to T
    S_grid : jnp.ndarray, shape (N_spatial,)
        Spatial grid points from S_min to S_max
    U : jnp.ndarray, shape (N_timesteps + 1, N_spatial)
        Option value function U(t, S) on grid
    b_grid : jnp.ndarray, shape (N_timesteps + 1, N_spatial)
        Volatility surface evaluated on grid
    payoff_grid : jnp.ndarray, shape (N_spatial,)
        Payoff function g(S) evaluated on spatial grid

    Notes
    -----
    - Backward Euler is unconditionally stable for diffusion problems
    - Early exercise enforced pointwise: U ← max(U, g) after each timestep
    - Boundary conditions: U(t, S_min) = g(S_min), U(t, S_max) = g(S_max)
    - Time complexity: O(N_timesteps * N_spatial²) for tridiagonal solve

    Examples
    --------
    >>> def constant_vol(t, S):
    ...     return 0.2 * jnp.ones_like(S)
    >>> t, S, U, b, g = solve_american_option(
    ...     T=1.0, S_min=50, S_max=150, N_timesteps=100, N_spatial=50,
    ...     r=0.05, K=100, option_type='put', volatility_surface=constant_vol
    ... )
    >>> print(f"Option value at t=0, S=100: {U[0, 25]:.4f}")
    """
    # Grid spacing
    dt = T / N_timesteps
    dS = (S_max - S_min) / (N_spatial - 1)

    # Create grids
    t_grid = jnp.linspace(0, T, N_timesteps + 1)
    S_grid = jnp.linspace(S_min, S_max, N_spatial)

    # Evaluate volatility surface on entire grid
    T_mesh, S_mesh = jnp.meshgrid(t_grid, S_grid, indexing='ij')
    b_grid = volatility_surface(T_mesh, S_mesh)

    # Check for reasonable volatility values
    if jnp.any(b_grid < 0):
        print(f"⚠️  Warning: Negative volatilities detected (min: {jnp.min(b_grid):.4f})")
    if jnp.any(b_grid > 2.0):
        print(f"⚠️  Warning: Extremely high volatilities detected (max: {jnp.max(b_grid):.4f})")

    # Compute payoff function
    payoff_grid = compute_payoff(S_grid, K, option_type)

    # Initialize value function
    U = jnp.zeros((N_timesteps + 1, N_spatial))

    # Boundary conditions (held constant across all time)
    U = U.at[:, 0].set(payoff_grid[0])    # At S_min
    U = U.at[:, -1].set(payoff_grid[-1])  # At S_max

    # Terminal condition at maturity
    U = U.at[-1, :].set(payoff_grid)

    # Backward timestepping (implicit scheme)
    for n in reversed(range(N_timesteps)):
        U_next = U[n + 1, :]                # Value at t_{n+1}
        b_current = b_grid[n, :]            # Volatility at t_n

        # Implicit backward Euler step: solve (I - dt*L) U^n = U^{n+1}
        U_continuation = apply_pde_operator(U_next, b_current, S_grid, r, dS, dt)

        # Enforce early exercise constraint
        U = U.at[n, :].set(jnp.maximum(U_continuation, payoff_grid))

    # Optional plotting
    if plot:
        _plot_solution(T_mesh, S_mesh, U, b_grid, K, option_type)

    return t_grid, S_grid, U, b_grid, payoff_grid


def _plot_solution(T_mesh, S_mesh, U, b_grid, K, option_type):
    """
    Internal function to plot option value and volatility surfaces.

    Parameters
    ----------
    T_mesh, S_mesh : jnp.ndarray
        Meshgrid arrays for time and space
    U : jnp.ndarray
        Option value function
    b_grid : jnp.ndarray
        Volatility surface
    K : float
        Strike price (for title)
    option_type : str
        'put' or 'call' (for title)
    """
    # Convert JAX arrays to numpy for matplotlib
    T_mesh_np = np.asarray(T_mesh)
    S_mesh_np = np.asarray(S_mesh)
    U_np = np.asarray(U)
    b_grid_np = np.asarray(b_grid)

    # Plot option value surface
    fig = plt.figure(figsize=(12, 5))

    # Subplot 1: Option value
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(T_mesh_np, S_mesh_np, U_np, cmap='viridis', alpha=0.9)
    ax1.set_title(f"American {option_type.capitalize()} Value (K={K})")
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Asset Price S")
    ax1.set_zlabel("Option Value U(t, S)")
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Subplot 2: Volatility surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(T_mesh_np, S_mesh_np, b_grid_np, cmap='plasma', alpha=0.9)
    ax2.set_title("Projected Volatility Surface b(t, S)")
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Asset Price S")
    ax2.set_zlabel("Volatility b")
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    plt.show()


def compute_exercise_boundary(t_grid, S_grid, U, payoff_grid, absolute_threshold=0.5):
    """
    Extract the early exercise boundary from the solution.

    The exercise boundary S*(t) is where the option transitions from the
    continuation region (U > g + threshold) to the exercise region (U ≈ g).

    Parameters
    ----------
    t_grid : jnp.ndarray, shape (N_timesteps + 1,)
        Time grid
    S_grid : jnp.ndarray, shape (N_spatial,)
        Spatial grid
    U : jnp.ndarray, shape (N_timesteps + 1, N_spatial)
        Option value function
    payoff_grid : jnp.ndarray, shape (N_spatial,)
        Payoff function g(S)
    absolute_threshold : float, default 0.5
        Absolute dollar threshold for time value

    Returns
    -------
    t_boundary : np.ndarray
        Time points where boundary is defined
    S_boundary : np.ndarray
        Asset prices at exercise boundary

    Notes
    -----
    Uses linear interpolation for sub-grid accuracy.
    Boundary is where time value crosses the threshold from below.
    """
    # Convert to numpy for processing
    t_grid_np = np.asarray(t_grid)
    S_grid_np = np.asarray(S_grid)
    U_np = np.asarray(U)
    payoff_grid_np = np.asarray(payoff_grid)

    t_boundary = []
    S_boundary = []

    for n, t in enumerate(t_grid_np):
        # Compute time value
        time_value = U_np[n, :] - payoff_grid_np

        # Find where time value exceeds threshold
        above_threshold = time_value > absolute_threshold

        if not np.any(above_threshold):
            # Exercise everywhere (boundary below domain)
            continue

        # Find first point above threshold (scanning from low S to high S)
        cross_idx = np.where(above_threshold)[0][0]

        # Skip if at edge
        if cross_idx == 0:
            continue

        # Linear interpolation between cross_idx-1 and cross_idx
        TV_before = time_value[cross_idx - 1]
        TV_after = time_value[cross_idx]
        S_before = S_grid_np[cross_idx - 1]
        S_after = S_grid_np[cross_idx]

        # Interpolate to find exact crossing
        if abs(TV_after - TV_before) > 1e-10:
            weight = (absolute_threshold - TV_before) / (TV_after - TV_before)
            S_star = S_before + weight * (S_after - S_before)
        else:
            S_star = S_before

        t_boundary.append(t)
        S_boundary.append(S_star)

    return np.array(t_boundary), np.array(S_boundary)


def check_stability_condition(dt, dS, b_max, r, S_max):
    """
    Check von Neumann stability condition for the discretisation.

    For explicit schemes, requires:
        dt ≤ dS² / (b²S² + r)

    Backward Euler is unconditionally stable, but this provides
    a guideline for reasonable timestep sizes.

    Parameters
    ----------
    dt : float
        Timestep size
    dS : float
        Spatial step size
    b_max : float
        Maximum volatility in domain
    r : float
        Risk-free rate
    S_max : float
        Maximum asset price

    Returns
    -------
    is_stable : bool
        True if stability condition satisfied
    stability_ratio : float
        dt / dt_max (should be < 1 for explicit schemes)
    """
    # Worst-case diffusion coefficient
    diffusion_coeff = b_max**2 * S_max**2 + r

    # Maximum stable timestep (for explicit schemes)
    dt_max = dS**2 / diffusion_coeff if diffusion_coeff > 0 else np.inf

    stability_ratio = dt / dt_max if dt_max > 0 else 0.0
    is_stable = stability_ratio <= 1.0

    return is_stable, stability_ratio
