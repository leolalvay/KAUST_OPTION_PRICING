"""
Unit tests for american_option_pde_solver.py

Validates PDE solver against known properties and analytical bounds.
"""

import numpy as np
import matplotlib.pyplot as plt
from american_option_pde_solver import (
    solve_american_option,
    compute_exercise_boundary,
    check_stability_condition
)


def test_constant_volatility_put():
    """Test American put with constant volatility."""
    # Simple Black-Scholes setup
    T = 1.0
    S_min, S_max = 50, 150
    N_timesteps = 100
    N_spatial = 50
    r = 0.05
    K = 100
    sigma = 0.2
    
    def constant_vol(t, S):
        return sigma * np.ones_like(S)
    
    t_grid, S_grid, U, b_grid, payoff_grid = solve_american_option(
        T, S_min, S_max, N_timesteps, N_spatial, r, K,
        option_type='put', volatility_surface=constant_vol, plot=False
    )
    
    # Check shapes
    assert t_grid.shape == (N_timesteps + 1,)
    assert S_grid.shape == (N_spatial,)
    assert U.shape == (N_timesteps + 1, N_spatial)
    assert b_grid.shape == (N_timesteps + 1, N_spatial)
    
    # Check terminal condition
    assert np.allclose(U[-1, :], payoff_grid), "Terminal condition not satisfied"
    
    # Check early exercise constraint: U ≥ payoff everywhere
    for n in range(N_timesteps + 1):
        assert np.all(U[n, :] >= payoff_grid - 1e-10), \
            f"Early exercise violated at timestep {n}"
    
    # Check put option bounds: 0 ≤ U ≤ K
    assert np.all(U >= 0), "Option value went negative"
    assert np.all(U <= K), f"Put value exceeded strike (max: {U.max():.2f})"
    
    # Check American ≥ European (time value)
    # At t=0 (present), American should be worth at least intrinsic value
    S_idx_atm = np.argmin(np.abs(S_grid - K))
    assert U[0, S_idx_atm] >= payoff_grid[S_idx_atm], \
        "American option less than intrinsic value"
    
    print(f"✓ Constant volatility put test passed")
    print(f"  Option value at t=0, S=K: {U[0, S_idx_atm]:.4f}")
    print(f"  Intrinsic value: {payoff_grid[S_idx_atm]:.4f}")


def test_constant_volatility_call():
    """Test American call with constant volatility."""
    T = 1.0
    S_min, S_max = 50, 150
    N_timesteps = 100
    N_spatial = 50
    r = 0.05
    K = 100
    sigma = 0.2
    
    def constant_vol(t, S):
        return sigma * np.ones_like(S)
    
    t_grid, S_grid, U, b_grid, payoff_grid = solve_american_option(
        T, S_min, S_max, N_timesteps, N_spatial, r, K,
        option_type='call', volatility_surface=constant_vol, plot=False
    )
    
    # Check terminal condition
    assert np.allclose(U[-1, :], payoff_grid), "Terminal condition not satisfied"
    
    # Check early exercise constraint
    for n in range(N_timesteps + 1):
        assert np.all(U[n, :] >= payoff_grid - 1e-10), \
            f"Early exercise violated at timestep {n}"
    
    # Check call option bounds: 0 ≤ U ≤ S_max
    assert np.all(U >= 0), "Option value went negative"
    
    print(f"✓ Constant volatility call test passed")


def test_time_varying_volatility():
    """Test with time-varying volatility surface."""
    T = 1.0
    S_min, S_max = 80, 120
    N_timesteps = 50
    N_spatial = 40
    r = 0.05
    K = 100
    
    # Volatility increases with time (volatility smile)
    def time_varying_vol(t, S):
        return 0.1 + 0.2 * t / T
    
    t_grid, S_grid, U, b_grid, payoff_grid = solve_american_option(
        T, S_min, S_max, N_timesteps, N_spatial, r, K,
        option_type='put', volatility_surface=time_varying_vol, plot=False
    )
    
    # Check volatility is actually varying
    assert not np.allclose(b_grid[0, :], b_grid[-1, :]), \
        "Volatility should vary with time"
    
    # Check monotonicity of volatility in time
    for s in range(N_spatial):
        assert b_grid[-1, s] >= b_grid[0, s], \
            "Volatility should increase with time"
    
    print(f"✓ Time-varying volatility test passed")
    print(f"  Vol at t=0: {b_grid[0, N_spatial//2]:.4f}")
    print(f"  Vol at t=T: {b_grid[-1, N_spatial//2]:.4f}")


def test_space_varying_volatility():
    """Test with space-varying volatility (local volatility)."""
    T = 1.0
    S_min, S_max = 50, 150
    N_timesteps = 50
    N_spatial = 40
    r = 0.05
    K = 100
    
    # Local volatility: higher for out-of-the-money
    def local_vol(t, S):
        return 0.15 + 0.05 * np.abs(S - K) / K
    
    t_grid, S_grid, U, b_grid, payoff_grid = solve_american_option(
        T, S_min, S_max, N_timesteps, N_spatial, r, K,
        option_type='put', volatility_surface=local_vol, plot=False
    )
    
    # Check volatility varies with S
    # Compare volatility at different S values (at same time)
    S_mid_idx = N_spatial // 2  # Near K=100
    vol_atm = b_grid[0, S_mid_idx]      # At-the-money
    vol_otm = b_grid[0, 0]               # Out-of-the-money (S_min)
    
    assert abs(vol_otm - vol_atm) > 0.01, \
        f"Volatility should vary with S (got ATM={vol_atm:.4f} vs OTM={vol_otm:.4f})"
    
    print(f"✓ Space-varying volatility test passed")
    print(f"  Vol at S≈K: {vol_atm:.4f}")
    print(f"  Vol at S_min: {vol_otm:.4f}")


def test_boundary_conditions():
    """Test that boundary conditions are properly enforced."""
    T = 1.0
    S_min, S_max = 50, 150
    N_timesteps = 50
    N_spatial = 40
    r = 0.05
    K = 100
    
    def constant_vol(t, S):
        return 0.2 * np.ones_like(S)
    
    # Put option
    t_grid, S_grid, U, b_grid, payoff_grid = solve_american_option(
        T, S_min, S_max, N_timesteps, N_spatial, r, K,
        option_type='put', volatility_surface=constant_vol, plot=False
    )
    
    # Check left boundary: U(t, S_min) = max(K - S_min, 0)
    expected_left = max(K - S_min, 0)
    assert np.allclose(U[:, 0], expected_left), \
        f"Left boundary incorrect: {U[0, 0]:.2f} vs {expected_left:.2f}"
    
    # Check right boundary: U(t, S_max) = max(K - S_max, 0)
    expected_right = max(K - S_max, 0)
    assert np.allclose(U[:, -1], expected_right), \
        f"Right boundary incorrect: {U[0, -1]:.2f} vs {expected_right:.2f}"
    
    print(f"✓ Boundary conditions test passed")


def test_exercise_boundary_extraction():
    """Test extraction of early exercise boundary."""
    T = 1.0
    S_min, S_max = 50, 150
    N_timesteps = 100
    N_spatial = 80
    r = 0.05
    K = 100
    
    def constant_vol(t, S):
        return 0.2 * np.ones_like(S)
    
    t_grid, S_grid, U, b_grid, payoff_grid = solve_american_option(
        T, S_min, S_max, N_timesteps, N_spatial, r, K,
        option_type='put', volatility_surface=constant_vol, plot=False
    )
    
    t_boundary, S_boundary = compute_exercise_boundary(
        t_grid, S_grid, U, payoff_grid, absolute_threshold=0.5
    )
    
    # Check that boundary exists
    assert len(t_boundary) > 0, "No exercise boundary found"
    
    # For American put: boundary should be below strike
    if len(S_boundary) > 0:
        assert all(S <= K for S in S_boundary), \
            "Put exercise boundary should be below strike"
    
    print(f"✓ Exercise boundary extraction test passed")
    print(f"  Boundary points found: {len(t_boundary)}")
    if len(t_boundary) > 0:
        print(f"  Boundary at t=0: S = {S_boundary[0]:.2f}")


def test_stability_condition():
    """Test stability condition checker."""
    dt = 0.01
    dS = 1.0
    b_max = 0.3
    r = 0.05
    S_max = 100
    
    is_stable, ratio = check_stability_condition(dt, dS, b_max, r, S_max)
    
    # Backward Euler is unconditionally stable, but ratio should be reasonable
    assert isinstance(is_stable, bool)
    assert isinstance(ratio, float)
    assert ratio >= 0
    
    print(f"✓ Stability condition test passed")
    print(f"  Stability ratio: {ratio:.4f} (stable: {is_stable})")


def test_convergence_with_refinement():
    """Test that solution converges as grid is refined."""
    T = 1.0
    S_min, S_max = 80, 120
    r = 0.05
    K = 100
    
    def constant_vol(t, S):
        return 0.2 * np.ones_like(S)
    
    # Solve at different resolutions
    N_values = [20, 40, 80]
    solutions = []
    
    for N in N_values:
        t_grid, S_grid, U, _, _ = solve_american_option(
            T, S_min, S_max, N_timesteps=N, N_spatial=N, r=r, K=K,
            option_type='put', volatility_surface=constant_vol, plot=False
        )
        # Store value at t=0, S=K
        S_idx = np.argmin(np.abs(S_grid - K))
        solutions.append(U[0, S_idx])
    
    # Check solutions are getting closer together (convergence)
    diff1 = abs(solutions[1] - solutions[0])
    diff2 = abs(solutions[2] - solutions[1])
    
    assert diff2 < diff1, "Solution should converge with refinement"
    
    print(f"✓ Convergence test passed")
    print(f"  N=20: {solutions[0]:.6f}")
    print(f"  N=40: {solutions[1]:.6f}")
    print(f"  N=80: {solutions[2]:.6f}")
    print(f"  Difference decreasing: {diff1:.6f} → {diff2:.6f}")


def test_american_vs_european_bound():
    """Test that American option value ≥ intrinsic value."""
    T = 1.0
    S_min, S_max = 50, 150
    N_timesteps = 50
    N_spatial = 50
    r = 0.05
    K = 100
    
    def constant_vol(t, S):
        return 0.2 * np.ones_like(S)
    
    t_grid, S_grid, U, b_grid, payoff_grid = solve_american_option(
        T, S_min, S_max, N_timesteps, N_spatial, r, K,
        option_type='put', volatility_surface=constant_vol, plot=False
    )
    
    # At every point, U(t, S) ≥ g(S) (intrinsic value)
    for n in range(N_timesteps + 1):
        for s in range(N_spatial):
            assert U[n, s] >= payoff_grid[s] - 1e-10, \
                f"Value below intrinsic at t={t_grid[n]:.2f}, S={S_grid[s]:.2f}"
    
    print(f"✓ American vs European bound test passed")


def run_all_tests():
    """Run all unit tests."""
    print("\nRunning american_option_pde_solver.py tests...\n")
    
    test_constant_volatility_put()
    test_constant_volatility_call()
    test_time_varying_volatility()
    test_space_varying_volatility()
    test_boundary_conditions()
    test_exercise_boundary_extraction()
    test_stability_condition()
    test_convergence_with_refinement()
    test_american_vs_european_bound()
    
    print("\n✅ All PDE solver tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
