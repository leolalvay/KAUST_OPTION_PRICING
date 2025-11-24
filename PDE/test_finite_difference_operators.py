"""
Unit tests for finite_difference_operators.py

Validates the implicit backward Euler scheme and payoff calculations.
"""

import numpy as np
from finite_difference_operators import apply_pde_operator, compute_payoff, thomas_algorithm


def test_payoff_put():
    """Test put option payoff calculation."""
    S = np.array([80, 90, 100, 110, 120])
    K = 100
    expected = np.array([20, 10, 0, 0, 0])

    payoff = compute_payoff(S, K, option_type="put")

    assert np.allclose(payoff, expected), f"Expected {expected}, got {payoff}"
    print("✓ Put payoff test passed")


def test_payoff_call():
    """Test call option payoff calculation."""
    S = np.array([80, 90, 100, 110, 120])
    K = 100
    expected = np.array([0, 0, 0, 10, 20])

    payoff = compute_payoff(S, K, option_type="call")

    assert np.allclose(payoff, expected), f"Expected {expected}, got {payoff}"
    print("✓ Call payoff test passed")


def test_thomas_algorithm():
    """Test Thomas algorithm against numpy solve."""
    n = 10

    # Random tridiagonal system
    np.random.seed(42)
    main = 4.0 + np.random.rand(n)  # Diagonally dominant
    lower = np.random.rand(n - 1)
    upper = np.random.rand(n - 1)
    rhs = np.random.rand(n)

    # Solve with Thomas algorithm
    x_thomas = thomas_algorithm(lower, main, upper, rhs)

    # Build full matrix and solve with numpy
    A = np.diag(main) + np.diag(lower, -1) + np.diag(upper, 1)
    x_numpy = np.linalg.solve(A, rhs)

    assert np.allclose(x_thomas, x_numpy, rtol=1e-10), \
        f"Thomas algorithm mismatch: max diff = {np.max(np.abs(x_thomas - x_numpy))}"

    print("✓ Thomas algorithm test passed")


def test_implicit_step_preserves_boundaries():
    """Test that implicit step preserves boundary values."""
    N_S = 20
    S_grid = np.linspace(50, 150, N_S)
    dS = S_grid[1] - S_grid[0]
    dt = 0.01

    # Initial condition with specific boundary values
    U_next = np.maximum(100 - S_grid, 0)  # Put payoff

    b = 0.2 * np.ones(N_S)
    r = 0.05

    U_current = apply_pde_operator(U_next, b, S_grid, r, dS, dt)

    # Boundary values should be preserved
    assert U_current[0] == U_next[0], "Left boundary should be preserved"
    assert U_current[-1] == U_next[-1], "Right boundary should be preserved"

    print("✓ Boundary preservation test passed")


def test_implicit_step_stability():
    """Test that implicit scheme is stable even with large dt."""
    N_S = 50
    S_grid = np.linspace(50, 150, N_S)
    dS = S_grid[1] - S_grid[0]

    # Use very large dt that would be unstable for explicit scheme
    dt = 1.0  # Much larger than explicit stability limit

    U_next = np.maximum(100 - S_grid, 0)  # Put payoff
    b = 0.3 * np.ones(N_S)  # High volatility
    r = 0.05

    U_current = apply_pde_operator(U_next, b, S_grid, r, dS, dt)

    # Solution should remain bounded (no explosion)
    assert np.all(np.isfinite(U_current)), "Solution should be finite"
    assert np.all(U_current >= -1e-10), "Solution should be non-negative (within tolerance)"
    assert np.max(U_current) < 1000, "Solution should not explode"

    print("✓ Implicit stability test passed")


def test_implicit_step_convergence():
    """Test that implicit scheme converges as dt -> 0."""
    N_S = 100
    S_grid = np.linspace(50, 150, N_S)
    dS = S_grid[1] - S_grid[0]

    U_next = np.maximum(100 - S_grid, 0)  # Put payoff
    b = 0.2 * np.ones(N_S)
    r = 0.05

    # Solve with different dt values
    dt_values = [0.1, 0.01, 0.001]
    solutions = []

    for dt in dt_values:
        U_current = apply_pde_operator(U_next, b, S_grid, r, dS, dt)
        solutions.append(U_current.copy())

    # Check convergence: differences should decrease
    diff1 = np.max(np.abs(solutions[1] - solutions[0]))
    diff2 = np.max(np.abs(solutions[2] - solutions[1]))

    assert diff2 < diff1, f"Solution should converge: diff1={diff1:.6f}, diff2={diff2:.6f}"

    print("✓ Implicit convergence test passed")


def test_volatility_sensitivity():
    """Test that solution is sensitive to volatility (the original bug check)."""
    N_S = 50
    S_grid = np.linspace(50, 150, N_S)
    dS = S_grid[1] - S_grid[0]
    dt = 0.01

    U_next = np.maximum(100 - S_grid, 0)  # Put payoff
    r = 0.05

    # Low volatility
    b_low = 0.1 * np.ones(N_S)
    U_low = apply_pde_operator(U_next, b_low, S_grid, r, dS, dt)

    # High volatility
    b_high = 0.4 * np.ones(N_S)
    U_high = apply_pde_operator(U_next, b_high, S_grid, r, dS, dt)

    # Solutions should be different (not identical like the old bug)
    diff = np.max(np.abs(U_high - U_low))
    assert diff > 0.01, f"Solutions should differ with volatility: max diff = {diff}"

    print(f"✓ Volatility sensitivity test passed (max diff = {diff:.4f})")


def run_all_tests():
    """Run all unit tests."""
    print("\nRunning finite difference operator tests...\n")

    test_payoff_put()
    test_payoff_call()
    test_thomas_algorithm()
    test_implicit_step_preserves_boundaries()
    test_implicit_step_stability()
    test_implicit_step_convergence()
    test_volatility_sensitivity()

    print("\n✅ All finite difference operator tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
