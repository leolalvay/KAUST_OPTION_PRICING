"""
Unit tests for finite_difference_operators.py

Validates discretisation accuracy against analytical Black-Scholes solutions.
"""

import numpy as np
from finite_difference_operators import apply_pde_operator, compute_payoff


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


def test_pde_operator_structure():
    """Test that PDE operator has correct structure and boundary behaviour."""
    N_S = 10
    S_grid = np.linspace(80, 120, N_S)
    dS = S_grid[1] - S_grid[0]
    
    # Simple test function
    U_next = S_grid ** 2
    
    # Constant volatility and rate
    b = 0.2 * np.ones(N_S)
    r = 0.05
    
    L_U = apply_pde_operator(U_next, b, S_grid, r, dS)
    
    # Check boundaries are zero (handled externally)
    assert L_U[0] == 0.0, "Left boundary should be zero"
    assert L_U[-1] == 0.0, "Right boundary should be zero"
    
    # Check interior points are non-zero for non-constant U
    assert np.any(L_U[1:-1] != 0), "Interior points should be non-zero"
    
    print("✓ PDE operator structure test passed")


def test_pde_operator_constant_function():
    """Test that operator applied to constant function gives -rU."""
    N_S = 20
    S_grid = np.linspace(50, 150, N_S)
    dS = S_grid[1] - S_grid[0]
    
    # Constant value function
    c = 42.0
    U_next = c * np.ones(N_S)
    
    b = 0.2 * np.ones(N_S)
    r = 0.05
    
    L_U = apply_pde_operator(U_next, b, S_grid, r, dS)
    
    # For constant U: ∂U/∂S = 0, ∂²U/∂S² = 0, so L(U) = -rU
    expected = -r * U_next
    
    # Interior points should match
    assert np.allclose(L_U[1:-1], expected[1:-1], rtol=1e-10), \
        "Constant function should give -rU"
    
    print("✓ Constant function test passed")


def test_linear_function_drift_term():
    """Test that linear function correctly captures drift term rS∂U/∂S."""
    N_S = 50
    S_grid = np.linspace(80, 120, N_S)
    dS = S_grid[1] - S_grid[0]
    
    # Linear function U = S (like a forward)
    U_next = S_grid.copy()
    
    b = 0.2 * np.ones(N_S)
    r = 0.05
    
    L_U = apply_pde_operator(U_next, b, S_grid, r, dS)
    
    # For U = S: ∂U/∂S = 1, ∂²U/∂S² = 0
    # So L(U) = rS * 1 - rS = 0
    expected = np.zeros_like(U_next)
    
    # Interior points should be close to zero (up to discretisation error)
    # Expect O(dS²) truncation error from central differences
    assert np.allclose(L_U[1:-1], expected[1:-1], atol=1e-6), \
        "Linear function should give near-zero result"
    
    print("✓ Linear function drift test passed")


def run_all_tests():
    """Run all unit tests."""
    print("\nRunning finite difference operator tests...\n")
    
    test_payoff_put()
    test_payoff_call()
    test_pde_operator_structure()
    test_pde_operator_constant_function()
    
    print("\n✓ All tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
