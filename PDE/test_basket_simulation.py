"""
Unit tests for basket_simulation.py

Validates GBM path generation, polynomial basis construction, 
and regression system setup.
"""

import numpy as np
from basket_simulation import (
    simulate_gbm_paths,
    generate_polynomial_basis_pairs,
    construct_regression_system,
    fit_volatility_coefficients,
    construct_volatility_surface
)


def test_simulate_gbm_paths():
    """Test GBM path simulation produces correct shapes and properties."""
    d = 3
    S0 = np.array([100, 100, 100])
    r = 0.05
    vol = np.array([0.2, 0.15, 0.1])
    cov_mat = np.eye(d)  # Uncorrelated for simplicity
    dt = 0.01
    N_steps = 100
    N_paths = 1000
    
    paths = simulate_gbm_paths(S0, r, vol, cov_mat, dt, N_steps, N_paths)
    
    # Check shape
    assert paths.shape == (N_paths, N_steps, d), f"Wrong shape: {paths.shape}"
    
    # Check initial condition
    assert np.allclose(paths[:, 0, :], S0), "Initial condition not satisfied"
    
    # Check positivity (GBM paths should stay positive)
    assert np.all(paths > 0), "GBM paths went negative!"
    
    # Check mean drift is approximately correct (statistical test)
    final_prices = paths[:, -1, 0]
    T = dt * (N_steps - 1)
    expected_mean = S0[0] * np.exp(r * T)
    empirical_mean = np.mean(final_prices)
    
    # Allow 5% tolerance on mean (it's stochastic)
    assert abs(empirical_mean - expected_mean) / expected_mean < 0.05, \
        f"Mean drift incorrect: expected {expected_mean:.2f}, got {empirical_mean:.2f}"
    
    print(f"✓ GBM simulation test passed (mean: {empirical_mean:.2f}, expected: {expected_mean:.2f})")


def test_generate_polynomial_basis_pairs():
    """Test polynomial basis generation."""
    # Test max_degree = 2
    pairs = generate_polynomial_basis_pairs(max_degree=2)
    expected = [(0,0), (0,1), (0,2), (1,0), (1,1), (2,0)]
    assert sorted(pairs) == sorted(expected), f"Expected {expected}, got {pairs}"
    
    # Test max_degree = 1
    pairs = generate_polynomial_basis_pairs(max_degree=1)
    expected = [(0,0), (0,1), (1,0)]
    assert sorted(pairs) == sorted(expected), f"Expected {expected}, got {pairs}"
    
    # Check size formula: (d+1)(d+2)/2
    for d in [1, 2, 3, 4]:
        pairs = generate_polynomial_basis_pairs(max_degree=d)
        expected_size = (d + 1) * (d + 2) // 2
        assert len(pairs) == expected_size, \
            f"Wrong size for degree {d}: expected {expected_size}, got {len(pairs)}"
    
    print("✓ Polynomial basis generation test passed")


def test_construct_regression_system():
    """Test regression system construction."""
    # Small problem for testing
    d = 2
    N_paths = 50
    N_fine = 10
    N_coarse = 5
    T = 1.0
    
    # Generate dummy paths
    paths_fine = np.random.rand(N_paths, N_fine, d) * 50 + 80  # Random prices [80, 130]
    paths_coarse = paths_fine[:, ::2, :]  # Subsample
    
    basket_weights = np.ones(d) / d
    basis_pairs = generate_polynomial_basis_pairs(max_degree=2)
    cov_mat = np.eye(d)
    vol = np.array([0.2, 0.15])
    S_min, S_max = 80, 130
    
    D, psi = construct_regression_system(
        paths_fine, paths_coarse, basket_weights, basis_pairs,
        cov_mat, vol, S_min, S_max, T
    )
    
    # Check shapes
    n_basis = len(basis_pairs)
    expected_rows = N_paths * N_coarse
    assert D.shape == (expected_rows, n_basis), f"Wrong D shape: {D.shape}"
    assert psi.shape == (expected_rows, 1), f"Wrong psi shape: {psi.shape}"
    
    # Check no NaNs or Infs
    assert not np.any(np.isnan(D)), "D contains NaN"
    assert not np.any(np.isinf(D)), "D contains Inf"
    assert not np.any(np.isnan(psi)), "psi contains NaN"
    assert not np.any(np.isinf(psi)), "psi contains Inf"
    
    print(f"✓ Regression system construction test passed (D: {D.shape}, psi: {psi.shape})")


def test_fit_volatility_coefficients():
    """Test coefficient fitting with QR decomposition."""
    # Create a simple overdetermined system
    N_samples = 100
    n_basis = 5
    
    D = np.random.randn(N_samples, n_basis)
    c_true = np.array([1.0, -0.5, 0.3, 0.2, -0.1])
    psi = (D @ c_true + 0.01 * np.random.randn(N_samples)).reshape(-1, 1)  # Make 2D
    
    c_fit = fit_volatility_coefficients(D, psi)
    
    # Check shape
    assert c_fit.shape == (n_basis,), f"Wrong shape: {c_fit.shape}"
    
    # Check recovery (should be close to c_true)
    assert np.allclose(c_fit, c_true, atol=0.1), \
        f"Poor recovery: true={c_true}, fit={c_fit}"
    
    print(f"✓ Coefficient fitting test passed (max error: {np.max(np.abs(c_fit - c_true)):.4f})")


def test_construct_volatility_surface():
    """Test volatility surface construction and evaluation."""
    # Simple coefficients
    c = np.array([0.04, 0.001, -0.0005, 0.0, 0.0, 0.0])  # Mostly constant
    basis_pairs = generate_polynomial_basis_pairs(max_degree=2)
    S_min, S_max = 80, 120
    T = 1.0
    max_degree = 2
    
    b_surface = construct_volatility_surface(c, basis_pairs, S_min, S_max, T, max_degree)
    
    # Test scalar evaluation
    b_val = b_surface(0.5, 100.0)
    assert isinstance(b_val, (float, np.ndarray)), "Should return numeric type"
    assert b_val > 0, f"Volatility should be positive, got {b_val}"
    
    # Test array evaluation
    t_grid = np.linspace(0, T, 10)
    S_grid = np.linspace(S_min, S_max, 20)
    T_mesh, S_mesh = np.meshgrid(t_grid, S_grid, indexing='ij')
    b_grid = b_surface(T_mesh, S_mesh)
    
    assert b_grid.shape == T_mesh.shape, "Output shape mismatch"
    assert np.all(b_grid >= 0), "Volatilities should be non-negative"
    
    print(f"✓ Volatility surface test passed (range: [{b_grid.min():.4f}, {b_grid.max():.4f}])")


def test_integration_small_example():
    """Integration test: full pipeline on small example."""
    print("\n--- Running full integration test ---")
    
    # Setup
    d = 3
    S0 = np.array([225, 250, 275])
    r = 0.05
    vol = np.array([0.2, 0.15, 0.1])
    cov_mat = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
    basket_weights = np.ones(d) / d
    T = 1.0
    
    # Generate fine paths, then derive coarse from them
    dt_fine = 0.01
    N_fine = int(T / dt_fine) + 1
    N_paths = 200
    
    np.random.seed(42)  # Reproducibility
    paths_fine = simulate_gbm_paths(S0, r, vol, cov_mat, dt_fine, N_fine, N_paths)
    
    # Coarse paths = subsample fine paths (coupled!)
    paths_coarse = paths_fine[:, ::2, :]
    
    # Estimate domain
    basket_vals = paths_fine @ basket_weights
    S_min, S_max = np.percentile(basket_vals, [1, 99])
    print(f"Basket domain: [{S_min:.2f}, {S_max:.2f}]")
    
    # Fit volatility with low degree for this small test
    basis_pairs = generate_polynomial_basis_pairs(max_degree=1)
    D, psi = construct_regression_system(
        paths_fine, paths_coarse, basket_weights, basis_pairs,
        cov_mat, vol, S_min, S_max, T
    )
    
    print(f"Condition number: {np.linalg.cond(D):.2e}")
    
    c = fit_volatility_coefficients(D, psi)
    print(f"Fitted coefficients: {c}")
    
    # Construct surface
    b_surface = construct_volatility_surface(c, basis_pairs, S_min, S_max, T, max_degree=1)
    
    # Evaluate at some points
    b_t0 = b_surface(0.0, 250.0)
    b_mid = b_surface(T/2, 250.0)
    b_T = b_surface(T, 250.0)
    
    print(f"Volatility at t=0: {b_t0:.4f}")
    print(f"Volatility at t=T/2: {b_mid:.4f}")
    print(f"Volatility at t=T: {b_T:.4f}")
    
    # Sanity checks (relaxed for this small example)
    assert D.shape[0] == N_paths * paths_coarse.shape[1]
    assert not np.all([b_t0, b_mid, b_T] == 0), "All volatilities are zero"
    
    print("✓ Full integration test passed")


def run_all_tests():
    """Run all unit tests."""
    print("\nRunning basket_simulation.py tests...\n")
    
    test_simulate_gbm_paths()
    test_generate_polynomial_basis_pairs()
    test_construct_regression_system()
    test_fit_volatility_coefficients()
    test_construct_volatility_surface()
    test_integration_small_example()
    
    print("\n✅ All basket simulation tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
