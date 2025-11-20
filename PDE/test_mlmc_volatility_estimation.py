"""
Unit tests for mlmc_volatility_estimation.py

Validates domain estimation, coupled path generation, and MLMC hierarchy.
"""

import numpy as np
from mlmc_volatility_estimation import (
    estimate_basket_domain,
    generate_coupled_paths,
    estimate_coefficients_at_level,
    aggregate_mlmc_coefficients
)
from basket_simulation import construct_volatility_surface, generate_polynomial_basis_pairs


def test_estimate_basket_domain():
    """Test pilot run for domain estimation."""
    d = 3
    S0 = np.array([100, 100, 100])
    T = 1.0
    h0 = 0.1
    r = 0.05
    vol = np.array([0.2, 0.2, 0.2])
    cov_mat = np.eye(d)
    max_degree = 2
    basket_weights = np.ones(d) / d
    
    np.random.seed(42)
    S_min, S_max, basket_paths = estimate_basket_domain(
        S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, N_pilot=1000
    )
    
    # Check outputs
    assert isinstance(S_min, (float, np.floating)), "S_min should be float"
    assert isinstance(S_max, (float, np.floating)), "S_max should be float"
    assert S_min < S_max, f"S_min ({S_min}) should be less than S_max ({S_max})"
    
    # Check basket paths shape
    dt = h0 * 2 ** (-max_degree)
    N_steps = int(T / dt)
    assert basket_paths.shape == (1000, N_steps), f"Wrong basket shape: {basket_paths.shape}"
    
    # Check domain is reasonable (around initial price)
    S_initial = S0.mean()
    assert S_min < S_initial < S_max, "Initial price should be in domain"
    
    print(f"✓ Domain estimation test passed (S_min={S_min:.2f}, S_max={S_max:.2f})")


def test_generate_coupled_paths():
    """Test coupled path generation for MLMC."""
    d = 2
    S0 = np.array([100, 100])
    r = 0.05
    vol = np.array([0.2, 0.15])
    cov_mat = np.eye(d)
    dt_fine = 0.01
    N_fine = 100
    N_paths = 50
    
    # Test level 0 (coarse paths should be zero)
    np.random.seed(42)
    paths_fine_l0, paths_coarse_l0 = generate_coupled_paths(
        S0, r, vol, cov_mat, dt_fine, N_fine, N_paths, level=0
    )
    
    assert paths_fine_l0.shape == (N_paths, N_fine, d), "Wrong fine shape"
    assert paths_coarse_l0.shape == (N_paths, N_fine//2, d), "Wrong coarse shape"
    assert np.all(paths_coarse_l0 == 0), "Level 0 coarse paths should be zero"
    
    # Test level 1 (coarse paths should be non-zero)
    np.random.seed(42)
    paths_fine_l1, paths_coarse_l1 = generate_coupled_paths(
        S0, r, vol, cov_mat, dt_fine, N_fine, N_paths, level=1
    )
    
    assert paths_coarse_l1.shape == (N_paths, N_fine//2, d), "Wrong coarse shape"
    assert np.any(paths_coarse_l1 != 0), "Level 1 coarse paths should be non-zero"
    
    # Check initial conditions
    assert np.allclose(paths_fine_l1[:, 0, :], S0), "Fine initial condition"
    assert np.allclose(paths_coarse_l1[:, 0, :], S0), "Coarse initial condition"
    
    # Check positivity
    assert np.all(paths_fine_l1 > 0), "Fine paths should stay positive"
    assert np.all(paths_coarse_l1 >= 0), "Coarse paths should be non-negative"
    
    print("✓ Coupled path generation test passed")


def test_estimate_coefficients_at_level():
    """Test single-level coefficient estimation."""
    d = 2
    S0 = np.array([100, 100])
    T = 1.0
    h0 = 0.1
    r = 0.05
    vol = np.array([0.2, 0.15])
    cov_mat = np.eye(d)
    max_degree = 2
    basket_weights = np.ones(d) / d
    S_min, S_max = 80, 120
    
    print("\n--- Testing single level estimation ---")
    
    np.random.seed(42)
    c = estimate_coefficients_at_level(
        S0, T, h0, level=0, r=r, cov_mat=cov_mat, vol=vol,
        max_degree=max_degree, basket_weights=basket_weights,
        S_min=S_min, S_max=S_max
    )
    
    # Check output shape
    n_basis_max = len(generate_polynomial_basis_pairs(max_degree))
    assert c.shape == (n_basis_max,), f"Wrong shape: {c.shape}"
    
    # Check some coefficients are non-zero (we fitted something)
    assert not np.all(c == 0), "All coefficients are zero"
    
    print(f"✓ Single-level estimation test passed (coefficients: {c})")


def test_aggregate_mlmc_coefficients_small():
    """Test full MLMC aggregation with small parameters."""
    d = 2
    S0 = np.array([100, 100])
    T = 1.0
    h0 = 0.2
    r = 0.05
    vol = np.array([0.2, 0.15])
    cov_mat = np.eye(d)
    max_degree = 1  # Small for speed
    basket_weights = np.ones(d) / d
    
    # Estimate domain first
    np.random.seed(42)
    S_min, S_max, _ = estimate_basket_domain(
        S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, N_pilot=500
    )
    
    print("\n--- Testing MLMC aggregation ---")
    
    # Run MLMC
    np.random.seed(42)
    c_total = aggregate_mlmc_coefficients(
        S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, S_min, S_max
    )
    
    # Check shape
    n_basis_max = len(generate_polynomial_basis_pairs(max_degree))
    assert c_total.shape == (n_basis_max,), f"Wrong shape: {c_total.shape}"
    
    # Build volatility surface
    basis_pairs = generate_polynomial_basis_pairs(max_degree)
    b_surface = construct_volatility_surface(c_total, basis_pairs, S_min, S_max, T, max_degree)
    
    # Evaluate at mid-point
    b_mid = b_surface(T/2, 100.0)
    print(f"Volatility at (t=T/2, S=100): {b_mid:.4f}")
    
    # Sanity check
    assert b_mid >= 0, "Volatility should be non-negative"
    
    print(f"✓ MLMC aggregation test passed (total coeff: {c_total})")


def test_integration_full_pipeline():
    """Integration test: complete workflow from domain estimation to surface."""
    print("\n" + "="*60)
    print("FULL PIPELINE INTEGRATION TEST")
    print("="*60 + "\n")
    
    # Setup problem
    d = 3
    S0 = np.array([225, 250, 275])
    T = 1.0
    h0 = 0.2
    r = 0.05
    vol = np.array([0.2, 0.15, 0.1])
    cov_mat = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    basket_weights = np.ones(d) / d
    max_degree = 1  # Keep small for testing
    
    # Step 1: Domain estimation
    print("Step 1: Estimating basket domain...")
    np.random.seed(123)
    S_min, S_max, _ = estimate_basket_domain(
        S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, N_pilot=1000
    )
    print(f"  Domain: [{S_min:.2f}, {S_max:.2f}]")
    
    # Step 2: MLMC coefficient estimation
    print("\nStep 2: Running MLMC hierarchy...")
    np.random.seed(123)
    c_total = aggregate_mlmc_coefficients(
        S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, S_min, S_max
    )
    print(f"  Total coefficients: {c_total}")
    
    # Step 3: Construct volatility surface
    print("\nStep 3: Building volatility surface...")
    basis_pairs = generate_polynomial_basis_pairs(max_degree)
    b_surface = construct_volatility_surface(c_total, basis_pairs, S_min, S_max, T, max_degree)
    
    # Step 4: Evaluate surface
    print("\nStep 4: Evaluating surface...")
    t_test = np.array([0.0, 0.5, 1.0])
    S_test = np.array([230, 250, 270])
    
    for t_val in t_test:
        for S_val in S_test:
            b_val = b_surface(t_val, S_val)
            print(f"  b(t={t_val:.1f}, S={S_val:.0f}) = {b_val:.4f}")
    
    print("\n" + "="*60)
    print("✓ Full pipeline integration test PASSED")
    print("="*60)


def run_all_tests():
    """Run all unit tests."""
    print("\nRunning mlmc_volatility_estimation.py tests...\n")
    
    test_estimate_basket_domain()
    test_generate_coupled_paths()
    test_estimate_coefficients_at_level()
    test_aggregate_mlmc_coefficients_small()
    test_integration_full_pipeline()
    
    print("\n✅ All MLMC tests passed!\n")


if __name__ == "__main__":
    run_all_tests()
