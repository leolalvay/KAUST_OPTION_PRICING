# This file is based on test.py from Amelie's work.
# JAX version with timing and PRNG

"""
Example: American Basket Option Pricing with MLMC and Markovian Projection (JAX Version)

Demonstrates the complete pipeline:
1. Estimate basket domain from pilot runs
2. Run MLMC to fit projected volatility surface b(t,S)
3. Solve American option PDE using backward Euler
4. Analyse solution and exercise boundary
"""

import time
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from JAX_mlmc_volatility_estimation import (
    estimate_basket_domain,
    aggregate_mlmc_coefficients
)
from JAX_basket_simulation import (
    generate_polynomial_basis_pairs,
    construct_volatility_surface
)
from JAX_american_option_pde_solver import (
    solve_american_option,
    compute_exercise_boundary
)


# Start timing
start_time = time.time()

# Initialize JAX PRNG with fixed seed
main_key = random.PRNGKey(42)

# ============================================================================
# Problem Setup
# ============================================================================

print("\n" + "="*70)
print("AMERICAN BASKET OPTION PRICING WITH MLMC + MARKOVIAN PROJECTION (JAX)")
print("="*70 + "\n")

# Basket parameters
d = 3  # Number of assets
S0 = jnp.linspace(225, 275, num=d)[:, jnp.newaxis]  # Initial prices
basket_weights = jnp.ones(d) / d  # Equal-weighted basket
r = 0.05  # Risk-free rate
vol = jnp.array([0.2, 0.15, 0.1])  # Asset volatilities
cov_mat = jnp.array([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.1],
    [0.3, 0.1, 1.0]
])  # Correlation matrix

# Option parameters
T = 1.0  # Maturity
K = 250  # Strike price
option_type = "put"

# MLMC parameters
h0 = 0.05  # Coarsest timestep
max_degree = 3  # Maximum polynomial degree (also max MLMC level)

# PDE solver parameters
N_timesteps = 200  # Number of time steps
N_spatial = 50  # Number of spatial points

print(f"Basket Configuration:")
print(f"  Assets: {d}")
print(f"  Initial prices: {np.asarray(S0.flatten())}")
print(f"  Volatilities: {np.asarray(vol)}")
print(f"  Basket weights: {np.asarray(basket_weights)}")
print(f"\nOption Details:")
print(f"  Type: American {option_type}")
print(f"  Strike: K = {K}")
print(f"  Maturity: T = {T}")
print(f"  Rate: r = {r}")


# ============================================================================
# Step 1: Domain Estimation via Pilot Run
# ============================================================================

print(f"\n{'-'*70}")
print("STEP 1: Estimating Basket Domain")
print(f"{'-'*70}\n")

main_key, subkey = random.split(main_key)
S_min, S_max, basket_paths = estimate_basket_domain(
    S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, N_pilot=10000, key=subkey
)

print(f"Basket domain: [{S_min:.2f}, {S_max:.2f}]")
print(f"  Pilot paths simulated: 10,000")
print(f"  Domain width: {S_max - S_min:.2f}")


# ============================================================================
# Step 2: MLMC Volatility Surface Estimation
# ============================================================================

print(f"\n{'-'*70}")
print("STEP 2: MLMC Coefficient Estimation")
print(f"{'-'*70}\n")

main_key, subkey = random.split(main_key)
c_total = aggregate_mlmc_coefficients(
    S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, S_min, S_max, subkey
)

print(f"\nTotal coefficients: {np.asarray(c_total)}")
print(f"Non-zero coefficients: {jnp.sum(jnp.abs(c_total) > 1e-10)}")

# Construct volatility surface
basis_pairs = generate_polynomial_basis_pairs(max_degree)
volatility_surface = construct_volatility_surface(
    c_total, basis_pairs, S_min, S_max, T, max_degree
)

# Sample volatility at a few points
print(f"\nVolatility surface samples:")
for t_val in [0.0, T/2, T]:
    for S_val in [230, 250, 270]:
        b_val = volatility_surface(t_val, S_val)
        print(f"  b(t={t_val:.1f}, S={S_val}) = {np.asarray(b_val):.4f}")


# ============================================================================
# Step 3: Solve American Option PDE
# ============================================================================

print(f"\n{'-'*70}")
print("STEP 3: Solving American Option PDE")
print(f"{'-'*70}\n")

t_grid, S_grid, U, b_grid, payoff_grid = solve_american_option(
    T, S_min, S_max, N_timesteps, N_spatial, r, K,
    option_type, volatility_surface, plot=False
)

print(f"PDE Grid:")
print(f"  Time steps: {N_timesteps}")
print(f"  Spatial points: {N_spatial}")
print(f"  dt = {T/N_timesteps:.6f}")
print(f"  dS = {(S_max - S_min)/(N_spatial - 1):.4f}")

# Find option value at key points
S_grid_np = np.asarray(S_grid)
U_np = np.asarray(U)
payoff_grid_np = np.asarray(payoff_grid)

S_idx_atm = np.argmin(np.abs(S_grid_np - K))
S_idx_itm = np.argmin(np.abs(S_grid_np - 0.9*K))
S_idx_otm = np.argmin(np.abs(S_grid_np - 1.1*K))

print(f"\nOption Values at t=0:")
print(f"  At-the-money (S≈{S_grid_np[S_idx_atm]:.2f}): {U_np[0, S_idx_atm]:.4f}")
print(f"  In-the-money (S≈{S_grid_np[S_idx_itm]:.2f}): {U_np[0, S_idx_itm]:.4f}")
print(f"  Out-of-the-money (S≈{S_grid_np[S_idx_otm]:.2f}): {U_np[0, S_idx_otm]:.4f}")

print(f"\nIntrinsic Values:")
print(f"  At-the-money: {payoff_grid_np[S_idx_atm]:.4f}")
print(f"  In-the-money: {payoff_grid_np[S_idx_itm]:.4f}")
print(f"  Out-of-the-money: {payoff_grid_np[S_idx_otm]:.4f}")


# ============================================================================
# Step 4: Exercise Boundary Analysis
# ============================================================================

print(f"\n{'-'*70}")
print("STEP 4: Exercise Boundary Analysis")
print(f"{'-'*70}\n")

t_boundary, S_boundary = compute_exercise_boundary(
    t_grid, S_grid, U, payoff_grid, absolute_threshold=0.5
)

print(f"Exercise boundary points found: {len(t_boundary)}")
if len(t_boundary) > 0:
    print(f"  Boundary at t=0: S* ≈ {S_boundary[0]:.2f}")
    print(f"  Boundary at t={t_boundary[-1]:.2f}: S* ≈ {S_boundary[-1]:.2f}")


# End timing (before plot)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n{'='*70}")
print(f"⏱️  JAX VERSION TOTAL EXECUTION TIME: {elapsed_time:.4f} seconds")
print(f"{'='*70}\n")


# ============================================================================
# Step 5: Visualization
# ============================================================================

print(f"\n{'-'*70}")
print("STEP 5: Generating Visualizations")
print(f"{'-'*70}\n")

# Create comprehensive plot
fig = plt.figure(figsize=(16, 10))

# Convert all arrays to numpy for plotting
t_grid_np = np.asarray(t_grid)
b_grid_np = np.asarray(b_grid)
T_mesh, S_mesh = np.meshgrid(t_grid_np, S_grid_np, indexing='ij')

# Subplot 1: Option value surface
ax1 = fig.add_subplot(231, projection='3d')
surf1 = ax1.plot_surface(T_mesh, S_mesh, U_np, cmap='viridis', alpha=0.9)
ax1.set_title(f"American {option_type.capitalize()} Value", fontsize=12, fontweight='bold')
ax1.set_xlabel("Time t")
ax1.set_ylabel("Basket Price S")
ax1.set_zlabel("Option Value")
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Subplot 2: Volatility surface
ax2 = fig.add_subplot(232, projection='3d')
surf2 = ax2.plot_surface(T_mesh, S_mesh, b_grid_np, cmap='plasma', alpha=0.9)
ax2.set_title("Projected Volatility b(t,S)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Time t")
ax2.set_ylabel("Basket Price S")
ax2.set_zlabel("Volatility")
fig.colorbar(surf2, ax=ax2, shrink=0.5)

# Subplot 3: Option value at t=0
ax3 = fig.add_subplot(233)
ax3.plot(S_grid_np, U_np[0, :], 'b-', linewidth=2, label='American value')
ax3.plot(S_grid_np, payoff_grid_np, 'r--', linewidth=2, label='Intrinsic value')
ax3.axvline(K, color='k', linestyle=':', alpha=0.5, label='Strike')
ax3.set_xlabel("Basket Price S")
ax3.set_ylabel("Option Value")
ax3.set_title("Value at t=0", fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Exercise boundary
ax4 = fig.add_subplot(234)
if len(t_boundary) > 0:
    ax4.plot(t_boundary, S_boundary, 'ro-', linewidth=2, markersize=3)
    ax4.axhline(K, color='k', linestyle=':', alpha=0.5, label='Strike')
    ax4.set_xlabel("Time t")
    ax4.set_ylabel("Exercise Boundary S*")
    ax4.set_title("Early Exercise Boundary", fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No exercise boundary detected',
             ha='center', va='center', transform=ax4.transAxes)

# Subplot 5: Time value (American - Intrinsic)
ax5 = fig.add_subplot(235)
time_value = U_np[0, :] - payoff_grid_np
ax5.plot(S_grid_np, time_value, 'g-', linewidth=2)
ax5.axvline(K, color='k', linestyle=':', alpha=0.5, label='Strike')
ax5.axhline(0, color='k', linestyle='-', alpha=0.3)
ax5.set_xlabel("Basket Price S")
ax5.set_ylabel("Time Value")
ax5.set_title("Time Value at t=0", fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Volatility at t=0
ax6 = fig.add_subplot(236)
ax6.plot(S_grid_np, b_grid_np[0, :], 'purple', linewidth=2)
ax6.axvline(K, color='k', linestyle=':', alpha=0.5, label='Strike')
ax6.set_xlabel("Basket Price S")
ax6.set_ylabel("Volatility b")
ax6.set_title("Volatility at t=0", fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('JAX_american_basket_option_analysis.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Summary Statistics
# ============================================================================

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")

print(f"Computational Details:")
print(f"  MLMC levels: {max_degree + 1}")
print(f"  Polynomial basis size: {len(basis_pairs)}")
print(f"  PDE grid: {N_timesteps} × {N_spatial}")
print(f"  Total PDE timesteps: {N_timesteps}")

print(f"\nKey Results:")
print(f"  Option value at S={S_grid_np[S_idx_atm]:.2f}: {U_np[0, S_idx_atm]:.4f}")
print(f"  Intrinsic value: {payoff_grid_np[S_idx_atm]:.4f}")
print(f"  Time value: {U_np[0, S_idx_atm] - payoff_grid_np[S_idx_atm]:.4f}")
if payoff_grid_np[S_idx_atm] > 0:
    print(f"  Early exercise premium: {(U_np[0, S_idx_atm] / payoff_grid_np[S_idx_atm] - 1)*100:.2f}%")

print(f"\nVolatility Statistics:")
print(f"  Mean volatility: {b_grid_np.mean():.4f}")
print(f"  Min volatility: {b_grid_np.min():.4f}")
print(f"  Max volatility: {b_grid_np.max():.4f}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}\n")

plt.show()
