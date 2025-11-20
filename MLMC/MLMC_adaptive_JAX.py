"""
Multi-Level Monte Carlo (MLMC) Estimator with Adaptive Sample Allocation - JAX Version
=======================================================================================

JAX-optimized implementation with JIT compilation for GPU/TPU acceleration.
Uses the same algorithm as MLMC_adaptive.py but with vectorized operations.

Key JAX Features:
    - JIT compilation for hot paths
    - Vectorization via vmap instead of for-loops
    - Efficient sequential operations via jax.lax.scan
    - Reproducible PRNG key management

Theory:
    MLMC decomposes the expectation:
        E[P_L] = E[P_0] + sum_{l=1}^L E[P_l - P_{l-1}]

    Optimal sample allocation minimises total cost subject to achieving
    root-mean-square error <= epsilon:
        N_l^opt = (2/eps^2) * sqrt(V_l/C_l) * sum_j sqrt(V_j * C_j)

    where V_l = Var[Y_l] and C_l = cost per sample at level l.
"""

# Enable 64-bit precision before importing JAX
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap, lax
import jax.random as random
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Import from Normal_MC package for comparisons
from Normal_MC import BS_call, run_mc_estimation_jax

# ============================================================================
# JAX-Optimized Helper Functions
# ============================================================================

def optimal_sample_allocation(V, C, eps):
    """
    Compute optimal number of samples per level for MLMC.

    Minimises total cost: sum_l N_l * C_l
    Subject to: RMSE <= eps

    Solution from Lagrange multipliers:
        N_l = (2/eps^2) * sqrt(V_l/C_l) * sum_j sqrt(V_j * C_j)

    Parameters
    ----------
    V : np.ndarray
        Variance at each level V_l = Var[Y_l]
    C : np.ndarray
        Cost per sample at each level (proportional to number of timesteps)
    eps : float
        Target RMSE accuracy

    Returns
    -------
    N_opt : np.ndarray (int)
        Optimal number of samples at each level
    """
    # Total weighted variance-cost product
    sum_sqrt_VC = np.sum(np.sqrt(V * C))

    # Optimal allocation formula
    mu = 2.0 * sum_sqrt_VC / eps**2
    N_opt = mu * np.sqrt(V / C)

    # Round up to ensure we meet accuracy target
    return np.ceil(N_opt).astype(int)


# ============================================================================
# JIT-Compiled Simulation Functions
# ============================================================================

@jit
def euler_maruyama_step(carry, dW):
    """Single Euler-Maruyama step for GBM."""
    S, r, sigma, h = carry
    S_new = S + r * S * h + sigma * S * dW
    return (S_new, r, sigma, h), S_new


def simulate_paths_at_level(key, l, N_l, T, S0, K, r, sigma, h0):
    """
    Simulate N_l coupled fine/coarse paths at level l using JAX.

    Returns sum of Y and sum of Y^2 for variance estimation.
    """
    # Timesteps at this level
    h_fine = h0 * (2.0 ** (-l))
    n_steps = int(T / h_fine)
    h_coarse = 2.0 * h_fine

    # Discount factor
    disc = math.exp(-r * T)

    # Generate all random increments at once
    # Shape: (N_l, n_steps)
    dW_fine_all = random.normal(key, shape=(N_l, n_steps)) * jnp.sqrt(h_fine)

    # Define single path simulation function
    def simulate_single_path(dW_fine):
        """Simulate one fine and one coarse path with coupled increments."""

        # Fine path using scan
        def fine_step(S, dW):
            S_new = S + r * S * h_fine + sigma * S * dW
            return S_new, S_new

        S_fine_final, _ = lax.scan(fine_step, S0, dW_fine)
        Pf = jnp.maximum(S_fine_final - K, 0.0)

        # Coarse path (only if l > 0)
        if l > 0:
            # Sum consecutive pairs of fine increments
            dW_coarse = dW_fine.reshape(-1, 2).sum(axis=1)

            def coarse_step(S, dW):
                S_new = S + r * S * h_coarse + sigma * S * dW
                return S_new, S_new

            S_coarse_final, _ = lax.scan(coarse_step, S0, dW_coarse)
            Pc = jnp.maximum(S_coarse_final - K, 0.0)
        else:
            Pc = 0.0

        # Correction term
        Y = disc * (Pf - Pc)
        return Y

    # Vectorize over all samples
    Y_all = vmap(simulate_single_path)(dW_fine_all)

    # Compute sums for statistics
    sum_Y = jnp.sum(Y_all)
    sum_Y_sq = jnp.sum(Y_all ** 2)

    return float(sum_Y), float(sum_Y_sq)


def mlmc_level_simulation_jax(key, l, N_l, T, S0, K, r, sigma, h0):
    """
    JAX wrapper for level simulation with proper JIT compilation.

    Creates a JIT-compiled version specific to this level's timestep count.
    """
    # Timesteps at this level
    h_fine = h0 * (2.0 ** (-l))
    n_steps = int(T / h_fine)
    h_coarse = 2.0 * h_fine

    # Discount factor
    disc = math.exp(-r * T)

    # Generate all random increments at once
    dW_fine_all = random.normal(key, shape=(N_l, n_steps)) * jnp.sqrt(h_fine)

    # JIT-compile a function for this specific level
    if l == 0:
        @jit
        def compute_corrections_l0(dW_fine_all):
            def simulate_single_path(dW_fine):
                def fine_step(S, dW):
                    S_new = S + r * S * h_fine + sigma * S * dW
                    return S_new, S_new

                S_fine_final, _ = lax.scan(fine_step, S0, dW_fine)
                Pf = jnp.maximum(S_fine_final - K, 0.0)
                Y = disc * Pf  # No coarse level for l=0
                return Y

            Y_all = vmap(simulate_single_path)(dW_fine_all)
            return jnp.sum(Y_all), jnp.sum(Y_all ** 2)

        sum_Y, sum_Y_sq = compute_corrections_l0(dW_fine_all)
    else:
        @jit
        def compute_corrections(dW_fine_all):
            def simulate_single_path(dW_fine):
                # Fine path
                def fine_step(S, dW):
                    S_new = S + r * S * h_fine + sigma * S * dW
                    return S_new, S_new

                S_fine_final, _ = lax.scan(fine_step, S0, dW_fine)
                Pf = jnp.maximum(S_fine_final - K, 0.0)

                # Coarse path with coupled increments
                dW_coarse = dW_fine.reshape(-1, 2).sum(axis=1)

                def coarse_step(S, dW):
                    S_new = S + r * S * h_coarse + sigma * S * dW
                    return S_new, S_new

                S_coarse_final, _ = lax.scan(coarse_step, S0, dW_coarse)
                Pc = jnp.maximum(S_coarse_final - K, 0.0)

                Y = disc * (Pf - Pc)
                return Y

            Y_all = vmap(simulate_single_path)(dW_fine_all)
            return jnp.sum(Y_all), jnp.sum(Y_all ** 2)

        sum_Y, sum_Y_sq = compute_corrections(dW_fine_all)

    return float(sum_Y), float(sum_Y_sq)


# ============================================================================
# Problem Parameters
# ============================================================================

# Option parameters (European call)
T = 1.0          # Time to maturity (years)
S0 = 100.0       # Initial stock price
K = 100.0        # Strike price
r = 0.05         # Risk-free rate
sigma = 0.2      # Volatility

# Target accuracies to test
epsilon_values = np.array([0.1, 0.05, 0.02, 0.01, 0.005])

# Convergence rate parameters (from theory/pilot studies)
alpha = 1.0      # Weak convergence rate: |E[Y_l]| ~ 2^(-alpha*l)
beta = 1.0       # Variance decay rate: Var[Y_l] ~ 2^(-beta*l)
gamma = 1.0      # Cost growth rate: C_l ~ 2^(gamma*l)

# Calculate analytical Black-Scholes price for reference
BS_price = BS_call(S0, K, T, r, sigma)

# Initialize master PRNG key for reproducibility
master_key = random.PRNGKey(42)

# Split key for MC estimation and MLMC
master_key, mc_key = random.split(master_key)

# Run JAX-accelerated standard MC estimation for comparison
print("Running JAX-accelerated standard MC estimation for comparison...")
mc_results = run_mc_estimation_jax(
    epsilon_values, T=T, S0=S0, K=K, r=r, sigma=sigma, N0=100000, verbose=False, key=mc_key
)
print("Done.\n")

# Storage for results across different epsilon values
results_N = []           # Sample allocation N_l for each epsilon
results_cost = []        # Total computational cost
results_cost_mc = []     # Actual standard MC cost for comparison
results_time = []        # Time per epsilon convergence

# Start total timer
total_start_time = time.time()

# ============================================================================
# Main MLMC Loop: Test Multiple Accuracy Levels
# ============================================================================

print(f"\n{'='*70}")
print("ADAPTIVE MLMC (JAX): Optimal Sample Allocation")
print(f"{'='*70}")
print(f"Option: European Call (S0={S0}, K={K}, T={T})")
print(f"Model: GBM (r={r}, sigma={sigma})")
print(f"Testing accuracy levels: {epsilon_values}")
print(f"{'='*70}\n")

# Split master key for each epsilon iteration
eps_keys = random.split(master_key, len(epsilon_values))

for eps_idx, eps in enumerate(epsilon_values):
    # Start timer for this epsilon
    eps_start_time = time.time()

    # Get key for this epsilon
    eps_key = eps_keys[eps_idx]

    print(f"\n{'='*70}")
    print(f"TARGET ACCURACY: eps = {eps}")
    print(f"{'='*70}")

    # ========================================================================
    # Initialisation
    # ========================================================================

    # Start with L = 3 levels (h = 0.5, 0.25, 0.125, 0.0625)
    L = 3
    h0 = 0.5

    # Initial pilot sample size (will be refined adaptively)
    N0_pilot = 100

    # Current sample counts at each level
    N = np.zeros(L + 1, dtype=int)

    # Additional samples needed at each level (initialise with pilot run)
    dN = np.full(L + 1, N0_pilot, dtype=int)

    # Accumulated sums for statistics
    # sum_stats[0, l] = sum of Y at level l
    # sum_stats[1, l] = sum of Y^2 at level l
    sum_stats = np.zeros((2, L + 1))

    # Cost per sample at each level (proportional to number of timesteps)
    levels = np.arange(L + 1)
    cost_per_sample = 2.0**(gamma * levels)

    # Key management for iterations
    iteration_key = eps_key

    # ========================================================================
    # Adaptive Refinement Loop
    # ========================================================================

    iteration = 0
    while np.any(dN > 0):
        iteration += 1
        print(f"\nIteration {iteration}: L = {L}, max(dN) = {dN.max()}")

        # Split key for this iteration
        iteration_key, *level_keys = random.split(iteration_key, L + 2)

        # ====================================================================
        # Step 1: Run Additional Samples Where Needed
        # ====================================================================
        for l in range(L + 1):
            if dN[l] > 0:
                # Run dN[l] new samples at level l using JAX
                sum_Y, sum_Y_sq = mlmc_level_simulation_jax(
                    level_keys[l], l, dN[l], T, S0, K, r, sigma, h0
                )

                # Update total sample count
                N[l] += dN[l]

                # Accumulate statistics
                sum_stats[0, l] += sum_Y
                sum_stats[1, l] += sum_Y_sq

        # ====================================================================
        # Step 2: Estimate Mean and Variance at Each Level
        # ====================================================================

        # Sample mean: E[Y_l] ~ (sum Y) / N_l
        mean_Y = np.abs(sum_stats[0, :] / N)

        # Sample variance: Var[Y_l] ~ E[Y^2] - E[Y]^2
        var_Y = np.maximum(0.0, sum_stats[1, :] / N - mean_Y**2)

        # ====================================================================
        # Step 3: Enforce Minimum Decay Rates
        # ====================================================================
        # This prevents underestimation of variance at fine levels due to
        # limited samples. We assume variance decays at least as fast as theory.

        for l in range(2, L + 1):
            # Mean should decay like 2^(-alpha*l)
            mean_Y[l] = max(mean_Y[l], 0.5 * mean_Y[l-1] / 2**alpha)

            # Variance should decay like 2^(-beta*l)
            var_Y[l] = max(var_Y[l], 0.5 * var_Y[l-1] / 2**beta)

        # ====================================================================
        # Step 4: Compute Optimal Sample Allocation
        # ====================================================================
        N_opt = optimal_sample_allocation(var_Y, cost_per_sample, eps)

        # Additional samples needed
        dN = np.maximum(N_opt - N, 0)

        print(f"  Current N: {N}")
        print(f"  Optimal N: {N_opt}")
        print(f"  Need dN: {dN}")

        # ====================================================================
        # Step 5: Check if We Need More Levels
        # ====================================================================

        # Convergence criterion: if all levels have enough samples (within 1%)
        converged = np.sum(dN > 0.01 * N) == 0

        if converged:
            # Estimate remaining bias from truncating at level L
            # Uses extrapolation from last 3 levels
            indices = np.array([-2, -1, 0])
            means_last_three = mean_Y[L + indices]

            # Extrapolate: |E[Y_L+1]| ~ max of Richardson extrapolation
            # Formula: sum_{i=L-2}^L E[Y_i] * 2^(alpha*i) / (2^alpha - 1)
            bias_estimate = np.max(means_last_three * 2.0**(alpha * indices)) / (2.0**alpha - 1.0)

            print(f"  Estimated remaining bias: {bias_estimate:.6f}")
            print(f"  Bias threshold (eps/sqrt(2)): {eps/math.sqrt(2):.6f}")

            if bias_estimate > eps / math.sqrt(2):
                # Need another level!
                print(f"  -> Adding level {L+1}")

                L += 1

                # Extend arrays
                N = np.append(N, 0)
                dN = np.append(dN, 0)
                sum_stats = np.hstack((sum_stats, np.zeros((2, 1))))

                # Extrapolate variance for new level
                var_Y = np.append(var_Y, var_Y[-1] / 2**beta)
                mean_Y = np.append(mean_Y, 0.0)

                # Update cost array
                levels = np.arange(L + 1)
                cost_per_sample = 2.0**(gamma * levels)

                # Recompute optimal allocation with new level
                N_opt = optimal_sample_allocation(var_Y, cost_per_sample, eps)
                dN = np.maximum(N_opt - N, 0)
            else:
                # Bias is acceptable, we're done!
                print(f"  [OK] Bias is acceptable, converged!")
                break

    # ========================================================================
    # Final MLMC Estimate
    # ========================================================================

    # Compute final mean at each level
    final_mean_Y = sum_stats[0, :] / N

    # MLMC estimator: sum of all corrections
    P_mlmc = np.sum(final_mean_Y)

    # Total computational cost (weighted by epsilon^2 for fair comparison)
    total_cost = (eps**2) * np.dot(N, cost_per_sample)

    # Get actual MC cost for this epsilon from pre-computed results
    eps_result_idx = np.where(epsilon_values == eps)[0][0]
    cost_mc_actual = mc_results['total_cost'][eps_result_idx]

    # Finest timestep for reference
    h_finest = h0 * 2**(-L)

    # ========================================================================
    # Summary for This Epsilon
    # ========================================================================

    print(f"\n{'='*70}")
    print(f"RESULTS FOR eps = {eps}")
    print(f"{'='*70}")
    print(f"MLMC option price estimate: {P_mlmc:.6f}")
    print(f"Black-Scholes analytical:   {BS_price:.6f}")
    print(f"MLMC error vs analytical:   {abs(P_mlmc - BS_price):.6f}")
    print(f"Total samples used: {N.sum():,}")
    print(f"Number of levels: {L + 1}")
    print(f"Finest timestep: {h_finest:.6f}")
    print(f"Sample distribution: {N}")
    print(f"Normalised cost MLMC (eps^2*C): {total_cost:.2f}")
    print(f"Normalised cost MC (eps^2*C):   {cost_mc_actual:.2f}")
    print(f"{'='*70}")

    # Record time for this epsilon
    eps_elapsed_time = time.time() - eps_start_time
    print(f"\nTIMING: eps = {eps} converged in {eps_elapsed_time:.2f} seconds")

    # Store results for plotting
    results_N.append(N.copy())
    results_cost.append(total_cost)
    results_cost_mc.append(cost_mc_actual)
    results_time.append(eps_elapsed_time)

# ============================================================================
# Visualisation: Complexity Analysis
# ============================================================================

print(f"\n{'='*70}")
print("GENERATING COMPLEXITY PLOTS")
print(f"{'='*70}\n")

markers = ['o', 'x', 'd', 's', '^']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ========================================================================
# Panel 1: Sample Allocation N_l vs Level l
# ========================================================================
# Shows how MLMC distributes samples across levels for different accuracies

for i, eps in enumerate(epsilon_values):
    N_levels = results_N[i]
    levels_plot = np.arange(len(N_levels))

    ax1.semilogy(levels_plot, N_levels,
                 marker=markers[i],
                 linestyle='--',
                 linewidth=2,
                 markersize=8,
                 label=f"eps = {eps}")

ax1.set_xlabel(r"Level $\ell$", fontsize=12)
ax1.set_ylabel(r"$N_\ell$ (samples)", fontsize=12)
ax1.legend(loc="best", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title("Optimal Sample Allocation (JAX)", fontsize=13, fontweight='bold')

# ========================================================================
# Panel 2: eps^2 x Cost vs eps (Complexity Analysis)
# ========================================================================
# If cost ~ eps^(-p), then eps^2 x cost ~ eps^(2-p)
# For optimal MLMC: p = 2, so plot should be flat
# For standard MC: p = 3, so plot grows like eps^(-1)

ax2.loglog(epsilon_values, results_cost,
           marker='o', linestyle='--', linewidth=2, markersize=8,
           color='orange', label='MLMC (JAX)')

ax2.loglog(epsilon_values, results_cost_mc,
           marker='d', linestyle='--', linewidth=2, markersize=8,
           color='green', label='Standard MC (JAX)')

ax2.set_xlabel(r"Accuracy $\epsilon$", fontsize=12)
ax2.set_ylabel(r"$\epsilon^2 \cdot C$ (cost)", fontsize=12)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_title("Complexity Comparison (JAX)", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('MLMC_adaptive_JAX.pdf',  bbox_inches='tight')
print("[OK] Complexity plots saved to: MLMC_adaptive_JAX.pdf")


# ============================================================================
# Final Summary
# ============================================================================

print(f"\n{'='*70}")
print("MLMC (JAX) COMPLEXITY ANALYSIS SUMMARY")
print(f"{'='*70}")
print(f"Black-Scholes analytical price: {BS_price:.6f}")
print(f"{'='*70}")
print(f"{'Epsilon':<12} {'MLMC Cost':<15} {'MC Cost':<15} {'Speedup':<10}")
print(f"{'-'*70}")
for i, eps in enumerate(epsilon_values):
    speedup = results_cost_mc[i] / results_cost[i]
    print(f"{eps:<12.3f} {results_cost[i]:<15.2f} {results_cost_mc[i]:<15.2f} {speedup:<10.1f}x")
print(f"{'='*70}")
print(f"\nKey insight: MLMC achieves O(eps^-2) complexity vs O(eps^-3) for standard MC")
print(f"This is a fundamental algorithmic improvement, like FFT vs DFT!\n")

# Print total execution time
total_elapsed_time = time.time() - total_start_time
print(f"{'='*70}")
print(f"TOTAL EXECUTION TIME: {total_elapsed_time:.2f} seconds")
print(f"{'='*70}\n")


plt.show()