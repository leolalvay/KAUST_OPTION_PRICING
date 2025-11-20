"""
GPU-Accelerated Single-Level Monte Carlo for European Call Options.

Uses JAX for automatic vectorisation and GPU acceleration. Demonstrates
computational cost scaling of single-level Monte Carlo as a function of
target accuracy epsilon, with comparison to analytical Black-Scholes.

Runs full simulations with N_needed samples (not just pilot runs).
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from BS_Analytic import BS_call

# Enable 64-bit precision (important for financial calculations)
jax.config.update("jax_enable_x64", True)

# Market parameters
T = 1.0          # Time to maturity (years)
vol = 0.2        # Volatility (annualised)
r = 0.05         # Risk-free rate
K = 100.0        # Strike price
S0 = 100.0       # Initial spot price

# Monte Carlo parameters
epsilon = jnp.array([0.1, 0.05, 0.02, 0.01, 0.005])
N0 = 50000       # Pilot samples for variance estimation


def simulate_gbm_path(key, S0, r, vol, dt, steps):
    """
    Simulate a single GBM path using Euler-Maruyama discretisation.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate.
    vol : float
        Volatility.
    dt : float
        Time step size.
    steps : int
        Number of time steps (must be static for JIT).
    
    Returns
    -------
    float
        Final stock price at maturity.
    
    Notes
    -----
    Implements: dS = r*S*dt + sigma*S*dW
    steps parameter is static (not traced) for JIT compilation.
    """
    # Generate all Brownian increments at once
    dW = random.normal(key, shape=(steps,)) * jnp.sqrt(dt)
    
    # Vectorised path simulation using scan for efficiency
    def step_fn(S, dw):
        S_new = S + S * r * dt + S * vol * dw
        return S_new, S_new
    
    S_final, _ = jax.lax.scan(step_fn, S0, dW)
    return S_final


def compute_payoff(key, S0, K, r, vol, T, dt, steps):
    """
    Compute discounted call option payoff for a single path.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    vol : float
        Volatility.
    T : float
        Time to maturity.
    dt : float
        Time step size.
    steps : int
        Number of time steps (must be static for JIT).
    
    Returns
    -------
    float
        Discounted payoff: exp(-rT) * max(S_T - K, 0)
    """
    S_final = simulate_gbm_path(key, S0, r, vol, dt, steps)
    payoff = jnp.maximum(S_final - K, 0.0)
    return jnp.exp(-r * T) * payoff


def run_monte_carlo(key, N, S0, K, r, vol, T, dt, steps):
    """
    Run Monte Carlo simulation with N samples.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key.
    N : int
        Number of Monte Carlo samples.
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    vol : float
        Volatility.
    T : float
        Time to maturity.
    dt : float
        Time step size.
    steps : int
        Number of time steps.
    
    Returns
    -------
    tuple
        (mean_price, variance, payoffs_array)
    
    Notes
    -----
    Processes samples in batches for GPU efficiency.
    JIT-compiles a version specific to this step count.
    """
    # Create JIT-compiled functions with steps as static argument
    # This compiles a new version for each unique step count
    compute_payoff_jit = jit(compute_payoff, static_argnums=(7,))
    compute_payoffs_batch = jit(vmap(compute_payoff_jit, in_axes=(0, None, None, None, None, None, None, None)),
                                 static_argnums=(7,))
    
    # Split key for N independent samples
    keys = random.split(key, N)
    
    # Compute all payoffs in parallel on GPU
    payoffs = compute_payoffs_batch(keys, S0, K, r, vol, T, dt, steps)
    
    # Block until computation is done (for timing purposes)
    payoffs.block_until_ready()
    
    mean_price = jnp.mean(payoffs)
    variance = jnp.var(payoffs, ddof=1)
    
    return mean_price, variance, payoffs


# Main execution
if __name__ == "__main__":
    # Initialise random key
    key = random.PRNGKey(42)
    
    # Calculate analytical Black-Scholes price
    BS_price = BS_call(S0, K, T, r, vol)
    print(f"Black-Scholes analytical price: {BS_price:.6f}\n")
    print("="*70)
    
    # Storage for results
    total_cost = []
    abs_errors = []
    pct_errors = []
    mc_prices = []
    samples_used = []
    total_times = []
    
    for eps in epsilon:
        # Set time step adaptively
        dt = eps / math.sqrt(2)
        steps = int(T / dt)
        
        # Split key for this iteration
        key, subkey = random.split(key)
        
        # STEP 1: Quick pilot run to estimate variance
        print(f"Running pilot simulation for ε = {float(eps):.3f} with {N0} samples...")
        start_time = time.time()
        mean_pilot, var_pilot, payoffs_pilot = run_monte_carlo(
            subkey, N0, S0, K, r, vol, T, dt, steps
        )
        pilot_time = time.time() - start_time
        
        # Determine required sample size
        N_needed = int(math.ceil(2 * float(var_pilot) / eps**2))
        print(f"  Pilot completed in {pilot_time:.3f}s")
        print(f"  Estimated samples needed: {N_needed:,}")
        
        # STEP 2: Full simulation with N_needed samples
        key, subkey = random.split(key)
        print(f"  Running full simulation with {N_needed:,} samples...")
        start_time = time.time()
        mean_full, var_full, payoffs_full = run_monte_carlo(
            subkey, N_needed, S0, K, r, vol, T, dt, steps
        )
        full_time = time.time() - start_time
        
        # Total computational cost
        eps2C = eps**2 * N_needed * steps
        total_cost.append(float(eps2C))
        
        # Calculate errors using the FULL simulation
        abs_error = abs(float(mean_full) - BS_price)
        pct_error = 100 * abs_error / BS_price
        
        # Store results for plotting
        abs_errors.append(abs_error)
        pct_errors.append(pct_error)
        mc_prices.append(float(mean_full))
        samples_used.append(N_needed)
        total_times.append(pilot_time + full_time)
        
        print(f"\nTarget accuracy ε = {float(eps):.3f}:")
        print(f"  MC price estimate:    {float(mean_full):.6f}")
        print(f"  Absolute error:       {abs_error:.6f}")
        print(f"  Percentage error:     {pct_error:.3f}%")
        print(f"  Steps per path:       {steps}")
        print(f"  Actual samples used:  {N_needed:,}")
        print(f"  Total cost (ε²C):     {eps2C:.2e}")
        print(f"  Pilot time:           {pilot_time:.3f} seconds")
        print(f"  Full simulation time: {full_time:.3f} seconds")
        print(f"  Total time:           {pilot_time + full_time:.3f} seconds")
        print(f"  Throughput:           {N_needed/full_time:,.0f} samples/sec")
        print("-"*70)
    
    # Save results
    data = np.column_stack((np.array(epsilon), np.array(total_cost)))
    np.savetxt(
        "mc_cost_jax.txt",
        data,
        fmt="%.6e",
        header="epsilon          total_cost",
        comments=""
    )
    
    print("\nResults saved to mc_cost_jax.txt")
    print(f"\nSummary: BS analytical = {BS_price:.6f}")
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    epsilon_array = np.array(epsilon)
    
    # Plot 1: Convergence to BS price
    ax1.semilogx(epsilon_array, mc_prices, 'o-', linewidth=2, markersize=8, label='MC Estimate')
    ax1.axhline(y=BS_price, color='r', linestyle='--', linewidth=2, label='BS Analytical')
    ax1.fill_between(epsilon_array, BS_price - epsilon_array, BS_price + epsilon_array, 
                      alpha=0.2, color='gray', label='Target accuracy ±ε')
    ax1.set_xlabel('Target Accuracy ε', fontsize=12)
    ax1.set_ylabel('Option Price', fontsize=12)
    ax1.set_title('Convergence to Black-Scholes Price', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: Percentage error vs epsilon (log-log)
    ax2.loglog(epsilon_array, pct_errors, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.loglog(epsilon_array, 100*epsilon_array, '--', linewidth=2, color='red', 
               label='100% × ε (reference)', alpha=0.7)
    ax2.set_xlabel('Target Accuracy ε', fontsize=12)
    ax2.set_ylabel('Percentage Error (%)', fontsize=12)
    ax2.set_title('Error Convergence Rate', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.invert_xaxis()
    
    # Plot 3: Computational cost scaling (ε² × Cost)
    ax3.loglog(epsilon_array, total_cost, 'o-', linewidth=2, markersize=8, color='green')
    # Add reference line for O(ε^-3) scaling
    eps_ref = epsilon_array[0]
    cost_ref = total_cost[0]
    reference_line = cost_ref * (epsilon_array / eps_ref)**(-3)
    ax3.loglog(epsilon_array, reference_line, '--', linewidth=2, color='red', 
               label='O(ε⁻³) reference', alpha=0.7)
    ax3.set_xlabel('Target Accuracy ε', fontsize=12)
    ax3.set_ylabel('ε² × Cost', fontsize=12)
    ax3.set_title('Computational Cost Scaling (Should be ~Constant)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.invert_xaxis()
    
    # Plot 4: Computation time vs samples
    ax4.loglog(samples_used, total_times, 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Number of Samples', fontsize=12)
    ax4.set_ylabel('Total Computation Time (s)', fontsize=12)
    ax4.set_title('GPU Performance Scaling', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    
    # Add text annotation with average throughput
    avg_throughput = np.mean([samples_used[i]/total_times[i] for i in range(len(samples_used))])
    ax4.text(0.05, 0.95, f'Avg throughput: {avg_throughput:,.0f} samples/s', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('mc_convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'mc_convergence_analysis.png'")
    plt.show()