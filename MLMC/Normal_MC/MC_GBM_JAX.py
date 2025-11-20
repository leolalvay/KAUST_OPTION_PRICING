"""
GPU-Accelerated Multi-Level Monte Carlo for European Call Options.

Uses JAX for automatic vectorisation and GPU acceleration. Demonstrates
computational cost scaling of single-level Monte Carlo as a function of
target accuracy epsilon, with comparison to analytical Black-Scholes.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
import math
from BS_Analytic import BS_call
import time

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
N0 = int(10e6)       # Pilot samples for variance estimation


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
    
    total_cost = []
    
    for eps in epsilon:
        # Set time step adaptively
        dt = eps / math.sqrt(2)
        steps = int(T / dt)
        
        # Split key for this iteration
        key, subkey = random.split(key)
        
        # Pilot run with N0 samples
        print(f"Running pilot simulation for ε = {eps:.3f} with {N0} samples...")
        start_time = time.time()
        mean_pilot, var_pilot, payoffs_pilot = run_monte_carlo(
            subkey, N0, S0, K, r, vol, T, dt, steps
        )
        elapsed_time = time.time() - start_time
        # Determine required sample size
        N_needed = int(math.ceil(2 * float(var_pilot) / eps**2))
        
        # Total computational cost
        eps2C = eps**2 * N_needed * steps
        total_cost.append(float(eps2C))
        
        # Calculate errors
        abs_error = abs(float(mean_pilot) - BS_price)
        pct_error = 100 * abs_error / BS_price
        
        print(f"Target accuracy ε = {eps:.3f}:")
        print(f"  MC price estimate:    {float(mean_pilot):.6f}")
        print(f"  Absolute error:       {abs_error:.6f}")
        print(f"  Percentage error:     {pct_error:.3f}%")
        print(f"  Steps per path:       {steps}")
        print(f"  Samples needed:       {N_needed}")
        print(f"  Total cost (ε²C):     {eps2C:.2e}")
        print(f"  Computation time:     {elapsed_time:.3f} seconds")
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