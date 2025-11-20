"""
JAX-Accelerated Standard Monte Carlo Estimator for European Call Options.

Provides JAX-optimized MC cost estimation for comparison with MLMC.
Uses JIT compilation and vectorization for GPU acceleration.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap, lax
import jax.random as random
import numpy as np
import math
from .BS_Analytic import BS_call


def run_mc_estimation_jax(epsilon_values, T=1.0, S0=100.0, K=100.0, r=0.05, sigma=0.2,
                          N0=100000, verbose=False, key=None):
    """
    Run JAX-accelerated standard Monte Carlo estimation for European call option pricing.

    Parameters
    ----------
    epsilon_values : array-like
        Target accuracies to test
    T : float
        Time to maturity (years)
    S0 : float
        Initial stock price
    K : float
        Strike price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    N0 : int
        Number of pilot samples for variance estimation
    verbose : bool
        Whether to print progress
    key : jax.random.PRNGKey, optional
        Random key for reproducibility. If None, uses PRNGKey(42).

    Returns
    -------
    dict
        Dictionary containing:
        - 'epsilon': array of epsilon values
        - 'total_cost': array of eps^2 x C costs (normalised)
        - 'raw_cost': array of N x steps (unnormalised total work)
        - 'prices': array of MC price estimates
        - 'samples_needed': array of N values
        - 'steps': array of timesteps used
        - 'bs_price': analytical Black-Scholes price
    """
    epsilon_values = np.asarray(epsilon_values)
    total_cost = []
    raw_cost = []
    prices = []
    samples_needed = []
    steps_list = []

    # Initialize PRNG key if not provided
    if key is None:
        key = random.PRNGKey(42)

    # Calculate analytical Black-Scholes price for comparison
    BS_price = BS_call(S0, K, T, r, sigma)

    if verbose:
        print(f"Black-Scholes analytical price: {BS_price:.6f}\n")
        print("="*70)

    for eps in epsilon_values:
        # Set time step adaptively: dt ~ epsilon / sqrt(2)
        # This balances discretisation bias with statistical error
        dt = float(eps) / math.sqrt(2)
        steps = int(T / dt)

        # Split key for this iteration
        key, subkey = random.split(key)

        # Run JAX-accelerated pilot simulation
        mean_price, var_po = _run_mc_pilot_jax(subkey, N0, S0, K, r, sigma, T, dt, steps)

        # Determine number of samples needed for target accuracy epsilon
        # From CLT: Var[sample mean] = var_po / N
        # To achieve MSE ~ epsilon^2, we need N ~ 2 * var_po / epsilon^2
        N_needed = math.ceil(2 * float(var_po) / float(eps)**2)

        # Total computational cost: (samples) x (steps per sample)
        raw = N_needed * steps
        # Scaled by epsilon^2 to show the O(epsilon^-3) complexity
        eps2C = float(eps)**2 * raw

        total_cost.append(eps2C)
        raw_cost.append(raw)

        # Store results
        prices.append(float(mean_price))
        samples_needed.append(N_needed)
        steps_list.append(steps)

        if verbose:
            # Calculate absolute and percentage errors
            abs_error = abs(float(mean_price) - BS_price)
            pct_error = 100 * abs_error / BS_price

            print(f"Target accuracy eps = {eps:.3f}:")
            print(f"  MC price estimate:    {float(mean_price):.6f}")
            print(f"  Absolute error:       {abs_error:.6f}")
            print(f"  Percentage error:     {pct_error:.3f}%")
            print(f"  Steps per path:       {steps}")
            print(f"  Samples needed:       {N_needed}")
            print(f"  Total cost (eps^2*C): {eps2C:.2e}")
            print("-"*70)

    return {
        'epsilon': epsilon_values,
        'total_cost': np.array(total_cost),
        'raw_cost': np.array(raw_cost),
        'prices': np.array(prices),
        'samples_needed': np.array(samples_needed),
        'steps': np.array(steps_list),
        'bs_price': BS_price
    }


def _run_mc_pilot_jax(key, N, S0, K, r, sigma, T, dt, steps):
    """
    Run JAX-accelerated Monte Carlo pilot simulation.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    N : int
        Number of samples
    S0, K, r, sigma, T, dt : float
        Option parameters
    steps : int
        Number of timesteps

    Returns
    -------
    tuple
        (mean_price, variance)
    """
    # Generate all random increments: shape (N, steps)
    dW_all = random.normal(key, shape=(N, steps)) * jnp.sqrt(dt)

    # Discount factor
    disc = jnp.exp(-r * T)

    # JIT-compile the path simulation for this step count
    @jit
    def compute_payoffs(dW_all):
        def simulate_single_path(dW):
            """Simulate one GBM path and return discounted payoff."""
            def step_fn(S, dw):
                S_new = S + r * S * dt + sigma * S * dw
                return S_new, S_new

            S_final, _ = lax.scan(step_fn, S0, dW)
            payoff = jnp.maximum(S_final - K, 0.0)
            return disc * payoff

        # Vectorize over all samples
        payoffs = vmap(simulate_single_path)(dW_all)
        return payoffs

    # Compute all payoffs
    payoffs = compute_payoffs(dW_all)

    # Block until computation is done
    payoffs.block_until_ready()

    # Compute statistics
    mean_price = jnp.mean(payoffs)
    variance = jnp.var(payoffs, ddof=1)

    return mean_price, variance


if __name__ == "__main__":
    # Test the module
    epsilon = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    results = run_mc_estimation_jax(epsilon, verbose=True)
    print(f"\nBS Price: {results['bs_price']:.6f}")
