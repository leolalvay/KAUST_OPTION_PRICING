"""
Standard Monte Carlo Estimator Module for European Call Options.

Provides functions to compute MC costs for comparison with MLMC.
Uses the same methodology as MC_GBM.py but in a reusable format.
"""

import numpy as np
import math
from .BS_Analytic import BS_call


def run_mc_estimation(epsilon_values, T=1.0, S0=100.0, K=100.0, r=0.05, sigma=0.2,
                      N0=100000, verbose=False):
    """
    Run standard Monte Carlo estimation for European call option pricing.

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

    Returns
    -------
    dict
        Dictionary containing:
        - 'epsilon': array of epsilon values
        - 'total_cost': array of ε²×C costs (normalised)
        - 'raw_cost': array of N×steps (unnormalised total work)
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

    # Calculate analytical Black-Scholes price for comparison
    BS_price = BS_call(S0, K, T, r, sigma)

    if verbose:
        print(f"Black-Scholes analytical price: {BS_price:.6f}\n")
        print("="*70)

    for eps in epsilon_values:
        # Set time step adaptively: dt ~ epsilon / sqrt(2)
        # This balances discretisation bias with statistical error
        dt = eps / math.sqrt(2)
        steps = int(T / dt)

        # Pilot run: estimate variance with N0 samples
        payoffs = np.empty(N0)
        for i in range(N0):
            # Generate Brownian increments for the entire path
            dW_f = np.random.normal(size=steps) * math.sqrt(dt)

            # Euler-Maruyama discretisation of GBM: dS = r*S*dt + sigma*S*dW
            S = S0
            for dw in dW_f:
                S += S * r * dt + S * sigma * dw

            # Discounted payoff for European call
            payoffs[i] = math.exp(-r * T) * max(S - K, 0)

        # Estimate variance from pilot samples
        var_po = np.var(payoffs, ddof=1)

        # Determine number of samples needed for target accuracy epsilon
        # From CLT: Var[sample mean] = var_po / N
        # To achieve MSE ~ epsilon^2, we need N ~ 2 * var_po / epsilon^2
        N_needed = math.ceil(2 * var_po / eps**2)

        # Total computational cost: (samples) × (steps per sample)
        raw = N_needed * steps
        # Scaled by epsilon^2 to show the O(epsilon^-3) complexity
        eps2C = eps**2 * raw

        total_cost.append(eps2C)
        raw_cost.append(raw)

        # Store results
        MCMC_price = np.mean(payoffs)
        prices.append(MCMC_price)
        samples_needed.append(N_needed)
        steps_list.append(steps)

        if verbose:
            # Calculate absolute and percentage errors
            abs_error = abs(MCMC_price - BS_price)
            pct_error = 100 * abs_error / BS_price

            print(f"Target accuracy ε = {eps:.3f}:")
            print(f"  MC price estimate:    {MCMC_price:.6f}")
            print(f"  Absolute error:       {abs_error:.6f}")
            print(f"  Percentage error:     {pct_error:.3f}%")
            print(f"  Steps per path:       {steps}")
            print(f"  Samples needed:       {N_needed}")
            print(f"  Total cost (ε²C):     {eps2C:.2e}")
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


if __name__ == "__main__":
    # Test the module
    epsilon = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    results = run_mc_estimation(epsilon, verbose=True)
    print(f"\nBS Price: {results['bs_price']:.6f}")
