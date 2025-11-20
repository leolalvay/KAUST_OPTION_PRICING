"""
Single-Level Monte Carlo for European Call Options under Geometric Brownian Motion.

This script demonstrates the computational cost scaling of standard Monte Carlo
for pricing European call options as a function of target accuracy epsilon.
The total cost grows as O(epsilon^-3) in the single-level regime.

Compares MC estimates against the analytical Black-Scholes solution.
"""

import numpy as np 
import math
import matplotlib.pyplot as plt
from BS_Analytic import BS_call

# Market parameters
dt = 0.0625      # Time step (will be overwritten in loop)
T = 1.0          # Time to maturity (years)
vol = 0.2        # Volatility (annualised)
r = 0.05         # Risk-free rate
K = 100          # Strike price
S0 = 100         # Initial spot price

# Monte Carlo parameters
epsilon = np.array([0.1, 0.05, 0.02, 0.01, 0.005])  # Target accuracies
N0 = 100000       # Number of pilot samples for variance estimation
total_cost = []  # Store computational cost for each epsilon

# Calculate analytical Black-Scholes price for comparison
BS_price = BS_call(S0, K, T, r, vol)
print(f"Black-Scholes analytical price: {BS_price:.6f}\n")
print("="*70)

for eps in epsilon: 
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
            S += S * r * dt + S * vol * dw 
        
        # Discounted payoff for European call
        payoffs[i] = math.exp(-r * T) * max(S - K, 0)

    # Estimate variance from pilot samples
    var_po = np.var(payoffs, ddof=1)
    
    # Determine number of samples needed for target accuracy epsilon
    # From CLT: Var[sample mean] = var_po / N
    # To achieve MSE ~ epsilon^2, we need N ~ 2 * var_po / epsilon^2
    N_needed = math.ceil(2 * var_po / eps**2)
    
    # Total computational cost: (samples) × (steps per sample)
    # Scaled by epsilon^2 to show the O(epsilon^-3) complexity
    eps2C = eps**2 * N_needed * steps
    total_cost.append(eps2C)

    # Report current accuracy level and estimated price
    MCMC_price = np.mean(payoffs)
    
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


# Save results to file for later analysis
data = np.column_stack((epsilon, total_cost))
np.savetxt(
    "mc_cost.txt", 
    data, 
    fmt="%.6e",  # Scientific notation with 6 decimal places
    header="epsilon          total_cost",
    comments=""  # Suppress default '#' comment character
)

print("\nResults saved to mc_cost.txt")
print(f"\nSummary: BS analytical = {BS_price:.6f}")