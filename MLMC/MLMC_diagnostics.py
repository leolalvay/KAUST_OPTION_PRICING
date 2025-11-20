"""
Multi-Level Monte Carlo (MLMC) Convergence Diagnostics
========================================================

Validates MLMC convergence rates for European call option under GBM.

Theory:
    - Variance decay: Var[P_l - P_{l-1}] = O(2^(-beta*l)), beta ≈ 2
    - Mean decay: |E[P_l - P_{l-1}]| = O(2^(-alpha*l)), alpha ≈ 1
    - These rates ensure O(epsilon^-2) complexity vs O(epsilon^-3) for standard MC

Physics Analogy:
    Levels are like detector resolutions. The *difference* between levels has
    much lower variance than absolute measurements (like common-mode rejection).
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Import from Normal_MC package for analytical solution
from Normal_MC import BS_call

# ============================================================================
# Problem Parameters
# ============================================================================

# Option parameters (European call)
T = 1.0                     # Time to maturity (years)
S0 = 100.0                  # Initial stock price
K = 100.0                   # Strike price (at-the-money)
r = 0.05                    # Risk-free rate
sigma = 0.2                 # Volatility

# MLMC parameters
N_pilot = 200_000           # Fixed samples per level (pilot study)
L_max = 8                   # Maximum refinement level
h0 = 0.5                    # Coarsest timestep

# Derived quantities
levels = np.arange(L_max + 1)  # l = 0, 1, ..., L_max
disc = math.exp(-r * T)        # Discount factor

# Calculate analytical Black-Scholes price for reference
BS_price = BS_call(S0, K, T, r, sigma)

# ============================================================================
# Storage Arrays for Statistics
# ============================================================================

# Fine path statistics (P_l computed with timestep h_l)
sum_Pf = np.zeros(L_max + 1)
sum_Pf_sq = np.zeros(L_max + 1)

# Coarse path statistics (P_{l-1} computed with timestep 2*h_l)
sum_Pc = np.zeros(L_max + 1)
sum_Pc_sq = np.zeros(L_max + 1)

# Correction statistics (Y_l = disc * (P_l - P_{l-1}))
sum_Y = np.zeros(L_max + 1)
sum_Y_sq = np.zeros(L_max + 1)
sum_Y_cubed = np.zeros(L_max + 1)
sum_Y_fourth = np.zeros(L_max + 1)

# ============================================================================
# Main Simulation Loop
# ============================================================================

print(f"\n{'='*70}")
print("MLMC CONVERGENCE DIAGNOSTICS")
print(f"{'='*70}")
print(f"Option: European Call (S0={S0}, K={K}, T={T})")
print(f"Model: GBM (r={r}, sigma={sigma})")
print(f"Black-Scholes analytical price: {BS_price:.6f}")
print(f"Samples per level: {N_pilot:,}")
print(f"{'='*70}\n")

for l in levels:
    # Timestep at this level: h_l = h0 * 2^(-l)
    h_fine = h0 * 2.0**(-l)
    n_steps = int(T / h_fine)
    h_coarse = 2.0 * h_fine
    
    print(f"Level {l}: h = {h_fine:.6f}, n_steps = {n_steps}")
    
    # Accumulators for this level
    acc_Pf = 0.0
    acc_Pf_sq = 0.0
    acc_Pc = 0.0
    acc_Pc_sq = 0.0
    acc_Y = 0.0
    acc_Y_sq = 0.0
    acc_Y_cubed = 0.0
    acc_Y_fourth = 0.0
    
    # Monte Carlo loop: run N_pilot simulations
    for n in range(N_pilot):
        # ====================================================================
        # Fine Path Simulation (timestep h_l)
        # ====================================================================
        
        # Generate Brownian increments: dW ~ N(0, h_l)
        dW_fine = np.random.normal(loc=0.0, scale=math.sqrt(h_fine), size=n_steps)
        
        # Euler-Maruyama scheme: S_{n+1} = S_n + r*S_n*h + sigma*S_n*dW_n
        S = S0
        for dW in dW_fine:
            S += r * S * h_fine + sigma * S * dW
        
        # Option payoff at maturity: max(S_T - K, 0)
        Pf = max(S - K, 0.0)
        
        # Accumulate fine path statistics
        acc_Pf += Pf
        acc_Pf_sq += Pf**2
        
        # ====================================================================
        # Coarse Path Simulation (timestep 2*h_l)
        # ====================================================================
        
        if l > 0:
            # KEY MLMC IDEA: Reuse same Brownian increments for correlation!
            # Coarse path uses sums of consecutive fine increments:
            #   dW_coarse[i] = dW_fine[2*i] + dW_fine[2*i+1]
            #
            # This creates high correlation between payoffs, reducing variance
            # of the difference (like common-mode noise rejection).
            dW_coarse = dW_fine.reshape(-1, 2).sum(axis=1)
            
            # Simulate coarse path with same Euler-Maruyama scheme
            S = S0
            for dW in dW_coarse:
                S += r * S * h_coarse + sigma * S * dW
            
            Pc = max(S - K, 0.0)
        else:
            # Level 0 has no coarser level, set P_{-1} = 0 by convention
            Pc = 0.0
        
        # Accumulate coarse path statistics
        acc_Pc += Pc
        acc_Pc_sq += Pc**2
        
        # ====================================================================
        # Correction (Telescoping Sum Term)
        # ====================================================================
        
        # MLMC estimator uses: E[P_L] = E[P_0] + sum_{l=1}^L E[P_l - P_{l-1}]
        # The correction Y_l = disc*(P_l - P_{l-1}) has much lower variance!
        Y = disc * (Pf - Pc)
        
        # Accumulate moments for statistical analysis
        acc_Y += Y
        acc_Y_sq += Y**2
        acc_Y_cubed += Y**3
        acc_Y_fourth += Y**4
    
    # Store accumulated statistics for this level
    sum_Pf[l] = acc_Pf
    sum_Pf_sq[l] = acc_Pf_sq
    sum_Pc[l] = acc_Pc
    sum_Pc_sq[l] = acc_Pc_sq
    sum_Y[l] = acc_Y
    sum_Y_sq[l] = acc_Y_sq
    sum_Y_cubed[l] = acc_Y_cubed
    sum_Y_fourth[l] = acc_Y_fourth

# ============================================================================
# Compute Statistics from Accumulated Sums
# ============================================================================

# Sample means
mean_Pf = sum_Pf / N_pilot
mean_Pc = sum_Pc / N_pilot
mean_Y = sum_Y / N_pilot

# Sample variances (using E[X^2] - E[X]^2)
var_Pf = sum_Pf_sq / N_pilot - mean_Pf**2
var_Pc = sum_Pc_sq / N_pilot - mean_Pc**2
var_Y = sum_Y_sq / N_pilot - mean_Y**2

# Fourth central moment for kurtosis
# Formula: m4_central = E[X^4] - 4*mu*E[X^3] + 6*mu^2*E[X^2] - 3*mu^4
# Derived from binomial expansion of (X - mu)^4
m1 = mean_Y
m2 = sum_Y_sq / N_pilot
m3 = sum_Y_cubed / N_pilot
m4 = sum_Y_fourth / N_pilot

m4_central = (
    m4 
    - 4*m1*m3 
    + 6*(m1**2)*m2 
    - 3*m1**4
)

# Kurtosis = E[(X-mu)^4] / sigma^4 (Gaussian has kurtosis = 3)
kurt_Y = m4_central / (var_Y**2)

# Print summary statistics
print(f"\n{'='*70}")
print("LEVEL STATISTICS")
print(f"{'='*70}")
for l in levels:
    if l > 0:  # Only print correction stats for levels with coarse comparison
        print(f"Level {l}:")
        print(f"  E[Pf] = {mean_Pf[l]:.4f}, Var[Pf] = {var_Pf[l]:.2f}")
        print(f"  E[Y] = {mean_Y[l]:.6f}, Var[Y] = {var_Y[l]:.6f}, "
              f"Kurt[Y] = {kurt_Y[l]:.2f}\n")

# ============================================================================
# Estimate Convergence Rates
# ============================================================================

# Fit lines to log2(variance) and log2(|mean|) vs level to estimate decay rates
levels_fit = levels[1:]  # Exclude level 0 (no coarse comparison)

# Variance decay rate: Var[Y_l] ~ 2^(-beta*l), expect beta ≈ 2 for GBM
log2_var_Y = np.log2(var_Y[1:])
beta = -np.polyfit(levels_fit, log2_var_Y, 1)[0]

# Weak convergence rate: |E[Y_l]| ~ 2^(-alpha*l), expect alpha ≈ 1 for GBM
log2_mean_Y = np.log2(np.abs(mean_Y[1:]))
alpha = -np.polyfit(levels_fit, log2_mean_Y, 1)[0]

# Compute MLMC estimate (sum of all level corrections)
# Note: mean_Y already contains discounted values
P_mlmc = np.sum(mean_Y)

print(f"{'='*70}")
print("CONVERGENCE RATE ESTIMATES")
print(f"{'='*70}")
print(f"Variance decay rate (beta): {beta:.2f} (theory: 2.0)")
print(f"Weak convergence rate (alpha): {alpha:.2f} (theory: 1.0)")
print(f"Kurtosis at finest level: {kurt_Y[L_max]:.2f} (Gaussian: 3.0)")
print(f"{'='*70}")
print(f"\nMLMC PRICE ESTIMATE COMPARISON")
print(f"{'='*70}")
print(f"MLMC estimate (L={L_max}):      {P_mlmc:.6f}")
print(f"Black-Scholes analytical:   {BS_price:.6f}")
print(f"Absolute error:             {abs(P_mlmc - BS_price):.6f}")
print(f"Relative error:             {100*abs(P_mlmc - BS_price)/BS_price:.4f}%")
print(f"{'='*70}\n")

# ============================================================================
# Diagnostic Plots
# ============================================================================

# Prepare data for plotting (skip level 0 since it has no coarse comparison)
plot_levels = levels[1:]
log2_var_Pf = np.log2(var_Pf[1:])
log2_var_Y = np.log2(var_Y[1:])
log2_mean_Pf = np.log2(np.abs(mean_Pf[1:]))
log2_mean_Y = np.log2(np.abs(mean_Y[1:]))
plot_kurt_Y = kurt_Y[1:]

# Create three-panel figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# ========== Panel 1: Variance Decay ==========
ax1.plot(plot_levels, log2_var_Pf, 'o-', color='orange', 
         label=r'$\mathrm{Var}[P_\ell]$', linewidth=2, markersize=8)
ax1.plot(plot_levels, log2_var_Y, 'x--', color='blue',
         label=r'$\mathrm{Var}[P_\ell - P_{\ell-1}]$', linewidth=2, markersize=8)

# Add theoretical reference line (slope = -2)
if len(plot_levels) >= 2:
    reference_levels = plot_levels[[0, -1]]
    slope_theory = -2.0
    intercept = log2_var_Y[0] - slope_theory * plot_levels[0]
    ax1.plot(reference_levels, slope_theory * reference_levels + intercept, 
             'k:', linewidth=1.5, label=r'Slope $-2$ (theory)')

ax1.set_xlim(0, plot_levels[-1] + 0.5)
ax1.set_ylim(-7, 10)
ax1.set_xlabel(r'Level $\ell$', fontsize=12)
ax1.set_ylabel(r'$\log_2(\mathrm{variance})$', fontsize=12)
ax1.legend(loc='lower left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Variance Decay Rate', fontsize=13, fontweight='bold')

# ========== Panel 2: Mean Decay ==========
ax2.plot(plot_levels, log2_mean_Pf, 'o-', color='orange',
         label=r'$|E[P_\ell]|$', linewidth=2, markersize=8)
ax2.plot(plot_levels, log2_mean_Y, 'x--', color='blue',
         label=r'$|E[P_\ell - P_{\ell-1}]|$', linewidth=2, markersize=8)

# Add theoretical reference line (slope = -1)
if len(plot_levels) >= 2:
    slope_theory = -1.0
    intercept = log2_mean_Y[0] - slope_theory * plot_levels[0]
    ax2.plot(reference_levels, slope_theory * reference_levels + intercept,
             'k:', linewidth=1.5, label=r'Slope $-1$ (theory)')

ax2.set_xlim(0, plot_levels[-1] + 0.5)
ax2.set_ylim(-16, 7)
ax2.set_xlabel(r'Level $\ell$', fontsize=12)
ax2.set_ylabel(r'$\log_2(|\mathrm{mean}|)$', fontsize=12)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Weak Convergence Rate', fontsize=13, fontweight='bold')

# ========== Panel 3: Kurtosis ==========
ax3.plot(plot_levels, plot_kurt_Y, 'x--', color='green', 
         linewidth=2, markersize=8)
ax3.axhline(y=3, color='red', linestyle=':', linewidth=2, 
            label='Gaussian (Kurt=3)')

ax3.set_xlim(0, plot_levels[-1] + 0.5)
ax3.set_xlabel(r'Level $\ell$', fontsize=12)
ax3.set_ylabel('Kurtosis', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_title('Heavy Tails Check', fontsize=13, fontweight='bold')

plt.tight_layout()
#plt.savefig('mlmc_diagnostics_simple.pdf', dpi=300, bbox_inches='tight')
plt.show()
