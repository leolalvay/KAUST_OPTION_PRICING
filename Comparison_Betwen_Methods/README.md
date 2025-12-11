# Comparison Framework: MLMC vs Laplace Approximation

A framework for comparing Multi-Level Monte Carlo (MLMC) and Laplace approximation methods for estimating projected volatility surfaces in American basket option pricing.

## Background

When pricing **American basket options** on $d$ correlated assets, the curse of dimensionality makes direct PDE methods intractable. Using **Gyöngy's Lemma** (1986), the high-dimensional basket dynamics can be "projected" onto a single 1D process:

$$d\bar{S}(t) = a(t, \bar{S}) \, dt + b(t, \bar{S}) \, dW(t)$$

The key computational challenge is estimating the projected volatility coefficient $b(t, S)$:

$$b^2(t, s) = \mathbb{E}\left[\text{Basket Variance} \,|\, \text{Basket Value} = s\right]$$

This framework compares two approaches:

| Method | Approach | Complexity |
|--------|----------|------------|
| **MLMC** | Monte Carlo with polynomial regression | $\mathcal{O}(\varepsilon^{-2})$ |
| **Laplace** | Analytical Taylor expansion at mode | Deterministic |

## Reference

**Bayer, C., Häppölä, J., & Tempone, R. (2017)**  
*Implied Stopping Rules for American Basket Options from Markovian Projection*  
arXiv:1705.00558

## Directory Structure

```
Comparison/
├── config.py                    # Problem parameters (Paper Eq. 56)
├── paths.py                     # Import path configuration
├── run_all_experiments.py       # Master script
├── README.md                    # This file
│
├── methods/                     # Volatility estimation wrappers
│   ├── __init__.py
│   ├── common.py               # VolatilitySurfaceResult dataclass
│   ├── mlmc_wrapper.py         # MLMC method wrapper
│   └── laplace_wrapper.py      # Laplace method wrapper
│
├── experiments/                 # Individual experiments
│   ├── __init__.py
│   ├── exp1_surface_comparison.py
│   ├── exp2_mlmc_convergence.py
│   ├── exp3_dimension_scaling.py
│   ├── exp4_option_pricing.py
│   └── exp5_parameter_sensitivity.py
│
├── visualisation/              # Plotting utilities
│   ├── __init__.py
│   ├── surface_plots.py
│   ├── error_plots.py
│   └── convergence_plots.py
│
└── results/                    # Output directory
    ├── figures/
    └── tables/
```

## Quick Start

### Run All Experiments

```bash
cd Comparison
python run_all_experiments.py
```

### Run Specific Experiment

```bash
python run_all_experiments.py --exp 1    # Surface comparison only
python run_all_experiments.py --quick    # Fast mode (fewer runs)
```

### Run Individual Experiment

```bash
python experiments/exp1_surface_comparison.py
python experiments/exp2_mlmc_convergence.py --n-runs 10
```

## Experiments

### Experiment 1: Surface Comparison
Direct comparison of $b^2(t, S)$ surfaces from both methods on the paper's 3D Black-Scholes test case.

**Outputs:**
- `exp1_surface_comparison.png`: Side-by-side 3D surfaces
- `exp1_difference_heatmap.png`: Difference visualisation
- `exp1_summary_panel.png`: Comprehensive comparison
- `exp1_metrics.md`: Accuracy statistics

### Experiment 2: MLMC Convergence Study
Verify MLMC convergence with multiple runs (default: 20).

**Outputs:**
- `exp2_convergence_curves.png`: Mean, std, and Laplace comparison
- `exp2_confidence_bands.png`: Time evolution with error bands
- `exp2_convergence_rate.png`: Verification of $1/\sqrt{n}$ convergence
- `exp2_convergence_stats.md`: Statistics table

### Experiment 3: Dimension Scaling
Test how both methods scale with basket dimension $d = 2, 3, 5, 10$.

**Outputs:**
- `exp3_time_vs_dimension.png`: Computation time bar chart
- `exp3_error_vs_dimension.png`: L2 and L∞ errors vs dimension
- `exp3_scaling_log.png`: Log-scale scaling behaviour
- `exp3_dimension_scaling.md`: Summary table

### Experiment 4: Option Pricing Comparison
Use volatility surfaces to price American basket options via PDE solving.

**Outputs:**
- `exp4_price_comparison.png`: Option prices at t=0
- `exp4_price_difference.png`: Absolute and relative differences
- `exp4_price_evolution.png`: Price dynamics over time
- `exp4_option_prices.md`: Price comparison table

### Experiment 5: Parameter Sensitivity
Study how methods respond to changes in market parameters.

**Parameters Varied:**
- Volatility ($\sigma$): 0.05 to 0.50
- Correlation ($\rho$): -0.5 to 0.9
- Interest rate ($r$): 0.00 to 0.15
- Maturity ($T$): 0.1 to 2.0
- Moneyness ($K/S_0$): 0.8 to 1.2

**Outputs:**
- `exp5_sensitivity_*.png`: Individual parameter plots
- `exp5_sensitivity_summary.png`: All parameters overview
- `exp5_parameter_sensitivity.md`: Full results table

## Default Parameters (Paper Equation 56)

```python
r = 0.05                              # Risk-free rate
sigma = [0.2, 0.15, 0.1]             # Asset volatilities
corr_matrix = [[1.0, 0.8, 0.3],      # Correlation
               [0.8, 1.0, 0.1],
               [0.3, 0.1, 1.0]]
P1 = [1, 1, 1]                       # Equal weights
x0 = [100, 100, 100]                 # Initial prices
T = 0.5                              # Maturity
K = 300                              # Strike (ATM)
```

## Accuracy Metrics

The framework computes symmetric accuracy metrics (neither method is treated as ground truth):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| L2 Relative Error | $\|b^2_{MLMC} - b^2_{Laplace}\|_2 / \|b^2_{mean}\|_2$ | Overall difference |
| Symmetric Relative | $2\|a-b\| / (\|a\|+\|b\|)$ | Unbiased comparison |
| Correlation | Pearson $r$ | Agreement pattern |
| Bias | $\mathbb{E}[b^2_{MLMC} - b^2_{Laplace}]$ | Systematic offset |

## Physics Analogy

For those with a physics background, the Markovian projection is analogous to **coarse-graining** in statistical mechanics:

- The full system has $d$ degrees of freedom (asset prices)
- We integrate out $d-1$ "irrelevant" variables
- The remaining effective theory has one degree of freedom (basket value)
- The projected volatility $b(t,S)$ is like an **effective mass** or **renormalised coupling**

The Laplace approximation is similar to the **saddle-point method** in path integrals.

## Dependencies

- NumPy, SciPy, Matplotlib
- Python 3.8+

## Author

Wadoud (KAUST Internship, 2024-2025)

Part of Project 1: Multilevel Regression + Markovian Projection for American Options
