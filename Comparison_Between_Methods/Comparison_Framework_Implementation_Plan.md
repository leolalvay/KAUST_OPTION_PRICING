# Comparison Framework Implementation Plan

## Handoff Document for Continuation

**Date:** December 2024  
**Purpose:** This document captures the complete context, decisions, and implementation plan for building a comparison framework between MLMC and Laplace approximation methods for estimating projected volatility in American basket option pricing. A new Claude instance should be able to read this document and implement the entire framework.

---

## Table of Contents

1. [Project Background](#1-project-background)
2. [Mathematical Context: Coefficients a and b](#2-mathematical-context-coefficients-a-and-b)
3. [Existing Codebase Structure](#3-existing-codebase-structure)
4. [Comparison Framework Design](#4-comparison-framework-design)
5. [Technical Implementation Details](#5-technical-implementation-details)
6. [Experiments to Implement](#6-experiments-to-implement)
7. [Design Decisions Made](#7-design-decisions-made)
8. [Open Questions Resolved](#8-open-questions-resolved)
9. [Files to Create](#9-files-to-create)
10. [References](#10-references)

---

## 1. Project Background

### 1.1 The Research Problem

The project prices **American basket options** on $d$ correlated assets. The challenge is the **curse of dimensionality**: solving a $d$-dimensional PDE is computationally intractable for large $d$.

### 1.2 The Solution: Markovian Projection

Using **Gyöngy's Lemma** (1986), the high-dimensional basket dynamics are "projected" onto a single 1D process $\bar{S}(t)$ that has the same marginal distributions as the true basket $S(t) = \sum_i w_i X_i(t)$.

The projected 1D SDE is:

$$d\bar{S}(t) = a(t, \bar{S}) \, dt + b(t, \bar{S}) \, dW(t)$$

where:
- $a(t, S)$ is the **drift coefficient** (trivial to compute)
- $b(t, S)$ is the **volatility coefficient** (the computational challenge)

### 1.3 The Core Computational Challenge

Estimating the projected volatility $b(t, S)$ requires computing:

$$b^2(t, s) = \mathbb{E}\left[\text{Instantaneous Basket Variance} \,\big|\, \text{Basket Value} = s\right]$$

This conditional expectation involves high-dimensional integrals over the hyperplane $\{x : P_1 \cdot x = s\}$.

### 1.4 Two Competing Methods

| Method | Approach | Reference |
|--------|----------|-----------|
| **Laplace Approximation** | Analytical approximation of the integral using Taylor expansion around the extremal point | Bayer, Häppölä, Tempone (2017) - Paper Eq. 41 |
| **MLMC + Regression** | Monte Carlo simulation with polynomial regression, using Multi-Level variance reduction | Project codebase |

### 1.5 Project Goal

Build a **comparison framework** that:
1. Runs both methods on identical parameters
2. Compares the resulting $b^2(t, S)$ surfaces
3. Measures accuracy differences
4. Measures computational cost
5. Propagates both through PDE solver to compare final option prices
6. Reports results symmetrically without declaring either as "ground truth"

---

## 2. Mathematical Context: Coefficients a and b

### 2.1 The High-Dimensional Model

Consider $d$ assets following correlated geometric Brownian motion:

$$dX_i(t) = r X_i(t) \, dt + \sigma_i X_i(t) \, dW_i(t), \quad i = 1, \ldots, d$$

where:
- $r$ is the risk-free interest rate
- $\sigma_i$ is the volatility of asset $i$
- $dW_i \cdot dW_j = \rho_{ij} \, dt$ (correlated Brownian motions)

The basket value is:

$$S(t) = P_1 \cdot \mathbf{X}(t) = \sum_{i=1}^{d} w_i X_i(t)$$

### 2.2 Coefficient $a$: The Drift (Trivial)

**Definition (Paper Eq. 12):**

$$\bar{a}(t, s) = \mathbb{E}\left[P_1 \cdot a(t, \mathbf{X}(t)) \,\big|\, P_1 \mathbf{X}(t) = s\right]$$

**Why it's trivial:** Under risk-neutral measure, $a(t, \mathbf{x}) = r\mathbf{x}$. By linearity of expectation:

$$\bar{a}(t, s) = r \cdot \mathbb{E}\left[P_1 \mathbf{X}(t) \,\big|\, P_1 \mathbf{X}(t) = s\right] = r \cdot s$$

**Result:** $a(t, S) = rS$ — hardcoded directly into PDE solver, no estimation needed.

### 2.3 Coefficient $b$: The Volatility (Hard)

**Definition (Paper Eq. 13):**

$$\bar{b}^2(t, s) = \mathbb{E}\left[(P_1 \, \mathbf{b}\mathbf{b}^T P_1^T)(t, \mathbf{X}(t)) \,\big|\, P_1 \mathbf{X}(t) = s\right]$$

For Black-Scholes:

$$b^2(t, s) = \mathbb{E}\left[\sum_{i,j} w_i w_j \sigma_i \sigma_j \rho_{ij} X_i(t) X_j(t) \,\Big|\, \text{Basket} = s\right]$$

**Why it's hard:**
1. **Non-linearity:** Volatility of a sum ≠ sum of volatilities
2. **Conditional expectation:** Must integrate over all asset configurations satisfying basket constraint
3. **Unknown density:** Sum of log-normals has no closed-form density
4. **High-dimensional integration:** Integral over $(d-1)$-dimensional hyperplane

### 2.4 Summary Table

| Coefficient | Represents | Equation | Difficulty | Method |
|:-----------:|:----------:|:--------:|:----------:|:-------|
| $a$ | Drift | $a(t, S) = r \cdot S$ | Trivial | Analytical |
| $b$ | Volatility | $b(t, S) = \sqrt{\mathbb{E}[\text{Var} \mid S = s]}$ | Hard | MLMC or Laplace |

---

## 3. Existing Codebase Structure

### 3.1 MLMC Implementation Location

**Primary files in `PDE/` folder:**

| File | Key Functions | Purpose |
|------|---------------|---------|
| `PDE/mlmc_volatility_estimation.py` | `aggregate_mlmc_coefficients()`, `estimate_basket_domain()`, `estimate_coefficients_at_level()` | Main MLMC coefficient estimation |
| `PDE/basket_simulation.py` | `construct_volatility_surface()`, `generate_polynomial_basis_pairs()`, `simulate_gbm_paths()` | Path simulation and regression |
| `PDE/american_option_pde_solver.py` | `solve_american_option()` | PDE solver with early exercise |
| `PDE/finite_difference_operators.py` | `apply_pde_operator()`, `thomas_algorithm()` | Finite difference discretisation |

**Key function signature (from project knowledge):**
```python
def aggregate_mlmc_coefficients(
    S0: np.ndarray,           # Initial asset prices, shape (d,)
    T: float,                 # Maturity time
    h0: float,                # Coarsest timestep
    r: float,                 # Risk-free rate
    cov_mat: np.ndarray,      # Correlation matrix, shape (d, d)
    vol: np.ndarray,          # Asset volatilities, shape (d,)
    max_degree: int,          # Maximum MLMC level / polynomial degree
    basket_weights: np.ndarray,  # Portfolio weights, shape (d,)
    S_min: float,             # Spatial domain lower bound
    S_max: float              # Spatial domain upper bound
) -> np.ndarray:              # Coefficient vector
```

### 3.2 Laplace Approximation Implementation Location

**Primary files in `Laplace_Replication/` folder:**

| File | Key Functions | Purpose |
|------|---------------|---------|
| `Laplace_Replication/laplace_volatility.py` | `laplace_approximation_volatility_squared()`, `compute_volatility_surface()` | Laplace approximation for $b^2(t,s)$ |
| `Laplace_Replication/pde_solver.py` | `solve_american_option_pde()`, `monte_carlo_lower_bound()` | PDE solver for Laplace method |
| `Laplace_Replication/replicate_paper.py` | `run_3d_black_scholes_test()` | Full pipeline replication |
| `Laplace_Replication/detailed_comparison.py` | `run_detailed_comparison()` | Numerical validation vs paper |

**Key function signature (from project knowledge):**
```python
def compute_volatility_surface(
    t_grid: np.ndarray,       # Time points
    s_grid: np.ndarray,       # Basket value points
    P1: np.ndarray,           # Portfolio weights
    x0: np.ndarray,           # Initial asset prices
    r: float,                 # Risk-free rate
    sigma: np.ndarray,        # Asset volatilities
    corr_chol: np.ndarray     # Cholesky factor of correlation matrix
) -> np.ndarray:              # b² surface, shape (len(t_grid), len(s_grid))
```

### 3.3 Important Notes on Existing Code

1. **`__init__.py` files:** User has confirmed these now exist (as empty files) in both `PDE/` and `Laplace_Replication/`

2. **Laplace returns $b^2$, not $b$:** The function `laplace_approximation_volatility_squared()` returns volatility SQUARED. This matches Paper Figure 1a (which shows values ~1200-1500, not ~35-40).

3. **MLMC returns coefficients:** The MLMC method returns polynomial coefficients that must be used with `construct_volatility_surface()` to create a callable $b(t, S)$ function.

4. **Domain bounds:** MLMC requires `S_min`, `S_max` from a pilot run. Laplace can evaluate at any point but should use the same domain for fair comparison.

5. **PDE solver note:** The solver uses $b^2 S^2$ in the diffusion term (characteristic of GBM). A previous bug fix ensured this $S^2$ factor is included.

### 3.4 L2_Regression Folder (Historical)

The `L2_Regression/` folder contains Amelie's original code that was refactored into `PDE/`. Key file:
- `L2_Regression/Legendre_Polynomials/Multi_Level/Fast_Programs/FML_utils.py` contains `make_c()` function

This is referenced for historical context but the `PDE/` version should be used.

---

## 4. Comparison Framework Design

### 4.1 Directory Structure

```
Comparison/
├── paths.py                        # Import helper (adds project root to sys.path)
├── config.py                       # Shared parameters (paper's Eq. 56)
│
├── methods/
│   ├── __init__.py
│   ├── mlmc_wrapper.py             # Thin wrapper around PDE/ code
│   ├── laplace_wrapper.py          # Thin wrapper around Laplace_Replication/
│   └── common.py                   # VolatilitySurfaceResult dataclass
│
├── experiments/
│   ├── __init__.py
│   ├── exp1_surface_comparison.py      # Direct b²(t,s) comparison
│   ├── exp2_mlmc_convergence.py        # Convergence study with 20 runs
│   ├── exp3_dimension_scaling.py       # d = 2, 3, 5, 10
│   ├── exp4_option_pricing.py          # End-to-end PDE comparison
│   └── exp5_parameter_sensitivity.py   # Vary σ, ρ, K, T, r
│
├── visualisation/
│   ├── __init__.py
│   ├── surface_plots.py            # 3D surface comparisons
│   ├── error_plots.py              # Heatmaps of differences
│   └── convergence_plots.py        # MLMC convergence curves
│
├── run_all_experiments.py          # Master script
├── results/                        # Output directory
│   ├── figures/
│   └── tables/
└── README.md
```

### 4.2 Common Interface

Both methods should return a standardised result object:

```python
from dataclasses import dataclass
from typing import Callable, Optional, Dict
import numpy as np

@dataclass
class VolatilitySurfaceResult:
    """Common return type for both volatility estimation methods."""
    
    # The callable surface b(t, s) -> float
    b_surface: Callable[[float, float], float]
    
    # The squared version b²(t, s) for direct comparison
    b_squared_surface: Callable[[float, float], float]
    
    # Grid data for plotting
    t_grid: np.ndarray
    s_grid: np.ndarray
    b_squared_values: np.ndarray  # Shape (len(t_grid), len(s_grid))
    
    # Metadata
    method_name: str              # "MLMC" or "Laplace"
    computation_time: float       # Wall-clock seconds
    parameters: Dict              # All input parameters for reproducibility
    
    # MLMC-specific (None for Laplace)
    coefficients: Optional[np.ndarray] = None
    n_samples: Optional[int] = None
    mlmc_levels: Optional[int] = None
```

### 4.3 Configuration

Use the paper's 3D Black-Scholes example (Equation 56):

```python
# config.py
import numpy as np
from dataclasses import dataclass
from scipy.linalg import cholesky

@dataclass
class ProblemParameters:
    """Parameters for the 3D Black-Scholes test case (Paper Eq. 56)."""
    
    # Model parameters
    r: float = 0.05                                    # Risk-free rate
    sigma: np.ndarray = None                           # Volatilities
    corr_matrix: np.ndarray = None                     # Correlation matrix
    
    # Portfolio
    P1: np.ndarray = None                              # Basket weights
    x0: np.ndarray = None                              # Initial prices
    
    # Option parameters
    T: float = 0.5                                     # Maturity
    K: float = 300.0                                   # Strike (ATM)
    option_type: str = "put"                           # Put option
    
    # Grid parameters
    N_t: int = 50                                      # Time grid points
    N_s: int = 100                                     # Spatial grid points
    
    # MLMC parameters
    max_degree: int = 3                                # Polynomial degree / MLMC levels
    h0: float = 0.1                                    # Coarsest timestep
    
    # Reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        if self.sigma is None:
            self.sigma = np.array([0.2, 0.15, 0.1])
        if self.corr_matrix is None:
            self.corr_matrix = np.array([
                [1.0, 0.8, 0.3],
                [0.8, 1.0, 0.1],
                [0.3, 0.1, 1.0]
            ])
        if self.P1 is None:
            self.P1 = np.array([1.0, 1.0, 1.0])
        if self.x0 is None:
            self.x0 = np.array([100.0, 100.0, 100.0])
    
    @property
    def d(self) -> int:
        """Number of assets."""
        return len(self.x0)
    
    @property
    def S0(self) -> float:
        """Initial basket value."""
        return float(np.dot(self.P1, self.x0))
    
    @property
    def corr_chol(self) -> np.ndarray:
        """Cholesky factor of correlation matrix."""
        return cholesky(self.corr_matrix, lower=True)


# Default parameters (paper's Eq. 56)
DEFAULT_PARAMS = ProblemParameters()
```

---

## 5. Technical Implementation Details

### 5.1 Import Strategy

Use `sys.path` manipulation to enable cross-folder imports:

**`Comparison/paths.py`:**
```python
"""
Path configuration for the Comparison module.
Adds project root to Python path for cross-folder imports.
"""
import sys
from pathlib import Path

# Get the project root (parent of Comparison/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add to Python path if not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Verify imports work
def verify_imports():
    """Test that all required modules can be imported."""
    try:
        from PDE.mlmc_volatility_estimation import aggregate_mlmc_coefficients
        from PDE.basket_simulation import construct_volatility_surface
        from Laplace_Replication.laplace_volatility import compute_volatility_surface
        print("✓ All imports verified successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    verify_imports()
```

**Usage in every script:**
```python
# At the TOP of every script in Comparison/
import paths  # This modifies sys.path

# Now imports work
from PDE.mlmc_volatility_estimation import aggregate_mlmc_coefficients
```

### 5.2 MLMC Wrapper

**`Comparison/methods/mlmc_wrapper.py`:**
```python
"""
Wrapper for MLMC volatility estimation method.
"""
import paths  # Must be first

import time
import numpy as np
from typing import Callable, Tuple

from PDE.mlmc_volatility_estimation import (
    aggregate_mlmc_coefficients,
    estimate_basket_domain,
)
from PDE.basket_simulation import (
    construct_volatility_surface,
    generate_polynomial_basis_pairs,
)

from .common import VolatilitySurfaceResult


def estimate_volatility_mlmc(
    params,  # ProblemParameters
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    random_seed: int = 42
) -> VolatilitySurfaceResult:
    """
    Estimate projected volatility surface using MLMC + polynomial regression.
    
    Parameters
    ----------
    params : ProblemParameters
        Problem configuration
    t_grid : np.ndarray
        Time points for evaluation
    s_grid : np.ndarray
        Basket values for evaluation
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    VolatilitySurfaceResult
        Standardised result object
    """
    np.random.seed(random_seed)
    start_time = time.perf_counter()
    
    # Step 1: Estimate domain bounds via pilot run
    S_min, S_max, _ = estimate_basket_domain(
        S0=params.x0,
        T=params.T,
        h0=params.h0,
        r=params.r,
        cov_mat=params.corr_matrix,
        vol=params.sigma,
        max_degree=params.max_degree,
        basket_weights=params.P1,
        N_pilot=10000
    )
    
    # Step 2: Run MLMC coefficient estimation
    coefficients = aggregate_mlmc_coefficients(
        S0=params.x0,
        T=params.T,
        h0=params.h0,
        r=params.r,
        cov_mat=params.corr_matrix,
        vol=params.sigma,
        max_degree=params.max_degree,
        basket_weights=params.P1,
        S_min=S_min,
        S_max=S_max
    )
    
    # Step 3: Construct callable volatility surface
    basis_pairs = generate_polynomial_basis_pairs(params.max_degree)
    b_surface_func = construct_volatility_surface(
        coefficients, basis_pairs, S_min, S_max, params.T, params.max_degree
    )
    
    # Step 4: Evaluate on grid
    b_squared_values = np.zeros((len(t_grid), len(s_grid)))
    for i, t in enumerate(t_grid):
        for j, s in enumerate(s_grid):
            b_val = b_surface_func(t, s)
            b_squared_values[i, j] = b_val ** 2
    
    computation_time = time.perf_counter() - start_time
    
    # Create b² callable
    def b_squared_surface(t: float, s: float) -> float:
        return b_surface_func(t, s) ** 2
    
    return VolatilitySurfaceResult(
        b_surface=b_surface_func,
        b_squared_surface=b_squared_surface,
        t_grid=t_grid,
        s_grid=s_grid,
        b_squared_values=b_squared_values,
        method_name="MLMC",
        computation_time=computation_time,
        parameters={
            "S_min": S_min,
            "S_max": S_max,
            "max_degree": params.max_degree,
            "h0": params.h0,
            "random_seed": random_seed,
        },
        coefficients=coefficients,
        n_samples=None,  # Could track this in aggregate_mlmc_coefficients
        mlmc_levels=params.max_degree + 1
    )
```

### 5.3 Laplace Wrapper

**`Comparison/methods/laplace_wrapper.py`:**
```python
"""
Wrapper for Laplace approximation volatility estimation method.
"""
import paths  # Must be first

import time
import numpy as np
from typing import Callable
from scipy.interpolate import RectBivariateSpline

from Laplace_Replication.laplace_volatility import (
    compute_volatility_surface as laplace_compute_surface,
    laplace_approximation_volatility_squared,
)

from .common import VolatilitySurfaceResult


def estimate_volatility_laplace(
    params,  # ProblemParameters
    t_grid: np.ndarray,
    s_grid: np.ndarray
) -> VolatilitySurfaceResult:
    """
    Estimate projected volatility surface using Laplace approximation.
    
    Parameters
    ----------
    params : ProblemParameters
        Problem configuration
    t_grid : np.ndarray
        Time points for evaluation
    s_grid : np.ndarray
        Basket values for evaluation
        
    Returns
    -------
    VolatilitySurfaceResult
        Standardised result object
    """
    start_time = time.perf_counter()
    
    # Compute b² surface using Laplace approximation
    b_squared_values = laplace_compute_surface(
        t_grid=t_grid,
        s_grid=s_grid,
        P1=params.P1,
        x0=params.x0,
        r=params.r,
        sigma=params.sigma,
        corr_chol=params.corr_chol
    )
    
    computation_time = time.perf_counter() - start_time
    
    # Create interpolated callable for b²
    # Handle NaN values first
    b_sq_clean = np.nan_to_num(b_squared_values, nan=np.nanmean(b_squared_values))
    interp = RectBivariateSpline(t_grid, s_grid, b_sq_clean)
    
    def b_squared_surface(t: float, s: float) -> float:
        return float(interp(t, s))
    
    def b_surface(t: float, s: float) -> float:
        return np.sqrt(max(b_squared_surface(t, s), 0))
    
    return VolatilitySurfaceResult(
        b_surface=b_surface,
        b_squared_surface=b_squared_surface,
        t_grid=t_grid,
        s_grid=s_grid,
        b_squared_values=b_squared_values,
        method_name="Laplace",
        computation_time=computation_time,
        parameters={
            "method": "Laplace approximation (Paper Eq. 41)",
        },
        coefficients=None,
        n_samples=None,
        mlmc_levels=None
    )
```

### 5.4 Accuracy Metrics

```python
def compute_accuracy_metrics(
    result_mlmc: VolatilitySurfaceResult,
    result_laplace: VolatilitySurfaceResult
) -> dict:
    """
    Compute accuracy metrics comparing two volatility surfaces.
    
    Returns dict with:
    - l2_relative_error: ||b²_MLMC - b²_Laplace||_2 / ||b²_Laplace||_2
    - linf_error: max|b²_MLMC - b²_Laplace|
    - mean_relative_error: mean(|b²_MLMC - b²_Laplace| / b²_Laplace)
    - bias: mean(b²_MLMC - b²_Laplace)
    - pointwise_relative_error: 2D array for heatmap
    """
    b2_mlmc = result_mlmc.b_squared_values
    b2_laplace = result_laplace.b_squared_values
    
    # Handle NaN values
    valid = np.isfinite(b2_mlmc) & np.isfinite(b2_laplace) & (b2_laplace > 0)
    
    diff = b2_mlmc - b2_laplace
    
    # L2 relative error
    l2_error = np.sqrt(np.sum(diff[valid]**2))
    l2_norm = np.sqrt(np.sum(b2_laplace[valid]**2))
    l2_relative = l2_error / l2_norm if l2_norm > 0 else np.inf
    
    # L-infinity error
    linf_error = np.max(np.abs(diff[valid]))
    
    # Mean relative error
    rel_errors = np.abs(diff[valid]) / b2_laplace[valid]
    mean_relative = np.mean(rel_errors)
    
    # Bias
    bias = np.mean(diff[valid])
    
    # Pointwise relative error (for heatmap)
    pointwise = np.full_like(diff, np.nan)
    pointwise[valid] = np.abs(diff[valid]) / b2_laplace[valid]
    
    return {
        "l2_relative_error": l2_relative,
        "linf_error": linf_error,
        "mean_relative_error": mean_relative,
        "bias": bias,
        "pointwise_relative_error": pointwise,
    }
```

---

## 6. Experiments to Implement

### 6.1 Experiment 1: Surface Comparison (Fixed Parameters)

**Purpose:** Direct comparison of $b^2(t, S)$ surfaces from both methods.

**Steps:**
1. Load paper's 3D parameters (Eq. 56)
2. Run MLMC with fixed seed
3. Run Laplace approximation
4. Compute accuracy metrics
5. Generate comparison plots (side-by-side surfaces, difference heatmap)

**Output:** 
- `results/figures/exp1_surface_comparison.png`
- `results/tables/exp1_metrics.md`

### 6.2 Experiment 2: MLMC Convergence Study

**Purpose:** Verify MLMC converges to Laplace result as samples increase.

**Steps:**
1. Run Laplace once (deterministic reference)
2. Run MLMC 20 times with different seeds
3. Compute mean and std of MLMC estimates
4. Vary MLMC sample size: multiply base samples by factors [0.1, 0.5, 1, 2, 5, 10]
5. Plot L2 error vs computation time
6. Verify $\mathcal{O}(\epsilon^{-2})$ complexity

**Output:**
- `results/figures/exp2_convergence_curves.png`
- `results/figures/exp2_error_vs_time.png`
- `results/tables/exp2_convergence_stats.md`

### 6.3 Experiment 3: Dimension Scaling

**Purpose:** Compare how both methods scale with number of assets $d$.

**Steps:**
1. Create problem instances for $d = 2, 3, 5, 10$
2. For each $d$:
   - Generate appropriate correlation matrices
   - Run both methods
   - Record computation time
3. Plot time vs $d$ for both methods
4. Note: Laplace optimisation is in $(d-1)$ dimensions

**Output:**
- `results/figures/exp3_dimension_scaling.png`
- `results/tables/exp3_timing.md`

### 6.4 Experiment 4: End-to-End Option Pricing

**Purpose:** Compare final American put prices using both volatility surfaces.

**Steps:**
1. Estimate $b(t, S)$ using both methods
2. Solve American option PDE with MLMC-estimated volatility
3. Solve American option PDE with Laplace-estimated volatility
4. Compare:
   - Option prices at various strikes
   - Exercise boundaries
   - Early exercise premium
5. Compare both to paper's Table 1 / Figure 4 if applicable

**Output:**
- `results/figures/exp4_option_prices.png`
- `results/figures/exp4_exercise_boundaries.png`
- `results/tables/exp4_prices.md`

### 6.5 Experiment 5: Parameter Sensitivity

**Purpose:** Check if methods agree across parameter space.

**Steps:**
1. Define parameter sweeps:
   - Volatility: $\sigma_1 \in [0.1, 0.3]$
   - Correlation: $\rho_{12} \in [0.1, 0.9]$
   - Moneyness: $K/S_0 \in [0.9, 1.1]$
   - Maturity: $T \in [0.25, 1.0]$
   - Interest rate: $r \in [0.01, 0.10]$
2. For each parameter value:
   - Run both methods
   - Compute L2 error
3. Identify regimes where methods diverge

**Output:**
- `results/figures/exp5_sensitivity_*.png` (one per parameter)
- `results/tables/exp5_sensitivity.md`

---

## 7. Design Decisions Made

The following decisions were agreed upon in conversation:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Import strategy | `sys.path` manipulation via `paths.py` | No code duplication, works immediately |
| Domain bounds | Use MLMC pilot-estimated bounds for both | Fair comparison, consistent domain |
| Random seeds | Fixed seeds for reproducibility | Essential for debugging and replication |
| Ground truth | Neither; report agreement/disagreement symmetrically | Both are approximations |
| Number of MLMC runs | 20 | Sufficient for confidence intervals |
| Base parameters | Paper's Eq. 56 (3D Black-Scholes) | Validated in original paper |
| JAX version | Not included for now | Simplicity first; can add later |
| Experiment structure | Separate scripts that share config | Modularity, can run independently |

---

## 8. Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Do PDE/ and Laplace_Replication/ have `__init__.py`? | User created empty ones |
| Fixed domain or pilot-estimated? | Pilot-estimated, shared between methods |
| How many MLMC runs for confidence intervals? | 20 runs |
| Include JAX/GPU version? | Not now, later |
| Which experiment first? | All equally important, implement all |
| Should experiments be one script or separate? | Separate scripts, shared infrastructure |

---

## 9. Files to Create

### 9.1 File Checklist

```
Comparison/
├── paths.py                              [ ] Create
├── config.py                             [ ] Create
│
├── methods/
│   ├── __init__.py                       [ ] Create
│   ├── common.py                         [ ] Create (VolatilitySurfaceResult)
│   ├── mlmc_wrapper.py                   [ ] Create
│   └── laplace_wrapper.py                [ ] Create
│
├── experiments/
│   ├── __init__.py                       [ ] Create
│   ├── exp1_surface_comparison.py        [ ] Create
│   ├── exp2_mlmc_convergence.py          [ ] Create
│   ├── exp3_dimension_scaling.py         [ ] Create
│   ├── exp4_option_pricing.py            [ ] Create
│   └── exp5_parameter_sensitivity.py     [ ] Create
│
├── visualisation/
│   ├── __init__.py                       [ ] Create
│   ├── surface_plots.py                  [ ] Create
│   ├── error_plots.py                    [ ] Create
│   └── convergence_plots.py              [ ] Create
│
├── run_all_experiments.py                [ ] Create
├── results/                              [ ] Create directory
│   ├── figures/                          [ ] Create directory
│   └── tables/                           [ ] Create directory
└── README.md                             [ ] Create
```

### 9.2 Implementation Order

1. **Infrastructure first:**
   - `paths.py` (import helper)
   - `config.py` (parameters)
   - `methods/common.py` (dataclass)
   - `methods/__init__.py`

2. **Wrappers:**
   - `methods/mlmc_wrapper.py`
   - `methods/laplace_wrapper.py`

3. **Visualisation utilities:**
   - `visualisation/surface_plots.py`
   - `visualisation/error_plots.py`
   - `visualisation/convergence_plots.py`

4. **Experiments (in order):**
   - `exp1_surface_comparison.py` (simplest, validates wrappers work)
   - `exp2_mlmc_convergence.py`
   - `exp4_option_pricing.py` (depends on PDE solver integration)
   - `exp3_dimension_scaling.py`
   - `exp5_parameter_sensitivity.py`

5. **Master script:**
   - `run_all_experiments.py`
   - `README.md`

---

## 10. References

### 10.1 Paper Reference

**Bayer, C., Häppölä, J., & Tempone, R. (2017)**  
*Implied Stopping Rules for American Basket Options from Markovian Projection*  
arXiv:1705.00558

Key equations:
- **Eq. 11:** Projected 1D SDE
- **Eq. 12:** Drift projection (trivial)
- **Eq. 13:** Volatility projection (the challenge)
- **Eq. 41:** Laplace approximation formula
- **Eq. 56:** 3D Black-Scholes test case parameters

Key figures for validation:
- **Figure 1a:** $b^2(t, s)$ surface (values ~700-2500)
- **Figure 2b:** Implied volatility
- **Figure 4:** Option prices vs strike
- **Figure 5a:** Value function surface
- **Figure 5b:** Exercise boundary

### 10.2 Code References

| Purpose | File Path |
|---------|-----------|
| MLMC coefficient aggregation | `PDE/mlmc_volatility_estimation.py` |
| Basket simulation | `PDE/basket_simulation.py` |
| Volatility surface construction | `PDE/basket_simulation.py` → `construct_volatility_surface()` |
| PDE solver (MLMC) | `PDE/american_option_pde_solver.py` |
| Laplace volatility | `Laplace_Replication/laplace_volatility.py` |
| PDE solver (Laplace) | `Laplace_Replication/pde_solver.py` |
| Paper replication | `Laplace_Replication/replicate_paper.py` |
| Detailed comparison | `Laplace_Replication/detailed_comparison.py` |

### 10.3 Theory References

- **Gyöngy, I. (1986):** *Mimicking the one-dimensional marginal distributions of processes having an Itô differential*. Probability Theory and Related Fields, 71, 501–516.

---

## Appendix A: User Context

**User background:** Wadoud is a KAUST intern with computational physics background from Imperial College. Previous projects include neutrino oscillations, Higgs analysis, quantum algorithms, and optimisation. Appreciates physics analogies and understands mathematical rigour.

**Coding preferences:**
- UK English throughout
- Descriptive variable names for non-mathematical variables
- Preserve mathematical notation (e.g., `sigma`, `rho`) where appropriate
- Comprehensive docstrings
- Credit original work

**Project context:** This comparison framework is part of a larger effort to produce publication-grade results comparing MLMC methods to the paper's Laplace approximation approach.

---

## Appendix B: Quick Start for New Claude Instance

1. **Read this document fully** to understand context
2. **Verify imports work** by checking `PDE/__init__.py` and `Laplace_Replication/__init__.py` exist
3. **Start with infrastructure** (`paths.py`, `config.py`, `methods/common.py`)
4. **Test wrappers individually** before running experiments
5. **Run Experiment 1 first** as it validates the entire pipeline
6. **Use fixed random seed 42** for reproducibility during development
7. **Save all outputs** to `results/figures/` and `results/tables/`

---

*End of handoff document.*
