# Error Analysis Correction: Implementation Plan

**Author:** Wadoud Charbak  
**Date:** December 2024  
**Supervisor Feedback:** Current L² and L∞ error analysis is fundamentally incorrect  
**Status:** Planning Phase

---

## Executive Summary

Your supervisor identified that the current error analysis in `Comparison_Between_Methods` treats Laplace as "ground truth" and computes L² error as a simple difference between methods. This is **methodologically incorrect** for two reasons:

1. **Neither method is truth** — both MLMC and Laplace are approximations
2. **MLMC error estimation** should use the **telescoping sum structure** to estimate bias from level differences, not from comparison to another approximate method

This document provides a comprehensive plan to implement **correct error analysis** based on the foundational MLMC theory from Giles (2015) and proper cross-validation methodology.

---

## Table of Contents

1. [What's Currently Wrong](#1-whats-currently-wrong)
2. [The Correct Theory](#2-the-correct-theory)
3. [Files to Modify](#3-files-to-modify)
4. [New Files to Create](#4-new-files-to-create)
5. [Detailed Implementation Steps](#5-detailed-implementation-steps)
6. [Updated Experiment Designs](#6-updated-experiment-designs)
7. [Validation Checklist](#7-validation-checklist)
8. [References](#8-references)

---

## 1. What's Currently Wrong

### 1.1 Current Implementation in `common.py`

```python
# WRONG: Treating Laplace as "truth"
def compute_accuracy_metrics(b2_mlmc, b2_laplace):
    diff = b2_mlmc - b2_laplace
    l2_error = np.sqrt(np.mean(diff ** 2))
    l2_norm = np.sqrt(np.mean(b2_laplace ** 2))  # Using Laplace as reference!
    l2_relative = l2_error / l2_norm
```

**Problems:**
1. Normalising by Laplace values implies Laplace is the correct answer
2. This measures **method disagreement**, not **error**
3. No distinction between **bias** (systematic) and **variance** (stochastic)
4. No use of MLMC's internal structure for error estimation

### 1.2 "Square of Difference" vs "Difference of Squares"

Your supervisor's comment about this refers to **variance estimation**:

**WRONG (difference of squares):**
$$\mathbb{E}[P_\ell^2] - \mathbb{E}[P_{\ell-1}^2]$$

**CORRECT (variance of difference):**
$$\text{Var}[P_\ell - P_{\ell-1}] = \mathbb{E}[(P_\ell - P_{\ell-1})^2] - (\mathbb{E}[P_\ell - P_{\ell-1}])^2$$

The correct quantity is the **variance of the coupled difference**, which exploits the correlation between levels.

### 1.3 "Approximation to Truth"

Your supervisor mentioned approximating truth from MLMC level differences. This refers to **Richardson extrapolation**:

$$\mathbb{E}[P - P_L] \approx \frac{\mathbb{E}[P_L - P_{L-1}]}{2^\alpha - 1}$$

where $\alpha$ is the weak convergence rate. The bias can be estimated **without knowing the true value** by using the decay rate of level differences.

---

## 2. The Correct Theory

### 2.1 MLMC Error Decomposition

The Mean Square Error (MSE) of the MLMC estimator decomposes as:

$$\text{MSE} = \underbrace{\sum_{\ell=0}^{L} \frac{V_\ell}{N_\ell}}_{\text{Variance (statistical)}} + \underbrace{(\mathbb{E}[P_L] - \mathbb{E}[P])^2}_{\text{Bias}^2 \text{ (discretisation)}}$$

**Key insight:** These two components have different sources and different remedies:
- **Variance** → Reduced by more samples (averaging)
- **Bias** → Reduced by more levels (finer discretisation)

### 2.2 Estimating Bias Without Truth

From the weak convergence assumption $|\mathbb{E}[P_\ell - P]| \sim c \cdot 2^{-\alpha \ell}$:

$$\text{Bias estimate} = \frac{|m_L|}{2^\alpha - 1}$$

where $m_L = \hat{\mathbb{E}}[P_L - P_{L-1}]$ is the sample mean of the finest level correction.

**For robustness** (from Giles' mlmc.m):
$$\text{Bias estimate} = \max\left(\frac{|m_L|}{2^\alpha - 1}, \frac{|m_{L-1}|}{2^{2\alpha} - 2^\alpha}\right)$$

### 2.3 Estimating Variance Rate β

The variance decay rate $\beta$ is estimated from:
$$\log_2(V_\ell) \approx \text{const} - \beta \cdot \ell$$

Linear regression on $(\ell, \log_2 V_\ell)$ gives the slope $-\beta$.

**Expected values:**
- Euler-Maruyama with smooth payoffs: $\beta \approx 2$
- Euler-Maruyama with non-smooth payoffs: $\beta \approx 1$
- With antithetic/OT coupling: potentially $\beta > 2$

### 2.4 Cross-Validation Between Methods

When comparing MLMC and Laplace (neither is truth):

1. **Self-convergence study** for each method separately
2. **Richardson extrapolation** to estimate each method's error
3. **Bland-Altman analysis** for method agreement
4. **Option price comparison** as the ultimate validation (prices should agree even if surfaces differ)

---

## 3. Files to Modify

### 3.1 `methods/common.py`

**Current:** Computes L² error treating Laplace as truth

**Changes needed:**
1. Rename `compute_accuracy_metrics()` → `compute_method_agreement()` (honest naming)
2. Add new function `compute_mlmc_error_estimates()` for proper MLMC diagnostics
3. Add new function `compute_bland_altman_metrics()` for cross-validation
4. Add new function `compute_richardson_extrapolation()` for bias estimation

### 3.2 `methods/mlmc_ot_estimator.py` (or equivalent)

**Changes needed:**
1. Return **level-by-level statistics** (means, variances, sample counts)
2. Compute **variance of coupled differences** correctly
3. Return **convergence rate estimates** (α, β)
4. Add **diagnostic quantities** (correlation between levels, kurtosis)

### 3.3 `experiments/exp2_mlmc_convergence.py`

**Current:** Runs MLMC 20 times and compares to Laplace

**Changes needed:**
1. Remove "error to Laplace" as primary metric
2. Add **bias estimation** using Richardson extrapolation
3. Add **variance decay rate** verification
4. Separate **stochastic uncertainty** from **discretisation bias**
5. Plot proper **MLMC diagnostic plots** (log₂V vs ℓ, log₂|m| vs ℓ)

### 3.4 `visualisation/error_plots.py`

**Changes needed:**
1. Add `plot_mlmc_diagnostics()` — standard Giles-style diagnostic plot
2. Add `plot_bland_altman()` — for method comparison
3. Update existing plots to use correct terminology ("agreement" not "error")

### 3.5 `results_analysis_FINAL.md`

**Changes needed:**
1. Reframe narrative from "error" to "method agreement"
2. Add new sections on MLMC self-consistency
3. Add Richardson extrapolation results
4. Update conclusions based on proper error decomposition

---

## 4. New Files to Create

### 4.1 `methods/mlmc_diagnostics.py`

New module for MLMC-specific error analysis:

```python
"""
MLMC Diagnostic Tools

Implements proper MLMC error estimation following Giles (2015):
- Bias estimation via Richardson extrapolation
- Variance decay rate estimation
- Convergence rate verification
- Self-consistency checks

References
----------
[1] Giles, M.B. (2015). "Multilevel Monte Carlo methods."
    Acta Numerica, 24, 259-328. DOI: 10.1017/S096249291500001X
"""

def estimate_convergence_rates(level_means, level_variances, level_costs):
    """
    Estimate weak (α) and strong (β) convergence rates.
    
    Parameters
    ----------
    level_means : array, shape (L+1,)
        Sample means E[Y_ℓ] for ℓ = 0, 1, ..., L
        where Y_0 = P_0 and Y_ℓ = P_ℓ - P_{ℓ-1} for ℓ > 0
    level_variances : array, shape (L+1,)
        Sample variances Var[Y_ℓ]
    level_costs : array, shape (L+1,)
        Computational cost per sample at each level
    
    Returns
    -------
    dict with keys:
        'alpha': weak convergence rate
        'beta': variance decay rate
        'gamma': cost growth rate
        'alpha_se': standard error of alpha estimate
        'beta_se': standard error of beta estimate
    """
    pass


def estimate_bias_richardson(level_means, alpha, L):
    """
    Estimate discretisation bias using Richardson extrapolation.
    
    The bias E[P_L - P] is approximated by:
        bias ≈ m_L / (2^α - 1)
    
    where m_L = E[P_L - P_{L-1}] is the finest level correction.
    
    Parameters
    ----------
    level_means : array
        Sample means of level corrections
    alpha : float
        Weak convergence rate
    L : int
        Number of levels (finest level index)
    
    Returns
    -------
    float
        Estimated bias
    """
    pass


def compute_variance_of_difference(fine_samples, coarse_samples):
    """
    Compute Var[P_fine - P_coarse] correctly.
    
    CORRECT: Var[Y] = E[Y²] - E[Y]² where Y = P_f - P_c
    WRONG: E[P_f²] - E[P_c²]  (this is difference of second moments)
    
    Parameters
    ----------
    fine_samples : array, shape (N,)
        Samples from fine level
    coarse_samples : array, shape (N,)
        Coupled samples from coarse level (SAME random seed)
    
    Returns
    -------
    float
        Variance of the coupled difference
    """
    Y = fine_samples - coarse_samples
    return np.var(Y, ddof=1)  # Unbiased estimator


def variance_reduction_factor(fine_samples, coarse_samples):
    """
    Compute the variance reduction factor from coupling.
    
    VRF = (Var[P_f] + Var[P_c]) / Var[P_f - P_c]
    
    Should be >> 1 for effective coupling (typically 10-1000).
    """
    pass


def mlmc_consistency_check(level_means, level_variances):
    """
    Check MLMC consistency: (a - b + c) / σ should be O(1).
    
    From Giles (2015): If we define
        a = E[P_ℓ]      (from level ℓ alone)
        b = E[P_{ℓ-1}]  (from level ℓ-1 alone)  
        c = E[Y_ℓ]      (from coupled difference)
    
    Then a - b + c should be close to zero (up to sampling noise).
    
    Returns
    -------
    array
        Consistency statistic for each level (should be < 3)
    """
    pass
```

### 4.2 `methods/cross_validation.py`

New module for comparing methods without assuming truth:

```python
"""
Cross-Validation Tools for Method Comparison

When comparing two numerical methods (MLMC vs Laplace), neither of which
is ground truth, we need careful statistical methodology.

Approaches implemented:
1. Self-convergence: Each method's internal convergence
2. Richardson extrapolation: Estimate individual method errors
3. Bland-Altman analysis: Quantify method agreement
4. Option price comparison: Ultimate validation metric

References
----------
[1] Bland, J.M., Altman, D.G. (1986). "Statistical methods for 
    assessing agreement between two methods of clinical measurement."
    The Lancet, 327(8476), 307-310.
"""

def bland_altman_analysis(method1_values, method2_values):
    """
    Compute Bland-Altman statistics for method comparison.
    
    Parameters
    ----------
    method1_values : array
        Values from first method (e.g., MLMC b² surface flattened)
    method2_values : array
        Values from second method (e.g., Laplace b² surface flattened)
    
    Returns
    -------
    dict with keys:
        'mean_difference': Average (M1 - M2), a.k.a. bias
        'std_difference': Std of differences
        'lower_loa': Lower 95% limit of agreement
        'upper_loa': Upper 95% limit of agreement
        'mean_value': Average of (M1 + M2)/2 for plotting
        'differences': Individual differences for plotting
    
    Notes
    -----
    The Bland-Altman plot shows differences (y-axis) vs means (x-axis).
    If methods agree perfectly, all points lie on y=0.
    The limits of agreement (LOA) define the expected range of differences.
    """
    pass


def grid_convergence_index(values_fine, values_coarse, order, safety_factor=1.25):
    """
    Compute the Grid Convergence Index (GCI) uncertainty estimate.
    
    GCI = F_s * |f_fine - f_coarse| / (r^p - 1)
    
    where:
        F_s = safety factor (typically 1.25 for 3+ grids, 3.0 for 2 grids)
        r = refinement ratio (typically 2)
        p = order of convergence
    
    Parameters
    ----------
    values_fine : array
        Values on finer grid/discretisation
    values_coarse : array
        Values on coarser grid/discretisation  
    order : float
        Estimated order of convergence
    safety_factor : float
        GCI safety factor (default 1.25)
    
    Returns
    -------
    array
        GCI uncertainty estimate (same shape as inputs)
    
    References
    ----------
    Roache, P.J. (1994). "Perspective: A method for uniform reporting 
    of grid refinement studies." J. Fluids Engineering, 116(3), 405-413.
    """
    pass


def compute_method_agreement_metrics(surface1, surface2, grid_weights=None):
    """
    Compute agreement metrics between two surfaces.
    
    Note: This measures AGREEMENT, not ERROR (neither is truth).
    
    Parameters
    ----------
    surface1, surface2 : array, shape (N_t, N_s)
        Volatility surfaces to compare
    grid_weights : array, optional
        Quadrature weights for proper L² norm
    
    Returns
    -------
    dict with keys:
        'l2_disagreement': √(∫∫(σ₁-σ₂)² dt ds) / √(∫∫σ_mean² dt ds)
        'linf_disagreement': max|σ₁ - σ₂|
        'correlation': Pearson correlation
        'rmse': Root mean square difference
        'bias': mean(σ₁ - σ₂)
        'bland_altman': Bland-Altman statistics
    """
    pass
```

### 4.3 `theory/Error_Analysis_Theory.md`

Comprehensive theory document for the folder:

```markdown
# Error Analysis Theory for MLMC vs Laplace Comparison

## 1. Introduction

This document describes the correct error analysis framework...

[Include the full theory from the research we just did]

## 2. MLMC Error Estimation

### 2.1 The Telescoping Sum Identity
...

### 2.2 Bias Estimation via Richardson Extrapolation
...

## 3. Laplace Approximation Error

### 3.1 Saddle-Point Error Structure
...

### 3.2 Primal-Dual Bounds
...

## 4. Cross-Validation Methodology

### 4.1 When Neither Method is Truth
...

### 4.2 Bland-Altman Analysis
...

## 5. Implementation Notes
...

## References
[1] Giles (2015)...
[2] Bayer, Häppölä, Tempone (2017)...
```

---

## 5. Detailed Implementation Steps

### Phase 1: Core Infrastructure

#### Step 1.1: Create `methods/mlmc_diagnostics.py`

```python
# Key functions to implement:

def compute_level_statistics(all_level_samples):
    """
    Compute per-level statistics from MLMC run.
    
    Returns
    -------
    dict with keys per level:
        'mean': E[Y_ℓ]
        'variance': Var[Y_ℓ]  
        'n_samples': N_ℓ
        'cost': C_ℓ
        'kurtosis': Kurt[Y_ℓ]
    """
    stats = {}
    for level, samples in enumerate(all_level_samples):
        Y = samples['fine'] - samples['coarse'] if level > 0 else samples['fine']
        stats[level] = {
            'mean': np.mean(Y),
            'variance': np.var(Y, ddof=1),
            'n_samples': len(Y),
            'kurtosis': scipy.stats.kurtosis(Y, fisher=True) + 3,
        }
    return stats


def estimate_alpha_beta(level_stats):
    """
    Estimate convergence rates from level statistics.
    
    α: weak convergence (|E[Y_ℓ]| ~ 2^{-αℓ})
    β: variance decay (Var[Y_ℓ] ~ 2^{-βℓ})
    """
    levels = np.array(list(level_stats.keys()))
    
    # Exclude level 0 for rate estimation
    levels_fit = levels[levels > 0]
    
    # Fit log₂|mean| vs level for α
    log2_means = np.log2(np.abs([level_stats[l]['mean'] for l in levels_fit]))
    alpha, _ = -np.polyfit(levels_fit, log2_means, 1)
    
    # Fit log₂(variance) vs level for β  
    log2_vars = np.log2([level_stats[l]['variance'] for l in levels_fit])
    beta, _ = -np.polyfit(levels_fit, log2_vars, 1)
    
    # Enforce minimum values for robustness
    alpha = max(alpha, 0.5)
    beta = max(beta, 0.5)
    
    return alpha, beta


def estimate_bias(level_stats, alpha):
    """
    Richardson extrapolation bias estimate.
    
    bias ≈ |m_L| / (2^α - 1)
    """
    L = max(level_stats.keys())
    m_L = level_stats[L]['mean']
    
    # Robust estimate using last two levels
    bias_L = abs(m_L) / (2**alpha - 1)
    
    if L >= 1:
        m_Lm1 = level_stats[L-1]['mean']
        bias_Lm1 = abs(m_Lm1) / (2**(2*alpha) - 2**alpha)
        return max(bias_L, bias_Lm1)
    
    return bias_L


def estimate_statistical_error(level_stats):
    """
    Estimate total statistical error (variance component).
    
    Var[Y_MLMC] = Σ_ℓ V_ℓ / N_ℓ
    
    Returns standard error = √(Var[Y_MLMC])
    """
    total_var = sum(
        stats['variance'] / stats['n_samples'] 
        for stats in level_stats.values()
    )
    return np.sqrt(total_var)
```

#### Step 1.2: Update `methods/common.py`

Rename functions and add proper cross-validation metrics:

```python
# OLD (remove or deprecate)
def compute_accuracy_metrics(b2_mlmc, b2_laplace):
    ...

# NEW: Honest naming
def compute_method_agreement(b2_method1, b2_method2, method1_name="MLMC", method2_name="Laplace"):
    """
    Compute agreement metrics between two volatility surfaces.
    
    NOTE: This measures how much the methods AGREE, not how accurate 
    either one is. Neither method is assumed to be "truth".
    """
    ...
    return {
        "l2_disagreement": ...,  # NOT "l2_error"
        "linf_disagreement": ...,
        "bias": ...,  # systematic difference (method1 - method2)
        "correlation": ...,
        "bland_altman": bland_altman_analysis(b2_method1.flatten(), b2_method2.flatten()),
    }


# NEW: MLMC-specific error estimation
def compute_mlmc_error_estimates(level_statistics):
    """
    Compute MLMC error estimates using internal structure.
    
    Returns
    -------
    dict with keys:
        'alpha': weak convergence rate
        'beta': variance decay rate  
        'bias_estimate': Richardson extrapolation bias
        'statistical_error': standard error from sampling
        'total_rmse_estimate': √(bias² + statistical_error²)
    """
    from .mlmc_diagnostics import estimate_alpha_beta, estimate_bias, estimate_statistical_error
    
    alpha, beta = estimate_alpha_beta(level_statistics)
    bias = estimate_bias(level_statistics, alpha)
    stat_err = estimate_statistical_error(level_statistics)
    
    return {
        'alpha': alpha,
        'beta': beta,
        'bias_estimate': bias,
        'statistical_error': stat_err,
        'total_rmse_estimate': np.sqrt(bias**2 + stat_err**2),
    }
```

### Phase 2: Update MLMC Wrapper

#### Step 2.1: Modify `methods/mlmc_ot_estimator.py`

The MLMC wrapper must return **level-by-level statistics**:

```python
@dataclass
class MLMCResult:
    """Extended result with diagnostic information."""
    
    # Surface output
    b_squared_values: np.ndarray
    t_grid: np.ndarray
    s_grid: np.ndarray
    computation_time: float
    
    # NEW: Level-by-level statistics for error analysis
    level_statistics: Dict[int, Dict[str, float]]
    # Format: {level: {'mean': ..., 'variance': ..., 'n_samples': ..., 'cost': ...}}
    
    # NEW: Convergence rate estimates  
    alpha_estimate: float  # weak convergence rate
    beta_estimate: float   # variance decay rate
    
    # NEW: Error estimates
    bias_estimate: float
    statistical_error: float


def estimate_volatility_mlmc(..., return_diagnostics=True):
    """
    Estimate volatility surface using MLMC.
    
    Parameters
    ----------
    ...
    return_diagnostics : bool
        If True, return level-by-level statistics for error analysis.
    
    Returns
    -------
    MLMCResult
        Extended result object with diagnostics.
    """
    # ... existing computation ...
    
    if return_diagnostics:
        # Collect level statistics during MLMC aggregation
        level_stats = {}
        for level in range(max_degree + 1):
            # Get samples from this level
            fine_samples, coarse_samples = get_level_samples(level, ...)
            
            if level == 0:
                Y = fine_samples
            else:
                Y = fine_samples - coarse_samples  # Coupled difference
            
            level_stats[level] = {
                'mean': np.mean(Y, axis=0),  # Per-grid-point means
                'variance': np.var(Y, axis=0, ddof=1),
                'n_samples': len(Y),
                'cost': compute_level_cost(level, ...),
            }
        
        # Estimate convergence rates
        alpha, beta = estimate_alpha_beta(level_stats)
        bias = estimate_bias(level_stats, alpha)
        stat_err = estimate_statistical_error(level_stats)
        
        return MLMCResult(
            b_squared_values=b2_surface,
            t_grid=t_grid,
            s_grid=s_grid,
            computation_time=elapsed,
            level_statistics=level_stats,
            alpha_estimate=alpha,
            beta_estimate=beta,
            bias_estimate=bias,
            statistical_error=stat_err,
        )
```

### Phase 3: Update Experiments

#### Step 3.1: Update `experiments/exp2_mlmc_convergence.py`

This is the main experiment that needs fundamental restructuring:

```python
def run_experiment(...):
    """
    Experiment 2: MLMC Self-Convergence Study
    
    CORRECTED APPROACH:
    - DO NOT treat Laplace as "truth"
    - Use MLMC level differences to estimate bias
    - Separate bias (discretisation) from variance (statistical)
    - Use Richardson extrapolation for error estimation
    """
    
    # Run MLMC with diagnostics
    result_mlmc = estimate_volatility_mlmc(params, ..., return_diagnostics=True)
    
    # Extract level statistics
    level_stats = result_mlmc.level_statistics
    
    # Estimate convergence rates
    alpha = result_mlmc.alpha_estimate
    beta = result_mlmc.beta_estimate
    
    print(f"Weak convergence rate α: {alpha:.2f} (theory: 1.0 for Euler)")
    print(f"Variance decay rate β: {beta:.2f} (theory: 2.0 for smooth payoffs)")
    
    # Estimate errors
    bias_estimate = result_mlmc.bias_estimate
    stat_error = result_mlmc.statistical_error
    
    print(f"Estimated bias (Richardson): {bias_estimate:.2e}")
    print(f"Statistical error (std): {stat_error:.2e}")
    print(f"Total RMSE estimate: {np.sqrt(bias_estimate**2 + stat_error**2):.2e}")
    
    # Generate MLMC diagnostic plots (Giles-style)
    plot_mlmc_diagnostic_panel(level_stats, alpha, beta, ...)
    
    # OPTIONAL: Compare to Laplace for method agreement (not error)
    result_laplace = estimate_volatility_laplace(params, ...)
    agreement = compute_method_agreement(
        result_mlmc.b_squared_values, 
        result_laplace.b_squared_values
    )
    
    print(f"\nMethod Agreement (MLMC vs Laplace):")
    print(f"  L² disagreement: {agreement['l2_disagreement']:.4f}")
    print(f"  Correlation: {agreement['correlation']:.4f}")
    print(f"  Bias (MLMC - Laplace): {agreement['bias']:.4f}")
```

#### Step 3.2: Add MLMC Diagnostic Plot

```python
def plot_mlmc_diagnostic_panel(level_stats, alpha, beta, save_path=None):
    """
    Standard Giles-style MLMC diagnostic plot.
    
    Four panels:
    1. log₂|E[Y_ℓ]| vs ℓ (weak convergence)
    2. log₂(Var[Y_ℓ]) vs ℓ (variance decay)
    3. E[Y_ℓ] vs ℓ (level corrections)
    4. Consistency check
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    levels = np.array(list(level_stats.keys()))
    means = np.array([level_stats[l]['mean'] for l in levels])
    variances = np.array([level_stats[l]['variance'] for l in levels])
    
    # Panel 1: Weak convergence
    ax1 = axes[0, 0]
    ax1.plot(levels[1:], np.log2(np.abs(means[1:])), 'bo-', label='|E[Y_ℓ]|')
    ax1.plot(levels[1:], -alpha * levels[1:] + np.log2(np.abs(means[1])) + alpha, 
             'r--', label=f'Slope = -{alpha:.2f}')
    ax1.set_xlabel('Level ℓ')
    ax1.set_ylabel('log₂|E[Y_ℓ]|')
    ax1.set_title(f'Weak Convergence (α = {alpha:.2f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Variance decay
    ax2 = axes[0, 1]
    ax2.plot(levels[1:], np.log2(variances[1:]), 'bo-', label='Var[Y_ℓ]')
    ax2.plot(levels[1:], -beta * levels[1:] + np.log2(variances[1]) + beta,
             'r--', label=f'Slope = -{beta:.2f}')
    ax2.set_xlabel('Level ℓ')
    ax2.set_ylabel('log₂(Var[Y_ℓ])')
    ax2.set_title(f'Variance Decay (β = {beta:.2f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Level corrections
    ax3 = axes[1, 0]
    ax3.bar(levels, means, color='steelblue', alpha=0.7)
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.set_xlabel('Level ℓ')
    ax3.set_ylabel('E[Y_ℓ]')
    ax3.set_title('Level Corrections (Telescoping Sum)')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Kurtosis (normality check)
    ax4 = axes[1, 1]
    kurtosis = [level_stats[l].get('kurtosis', 3) for l in levels]
    ax4.bar(levels, kurtosis, color='orange', alpha=0.7)
    ax4.axhline(3, color='r', linestyle='--', label='Gaussian (κ=3)')
    ax4.axhline(100, color='r', linestyle=':', label='Warning threshold')
    ax4.set_xlabel('Level ℓ')
    ax4.set_ylabel('Kurtosis')
    ax4.set_title('Sample Kurtosis (Normality Check)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
```

### Phase 4: Update Visualisation

#### Step 4.1: Add Bland-Altman Plot

```python
def plot_bland_altman(method1_values, method2_values, 
                      method1_name="MLMC", method2_name="Laplace",
                      save_path=None):
    """
    Bland-Altman plot for method comparison.
    
    X-axis: Mean of two methods (M1 + M2)/2
    Y-axis: Difference (M1 - M2)
    
    Horizontal lines show mean difference and 95% limits of agreement.
    """
    m1 = method1_values.flatten()
    m2 = method2_values.flatten()
    
    # Filter valid values
    valid = np.isfinite(m1) & np.isfinite(m2) & (m1 > 0) & (m2 > 0)
    m1, m2 = m1[valid], m2[valid]
    
    means = (m1 + m2) / 2
    diffs = m1 - m2
    
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    lower_loa = mean_diff - 1.96 * std_diff
    upper_loa = mean_diff + 1.96 * std_diff
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(means, diffs, alpha=0.3, s=10, c='steelblue')
    
    # Mean difference line
    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2,
               label=f'Mean diff: {mean_diff:.2f}')
    
    # Limits of agreement
    ax.axhline(upper_loa, color='red', linestyle='--', linewidth=1.5,
               label=f'+1.96 SD: {upper_loa:.2f}')
    ax.axhline(lower_loa, color='red', linestyle='--', linewidth=1.5,
               label=f'-1.96 SD: {lower_loa:.2f}')
    
    # Zero line
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    ax.set_xlabel(f'Mean of {method1_name} and {method2_name}', fontsize=12)
    ax.set_ylabel(f'{method1_name} − {method2_name}', fontsize=12)
    ax.set_title('Bland-Altman Plot: Method Agreement', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
```

### Phase 5: Update Documentation

#### Step 5.1: Create `theory/Error_Analysis_Theory.md`

Full theory document with:
1. MLMC error decomposition
2. Richardson extrapolation
3. Variance estimation (correct formula)
4. Cross-validation methodology
5. References to Giles (2015) and Bayer et al. (2017)

#### Step 5.2: Update `results_analysis_FINAL.md`

Reframe the entire narrative:
- Change "L² error" to "L² disagreement" or "method discrepancy"
- Add new section on MLMC self-convergence
- Add Richardson extrapolation results
- Add Bland-Altman analysis
- Update conclusions

#### Step 5.3: Update `README.md`

Add section explaining the error analysis methodology.

---

## 6. Updated Experiment Designs

### 6.1 Experiment 1: Surface Comparison (Updated)

**Old approach:** Compare surfaces, report "error" to Laplace
**New approach:** Compare surfaces, report "agreement" metrics, acknowledge neither is truth

### 6.2 Experiment 2: MLMC Self-Convergence (Major Update)

**Old approach:** Run MLMC 20 times, compare to Laplace
**New approach:**
1. Run MLMC with level-by-level diagnostics
2. Estimate α, β from level differences
3. Estimate bias via Richardson extrapolation
4. Estimate statistical error from variance decomposition
5. Generate Giles-style diagnostic plots
6. (Optional) Compare to Laplace as secondary validation

### 6.3 Experiment 3: Dimension Scaling (Minor Update)

**Change:** Report method disagreement, not "error"

### 6.4 Experiment 4: Option Pricing (Important)

**Add:** This is the **ultimate validation**. Even if surfaces disagree, if option prices agree, both methods are practically equivalent for their intended purpose.

### 6.5 Experiment 5: Parameter Sensitivity (Minor Update)

**Change:** Report method disagreement, not "error"

---

## 7. Validation Checklist

After implementation, verify:

### 7.1 MLMC Diagnostics

- [ ] α estimate is ~1.0 for Euler-Maruyama
- [ ] β estimate is ~1.0 to 2.0 depending on payoff smoothness
- [ ] log₂|E[Yₗ]| vs ℓ is approximately linear
- [ ] log₂(Var[Yₗ]) vs ℓ is approximately linear
- [ ] Kurtosis < 100 at all levels (otherwise CLT unreliable)
- [ ] Variance reduction factor >> 1 (coupling is working)

### 7.2 Richardson Extrapolation

- [ ] Bias estimate is reasonable (not larger than solution itself)
- [ ] Using robust estimate from last 2-3 levels
- [ ] α ≥ 0.5 enforced for stability

### 7.3 Cross-Validation

- [ ] Bland-Altman plot shows no obvious trends
- [ ] Method disagreement is interpretable (not presented as "error")
- [ ] Option prices agree even if surfaces disagree

### 7.4 Documentation

- [ ] Theory document explains all formulas with references
- [ ] Results analysis uses correct terminology
- [ ] README explains the methodology

---

## 8. References

### Primary Sources

1. **Giles, M.B. (2015).** "Multilevel Monte Carlo methods."  
   Acta Numerica, 24, 259-328.  
   DOI: 10.1017/S096249291500001X  
   [PDF](https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf)  
   *The foundational MLMC paper with complete error analysis theory.*

2. **Bayer, C., Häppölä, J., Tempone, R. (2017).** "Implied Stopping Rules for American Basket Options from Markovian Projection."  
   arXiv:1705.00558  
   *The paper we're trying to improve upon.*

### Supporting Sources

3. **Giles, M.B. (2008).** "Multilevel Monte Carlo Path Simulation."  
   Operations Research, 56(3), 607-617.  
   *Original MLMC paper for SDEs.*

4. **Bland, J.M., Altman, D.G. (1986).** "Statistical methods for assessing agreement between two methods of clinical measurement."  
   The Lancet, 327(8476), 307-310.  
   *Methodology for comparing methods without ground truth.*

5. **Roache, P.J. (1994).** "Perspective: A method for uniform reporting of grid refinement studies."  
   Journal of Fluids Engineering, 116(3), 405-413.  
   *Grid Convergence Index methodology.*

### Code References

6. **Giles' MLMC code:** https://people.maths.ox.ac.uk/gilesm/mlmc/  
   *Reference implementations with diagnostic functions.*

---

## Summary

The key changes are:

1. **Stop treating Laplace as truth** — rename "error" to "disagreement"
2. **Use MLMC's internal structure** — level differences give bias estimates
3. **Compute variance correctly** — variance of (P_fine - P_coarse), not difference of variances
4. **Add Richardson extrapolation** — estimate bias without knowing truth
5. **Add proper diagnostics** — α, β estimation, Giles-style plots
6. **Cross-validate properly** — Bland-Altman analysis, option price comparison

This transforms your error analysis from "wrong" to "publication-quality rigorous".
