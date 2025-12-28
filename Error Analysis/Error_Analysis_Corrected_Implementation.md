# Error Analysis for Regression-Based MLMC: Theory and Implementation

**Author:** Wadoud Charbak  
**Date:** December 2024  
**Purpose:** Correct the error analysis in `Comparison_Between_Methods` based on supervisor feedback  
**Status:** Implementation Plan

---

# Part 1: Theory

## 1.1 The Core Problem: We Don't Know Truth

The projected volatility coefficient $\bar{b}^2(t, s)$ is defined as a conditional expectation:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\vec{P}^{\,T} b(t, \vec{X}(t))\, b(t, \vec{X}(t))^T \vec{P} \;\Big|\; \vec{P} \cdot \vec{X}(t) = s\right]$$

This cannot be computed analytically. Both MLMC and Laplace are approximations. **Neither is ground truth.**

Therefore, any code that computes:
```python
l2_error = ||MLMC - Laplace|| / ||Laplace||
```

...and calls this "error" is **methodologically incorrect**. This measures **disagreement**, not error.

---

## 1.2 What We Can Actually Measure

### 1.2.1 MLMC Self-Convergence

MLMC produces intermediate surfaces at each level. The telescoping sum is:

$$\bar{b}^2_L = \bar{b}^2_0 + \sum_{\ell=1}^{L} (\bar{b}^2_\ell - \bar{b}^2_{\ell-1})$$

The **level correction norms** are observable:

$$m_\ell = \|\bar{b}^2_\ell - \bar{b}^2_{\ell-1}\|_{L^2}$$

If the method is converging, $m_\ell$ should decay geometrically:

$$m_\ell \sim c \cdot 2^{-\alpha \ell}$$

where $\alpha$ is the weak convergence rate.

### 1.2.2 Richardson Extrapolation for Bias

Without knowing truth, we can still estimate the remaining discretisation bias:

$$\text{Bias estimate} = \frac{m_L}{2^\alpha - 1}$$

This uses the assumption that the error decays geometrically. The finer the level, the smaller the remaining error.

### 1.2.3 Statistical Uncertainty (Across Runs)

For regression-based MLMC, all Monte Carlo samples contribute to a single polynomial fit. There's no natural "per-sample" variance within a run.

Instead, we estimate statistical uncertainty by running MLMC multiple times with different seeds:

$$\text{Statistical uncertainty} = \frac{\sigma_{\text{across runs}}}{\sqrt{K}}$$

where $K$ is the number of runs. This follows the standard $1/\sqrt{n}$ convergence.

**Your current 20-run approach is correct for this.** The problem is calling it "error to Laplace".

### 1.2.4 Method Agreement (Cross-Validation)

When comparing MLMC and Laplace, we measure **agreement**, not error:

- **L² disagreement**: How much do the surfaces differ in L² norm?
- **Correlation**: Do they have the same shape?
- **Bias**: Is one systematically higher/lower?

Neither method is privileged as truth.

### 1.2.5 End-to-End Validation

The ultimate test: do both methods produce the same **option prices**? 

Your experiments show prices agree within ~0.5%. This is the strongest validation, because prices are what we actually care about.

---

## 1.3 Correct Variance Estimation: "Square of Difference" vs "Difference of Squares"

Your supervisor flagged this. Here's the distinction:

**WRONG (difference of squares):**
$$\mathbb{E}[P_\ell^2] - \mathbb{E}[P_{\ell-1}^2]$$

This is the difference of second moments. It doesn't account for correlation.

**CORRECT (variance of coupled difference):**
$$V_\ell = \text{Var}[P_\ell - P_{\ell-1}] = \mathbb{E}[(P_\ell - P_{\ell-1})^2] - (\mathbb{E}[P_\ell - P_{\ell-1}])^2$$

This exploits the correlation from using the same Brownian paths. When $P_\ell$ and $P_{\ell-1}$ are highly correlated, their difference has much smaller variance than either individually.

**For regression-based MLMC on surfaces**, the analogous quantity is:

$$V_\ell = \|\bar{b}^2_\ell - \bar{b}^2_{\ell-1}\|_{L^2}^2$$

computed from the coupled surfaces (same random seed for both levels).

---

## 1.4 Summary: What Diagnostics We Need

| Diagnostic | What It Measures | How to Compute |
|------------|------------------|----------------|
| Level correction $m_\ell$ | Bias decay rate | $\|\bar{b}^2_\ell - \bar{b}^2_{\ell-1}\|_{L^2}$ |
| Convergence rate $\alpha$ | Weak convergence | Linear fit of $\log_2(m_\ell)$ vs $\ell$ |
| Richardson bias | Remaining discretisation error | $m_L / (2^\alpha - 1)$ |
| Across-run std | Statistical uncertainty | Std of surfaces from $K$ runs |
| Method agreement | MLMC vs Laplace difference | $\|\bar{b}^2_{\text{MLMC}} - \bar{b}^2_{\text{Laplace}}\|_{L^2}$ |
| Option price agreement | End-to-end validation | Compare prices from both methods |

---

# Part 2: Implementation Changes

## 2.1 Overview of Current Code Structure

```
Comparison_Between_Methods/
├── methods/
│   ├── common.py                 # VolatilitySurfaceResult, compute_accuracy_metrics
│   ├── mlmc_ot_estimator.py      # estimate_volatility_mlmc, make_c, make_c_ot
│   └── laplace_wrapper.py        # estimate_volatility_laplace
├── experiments/
│   ├── exp1_surface_comparison.py
│   ├── exp2_mlmc_convergence.py
│   ├── exp3_dimension_scaling.py
│   ├── exp4_option_pricing.py
│   └── exp5_parameter_sensitivity.py
├── visualisation/
│   ├── surface_plots.py
│   ├── error_plots.py
│   └── convergence_plots.py
└── results/
    ├── figures/
    └── tables/
```

---

## 2.2 File: `methods/common.py`

### 2.2.1 RENAME: `compute_accuracy_metrics` → `compute_method_agreement`

**Current (WRONG):**
```python
def compute_accuracy_metrics(b2_mlmc, b2_laplace) -> Dict[str, Any]:
    """
    Compute accuracy metrics comparing two volatility surfaces.
    ...
    Returns dict with:
    - l2_relative_error: ||b²_MLMC - b²_Laplace||_2 / ||b²_mean||_2
    ...
    """
```

**Replace with:**
```python
def compute_method_agreement(
    b2_method1: np.ndarray,
    b2_method2: np.ndarray,
    method1_name: str = "MLMC",
    method2_name: str = "Laplace"
) -> Dict[str, Any]:
    """
    Compute agreement metrics between two volatility surfaces.
    
    NOTE: This measures how much the methods AGREE, not how accurate
    either one is. Neither method is assumed to be ground truth.
    
    Parameters
    ----------
    b2_method1, b2_method2 : np.ndarray
        Volatility surfaces to compare.
    method1_name, method2_name : str
        Names for reporting.
        
    Returns
    -------
    dict with keys:
        'l2_disagreement': Relative L² norm of difference
        'linf_disagreement': Maximum absolute difference
        'rmse': Root mean square difference  
        'bias': mean(method1 - method2)
        'correlation': Pearson correlation
        'method1_name', 'method2_name': For labelling
    """
    # ... same computation as before, but with honest naming
    
    return {
        "l2_disagreement": l2_relative,      # NOT "l2_error"
        "linf_disagreement": linf_relative,  # NOT "linf_error"
        "rmse": rmse,
        "bias": bias,
        "correlation": correlation,
        "method1_name": method1_name,
        "method2_name": method2_name,
    }
```

**Changes:**
- Rename function
- Rename return keys from `*_error` to `*_disagreement`
- Update docstring to be honest about what we're measuring
- Make method names parameters for flexibility

### 2.2.2 ADD: `compute_mlmc_level_diagnostics`

**Add this new function:**
```python
def compute_mlmc_level_diagnostics(level_surfaces: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """
    Compute MLMC self-convergence diagnostics from level-by-level surfaces.
    
    Parameters
    ----------
    level_surfaces : dict
        Dictionary mapping level index to cumulative surface at that level.
        level_surfaces[0] = b²_0
        level_surfaces[1] = b²_0 + (b²_1 - b²_0) = b²_1
        ...
        level_surfaces[L] = final surface
        
    Returns
    -------
    dict with keys:
        'level_corrections': dict mapping level -> ||b²_ℓ - b²_{ℓ-1}||_{L²}
        'alpha': estimated weak convergence rate
        'bias_estimate': Richardson extrapolation bias estimate
        'convergence_quality': 'good' / 'acceptable' / 'poor'
    """
    L = max(level_surfaces.keys())
    
    # Compute level correction norms
    corrections = {}
    for ell in range(1, L + 1):
        diff = level_surfaces[ell] - level_surfaces[ell - 1]
        valid = np.isfinite(diff)
        corrections[ell] = np.sqrt(np.mean(diff[valid]**2))  # L² norm
    
    # Estimate convergence rate α via linear regression
    if len(corrections) >= 2:
        levels = np.array(list(corrections.keys()))
        log2_m = np.log2(np.array([corrections[ell] for ell in levels]))
        
        # Fit: log₂(m) = const - α * ℓ
        slope, intercept = np.polyfit(levels, log2_m, 1)
        alpha = -slope
        alpha = max(alpha, 0.5)  # Floor for robustness
    else:
        alpha = 1.0  # Default assumption
    
    # Richardson extrapolation bias estimate
    m_L = corrections[L]
    bias_estimate = m_L / (2**alpha - 1)
    
    # Assess convergence quality
    if alpha >= 0.9 and bias_estimate < 0.01:
        quality = 'good'
    elif alpha >= 0.5 and bias_estimate < 0.05:
        quality = 'acceptable'
    else:
        quality = 'poor'
    
    return {
        'level_corrections': corrections,
        'alpha': alpha,
        'bias_estimate': bias_estimate,
        'convergence_quality': quality,
        'n_levels': L + 1,
    }
```

### 2.2.3 ADD: `compute_across_run_uncertainty`

**Add this new function:**
```python
def compute_across_run_uncertainty(all_surfaces: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical uncertainty from multiple MLMC runs.
    
    For regression-based MLMC, statistical uncertainty is estimated
    by running the full procedure multiple times with different seeds.
    
    Parameters
    ----------
    all_surfaces : np.ndarray, shape (n_runs, n_t, n_s)
        Surfaces from multiple independent runs.
        
    Returns
    -------
    dict with keys:
        'mean_surface': Mean across runs
        'std_surface': Pointwise std across runs
        'relative_std': Mean relative standard deviation
        'standard_error': Standard error of the mean (std / sqrt(n))
    """
    n_runs = all_surfaces.shape[0]
    
    mean_surface = np.mean(all_surfaces, axis=0)
    std_surface = np.std(all_surfaces, axis=0, ddof=1)
    
    # Relative std (avoiding division by zero)
    valid = np.abs(mean_surface) > 1e-10
    relative_std = np.mean(std_surface[valid] / np.abs(mean_surface[valid]))
    
    # Standard error of the mean
    standard_error = relative_std / np.sqrt(n_runs)
    
    return {
        'mean_surface': mean_surface,
        'std_surface': std_surface,
        'relative_std': relative_std,
        'standard_error': standard_error,
        'n_runs': n_runs,
    }
```

### 2.2.4 UPDATE: `format_metrics_table` and `save_metrics_markdown`

**Change all occurrences of "error" to "disagreement" in output strings.**

---

## 2.3 File: `methods/mlmc_ot_estimator.py`

### 2.3.1 MODIFY: `make_c` and `make_c_ot` to return level information

**Current:**
```python
def make_c(...) -> np.ndarray:
    """..."""
    c_total = np.zeros(len(tot_degree_poly(max_deg)))
    
    for level in range(max_deg + 1):
        c_l = mlmc_level(...)
        c_total += c_l
    
    return c_total
```

**Replace with:**
```python
def make_c(..., return_level_info: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Compute full coefficient vector via MLMC telescoping sum.
    
    Parameters
    ----------
    ...
    return_level_info : bool, optional
        If True, also return intermediate level coefficients for diagnostics.
        
    Returns
    -------
    c_total : np.ndarray
        Aggregated coefficients.
    level_info : dict (only if return_level_info=True)
        Dictionary with level-by-level coefficients and cumulative sums.
    """
    c_total = np.zeros(len(tot_degree_poly(max_deg)))
    
    level_info = {'level_coefficients': {}, 'cumulative_coefficients': {}}
    
    for level in range(max_deg + 1):
        c_l = mlmc_level(...)
        level_info['level_coefficients'][level] = c_l.copy()
        c_total += c_l
        level_info['cumulative_coefficients'][level] = c_total.copy()
    
    if return_level_info:
        return c_total, level_info
    return c_total
```

**Do the same for `make_c_ot`.**

### 2.3.2 MODIFY: `estimate_volatility_mlmc` to compute level surfaces

**Add after coefficient estimation:**
```python
# If requested, compute level-by-level surfaces for diagnostics
if return_diagnostics:
    level_surfaces = {}
    for level, cumul_coefs in level_info['cumulative_coefficients'].items():
        # Evaluate cumulative surface at this level
        surface_at_level = np.zeros((len(t_grid), len(s_grid)))
        b2_func = make_b_squared(cumul_coefs, pairs, s_min, s_max, 
                                  params.T, level)  # Use level-appropriate degree
        for i, t in enumerate(t_grid):
            for j, s in enumerate(s_grid):
                surface_at_level[i, j] = b2_func(t, s)
        level_surfaces[level] = surface_at_level
```

### 2.3.3 UPDATE: `VolatilitySurfaceResult` dataclass

**Add new optional field:**
```python
@dataclass
class VolatilitySurfaceResult:
    # ... existing fields ...
    
    # NEW: Level-by-level diagnostics (optional)
    level_surfaces: Optional[Dict[int, np.ndarray]] = None
    level_diagnostics: Optional[Dict[str, Any]] = None
```

---

## 2.4 File: `experiments/exp2_mlmc_convergence.py`

This experiment needs the most significant changes.

### 2.4.1 REMOVE: "L2 error to Laplace" as primary metric

**Current (WRONG):**
```python
# Compute running average L² error to Laplace (total error)
l2_errors = []
for n in range(1, n_runs + 1):
    running_mean = np.mean(all_b_squared[:n], axis=0)
    diff = running_mean - result_laplace.b_squared_values  # Treating Laplace as truth!
    l2 = np.sqrt(np.mean(diff[valid] ** 2))
    l2_errors.append(l2)
```

**Replace with two separate analyses:**

```python
# ANALYSIS 1: MLMC Self-Convergence (using level diagnostics)
# This estimates bias via Richardson extrapolation, not comparison to Laplace

result_with_diagnostics = estimate_volatility_mlmc(
    params, t_grid, s_grid,
    return_diagnostics=True,  # NEW FLAG
    ...
)
level_diagnostics = compute_mlmc_level_diagnostics(result_with_diagnostics.level_surfaces)

print(f"MLMC Self-Convergence Diagnostics:")
print(f"  Weak convergence rate α: {level_diagnostics['alpha']:.2f}")
print(f"  Richardson bias estimate: {level_diagnostics['bias_estimate']:.2e}")
print(f"  Convergence quality: {level_diagnostics['convergence_quality']}")


# ANALYSIS 2: Statistical Uncertainty (across multiple runs)
# This is what your 20-run study already does correctly

all_b_squared = []
for i in range(n_runs):
    result = estimate_volatility_mlmc(params, t_grid, s_grid, random_seed=base_seed+i)
    all_b_squared.append(result.b_squared_values)
all_b_squared = np.array(all_b_squared)

uncertainty = compute_across_run_uncertainty(all_b_squared)

print(f"\nStatistical Uncertainty ({n_runs} runs):")
print(f"  Relative std: {uncertainty['relative_std']:.4f}")
print(f"  Standard error: {uncertainty['standard_error']:.4f}")


# ANALYSIS 3: Method Agreement (MLMC vs Laplace) - OPTIONAL, SECONDARY
# Clearly labelled as "agreement", not "error"

result_laplace = estimate_volatility_laplace(params, t_grid, s_grid)
agreement = compute_method_agreement(
    np.mean(all_b_squared, axis=0),
    result_laplace.b_squared_values
)

print(f"\nMethod Agreement (MLMC vs Laplace):")
print(f"  L² disagreement: {agreement['l2_disagreement']:.4f}")
print(f"  Correlation: {agreement['correlation']:.4f}")
print(f"  NOTE: Neither method is ground truth")
```

### 2.4.2 UPDATE: Convergence plots

**Current plot labels (WRONG):**
- "Total L² error (converges to ...)"
- "Stochastic uncertainty"

**New plot labels:**
- "Statistical uncertainty (std error)" — the thing that decreases as 1/√n
- "Method agreement (MLMC vs Laplace)" — if shown, clearly secondary

**Add new plot: MLMC Diagnostic Panel**
```python
def plot_mlmc_diagnostics(level_diagnostics, save_path=None):
    """
    Giles-style MLMC diagnostic plot.
    
    Two panels:
    1. log₂(m_ℓ) vs ℓ — should be linear with slope -α
    2. Level correction values — bar chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    corrections = level_diagnostics['level_corrections']
    levels = np.array(list(corrections.keys()))
    m_values = np.array([corrections[l] for l in levels])
    alpha = level_diagnostics['alpha']
    
    # Panel 1: Log-scale convergence
    ax1 = axes[0]
    ax1.plot(levels, np.log2(m_values), 'bo-', markersize=8, linewidth=2, label='Observed')
    
    # Reference line with estimated slope
    ref_line = np.log2(m_values[0]) - alpha * (levels - levels[0])
    ax1.plot(levels, ref_line, 'r--', linewidth=1.5, label=f'Slope = -{alpha:.2f}')
    
    ax1.set_xlabel('Level ℓ', fontsize=12)
    ax1.set_ylabel(r'$\log_2 \| b^2_\ell - b^2_{\ell-1} \|_{L^2}$', fontsize=12)
    ax1.set_title(f'Level Correction Decay (α = {alpha:.2f})', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Bar chart of corrections
    ax2 = axes[1]
    ax2.bar(levels, m_values, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Level ℓ', fontsize=12)
    ax2.set_ylabel(r'$\| b^2_\ell - b^2_{\ell-1} \|_{L^2}$', fontsize=12)
    ax2.set_title('Level Corrections (Telescoping Sum)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add Richardson bias annotation
    bias = level_diagnostics['bias_estimate']
    ax2.annotate(f'Richardson bias ≈ {bias:.2e}', 
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
```

---

## 2.5 Files: Other Experiments

### 2.5.1 `exp1_surface_comparison.py`

**Changes:**
- Replace `compute_accuracy_metrics` → `compute_method_agreement`
- Update printed output: "L² disagreement" not "L² error"
- Update saved markdown table headings

### 2.5.2 `exp3_dimension_scaling.py`

**Changes:**
- Same renaming as exp1
- Keep the comparison, but label honestly

### 2.5.3 `exp4_option_pricing.py`

**No major changes needed.** This experiment already does the right thing: comparing option prices. This is the most meaningful validation.

**Minor change:** Add a note in output emphasising this is the key validation.

### 2.5.4 `exp5_parameter_sensitivity.py`

**Changes:**
- Same renaming as exp1

---

## 2.6 File: `visualisation/error_plots.py`

### 2.6.1 RENAME file

Consider renaming to `comparison_plots.py` or `diagnostic_plots.py` since we're not measuring "error".

### 2.6.2 UPDATE function names and labels

- `plot_summary_panel`: Update axis labels from "Error" to "Difference" or "Disagreement"
- Add new function `plot_mlmc_diagnostics` (shown above)
- Add new function `plot_uncertainty_convergence` for the across-run study

---

## 2.7 File: `results/tables/*.md`

### 2.7.1 UPDATE all table headings

**Current:**
```markdown
| Metric | Value |
|--------|-------|
| L2 relative error | 0.005732 |
```

**Replace with:**
```markdown
| Metric | Value |
|--------|-------|
| L² disagreement (MLMC vs Laplace) | 0.005732 |
```

---

## 2.8 File: `results_analysis_FINAL.md`

### 2.8.1 Major narrative update needed

**Current framing (WRONG):**
> "The ~19% disagreement with Laplace is structural, not statistical..."
> "Total L² error is flat at ~18.9%..."

**New framing:**
> "MLMC and Laplace show ~X% L² disagreement in the volatility surface, but this does not indicate which method is more accurate — neither is ground truth."
> 
> "MLMC self-convergence diagnostics show α = X.X, with Richardson-estimated bias of Y.Y. This indicates the MLMC procedure is internally consistent."
>
> "Most importantly, option prices from both methods agree within 0.5%, validating that both approaches are fit for purpose."

---

## 2.9 Summary: Action Items

### Files to MODIFY:

| File | Changes |
|------|---------|
| `methods/common.py` | Rename function, add 2 new functions, update terminology |
| `methods/mlmc_ot_estimator.py` | Add `return_level_info` flag, store intermediate surfaces |
| `experiments/exp2_mlmc_convergence.py` | Major restructure: add self-convergence, rename metrics |
| `experiments/exp1_surface_comparison.py` | Rename function calls, update labels |
| `experiments/exp3_dimension_scaling.py` | Rename function calls, update labels |
| `experiments/exp5_parameter_sensitivity.py` | Rename function calls, update labels |
| `visualisation/error_plots.py` | Update labels, add diagnostic plot |
| `results_analysis_FINAL.md` | Rewrite narrative |

### Files that are FINE:

| File | Why |
|------|-----|
| `experiments/exp4_option_pricing.py` | Already does correct validation (prices) |
| `methods/laplace_wrapper.py` | No error analysis code |
| `config.py` | No error analysis code |

### NEW files to create:

| File | Purpose |
|------|---------|
| `theory/Error_Analysis_Theory.md` | Full documentation of methodology |

---

## 2.10 Validation Checklist

After implementation, verify:

- [ ] No function is called `compute_*_error` when comparing MLMC to Laplace
- [ ] All plots use "disagreement" or "difference", not "error"
- [ ] MLMC self-convergence diagnostics are computed (α, Richardson bias)
- [ ] Across-run uncertainty is computed and labelled correctly
- [ ] exp4 (option pricing) is highlighted as the key validation
- [ ] Results narrative doesn't claim either method is "correct"
- [ ] Level-by-level surfaces are stored for diagnostic purposes

---

# Part 3: Quick Reference

## What to call things

| Concept | WRONG term | CORRECT term |
|---------|------------|--------------|
| MLMC vs Laplace surface difference | "error" | "disagreement" or "difference" |
| Spread across multiple MLMC runs | "error" | "statistical uncertainty" |
| Richardson extrapolation estimate | — | "bias estimate" or "discretisation error estimate" |
| Option price difference | — | "price agreement" (and note both prices are approximations) |

## Key formulas

**Level correction norm:**
$$m_\ell = \|\bar{b}^2_\ell - \bar{b}^2_{\ell-1}\|_{L^2}$$

**Convergence rate (fit from data):**
$$\alpha \text{ such that } m_\ell \approx c \cdot 2^{-\alpha \ell}$$

**Richardson bias estimate:**
$$\text{Bias} \approx \frac{m_L}{2^\alpha - 1}$$

**Standard error (across K runs):**
$$\text{SE} = \frac{\sigma_{\text{runs}}}{\sqrt{K}}$$

---

## References

1. **Giles, M.B. (2015).** "Multilevel Monte Carlo methods." Acta Numerica, 24, 259-328.
2. **Bayer, C., Häppölä, J., Tempone, R. (2017).** "Implied Stopping Rules for American Basket Options from Markovian Projection." arXiv:1705.00558
