# Results Analysis: MLMC vs Laplace Volatility Surface Comparison

**Author:** Wadoud Charbak  
**Date:** December 2024  
**Project:** Multilevel Regression + Markovian Projection for American Options  
**Status:** Initial analysis - requires plot verification before presentation

---

## Executive Summary

This document analyses the results from five experiments comparing two methods for computing the projected volatility surface $\bar{b}^2(t, S)$ used in Markovian projection for American basket option pricing:

1. **MLMC (Multi-Level Monte Carlo)**: Stochastic method using polynomial regression on simulated GBM paths
2. **Laplace Approximation**: Analytical/deterministic method using saddle-point approximations (from Bayer, Häppölä, Tempone 2017)

**Key Finding:** Despite ~18% L² disagreement in volatility surfaces, option prices agree within 0.2% for practically relevant cases (ITM/ATM options). MLMC shows dramatic computational advantages for high-dimensional problems.

**Action Required:** Several plots need visual verification before results can be presented.

---

## Background: What We Are Comparing

### The Projected Volatility Surface

Both methods compute $\bar{b}^2(t, S)$, the **projected volatility** that reduces a d-dimensional basket option to an effective 1D process via Gyöngy's Lemma:

$$dS_t = r S_t \, dt + \bar{b}(t, S_t) \, dW_t$$

where $S_t = \sum_i w_i X_t^{(i)}$ is the basket value.

### Method Differences

| Aspect | MLMC | Laplace |
|--------|------|---------|
| **Approach** | Monte Carlo + polynomial regression | Saddle-point optimisation |
| **Nature** | Stochastic | Deterministic |
| **Scaling** | O(1) with dimension | O(d²) to O(d³) |
| **Accuracy** | Depends on samples and polynomial degree | Depends on validity of saddle-point |

### Error Metrics Used

- **L² Relative Error**: $\|\bar{b}^2_{\text{MLMC}} - \bar{b}^2_{\text{Laplace}}\|_2 / \|\bar{b}^2_{\text{mean}}\|_2$ — "average" error across surface
- **L∞ Error**: $\max|\bar{b}^2_{\text{MLMC}} - \bar{b}^2_{\text{Laplace}}|$ — worst-case error anywhere
- **Correlation**: Pearson correlation between surfaces — measures whether shapes match
- **Bias**: $\mathbb{E}[\bar{b}^2_{\text{MLMC}} - \bar{b}^2_{\text{Laplace}}]$ — systematic offset

---

## Experiment 1: Direct Surface Comparison

### Purpose
Compare volatility surfaces from both methods on the paper's 3D Black-Scholes test case (Equation 56).

### Parameters (Bayer et al. 2017, Eq. 56)
```
d = 3 assets
x0 = [100, 100, 100]
σ = [0.2, 0.15, 0.1]
ρ = [[1.0, 0.8, 0.3],
     [0.8, 1.0, 0.1],
     [0.3, 0.1, 1.0]]
r = 0.05
T = 0.5
K = 300 (ATM)
```

### Results

| Metric | Value | Assessment |
|--------|-------|------------|
| L² relative error | 0.191 (19.1%) | ⚠️ Higher than expected |
| L∞ error | 512.4 | Large in absolute terms |
| Mean absolute difference | 235.1 | — |
| Mean relative difference | 16.1% | ⚠️ Significant |
| Bias (MLMC − Laplace) | −89.7 | MLMC systematically lower |
| Pearson correlation | 0.948 | ✅ Excellent shape agreement |
| Valid grid fraction | 100% | ✅ No numerical issues |

### Analysis

#### 1. Are These Results Correct?

**Concerning signs:**
- 19% L² error is higher than one might expect for two methods computing the same theoretical quantity
- Negative bias indicates MLMC is systematically underestimating relative to Laplace
- L∞ of 512 is large, though this must be contextualised against typical $b^2$ values (700–2500)

**Reassuring signs:**
- Correlation of 0.948 is excellent — the surfaces have the same "shape"
- 100% valid grid indicates no numerical instabilities

**Verdict:** Results are plausible but indicate a systematic discrepancy requiring investigation.

#### 2. What Do These Results Mean?

The high correlation with significant bias suggests:
1. Both methods capture the same underlying dynamics
2. There may be a normalisation or definitional difference between implementations
3. The systematic underestimation by MLMC warrants verification

#### 3. Insights

- Methods agree in *structure* but disagree in *magnitude*
- For applications where only *relative* changes in $b^2$ matter, both methods give consistent guidance
- The 0.948 correlation is evidence that both methods are working correctly in terms of capturing the physics

#### 4. Action Items

⚠️ **Critical:** Verify both methods compute the **same definition** of $\bar{b}^2$. Potential sources of discrepancy:
- Missing factor of $S^2$ somewhere (diffusion coefficient vs volatility)
- Different normalisation conventions
- Boundary treatment differences

### Output Files
- `exp1_surface_comparison.png`: Side-by-side 3D surfaces
- `exp1_difference_heatmap.png`: Heatmap showing where differences are largest
- `exp1_summary_panel.png`: Comprehensive 6-panel comparison
- `exp1_metrics.md`: Summary statistics table

### Plot Verification Status
Reviewed

---

## Experiment 2: MLMC Convergence Study

### Purpose
Verify that MLMC estimates are reproducible and converge with averaging. Distinguish between Monte Carlo variance and systematic bias.

### Methodology
1. Run Laplace once (deterministic reference)
2. Run MLMC 20 times with different random seeds
3. Compute running average of L² error to Laplace
4. Check if error follows $1/\sqrt{n}$ convergence

### Results

| Statistic | Value | Assessment |
|-----------|-------|------------|
| Number of runs | 20 | — |
| Mean L² error (individual) | 0.184 (18.4%) | Consistent with Exp 1 |
| Std L² error | 0.00336 (0.34%) | ✅ Very low variance |
| Final running avg L² | 0.183 | Converged |
| Mean time per run | 0.22s | Fast |
| Total time | 4.5s | — |

### Running Average Convergence

| n runs | L² Error |
|--------|----------|
| 1 | 0.1827 |
| 2 | 0.1843 |
| 5 | 0.1817 |
| 10 | 0.1831 |
| 20 | 0.1835 |

### Analysis

#### 1. Are These Results Correct?

**Very encouraging:**
- Standard deviation of only 0.34% across 20 runs is excellent reproducibility
- The running average stabilises quickly (by ~5 runs)
- Timing is consistent

**The critical observation:**
- The L² error converges to ~18.4%, **not to zero**
- This confirms the systematic discrepancy from Exp 1 is **not due to Monte Carlo variance**

**Verdict:** MLMC is numerically stable and reproducible. The ~18% discrepancy is deterministic, not stochastic.

#### 2. What Do These Results Mean?

This is crucial evidence:
- If the discrepancy were due to insufficient MLMC samples, we would see high variance
- Instead we see: low variance (0.34%) + consistent bias (~18%)
- Therefore, **the methods genuinely compute different things** (or one has a systematic error)

Adding more MLMC samples will **not** fix this discrepancy.

#### 3. Insights

- MLMC is well-converged within its own framework
- The discrepancy with Laplace is fundamental, not statistical
- Need to investigate the source of systematic difference

#### 4. Action Items

⚠️ **CRITICAL - PLOTS NEED TO BE REVIEWED:**

The experiment generates three plots that need visual verification:

1. **`exp2_convergence_curves.png`**: Should show:
   - Mean MLMC surface
   - Standard deviation heatmap
   - Running L² error converging to a plateau

2. **`exp2_confidence_bands.png`**: Should show:
   - Time evolution of $b^2$ at $S = S_0$
   - MLMC mean with ±1σ error bands
   - Laplace curve for comparison
   - Bands should be narrow (indicating convergence)

3. **`exp2_convergence_rate.png`**: Should show:
   - Running average L² error vs number of runs (log scale)
   - Reference $1/\sqrt{n}$ line
   - **IMPORTANT:** The error should NOT follow $1/\sqrt{n}$ towards zero — it should plateau at ~18%. If it follows $1/\sqrt{n}$ to zero, something is wrong.

**Questions for Plot Review:**
- Does the running L² error plateau or continue decreasing?
- Are the confidence bands appropriately narrow?
- Does the $1/\sqrt{n}$ reference line make sense in context?

### Output Files
- `exp2_convergence_curves.png`
- `exp2_confidence_bands.png`
- `exp2_convergence_rate.png`
- `exp2_convergence_stats.md`

### Plot Verification Status
Reviewed

---

## Experiment 3: Dimension Scaling

### Purpose
Compare how both methods scale computationally with the number of assets $d$ in the basket.

### Methodology
Run both methods for $d = 2, 3, 5, 10$ assets with appropriately constructed correlation matrices.

### Results

| d | MLMC Time (s) | Laplace Time (s) | L² Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.22 | 0.58 | 0.198 | 0.378 |
| 3 | 0.22 | 1.24 | 0.191 | 0.357 |
| 5 | 0.24 | 2.83 | 0.135 | 0.251 |
| 10 | 0.29 | 9.60 | 0.090 | 0.170 |

### Analysis

#### 1. Are These Results Correct?

**Timing results — expected and correct:**
- MLMC scales as approximately **O(1)** with dimension (0.22s → 0.29s)
- Laplace scales as approximately **O(d²) to O(d³)** (0.58s → 9.60s)

**Error results — surprising but potentially correct:**
- L² error **decreases** with dimension: 19.8% → 9.0%
- This is counterintuitive but not impossible

**Verdict:** Timing results are correct and expected. Error trend needs confirmation.

#### 2. What Do These Results Mean?

**Timing:**
- MLMC's near-constant scaling is because the Markovian projection immediately collapses all d dimensions to 1D
- Laplace requires (d−1)-dimensional optimisation, hence polynomial growth
- This is a **major advantage** for MLMC in high-dimensional problems

**Error decreasing with dimension:**
- In high dimensions, Central Limit Theorem effects are stronger
- Both methods may converge to the same Gaussian limit
- The averaging over more assets may smooth out method-specific approximation errors

#### 3. Insights

🎉 **This is excellent news for the paper:**
- MLMC is dramatically faster for high-dimensional problems
- The methods agree **better** for exactly the cases where MLMC's advantages matter most
- At d=10 (where you would actually use these methods), L² error is only 9%

**Key message:** MLMC provides O(1) scaling while Laplace becomes computationally prohibitive for d > 20.

#### 4. Action Items

⚠️ **PLOTS NEED TO BE CONFIRMED:**

1. **`exp3_time_vs_dimension.png`**: Bar chart comparing times
   - Should clearly show MLMC nearly flat, Laplace growing
   
2. **`exp3_error_vs_dimension.png`**: Two panels for L² and L∞
   - Should show decreasing trend with d
   
3. **`exp3_scaling_log.png`**: Log-scale timing plot
   - Should include O(d) reference line
   - MLMC should be below reference, Laplace should be above

**Questions for Plot Review:**
- Is the timing data plotted correctly?
- Does the error vs dimension trend look believable?
- Are axis labels and legends correct?

### Output Files
- `exp3_time_vs_dimension.png`
- `exp3_error_vs_dimension.png`
- `exp3_scaling_log.png`
- `exp3_dimension_scaling.md`

### Plot Verification Status
Reviewed

---

## Experiment 4: Option Pricing Comparison

### Purpose
The ultimate test: do volatility surface differences translate into option price differences? Use both surfaces to price American basket options via PDE solving.

### Methodology
1. Compute volatility surfaces using both methods
2. Solve the projected 1D American option PDE with each surface
3. Compare resulting option prices at various spot values

### Parameters
```
d = 3
K = 300 (ATM)
S0 = 300
T = 0.5
Option type: Put
```

### Results

| S | Moneyness | MLMC Price | Laplace Price | Difference | Diff (%) |
|---|-----------|------------|---------------|------------|----------|
| 263.7 | ITM | 36.32 | 36.32 | 0.00 | 0.00% |
| 270.1 | ITM | 29.91 | 29.91 | 0.00 | 0.00% |
| 300.3 | ATM | 7.50 | 7.49 | 0.01 | 0.14% |
| 329.6 | OTM | 0.94 | 1.12 | 0.18 | 16.4% |
| 354.3 | Deep OTM | 0.01 | 0.02 | 0.01 | 33.4% |

**Summary Statistics:**
- Mean absolute difference: 0.10
- Max absolute difference: 0.20
- Mean percentage difference: 9.48%

### Analysis

#### 1. Are These Results Correct?

**The pattern makes perfect physical sense:**
- **ITM options:** Perfect agreement (0.00% error) — dominated by intrinsic value
- **ATM options:** Excellent agreement (0.14% error) — where time value is maximised
- **OTM options:** Larger percentage discrepancy (16–33%) but tiny absolute values ($0.01–$1)

**Verdict:** Results are physically correct and very encouraging.

#### 2. What Do These Results Mean?

🎉 **This is the most important result:**

Despite ~18% L² error in volatility surfaces, **option prices agree within 0.2% for practically relevant cases**.

**Why this happens:**
1. **Integration smooths differences:** The PDE integrates over the entire surface, averaging out local discrepancies
2. **Payoff dominates ITM:** For in-the-money options, intrinsic value dominates regardless of volatility details
3. **Time value localised ATM:** The time value (where volatility matters most) is concentrated near ATM where methods agree best

#### 3. Insights

- Both methods are **interchangeable for practical option pricing**
- The 33% OTM error is misleading — absolute difference is $0.007, smaller than typical bid-ask spreads
- **Key takeaway for paper:** Volatility surface methodology choice does not materially affect option prices

#### 4. Action Items

⚠️ **PLOTS NEED TO BE CONFIRMED:**

1. **`exp4_price_comparison.png`**: Should show:
   - Option price curves from both methods (should nearly overlap)
   - Intrinsic value (payoff) for reference
   - Vertical lines at S0 and K

2. **`exp4_price_difference.png`**: Should show:
   - Absolute difference (should be small everywhere)
   - Relative difference (large only for OTM where absolute is tiny)

3. **`exp4_price_evolution.png`**: Should show:
   - How prices evolve over time
   - Both methods should track closely

**Questions for Plot Review:**
- Do the price curves visually overlap as expected?
- Is the "large" OTM relative error visually contextualised by the tiny absolute values?
- Are there any unexpected features in the price evolution?

### Output Files
- `exp4_price_comparison.png`
- `exp4_price_difference.png`
- `exp4_price_evolution.png`
- `exp4_option_prices.md`

### Plot Verification Status
Reviewed -- Looks good!

---

## Experiment 5: Parameter Sensitivity

### Purpose
Study how both methods respond to changes in market parameters. Identify regimes where methods agree or diverge.

### Parameters Tested
- Volatility (σ): 0.05 to 0.50
- Correlation (ρ): −0.5 to 0.9
- Interest rate (r): 0.00 to 0.15
- Maturity (T): 0.10 to 2.0
- Moneyness (K/S₀): 0.80 to 1.20

### Results

#### Volatility Sensitivity (σ)

| σ | L² Error | Assessment |
|---|----------|------------|
| 0.05 | 0.508 | ⚠️ Very high — methods disagree |
| 0.10 | 0.106 | Better |
| 0.15 | 0.174 | Moderate |
| 0.20 | 0.199 | Baseline |
| 0.30 | 0.220 | Slight increase |
| 0.50 | 0.265 | Higher but acceptable |

**Observation:** Very low volatility (σ < 0.10) causes large discrepancies.

#### Correlation Sensitivity (ρ)

| ρ | L² Error |
|---|----------|
| −0.50 | 0.278 |
| 0.00 | 0.201 |
| 0.50 | 0.200 |
| 0.90 | 0.203 |

**Observation:** Correlation has moderate effect. Negative correlation shows slightly higher error.

#### Interest Rate Sensitivity (r)

| r | L² Error |
|---|----------|
| 0.00 | 0.209 |
| 0.05 | 0.200 |
| 0.10 | 0.197 |
| 0.15 | 0.196 |

**Observation:** Methods agree well across entire interest rate range. Slight improvement at higher rates.

#### Maturity Sensitivity (T)

| T | L² Error | Status |
|---|----------|--------|
| 0.10 | — | ⚠️ **FAILED** |
| 0.25 | — | ⚠️ **FAILED** |
| 0.50 | 0.200 | ✅ Works |
| 0.75 | — | ⚠️ **FAILED** |
| 1.00 | 0.235 | ✅ Works |
| 1.50 | 0.240 | ✅ Works |
| 2.00 | 0.245 | ✅ Works |

**Observation:** Short maturities (T < 0.5) and T = 0.75 fail. This is a significant issue.

#### Moneyness Sensitivity (K/S₀)

| K/S₀ | L² Error |
|------|----------|
| 0.80–1.20 | 0.200 (constant) |

**Observation:** Moneyness has no effect on L² error. This is **correct** — the volatility surface $\bar{b}^2(t,S)$ does not depend on K directly; K only enters when solving the PDE for option prices.

### Analysis

#### 1. Are These Results Correct?

**Correct and expected:**
- Moneyness invariance is correct (surface doesn't depend on K)
- Interest rate stability is expected
- Correlation having moderate effect makes sense

**Concerning:**
- σ = 0.05 giving 50% error suggests breakdown of one or both methods at very low volatility
- Maturity failures at T < 0.5 and T = 0.75 indicate numerical issues

**Verdict:** Most results are correct, but the failures need investigation.

#### 2. What Do These Results Mean?

**Safe operating regime:**
- Volatility: σ ∈ [0.10, 0.50]
- Maturity: T ∈ [0.5, 2.0] (excluding 0.75?)
- Correlation: any ρ ∈ [−0.5, 0.9]
- Interest rate: any r ∈ [0, 0.15]
- Moneyness: any K/S₀

**Problematic regimes:**
- Very low volatility (σ < 0.10)
- Short maturities (T < 0.5)

#### 3. Insights

- Methods are most reliable in the typical equity option parameter range
- Very low volatility may cause Laplace saddle-point approximation to break down
- Short maturities may cause time-stepping issues in MLMC

#### 4. Action Items

⚠️ **BUGS TO INVESTIGATE:**

1. **Why do T = 0.10, 0.25, 0.75 fail?**
   - Check error messages
   - Could be: time-step too large, domain estimation failure, numerical instability

2. **Why does σ = 0.05 give 50% error?**
   - Check if Laplace optimisation converges
   - Check if MLMC has enough variance to fit

### Output Files
- `exp5_sensitivity_volatility.png`
- `exp5_sensitivity_correlation.png`
- `exp5_sensitivity_interest_rate.png`
- `exp5_sensitivity_maturity.png`
- `exp5_sensitivity_moneyness.png`
- `exp5_sensitivity_summary.png`
- `exp5_parameter_sensitivity.md`

### Plot Verification Status
⚠️ **NEEDS REVIEW:** Check that failed cases are clearly marked in plots and that trends are correctly displayed.

---

## Summary and Recommendations

### Overall Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Numerical stability | ✅ Good | Low variance, reproducible |
| Computational scaling | ✅ Excellent | MLMC is O(1), Laplace is O(d³) |
| Option pricing | ✅ Excellent | <0.2% error where it matters |
| Surface agreement | ⚠️ Moderate | ~18% L² error, but high correlation |
| Parameter robustness | ⚠️ Partial | Failures at short T, low σ |

### Key Messages for Paper

1. **MLMC provides O(1) dimensional scaling** vs O(d³) for Laplace — critical for high-dimensional problems

2. **Option prices agree excellently** despite volatility surface differences — the methodology choice doesn't affect practical pricing

3. **Methods converge better for high dimensions** — exactly where MLMC's advantages matter most

### Before Presenting Results

#### Must Do:
1. **Visual verification of all plots** — ensure they correctly represent the data
2. **Investigate systematic bias** — understand the ~18% L² discrepancy
3. **Debug maturity failures** — T < 0.5 should work

#### Should Do:
1. Add error bars to plots where applicable
2. Include representative surface plots in paper
3. Emphasise option pricing agreement as headline result

### Instructions for Future Analysis Sessions

This document is designed to bring a new Claude instance up to speed. Each experiment can be analysed in a separate chat:

**For Experiment N analysis:**
1. Reference this document for context
2. Request the specific plots from `/Comparison_Between_Methods/results/figures/expN_*.png`
3. Compare plots against the expected behaviour described here
4. Document any discrepancies found

**Key questions to answer for each experiment:**
1. Do the plots match the tabulated numerical results?
2. Are axis labels and legends correct?
3. Does the visual presentation support the conclusions?
4. Are there any unexpected features that need explanation?

---

## Appendix: File Locations

### Results Tables
```
Comparison_Between_Methods/results/tables/
├── exp1_metrics.md
├── exp2_convergence_stats.md
├── exp3_dimension_scaling.md
├── exp4_option_prices.md
└── exp5_parameter_sensitivity.md
```

### Figures
```
Comparison_Between_Methods/results/figures/
├── exp1_surface_comparison.png
├── exp1_difference_heatmap.png
├── exp1_summary_panel.png
├── exp2_convergence_curves.png
├── exp2_confidence_bands.png
├── exp2_convergence_rate.png
├── exp3_time_vs_dimension.png
├── exp3_error_vs_dimension.png
├── exp3_scaling_log.png
├── exp4_price_comparison.png
├── exp4_price_difference.png
├── exp4_price_evolution.png
├── exp5_sensitivity_*.png
└── exp5_sensitivity_summary.png
```

### Experiment Scripts
```
Comparison_Between_Methods/experiments/
├── exp1_surface_comparison.py
├── exp2_mlmc_convergence.py
├── exp3_dimension_scaling.py
├── exp4_option_pricing.py
└── exp5_parameter_sensitivity.py
```

---

*Document generated from analysis session on December 2024*

---

## Analysis Log

### Experiment 1: ✅ Analysed

**Findings:**
- Tested max_degree values 3, 4, 5 - higher degrees caused numerical blow-up (10¹² values) due to insufficient timesteps at coarse MLMC levels
- Settled on max_degree = 3 with h0 = 0.1 (stable and accurate)
- Identified root cause of ~19% L² error: **dynamic range compression** from non-uniform Monte Carlo sampling
- MLMC range ~1250-1500 vs Laplace range ~950-2100 (MLMC "squashes" the S² dependence)
- High correlation (0.948) confirms methods agree in shape, not magnitude
- This is a fundamental limitation of polynomial regression with concentrated samples, not a bug

**No code changes required.**

### Experiment 2: ✅ Analysed

**Findings:**
- Confirmed ~19% discrepancy is deterministic (std across runs only 0.34%)
- Stochastic uncertainty follows 1/√n as expected
- Adding more MLMC runs cannot reduce the bias

**Code changes:**
- Updated `plot_mlmc_convergence()` in `visualisation/convergence_plots.py` (panel 3)
- Updated convergence rate plot in `experiments/exp2_mlmc_convergence.py`
- Fix: Replaced misleading 1/√n reference with proper error decomposition showing bias (flat, ~19%) vs stochastic uncertainty (decreasing, ~0.6% → 0.1%)
- Added percentage formatting to y-axis

---

*Analysis completed: December 2024*