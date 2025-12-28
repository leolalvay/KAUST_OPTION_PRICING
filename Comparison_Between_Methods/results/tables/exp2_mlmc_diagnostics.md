# Experiment 2: MLMC Diagnostics and Uncertainty Analysis

## 1. MLMC Self-Convergence (Level Diagnostics)

These diagnostics assess MLMC internal convergence.

| Metric | Value | Status |
|--------|-------|--------|
| Weak convergence rate α | -0.062 | WARN |
| Variance decay rate β | 1.842 | OK |
| Richardson bias | nan | - |
| Convergence quality | acceptable | - |

### Per-Level Statistics

| Level | Variance | Correlation | VRF | Kurtosis |
|-------|----------|-------------|-----|----------|
| 0 | 4.29e+04 | 1.000 | 2.0 | 2.1 |
| 1 | 2.20e-01 | 1.000 | 437336.2 | 4.7 |
| 2 | 8.15e-03 | 1.000 | 12134912.1 | 8.3 |
| 3 | 1.71e-02 | 1.000 | 4631050.1 | 4.0 |

### Warnings

- Weak convergence rate α = -0.06 < 0.5 (slow convergence)

## 2. Statistical Uncertainty (Across-Run)

| Metric | Value |
|--------|-------|
| Number of runs | 20 |
| Mean relative std | 0.22% |
| Mean relative SE | 0.05% |
| Mean time per run | 0.22 s |
| Total time | 4.4 s |

## 3. Method Agreement (MLMC vs Laplace)

**IMPORTANT**: Neither method is ground truth. These metrics
measure disagreement, not error.

| Metric | Value |
|--------|-------|
| L² disagreement | 0.001810 |
| L∞ disagreement | 15.6158 |
| Mean relative diff | 0.13% |
| Correlation | 1.0000 |
| Bias (MLMC - Laplace) | -1.5349 |

