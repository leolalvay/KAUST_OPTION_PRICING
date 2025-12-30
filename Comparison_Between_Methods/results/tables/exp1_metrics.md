# Method Agreement: MLMC vs Laplace Approximation

**IMPORTANT**: Neither method is ground truth. These metrics measure
disagreement between methods, not error.

## Summary Statistics

| Metric | Value |
|--------|-------|
| L² disagreement | 0.005743 |
| L∞ disagreement | 42.6993 |
| Mean absolute difference | 5.8625 |
| Mean relative difference | 0.29% |
| Bias (MLMC - Laplace) | -5.7417 |
| Pearson correlation | 1.0000 |
| Valid grid fraction | 100.0% |

## Parameters

```
============================================================
Problem Parameters (Bayer et al. 2017, Eq. 56)
============================================================
  Number of assets: d = 3
  Initial prices: x0 = [100. 100. 100.]
  Initial basket: S0 = 300.0
  Basket weights: P1 = [1. 1. 1.]

  Risk-free rate: r = 0.05
  Volatilities: σ = [0.2  0.15 0.1 ]
  Correlation matrix:
    [1.  0.8 0.3]
    [0.8 1.  0.1]
    [0.3 0.1 1. ]

  Option type: put
  Strike: K = 300.0
  Maturity: T = 0.5

  PDE grid: 50 × 100
  MLMC max degree: 3
  MLMC coarsest timestep: h0 = 0.05
  Random seed: 42
============================================================
```

## Notes

- **Neither method is treated as ground truth**
- Relative differences use symmetric formula: 2|a-b|/(|a|+|b|)
- L² disagreement normalised by mean surface magnitude
- High correlation (> 0.95) indicates good agreement
