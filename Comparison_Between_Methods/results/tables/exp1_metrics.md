# Comparison Metrics: MLMC vs Laplace Approximation

## Summary Statistics

| Metric | Value |
|--------|-------|
| L2 relative error | 2.000000 |
| L∞ error | 28416316569.0657 |
| Mean absolute difference | 9098743280.7659 |
| Mean relative difference | 200.00% |
| Bias (MLMC - Laplace) | 9098742844.1152 |
| Pearson correlation | 0.6939 |
| Valid grid fraction | 44.5% |

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
  MLMC coarsest timestep: h0 = 0.1
  Random seed: 42
============================================================
```

## Notes

- Neither method is treated as ground truth
- Relative differences use symmetric formula: 2|a-b|/(|a|+|b|)
- L2 error normalised by mean surface magnitude
