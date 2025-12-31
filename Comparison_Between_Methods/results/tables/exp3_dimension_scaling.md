# Experiment 3: Dimension Scaling Results

**NOTE**: Disagreement metrics measure difference between MLMC and Laplace.
Neither method is ground truth.

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L² Disagreement | L∞ Disagreement |
|---|---------------|------------------|-----------------|------------------|
| 2 | 0.19 | 0.64 | 0.0018 | 10.3136 |
| 3 | 0.21 | 1.32 | 0.0057 | 46.5437 |
| 5 | 0.25 | 2.90 | 0.0028 | 40.2860 |
| 7 | 0.30 | 5.18 | 0.0052 | 118.2496 |
| 10 | 0.37 | 9.92 | 0.0103 | 265.5721 |
| 15 | 0.46 | 23.02 | 0.0229 | 1031.5011 |
| 20 | 0.59 | 39.06 | 0.0448 | 3028.2342 |
| 30 | 0.84 | 87.40 | 0.0961 | 11595.2870 |
| 50 | 1.36 | 251.28 | 0.2299 | 46740.1620 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
