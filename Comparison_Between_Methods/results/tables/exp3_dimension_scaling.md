# Experiment 3: Dimension Scaling Results

**NOTE**: Disagreement metrics measure difference between MLMC and Laplace.
Neither method is ground truth.

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L² Disagreement | L∞ Disagreement |
|---|---------------|------------------|-----------------|------------------|
| 2 | 0.19 | 0.65 | 0.0018 | 10.3136 |
| 3 | 0.22 | 1.35 | 0.0057 | 46.5437 |
| 5 | 0.28 | 3.08 | 0.0028 | 40.2860 |
| 10 | 0.39 | 10.05 | 0.0103 | 265.5721 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
