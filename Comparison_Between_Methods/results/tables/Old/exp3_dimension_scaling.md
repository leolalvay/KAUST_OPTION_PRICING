# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.44 | 0.72 | 0.2087 | 0.4027 |
| 3 | 0.45 | 1.26 | 0.1972 | 0.3728 |
| 5 | 0.49 | 2.88 | 0.1374 | 0.2509 |
| 10 | 0.60 | 9.59 | 0.0907 | 0.1692 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
