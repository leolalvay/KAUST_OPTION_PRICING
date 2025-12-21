# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.83 | 0.59 | 0.2073 | 0.3964 |
| 3 | 0.87 | 1.25 | 0.2004 | 0.3793 |
| 5 | 0.94 | 2.91 | 0.1351 | 0.2450 |
| 10 | 1.14 | 9.59 | 0.0914 | 0.1633 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
