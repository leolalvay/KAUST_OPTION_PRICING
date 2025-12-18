# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.22 | 0.58 | 0.1984 | 0.3775 |
| 3 | 0.22 | 1.24 | 0.1911 | 0.3570 |
| 5 | 0.24 | 2.83 | 0.1347 | 0.2511 |
| 10 | 0.29 | 9.60 | 0.0904 | 0.1702 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
