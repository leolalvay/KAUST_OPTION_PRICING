# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.38 | 0.60 | 2.0000 | 9.7498 |
| 3 | 0.40 | 1.32 | 2.0000 | 4.7120 |
| 5 | 0.42 | 3.22 | 2.0000 | 3.7503 |
| 10 | 0.49 | 9.82 | 1.9980 | 6.3118 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
