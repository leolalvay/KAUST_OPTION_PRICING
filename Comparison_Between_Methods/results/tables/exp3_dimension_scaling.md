# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.36 | 0.59 | 2.0000 | 9.7503 |
| 3 | 0.37 | 1.27 | 2.0000 | 4.7091 |
| 5 | 0.41 | 2.82 | 2.0000 | 3.6638 |
| 10 | 0.46 | 9.64 | 1.9980 | 6.2870 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
