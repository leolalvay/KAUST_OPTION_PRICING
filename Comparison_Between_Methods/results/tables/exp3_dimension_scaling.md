# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.40 | 0.59 | 2.0000 | 2.7308 |
| 3 | 0.41 | 1.26 | 2.0000 | 2.7155 |
| 5 | 0.46 | 2.95 | 2.0000 | 2.5053 |
| 10 | 0.52 | 9.80 | 2.0000 | 2.3776 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
