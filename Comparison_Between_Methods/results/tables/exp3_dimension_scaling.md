# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.38 | 0.64 | 2.0000 | 10.0273 |
| 3 | 0.39 | 1.35 | 2.0000 | 4.7220 |
| 5 | 0.41 | 3.05 | 2.0000 | 7.3091 |
| 10 | 0.50 | 10.68 | 1.9991 | 5.0095 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
