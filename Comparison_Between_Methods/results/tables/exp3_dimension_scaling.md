# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 3.44 | 0.65 | 0.0018 | 0.0115 |
| 3 | 3.65 | 1.32 | 0.0087 | 0.0676 |
| 5 | 4.24 | 2.91 | 0.0038 | 0.0222 |
| 10 | 5.10 | 9.90 | 0.0105 | 0.0468 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
