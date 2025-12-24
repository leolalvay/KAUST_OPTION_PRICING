# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.19 | 0.65 | 0.0018 | 0.0084 |
| 3 | 0.21 | 1.36 | 0.0057 | 0.0266 |
| 5 | 0.25 | 3.00 | 0.0028 | 0.0139 |
| 10 | 0.36 | 9.90 | 0.0103 | 0.0501 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
