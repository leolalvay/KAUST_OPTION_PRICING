# Experiment 3: Dimension Scaling Results

## Summary

| d | MLMC Time (s) | Laplace Time (s) | L2 Error | L∞ Error |
|---|---------------|------------------|----------|----------|
| 2 | 0.19 | 0.64 | 1.1996 | 1.9142 |
| 3 | 0.22 | 1.36 | 1.6015 | 2.5515 |
| 5 | 0.27 | 3.17 | 1.8460 | 2.6184 |
| 10 | 0.35 | 10.20 | 1.9607 | 2.5699 |

## Key Observations

- Markovian projection reduces d-dimensional problem to 1D PDE
- Main computational cost scales with number of Monte Carlo paths
- Laplace approximation remains purely analytical
- Both methods avoid exponential curse of dimensionality
