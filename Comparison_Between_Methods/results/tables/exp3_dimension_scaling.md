# Experiment 3: Dimension Scaling Results

**NOTE**: Disagreement metrics measure difference between MLMC and Laplace.
Neither method is ground truth.

**Threading disabled**: False

## Wall-Clock Time Summary

| d | MLMC Time (s) | Laplace Time (s) | L² Disagreement | L∞ Disagreement |
|---|---------------|------------------|-----------------|------------------|
| 2 | 0.39 | 2.27 | 0.0018 | 10.3136 |
| 3 | 0.41 | 5.00 | 0.0057 | 46.5437 |
| 5 | 0.48 | 11.06 | 0.0028 | 40.2860 |
| 10 | 0.59 | 34.46 | 0.0103 | 265.5721 |

## Hardware-Independent Cost Metrics

| d | MLMC CPU (s) | Laplace CPU (s) | MLMC Calls | Laplace Calls | MLMC Mem (MB) | Laplace Mem (MB) |
|---|--------------|-----------------|-----------|---------------|---------------|------------------|
| 2 | 0.39 | 2.27 | 143,829 | 1,882,129 | 30.6 | 0.3 |
| 3 | 0.41 | 4.97 | 143,829 | 4,045,368 | 36.6 | 0.3 |
| 5 | 0.48 | 11.01 | 143,829 | 9,176,723 | 48.8 | 0.3 |
| 10 | 0.59 | 34.37 | 143,829 | 27,891,525 | 79.4 | 0.3 |

## Key Observations

- **Function calls**: Hardware-independent metric (same results on any computer)
- **CPU time**: More consistent than wall time (excludes sleep/IO)
- **Peak memory**: Shows memory complexity scaling
- Markovian projection reduces d-dimensional problem to 1D PDE
