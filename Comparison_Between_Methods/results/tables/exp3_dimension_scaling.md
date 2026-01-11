# Experiment 3: Dimension Scaling Results

**NOTE**: Disagreement metrics measure difference between MLMC and Laplace.
Neither method is ground truth.

**Threading disabled**: False

## Wall-Clock Time Summary

| d | MLMC Time (s) | Laplace Time (s) | L² Disagreement | L∞ Disagreement |
|---|---------------|------------------|-----------------|------------------|
| 2 | 0.39 | 2.23 | 0.0018 | 10.3136 |
| 3 | 0.41 | 4.80 | 0.0057 | 46.5437 |
| 5 | 0.47 | 11.03 | 0.0028 | 40.2860 |
| 10 | 0.56 | 33.83 | 0.0103 | 265.5721 |
| 20 | 0.88 | 129.27 | 0.0448 | 3028.2342 |
| 50 | 1.53 | 714.73 | 0.2299 | 46740.1620 |
| 100 | 2.89 | 3815.46 | 1.8745 | 8335971.8830 |

## Hardware-Independent Cost Metrics

| d | MLMC CPU (s) | Laplace CPU (s) | MLMC Calls | Laplace Calls | MLMC Mem (MB) | Laplace Mem (MB) |
|---|--------------|-----------------|-----------|---------------|---------------|------------------|
| 2 | 0.39 | 2.23 | 143,829 | 1,882,129 | 30.6 | 0.3 |
| 3 | 0.41 | 4.80 | 143,829 | 4,045,368 | 36.6 | 0.3 |
| 5 | 0.47 | 10.95 | 143,829 | 9,176,723 | 48.8 | 0.3 |
| 10 | 0.56 | 33.74 | 143,829 | 27,891,525 | 79.4 | 0.3 |
| 20 | 0.87 | 128.98 | 143,829 | 105,791,457 | 140.4 | 0.3 |
| 50 | 1.56 | 713.45 | 143,829 | 544,758,689 | 328.1 | 0.5 |
| 100 | 2.94 | 3766.12 | 143,829 | 1,902,682,059 | 656.2 | 1.1 |

## Key Observations

- **Function calls**: Hardware-independent metric (same results on any computer)
- **CPU time**: More consistent than wall time (excludes sleep/IO)
- **Peak memory**: Shows memory complexity scaling
- Markovian projection reduces d-dimensional problem to 1D PDE
