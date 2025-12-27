# Experiment 4: Option Pricing Results

## Parameters

- Dimension: d = 3
- Strike: K = 300.0
- Initial basket: S0 = 300.0
- Maturity: T = 0.5
- Option type: put

## Price Comparison at Key Spot Values

| S | MLMC Price | Laplace Price | Difference | Diff (%) |
|---|------------|---------------|------------|----------|
| 237.2 | 62.7575 | 62.7575 | 0.0000 | 0.00 |
| 240.7 | 59.3222 | 59.3222 | 0.0000 | 0.00 |
| 269.9 | 30.1223 | 30.1223 | 0.0000 | 0.00 |
| 300.8 | 7.2995 | 7.3012 | 0.0017 | 0.02 |
| 330.0 | 1.1056 | 1.1063 | 0.0007 | 0.06 |
| 359.2 | 0.1061 | 0.1059 | 0.0002 | 0.15 |
| 390.1 | 0.0062 | 0.0061 | 0.0001 | N/A |

## Statistics

- Mean absolute difference: 0.0005
- Max absolute difference: 0.0017
- Mean percentage difference: 0.13%
