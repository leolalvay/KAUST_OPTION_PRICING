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
| 237.3 | 62.7302 | 62.7302 | 0.0000 | 0.00 |
| 240.6 | 59.3871 | 59.3871 | 0.0000 | 0.00 |
| 270.7 | 29.2995 | 29.2995 | 0.0000 | 0.00 |
| 300.8 | 7.3048 | 7.3043 | 0.0005 | 0.01 |
| 329.2 | 1.1707 | 1.1720 | 0.0014 | 0.12 |
| 359.3 | 0.1042 | 0.1050 | 0.0008 | 0.73 |
| 389.4 | 0.0062 | 0.0063 | 0.0001 | N/A |

## Statistics

- Mean absolute difference: 0.0005
- Max absolute difference: 0.0014
- Mean percentage difference: 0.30%
