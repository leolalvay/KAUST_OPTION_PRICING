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
| 300.8 | 1.2571 | 7.3043 | 6.0472 | 82.79 |
| 329.2 | 0.0001 | 1.1720 | 1.1720 | 99.99 |
| 359.3 | 0.0000 | 0.1050 | 0.1050 | 100.00 |
| 389.4 | 0.0000 | 0.0063 | 0.0063 | N/A |

## Statistics

- Mean absolute difference: 1.0528
- Max absolute difference: 6.1076
- Mean percentage difference: 60.66%
