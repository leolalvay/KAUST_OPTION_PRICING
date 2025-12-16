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
| 211.2 | 88.7755 | 88.7755 | 0.0000 | 0.00 |
| 241.8 | 58.1633 | 58.1633 | 0.0000 | 0.00 |
| 272.4 | 27.5510 | 27.5510 | 0.0000 | 0.00 |
| 296.9 | 4.4554 | 8.9347 | 4.4793 | 50.13 |
| 327.6 | 0.0325 | 1.3157 | 1.2832 | 97.53 |
| 358.2 | 0.0001 | 0.1228 | 0.1227 | 99.96 |
| 388.8 | 0.0000 | 0.0087 | 0.0087 | N/A |

## Statistics

- Mean absolute difference: 0.4940
- Max absolute difference: 4.4793
- Mean percentage difference: 36.21%
