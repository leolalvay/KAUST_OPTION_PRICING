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
| 210.6 | 89.3939 | 89.3939 | 0.0000 | 0.00 |
| 240.9 | 59.0909 | 59.0909 | 0.0000 | 0.00 |
| 271.2 | 28.7879 | 28.7879 | 0.0000 | 0.00 |
| 298.5 | 1.5152 | 8.2814 | 6.7663 | 81.70 |
| 328.8 | 1.1568 | 1.2084 | 0.0515 | 4.26 |
| 359.1 | 0.8746 | 0.1077 | 0.7669 | 712.29 |
| 389.4 | 0.5924 | 0.0068 | 0.5855 | N/A |

## Statistics

- Mean absolute difference: 1.4744
- Max absolute difference: 85.6434
- Mean percentage difference: 309.32%
