# MLMC Sample Allocation Comparison

This folder contains a standalone experiment to compare different sample allocation strategies for MLMC polynomial regression.

## Required Files

You need to copy the following files from your main project into this folder:

1. **`config.py`** - Problem parameters configuration
2. **`common.py`** - Contains `VolatilitySurfaceResult` dataclass
3. **`mlmc_ot_estimator.py`** - Original MLMC implementation (needs modification, see below)

## Files Provided

4. **`mlmc_flexible_allocation.py`** - Modified MLMC with flexible sample allocation
5. **`MLMC_Sample_Allocation_Comparison.ipynb`** - Jupyter notebook for experiments
6. **`README.md`** - This file


## Folder Structure

After setup, your folder should look like:

```
sample_allocation_comparison/
├── config.py                              # Copy from your project
├── common.py                              # Copy from your project
├── mlmc_ot_estimator.py                   # Copy from your project (+ modify import)
├── mlmc_flexible_allocation.py            # Provided
├── MLMC_Sample_Allocation_Comparison.ipynb # Provided
└── README.md                              # This file
```

## Running the Notebook

1. Ensure all files are in the same folder
2. Open `MLMC_Sample_Allocation_Comparison.ipynb` in Jupyter
3. Run all cells

The notebook will:
- **Part 1:** Test 10 different values of C (10, 20, 30, 40, 50, 60, 80, 100, 120, 150) with the quadratic formula
- **Part 2:** Test 5 different allocation formulas (quadratic, log, sqrt, linear, log_squared) with C=80

Each configuration is tested with 10 random seeds for statistical robustness.

## Available Allocation Formulas

| Formula | Expression | Use Case |
|---------|------------|----------|
| `quadratic` | $M_\ell = C \cdot m^2$ | Current (conservative) |
| `log` | $M_\ell = C \cdot m \log(m+1)$ | Cohen-Migliorati optimal |
| `sqrt` | $M_\ell = C \cdot m^{1.5}$ | Intermediate |
| `linear` | $M_\ell = C \cdot m$ | Minimum for uniqueness |
| `log_squared` | $M_\ell = C \cdot m \log^2(m+1)$ | Extra conservative |

## Author

Wadoud Bouslama (KAUST Internship)  
January 2026
