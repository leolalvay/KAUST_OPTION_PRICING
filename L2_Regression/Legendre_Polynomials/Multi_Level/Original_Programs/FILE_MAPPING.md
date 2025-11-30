# Multi-Level File Consolidation Map

**Date:** November 30, 2024  
**Project:** Multi-Level Markovian Projection Refactoring  
**Original Source:** Amelie's originalprograms/ folder (10 files)  
**Refactored Output:** 4 core files + 1 notebook

---

## Overview

We consolidated 10 original research files into 4 production-quality Python files plus one comprehensive notebook. This reduces duplication, improves maintainability, and follows the established Single-Level conventions.

**Key principles:**
- Import shared utilities from Single-Level (no duplication)
- Group related functionality together
- Separate computation from visualisation
- British spelling throughout
- Credit Amelie in every file
- ML_ prefix for all Multi-Level files

---

## File Mapping Details

### ✅ Core File 1: ML_level_utilities.py (7.9 KB)

**Merged from:**
- `DM_utils_ML.py` (partial - only ML-specific functions)

**What's included:**
- ✅ `normaleq_components_ML(paths_f, paths_c, ...)` - Constructs D and ψ for b_fine² - b_coarse²
- ✅ `fit_local_vol(D, psi)` - QR solver with column normalisation
- ✅ `make_b_bar(c, pairs, ...)` - Callable volatility function with negative checking

**What's excluded (imported from SL instead):**
- ❌ `GBM_paths()` - Identical to Single-Level version
- ❌ `tot_degree_poly()` - Identical to Single-Level version

**Why separate?**
These are the ONLY functions that differ between Single-Level and Multi-Level. Everything else is shared.

**Key difference from SL:**
`normaleq_components_ML()` computes ψ = b_fine² - b_coarse² (difference for telescoping sum) instead of just b².

---

### ✅ Core File 2: ML_telescoping_sum.py (9.3 KB)

**Merged from:**
- `DM_ML.py` (complete - all functions)

**What's included:**
- ✅ `scalings_l0(x0, T, h0, ...)` - Pilot run for domain estimation
- ✅ `mlmc_l(x0, T, h0, l, ...)` - Level-l estimator with coupled fine/coarse paths
- ✅ `make_c(x0, T, h0, ...)` - Telescoping sum: c_total = Σ_{l=0}^L c_l

**Key features:**
- Level-dependent timesteps: dt_l = h0 * 2^(-l)
- Level-dependent polynomial degrees: deg_l = max_deg - l
- Adaptive sample sizes: M_l ∝ (# basis functions)²
- Brownian coupling: Z_coarse[i] = Z_fine[2i] + Z_fine[2i+1]

**Physics analogy:**
Like perturbation theory in QFT: ground state (level 0) + corrections (levels 1-L).

---

### ✅ Core File 3: ML_optimal_transport.py (18 KB) 💎

**Merged from:**
- `ML_BS_OTmap.py` (complete - all functions)
- `DM_ML_OT.py` (complete - all functions)

**What's included:**

**Part 1: OT Infrastructure (from ML_BS_OTmap.py)**
- ✅ `GaussianBrenierMap` class - Optimal transport between Gaussians
  - `__init__(mu_f, C_f, mu_c, C_c)` - Computes transformation matrix A
  - `map(x)` - Applies T(x) = μ_c + A(x - μ_f)
- ✅ `identity_map(x)` - For time t=0 (no transformation needed)
- ✅ `logpaths_maps(x0, T, h0, l, ...)` - Generate paths + build OT maps
- ✅ `apply_maps(paths_f, maps, ...)` - Transform fine → estimated coarse
- ✅ `validate_maps(logpaths_f, logpaths_c, maps, ...)` - Check map quality

**Part 2: MLMC with OT (from DM_ML_OT.py)**
- ✅ `mlmc_l_OT(x0, T, h0, l, ...)` - Level-l estimator using OT maps
- ✅ `make_c_OT(x0, T, h0, ...)` - Hybrid: c_0 (standard) + Σ c_l^{OT} (levels 1-L)

**Why merge these two files?**
The OT map infrastructure and OT-MLMC usage are conceptually inseparable. Keeping them together makes the implementation clearer.

**Key innovation:**
Instead of sampling coarse paths, we TRANSFORM fine paths using Brenier maps. This minimises E[||X_f - X_c||²], giving better variance reduction.

**Mathematical foundation:**
Brenier's theorem: For Gaussians, the optimal map T: N(μ_f, C_f) → N(μ_c, C_c) is:
    T(x) = μ_c + A(x - μ_f)
where A = C_f^{-1/2} (C_f^{1/2} C_c C_f^{1/2})^{1/2} C_f^{-1/2}

---

### ✅ Core File 4: ML_weak_error_analysis.py (11 KB)

**Merged from:**
- `ML_WE.py` (refactored and enhanced)

**What's included:**
- ✅ `compute_weak_error_ML(b_bar, x0, ...)` - Weak error across sample sizes
- ✅ `plot_weak_error_convergence(M_samples, ...)` - Publication-quality plots
- ✅ Main script with comprehensive testing

**Changes from original:**
- Renamed `plot_weakerror()` → `compute_weak_error_ML()` (clearer naming)
- Separated computation from plotting (modularity)
- Added detailed docstrings
- Uses `plots/WeakError/` directory (not `FIGS/`)
- Enhanced plotting with error bars and individual trial points

**What it validates:**
- Monte Carlo convergence: weak error ∝ M^{-1/2}
- Pricing accuracy: |E[g(X_T)] - E[g(S̄_T)]| / |E[g(X_T)]|
- MLMC maintains accuracy whilst reducing cost

---

## ❌ Files Not Yet Refactored (Will go in ML_Results.ipynb)

These are single-use visualization/comparison scripts. Instead of refactoring them as standalone .py files, we'll combine them into ONE comprehensive Jupyter notebook for better interactivity and presentation.

### ML_volsurf.py → Section 1 of ML_Results.ipynb
**Purpose:** 3D volatility surface visualization  
**Saves to:** `plots/VolSurf/VolSurf_maxdeg{N}.pdf`  
**What it does:**
- Generates 3D wireframe/surface plots of b̄(t, S)
- Compares surfaces across polynomial degrees
- Validates smoothness and positivity

### ML_projerror.py → Section 2 of ML_Results.ipynb
**Purpose:** L² projection error (approximation quality)  
**What it does:**
- Computes ||b̄(t,S) - b(X)||₂ on validation paths
- Measures regression fit quality
- Reports standard error, absolute error, relative error

### pairsscatter.py → Section 3 of ML_Results.ipynb
**Purpose:** Visualize fine/coarse path correlation  
**Saves to:** `plots/PairScatter/PairDistribution_{l}.pdf`  
**What it does:**
- Scatter plots of X_coarse vs X_fine at terminal time
- Validates coupling strength (high correlation = good variance reduction)
- Shows correlation improves with OT maps

### singlemulticomp.py → Section 4 of ML_Results.ipynb
**Purpose:** Compare SL vs ML vs ML+OT  
**Saves to:** `plots/SinglevsMultierror.pdf`  
**What it does:**
- Computes RMS difference between b̄_SL and b̄_ML
- Compares all three methods (Single-Level, Multi-Level, Multi-Level+OT)
- Shows OT gives best variance reduction
- Demonstrates computational cost savings

---

## Plot Directory Structure

Following Amelie's organization (renamed FIGS → plots):

```
plots/
├── PairScatter/           # Fine vs coarse correlation plots
│   └── PairDistribution_{l}.pdf
├── VolSurf/               # Standard MLMC volatility surfaces
│   └── VolSurf_maxdeg{N}.pdf
├── VolSurfOT/             # OT-MLMC volatility surfaces
│   └── OTVolSurf_maxdeg{N}.pdf
└── WeakError/             # Convergence analysis
    └── ML_WeakErrors.pdf
```

**All refactored files use `plots/` instead of `FIGS/`.**

---

## Files Completely Skipped

### DM_utils_SL.py - ❌ SKIPPED
**Why:** We already have `SL_legendre_utilities.py` from Single-Level phase.  
**Action:** Import from Single-Level when needed.

---

## Summary Statistics

| Category | Original | Refactored | Status |
|----------|----------|------------|--------|
| **Core utilities** | DM_utils_ML.py (partial) + DM_utils_SL.py | ML_level_utilities.py | ✅ DONE |
| **Standard MLMC** | DM_ML.py | ML_telescoping_sum.py | ✅ DONE |
| **Optimal Transport** | ML_BS_OTmap.py + DM_ML_OT.py | ML_optimal_transport.py | ✅ DONE |
| **Validation** | ML_WE.py | ML_weak_error_analysis.py | ✅ DONE |
| **Visualization** | ML_volsurf.py + ML_projerror.py + pairsscatter.py + singlemulticomp.py | ML_Results.ipynb | 🔜 NEXT |
| **Theory** | (none) | ML_Theory.md | 🔜 NEXT |

**Total:** 10 original files → 4 core .py files + 1 notebook + 1 theory doc

---

## Key Improvements Over Original

1. **No duplication:** Shared utilities imported from SL, not copied
2. **Better organization:** Related functions grouped logically
3. **Consistent naming:** ML_ prefix, British spelling, clear function names
4. **Comprehensive docs:** Detailed docstrings with theory, parameters, notes
5. **Modular design:** Computation separated from visualization
6. **Publication-ready:** Clean code suitable for academic paper supplementary materials

---

## Next Steps

1. ✅ **Test ML_weak_error_analysis.py** - Ensure it runs and generates plots correctly
2. 🔜 **Create ML_Theory.md** - Comprehensive theory document (telescoping sums, OT theory, complexity analysis)
3. 🔜 **Create ML_Results.ipynb** - Interactive notebook with all comparisons and plots
4. 🔜 **Validate results** - Compare with Amelie's original outputs to ensure correctness

---

**END OF FILE MAPPING DOCUMENT**
