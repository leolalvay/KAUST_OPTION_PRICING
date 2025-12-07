# Single-Level vs Multi-Level: A Complete Clarification

**Date:** December 7, 2024  
**Project:** Multi-Level Markovian Projection for American Options  
**Author:** Wadoud (with Claude)

---

## Overview

The terminology "single-level" (SL) vs "multi-level" (ML) refers to **how many timestep resolutions** are used in the Monte Carlo estimation, NOT the dimensionality of the basket or whether Markovian projection is applied.

**Key Point:** Both approaches use Markovian projection to reduce d-dimensional baskets to 1D. Both use MLMC-adjacent techniques like regression on polynomial basis. The difference is whether you use **one fixed timestep** or **multiple hierarchical timesteps combined via telescoping sum**.

---

## The Core Distinction

### Single-Level (SL_)
- Run Monte Carlo at **one fixed timestep resolution** (e.g., N = 64)
- Generate paths, compute regression coefficients, done
- Standard Monte Carlo applied once
- **Computes:** E[Y_L] directly at finest level L

### Multi-Level (ML_)
- Run Monte Carlo at **multiple timestep resolutions** simultaneously (N = 4, 8, 16, 32, 64...)
- Combine results using the **telescoping sum formula**
- Full MLMC machinery with variance reduction
- **Computes:** E[Y_L] = E[Y_0] + Σ(E[Y_ℓ] - E[Y_ℓ₋₁])

---

## The Telescoping Sum Formula

This is the mathematical heart of MLMC:

```
E[Y_L] = E[Y_0] + Σ_{ℓ=1}^{L} (E[Y_ℓ] - E[Y_ℓ₋₁])
```

where Y_ℓ represents your quantity of interest (volatility surface coefficients) at level ℓ.

**Why this works:**
- **Coarse level (ℓ=0):** Cheap to compute E[Y_0], captures rough structure
- **Correction terms:** Each (E[Y_ℓ] - E[Y_ℓ₋₁]) has low variance due to coupling
- **Result:** Same accuracy as single-level at L, but much cheaper overall

**Physics Analogy:** Like perturbation theory in quantum mechanics:
- Single-level = Diagonalising the full Hamiltonian (expensive!)
- Multi-level = H₀ eigenstate + first-order correction + second-order + ...

Each MLMC level is like adding another perturbative order.

---

## Code Implementation: Where to Find It

### 1. Single-Level: Direct Computation

**File:** `SL_surface_visualisation.py` or `SL_single_level.py`

```python
# Single fixed timestep
dt = T / N  # e.g., N = 64
paths = GBM_paths(x0, r, vol, cov_mat, dt, N, M)

# Compute D and ψ for b² (absolute volatility)
D, psi = normaleq_components_SL(paths, P1, pairs, ...)

# Solve: Dc = ψ
c = fit_local_vol(D, psi)  # Done! That's your answer.
```

You get **one set of coefficients** `c` at your chosen resolution N.

---

### 2. Multi-Level: Telescoping Sum

**File:** `ML_telescoping_sum.py`

The key function that implements the telescoping sum:

```python
def make_c(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min0, s_max0, C=80):
    """
    Aggregate coefficients across all MLMC levels via telescoping sum.
    
    Computes:
        c_total = Σ_{l=0}^{max_deg} c_l
    
    where c_l are the level-specific corrections
    """
    return sum(mlmc_l(x0, T, h0, l, r, cov_mat, vol, max_deg, P1, s_min0, 
                      s_max0, C) 
               for l in range(max_deg + 1))  # ← TELESCOPING SUM!
```

This **literally sums** corrections from multiple levels:  
`c_total = c₀ + c₁ + c₂ + c₃`

---

### 3. What Each Level Does: `mlmc_l()`

**File:** `ML_telescoping_sum.py`

Each level computes a **correction term** using coupled fine/coarse paths:

```python
def mlmc_l(x0, T, h0, l, r, cov_mat, vol, max_deg, ...):
    """
    Estimate level-l correction coefficients using coupled fine/coarse paths.
    """
    # Level-dependent timesteps
    hl_f = h0 * 2 ** (-l)     # Fine timestep: h₀/2^l
    hl_c = 2 * hl_f           # Coarse timestep: 2×fine
    
    N_f = int(round(T / hl_f))  # e.g., level 2: N_f = 16
    N_c = N_f // 2              # e.g., level 2: N_c = 8
    
    # Generate COUPLED paths at both resolutions
    paths_f = generate_fine_paths(...)    # N_f timesteps
    paths_c = generate_coarse_paths(...)  # N_c timesteps
    
    # HERE'S THE KEY DIFFERENCE FROM SINGLE-LEVEL:
    # Compute ψ = b²_fine - b²_coarse (the DIFFERENCE!)
    D, psi = normaleq_components_ML(paths_f, paths_c, ...)
    
    # Solve for this level's correction
    c_l = fit_local_vol(D, psi)
    
    return c_l  # This is added to the telescoping sum
```

---

### 4. The Critical Difference in Regression Target

**Multi-Level** (`ML_level_utilities.py`):

```python
def normaleq_components_ML(paths_f, paths_c, P1, pairs, ...):
    """
    ML version: computes ψ = b²_fine - b²_coarse (DIFFERENCE for telescoping)
    """
    # Compute basket volatilities at both resolutions
    b_squared_fine = compute_volatility(paths_f, ...)
    b_squared_coarse = compute_volatility(paths_c, ...)
    
    # The MLMC magic: fit the DIFFERENCE
    psi = b_squared_fine - b_squared_coarse  # ← THIS IS NEW!
    
    return D, psi
```

**Single-Level** (`SL_level_utilities.py`):

```python
def normaleq_components_SL(paths, P1, pairs, ...):
    """
    SL version: computes ψ = b² (absolute volatility)
    """
    # Just compute volatility at THIS resolution
    b_squared = compute_volatility(paths, ...)
    
    psi = b_squared  # ← Fit absolute value, not difference!
    
    return D, psi
```

**Key Insight:** Multi-level fits the **difference** between resolutions, not the absolute values. This difference has much lower variance when paths are properly coupled!

---

## The Coupling Mechanism

**File:** `ML_telescoping_sum.py` (inside `mlmc_l()`)

The variance reduction comes from **reusing random numbers** between fine and coarse paths:

```python
# Generate random increments ONCE
Z = np.random.randn(M_l, N_f, d)  # e.g., 64 timesteps for fine level

# ========================================================================
# FINE paths: use Z directly
# ========================================================================
for n in range(1, N_f):
    Z_n = Z[:, n-1, :]
    dW = (Z_n @ G.T) * math.sqrt(hl_f)
    X = X + r * X * hl_f + sigma * dW
    paths_f[:, n, :] = X

# ========================================================================
# COARSE paths: SUM consecutive pairs of fine increments
# ========================================================================
Z_c = Z.reshape(M_l, N_c, 2, d).sum(axis=2)  # ← COUPLING!
# Z_coarse[0] = Z_fine[0] + Z_fine[1]
# Z_coarse[1] = Z_fine[2] + Z_fine[3]
# etc.

for n in range(1, N_c):
    Z_n = Z_c[:, n-1, :]
    dW = (Z_n @ G.T) * math.sqrt(hl_f)  # Note: sqrt(hl_f), not sqrt(hl_c)!
    X = X + r * X * hl_c + sigma * dW
    paths_c[:, n, :] = X
```

**Why This Works:**
- Fine and coarse paths use **correlated random numbers**
- This makes paths_f and paths_c **highly correlated**
- Therefore b²_fine and b²_coarse are highly correlated
- The difference (b²_fine - b²_coarse) has **much lower variance**
- Need far fewer samples to estimate the correction accurately!

**Thermodynamics Analogy:** Like measuring temperature differences instead of absolute temperatures. The difference between two correlated measurements has lower uncertainty than either measurement alone.

---

## Computational Cost Comparison

### Single-Level at N = 64
- Need M samples at finest resolution
- Cost: M × 64 × (operations per timestep)
- Complexity to achieve error ε: **O(ε⁻³)**

### Multi-Level (levels 0, 1, 2, 3, 4)
- Level 0 (N = 4): Many samples (cheap!)
- Level 1 (N = 8): Fewer samples
- Level 2 (N = 16): Even fewer
- Level 3 (N = 32): Fewer still
- Level 4 (N = 64): Very few samples (expensive per sample, but not many needed!)

**Total cost:** Dominated by cheap coarse levels  
**Complexity to achieve error ε:** **O(ε⁻²(log ε)²)** 

**Example with max_deg = 3:**
- Level 0: ~1280 samples (deg 3, 10 basis functions)
- Level 1: ~720 samples (deg 2, 6 basis functions)
- Level 2: ~320 samples (deg 1, 3 basis functions)
- Level 3: ~80 samples (deg 0, 1 basis function)
- **Total:** ~2400 samples vs ~6400 for equivalent single-level

---

## Level-Dependent Features

Multi-level has several parameters that vary by level:

| Parameter | Formula | Example (max_deg=3) |
|-----------|---------|---------------------|
| **Timestep** | h_ℓ = h₀ · 2⁻ˡ | h₀, h₀/2, h₀/4, h₀/8 |
| **Number of steps** | N_ℓ = T/h_ℓ | 4, 8, 16, 32 |
| **Polynomial degree** | deg_ℓ = max_deg - ℓ | 3, 2, 1, 0 |
| **Basis size** | dim(V_ℓ) | 10, 6, 3, 1 |
| **Sample size** | M_ℓ ∝ dim(V_ℓ)² | 1280, 720, 320, 80 |

**Key Principle:** Coarse levels use high-degree polynomials (capture complex surface features), fine levels use low-degree polynomials (small corrections only).

---

## Quick Reference: Finding It In Code

| Feature | Single-Level | Multi-Level |
|---------|--------------|-------------|
| **Main file** | `SL_surface_visualisation.py` | `ML_telescoping_sum.py` |
| **Key function** | Direct computation in script | `make_c()` |
| **Loop structure** | One pass at fixed N | `sum(mlmc_l(...) for l in range(L))` |
| **Regression target** | `ψ = b²` | `ψ = b²_fine - b²_coarse` |
| **Path generation** | One resolution | Two coupled resolutions per level |
| **Output** | `c` (coefficients at level L) | `c_total = Σ c_l` (sum of corrections) |
| **Utilities file** | `SL_level_utilities.py` | `ML_level_utilities.py` |
| **Normal equations** | `normaleq_components_SL()` | `normaleq_components_ML()` |
| **Complexity** | O(ε⁻³) | O(ε⁻²(log ε)²) |

---

## Memory-Efficient Variants

The **Fast Multi-Level (FML_)** files implement the same telescoping sum but with accumulated normal equations:

**Standard ML:** Store full design matrix D (size M×N × dimV), then solve

**Fast ML:** Accumulate G = D^T D and g = D^T ψ incrementally
- Memory: O(MN·dimV) → **O(dimV²)**
- Same result, much lower memory footprint
- Essential for large-scale problems

**Files:**
- `FML_utils.py` - Accumulated normal equations implementation
- `FML_optimal_transport.py` - Same but with OT coupling (best variance reduction)

---

## Summary

**"Single-level" vs "Multi-level"** refers to the **timestep hierarchy**, NOT dimensionality:

- **Single-level:** One timestep resolution, compute E[Y_L] directly
- **Multi-level:** Multiple timestep resolutions, combine via telescoping sum
- **The "levels":** Refer to h₀, h₀/2, h₀/4, h₀/8, ... (timestep refinement)
- **Variance reduction:** Comes from coupling fine/coarse paths with correlated random numbers
- **Cost reduction:** O(ε⁻³) → O(ε⁻²(log ε)²) by doing most work at cheap coarse levels

Both methods use Markovian projection, regression on polynomial basis, and can handle d-dimensional baskets. The multi-level approach is simply a more sophisticated way to organize the Monte Carlo sampling.

---

## Further Reading

- **Single-Level Theory:** `SL_Theory.md` (880 lines, comprehensive)
- **Multi-Level Theory:** `ML_Theory.md` (~300 lines)
- **Fast Implementation:** `FML_Theory.md` (~370 lines)
- **File Mapping:** `FILE_MAPPING.md` (shows old → new file consolidation)
- **Optimal Transport:** `optimal_transport_tutorial.ipynb` (interactive)

---

**END OF CLARIFICATION**
