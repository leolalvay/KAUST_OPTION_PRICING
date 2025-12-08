# Dimension Cases in MLMC + Markovian Projection

**Author:** Wadoud Charbak (KAUST Intern)  
**Date:** December 2024  
**Purpose:** Clarify all dimensional aspects of the American Basket Option pricing pipeline

---

## Table of Contents

1. [Overview: The Four Dimensions](#1-overview-the-four-dimensions)
2. [Asset Dimension (d)](#2-asset-dimension-d)
3. [Markovian Projection (d → 1)](#3-markovian-projection-d--1)
4. [Polynomial Basis Dimension (P_ℓ)](#4-polynomial-basis-dimension-p_ℓ)
5. [Time Discretisation (N_ℓ)](#5-time-discretisation-n_ℓ)
6. [Optimal Transport Map Dimensions](#6-optimal-transport-map-dimensions)
7. [Complete Pipeline Diagram](#7-complete-pipeline-diagram)
8. [Summary Tables](#8-summary-tables)

---

## 1. Overview: The Four Dimensions

There are multiple "dimensions" at play in this project, and conflating them causes confusion. Here is the complete picture:

| Concept | Symbol | What It Represents | How It Changes |
|---------|--------|-------------------|----------------|
| **Asset dimension** | d | Number of stocks in basket | **Fixed** throughout |
| **Spatial dimension** | 1 | Dimension of projected process | **Reduced** from d to 1 |
| **Polynomial basis dimension** | P_ℓ | Number of basis functions at level ℓ | **Decreases** with level |
| **Time discretisation** | N_ℓ | Number of timesteps at level ℓ | **Increases** with level |
| **Transport map dimension** | d × d | Dimension of Brenier matrix A | **Fixed** at d × d |

---

## 2. Asset Dimension (d)

### What It Is

The number of correlated assets in your basket. This is the fundamental dimension of the problem.

### In the Code

**File:** `example_american_basket_pricing.py`

```python
d = 3  # Number of assets
S0 = np.linspace(225, 275, num=d)[:, np.newaxis]  # Initial prices
vol = np.array([0.2, 0.15, 0.1])  # Asset volatilities
cov_mat = np.array([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.1],
    [0.3, 0.1, 1.0]
])  # Correlation matrix (d × d)
```

### The Curse of Dimensionality

Solving the full problem directly requires an N^d grid. For d = 10 assets with N = 100 grid points per dimension:

$$\text{Grid points} = 100^{10} = 10^{20}$$

This is computationally impossible.

### Physics Analogy: Many-Body Problem

Think of each asset as a degree of freedom in a particle system:
- A 3-asset basket is like a **3-body problem**
- A 50-asset basket is like a **50-body problem**
- The configuration space (phase space) explodes exponentially with d

Just as we cannot solve the full quantum many-body Schrödinger equation directly, we cannot solve the d-dimensional Black-Scholes PDE directly for large d.

---

## 3. Markovian Projection (d → 1)

### This Is the Big Dimensional Reduction

**Before projection:** d-dimensional process **X**(t) ∈ ℝ^d

$$dX_i(t) = r X_i(t) \, dt + \sigma_i X_i(t) \, dW_i(t), \quad i = 1, \ldots, d$$

**After projection:** 1-dimensional process S̄(t) ∈ ℝ

$$d\bar{S}(t) = r \bar{S}(t) \, dt + \bar{b}(t, \bar{S}(t)) \, dW(t)$$

### How It Works: Gyöngy's Lemma

Gyöngy's Lemma (1986) tells us that if we choose b̄²(t, s) correctly:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\|\boldsymbol{\sigma}(\mathbf{X}_t)\|^2 \mid S_t = s\right]$$

then the marginal distributions match:

$$\mathcal{L}(S_t) = \mathcal{L}(\bar{S}_t) \quad \forall t$$

where S_t = (1/d) Σᵢ Xᵢ(t) is the true basket value.

### In the Code

**File:** `SL_legendre_utilities.py`

```python
# Generate d-dimensional paths
paths = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t)  # Shape: (M, N_t, d)

# Project to 1D basket
basket_paths = paths @ basket_weights  # Shape: (M, N_t) — now 1D!
```

### What Is Preserved vs Lost

| Preserved | Lost |
|-----------|------|
| Marginal distributions at each t | Full path structure |
| Option prices (European) | Joint distributions |
| Terminal payoff distribution | Correlation dynamics |

### Physics Analogy: Born-Oppenheimer Approximation

This is exactly like the Born-Oppenheimer approximation in quantum chemistry:

| Quantum Chemistry | Financial Mathematics |
|-------------------|----------------------|
| Full Hamiltonian (electrons + nuclei) | Full d-dimensional SDE |
| Integrate out electrons | Integrate out individual assets |
| Effective nuclear potential | Effective volatility b̄(t, s) |
| Energy eigenvalues preserved | Option prices preserved |

The wavefunction for the full system is different from the reduced wavefunction, but the observable energies match. Similarly, the paths are different, but the pricing-relevant distributions match.

### Complexity Reduction

| Approach | Grid Complexity | For d = 10, N = 100 |
|----------|-----------------|---------------------|
| Direct PDE | O(N^d) | 10²⁰ points |
| Markovian Projection | O(N²) | 10⁴ points |

This is a reduction by a factor of 10¹⁶!

---

## 4. Polynomial Basis Dimension (P_ℓ)

### What It Is

The number of basis functions used to approximate b̄²(t, s) at each MLMC level.

### The Formula

For total-degree polynomials in 2D (time t and space s):

$$P_d = \binom{d+2}{2} = \frac{(d+1)(d+2)}{2}$$

| Polynomial Degree d | Basis Size P_d | Basis Functions |
|---------------------|----------------|-----------------|
| 0 | 1 | {1} |
| 1 | 3 | {1, t, s} |
| 2 | 6 | {1, t, s, t², ts, s²} |
| 3 | 10 | {1, t, s, t², ts, s², t³, t²s, ts², s³} |

### In the Code

**File:** `FML_utils.py`

```python
def tot_degree_poly(maxdeg=3):
    """Generate index pairs for total-degree polynomial basis."""
    return [(i, j) for i in range(maxdeg + 1) 
            for j in range(maxdeg + 1) if (i + j <= maxdeg)]
```

### Why It Decreases with MLMC Level

This is the **multi-resolution principle**:

$$\text{deg}_\ell = \text{max\_deg} - \ell$$

| Level ℓ | Degree d_ℓ | Basis Size P_ℓ | Sample Size M_ℓ |
|---------|------------|----------------|-----------------|
| 0 | 3 | 10 | 80 × 100 = 8000 |
| 1 | 2 | 6 | 80 × 36 = 2880 |
| 2 | 1 | 3 | 80 × 9 = 720 |
| 3 | 0 | 1 | 80 × 1 = 80 |

**In the code** (`ML_telescoping_sum.py`):

```python
# Level-dependent polynomial degree
l_V = max_deg - l
print(f"Level {l}: polynomial degree = {l_V}")

pairs = tot_degree_poly(l_V)
dimV = len(pairs)

# Sample size scales with basis dimension
M_l = max(C, int(C * dimV**2))
```

### Physics Analogy: Renormalisation Group

Think of this like a Fourier series or wavelet decomposition:

| Level | Physics Analogue | What It Captures |
|-------|------------------|------------------|
| Level 0 (coarse) | Low-energy effective theory | Global shape, many modes |
| Level 3 (fine) | High-energy corrections | Tiny corrections, few modes |

This is exactly the **renormalisation group** idea from QFT:
- Coarse scales need many degrees of freedom to capture complex structure
- Fine scales only correct small details

### Another Analogy: Image Compression (JPEG)

| MLMC Level | JPEG Analogue |
|------------|---------------|
| Level 0 | Blurry thumbnail (many DCT coefficients) |
| Level 1 | Add medium detail |
| Level 2 | Add fine detail |
| Level 3 | Add sharpening (few coefficients) |

---

## 5. Time Discretisation (N_ℓ)

### What It Is

The number of timesteps at each MLMC level.

### How It Changes

$$h_\ell = h_0 \cdot 2^{-\ell}, \qquad N_\ell = \frac{T}{h_\ell} = N_0 \cdot 2^\ell$$

| Level ℓ | Timestep h_ℓ | Number of Steps N_ℓ |
|---------|--------------|---------------------|
| 0 | h₀ = 0.05 | 20 |
| 1 | h₀/2 = 0.025 | 40 |
| 2 | h₀/4 = 0.0125 | 80 |
| 3 | h₀/8 = 0.00625 | 160 |

### In the Code

**File:** `ML_telescoping_sum.py`

```python
# Level-dependent timesteps
hl_f = h0 * 2 ** (-l)
N_f = int(round(T / hl_f))

hl_c = 2 * hl_f  # Coarse timestep is double
N_c = N_f // 2   # Half as many coarse steps
```

### This Is NOT a Dimensional Reduction

Time discretisation is a **resolution change**, not a dimensional reduction. But it is crucial for:
1. Understanding the fine/coarse coupling
2. The telescoping sum structure
3. Variance reduction through path correlation

### The Fine-Coarse Relationship

At each MLMC level ℓ > 0:

```
Fine path:   |--•--•--•--•--•--•--•--•--|  (N_f steps of size h_f)
             
Coarse path: |----•----•----•----•----|   (N_c = N_f/2 steps of size 2h_f)
```

The coupling uses:
$$\Delta W_n^{\text{coarse}} = \Delta W_{2n}^{\text{fine}} + \Delta W_{2n+1}^{\text{fine}}$$

---

## 6. Optimal Transport Map Dimensions

### The Key Insight Your Supervisor Likely Wants

**The Brenier map operates in the full d-dimensional asset space, NOT in the 1D projected space!**

### The Map Structure

The Gaussian-Brenier map is:

$$T: \mathbb{R}^d \to \mathbb{R}^d$$

$$T(\mathbf{x}) = \boldsymbol{\mu}_c + A(\mathbf{x} - \boldsymbol{\mu}_f)$$

where:
- **μ_f** ∈ ℝ^d is the mean of fine log-paths
- **μ_c** ∈ ℝ^d is the mean of coarse log-paths  
- **C_f** ∈ ℝ^(d×d) is the covariance of fine log-paths
- **C_c** ∈ ℝ^(d×d) is the covariance of coarse log-paths
- **A** ∈ ℝ^(d×d) is the transport matrix

### The Transport Matrix Formula

$$A = C_f^{-1/2} \left( C_f^{1/2} C_c C_f^{1/2} \right)^{1/2} C_f^{-1/2}$$

### In the Code

**File:** `FML_optimal_transport.py`

```python
class GaussianBrenierMap:
    """
    Optimal transport map between two Gaussian distributions.
    
    The matrix A is computed via the formula:
        A = C_f^{-1/2} (C_f^{1/2} C_c C_f^{1/2})^{1/2} C_f^{-1/2}
    """
    
    def __init__(self, mu_f, C_f, mu_c, C_c):
        self.mu_f = np.asarray(mu_f)        # Shape: (d,)
        self.mu_c = np.asarray(mu_c)        # Shape: (d,)
        self.C_f = np.asarray(C_f)          # Shape: (d, d)
        self.C_c = np.asarray(C_c)          # Shape: (d, d)

        # Eigendecomposition of C_f for numerical stability
        eig_f, U_f = np.linalg.eigh(self.C_f)
        sqrt_C_f = U_f @ np.diag(np.sqrt(eig_f)) @ U_f.T      # (d, d)
        invsqrt_C_f = U_f @ np.diag(1.0 / np.sqrt(eig_f)) @ U_f.T  # (d, d)

        # Middle matrix M = C_f^{1/2} C_c C_f^{1/2}
        M = sqrt_C_f @ self.C_c @ sqrt_C_f  # (d, d)

        # Eigendecomposition of M
        eig_M, U_M = np.linalg.eigh(M)
        sqrt_M = U_M @ np.diag(np.sqrt(eig_M)) @ U_M.T  # (d, d)

        # Brenier map: A = C_f^{-1/2} M^{1/2} C_f^{-1/2}
        self.A = invsqrt_C_f @ sqrt_M @ invsqrt_C_f  # (d, d)

    def map(self, x):
        """Apply the optimal transport map."""
        x = np.asarray(x)  # Shape: (M, d) or (d,)
        return self.mu_c + (x - self.mu_f) @ self.A.T  # Output: same shape as input
```

### Why Log-Space?

GBM produces **log-normal** distributions, not Gaussian. But:

$$\log X_i(t) \sim \mathcal{N}\left( \log x_{i,0} + \left(r - \frac{\sigma_i^2}{2}\right)t, \; \sigma_i^2 t \right)$$

So **log(X)** is multivariate Gaussian, making Brenier's theorem applicable.

**File:** `ML_optimal_transport.py`

```python
# Transform to log-space for Gaussian assumption
logpaths_c = np.log(paths_c)  # Shape: (M, N_c, d)
logpaths_f = np.log(reducedpaths_f)  # Shape: (M, N_c, d)

# Compute empirical means and covariances
mu_c = np.mean(logpaths_c, axis=0)  # Shape: (N_c, d)
mu_f = np.mean(logpaths_f, axis=0)  # Shape: (N_c, d)

# Covariances at each timestep
C_c = np.array([np.cov(logpaths_c[:, n, :].T, bias=False) 
                for n in range(N_c)])  # Shape: (N_c, d, d)
C_f = np.array([np.cov(logpaths_f[:, n, :].T, bias=False) 
                for n in range(N_c)])  # Shape: (N_c, d, d)
```

### The Dimensional Flow in OT

```
Input:  X_f ∈ ℝ^(M × d)     (M fine paths, each d-dimensional)
        ↓
        log(X_f) ∈ ℝ^(M × d)  (still d-dimensional)
        ↓
        T(log(X_f)) ∈ ℝ^(M × d)  (Brenier map, d → d)
        ↓
        exp(T(log(X_f))) ∈ ℝ^(M × d)  (back to price space)
        ↓
Output: X_c^{est} ∈ ℝ^(M × d)  (estimated coarse paths, d-dimensional)
```

**Key point:** The transport map never reduces dimension. It maps d-dimensional fine paths to d-dimensional coarse path estimates.

### Where Does Dimension Reduction Happen?

The Markovian projection happens **after** the OT coupling:

```
X_f ∈ ℝ^d  ──[OT map]──→  X_c^{est} ∈ ℝ^d
    │                          │
    │ basket_weights           │ basket_weights
    ↓                          ↓
S_f ∈ ℝ¹   ←─[difference]──   S_c^{est} ∈ ℝ¹
                │
                ↓
        ψ = b²(X_f) - b²(X_c^{est})  (response for regression)
```

### Physics Analogy: Phase Space Transformation

Think of the Brenier map like a **canonical transformation** in classical mechanics:

| Classical Mechanics | Optimal Transport |
|---------------------|-------------------|
| Phase space (q, p) ∈ ℝ^(2n) | Asset space X ∈ ℝ^d |
| Canonical transformation | Brenier map T |
| Preserves Hamiltonian structure | Preserves Gaussian structure |
| Symplectic matrix | Transport matrix A |
| Volume-preserving | Measure-preserving (pushforward) |

The transformation operates in the full phase space, not a reduced space.

### Time-Dependent Maps

A separate Brenier map is constructed at **each coarse timestep**:

**File:** `ML_optimal_transport.py`

```python
# Construct Brenier maps at each timestep
maps = [identity_map]  # Time 0: no transformation needed

for n in range(1, N_c):
    maps.append(GaussianBrenierMap(mu_f[n], C_f[n], mu_c[n], C_c[n]))
```

So we have a sequence of d × d transport matrices:

$$\{A_0 = I, \; A_1, \; A_2, \; \ldots, \; A_{N_c-1}\}$$

Each A_n captures how the fine and coarse distributions differ at time t_n.

---

## 7. Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FULL PROBLEM (Intractable)                      │
│                                                                         │
│   d assets: X₁(t), X₂(t), ..., X_d(t)  ∈ ℝᵈ                            │
│   Grid complexity: O(Nᵈ) ← CURSE OF DIMENSIONALITY                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ GYÖNGY'S LEMMA (Markovian Projection)
                                 │ Spatial dimension: d → 1
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PROJECTED PROBLEM (Tractable)                      │
│                                                                         │
│   1D process: S̄(t) ∈ ℝ¹                                                │
│   Need to estimate: b̄²(t, s) = E[σ²|S_t = s]                           │
│   Grid complexity: O(N²) ← TRACTABLE!                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ MLMC ESTIMATION
                                 │ (paths still d-dimensional internally!)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MLMC HIERARCHY                                 │
│                                                                         │
│   For each level ℓ = 0, 1, ..., L:                                     │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Generate d-dimensional paths:                                   │  │
│   │    X_f ∈ ℝᵈ (fine)    X_c ∈ ℝᵈ (coarse)                         │  │
│   │                                                                  │  │
│   │  [If using OT] Apply Brenier map T: ℝᵈ → ℝᵈ                     │  │
│   │    X_c^{est} = exp(T(log(X_f)))                                  │  │
│   │                                                                  │  │
│   │  Project to 1D:                                                  │  │
│   │    S_f = weights · X_f ∈ ℝ¹                                      │  │
│   │    S_c = weights · X_c^{est} ∈ ℝ¹                                │  │
│   │                                                                  │  │
│   │  Compute squared diffusion:                                      │  │
│   │    ψ = ||σ(X_f)||² - ||σ(X_c)||²                                 │  │
│   │                                                                  │  │
│   │  Fit polynomial (dimension P_ℓ):                                 │  │
│   │    b̄²(t,s) ≈ Σ cₖ φₖ(t,s)  where k = 1, ..., P_ℓ               │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   Level 0: h₀,      deg = 3,  P = 10 basis,  M = 8000 samples          │
│   Level 1: h₀/2,    deg = 2,  P = 6 basis,   M = 2880 samples          │
│   Level 2: h₀/4,    deg = 1,  P = 3 basis,   M = 720 samples           │
│   Level 3: h₀/8,    deg = 0,  P = 1 basis,   M = 80 samples            │
│                                                                         │
│   Telescoping sum: c_total = c₀ + c₁ + c₂ + c₃                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ PDE SOLVER
                                 │ (1D backward Euler)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OPTION PRICE                                  │
│                                                                         │
│   Solve 1D PDE: ∂U/∂t + ½b̄²∂²U/∂S² + rS∂U/∂S - rU = 0                │
│   with early exercise constraint: U ≥ payoff                           │
│                                                                         │
│   Output: U(0, S₀) = American basket option value                      │
│           S*(t) = Exercise boundary                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Summary Tables

### Table 1: All Dimensions at a Glance

| Dimension Type | Symbol | Typical Value | Changes? | Where in Pipeline |
|----------------|--------|---------------|----------|-------------------|
| Asset dimension | d | 3-50 | Fixed | Throughout |
| Projected dimension | 1 | 1 | Reduced from d | After Gyöngy projection |
| Polynomial basis | P_ℓ | 1-10 | Decreases with ℓ | MLMC regression |
| Fine timesteps | N_ℓ | 20-160 | Increases with ℓ | Path simulation |
| Coarse timesteps | N_ℓ/2 | 10-80 | Half of fine | Path simulation |
| Transport matrix | d × d | 3×3 to 50×50 | Fixed at d×d | OT coupling |
| Covariance matrices | d × d | 3×3 to 50×50 | Fixed at d×d | OT coupling |

### Table 2: What Changes at Each MLMC Level

| Level ℓ | Timestep h_ℓ | Steps N_ℓ | Poly Degree | Basis P_ℓ | Samples M_ℓ |
|---------|--------------|-----------|-------------|-----------|-------------|
| 0 | h₀ | 20 | 3 | 10 | 8000 |
| 1 | h₀/2 | 40 | 2 | 6 | 2880 |
| 2 | h₀/4 | 80 | 1 | 3 | 720 |
| 3 | h₀/8 | 160 | 0 | 1 | 80 |

### Table 3: Dimensional Flow Through Operations

| Operation | Input Dimension | Output Dimension | Notes |
|-----------|-----------------|------------------|-------|
| GBM simulation | d | d | d-dimensional paths |
| Cholesky decomposition | d × d | d × d | Correlation → increments |
| Log transformation | d | d | For Gaussian OT |
| Brenier map T | d | d | Optimal coupling |
| Exp transformation | d | d | Back to prices |
| Basket projection | d | 1 | Markovian projection |
| Polynomial evaluation | 2 (t, s) | P_ℓ | Basis functions |
| PDE solve | 1 | 1 | 1D spatial grid |

### Table 4: Code Files and Their Dimensional Roles

| File | Primary Dimensional Operation |
|------|-------------------------------|
| `example_american_basket_pricing.py` | Sets d, initialises d-dim parameters |
| `GBM_paths()` in utilities | Generates d-dimensional paths |
| `FML_optimal_transport.py` | d × d Brenier matrices |
| `tot_degree_poly()` | Generates P_ℓ basis pairs |
| `mlmc_level()` | Level-dependent P_ℓ and N_ℓ |
| `basket_weights @ paths` | Projects d → 1 |
| `solve_american_option()` | 1D PDE solver |

---

## Appendix: Quick Reference for Common Questions

**Q: What dimension are the paths?**  
A: Always d-dimensional until projected to basket.

**Q: What dimension is the Brenier map?**  
A: T: ℝ^d → ℝ^d (operates in full asset space).

**Q: Where does d → 1 happen?**  
A: When computing basket value S = weights · X.

**Q: Why does polynomial degree decrease with level?**  
A: Multi-resolution principle: coarse levels capture global shape, fine levels only correct small errors.

**Q: What's the dimension of C_f and C_c?**  
A: Both are d × d covariance matrices.

**Q: What's preserved by Markovian projection?**  
A: Marginal distributions of the basket at each time t.

**Q: What's lost by Markovian projection?**  
A: Full path structure and joint distributions across time.
