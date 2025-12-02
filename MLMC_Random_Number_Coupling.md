# MLMC Random Number Generation and Level Coupling

**Author:** Wadoud Charbak  
**Date:** December 2024  
**Context:** KAUST Internship - American Basket Options Project

---

## Table of Contents

1. [What Distribution Do We Use?](#1-what-distribution-do-we-use)
2. [Why Gaussian Specifically?](#2-why-gaussian-specifically)
3. [Are Random Numbers Related Between Levels?](#3-are-random-numbers-related-between-levels)
4. [The Coupling Mechanism Explained](#4-the-coupling-mechanism-explained)
5. [Concrete Numerical Example](#5-concrete-numerical-example)
6. [Why Coupling Preserves Correct Statistics](#6-why-coupling-preserves-correct-statistics)
7. [The Variance Reduction Effect](#7-the-variance-reduction-effect)
8. [Summary](#8-summary)

---

## 1. What Distribution Do We Use?

We use **standard normal (Gaussian) random numbers**:

$$Z \sim \mathcal{N}(0, 1)$$

These are then scaled to create proper Brownian increments:

$$\Delta W = Z \cdot \sqrt{\Delta t}$$

### Code Example

From `SL_legendre_utilities.py`:

```python
def GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t):
    G = np.linalg.cholesky(cov_mat)  # Cholesky factor for correlations
    sqrtdt = math.sqrt(dt)
    
    for n in range(1, N_t):
        Z = np.random.randn(M_t, d)  # Independent Gaussian draws
        sigma = X * vol
        dW = Z @ G.T  # Correlated Brownian increments
        X = X + r * X * dt + sigma * dW * sqrtdt
```

The `np.random.randn(M_t, d)` generates standard normals Z ~ N(0, 1), then multiplied by √dt to get the correct variance.

---

## 2. Why Gaussian Specifically?

This comes directly from the **definition of Brownian motion**. A Wiener process W(t) has the property that increments are normally distributed:

$$W(t + \Delta t) - W(t) \sim \mathcal{N}(0, \Delta t)$$

### Physics Analogy

The Gaussian distribution here plays the same role as wave function spreading in quantum mechanics. Just as the position uncertainty of a free particle grows as √t, so does the standard deviation of Brownian motion.

### What Happens If We Change the Distribution?

Using a different distribution (uniform, exponential, etc.) would **break the fundamental mathematics**:

| Property | Gaussian ✓ | Other Distributions ✗ |
|----------|------------|----------------------|
| Brownian motion definition | Correct increments | Violates axioms |
| Central Limit Theorem | Sum of increments → Gaussian | May not converge correctly |
| Itô calculus | Valid stochastic integrals | Breaks Itô's lemma |
| Log-normal stock prices | GBM gives S > 0 | May give negative prices |

**Physics analogy**: It's like asking "what if we used a non-Hermitian Hamiltonian in quantum mechanics?" You'd lose probability conservation and unitarity. Similarly, non-Gaussian noise breaks the mathematical framework that makes Black-Scholes theory work.

---

## 3. Are Random Numbers Related Between Levels?

**Yes! This is the key MLMC innovation.**

The random numbers are **deliberately coupled between levels**, and this coupling is what makes MLMC vastly more efficient than standard Monte Carlo.

### The Core Idea

- Generate random numbers at the **finest level**
- Coarser levels **reuse the same random numbers**, just combined differently
- This creates high correlation between fine and coarse paths
- High correlation means Var[P_fine - P_coarse] is very small

---

## 4. The Coupling Mechanism Explained

### The Code

From `ML_telescoping_sum.py`:

```python
# Generate random increments (shared for coupling!)
Z = np.random.randn(M_l, N_f, d)

# ========================================================================
# Fine paths (timestep h_fine)
# ========================================================================
X = X0.copy()
for n in range(1, N_f):
    Z_n = Z[:, n-1, :]  # Use Z directly
    dW = (Z_n @ G.T) * math.sqrt(hl_f)
    X = X + r * X * hl_f + sigma * dW

# ========================================================================
# Coarse paths (timestep h_coarse = 2*h_fine)
# ========================================================================
# COUPLING: Sum consecutive pairs of fine increments
Z_c = Z.reshape(M_l, N_c, 2, d).sum(axis=2)  # ← THE MAGIC LINE!

X = X0.copy()
for n in range(1, N_c):
    Z_n = Z_c[:, n-1, :]  # Use SUMMED Z values
    dW = (Z_n @ G.T) * math.sqrt(hl_f)
    X = X + r * X * hl_c + sigma * dW
```

### The Mathematical Relationship

The coarse path uses summed pairs of fine increments:

$$\Delta W_n^{\text{coarse}} = \Delta W_{2n}^{\text{fine}} + \Delta W_{2n+1}^{\text{fine}}$$

This works because of the **Brownian bridge property**.

---

## 5. Concrete Numerical Example

### Setup

```python
M_l = 2    # 2 Monte Carlo paths
N_f = 6    # 6 fine timesteps
N_c = 3    # 3 coarse timesteps (half of fine)
d = 1      # 1 asset (keeps it simple)
```

### Step 1: Generate Fine Random Numbers

```python
Z = np.random.randn(M_l, N_f, d)  # Shape: (2, 6, 1)
```

Suppose we get these values:

```
Z = [
    # Path 0: 6 fine timesteps
    [[0.5], [1.2], [-0.3], [0.8], [0.1], [-0.7]],
    
    # Path 1: 6 fine timesteps  
    [[0.9], [-0.4], [0.6], [0.2], [-0.5], [1.1]]
]

Shape: (2, 6, 1)
        ↑  ↑  ↑
        |  |  └── 1 asset
        |  └───── 6 timesteps
        └──────── 2 paths
```

### Step 2: Reshape to Group Pairs

```python
Z.reshape(M_l, N_c, 2, d)  # Shape: (2, 3, 2, 1)
```

This **groups consecutive pairs** together:

```
Z_reshaped = [
    # Path 0
    [
        [[0.5], [1.2]],    # Fine steps 0,1 → coarse step 0
        [[-0.3], [0.8]],   # Fine steps 2,3 → coarse step 1
        [[0.1], [-0.7]]    # Fine steps 4,5 → coarse step 2
    ],
    
    # Path 1
    [
        [[0.9], [-0.4]],   # Fine steps 0,1 → coarse step 0
        [[0.6], [0.2]],    # Fine steps 2,3 → coarse step 1
        [[-0.5], [1.1]]    # Fine steps 4,5 → coarse step 2
    ]
]

Shape: (2, 3, 2, 1)
        ↑  ↑  ↑  ↑
        |  |  |  └── 1 asset
        |  |  └───── 2 fine steps per coarse step (the pair!)
        |  └──────── 3 coarse timesteps
        └─────────── 2 paths
```

### Step 3: Sum Along Axis 2 (The Magic!)

```python
Z_c = Z.reshape(M_l, N_c, 2, d).sum(axis=2)  # Shape: (2, 3, 1)
```

The `.sum(axis=2)` adds together each pair:

```
Z_c = [
    # Path 0
    [
        [0.5 + 1.2],      # = [1.7]   ← sum of fine steps 0,1
        [-0.3 + 0.8],     # = [0.5]   ← sum of fine steps 2,3
        [0.1 + (-0.7)]    # = [-0.6]  ← sum of fine steps 4,5
    ],
    
    # Path 1
    [
        [0.9 + (-0.4)],   # = [0.5]
        [0.6 + 0.2],      # = [0.8]
        [-0.5 + 1.1]      # = [0.6]
    ]
]

Final Z_c = [
    [[1.7], [0.5], [-0.6]],   # Path 0: 3 coarse increments
    [[0.5], [0.8], [0.6]]     # Path 1: 3 coarse increments
]

Shape: (2, 3, 1)
```

### Visual Diagram

```
FINE PATH (6 steps):
   Z[0]   Z[1]   Z[2]   Z[3]   Z[4]   Z[5]
   0.5    1.2   -0.3    0.8    0.1   -0.7
    ↓      ↓      ↓      ↓      ↓      ↓
    └──┬───┘      └──┬───┘      └──┬───┘
       ↓             ↓             ↓
      SUM           SUM           SUM
       ↓             ↓             ↓
      1.7           0.5          -0.6
       ↓             ↓             ↓
   Z_c[0]        Z_c[1]        Z_c[2]

COARSE PATH (3 steps)
```

### Python Verification Script

```python
import numpy as np

# Small example
M_l, N_f, d = 2, 6, 1
N_c = N_f // 2

# Generate fine randoms
np.random.seed(42)  # For reproducibility
Z = np.random.randn(M_l, N_f, d)

print("Fine Z shape:", Z.shape)
print("Fine Z values (Path 0):", Z[0, :, 0])

# The magic line
Z_c = Z.reshape(M_l, N_c, 2, d).sum(axis=2)

print("\nCoarse Z_c shape:", Z_c.shape)
print("Coarse Z_c values (Path 0):", Z_c[0, :, 0])

# Verify: first coarse = sum of first two fine
print("\nVerification:")
print(f"Z[0,0] + Z[0,1] = {Z[0,0,0]:.4f} + {Z[0,1,0]:.4f} = {Z[0,0,0] + Z[0,1,0]:.4f}")
print(f"Z_c[0,0] = {Z_c[0,0,0]:.4f}")
```

---

## 6. Why Coupling Preserves Correct Statistics

If each fine increment is standard normal:

$$Z_{\text{fine}} \sim \mathcal{N}(0, 1)$$

Then the sum of two independent standard normals is:

$$Z_{\text{coarse}} = Z_{\text{fine}}^{(1)} + Z_{\text{fine}}^{(2)} \sim \mathcal{N}(0, 2)$$

When we multiply by √h_fine to get actual Brownian increments:

| Path Type | Increment | Distribution |
|-----------|-----------|--------------|
| Fine | Z × √h_f | N(0, h_f) ✓ |
| Coarse | (Z₁ + Z₂) × √h_f | N(0, 2h_f) = N(0, h_c) ✓ |

Both get the **correct variance** for their respective timesteps!

### Mathematical Proof

For the Brownian bridge property, if:
- $\Delta W_1 \sim \mathcal{N}(0, h_f)$
- $\Delta W_2 \sim \mathcal{N}(0, h_f)$
- $\Delta W_1$ and $\Delta W_2$ are independent

Then:
$$\Delta W_1 + \Delta W_2 \sim \mathcal{N}(0 + 0, h_f + h_f) = \mathcal{N}(0, 2h_f) = \mathcal{N}(0, h_c)$$

This is exactly what a coarse Brownian increment should be!

---

## 7. The Variance Reduction Effect

### The Correlation Structure

Both fine and coarse paths are driven by **exactly the same random numbers**, just combined differently:

| Time Interval | Fine Path Uses | Coarse Path Uses |
|---------------|----------------|------------------|
| 0 to h_c | Z[0]=0.5, Z[1]=1.2 separately | Z[0]+Z[1]=1.7 together |
| h_c to 2h_c | Z[2]=-0.3, Z[3]=0.8 separately | Z[2]+Z[3]=0.5 together |
| 2h_c to 3h_c | Z[4]=0.1, Z[5]=-0.7 separately | Z[4]+Z[5]=-0.6 together |

If the fine path drifts upward (positive Z values), the coarse path **also drifts upward**.

### Correlation Analysis

From ML_Theory.md:

> For the specific coupling:
> - E[W^f(t) W^c(t)] = t (they share all increments)
> - Var[W^f(t)] = t
> - Var[W^c(t)] = t
>
> Therefore: **Corr(W^f(t), W^c(t)) = 1** (perfect at Brownian level!)

### The Key Result

Because of this near-perfect correlation:

$$\text{Var}[P_{\text{fine}} - P_{\text{coarse}}] \ll \text{Var}[P_{\text{fine}}]$$

This means the **correction terms** in the MLMC telescoping sum have tiny variance, so we need far fewer samples at expensive fine levels.

### Physics Analogy: Common-Mode Noise Rejection

Think of it like differential measurements in experimental physics:

| MLMC Concept | Physics Equivalent |
|--------------|-------------------|
| Fine path | Far detector (DUNE) |
| Coarse path | Near detector (DUNE) |
| Shared randomness | Same neutrino beam |
| Difference P_f - P_c | Oscillation signal |
| Variance reduction | Systematic cancellation |

The near and far detectors see the same beam, so taking ratios cancels flux uncertainties. Similarly, MLMC fine and coarse paths share the same "beam" (Brownian motion), so their difference has much smaller variance than either individually.

---

## 8. Summary

### Key Points

| Question | Answer |
|----------|--------|
| What distribution? | Gaussian N(0,1), scaled by √Δt |
| Why Gaussian? | Required by Brownian motion definition and Itô calculus |
| Are levels related? | **YES!** Coarse paths sum consecutive fine increments |
| Why relate them? | Creates high correlation, dramatically reduces variance of differences |
| Net effect | MLMC complexity O(ε⁻²(log ε)²) vs standard MC's O(ε⁻³) |

### The Magic Line Explained

```python
Z_c = Z.reshape(M_l, N_c, 2, d).sum(axis=2)
```

This does exactly one thing: **pairs up consecutive fine random numbers and adds them together** to create the coarse random numbers.

### Why This Matters

The coupling ensures:
1. The coarse path has the **correct statistical distribution**
2. The fine and coarse paths are **maximally correlated** (same underlying randomness)
3. Therefore **Var[Y_fine - Y_coarse] is tiny**
4. MLMC achieves massive computational savings

### Visual Summary of Level Structure

```
Level 0 (coarsest):  ●─────────────────●─────────────────●
                     
Level 1:             ●────────●────────●────────●────────●
                          ↑        ↑        ↑        ↑
                          └────┬───┘        └────┬───┘
                               ↓                  ↓
                           Sum pairs          Sum pairs
                         
Level 2 (finest):    ●────●────●────●────●────●────●────●
                     
                     All levels share the SAME underlying Z values!
```

The telescoping sum then becomes:

$$\mathbb{E}[Y_L] = \mathbb{E}[Y_0] + \sum_{\ell=1}^{L} \mathbb{E}[\underbrace{Y_\ell - Y_{\ell-1}}_{\text{small variance!}}]$$

Each correction term has small variance (because of coupling), so you need fewer samples at fine levels where computation is expensive.

---

## References

- Giles, M.B. (2008). "Multilevel Monte Carlo Path Simulation." Operations Research.
- Project files: `ML_telescoping_sum.py`, `ML_Theory.md`, `SL_legendre_utilities.py`
