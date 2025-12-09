# Optimal Transport Dimension Summary

**Key Point:** The Brenier map operates in **d-dimensional** asset space, not 1D.

---

## The Map Signature

$$T: \mathbb{R}^d \to \mathbb{R}^d$$

$$T(\mathbf{x}) = \boldsymbol{\mu}_c + A(\mathbf{x} - \boldsymbol{\mu}_f)$$

| Object | Dimension | Description |
|--------|-----------|-------------|
| **x** (input) | d | Log of fine asset prices |
| **μ_f** | d | Mean of fine log-paths |
| **μ_c** | d | Mean of coarse log-paths |
| **C_f** | d × d | Covariance of fine log-paths |
| **C_c** | d × d | Covariance of coarse log-paths |
| **A** | d × d | Transport matrix |
| **T(x)** (output) | d | Estimated coarse log-prices |

---

## The Transport Matrix

$$A = C_f^{-1/2} \left( C_f^{1/2} C_c C_f^{1/2} \right)^{1/2} C_f^{-1/2} \in \mathbb{R}^{d \times d}$$

Computed via eigendecomposition:
$$\begin{align}
C_f &= U_f \Lambda_f U_f^T \quad \Rightarrow \quad C_f^{1/2} = U_f \Lambda_f^{1/2} U_f^T \\
M &= C_f^{1/2} C_c C_f^{1/2} \quad \text{(symmetric positive definite)} \\
M &= U_M \Lambda_M U_M^T \quad \Rightarrow \quad M^{1/2} = U_M \Lambda_M^{1/2} U_M^T \\
A &= C_f^{-1/2} M^{1/2} C_f^{-1/2}
\end{align}$$

---

## Dimensional Flow

```
┌─────────────────────────────────────────────────────────────┐
│  FINE PATHS                                                 │
│  X_f ∈ ℝ^{M × N × d}                                        │
│  (M paths, N timesteps, d assets)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓  log transform (elementwise)
┌─────────────────────────────────────────────────────────────┐
│  LOG FINE PATHS                                             │
│  log(X_f) ∈ ℝ^{M × N × d}     ← Still d-dimensional!       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓  Brenier map T: ℝ^d → ℝ^d (at each timestep)
┌─────────────────────────────────────────────────────────────┐
│  TRANSPORTED LOG PATHS                                      │
│  T(log(X_f)) ∈ ℝ^{M × N × d}  ← Still d-dimensional!       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓  exp transform (elementwise)
┌─────────────────────────────────────────────────────────────┐
│  ESTIMATED COARSE PATHS                                     │
│  X_c^{est} ∈ ℝ^{M × N × d}    ← Still d-dimensional!       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓  Markovian projection: weights · X  (NOW d → 1)
┌─────────────────────────────────────────────────────────────┐
│  PROJECTED BASKET                                           │
│  S_c = w^T X_c^{est} ∈ ℝ^{M × N}   ← Now 1-dimensional!    │
└─────────────────────────────────────────────────────────────┘
```

---

## Why d-Dimensional OT Before 1D Projection?

**Exploits full correlation structure.** The d × d covariance matrices capture how all assets co-move. Projecting first would lose this information.

**Gaussian assumption exact.** Log-GBM is multivariate Gaussian in ℝ^d. Brenier's theorem gives closed-form optimal map.

**Provably optimal coupling.** Minimises E[||X_f - T(X_f)||²] among all valid transport maps.

---

## Time-Dependent Maps

At each coarse timestep t_n, we have a **different** d × d transport matrix:

$$\{A_0 = I_d, \; A_1, \; A_2, \; \ldots, \; A_{N_c-1}\} \quad \text{where each } A_n \in \mathbb{R}^{d \times d}$$

The statistics μ_f, μ_c, C_f, C_c change with time, so the optimal coupling changes too.

---

## Code Reference

```python
# In GaussianBrenierMap.__init__():
self.C_f = np.asarray(C_f)          # Shape: (d, d)
self.C_c = np.asarray(C_c)          # Shape: (d, d)
self.A = invsqrt_C_f @ sqrt_M @ invsqrt_C_f  # Shape: (d, d)

# In map():
def map(self, x):
    x = np.asarray(x)  # Shape: (M, d)
    return self.mu_c + (x - self.mu_f) @ self.A.T  # Shape: (M, d)
```

---

## Summary

| Stage | Dimension | Operation |
|-------|-----------|-----------|
| Asset paths | d | GBM simulation |
| Log transform | d | Elementwise log |
| Brenier map | d → d | Affine transform with d×d matrix |
| Exp transform | d | Elementwise exp |
| **Basket projection** | **d → 1** | **Inner product with weights** |
| PDE solve | 1 | 1D spatial grid |

**The OT coupling happens in ℝ^d. The dimension reduction to ℝ¹ happens afterwards.**
