# What Are the Legendre Polynomials Actually Fitting?

**Purpose:** Clear explanation of the regression target in our MLMC + Markovian projection framework.

**Code Reference:** All snippets from `Comparison_Between_Methods/methods/mlmc_ot_estimator.py`

---

## The Short Answer

As I said in our meeting and I want to properly clarify, we are fitting **b²(t, S)**, the **projected volatility squared**, the 2D surface that tells us: *"Given that the basket has value S at time t, what is the expected instantaneous variance of that basket?"*

This is defined by **Gyöngy's Lemma** as a conditional expectation:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\text{Instantaneous Basket Variance} \;\Big|\; \text{Basket Value} = s\right]$$

The Legendre polynomial regression approximates this unknown surface as:

$$\bar{b}^2(t, s) \approx \sum_{p} c_p \, \tilde{P}_{i_1}\!\left(\frac{2t}{T} - 1\right) \, \tilde{P}_{i_2}\!\left(\frac{2(s - s_{\min})}{s_{\max} - s_{\min}} - 1\right)$$

where $\tilde{P}_n$ are orthonormalised Legendre polynomials and $c_p$ are the coefficients we solve for.

---

## The Full Mathematical Picture

### 1. The High-Dimensional Problem

We have $d$ correlated assets following GBM:

$$dX_i(t) = r X_i(t) \, dt + \sigma_i X_i(t) \, dW_i(t), \quad \text{with } dW_i \cdot dW_j = \rho_{ij} \, dt$$

The basket value is:

$$S(t) = \sum_{i=1}^{d} w_i X_i(t) = \vec{P}_1 \cdot \vec{X}(t)$$

where $\vec{P}_1 = (w_1, \ldots, w_d)$ are the basket weights.

### 2. What Gyöngy's Lemma Guarantees

**Theorem (Gyöngy 1986):** There exists a 1D process $\bar{S}(t)$ satisfying:

$$d\bar{S}(t) = r\bar{S}(t) \, dt + \bar{b}(t, \bar{S}) \, dW(t)$$

such that $\text{Law}(S(t)) = \text{Law}(\bar{S}(t))$ for all $t \geq 0$, **if and only if** we choose:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\left\|\vec{P}_1^\top \boldsymbol{\sigma}(t, \vec{X})\right\|^2 \;\Big|\; \vec{P}_1 \cdot \vec{X}(t) = s\right]$$

**Key insight:** The paths of $S(t)$ and $\bar{S}(t)$ are different, but their **marginal distributions** match at every time $t$. This is sufficient for option pricing.

### 3. Expanding the Conditional Expectation

For our GBM model with $\boldsymbol{\sigma}(t, \vec{X}) = \text{diag}(\sigma_1 X_1, \ldots, \sigma_d X_d)$:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\sum_{i,j} w_i w_j \sigma_i \sigma_j \rho_{ij} X_i(t) X_j(t) \;\Bigg|\; \sum_k w_k X_k(t) = s\right]$$

Or in matrix notation:

$$\bar{b}^2(t, s) = \mathbb{E}\left[(\vec{w} \odot \vec{\sigma} \odot \vec{X})^\top \boldsymbol{\rho} \, (\vec{w} \odot \vec{\sigma} \odot \vec{X}) \;\Big|\; \vec{w} \cdot \vec{X} = s\right]$$

**This cannot be computed analytically** because:
1. The sum of log-normals (the basket) has no closed-form density
2. The conditioning constrains us to a $(d-1)$-dimensional hyperplane
3. The integrand is nonlinear in $\vec{X}$

---

## The Regression Framework (with actual code)

Since we cannot compute $\bar{b}^2(t, s)$ analytically, we **approximate it via L² regression**.

### Step 1: The Target ψ — Instantaneous Basket Variance

At each simulation point, we compute the **exact instantaneous variance** from the known asset prices. This is the quantity that b²(t, S) is trying to predict.

From `mlmc_ot_estimator.py`, **Level 0** (single-level, no coarse correction):

```python
# Instantaneous variance (target for regression)
# h = b²S² = (P1 ⊙ σ ⊙ X)ᵀ Σ (P1 ⊙ σ ⊙ X)
# NOTE: We do NOT divide by d². That convention assumes normalised
# weights [1/d, ..., 1/d], but we use P1 directly (typically [1,1,1]).
# Laplace returns values ~1000-2000, matching this formula.
w_sigma_f = X_f * vol * P1  # Shape: (B, d) = wᵢσᵢXᵢ
b_sq = ((w_sigma_f @ cov_mat) * w_sigma_f).sum(axis=1)
psi_n = b_sq.reshape(-1, 1)
```

**What this computes:**

$$\psi = \sum_{i,j} (w_i \sigma_i X_i) \rho_{ij} (w_j \sigma_j X_j) = (\vec{w} \odot \vec{\sigma} \odot \vec{X})^\top \boldsymbol{\rho} \, (\vec{w} \odot \vec{\sigma} \odot \vec{X})$$

This is the **instantaneous variance of the basket** at that specific point. It's a noisy observation of $\bar{b}^2(t, S)$ because many different asset configurations $\vec{X}$ can produce the same basket value $S$.

### Step 2: The Design Matrix D — Legendre Polynomial Basis

We precompute Legendre polynomials on the time grid:

```python
# Precompute Legendre values for time grid
t_grid = np.arange(N_c) * hl_c
t_scaled = 2.0 * t_grid / T - 1.0           # Map [0, T] → [-1, 1]
deg_t = l_V
deg_s = l_V
norm_t = np.sqrt(2 * np.arange(deg_t + 1) + 1)  # Orthonormalisation: √(2n+1)
norm_s = np.sqrt(2 * np.arange(deg_s + 1) + 1)
VT = legvander(t_scaled, deg_t) * norm_t        # Shape: (N_c, deg_t+1)
```

Then at each timestep, we build the design matrix rows:

```python
# Build design matrix row block
D_n = np.empty((B, dimV), dtype=float)
trow = VT[n, :]                                  # Pre-computed time polynomials
s = X_f @ P1                                     # Basket value S = P1·X
s_scaled = np.clip(2.0 * (s - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
VS = legvander(s_scaled, deg_s) * norm_s         # Space polynomials

for p, (i1, i2) in enumerate(pairs):
    D_n[:, p] = trow[i1] * VS[:, i2]             # Tensor product: P̃_{i1}(t) · P̃_{i2}(S)
```

**What this builds:**

$$D_{mn,p} = \tilde{P}_{i_1}\!\left(\frac{2t_n}{T} - 1\right) \cdot \tilde{P}_{i_2}\!\left(\frac{2(S_m^n - s_{\min})}{s_{\max} - s_{\min}} - 1\right)$$

where $\tilde{P}_k(x) = \sqrt{2k+1} \cdot P_k(x)$ are **orthonormalised** Legendre polynomials.

### Step 3: Accumulated Normal Equations (Memory-Efficient)

Instead of storing the full design matrix $D \in \mathbb{R}^{MN \times \text{dimV}}$, we accumulate:

```python
# Accumulators for normal equations
G = np.zeros((dimV, dimV))
g = np.zeros((dimV, 1))

# ... inside the simulation loop:
G += D_n.T @ D_n    # Gram matrix accumulation
g += D_n.T @ psi_n  # Moment vector accumulation
```

This reduces memory from $O(MN \cdot \text{dimV})$ to $O(\text{dimV}^2)$ — a factor of ~$10^7$ for typical problems!

### Step 4: Solve via Cholesky Decomposition

```python
# Solve normal equations via Cholesky
G_reg = 0.5 * (G + G.T)  # Ensure symmetry

# Check condition number
lam = np.linalg.eigvalsh(G_reg)
lam_min = max(lam[0], 1e-300)
cond_approx = np.sqrt(lam[-1] / lam_min)

if verbose:
    print(f"    Condition number ≈ {cond_approx:.2e}")

# Add regularisation if ill-conditioned
if cond_approx > 1e12:
    reg = 1e-10 * lam[-1]
    G_reg += reg * np.eye(dimV)
    if verbose:
        print(f"    Added regularisation: {reg:.2e}")

L = np.linalg.cholesky(G_reg)
y = np.linalg.solve(L, g)
c = np.linalg.solve(L.T, y).ravel()
```

This solves $Gc = g$ where $G = D^\top D$ and $g = D^\top \psi$, giving us the coefficient vector $c$.

---

## The MLMC Telescoping: Fitting Differences

For levels $\ell > 0$, we don't fit $b^2$ directly — we fit the **difference** between fine and coarse discretisations:

```python
# Telescoping difference: b²_fine - b²_coarse
# h = b²S² = (P1 ⊙ σ ⊙ X)ᵀ Σ (P1 ⊙ σ ⊙ X)
w_sigma_f = X_f * vol * P1
b_f = ((w_sigma_f @ cov_mat) * w_sigma_f).sum(axis=1)
w_sigma_c = X_c * vol * P1
b_c = ((w_sigma_c @ cov_mat) * w_sigma_c).sum(axis=1)
psi_n = (b_f - b_c).reshape(-1, 1)
```

**Why this works:** The fine and coarse paths are **coupled** (they use the same Brownian increments), so $b_f - b_c$ has much lower variance than $b_f$ alone. The telescoping sum reconstructs the full answer:

$$\mathbf{c}_{\text{total}} = \mathbf{c}_0 + \sum_{\ell=1}^{L} (\mathbf{c}_\ell^{\text{fine}} - \mathbf{c}_\ell^{\text{coarse}})$$

From `make_c()`:

```python
def make_c(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min, s_max, ...):
    """
    Compute full coefficient vector via MLMC telescoping sum.
    
    Implements c = Σ_{l=0}^{max_deg} c_l where each c_l is computed
    using accumulated normal equations at level l with Brownian coupling.
    """
    c_total = np.zeros(len(tot_degree_poly(max_deg)))
    
    for level in range(max_deg + 1):
        c_l = mlmc_level(x0, T, h0, level, r, cov_mat, vol, max_deg,
                         P1, s_min, s_max, C, batch_size, verbose)
        c_total += c_l
    
    return c_total
```

---

## Constructing the Volatility Surface

Once we have coefficients, we build the callable b̄(t, S):

```python
def make_b_bar(c, pairs, s_min, s_max, T, max_deg):
    """
    Construct callable b̄(t, S) from fitted coefficients.
    """
    norm = np.sqrt(2 * np.arange(max_deg + 1) + 1)
    
    def b_bar(t, S):
        # Scale to [-1, 1]
        tau = 2 * t / T - 1
        xi = np.clip(2 * (S - s_min) / (s_max - s_min) - 1, -1, 1)
        
        # Evaluate polynomial: b² = Σ cₚ P̃_{i1}(τ) P̃_{i2}(ξ)
        Pt = legval(tau, np.eye(max_deg + 1)) * norm
        Ps = legval(xi, np.eye(max_deg + 1)) * norm
        
        b_squared = sum(c[p] * Pt[i1] * Ps[i2] for p, (i1, i2) in enumerate(pairs))
        
        return np.sqrt(max(b_squared, 0))  # Safety floor
    
    return b_bar
```

---

## How b²(t, S) Is Used in the PDE

The fitted surface enters the **1D Black-Scholes PDE**:

$$\frac{\partial V}{\partial t} + \frac{1}{2} \bar{b}^2(t, S) \cdot S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

**Critical:** The diffusion term is $\bar{b}^2 S^2$, not just $\bar{b}^2$. The $S^2$ factor comes from the multiplicative noise structure of GBM. (This was a bug we fixed earlier — without the $S^2$, volatility had almost no effect on prices!)

---

## Summary: The Complete Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     WHAT WE'RE FITTING                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: d-dimensional GBM paths X_i(t)                                  │
│                                                                         │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  At each (path m, time n):                                       │   │
│  │                                                                  │   │
│  │  Predictor: (t_n, S_m^n) where S = P1·X                          │   │
│  │                                                                  │   │
│  │  Target: ψ = (w ⊙ σ ⊙ X)ᵀ ρ (w ⊙ σ ⊙ X)                          │   │
│  │           = instantaneous basket variance                        │   │
│  │                                                                  │   │
│  │  Code: w_sigma = X * vol * P1                                    │   │
│  │        psi = ((w_sigma @ cov_mat) * w_sigma).sum()               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  L² Regression via Accumulated Normal Equations:                 │   │
│  │                                                                  │   │
│  │  G += D_n.T @ D_n    (Gram matrix)                               │   │
│  │  g += D_n.T @ psi_n  (moment vector)                             │   │
│  │  Solve: Gc = g via Cholesky                                      │   │
│  │                                                                  │   │
│  │  where D[idx, p] = P̃_{i₁}(t_scaled) · P̃_{i₂}(S_scaled)          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  OUTPUT: Coefficient vector c                                           │
│                                                                         │
│          b̄²(t, S) = Σₚ cₚ P̃_{i₁}(t) P̃_{i₂}(S)                         │
│                                                                         │
│          This is the PROJECTED VOLATILITY SURFACE                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Points

1. **What we fit:** The surface $\bar{b}^2(t, S)$ — expected basket variance conditional on basket value

2. **Why we fit it:** Gyöngy's Lemma says this function makes the 1D projection match the d-dimensional basket's distribution

3. **What ψ is:** The instantaneous variance $(\vec{w} \odot \vec{\sigma} \odot \vec{X})^\top \rho (\vec{w} \odot \vec{\sigma} \odot \vec{X})$ computed exactly from simulated asset prices

4. **What the regression does:** Finds the polynomial that best predicts ψ from (t, S) — this approximates the conditional expectation

5. **Why Legendre:** Orthogonality gives ~100× better conditioning than monomials

6. **Why accumulated normal equations:** Reduces memory from O(MN·dimV) to O(dimV²)

7. **How MLMC helps:** Fits differences between levels (low variance) instead of absolute values

---

*Document created for to clarify my conversation with Erik*
