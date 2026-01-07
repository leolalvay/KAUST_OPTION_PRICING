# What Are the Legendre Polynomials Actually Fitting?

**Purpose:** Clear explanation of the regression target in our MLMC + Markovian projection framework.

**Code Reference:** All snippets from `Comparison_Between_Methods/methods/mlmc_ot_estimator.py`

---

# Part 1: The Big Picture

*What function are we ultimately trying to approximate?*

---

## The Short Answer

The ultimate goal is to approximate **$\bar{b}^2(t, S)$**, the **projected volatility squared** — a 2D surface that tells us: *"Given that the basket has value S at time t, what is the expected instantaneous variance of that basket?"*

This is defined by **Gyöngy's Lemma** as a conditional expectation:

$$\bar{b}^2(t, s) = \mathbb{E}\left[(P \cdot b(t, X_t))(P \cdot b(t, X_t))^\top \;\Big|\; P \cdot X_t = s\right]$$

We approximate this surface using a Legendre polynomial expansion:

$$\bar{b}^2(t, s) \approx \sum_{p} c_p \, \tilde{P}_{i_1}\!\left(\frac{2t}{T} - 1\right) \, \tilde{P}_{i_2}\!\left(\frac{2(s - s_{\min})}{s_{\max} - s_{\min}} - 1\right)$$

where $\tilde{P}_n$ are orthonormalised Legendre polynomials and $c_p$ are the coefficients we solve for.

**Important clarification:** While $\bar{b}^2(t,s)$ is the *target function* we want to approximate, the MLMC method does not directly regress samples of $\bar{b}^2$ from a single discretisation level. Instead, it uses a telescoping sum of level-wise regressions (detailed in Part 2).

---

## The Full Mathematical Picture

### 1. The High-Dimensional Problem

We have $d$ correlated assets following GBM:

$$dX_i(t) = r X_i(t) \, dt + \sigma_i X_i(t) \, dW_i(t), \quad \text{with } dW_i \cdot dW_j = \rho_{ij} \, dt$$

The basket value is:

$$S(t) = \sum_{i=1}^{d} w_i X_i(t) = P \cdot X(t)$$

where $P = (w_1, \ldots, w_d)$ are the basket weights.

### 2. What Gyöngy's Lemma Guarantees

**Theorem (Gyöngy 1986):** There exists a 1D process $\bar{S}(t)$ satisfying:

$$d\bar{S}(t) = r\bar{S}(t) \, dt + \bar{b}(t, \bar{S}) \, dW(t)$$

such that $\text{Law}(S(t)) = \text{Law}(\bar{S}(t))$ for all $t \geq 0$, **if and only if** we choose:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\left\|P^\top \boldsymbol{\sigma}(t, X)\right\|^2 \;\Big|\; P \cdot X(t) = s\right]$$

**Key insight:** The paths of $S(t)$ and $\bar{S}(t)$ are different, but their **marginal distributions** match at every time $t$. This is sufficient for option pricing.

### 3. Expanding the Conditional Expectation

For our GBM model with $\boldsymbol{\sigma}(t, X) = \text{diag}(\sigma_1 X_1, \ldots, \sigma_d X_d)$:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\sum_{i,j} w_i w_j \sigma_i \sigma_j \rho_{ij} X_i(t) X_j(t) \;\Bigg|\; \sum_k w_k X_k(t) = s\right]$$

Or in matrix notation:

$$\bar{b}^2(t, s) = \mathbb{E}\left[(P \odot \sigma \odot X)^\top \rho \, (P \odot \sigma \odot X) \;\Big|\; P \cdot X = s\right]$$

**This cannot be computed analytically** because:
1. The sum of log-normals (the basket) has no closed-form density
2. The conditioning constrains us to a $(d-1)$-dimensional hyperplane
3. The integrand is nonlinear in $X$

---

## The Single-Level Regression Framework

To build intuition, consider first the **single-level** approach (before introducing MLMC).

### The Target ψ — Instantaneous Basket Variance

At each simulation point, we compute the **exact instantaneous variance** from the known asset prices:

```python
# Instantaneous variance (target for regression)
# ψ = (P ⊙ σ ⊙ X)ᵀ ρ (P ⊙ σ ⊙ X)
w_sigma_f = X_f * vol * P1  # Shape: (B, d) = wᵢσᵢXᵢ
b_sq = ((w_sigma_f @ cov_mat) * w_sigma_f).sum(axis=1)
psi_n = b_sq.reshape(-1, 1)
```

**What this computes:**

$$\psi = \sum_{i,j} (w_i \sigma_i X_i) \rho_{ij} (w_j \sigma_j X_j) = (P \odot \sigma \odot X)^\top \rho \, (P \odot \sigma \odot X)$$

This is the **instantaneous variance of the basket** at that specific point. It's a noisy observation of $\bar{b}^2(t, S)$ because many different asset configurations $X$ can produce the same basket value $S$.

### The Design Matrix D — Legendre Polynomial Basis

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
s = X_f @ P1                                     # Basket value S = P·X
s_scaled = np.clip(2.0 * (s - s_min) / (s_max - s_min) - 1.0, -1.0, 1.0)
VS = legvander(s_scaled, deg_s) * norm_s         # Space polynomials

for p, (i1, i2) in enumerate(pairs):
    D_n[:, p] = trow[i1] * VS[:, i2]             # Tensor product: P̃_{i1}(t) · P̃_{i2}(S)
```

**What this builds:**

$$D_{mn,p} = \tilde{P}_{i_1}\!\left(\frac{2t_n}{T} - 1\right) \cdot \tilde{P}_{i_2}\!\left(\frac{2(S_m^n - s_{\min})}{s_{\max} - s_{\min}} - 1\right)$$

where $\tilde{P}_k(x) = \sqrt{2k+1} \cdot P_k(x)$ are **orthonormalised** Legendre polynomials.

### Accumulated Normal Equations (Memory-Efficient)

Instead of storing the full design matrix $D \in \mathbb{R}^{MN \times \text{dimV}}$, we accumulate:

```python
# Accumulators for normal equations
G = np.zeros((dimV, dimV))
g = np.zeros((dimV, 1))

# ... inside the simulation loop:
G += D_n.T @ D_n    # Gram matrix accumulation
g += D_n.T @ psi_n  # Moment vector accumulation
```

This reduces memory from $O(MN \cdot \text{dimV})$ to $O(\text{dimV}^2)$.

### Solve via Cholesky Decomposition

```python
# Solve normal equations via Cholesky
G_reg = 0.5 * (G + G.T)  # Ensure symmetry

# Check condition number
lam = np.linalg.eigvalsh(G_reg)
lam_min = max(lam[0], 1e-300)
cond_approx = np.sqrt(lam[-1] / lam_min)

# Add regularisation if ill-conditioned
if cond_approx > 1e12:
    reg = 1e-10 * lam[-1]
    G_reg += reg * np.eye(dimV)

L = np.linalg.cholesky(G_reg)
y = np.linalg.solve(L, g)
c = np.linalg.solve(L.T, y).ravel()
```

This solves $Gc = g$ where $G = D^\top D$ and $g = D^\top \psi$, giving us the coefficient vector $c$.

---

## Constructing the Volatility Surface

Once we have coefficients, we build the callable $\bar{b}(t, S)$:

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

## How $\bar{b}^2(t, S)$ Is Used in the PDE

The fitted surface enters the **1D Black-Scholes PDE**:

$$\frac{\partial V}{\partial t} + \frac{1}{2} \bar{b}^2(t, S) \cdot S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

**Critical:** The diffusion term is $\bar{b}^2 \cdot S^2$, not just $\bar{b}^2$. The $S^2$ factor comes from the multiplicative noise structure of GBM.

---

## Summary of Part 1

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     THE ULTIMATE GOAL                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: d-dimensional GBM paths X_i(t)                                  │
│                                                                         │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  At each (path m, time n):                                       │   │
│  │                                                                  │   │
│  │  Predictor: (t_n, S_m^n) where S = P·X                           │   │
│  │                                                                  │   │
│  │  Target: ψ = (P ⊙ σ ⊙ X)ᵀ ρ (P ⊙ σ ⊙ X)                          │   │
│  │           = instantaneous basket variance                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  L² Regression: minimise ‖Dc - ψ‖²                               │   │
│  │                                                                  │   │
│  │  where D[idx, p] = P̃_{i₁}(t_scaled) · P̃_{i₂}(S_scaled)          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                          │                                              │
│                          ▼                                              │
│  OUTPUT: Coefficient vector c such that                                 │
│                                                                         │
│          b̄²(t, S) ≈ Σₚ cₚ P̃_{i₁}(t) P̃_{i₂}(S)                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
1. **What we want:** The surface $\bar{b}^2(t, S) = \mathbb{E}[\text{basket variance} \mid \text{basket value} = S]$
2. **Why:** Gyöngy's Lemma says this function makes the 1D projection match the d-dimensional basket's distribution
3. **How (conceptually):** Regress instantaneous variance samples against $(t, S)$ using Legendre polynomials

---

# Part 2: The Precise MLMC Estimator Structure

*What is the exact form of the multi-level regression estimator, and what are its variance properties?*

---

## Why This Matters

Part 1 describes *what* we're ultimately trying to approximate: the function $\bar{b}^2(t,s)$. However, the MLMC method does **not** directly regress samples of $\bar{b}^2$ from one fixed discretisation level. Instead, it constructs an estimator via a **telescoping sum of level-wise regressions**.

To properly analyse sample allocation (i.e., how many samples $M_\ell$ to use at each level), we need the **explicit form of the full estimator** and its **variance structure**.

---

## The MLMC Regression Estimator: Explicit Form

### Level-Wise Coefficient Estimators

At each MLMC level $\ell$, we solve a separate regression problem.

**Level 0** (coarsest, no correction):
$$\hat{c}_0 = (D_0^\top D_0)^{-1} D_0^\top \psi_0$$

where:
- $D_0 \in \mathbb{R}^{M_0 N_0 \times \dim(V_L)}$ is the design matrix at level 0
- $\psi_0 \in \mathbb{R}^{M_0 N_0}$ contains instantaneous variance samples $\psi^{(0)}_{m,n}$
- $N_0 = T / h_0$ is the number of timesteps at the coarsest level
- $\dim(V_L)$ is the polynomial basis dimension at the finest level

**Level $\ell > 0$** (corrections):
$$\hat{c}_\ell = (D_\ell^\top D_\ell)^{-1} D_\ell^\top (\psi_\ell^f - \psi_\ell^c)$$

where:
- $D_\ell \in \mathbb{R}^{M_\ell N_\ell \times \dim(V_{L-\ell})}$ is the design matrix at level $\ell$
- $\psi_\ell^f$ contains fine-discretisation variance samples
- $\psi_\ell^c$ contains coarse-discretisation variance samples (coupled paths)
- $N_\ell = T / h_\ell$ with $h_\ell = h_0 \cdot 2^{-\ell}$
- $\dim(V_{L-\ell})$ decreases with level (multi-resolution)

**Crucial:** The regression target at level $\ell > 0$ is the **difference** $\psi_\ell^f - \psi_\ell^c$, not $\psi_\ell^f$ directly.

### The Telescoping Sum

The final coefficient estimator is:

$$\boxed{\hat{c}_{\text{total}} = \hat{c}_0 + \sum_{\ell=1}^{L} \hat{c}_\ell}$$

Expanding:

$$\hat{c}_{\text{total}} = (D_0^\top D_0)^{-1} D_0^\top \psi_0 + \sum_{\ell=1}^{L} (D_\ell^\top D_\ell)^{-1} D_\ell^\top (\psi_\ell^f - \psi_\ell^c)$$

This is the **complete expression** for the MLMC regression estimator.

### Code Correspondence

From `make_c()` in `mlmc_ot_estimator.py`:

```python
def make_c(x0, T, h0, r, cov_mat, vol, max_deg, P1, s_min, s_max, ...):
    """
    Compute full coefficient vector via MLMC telescoping sum.
    
    Implements c = Σ_{l=0}^{max_deg} c_l where each c_l is computed
    using accumulated normal equations at level l.
    """
    c_total = np.zeros(len(tot_degree_poly(max_deg)))
    
    for level in range(max_deg + 1):
        c_l = mlmc_level(...)  # Computes (D_l^T D_l)^{-1} D_l^T ψ_l
        c_total += c_l         # Telescoping sum
    
    return c_total
```

And at each level, from `mlmc_level()`:

```python
# Level 0: ψ = b² directly
if level == 0:
    w_sigma_f = X_f * vol * P1
    b_sq = ((w_sigma_f @ cov_mat) * w_sigma_f).sum(axis=1)
    psi_n = b_sq.reshape(-1, 1)

# Level > 0: ψ = b²_fine - b²_coarse
else:
    w_sigma_f = X_f * vol * P1
    b_f = ((w_sigma_f @ cov_mat) * w_sigma_f).sum(axis=1)
    w_sigma_c = X_c * vol * P1
    b_c = ((w_sigma_c @ cov_mat) * w_sigma_c).sum(axis=1)
    psi_n = (b_f - b_c).reshape(-1, 1)  # THE DIFFERENCE
```

---

## Variance Structure of the Estimator

### Single-Level Regression Variance

For a standard least-squares estimator $\hat{c} = (D^\top D)^{-1} D^\top \psi$, assuming $\psi = D c_{\text{true}} + \epsilon$ with $\text{Var}(\epsilon) = \sigma^2 I$:

$$\text{Cov}(\hat{c}) = \sigma^2 (D^\top D)^{-1}$$

More generally, if the residuals have heterogeneous variance:

$$\text{Cov}(\hat{c}) = (D^\top D)^{-1} D^\top \text{Cov}(\psi) D (D^\top D)^{-1}$$

### MLMC Regression Variance

Since the levels are computed from **independent samples**, the total covariance is:

$$\text{Cov}(\hat{c}_{\text{total}}) = \text{Cov}(\hat{c}_0) + \sum_{\ell=1}^{L} \text{Cov}(\hat{c}_\ell)$$

For each level:

$$\text{Cov}(\hat{c}_\ell) = \sigma_\ell^2 \, (D_\ell^\top D_\ell)^{-1}$$

where $\sigma_\ell^2$ is the residual variance at level $\ell$:
- Level 0: $\sigma_0^2 = \text{Var}(\psi^{(0)} - D_0 c_{\text{true}})$
- Level $\ell > 0$: $\sigma_\ell^2 = \text{Var}((\psi_\ell^f - \psi_\ell^c) - D_\ell \Delta c_\ell)$

**Key MLMC property:** The variance of the difference $\psi_\ell^f - \psi_\ell^c$ decreases as $\ell$ increases because:
1. Fine and coarse paths are **coupled** (same Brownian increments)
2. As $h_\ell \to 0$, fine and coarse discretisations converge

Specifically, for smooth payoffs with strong coupling:
$$\text{Var}(\psi_\ell^f - \psi_\ell^c) = O(h_\ell^{2\alpha})$$

for some $\alpha > 0$ (typically $\alpha = 1$ for Euler-Maruyama).

### The Gram Matrix Factor

The term $(D_\ell^\top D_\ell)^{-1}$ depends on:
1. **Sample size** $M_\ell$: More samples → larger $D_\ell^\top D_\ell$ → smaller inverse
2. **Basis dimension** $\dim(V_{L-\ell})$: More basis functions → larger matrix to invert
3. **Condition number** $\kappa(D_\ell)$: Poorly conditioned → amplifies variance

For well-conditioned regression with $M_\ell$ samples:

$$(D_\ell^\top D_\ell)^{-1} \approx \frac{1}{M_\ell N_\ell} \cdot (\text{something depending on basis})$$

---

## Implications for Sample Allocation

### Standard MLMC (Scalar Estimation)

For estimating a scalar $\mathbb{E}[P(X_T)]$, optimal allocation is:

$$M_\ell^* \propto \sqrt{\frac{V_\ell}{C_\ell}}$$

where $V_\ell = \text{Var}(P_\ell - P_{\ell-1})$ and $C_\ell$ is the cost per sample.

### MLMC Regression (Our Case)

The constraint is now on the **coefficient covariance matrix**:

$$\|\text{Cov}(\hat{c}_{\text{total}})\| \leq \varepsilon^2$$

for some matrix norm (e.g., spectral norm, Frobenius norm).

This expands to:

$$\left\| \sigma_0^2 (D_0^\top D_0)^{-1} + \sum_{\ell=1}^{L} \sigma_\ell^2 (D_\ell^\top D_\ell)^{-1} \right\| \leq \varepsilon^2$$

The optimal allocation must balance:
1. **Variance of targets** $\sigma_\ell^2$ (decreases with $\ell$ due to coupling)
2. **Gram matrix structure** $(D_\ell^\top D_\ell)^{-1}$ (depends on $M_\ell$ and basis)
3. **Cost per sample** $C_\ell$ (increases with $\ell$ due to finer timesteps)
4. **Basis dimension** $\dim(V_{L-\ell})$ (decreases with $\ell$ in our multi-resolution scheme)

### Current Implementation

Our current allocation uses:

$$M_\ell = C \cdot \dim(V_{L-\ell})^2$$

where $C = 80$ is a constant. This is motivated by:
- Cohen-Migliorati bounds suggesting $M \gtrsim \dim(V)^2$ for stable regression
- Empirical observation that this produces stable results

**However:** This is heuristic, not derived from a formal optimisation of the MLMC regression variance. A rigorous derivation would require:

1. Explicit expression for $\sigma_\ell^2$ in terms of problem parameters
2. Analysis of how $(D_\ell^\top D_\ell)^{-1}$ scales with $M_\ell$
3. Lagrange multiplier optimisation subject to total cost constraint

---

## Summary of Part 2

The complete MLMC regression estimator is:

$$\hat{c}_{\text{total}} = \underbrace{(D_0^\top D_0)^{-1} D_0^\top \psi_0}_{\text{Level 0: direct regression}} + \sum_{\ell=1}^{L} \underbrace{(D_\ell^\top D_\ell)^{-1} D_\ell^\top (\psi_\ell^f - \psi_\ell^c)}_{\text{Level } \ell: \text{ difference regression}}$$

With variance:

$$\text{Cov}(\hat{c}_{\text{total}}) = \sum_{\ell=0}^{L} \sigma_\ell^2 (D_\ell^\top D_\ell)^{-1}$$

**Key distinctions from Part 1:**
- We don't regress $\bar{b}^2(t,s)$ directly from one discretisation
- We regress **differences** $\psi^f - \psi^c$ at each level $\ell > 0$
- The telescoping structure is essential for variance reduction
- Optimal sample allocation requires analysing this full structure



---

*Document created to clarify conversation with Erik — Wadoud Charbak, January 2025*
