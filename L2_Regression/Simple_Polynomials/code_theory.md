# SimplePolynomials - Theoretical Background

**Based on:** DM_simplepoly.py, DM_tests.py, DM_tests2.py from Amelie's work  
**Date:** November 2024

---

## Overview

The SimplePolynomials folder contains **production-ready** implementations of single-level Markovian projection. These build on the proof-of-concept code from Rough_programs by adding:
1. Vectorised path generation (10-50× faster)
2. Weak error analysis (quantitative validation)
3. Convergence testing (how errors decay with sample size)

---

## Core Mathematical Framework

### Markovian Projection via $L^2$ Regression

**Given:** $d$-dimensional correlated GBM system

$$d\mathbf{X}_t = r\mathbf{X}_t \, dt + \text{diag}(\boldsymbol{\sigma} \odot \mathbf{X}_t) \cdot G \, d\mathbf{W}_t$$

where $G$ is the Cholesky factor: $\boldsymbol{\Sigma} = GG^{\top}$

**Goal:** Find 1D SDE for basket $\bar{S}_t = P_1^{\top} \mathbf{X}_t$

$$d\bar{S}_t = r\bar{S}_t \, dt + \bar{b}(t, \bar{S}_t) \, dB_t$$

such that $\text{Law}(\bar{S}_T) = \text{Law}(P_1^{\top} \mathbf{X}_T)$

**Method:** Fit $\bar{b}^2(t,s)$ to match instantaneous basket variance:

$$\bar{b}^2(t,s) \approx \sum_{p=1}^{P} c_p \varphi_p(t,s)$$

where $\{\varphi_p\}$ are polynomial basis functions.

---

## Polynomial Basis Choice

### Simple Monomials with Rescaling

The basis functions are:

$$\varphi_p(t,s) = t_c^{i_1} \cdot s_c^{i_2}$$

where rescaling ensures $t_c, s_c \in [-1, 1]$ approximately:

$$t_c = \frac{t - \mu_t}{\sigma_t}, \quad s_c = \frac{s - \mu_s}{\sigma_s}$$

**Standard choice:** $\{1, s_c, s_c^2, t_c\}$ corresponding to `pairs = [(0,0), (0,1), (0,2), (1,0)]`

**Why this degree structure?**
- **Quadratic in $s$:** Basket variance $\sim S^2$ for GBM (from $\sigma^2 X^2$ terms)
- **Linear in $t$:** Time dependence usually weaker (nearly constant vol over short horizons)

---

## Instantaneous Basket Variance

For equal-weighted basket with weights $w_i = 1/d$:

$$\text{Var}(d\bar{S}) = \mathbf{w}^{\top} \boldsymbol{\Sigma}_{\text{basket}} \mathbf{w}$$

where the basket covariance matrix is:

$$\boldsymbol{\Sigma}_{\text{basket}} = \text{diag}(\boldsymbol{\sigma} \odot \mathbf{X}) \cdot \boldsymbol{\Sigma} \cdot \text{diag}(\boldsymbol{\sigma} \odot \mathbf{X})$$

For equal weights:

$$\text{Var}(d\bar{S}) = \frac{1}{d^2} \sum_{i,j} \sigma_i \sigma_j X_i X_j \Sigma_{ij}$$

**Implementation:**
```python
sigma_matrix = np.diag(vol * X)
basket_cov = sigma_matrix @ cov_mat @ sigma_matrix
psi[idx] = basket_cov.sum() / d**2
```

---

## Weak Error: The Key Validation Metric

### Definition

**Weak error** measures how well the projected process prices options:

$$\varepsilon_{\text{weak}} = \frac{|\mathbb{E}_{\text{true}}[g(S_T)] - \mathbb{E}_{\text{proj}}[g(\bar{S}_T)]|}{|\mathbb{E}_{\text{true}}[g(S_T)]|}$$

where $g(\cdot)$ is the payoff function (e.g., $g(S) = \max(S - K, 0)$ for a call).

**Why weak error?**
- **Strong convergence** (path-wise matching) is impossible with dimension reduction
- **Weak convergence** (distributional matching) is sufficient for option pricing
- We only care about $\mathbb{E}[\text{payoff}]$, not individual path behaviour

### Physical Analogy: Observable vs Microscopic States

Think of this like comparing **macroscopic observables** in statistical mechanics:

**True process ($d$-dimensional):** Full microstate $\mathbf{X} = (X_1, ..., X_d)$
- Like knowing positions and momenta of all particles
- High-dimensional, expensive to simulate

**Projected process (1D):** Effective variable $\bar{S} = P_1^{\top} \mathbf{X}$
- Like knowing only total energy or pressure
- Low-dimensional, cheap to simulate

**Weak convergence:** Macroscopic observables match (temperature, pressure, energy)
- Microstates differ, but thermodynamic quantities agree
- Analogous to: option prices match, but individual paths differ

---

## Convergence Analysis

### Two Sources of Error

1. **Regression Error** (from fitting $\bar{b}^2$)
   - Decreases with training paths $M_t$: $\varepsilon_{\text{reg}} \sim M_t^{-\alpha}$
   - Depends on polynomial degree (higher → more flexible but needs more data)

2. **Monte Carlo Error** (from estimating expectations)
   - Decreases with validation samples: $\varepsilon_{\text{MC}} \sim M^{-1/2}$
   - Standard MC convergence rate

**Total error:**

$$\varepsilon_{\text{total}} \approx \varepsilon_{\text{reg}} + \varepsilon_{\text{MC}}$$

### Expected Behaviour

For fixed regression basis (fixed $M_t$), varying validation samples $M$:

$$\varepsilon_{\text{weak}} \approx \varepsilon_{\text{reg}} + \frac{C}{\sqrt{M}}$$

**Plot interpretation:**
- **Small $M$:** MC error dominates, error $\sim 1/\sqrt{M}$
- **Large $M$:** MC error negligible, error plateaus at $\varepsilon_{\text{reg}}$
- **Saturation point:** When MC error ≈ regression error

---

## Vectorisation Strategy

### Path Generation

**Old (loop-based):**
```python
for m in range(M_t):
    X = x0
    for n in range(N_t):
        # Generate one path, one step at a time
```

**New (vectorised):**
```python
X = np.tile(x0, (M_t, 1))  # All paths start at x0
for n in range(N_t):
    Z = np.random.randn(M_t, d)  # All noise at once
    X = X + r*X*dt + sigma*dW*sqrtdt  # All paths updated simultaneously
```

**Speed gain:** ~10× for path generation

### Weak Error Computation

**Old (nested loops):**
```python
for trial in range(10):
    for m in range(M):
        # Simulate one path
```

**New (fully vectorised):**
```python
Z = np.random.randn(trials, M, N_t)  # All noise pre-generated
S = np.full((trials, M), S0)  # All paths initialized
for n in range(N_t):
    S = S + r*S*dt + b_bar(t[n], S)*Z[:,:,n]*sqrtdt  # Broadcast over trials and paths
```

**Speed gain:** ~50× for weak error analysis

**Key insight:** Pre-generate ALL random numbers, then use broadcasting to update all paths simultaneously.

---

## Exit Percentage: Domain Validation

During training, we fit $\bar{b}(t,s)$ on domain:

$$s \in [s_{\min}(t), s_{\max}(t)]$$

where:
- $s_{\min}(t) = 1^{\text{st}}$ percentile of basket at time $t$
- $s_{\max}(t) = 99^{\text{th}}$ percentile

**Problem:** When simulating $\bar{S}$, it might leave this domain!

**Exit percentage** tracks:

$$\text{Exit\%} = \frac{\text{\# paths where } \bar{S}(t) \notin [s_{\min}(t), s_{\max}(t)]}{\text{Total paths}} \times 100$$

**Interpretation:**
- **< 1%:** Excellent, $\bar{b}$ rarely extrapolates
- **1-5%:** Good, minor extrapolation
- **> 10%:** Warning, consider wider domain or more training data

**Physical analogy:** Like measuring what fraction of particles in a gas exceed escape velocity – if many particles escape, your container (domain) is too small!

---

## Comparison with Rough_programs

### From Prototype to Production

| Aspect | Rough_programs | SimplePolynomials |
|--------|----------------|-------------------|
| **Path generation** | Loop-based (slow) | Vectorised (10× faster) |
| **Validation** | Visual only (histogram) | Quantitative (weak error) |
| **Code structure** | Monolithic script | Modular functions |
| **Diagnostics** | Basic (residuals) | Comprehensive (error analysis) |
| **Testing** | Single run | Convergence studies |
| **Output** | Plots only | Plots + CSV data |

### When to Use Simple Polynomials

**Advantages:**
- Easy to interpret (monomials)
- Fast to evaluate
- Good for low degrees (≤ 3)

**Disadvantages:**
- Poor conditioning for high degrees ($\text{cond}(D) \sim 10^4$ for degree 3)
- Doesn't scale well to degree > 5

**Recommendation:** Use for quick tests and low-degree fits. For production with higher degrees, use Legendre polynomials (next folder).

---

## Key Results and Typical Performance

### Weak Error Benchmarks

For the standard test case:
- $d = 3$ assets
- Correlations: $\rho \in \{0.1, 0.3, 0.8\}$
- Volatilities: $\sigma \in \{0.1, 0.15, 0.2\}$
- Time horizon: $T = 1$ year
- Polynomial degree: 2 in space, 1 in time

**Expected weak errors:**
- $M_t = 200$ training paths: $\varepsilon_{\text{weak}} \approx 1-2\%$
- $M_t = 400$ training paths: $\varepsilon_{\text{weak}} \approx 0.5-1\%$

**Validation:** With $M = 64000$ validation paths, MC error $< 0.1\%$

### Condition Numbers

Simple monomials (rescaled):
- Degree 1: $\text{cond}(D) \sim 10^2$
- Degree 2: $\text{cond}(D) \sim 10^3$
- Degree 3: $\text{cond}(D) \sim 10^4$

**Warning threshold:** $\text{cond}(D) > 10^6$ indicates numerical instability

---

## Connection to Later Developments

### Why Multi-Level?

The single-level approach uses fixed time discretization $\Delta t$. Multi-level Monte Carlo (MLMC) improves this by:

1. **Telescoping sum:** 
$$\mathbb{E}[\bar{b}^2_L] = \mathbb{E}[\bar{b}^2_0] + \sum_{l=1}^{L} \mathbb{E}[\bar{b}^2_l - \bar{b}^2_{l-1}]$$

2. **Level-dependent degrees:**
   - Coarse levels ($l=0$): High-degree polynomials (capture main features)
   - Fine levels ($l>0$): Low-degree corrections (cheap to compute)

3. **Optimal sample allocation:**
   - More samples on cheap levels
   - Fewer samples on expensive levels

**Complexity improvement:**
- Single-level: $\mathcal{O}(\varepsilon^{-3})$ for tolerance $\varepsilon$
- Multi-level: $\mathcal{O}(\varepsilon^{-2})$ (same as MLMC for expectations!)

---

## Practical Tips

### Choosing Polynomial Degree

**Too low (degree 1):**
- Underfitting: can't capture basket variance structure
- High weak error (> 5%)
- Smooth but inaccurate $\bar{b}(t,s)$

**Too high (degree > 4 for monomials):**
- Overfitting: fits noise in training data
- Poor conditioning: $\text{cond}(D) > 10^6$
- Oscillatory $\bar{b}(t,s)$

**Sweet spot:** Degree 2-3 for simple monomials

### Choosing Training Sample Size

**Rule of thumb:** $M_t \geq 10 \times P$ where $P$ is number of basis functions

**Example:**
- 4 basis functions → $M_t \geq 40$ paths (but use 100-200 for safety)
- 10 basis functions → $M_t \geq 100$ paths (use 200-400)

**Validation:** Check that increasing $M_t$ reduces weak error

---

## Physics Analogies Summary

| Financial Concept | Physics Analogy |
|-------------------|-----------------|
| High-dimensional SDE | Full microstate (all particles) |
| 1D projection | Macroscopic observable (pressure, energy) |
| Weak convergence | Thermodynamic equivalence |
| Weak error | Error in macroscopic measurement |
| Polynomial basis | Basis expansion (like Fourier series) |
| QR decomposition | Gram-Schmidt orthogonalization |
| Exit percentage | Escape velocity fraction |
| Condition number | Measurement sensitivity / ill-posedness |

---

## Summary

**SimplePolynomials provides:**
1. **Production-ready single-level implementation**
   - Clean, modular code
   - Vectorised for speed
   - Comprehensive validation

2. **Weak error analysis framework**
   - Quantitative pricing accuracy
   - Convergence diagnostics
   - CSV output for further analysis

3. **Bridge to advanced methods**
   - Foundation for multi-level approaches
   - Baseline for comparing Legendre polynomials
   - Template for custom basis functions

**Next steps:** LegendrePolynomials folder for better conditioning and higher degrees.

---

**Document created:** November 2024  
**Based on:** Amelie's SimplePolynomials folder  
**Covers:** DM_simplepoly.py, DM_tests.py, DM_tests2.py
