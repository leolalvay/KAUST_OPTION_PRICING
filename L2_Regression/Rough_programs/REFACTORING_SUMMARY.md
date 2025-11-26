# Markovian Projection - Refactored Code Summary

## Files Delivered

### 1. **DM_org_refactored.py** (Basic Implementation)
- Based on: `DM_org.py` from Amelie's work
- Polynomial basis: Simple monomials {1, s, t, ts}
- Assets: Independent GBM with identical volatility
- Training paths: 100
- Basis functions: 4

**Results:**
- Condition number: $\sim 1.09 \times 10^4$
- Validation errors: 4.62% mean, 0.29% std
- Status: ✅ Excellent projection

### 2. **DesignMatrix_refactored.py** (Advanced Implementation)
- Based on: `DesignMatrix.py` from Amelie's work
- Polynomial basis: Orthonormalized Legendre polynomials
- Assets: Correlated GBM with different volatilities
- Training paths: 200
- Basis functions: 16 (degree 3 in both t and s)

**Results:**
- Condition number: $\sim 7.12 \times 10^2$ (15× better!)
- Validation errors: 1.29% mean, 0.94% std
- Status: ✅ Excellent projection

---

## Key Improvements in Refactoring

### Structure & Documentation
✅ Comprehensive module docstrings with purpose, method, and physical analogies
✅ Every function has detailed docstrings (Parameters, Returns, Notes)
✅ Step-by-step workflow with clear progress indicators
✅ Educational comments connecting to physics/maths concepts

### Variable Naming
✅ Mathematical variables preserved: d, r, vol, X, t, s, D, c, psi
✅ Descriptive names for clarity: stock_price_paths, basket_weights
✅ Standard notation: S_bar (not projected_basket)
✅ Inline documentation for all variable declarations

### Code Quality
✅ Modular functions (each does one thing well)
✅ No magic numbers (everything parameterized)
✅ Proper error checking and diagnostics
✅ Publication-quality plots with clear labels

### Educational Features
✅ Print statements showing progress at each step
✅ Diagnostic warnings (conditioning, negative variance)
✅ Success criteria with visual feedback (✅, ✓, ⚠️)
✅ Physics analogies throughout (QM, thermodynamics, etc.)

---

## Comparison Table

| Feature | DM_org_refactored | DesignMatrix_refactored |
|---------|-------------------|-------------------------|
| **Basis Functions** | Monomials $(1, s, t, ts)$ | Legendre polynomials $\tilde{P}_i(t) \otimes \tilde{P}_j(s)$ |
| **Number of Basis** | 4 | 16 |
| **Polynomial Degree** | 1 | 3 |
| **Asset Correlations** | Independent | Correlated ($\rho=0.8, 0.3, 0.1$) |
| **Volatilities** | Equal ($\sigma=0.15$) | Different ($\sigma = [0.2, 0.15, 0.1]$) |
| **Training Paths** | 100 | 200 |
| **Condition Number** | $\sim 10^4$ | $\sim 10^2$ |
| **Mean Error** | 4.62% | 1.29% |
| **Std Error** | 0.29% | 0.94% |
| **Data Rescaling** | No | Yes (to $[-1,1]$) |
| **Visualization** | Histogram only | Histogram + 3D surface |

---

## Mathematical Background

### Why Legendre Polynomials?

**Problem with monomials:** Powers of the same variable $(1, t, t^2, t^3)$ become 
nearly collinear for large datasets, causing:
- Large condition numbers $\text{cond}(\mathbf{D}) \sim 10^4$ to $10^8$
- Numerical instability in solving $\mathbf{D} \cdot \mathbf{c} = \boldsymbol{\psi}$
- Loss of precision in fitted coefficients

**Solution - Orthogonal polynomials:** Legendre polynomials satisfy:

$$\int_{-1}^{1} P_m(x) P_n(x) \, dx = \frac{2}{2n+1} \delta_{mn}$$

After orthonormalization with factor $\sqrt{(2n+1)/2}$:

$$\langle \tilde{P}_m, \tilde{P}_n \rangle = \delta_{mn}$$

This gives:
- Much smaller condition numbers $\text{cond}(\mathbf{D}) \sim 10^2$ to $10^3$
- Better numerical stability
- More accurate coefficient estimates

**Physical analogy:** Like using eigenfunctions instead of arbitrary basis states 
in quantum mechanics - you diagonalize the problem!

### The Correlated GBM Model

For $d$ assets with correlation matrix $\boldsymbol{\Sigma}$:

$$dX_i = r X_i \, dt + \sigma_i X_i (G \cdot dW)_i$$

where $G$ is the Cholesky factor: $\boldsymbol{\Sigma} = G G^{\top}$

The basket variance becomes:

$$\text{Var}(d\bar{S}) = \mathbf{w}^{\top} \cdot \text{diag}(\boldsymbol{\sigma} \odot \mathbf{X}) \cdot \boldsymbol{\Sigma} \cdot \text{diag}(\boldsymbol{\sigma} \odot \mathbf{X}) \cdot \mathbf{w}$$

This generalizes the simple case (independent assets):

$$\text{Var}(d\bar{S}) = \frac{\sigma^2}{d^2} \sum_{i=1}^{d} X_i^2$$

---

## Next Steps in Amelie's Code

According to the project guide, the progression is:

1. ✅ **DM_org.py** → Simple concept (DONE)
2. ✅ **DesignMatrix.py** → Legendre polynomials (DONE)
3. ⏭️ **L2_regression/SimplePolynomials/** → Production single-level
4. ⏭️ **L2_regression/LegendrePolynomials/Single_Level/** → Advanced single-level
5. ⏭️ **L2_regression/LegendrePolynomials/Multi_Level/** → MLMC + projection

The Multi-Level implementation is where the real computational gains happen:
- Telescoping sum: $\mathbb{E}[\bar{b}^2_L] = \mathbb{E}[\bar{b}^2_0] + \sum_{l=1}^{L} \mathbb{E}[\bar{b}^2_l - \bar{b}^2_{l-1}]$
- Different polynomial degrees per level
- Variance reduction through level coupling
- $\mathcal{O}(\varepsilon^{-2})$ complexity instead of $\mathcal{O}(\varepsilon^{-3})$

---

## Files Location

Both refactored files are in: `/mnt/user-data/outputs/`

1. `DM_org_refactored.py`
2. `DesignMatrix_refactored.py`

You can download them directly from Claude's interface!

---

## Usage Notes

### Running the Code

```bash
# Basic implementation
python DM_org_refactored.py

# Advanced implementation with Legendre
python DesignMatrix_refactored.py
```

Both scripts will:
1. Print detailed progress information
2. Display diagnostic statistics
3. Generate validation plots
4. Save plots as PDFs

### Customization

Key parameters to modify:

```python
# In setup_market_parameters()
d = 3                           # Number of assets
vol = np.array([0.2, 0.15, 0.1])  # Volatilities
num_training_paths = 200        # Training data size

# In setup_legendre_basis()
max_deg_t = 3                   # Time polynomial degree
max_deg_s = 3                   # Space polynomial degree
```

### Interpreting Results

**Good projection:**
- Condition number $< 10^6$
- Relative residual $< 0.05$
- Mean/std errors $< 10\%$
- Validation histograms overlap well

**Warning signs:**
- Condition number $> 10^8$ → Use orthogonal polynomials
- $\bar{b}^2(t,s) < 0$ → Increase training data or reduce degree
- Large errors $> 20\%$ → Check polynomial degree and basis choice

---

## Theoretical Background

### Gyöngy's Lemma

**Statement:** Let $\mathbf{X}$ be a $d$-dimensional diffusion with dynamics:

$$d\mathbf{X}_t = \boldsymbol{\mu}(\mathbf{X}_t) \, dt + \boldsymbol{\sigma}(\mathbf{X}_t) \, d\mathbf{W}_t$$

and let $\bar{S}_t = f(\mathbf{X}_t)$ be a scalar functional. Then there exists a 1D SDE:

$$d\bar{S}_t = r \bar{S}_t \, dt + \bar{b}(t, \bar{S}_t) \, dB_t$$

such that $\text{Law}(\bar{S}_t) = \text{Law}(f(\mathbf{X}_t))$ for all $t \in [0,T]$, where:

$$\bar{b}^2(t,s) = \mathbb{E}\left[\|\boldsymbol{\sigma}(\mathbf{X}_t)\|^2 \mid f(\mathbf{X}_t) = s\right]$$

**Practical implication:** We can replace the $d$-dimensional problem with a 
1D problem if we correctly estimate $\bar{b}(t,s)$!

### $L^2$ Regression

We estimate $\bar{b}^2(t,s)$ by:
1. Generating sample paths of $\mathbf{X}$
2. Computing local variance at each point: $\psi = \sigma^2(\mathbf{X})$
3. Fitting: $\bar{b}^2(t,\bar{S}) \approx \sum_p c_p \varphi_p(t,\bar{S})$ where $\varphi_p$ are basis functions
4. Solving: $\underset{\mathbf{c}}{\text{argmin}} \|\mathbf{D} \cdot \mathbf{c} - \boldsymbol{\psi}\|^2$

This is standard least squares regression!

---

**Created:** November 2024  
**Author:** Wadoud Charbak  
**Based on:** Amelie's American Option Pricing research code
