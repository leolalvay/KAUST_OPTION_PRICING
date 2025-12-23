# The Laplace Approximation Method for American Basket Options

**A Comprehensive Guide to Theory, Implementation, and Application**

Author: Wadoud Charbak (KAUST Internship)
Reference: Bayer, Häppölä, Tempone (2017) - "Implied Stopping Rules for American Basket Options from Markovian Projection"

---

## Table of Contents

1. [The Big Picture: Why Do We Need This?](#1-the-big-picture-why-do-we-need-this)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [The Laplace Approximation: Core Theory](#3-the-laplace-approximation-core-theory)
4. [Implementation Walkthrough](#4-implementation-walkthrough)
5. [From Volatility Surface to Option Price](#5-from-volatility-surface-to-option-price)
6. [Practical Example: Pricing a 3D Basket Put](#6-practical-example-pricing-a-3d-basket-put)
7. [Comparison with MLMC](#7-comparison-with-mlmc)
8. [Summary and Key Takeaways](#8-summary-and-key-takeaways)

---

## 1. The Big Picture: Why Do We Need This?

### The Problem: Curse of Dimensionality

Imagine you want to price an American put option on a basket of 10 stocks. Each stock follows its own random walk, and they're all correlated with each other. The "obvious" approach would be to solve the pricing PDE directly:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sum_{i,j} \sigma_i \sigma_j \rho_{ij} S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} + r\sum_i S_i \frac{\partial V}{\partial S_i} - rV = 0$$

But here's the problem: if you use $N$ grid points per dimension, you need $N^{10}$ total grid points. With just $N = 100$, that's $10^{20}$ points. Even the world's fastest supercomputer would take longer than the age of the universe!

**Physics Analogy:** This is like trying to simulate every molecule in a gas. Instead, we use thermodynamics (effective theories). Similarly, we'll project our 10D problem down to 1D.

### The Solution: Markovian Projection

The key insight comes from **Gyöngy's Lemma** (1986): we can replace the complicated 10-dimensional basket dynamics with a single 1D process that has the *same distribution* at every point in time.

The projected 1D SDE is:

$$d\bar{S}(t) = a(t, \bar{S}) \, dt + b(t, \bar{S}) \, dW(t)$$

where:
- $\bar{S}(t)$ is the projected basket value
- $a(t, S) = rS$ is the drift (trivially computed)
- $b(t, S)$ is the projected volatility (the hard part!)

**The entire computational challenge reduces to computing $b(t, S)$.**

---

## 2. Mathematical Foundation

### 2.1 The High-Dimensional Model

Consider $d$ assets following correlated Geometric Brownian Motion (GBM):

$$dX_i(t) = r X_i(t) \, dt + \sigma_i X_i(t) \, dW_i(t), \quad i = 1, \ldots, d$$

where:
- $X_i(t)$ is the price of asset $i$ at time $t$
- $r$ is the risk-free interest rate
- $\sigma_i$ is the volatility of asset $i$
- $dW_i \cdot dW_j = \rho_{ij} \, dt$ captures correlations

The basket value is:

$$S(t) = \sum_{i=1}^{d} w_i X_i(t) = P_1 \cdot \mathbf{X}(t)$$

where $P_1 = (w_1, \ldots, w_d)$ are the portfolio weights.

### 2.2 What Gyöngy's Lemma Tells Us

**Theorem (Gyöngy, 1986):** There exists a 1D process $\bar{S}(t)$ such that:

$$\text{Law}(S(t)) = \text{Law}(\bar{S}(t)) \quad \forall t \geq 0$$

if we choose:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\text{Instantaneous Basket Variance} \,\big|\, \text{Basket Value} = s\right]$$

**In plain English:** The projected volatility at basket value $s$ is the *average* of all possible volatilities across all asset configurations that could produce that basket value.

### 2.3 Why the Drift is Trivial

For the drift coefficient:

$$a(t, s) = \mathbb{E}\left[P_1 \cdot (r\mathbf{X}(t)) \,\big|\, P_1 \cdot \mathbf{X}(t) = s\right]$$

Since expectation is linear and all assets drift at rate $r$:

$$\boxed{a(t, s) = r \cdot s}$$

No computation needed! This can be hardcoded directly.

### 2.4 Why the Volatility is Difficult

The projected volatility squared (Paper Equation 13) is:

$$b^2(t, s) = \mathbb{E}\left[\sum_{i,j} w_i w_j \sigma_i \sigma_j \rho_{ij} X_i(t) X_j(t) \,\Bigg|\, \sum_k w_k X_k(t) = s\right]$$

This is challenging because:

1. **Non-linearity:** Volatility of a sum ≠ sum of volatilities
2. **Conditional expectation:** Must integrate over all configurations $\mathbf{X}$ satisfying $P_1 \cdot \mathbf{X} = s$
3. **Unknown density:** Sum of log-normals has no closed form
4. **High-dimensional integral:** Integration over $(d-1)$-dimensional hyperplane

**Physics Analogy:** This is like computing $\langle \phi^4 \rangle$ in quantum field theory. The constraint $P_1 \cdot \mathbf{X} = s$ is like a delta function, and we're doing a path integral over field configurations.

---

## 3. The Laplace Approximation: Core Theory

### 3.1 The Setup

We need to compute a ratio of integrals (Paper Equation 39):

$$b^2(t, s) = \frac{\int f(\mathbf{z}) \, d\mathbf{z}}{\int \tilde{f}(\mathbf{z}) \, d\mathbf{z}}$$

where:
- **Numerator integrand:** $f(\mathbf{z}) = \phi(\mathbf{x}(\mathbf{z})) \times (\text{basket variance at } \mathbf{x})$
- **Denominator integrand:** $\tilde{f}(\mathbf{z}) = \phi(\mathbf{x}(\mathbf{z}))$ (just the density)
- $\phi$ is the log-normal transition density
- $\mathbf{z} \in \mathbb{R}^{d-1}$ parametrises the constraint hyperplane $P_1 \cdot \mathbf{X} = s$

### 3.2 The Laplace Method: Saddle Point Approximation

The Laplace method approximates integrals of the form:

$$I = \int e^{M \cdot g(\mathbf{z})} \, d\mathbf{z}$$

For large $M$, the integral is dominated by the neighbourhood of the maximum of $g(\mathbf{z})$.

**Key Formula (Laplace Approximation):**

$$\int e^{g(\mathbf{z})} \, d\mathbf{z} \approx e^{g(\mathbf{z}^*)} \cdot \sqrt{\frac{(2\pi)^{d-1}}{|\det(-H_g)|}}$$

where:
- $\mathbf{z}^*$ is the mode (maximum) of $g$
- $H_g$ is the Hessian matrix of $g$ evaluated at $\mathbf{z}^*$

**Physics Analogy:** This is exactly the saddle-point approximation used in statistical mechanics to compute partition functions and in quantum mechanics for path integrals. The dominant contribution comes from the classical path (the extremum), with quantum corrections from fluctuations around it (captured by the Hessian).

### 3.3 Applying to Our Problem

For our volatility computation (Paper Equation 41):

$$b^2(t, s) = \exp(f^* - \tilde{f}^*) \cdot \sqrt{\frac{|\det(-H_{\tilde{f}})|}{|\det(-H_f)|}}$$

where:
- $f^* = f(\mathbf{z}^*)$ is the log-integrand at its mode for the numerator
- $\tilde{f}^* = \tilde{f}(\tilde{\mathbf{z}}^*)$ is the log-integrand at its mode for the denominator
- $H_f, H_{\tilde{f}}$ are the respective Hessians

### 3.4 Step-by-Step Algorithm

**Step 1: Parametrise the constraint hyperplane**

Given basket value $s$ and weights $P_1 = (w_1, \ldots, w_d)$:
- Use $\mathbf{z} = (X_2, X_3, \ldots, X_d)$ as free variables
- Compute $X_1 = (s - \sum_{j>1} w_j z_j) / w_1$

**Step 2: Define the log-integrands**

Numerator (for $b^2$):
$$\log f(\mathbf{z}) = \log \phi(\mathbf{x}(\mathbf{z}); \mathbf{x}_0, t) + \log(\text{basket variance at } \mathbf{x})$$

Denominator (normalisation):
$$\log \tilde{f}(\mathbf{z}) = \log \phi(\mathbf{x}(\mathbf{z}); \mathbf{x}_0, t)$$

**Step 3: Find the modes**

Use optimisation (L-BFGS-B with positivity constraints) to find:
- $\mathbf{z}^* = \arg\max \log f(\mathbf{z})$
- $\tilde{\mathbf{z}}^* = \arg\max \log \tilde{f}(\mathbf{z})$

**Step 4: Compute Hessians**

Use finite differences to approximate:
$$H_{ij} = \frac{\partial^2 \log f}{\partial z_i \partial z_j}\bigg|_{\mathbf{z} = \mathbf{z}^*}$$

**Step 5: Combine**

$$b^2(t, s) = \exp\left(f^* - \tilde{f}^* + \frac{1}{2}\left(\log|\det(-H_{\tilde{f}})| - \log|\det(-H_f)|\right)\right)$$

---

## 4. Implementation Walkthrough

### 4.1 File: `laplace_volatility.py`

This is the core implementation. Let's walk through the key functions:

#### 4.1.1 Log-Normal Density

```python
def black_scholes_log_density(x, x0, t, r, sigma, corr_chol):
    """
    Log density of multivariate Black-Scholes at time t.
    
    Under GBM: log(X_i(t)/X_i(0)) ~ N((r - σ²_i/2)t, σ²_i t)
    with correlation from corr_chol @ corr_chol.T
    """
    # Log returns
    log_returns = np.log(x / x0)
    
    # Mean under risk-neutral measure
    mu = (r - 0.5 * sigma**2) * t
    
    # Covariance: Cov[log X_i, log X_j] = σ_i σ_j ρ_ij t
    sigma_outer = np.outer(sigma, sigma)
    corr_matrix = corr_chol @ corr_chol.T
    cov_matrix = sigma_outer * corr_matrix * t
    
    # Multivariate normal log-density
    L = cholesky(cov_matrix, lower=True)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    y = solve_triangular(L, log_returns - mu, lower=True)
    quad_form = np.dot(y, y)
    
    # Jacobian for log-normal transformation
    log_jacobian = -np.sum(np.log(x))
    
    return -0.5 * (d * np.log(2*np.pi) + log_det + quad_form) + log_jacobian
```

**Physics Analogy:** This is like computing the action in the path integral formulation. The quadratic form $y^T y$ is analogous to $\int (\dot{x})^2 dt$ in classical mechanics.

#### 4.1.2 Numerator Integrand

```python
def projected_volatility_integrand_numerator(z, s, P1, x0, t, r, sigma, corr_chol):
    """
    Log of: density × basket_variance
    
    This is f(z) in our Laplace formula.
    """
    # Reconstruct full x from reduced coordinates z
    x = np.zeros(d)
    x[1:] = z
    x[0] = (s - np.dot(P1[1:], z)) / P1[0]  # Enforce constraint
    
    # Check positivity (asset prices must be positive)
    if x[0] <= 0 or np.any(z <= 0):
        return -np.inf, x
    
    # Log density
    log_phi = black_scholes_log_density(x, x0, t, r, sigma, corr_chol)
    
    # Basket variance: (P1 σ x)^T Corr (P1 σ x)
    corr_matrix = corr_chol @ corr_chol.T
    weighted = P1 * sigma * x
    vol_squared = weighted @ corr_matrix @ weighted
    
    return log_phi + np.log(vol_squared), x
```

#### 4.1.3 Mode Finding

```python
def find_laplace_mode(objective, z0, s, P1, x0, t, r, sigma, corr_chol):
    """
    Find maximum of log-integrand using L-BFGS-B.
    """
    def neg_objective(z):
        val, _ = objective(z, s, P1, x0, t, r, sigma, corr_chol)
        return -val if np.isfinite(val) else 1e10
    
    # Bounds: z_i > 0 (asset prices positive)
    bounds = [(1e-6, None) for _ in range(d - 1)]
    
    result = minimize(neg_objective, z0, method='L-BFGS-B', bounds=bounds)
    return result.x
```

#### 4.1.4 Hessian Computation

```python
def compute_hessian(objective, z_star, s, P1, x0, t, r, sigma, corr_chol, eps=1e-5):
    """
    Finite difference Hessian at mode.
    """
    d = len(z_star)
    H = np.zeros((d, d))
    
    f0, _ = objective(z_star, ...)
    
    for i in range(d):
        for j in range(i, d):
            # Four-point central difference for mixed partial
            z_pp, z_pm, z_mp, z_mm = [z_star.copy() for _ in range(4)]
            z_pp[i] += eps; z_pp[j] += eps
            z_pm[i] += eps; z_pm[j] -= eps
            z_mp[i] -= eps; z_mp[j] += eps
            z_mm[i] -= eps; z_mm[j] -= eps
            
            f_pp, _ = objective(z_pp, ...)
            f_pm, _ = objective(z_pm, ...)
            f_mp, _ = objective(z_mp, ...)
            f_mm, _ = objective(z_mm, ...)
            
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            H[j, i] = H[i, j]
    
    return H
```

#### 4.1.5 Main Laplace Function

```python
def laplace_approximation_volatility_squared(s, t, P1, x0, r, sigma, corr_chol):
    """
    Compute b²(t, s) using Laplace approximation (Paper Equation 41).
    
    Returns volatility SQUARED, not volatility.
    """
    # Handle t ≈ 0 (limiting case)
    if t < 1e-10:
        corr_matrix = corr_chol @ corr_chol.T
        weighted = P1 * sigma * x0
        return weighted @ corr_matrix @ weighted
    
    # Initial guess: forward prices scaled to match s
    forward = x0 * np.exp(r * t)
    scale = s / np.dot(P1, forward)
    z0 = forward[1:] * scale
    
    # Find modes for numerator and denominator
    z_star = find_laplace_mode(numerator_integrand, z0, ...)
    z_tilde = find_laplace_mode(denominator_integrand, z0, ...)
    
    # Evaluate at modes
    f_star, _ = numerator_integrand(z_star, ...)
    f_tilde, _ = denominator_integrand(z_tilde, ...)
    
    # Compute Hessians
    H_star = compute_hessian(numerator_integrand, z_star, ...)
    H_tilde = compute_hessian(denominator_integrand, z_tilde, ...)
    
    # Laplace formula
    det_H_star = np.linalg.det(-H_star)
    det_H_tilde = np.linalg.det(-H_tilde)
    
    log_b_squared = (f_star - f_tilde) + 0.5 * (np.log(det_H_tilde) - np.log(det_H_star))
    
    return np.exp(log_b_squared)
```

### 4.2 Computing the Full Surface

```python
def compute_volatility_surface(t_grid, s_grid, P1, x0, r, sigma, corr_chol):
    """
    Compute b²(t, s) over a grid.
    """
    N_t, N_s = len(t_grid), len(s_grid)
    b_squared_surface = np.zeros((N_t, N_s))
    
    for i, t in enumerate(t_grid):
        for j, s in enumerate(s_grid):
            b_squared_surface[i, j] = laplace_approximation_volatility_squared(
                s, t, P1, x0, r, sigma, corr_chol
            )
    
    return b_squared_surface
```

---

## 5. From Volatility Surface to Option Price

### 5.1 The 1D Pricing PDE

Once we have $b^2(t, s)$, we solve the 1D American option PDE (Paper Equation 18):

$$-\frac{\partial \bar{u}_A}{\partial t} = \max\left(\bar{\mathcal{L}}\bar{u}_A, 0\right)$$

where the differential operator is:

$$\bar{\mathcal{L}}u = \frac{1}{2}b^2(t, s)\frac{\partial^2 u}{\partial s^2} + rs\frac{\partial u}{\partial s} - ru$$

with terminal condition $\bar{u}_A(T, s) = g(s) = \max(K - s, 0)$ for a put.

### 5.2 Backward Euler Discretisation

The PDE solver in `pde_solver.py` uses:

**Finite difference stencil:**
$$\mathcal{L}_h u_i = A_i u_{i-1} - B_i u_i + C_i u_{i+1}$$

where:
- $A_i = \frac{b^2}{2\Delta s^2} + \frac{rs_i}{2\Delta s}$ (backward)
- $B_i = r + \frac{b^2}{\Delta s^2}$ (diagonal)
- $C_i = \frac{b^2}{2\Delta s^2} - \frac{rs_i}{2\Delta s}$ (forward)

**Backward Euler update:**
$$U_i^n = U_i^{n+1} + \Delta t \cdot \mathcal{L}_h U^{n+1}$$

**Early exercise enforcement:**
$$U_i^n \leftarrow \max(U_i^n, g(s_i))$$

### 5.3 File: `pde_solver.py`

```python
def solve_american_option_pde(b_squared_func, r, K, T, s_min, s_max, 
                               N_t=200, N_s=100, option_type='put'):
    """
    Solve 1D American option PDE using backward Euler.
    
    Note: b_squared_func returns b²(t,s), NOT b(t,s).
    The diffusion term is (1/2)b²∂²u/∂s².
    """
    # Grids
    dt = T / N_t
    ds = (s_max - s_min) / (N_s - 1)
    t_grid = np.linspace(0, T, N_t + 1)
    s_grid = np.linspace(s_min, s_max, N_s)
    
    # Payoff
    payoff = np.maximum(K - s_grid, 0)  # Put option
    
    # Terminal condition
    V = np.zeros((N_t + 1, N_s))
    V[-1, :] = payoff
    
    # Backward timestepping
    for n in range(N_t - 1, -1, -1):
        t = t_grid[n]
        
        # Get b² at this time
        b_sq = np.array([b_squared_func(t, s) for s in s_grid])
        
        # Finite difference coefficients
        A = (b_sq / (2*ds**2)) + (r * s_grid) / (2*ds)
        B = r + (b_sq / ds**2)
        C = (b_sq / (2*ds**2)) - (r * s_grid) / (2*ds)
        
        # Backward Euler step for interior points
        for i in range(1, N_s - 1):
            L_u = A[i] * V[n+1, i-1] - B[i] * V[n+1, i] + C[i] * V[n+1, i+1]
            V[n, i] = V[n+1, i] + dt * L_u
        
        # Boundary conditions
        V[n, 0] = payoff[0]
        V[n, -1] = payoff[-1]
        
        # Early exercise
        V[n, :] = np.maximum(V[n, :], payoff)
    
    return {'s_grid': s_grid, 't_grid': t_grid, 'values': V}
```

---

## 6. Practical Example: Pricing a 3D Basket Put

### 6.1 Parameters (Paper Equation 56)

```python
# Market parameters
r = 0.05                           # Risk-free rate
sigma = np.array([0.2, 0.15, 0.1]) # Individual volatilities

# Correlation matrix
corr_matrix = np.array([
    [1.0, 0.8, 0.3],
    [0.8, 1.0, 0.1],
    [0.3, 0.1, 1.0]
])
corr_chol = cholesky(corr_matrix, lower=True)

# Portfolio
P1 = np.array([1.0, 1.0, 1.0])     # Equal weights (sum to get basket)
x0 = np.array([100.0, 100.0, 100.0])  # Initial prices

# Option parameters
T = 0.5  # 6 months
K = 300  # At-the-money (basket starts at 300)
```

### 6.2 Expected Results

For the paper's 3D test case:
- **b² surface range:** ~700 to ~2500 (matching Paper Figure 1a)
- **Option price at K=300:** American premium ~15-20% over European
- **Exercise boundary:** Decreases from K as $t \to 0$

### 6.3 Running the Full Pipeline

```python
from laplace_volatility import compute_volatility_surface
from pde_solver import solve_american_option_pde

# Step 1: Compute volatility surface
t_grid = np.linspace(0.01, T, 15)  # Avoid t=0
s_grid = np.linspace(250, 400, 25)

b_squared_surface = compute_volatility_surface(
    t_grid, s_grid, P1, x0, r, sigma, corr_chol
)

# Step 2: Create interpolated function
from scipy.interpolate import RectBivariateSpline
interp = RectBivariateSpline(t_grid, s_grid, b_squared_surface)
b_squared_func = lambda t, s: float(interp(t, s))

# Step 3: Solve PDE
result = solve_american_option_pde(
    b_squared_func, r, K, T, 
    s_min=200, s_max=450, N_t=200, N_s=100
)

# Step 4: Extract option price at t=0, S=S0
S0 = np.dot(P1, x0)  # = 300
s_idx = np.argmin(np.abs(result['s_grid'] - S0))
option_price = result['values'][0, s_idx]

print(f"American Put Price: {option_price:.4f}")
```

---

## 7. Comparison with MLMC

### 7.1 Summary Comparison Table

| Aspect | Laplace Approximation | MLMC + Regression |
|--------|----------------------|-------------------|
| **Nature** | Deterministic (analytical) | Stochastic (Monte Carlo) |
| **Core Computation** | Optimisation + Hessian | Path simulation + regression |
| **Computational Complexity** | O(d³) per point (matrix ops) | O(1) per point (after setup) |
| **Memory** | O(d²) | O(dimV²) accumulated |
| **Dimension Scaling** | Excellent (Hessian size grows as d²) | Excellent (projection is dimension-independent) |
| **Bias** | Systematic (Taylor truncation) | Systematic (polynomial basis) |
| **Variance** | Zero (deterministic) | Controllable (increase samples) |
| **Numerical Stability** | Requires positive-definite Hessian | Requires good conditioning |
| **Near t=0 Behaviour** | Can be unstable | Naturally handles |
| **Implementation Complexity** | Moderate (optimisation, Hessians) | Higher (MLMC machinery) |
| **Parallelisation** | Per (t,s) point | Across levels and paths |



---


### Key Formulae

**Gyöngy's Lemma (the magic):**
$$b^2(t, s) = \mathbb{E}\left[\text{Basket Variance} \mid \text{Basket} = s\right]$$

**Laplace Approximation (the technique):**
$$\int e^{f(z)} dz \approx e^{f(z^*)} \cdot \sqrt{\frac{(2\pi)^{d-1}}{|\det(-H_f)|}}$$

**Final b² Formula (Paper Eq. 41):**
$$b^2 = \exp(f^* - \tilde{f}^*) \cdot \sqrt{\frac{|\det(-H_{\tilde{f}})|}{|\det(-H_f)|}}$$

### Physics Perspective

The Laplace method is the financial mathematics equivalent of:
- **Saddle-point approximation** in statistical mechanics
- **WKB approximation** in quantum mechanics
- **Steepest descent** in asymptotic analysis

We're essentially doing a **Gaussian integral around the classical solution**, with the Hessian capturing quantum/thermal fluctuations.

---

*Document created as part of KAUST internship project on American basket option pricing.*
*Reference: Bayer, Häppölä, Tempone (2017) - "Implied Stopping Rules for American Basket Options from Markovian Projection"*
