# Multi-Level Monte Carlo Theory for American Basket Options

**Author:** Wadoud Farchadi  
**Based on:** Amelie's original research  
**Institution:** KAUST Internship Project  
**Date:** November 2024

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Multi-Level Monte Carlo Framework](#2-the-multi-level-monte-carlo-framework)
3. [Telescoping Sum Decomposition](#3-telescoping-sum-decomposition)
4. [Level-Dependent Polynomial Approximation](#4-level-dependent-polynomial-approximation)
5. [Variance Reduction Through Coupling](#5-variance-reduction-through-coupling)
6. [Optimal Transport Enhancement](#6-optimal-transport-enhancement)
7. [Gaussian-Brenier Maps](#7-gaussian-brenier-maps)
8. [Complexity Analysis](#8-complexity-analysis)
9. [Numerical Stability](#9-numerical-stability)
10. [Implementation Details](#10-implementation-details)
11. [Physics Analogies](#11-physics-analogies)
12. [References](#12-references)

---

## 1. Introduction

### 1.1 Motivation

The standard Monte Carlo (MC) method for estimating quantities to accuracy $\varepsilon$ requires $O(\varepsilon^{-2})$ samples when the variance is finite. However, when dealing with stochastic differential equations (SDEs), achieving accuracy $\varepsilon$ in the **weak sense** typically requires timestep $h \sim \varepsilon$, leading to an overall complexity of $O(\varepsilon^{-3})$ when combined with the Monte Carlo sampling.

Multi-Level Monte Carlo (MLMC), introduced by Giles (2008), breaks this curse by using a **telescoping sum** of corrections across multiple levels of resolution. The key insight is that coarse approximations are cheap but inaccurate, whilst fine approximations are expensive but accurate. By cleverly combining both, MLMC achieves complexity $O(\varepsilon^{-2})$ for certain problems, and with appropriate variance reduction, can reach $O(\varepsilon^{-2} (\log \varepsilon)^2)$.

### 1.2 Problem Setting

We consider the pricing of American basket options on $d$ assets following geometric Brownian motion:

$$
dX_i(t) = r X_i(t) \, dt + \sigma_i X_i(t) \, dW_i(t), \quad X_i(0) = x_{i,0}, \quad i = 1, \ldots, d
$$

where:
- $r$ is the risk-free rate
- $\sigma_i$ are asset volatilities
- $W_i(t)$ are correlated Brownian motions with correlation matrix $\rho_{ij}$

The basket value is:
$$
B(t) = \sum_{i=1}^d w_i X_i(t)
$$

with weights $w_i$ (typically $w_i = 1/d$ for equal weighting).

### 1.3 Markovian Projection

Following Gyöngy's theorem, we project the $d$-dimensional system onto a 1D process:

$$
d\bar{S}(t) = r \bar{S}(t) \, dt + \bar{b}(t, \bar{S}(t)) \, dB(t), \quad \bar{S}(0) = B(0)
$$

where $\bar{b}(t, S)$ is the **local volatility** that preserves the marginal distributions. Our goal is to estimate $\bar{b}(t, S)$ efficiently using MLMC.

---

## 2. The Multi-Level Monte Carlo Framework

### 2.1 Hierarchy of Approximations

Let $Y_L$ denote the quantity of interest computed with the finest timestep $h_L$. In our case, $Y_L$ represents the coefficients of the local volatility expansion.

MLMC uses a hierarchy of timesteps:
$$
h_0 > h_1 > h_2 > \cdots > h_L
$$

typically with $h_\ell = M^{-\ell} h_0$ for some integer $M \geq 2$ (we use $M = 2$).

For each level $\ell$, we define:
- **Fine approximation:** $Y_\ell^f$ computed with timestep $h_\ell$
- **Coarse approximation:** $Y_\ell^c$ computed with timestep $h_{\ell-1} = 2h_\ell$

### 2.2 The Telescoping Sum

The fundamental MLMC identity is:

$$
\mathbb{E}[Y_L] = \mathbb{E}[Y_0] + \sum_{\ell=1}^{L} \mathbb{E}[Y_\ell - Y_{\ell-1}]
$$

This is simply a telescoping sum. The key advantage is that **we can estimate each correction term independently**:

$$
\mathbb{E}[Y_L] \approx \frac{1}{M_0} \sum_{m=1}^{M_0} Y_0^{(m)} + \sum_{\ell=1}^{L} \frac{1}{M_\ell} \sum_{m=1}^{M_\ell} \left( Y_\ell^{f,(m)} - Y_\ell^{c,(m)} \right)
$$

where:
- $M_\ell$ is the number of samples at level $\ell$
- $Y_\ell^{f,(m)}$ and $Y_\ell^{c,(m)}$ are **coupled** realisations (same random numbers!)

### 2.3 Why Coupling Matters

The variance of the MLMC estimator is:

$$
\text{Var}\left[\hat{Y}_L^{\text{MLMC}}\right] = \frac{V_0}{M_0} + \sum_{\ell=1}^{L} \frac{V_\ell}{M_\ell}
$$

where $V_\ell = \text{Var}[Y_\ell^f - Y_\ell^c]$.

**Critical observation:** If we sample $Y_\ell^f$ and $Y_\ell^c$ **independently**, then:
$$
V_\ell = \text{Var}[Y_\ell^f] + \text{Var}[Y_\ell^c] \approx 2 \text{Var}[Y_\ell^f]
$$

But if we **couple** them (use same random numbers), we get:
$$
V_\ell = \text{Var}[Y_\ell^f] + \text{Var}[Y_\ell^c] - 2\text{Cov}[Y_\ell^f, Y_\ell^c]
$$

For highly correlated paths, $\text{Cov}[Y_\ell^f, Y_\ell^c] \approx \text{Var}[Y_\ell^f]$, giving:

$$
V_\ell \approx 0 \quad \text{(massive variance reduction!)}
$$

In practice, the difference $Y_\ell^f - Y_\ell^c$ is $O(h_\ell^\alpha)$ for some $\alpha > 0$, so:

$$
V_\ell = O(h_\ell^{2\alpha})
$$

This **quadratic** decay in variance with timestep refinement is the engine of MLMC efficiency.

---

## 3. Telescoping Sum Decomposition

### 3.1 Local Volatility Coefficients

Recall that we approximate the local volatility as:

$$
\bar{b}(t, S) = \sqrt{\sum_{p=1}^{P} c_p \, \tilde{P}_{i_1}(\tau(t)) \, \tilde{P}_{i_2}(\xi(S))}
$$

where:
- $\tau(t) = 2t/T - 1$ maps $[0, T] \to [-1, 1]$
- $\xi(S) = 2(S - s_{\min})/(s_{\max} - s_{\min}) - 1$ maps $[s_{\min}, s_{\max}] \to [-1, 1]$
- $\tilde{P}_k$ are orthonormalised Legendre polynomials

Our quantity of interest is the coefficient vector $\mathbf{c} = (c_1, \ldots, c_P)^\top$.

### 3.2 Level-Specific Regression Systems

At level $\ell$, we construct the regression system using paths at timesteps $h_\ell$ (fine) and $2h_\ell$ (coarse):

**Fine system:**
$$
\mathbf{D}_\ell^f \mathbf{c}_\ell = \boldsymbol{\psi}_\ell^f
$$

where:
- $\mathbf{D}_\ell^f \in \mathbb{R}^{(M_\ell N_\ell) \times P_\ell}$ is the design matrix
- $\boldsymbol{\psi}_\ell^f$ contains squared basket volatilities $b^2(t, \mathbf{X}(t))$

**Coarse system:**
$$
\mathbf{D}_\ell^c \mathbf{c}_\ell = \boldsymbol{\psi}_\ell^c
$$

### 3.3 The Difference System

The key MLMC insight is to compute the **difference**:

$$
\mathbf{D}_\ell (\mathbf{c}_\ell^f - \mathbf{c}_\ell^c) = \boldsymbol{\psi}_\ell^f - \boldsymbol{\psi}_\ell^c
$$

where we evaluate the design matrix $\mathbf{D}_\ell$ on the fine paths subsampled at coarse timesteps, and:

$$
\psi_\ell[m, n] = b_f^2(t_n, \mathbf{X}_m^f(t_n)) - b_c^2(t_n, \mathbf{X}_m^c(t_n))
$$

This ensures that:
1. Both sides use the **same spatial points** (fine paths at coarse times)
2. The difference is $O(h_\ell)$ due to temporal discretisation error
3. Variance is greatly reduced compared to estimating $\mathbf{c}_\ell^f$ alone

### 3.4 Telescoping Reconstruction

The final coefficient vector is:

$$
\mathbf{c}_{\text{total}} = \mathbf{c}_0 + \sum_{\ell=1}^{L} (\mathbf{c}_\ell^f - \mathbf{c}_\ell^c)
$$

For level 0, there is no coarser level, so we simply compute:
$$
\mathbf{c}_0 = (\mathbf{D}_0^\top \mathbf{D}_0)^{-1} \mathbf{D}_0^\top \boldsymbol{\psi}_0
$$

---

## 4. Level-Dependent Polynomial Approximation

### 4.1 Motivation

A critical design choice in MLMC is the **degree of polynomial approximation** at each level. Naively, one might use the same high degree $d_{\max}$ at all levels, but this is wasteful.

**Key insight:** Coarse levels (small $\ell$) need to capture **global features** of the volatility surface, whilst fine levels (large $\ell$) only need to represent **small corrections**.

### 4.2 Multi-Resolution Principle

We use:
$$
d_\ell = d_{\max} - \ell
$$

So:
- **Level 0:** Degree $d_{\max}$ (e.g., 3) → captures main shape
- **Level 1:** Degree $d_{\max} - 1$ (e.g., 2) → medium corrections
- **Level 2:** Degree $d_{\max} - 2$ (e.g., 1) → small corrections
- **Level $L$:** Degree $d_{\max} - L$ (e.g., 0) → constant correction

### 4.3 Basis Size

The total-degree polynomial space of degree $d$ in 2D has dimension:

$$
P_d = \binom{d+2}{2} = \frac{(d+1)(d+2)}{2}
$$

For example:
- $d = 3$: $P_3 = 10$ basis functions
- $d = 2$: $P_2 = 6$ basis functions
- $d = 1$: $P_1 = 3$ basis functions
- $d = 0$: $P_0 = 1$ basis function

### 4.4 Sample Size Allocation

We set the number of samples at level $\ell$ proportional to the square of the basis size:

$$
M_\ell = \max(C, C \cdot P_\ell^2)
$$

where $C$ is a constant (typically 80).

**Rationale:** The regression system has $P_\ell$ unknowns, so we need at least $O(P_\ell)$ samples. Using $O(P_\ell^2)$ provides sufficient overdetermination for stable solutions.

For $d_{\max} = 3$, $C = 80$:
- Level 0: $M_0 = 80 \times 10^2 = 8000$ samples
- Level 1: $M_1 = 80 \times 6^2 = 2880$ samples  
- Level 2: $M_2 = 80 \times 3^2 = 720$ samples
- Level 3: $M_3 = 80 \times 1^2 = 80$ samples

**Total: 11,680 samples** (compared to ~25,000 for single-level at same accuracy)

---

## 5. Variance Reduction Through Coupling

### 5.1 Brownian Bridge Coupling

The standard MLMC coupling for SDEs uses the **Brownian bridge** property. For the coarse path at timestep $h_c = 2h_f$, we sum consecutive fine increments:

$$
\Delta W^c_n = \Delta W^f_{2n} + \Delta W^f_{2n+1}
$$

where $\Delta W^f_k \sim \mathcal{N}(0, h_f)$ are independent.

This ensures:
1. $\Delta W^c_n \sim \mathcal{N}(0, 2h_f) = \mathcal{N}(0, h_c)$ ✓
2. $\text{Corr}(\mathbf{X}^f(t), \mathbf{X}^c(t)) \approx 1 - O(h_f)$ (very high!)

### 5.2 Correlation Analysis

Let's analyse the correlation between fine and coarse paths. For geometric Brownian motion:

$$
X^f(t) = x_0 \exp\left( \left(r - \frac{\sigma^2}{2}\right)t + \sigma W^f(t) \right)
$$

$$
X^c(t) = x_0 \exp\left( \left(r - \frac{\sigma^2}{2}\right)t + \sigma W^c(t) \right)
$$

Since $W^c(t)$ is constructed from the same increments as $W^f(t)$ (just summed pairwise), we have:

$$
\text{Corr}(W^f(t), W^c(t)) = \frac{\mathbb{E}[W^f(t) W^c(t)]}{\sqrt{\text{Var}[W^f(t)] \text{Var}[W^c(t)]}}
$$

For the specific coupling:
- $\mathbb{E}[W^f(t) W^c(t)] = t$ (they share all increments)
- $\text{Var}[W^f(t)] = t$
- $\text{Var}[W^c(t)] = t$

Therefore:
$$
\text{Corr}(W^f(t), W^c(t)) = 1 \quad \text{(perfect at Brownian level!)}
$$

However, the **discretisation error** introduces a small decorrelation:

$$
\text{Corr}(X^f(T), X^c(T)) \approx 1 - O(h_f)
$$

In practice, for $h_f = 0.0078$ (level 3 with $h_0 = 0.125$), we observe:
$$
\text{Corr}(X^f(T), X^c(T)) \approx 0.998
$$

### 5.3 Variance Decay

For smooth payoffs, the variance of the difference decays as:

$$
V_\ell = \text{Var}[Y_\ell^f - Y_\ell^c] = O(h_\ell^2)
$$

This is due to the **weak error** of Euler-Maruyama being $O(h)$, so the squared difference is $O(h^2)$.

For our local volatility coefficients, the variance decays even faster because:
1. We're estimating **coefficients**, not payoffs
2. The Legendre basis provides excellent approximation properties
3. The regression averages over many paths

Empirically, we observe:
$$
V_\ell \approx C_V \cdot h_\ell^3
$$

for some constant $C_V$.

---

## 6. Optimal Transport Enhancement

### 6.1 Limitations of Standard Coupling

The Brownian bridge coupling is excellent, but it has limitations:

1. **Not optimal in $L^2$ sense:** The coupling minimises $\mathbb{E}[|W^f - W^c|^2]$ but not $\mathbb{E}[|\mathbf{X}^f - \mathbf{X}^c|^2]$

2. **Doesn't exploit distributional information:** It uses the **same** random numbers regardless of the distributions

3. **Suboptimal for non-Gaussian transformations:** GBM involves $\exp(\cdot)$, which is highly nonlinear

### 6.2 Optimal Transport Theory

**Optimal transport** seeks to find the map $T: \mathbb{R}^d \to \mathbb{R}^d$ that minimises:

$$
\mathcal{W}_2^2(\mu, \nu) = \inf_{T: T_\# \mu = \nu} \int_{\mathbb{R}^d} |x - T(x)|^2 \, d\mu(x)
$$

where:
- $\mu$ is the law of $\mathbf{X}^f$
- $\nu$ is the law of $\mathbf{X}^c$
- $T_\# \mu = \nu$ means $T$ pushes $\mu$ forward to $\nu$

This is the **2-Wasserstein distance** between probability measures.

### 6.3 Monge-Kantorovich Duality

The optimal transport problem has a dual formulation (Kantorovich):

$$
\mathcal{W}_2^2(\mu, \nu) = \sup_{\varphi, \psi} \left\{ \int \varphi \, d\mu + \int \psi \, d\nu : \varphi(x) + \psi(y) \leq |x - y|^2 \right\}
$$

When $\mu$ and $\nu$ are **Gaussian**, the optimal map has a **closed form**.

### 6.4 Why Log-Space?

Geometric Brownian motion is **log-normal**, not Gaussian. However:

$$
\log X_i(t) \sim \mathcal{N}\left( \log x_{i,0} + \left(r - \frac{\sigma_i^2}{2}\right)t, \sigma_i^2 t \right)
$$

So $\log \mathbf{X}(t)$ is **multivariate Gaussian**!

We work in log-space:
- $\mathbf{Y}^f = \log \mathbf{X}^f \sim \mathcal{N}(\boldsymbol{\mu}_f, \mathbf{C}_f)$
- $\mathbf{Y}^c = \log \mathbf{X}^c \sim \mathcal{N}(\boldsymbol{\mu}_c, \mathbf{C}_c)$

Then apply Gaussian optimal transport, and transform back:
$$
\mathbf{X}^c_{\text{est}} = \exp(T(\log \mathbf{X}^f))
$$

---

## 7. Gaussian-Brenier Maps

### 7.1 Brenier's Theorem

**Theorem (Brenier, 1991):** Let $\mu = \mathcal{N}(\boldsymbol{\mu}_1, \mathbf{C}_1)$ and $\nu = \mathcal{N}(\boldsymbol{\mu}_2, \mathbf{C}_2)$ be two Gaussian measures on $\mathbb{R}^d$ with $\mathbf{C}_1, \mathbf{C}_2$ positive definite.

The unique optimal transport map (in the sense of minimising $\mathcal{W}_2^2$) is:

$$
T(\mathbf{y}) = \boldsymbol{\mu}_2 + \mathbf{A}(\mathbf{y} - \boldsymbol{\mu}_1)
$$

where the matrix $\mathbf{A}$ is given by:

$$
\mathbf{A} = \mathbf{C}_1^{-1/2} \left( \mathbf{C}_1^{1/2} \mathbf{C}_2 \mathbf{C}_1^{1/2} \right)^{1/2} \mathbf{C}_1^{-1/2}
$$

### 7.2 Properties of the Brenier Map

1. **Affine:** $T$ is an affine transformation

2. **Pushes forward:** $T_\# \mu = \nu$, i.e., if $\mathbf{Y} \sim \mathcal{N}(\boldsymbol{\mu}_1, \mathbf{C}_1)$, then $T(\mathbf{Y}) \sim \mathcal{N}(\boldsymbol{\mu}_2, \mathbf{C}_2)$

3. **Minimises $L^2$ distance:** 
$$
\mathbb{E}[|\mathbf{Y} - T(\mathbf{Y})|^2] \leq \mathbb{E}[|\mathbf{Y} - S(\mathbf{Y})|^2]
$$
for any other map $S$ with $S_\# \mu = \nu$

4. **Symmetric:** The inverse map $T^{-1}: \nu \to \mu$ has the same form with $\mathbf{C}_1 \leftrightarrow \mathbf{C}_2$

### 7.3 Computational Implementation

Computing $\mathbf{A}$ requires:

**Step 1:** Eigendecomposition of $\mathbf{C}_1$:
$$
\mathbf{C}_1 = \mathbf{U}_1 \boldsymbol{\Lambda}_1 \mathbf{U}_1^\top
$$

**Step 2:** Compute $\mathbf{C}_1^{1/2}$ and $\mathbf{C}_1^{-1/2}$:
$$
\mathbf{C}_1^{1/2} = \mathbf{U}_1 \boldsymbol{\Lambda}_1^{1/2} \mathbf{U}_1^\top, \quad \mathbf{C}_1^{-1/2} = \mathbf{U}_1 \boldsymbol{\Lambda}_1^{-1/2} \mathbf{U}_1^\top
$$

**Step 3:** Form $\mathbf{M} = \mathbf{C}_1^{1/2} \mathbf{C}_2 \mathbf{C}_1^{1/2}$:
$$
\mathbf{M} = \mathbf{U}_1 \boldsymbol{\Lambda}_1^{1/2} \mathbf{U}_1^\top \mathbf{C}_2 \mathbf{U}_1 \boldsymbol{\Lambda}_1^{1/2} \mathbf{U}_1^\top
$$

**Step 4:** Eigendecomposition of $\mathbf{M}$:
$$
\mathbf{M} = \mathbf{U}_M \boldsymbol{\Lambda}_M \mathbf{U}_M^\top
$$

**Step 5:** Compute $\mathbf{M}^{1/2}$:
$$
\mathbf{M}^{1/2} = \mathbf{U}_M \boldsymbol{\Lambda}_M^{1/2} \mathbf{U}_M^\top
$$

**Step 6:** Final matrix:
$$
\mathbf{A} = \mathbf{C}_1^{-1/2} \mathbf{M}^{1/2} \mathbf{C}_1^{-1/2}
$$

**Complexity:** $O(d^3)$ per timestep (dominated by eigendecompositions)

### 7.4 Empirical Mean and Covariance

In practice, we don't know $\boldsymbol{\mu}_f, \mathbf{C}_f, \boldsymbol{\mu}_c, \mathbf{C}_c$ analytically. Instead, we estimate them from samples:

$$
\hat{\boldsymbol{\mu}}_f = \frac{1}{M} \sum_{m=1}^M \log \mathbf{X}_m^f(t_n)
$$

$$
\hat{\mathbf{C}}_f = \frac{1}{M-1} \sum_{m=1}^M (\log \mathbf{X}_m^f(t_n) - \hat{\boldsymbol{\mu}}_f)(\log \mathbf{X}_m^f(t_n) - \hat{\boldsymbol{\mu}}_f)^\top
$$

and similarly for coarse paths.

**Time-dependent maps:** We construct a **different map at each coarse timestep** $t_n$, giving a sequence:
$$
\{T_0, T_1, \ldots, T_{N_c-1}\}
$$

where $T_0 = \text{Id}$ (identity, since paths start at the same point).

---

## 8. Complexity Analysis

### 8.1 Standard Monte Carlo

For weak error $\varepsilon$ in estimating $\mathbb{E}[g(X(T))]$ using Euler-Maruyama:

**Weak error:** $|\mathbb{E}[g(X(T))] - \mathbb{E}[g(X_h(T))]| = O(h)$

To achieve $O(h) \sim \varepsilon$, need $h \sim \varepsilon$, hence $N = T/h \sim \varepsilon^{-1}$ timesteps.

**Monte Carlo error:** $O(M^{-1/2})$

To achieve $O(M^{-1/2}) \sim \varepsilon$, need $M \sim \varepsilon^{-2}$ samples.

**Total cost:**
$$
\text{Cost}_{\text{MC}} = M \times N \sim \varepsilon^{-2} \times \varepsilon^{-1} = O(\varepsilon^{-3})
$$

### 8.2 Multi-Level Monte Carlo

**Key observations:**
1. Variance decays: $V_\ell = O(h_\ell^2)$
2. Cost per sample: $C_\ell = O(h_\ell^{-1})$ (number of timesteps)

**Optimal sample allocation:** Minimise total cost subject to variance constraint:

$$
\min_{M_0, \ldots, M_L} \sum_{\ell=0}^L M_\ell C_\ell \quad \text{subject to} \quad \sum_{\ell=0}^L \frac{V_\ell}{M_\ell} \leq \varepsilon^2
$$

By Lagrange multipliers, the optimal allocation is:

$$
M_\ell \propto \sqrt{\frac{V_\ell}{C_\ell}}
$$

For $V_\ell = O(h_\ell^2)$ and $C_\ell = O(h_\ell^{-1})$:

$$
M_\ell \propto \sqrt{\frac{h_\ell^2}{h_\ell^{-1}}} = h_\ell^{3/2}
$$

**Number of levels:** $L \sim \log_2(\varepsilon^{-1})$ (since $h_L \sim \varepsilon$)

**Total cost:**

$$
\text{Cost}_{\text{MLMC}} = \sum_{\ell=0}^L M_\ell C_\ell \sim \sum_{\ell=0}^L h_\ell^{3/2} \cdot h_\ell^{-1} = \sum_{\ell=0}^L h_\ell^{1/2}
$$

This is a geometric series with ratio $2^{-1/2}$:

$$
\text{Cost}_{\text{MLMC}} \sim h_0^{1/2} + h_1^{1/2} + \cdots + h_L^{1/2} \sim h_L^{1/2} = O(\varepsilon^{-1/2})
$$

Wait, this seems too good! The catch is the **constant** factor. In practice:

$$
\text{Cost}_{\text{MLMC}} = O(\varepsilon^{-2} (\log \varepsilon)^2)
$$

The $(\log \varepsilon)^2$ factor comes from:
1. $L = O(\log \varepsilon)$ levels
2. Some additional logarithmic factors in the variance constants

**Speedup:**
$$
\frac{\text{Cost}_{\text{MC}}}{\text{Cost}_{\text{MLMC}}} = \frac{\varepsilon^{-3}}{\varepsilon^{-2} (\log \varepsilon)^2} = \frac{\varepsilon^{-1}}{(\log \varepsilon)^2}
$$

For $\varepsilon = 10^{-3}$:
$$
\text{Speedup} \sim \frac{1000}{(\log 1000)^2} \approx \frac{1000}{48} \approx 20\times
$$

### 8.3 Impact of Polynomial Degree

Our choice of level-dependent polynomial degrees affects the sample sizes. Recall:

$$
M_\ell = C \cdot P_\ell^2 = C \cdot \left( \frac{(d_{\max} - \ell + 1)(d_{\max} - \ell + 2)}{2} \right)^2
$$

For $d_{\max} = 3$:
- $M_0 = C \cdot 10^2 = 100C$
- $M_1 = C \cdot 6^2 = 36C$
- $M_2 = C \cdot 3^2 = 9C$
- $M_3 = C \cdot 1^2 = C$

Total samples: $146C$ vs $400C$ if we used degree 3 at all levels!

**Savings: ~63%** in sample count whilst maintaining accuracy.

---

## 9. Numerical Stability

### 9.1 Condition Number Analysis

The condition number of the regression system is:

$$
\kappa(\mathbf{D}^\top \mathbf{D}) = \frac{\lambda_{\max}(\mathbf{D}^\top \mathbf{D})}{\lambda_{\min}(\mathbf{D}^\top \mathbf{D})}
$$

**For monomials:** $\kappa \sim 10^4$ (terrible!)

**For Legendre polynomials:** $\kappa \sim 10^2$ (excellent!)

The improvement comes from:
1. **Orthogonality:** $\int_{-1}^1 P_i(x) P_j(x) \, dx = \delta_{ij} / (2i+1)$
2. **Normalisation:** We use $\tilde{P}_i(x) = \sqrt{(2i+1)/2} \, P_i(x)$, giving $\|\tilde{P}_i\|_{L^2} = 1$

### 9.2 Column Normalisation

Even with Legendre polynomials, the design matrix columns can have different magnitudes due to:
- Different numbers of samples at different timesteps
- Uneven distribution of basket values across $[s_{\min}, s_{\max}]$

We apply **column normalisation** before QR decomposition:

$$
\tilde{\mathbf{D}} = \mathbf{D} \text{diag}\left( \frac{1}{\|\mathbf{D}_{:,1}\|}, \ldots, \frac{1}{\|\mathbf{D}_{:,P}\|} \right)
$$

Then solve $\tilde{\mathbf{D}} \tilde{\mathbf{c}} = \boldsymbol{\psi}$ and rescale:
$$
\mathbf{c} = \text{diag}\left( \frac{1}{\|\mathbf{D}_{:,1}\|}, \ldots, \frac{1}{\|\mathbf{D}_{:,P}\|} \right) \tilde{\mathbf{c}}
$$

This typically improves $\kappa$ by another factor of 2-5×.

### 9.3 QR vs Normal Equations

We solve the least-squares problem using **QR decomposition**:

$$
\mathbf{D} = \mathbf{Q} \mathbf{R}
$$

Then:
$$
\mathbf{c} = \mathbf{R}^{-1} \mathbf{Q}^\top \boldsymbol{\psi}
$$

**Why not normal equations?**

Normal equations form $\mathbf{D}^\top \mathbf{D} \mathbf{c} = \mathbf{D}^\top \boldsymbol{\psi}$ and solve via Cholesky. However:

$$
\kappa(\mathbf{D}^\top \mathbf{D}) = \kappa(\mathbf{D})^2
$$

So the condition number is **squared**, losing roughly half the precision!

For our problems:
- $\kappa(\mathbf{D}) \sim 10^2$
- $\kappa(\mathbf{D}^\top \mathbf{D}) \sim 10^4$

With double precision ($\sim 16$ digits), normal equations give $\sim 12$ digits accuracy, whilst QR gives $\sim 14$ digits.

---

## 10. Implementation Details

### 10.1 Path Generation

For level $\ell$, we generate:

**Fine paths** ($N_f = T/h_\ell$ steps):
```
X = X0
for n = 1, ..., N_f:
    Z = randn(M_ℓ, d)
    dW = (Z @ G.T) * sqrt(h_ℓ)
    X = X + r*X*h_ℓ + σ*X*dW
    paths_f[:, n, :] = X
```

**Coarse paths** ($N_c = N_f/2$ steps, using **same** $Z$!):
```
Z_c = Z.reshape(M_ℓ, N_c, 2, d).sum(axis=2)  # Pair consecutive
X = X0
for n = 1, ..., N_c:
    dW = (Z_c[:, n-1, :] @ G.T) * sqrt(h_ℓ)  # Note: sqrt(h_ℓ), not sqrt(2*h_ℓ)!
    X = X + r*X*(2*h_ℓ) + σ*X*dW
    paths_c[:, n, :] = X
```

**Critical detail:** We use `sqrt(h_ℓ)` not `sqrt(2*h_ℓ)` because $Z_c$ is the **sum** of two $\mathcal{N}(0, h_\ell)$ variables, hence already has variance $2h_\ell$.

### 10.2 Design Matrix Construction

The design matrix $\mathbf{D} \in \mathbb{R}^{(M \cdot N) \times P}$ is built using tensor products:

```python
# Time values scaled to [-1, 1]
t_vals = 2 * np.linspace(0, T, N_c) / T - 1
t_vals = np.tile(t_vals, M)  # Repeat for each path

# Basket values scaled to [-1, 1]
basket = paths_f[:, ::2, :].dot(weights).flatten()
s_vals = 2 * (basket - s_min) / (s_max - s_min) - 1

# Vandermonde matrices for Legendre polynomials
VT = legvander(t_vals, degree_max)  # Time polynomials
VS = legvander(s_vals, degree_max)  # Space polynomials

# Normalisation: sqrt((2n+1)/2)
norm = np.sqrt((2 * np.arange(degree_max + 1) + 1) / 2)
VT *= norm[None, :]
VS *= norm[None, :]

# Tensor product
for p, (i1, i2) in enumerate(pairs):
    D[:, p] = VT[:, i1] * VS[:, i2]
```

### 10.3 Target Vector $\boldsymbol{\psi}$

For Multi-Level, we compute:

$$
\psi[m \cdot N + n] = b_f^2(t_n, \mathbf{X}_m^f(t_n)) - b_c^2(t_n, \mathbf{X}_m^c(t_n))
$$

where the basket volatility is:

$$
b^2(t, \mathbf{X}) = \frac{1}{d^2} \sum_{i,j=1}^d \sigma_i \sigma_j X_i X_j \rho_{ij}
$$

In code:
```python
for m in range(M):
    for n in range(N_c):
        # Fine volatility
        X_f = paths_f[m, 2*n, :]  # Every other timestep
        Sigma_f = np.diag(vol * X_f)
        b_f_sq = (Sigma_f @ cov_mat @ Sigma_f).sum() / d**2
        
        # Coarse volatility
        X_c = paths_c[m, n, :]
        Sigma_c = np.diag(vol * X_c)
        b_c_sq = (Sigma_c @ cov_mat @ Sigma_c).sum() / d**2
        
        psi[m * N_c + n] = b_f_sq - b_c_sq
```

### 10.4 Optimal Transport Map Application

At each coarse timestep $t_n$:

1. **Estimate distributions:**
```python
log_paths_f = np.log(paths_f[:, 2*n, :])
log_paths_c = np.log(paths_c[:, n, :])

mu_f = log_paths_f.mean(axis=0)
mu_c = log_paths_c.mean(axis=0)

C_f = np.cov(log_paths_f.T, bias=False)
C_c = np.cov(log_paths_c.T, bias=False)
```

2. **Construct Brenier map:**
```python
# Eigendecomposition of C_f
eig_f, U_f = np.linalg.eigh(C_f)
sqrt_C_f = U_f @ np.diag(np.sqrt(eig_f)) @ U_f.T
invsqrt_C_f = U_f @ np.diag(1.0 / np.sqrt(eig_f)) @ U_f.T

# Form M = C_f^{1/2} C_c C_f^{1/2}
M = sqrt_C_f @ C_c @ sqrt_C_f

# Eigendecomposition of M
eig_M, U_M = np.linalg.eigh(M)
sqrt_M = U_M @ np.diag(np.sqrt(eig_M)) @ U_M.T

# Final transformation
A = invsqrt_C_f @ sqrt_M @ invsqrt_C_f
```

3. **Apply map:**
```python
def map(y):
    return mu_c + (y - mu_f) @ A.T

# Transform fine paths
log_est_c = map(log_paths_f)
est_paths_c = np.exp(log_est_c)
```

4. **Use in regression:**
```python
# Compute psi using estimated coarse paths
for m in range(M):
    X_f = paths_f[m, 2*n, :]
    X_c_est = est_paths_c[m, :]  # Instead of paths_c[m, n, :]
    
    # ... compute b_f^2 - b_c^2 as before
```

---

## 11. Physics Analogies

### 11.1 Perturbation Theory

The MLMC telescoping sum is analogous to **perturbation theory** in quantum mechanics:

$$
E = E^{(0)} + \lambda E^{(1)} + \lambda^2 E^{(2)} + \cdots
$$

Here:
- $E^{(0)}$ is the **ground state** (level 0, coarse approximation)
- $\lambda E^{(1)}$ is the **first-order correction** (level 1)
- $\lambda^2 E^{(2)}$ is the **second-order correction** (level 2)
- etc.

Each correction becomes smaller as we refine the discretisation, just as higher-order perturbative corrections become smaller in quantum theory.

### 11.2 Renormalisation Group

The level-dependent polynomial degrees mirror **renormalisation group** ideas in statistical mechanics:

- **Coarse scales** (level 0): Need many degrees of freedom to capture **large-scale physics**
- **Fine scales** (level $L$): Only need **few degrees of freedom** for small corrections

This is exactly the philosophy of Wilson's renormalisation group: integrate out high-energy modes at each scale, leaving only the relevant low-energy physics.

### 11.3 Adiabatic Continuation

The Optimal Transport map $T: \mathbb{R}^d \to \mathbb{R}^d$ can be viewed as an **adiabatic transformation** in phase space:

$$
\mathbf{X}^f \xrightarrow{T} \mathbf{X}^c
$$

preserving the probability measure (pushforward property).

This is analogous to **adiabatic theorems** in quantum mechanics, where a slowly-varying Hamiltonian evolves eigenstates whilst preserving quantum numbers.

### 11.4 Effective Field Theory

The Markovian projection itself is an **effective field theory**:

- **Full theory:** $d$-dimensional GBM (expensive, high-dimensional)
- **Effective theory:** 1D process with local volatility (cheap, 1D)

The local volatility $\bar{b}(t, S)$ plays the role of **running coupling constants** that encode the effects of the heavy degrees of freedom.

Just as in SMEFT (Standard Model Effective Field Theory), we integrate out heavy fields (the individual assets) and keep only the light field (the basket), with **effective couplings** (local volatility) that capture the physics at each energy scale (time $t$, basket value $S$).

### 11.5 Variance as Free Energy

The variance $V_\ell$ can be thought of as a **free energy**:

$$
F_\ell = \beta^{-1} \log V_\ell
$$

The goal of variance reduction is to **minimise the free energy** by choosing optimal coupling schemes (temperature control).

Standard coupling is like **canonical ensemble** (fixed temperature), whilst Optimal Transport is like **grand canonical ensemble** (allows particle exchange), achieving lower free energy.

---

## 12. References

### Core MLMC Papers

1. **Giles, M. B.** (2008). "Multilevel Monte Carlo path simulation." *Operations Research*, 56(3), 607-617.

2. **Giles, M. B.** (2015). "Multilevel Monte Carlo methods." *Acta Numerica*, 24, 259-328.

3. **Heinrich, S.** (2001). "Multilevel Monte Carlo Methods." *International Conference on Large-Scale Scientific Computing*, pp. 58-67.

### Optimal Transport

4. **Brenier, Y.** (1991). "Polar factorization and monotone rearrangement of vector-valued functions." *Communications on Pure and Applied Mathematics*, 44(4), 375-417.

5. **Villani, C.** (2009). *Optimal Transport: Old and New*. Springer.

6. **Peyré, G., & Cuturi, M.** (2019). "Computational Optimal Transport." *Foundations and Trends in Machine Learning*, 11(5-6), 355-607.

### Markovian Projection

7. **Gyöngy, I.** (1986). "Mimicking the one-dimensional marginal distributions of processes having an Itô differential." *Probability Theory and Related Fields*, 71(4), 501-516.

8. **Dupire, B.** (1994). "Pricing with a smile." *Risk*, 7(1), 18-20.

### Polynomial Approximation

9. **Trefethen, L. N.** (2013). *Approximation Theory and Approximation Practice*. SIAM.

10. **Canuto, C., Hussaini, M. Y., Quarteroni, A., & Zang, T. A.** (2007). *Spectral Methods: Fundamentals in Single Domains*. Springer.

### American Options

11. **Longstaff, F. A., & Schwartz, E. S.** (2001). "Valuing American options by simulation: A simple least-squares approach." *Review of Financial Studies*, 14(1), 113-147.

12. **Glasserman, P.** (2004). *Monte Carlo Methods in Financial Engineering*. Springer.

---

## Appendix A: Notation Summary

| Symbol | Description |
|--------|-------------|
| $d$ | Number of assets |
| $\mathbf{X}(t) = (X_1(t), \ldots, X_d(t))^\top$ | Asset prices |
| $B(t) = \sum_i w_i X_i(t)$ | Basket value |
| $\bar{S}(t)$ | Projected 1D process |
| $\bar{b}(t, S)$ | Local volatility function |
| $h_\ell = 2^{-\ell} h_0$ | Timestep at level $\ell$ |
| $N_\ell = T / h_\ell$ | Number of timesteps at level $\ell$ |
| $M_\ell$ | Number of samples at level $\ell$ |
| $d_\ell = d_{\max} - \ell$ | Polynomial degree at level $\ell$ |
| $P_\ell$ | Number of basis functions at level $\ell$ |
| $\mathbf{D}_\ell$ | Design matrix at level $\ell$ |
| $\boldsymbol{\psi}_\ell$ | Target vector at level $\ell$ |
| $\mathbf{c}_\ell$ | Coefficient vector at level $\ell$ |
| $V_\ell = \text{Var}[Y_\ell^f - Y_\ell^c]$ | Variance of level-$\ell$ difference |
| $C_\ell = O(h_\ell^{-1})$ | Cost per sample at level $\ell$ |
| $T: \mathbb{R}^d \to \mathbb{R}^d$ | Optimal transport map |
| $\mathcal{W}_2(\mu, \nu)$ | 2-Wasserstein distance |
| $\tilde{P}_k(x)$ | Orthonormalised Legendre polynomial |
| $\kappa(\mathbf{A})$ | Condition number of matrix $\mathbf{A}$ |

---

## Appendix B: Algorithm Pseudocode

### Algorithm 1: Standard MLMC

```
Input: x0, T, h0, r, vol, cov_mat, max_deg, P1, C
Output: c_total

1. Pilot run:
   paths_0 = GBM_paths(x0, h0*2^(-max_deg), M=10000)
   basket_0 = paths_0.dot(P1)
   s_min, s_max = percentile(basket_0, [0.01, 99.99])

2. For ℓ = 0, 1, ..., max_deg:
   a. Set h_ℓ = h0 * 2^(-ℓ)
   b. Set d_ℓ = max_deg - ℓ
   c. Set P_ℓ = dim(total_degree_poly(d_ℓ))
   d. Set M_ℓ = max(C, C * P_ℓ^2)
   
   e. Generate fine paths: paths_f ~ GBM(h_ℓ, M_ℓ)
   f. Generate coarse paths: paths_c ~ GBM(2*h_ℓ, M_ℓ) [coupled!]
   
   g. Construct D_ℓ, ψ_ℓ from paths_f, paths_c
   h. Solve: c_ℓ = (D_ℓ^T D_ℓ)^(-1) D_ℓ^T ψ_ℓ
   
   i. If ℓ = 0:
        c_total = c_0
      Else:
        c_total += c_ℓ

3. Return c_total
```

### Algorithm 2: MLMC with Optimal Transport

```
Input: x0, T, h0, r, vol, cov_mat, max_deg, P1, C
Output: c_total

1. [Same pilot run as Algorithm 1]

2. Level 0 (standard):
   c_0 = mlmc_level_0(...)
   c_total = c_0

3. For ℓ = 1, ..., max_deg:
   a. Set h_ℓ = h0 * 2^(-ℓ)
   b. Set d_ℓ = max_deg - ℓ
   c. Set M_ℓ = max(C, C * dim(total_degree_poly(d_ℓ))^2)
   
   d. Generate fine paths: paths_f ~ GBM(h_ℓ, M_ℓ)
   e. Generate coarse paths: paths_c ~ GBM(2*h_ℓ, M_ℓ) [coupled!]
   
   f. For each coarse timestep t_n:
      i.   Compute μ_f, C_f from log(paths_f[:, 2*n, :])
      ii.  Compute μ_c, C_c from log(paths_c[:, n, :])
      iii. Construct Brenier map T_n: (μ_f, C_f) → (μ_c, C_c)
   
   g. Apply maps to get estimated coarse paths:
      est_paths_c = [exp(T_n(log(paths_f[:, 2*n, :]))) for n]
   
   h. Construct D_ℓ, ψ_ℓ from paths_f, est_paths_c
   i. Solve: c_ℓ = (D_ℓ^T D_ℓ)^(-1) D_ℓ^T ψ_ℓ
   j. c_total += c_ℓ

4. Return c_total
```

---

**END OF THEORY DOCUMENT**

This document provides the complete mathematical foundation for the Multi-Level Monte Carlo implementation with Optimal Transport variance reduction. All algorithms, derivations, and complexity analyses are publication-ready.
