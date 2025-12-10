# The Coefficients $a$ and $b$ in Markovian Projection

**Document Purpose:** Explanation of the drift and volatility coefficients in the projected 1D SDE for American basket option pricing.

---

## 1. Mathematical Context: Markovian Projection

The project aims to price high-dimensional American basket options by "projecting" the complex, multi-asset dynamics onto a single, one-dimensional Stochastic Differential Equation (SDE) that mimics the basket's behaviour.

### The High-Dimensional Problem

Consider a basket of $d$ assets following geometric Brownian motion (GBM):

$$dX_i(t) = r X_i(t) \, dt + \sigma_i X_i(t) \, dW_i(t), \quad i = 1, \ldots, d$$

where:
- $X_i(t)$ is the price of asset $i$ at time $t$
- $r$ is the risk-free interest rate
- $\sigma_i$ is the volatility of asset $i$
- $W_i(t)$ are correlated Brownian motions with $dW_i \cdot dW_j = \rho_{ij} \, dt$

The basket value is defined as:

$$S(t) = P_1 \cdot \mathbf{X}(t) = \sum_{i=1}^{d} w_i X_i(t)$$

where $P_1 = (w_1, \ldots, w_d)$ are the portfolio weights.

### The Projected 1D Process

The projected process $\bar{S}(t)$ follows the SDE (Paper Eq. 11):

$$d\bar{S}(t) = \underbrace{a(t, \bar{S})}_{\text{Drift}} dt + \underbrace{b(t, \bar{S})}_{\text{Diffusion}} dW(t)$$

**These are the two coefficients in question.** The drift $a(t, S)$ and diffusion $b(t, S)$ must be chosen such that the projected process $\bar{S}(t)$ has the same marginal distributions as the true basket $S(t)$ at all times. This is guaranteed by **Gyöngy's Lemma** (1986).

---

## 2. Coefficient $a$: The Drift — Why It Is Easy

### Definition (Paper Eq. 12)

The projected drift coefficient is defined as:

$$\bar{a}(t, s) = \mathbb{E}\left[P_1 \cdot a(t, \mathbf{X}(t)) \,\big|\, P_1 \mathbf{X}(t) = s\right]$$

### Why It Is Trivial

In the standard Black-Scholes risk-neutral framework:

1. **Risk-neutral drift:** Every individual asset $X_i$ drifts at the risk-free rate $r$, i.e., $a(t, \mathbf{x}) = r\mathbf{x}$ (Paper Eq. 2).

2. **Linearity of the basket:** The basket $S(t)$ is a linear sum of assets: $S = \sum_i w_i X_i$.

3. **Linearity of expectation:** Because expectation is a linear operator, the drift of the sum equals the sum of the drifts.

Substituting into the projection formula:

$$\bar{a}(t, s) = \mathbb{E}\left[P_1 \cdot (r\mathbf{X}(t)) \,\big|\, P_1 \mathbf{X}(t) = s\right] = r \cdot \mathbb{E}\left[P_1 \mathbf{X}(t) \,\big|\, P_1 \mathbf{X}(t) = s\right] = r \cdot s$$

### Result

$$\boxed{a(t, S) = r \cdot S}$$

**No simulation, no regression, no high-dimensional integrals required.** The drift is deterministic and can be hardcoded directly into the PDE solver as the term $rS \frac{\partial V}{\partial S}$.

---

## 3. Coefficient $b$: The Projected Volatility — Why It Is Hard

### Definition (Paper Eq. 13)

The projected volatility coefficient (squared) is defined as:

$$\bar{b}^2(t, s) = \mathbb{E}\left[(P_1 \, \mathbf{b}\mathbf{b}^T P_1^T)(t, \mathbf{X}(t)) \,\big|\, P_1 \mathbf{X}(t) = s\right]$$

For the Black-Scholes model with correlated assets, this expands to:

$$b^2(t, s) = \mathbb{E}\left[\frac{1}{d^2} \sum_{i,j} \sigma_i \sigma_j \rho_{ij} X_i(t) X_j(t) \,\Big|\, \text{Basket} = s\right]$$

In plain terms:

$$b^2(t, s) = \mathbb{E}\left[\text{Instantaneous Basket Variance} \,\big|\, \text{Basket Value} = s\right]$$

### Why It Is Computationally Challenging

Unlike the drift, estimating $b(t, S)$ is the **core computational challenge** of the project:

1. **Non-Linearity:** The volatility of a sum (the basket) is *not* the sum of the volatilities. It depends on the complex correlation structure between all $d$ assets through the covariance matrix.

2. **Conditional Expectation:** One must compute the expected instantaneous variance *conditioned on* the basket taking a specific value $s$. This requires integrating over all possible asset configurations $\mathbf{X}$ that satisfy $P_1 \mathbf{X} = s$.

3. **Unknown Density:** The sum of log-normal distributions (the basket of GBM assets) does not have a closed-form probability density. One cannot simply write down an analytical formula for this conditional expectation.

4. **High-Dimensional Integration:** The conditional expectation involves integrating over a $(d-1)$-dimensional hyperplane in $\mathbb{R}^d$, which becomes intractable for large $d$.

### Note on Notation

In the paper (Eq. 11), $\bar{b}$ refers to the **diffusion coefficient** in the SDE. In the Black-Scholes PDE, the diffusion term takes the form:

$$\frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}$$

where the factor $S^2$ is characteristic of multiplicative noise in GBM. The code must ensure the solver uses $b^2 S^2$ (not just $b^2$) in the finite difference discretisation.

---

## 4. Comparison of Solutions: Paper vs Project

The paper and the project codebase take different approaches to estimating the challenging coefficient $b(t, S)$:

### The Paper's Approach: Laplace Approximation

The original paper (Bayer, Häppölä, Tempone 2017) uses the **Laplace approximation** to analytically estimate the high-dimensional integrals in Eq. 13.

**Method:**
- Assumes the transition density $\phi(\mathbf{x}; \mathbf{x}_0)$ is known (Assumption 3.1)
- Finds the extremal point of the unimodal integrand
- Applies a second-order Taylor expansion around that point
- Results in a closed-form approximation (Paper Eq. 41)

**Advantages:**
- Fast evaluation once implemented
- No Monte Carlo noise
- Works well when density is sharply peaked

**Limitations:**
- Requires explicit knowledge of the transition density
- Accuracy degrades for non-Gaussian or multimodal distributions
- Extension to other models (beyond Black-Scholes) is non-trivial

### The Project's Approach: MLMC + Polynomial Regression

The project codebase estimates $b(t, S)$ using **Multi-Level Monte Carlo (MLMC)** combined with **polynomial regression**.

**Method:**
1. Simulate thousands of paths of the high-dimensional basket process
2. At each point $(t_n, S_m)$ along each path, record the instantaneous variance
3. Fit a polynomial surface $b^2(t, S) \approx \sum_p c_p \phi_p(t, S)$ using least-squares regression
4. Use MLMC telescoping to reduce variance efficiently

**The regression target:**

$$b^2(t, S) = \sum_p c_p \, P_{i_1}(t) \, P_{i_2}(S)$$

where $P_n$ are orthonormalised Legendre polynomials.

**Advantages:**
- Model-agnostic (works for any simulable process)
- MLMC reduces computational cost from $\mathcal{O}(\epsilon^{-3})$ to $\mathcal{O}(\epsilon^{-2} \log^2 \epsilon)$
- Naturally handles complex correlation structures
- Extensible to non-GBM models

**Limitations:**
- Requires careful tuning of polynomial degree and sample sizes
- Monte Carlo noise in coefficient estimates
- Memory management for large-scale problems

---

## 5. Summary Table

| Coefficient | Represents | Equation | Difficulty | How It Is Obtained |
|:-----------:|:----------:|:--------:|:----------:|:-------------------|
| $a$ | **Drift** | $a(t, S) = r \cdot S$ | **Easy** (Trivial) | Derived analytically from linearity of expectation under risk-neutral measure |
| $b$ | **Volatility** | $b(t, S) = \sqrt{\mathbb{E}[\text{Var} \mid S = s]}$ | **Hard** (The Challenge) | Estimated via **MLMC + Regression** (Code) or **Laplace Approximation** (Paper) |

---

## 6. The Project Goal

The primary objective is to:

1. **Estimate the coefficient $b(t, S)$** using Multi-Level Monte Carlo methods with Legendre polynomial basis functions

2. **Solve the 1D American option PDE** using the estimated volatility surface:
   $$\frac{\partial V}{\partial t} + \frac{1}{2} b^2(t, S) S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$
   with the early exercise constraint $V(t, S) \geq g(S) = (K - S)^+$

3. **Compare results to the original paper** by validating:
   - The MLMC-estimated $b(t, S)$ matches the paper's Laplace approximation
   - American option prices fall within the paper's upper/lower bounds
   - Early exercise boundaries align with the paper's Figure 5(b)

4. **Produce publication-grade results** demonstrating the MLMC approach as a viable (and potentially superior) alternative to analytical approximations for high-dimensional option pricing.

---

## References

- Bayer, C., Häppölä, J., & Tempone, R. (2017). *Implied Stopping Rules for American Basket Options from Markovian Projection*. arXiv:1705.00558
- Gyöngy, I. (1986). *Mimicking the one-dimensional marginal distributions of processes having an Itô differential*. Probability Theory and Related Fields, 71, 501–516.
