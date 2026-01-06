# Sample Allocation Strategies in MLMC for Polynomial Regression

**Author:** Wadoud Charbak  
**Project:** MLMC + Markovian Projection for American Basket Options  
**Date:** January 2026

---

## 1. The Big Picture

In our MLMC framework for volatility surface estimation, we need to choose how many Monte Carlo samples $M_\ell$ to use at each level $\ell$ of the telescoping sum. This choice affects computational cost, statistical accuracy, and numerical stability of the regression.

This document walks through the theory, explains our current implementation and its theoretical basis, and explores alternative allocation strategies that could improve efficiency.

> **Notation:** Throughout this document and in the literature, $N_\ell$ and $M_\ell$ are used interchangeably to denote the number of samples at level $\ell$. Giles typically uses $N_\ell$, whilst the polynomial regression literature often uses $M_\ell$. They refer to the same quantity.

---

## 2. Theoretical Foundations

### 2.1 The Telescoping Sum

The fundamental identity underpinning MLMC is:

$$\mathbb{E}[P_L] = \mathbb{E}[P_0] + \sum_{\ell=1}^{L} \mathbb{E}[P_\ell - P_{\ell-1}]$$

This is equation (114) in Amélie's notes. The key insight is that instead of computing $\mathbb{E}[P_L]$ directly (which requires the finest, most expensive resolution), we decompose it into a **sum of corrections**.

Think of it like computing the height of a building: rather than measuring from ground to roof in one go, you measure floor by floor. Each floor measurement is cheaper and the errors in measuring floor heights are smaller than errors in measuring the total height.

### 2.2 The Unbiased Estimator

The estimator becomes (equation 115 in Amélie's notes):

$$Y = \frac{1}{M_0} \sum_{n=1}^{M_0} P_0^{(0,n)} + \sum_{\ell=1}^{L} \frac{1}{M_\ell} \sum_{n=1}^{M_\ell} \left( P_\ell^{(\ell,n)} - P_{\ell-1}^{(\ell,n)} \right)$$

The notation $(\ell, n)$ means "the $n$-th sample at level $\ell$". Crucially, the fine path $P_\ell^{(\ell,n)}$ and coarse path $P_{\ell-1}^{(\ell,n)}$ are **coupled** — they use the same underlying Brownian increments. This coupling is what makes their difference have small variance.

Where $M_\ell$ is the number of samples at level $\ell$.

### 2.3 What is $M_\ell$ and Why Do We Choose It?

Here's where things get interesting. There are **two different contexts** for choosing $M_\ell$:

**Context 1: Standard MLMC (estimating expectations)**

Giles' optimal allocation minimises total cost for a fixed variance target:

$$M_\ell^{\text{opt}} = \frac{2}{\varepsilon^2} \sqrt{\frac{V_\ell}{C_\ell}} \sum_{j=0}^{L} \sqrt{V_j C_j}$$

where $V_\ell = \text{Var}[Y_\ell]$ and $C_\ell$ is cost per sample. This balances variance reduction against computational expense.

**Context 2: MLMC + Polynomial Regression (our framework)**

We're not just estimating an expectation — we're fitting **polynomial regression coefficients** at each level. This introduces an additional constraint: we need enough samples to reliably solve the least squares problem.

### 2.4 Why Does Polynomial Degree Decrease with Level?

In our code, the polynomial degree at level $\ell$ is:

$$d_\ell = d_{\max} - \ell$$

So with `max_deg = 3`:
- **Level 0:** Degree 3 → 10 basis functions (captures main shape)
- **Level 1:** Degree 2 → 6 basis functions (medium corrections)
- **Level 2:** Degree 1 → 3 basis functions (small corrections)
- **Level 3:** Degree 0 → 1 basis function (constant correction)

Why? Because coarse levels need to capture **global features** of the volatility surface (requiring high-degree polynomials), whilst fine levels only represent **small corrections** (requiring simple, low-degree polynomials).

This is the multi-resolution principle: don't use a sledgehammer to crack a nut. At fine levels, the corrections are tiny and smooth — using high-degree polynomials would waste resources and risk overfitting.

### 2.5 Variance and Cost Structure

For the MLMC estimator:

$$\text{Var}[\hat{Y}] = \sum_{\ell=0}^{L} \frac{V_\ell}{M_\ell}, \qquad \text{Cost}[\hat{Y}] = \sum_{\ell=0}^{L} M_\ell C_\ell$$

For Euler-Maruyama discretisation of SDEs with Lipschitz payoffs:
- $V_\ell = O(h_\ell) = O(2^{-\ell})$ — variance decays exponentially
- $C_\ell = O(h_\ell^{-1}) = O(2^\ell)$ — cost increases exponentially

### 2.6 Giles' MLMC Theorem

The key complexity result from Giles (2008, 2015):

**Theorem (Giles).** Under standard assumptions with weak convergence rate $\alpha$, variance decay rate $\beta$, and cost growth rate $\gamma$, MLMC achieves MSE $\leq \varepsilon^2$ with cost:

$$\text{Cost} \leq \begin{cases}
O(\varepsilon^{-2}) & \text{if } \beta > \gamma \\
O(\varepsilon^{-2} (\log \varepsilon)^2) & \text{if } \beta = \gamma \\
O(\varepsilon^{-2-(\gamma-\beta)/\alpha}) & \text{if } \beta < \gamma
\end{cases}$$

For standard SDEs with $\beta = \gamma = 1$, this gives $O(\varepsilon^{-2}(\log\varepsilon)^2)$, compared to $O(\varepsilon^{-3})$ for standard Monte Carlo — a massive improvement.

---

## 3. Our Current Implementation

### 3.1 The Formula

Our implementation uses:

$$M_\ell = \max\left(C, \, C \cdot \dim(V_{L-\ell})^2 \right)$$

where $C = 80$ is a base constant and $\dim(V_{L-\ell})$ is the number of polynomial basis functions at level $\ell$.

From `Comparison_Between_Methods/methods/mlmc_ot_estimator.py`, line 859:

```python
# Multi-resolution: polynomial degree decreases with level
l_V = max_deg - level
pairs = tot_degree_poly(l_V)
dimV = len(pairs)

# Sample size scales with basis dimension squared
M_l = max(C, int(C * dimV ** 2))
```

### 3.2 Concrete Numbers

For `max_deg = 3` with $C = 80$:

| Level $\ell$ | Degree | $\dim(V)$ | $M_\ell = 80 \times \dim^2$ | Timesteps |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 3 | 10 | 8,000 | 4 (cheap) |
| 1 | 2 | 6 | 2,880 | 8 |
| 2 | 1 | 3 | 720 | 16 |
| 3 | 0 | 1 | 80 | 32 (expensive) |

**Total:** 11,680 samples across all levels.

The beauty of this design: most samples are at the **cheap** coarse levels, very few at the **expensive** fine levels.

### 3.3 Theoretical Basis: Cohen-Migliorati Framework

Our $\dim(V)^2$ scaling is **not arbitrary** — it's adapted from the theoretical framework developed by Cohen, Migliorati, and collaborators for weighted least squares polynomial approximation.

**The key result** from Cohen & Migliorati (2017) and related work (Chkifa et al., 2015; Migliorati et al., 2014):

> For weighted least squares approximation in a polynomial space $V_m$ with $\dim(V_m) = m$, using an optimal sampling measure, stability and quasi-optimal accuracy are achieved when $N = O(m \log m)$.

However, this optimal bound requires:
1. **Sampling from the induced/Christoffel measure** (not uniform or arcsine)
2. **Weighted least squares** with specific weights depending on the sampling measure

For simpler sampling strategies (which we use), the required sample count is higher. The bound becomes:

$$N = C^d \cdot m \log m$$

where $C^d$ is dimension-dependent (empirically $\approx 4$ even for $d = 10$).

### 3.4 Why We Use $m^2$ Instead of $m \log m$

Our implementation is **more conservative** than the theoretical optimum:

| Approach | Formula | Samples for $m = 10$ |
|----------|---------|---------------------|
| Theoretical optimal | $C \cdot m \log m$ | $\approx 230$ |
| Our implementation | $C \cdot m^2$ | $8,000$ |

We use $m^2$ scaling because:

1. **We use arcsine/uniform sampling**, not optimal Christoffel sampling
2. **Safety margin:** Better to over-sample than have an unstable regression
3. **Robustness:** The $m^2$ scaling works reliably across different parameter regimes

The key diagnostic is the **condition number** of the Gram matrix $\mathbf{D}^T\mathbf{D}$. As long as it stays below $\sim 10^4$, the regression is stable.

### 3.5 Why $C = 80$?

I cannot find any theoretical derivation of this specific value; it has been found to work well across a range of test cases, so I believe this is why Amelie originally chose it. It was chosen to:

1. Ensure sufficient overdetermination of the regression system
2. Keep condition numbers manageable
3. Balance accuracy against computational cost

I could experiment with values between 40–150 to find the minimum that maintains stability for our typical problem parameters. Lower values would speed up computation if stability is maintained.

---

## 4. Alternative Sample Allocation Formulas

These are **drop-in replacements** — different ways to compute $M_\ell$ that would work directly in our existing code structure without changing the overall methodology.

### 4.1 Near-Optimal Polynomial Scaling

**Source:** Cohen & Migliorati (2017), Section 11.3 of Amélie's notes

**Formula:**
$$M_\ell = C \cdot \dim(V_\ell) \cdot \log(\dim(V_\ell) + 1)$$

**Implementation:**
```python
# Near-optimal scaling (Cohen-Migliorati bound)
M_l = max(C, int(C * dimV * np.log(dimV + 1)))
```

**Potential savings:** For $m = 10$, this reduces from 8,000 to roughly 2,000 samples — a 4× reduction.

| Pros | Cons |
|------|------|
| Theoretically justified | May require better sampling strategy |
| Significant computational savings | Could be less stable near domain boundaries |
| Matches literature bounds | Need to verify condition numbers remain OK |

**Verdict:** Worth testing. If condition numbers stay acceptable, this offers immediate speedup.

### 4.2 Adaptive Allocation

**Source:** Standard practice in MLMC implementations (Giles, 2015)

**Idea:** Rather than fixing $M_\ell$ upfront, start small and increase until the regression is stable.

**Algorithm:**
```python
def compute_M_l_adaptive(dimV, C_base=80, kappa_target=1e4):
    M_l = C_base * dimV  # Start with linear scaling
    
    while True:
        D, psi = build_regression_system(M_l, ...)
        kappa = np.linalg.cond(D.T @ D)
        
        if kappa < kappa_target:
            return M_l
        
        M_l = int(1.5 * M_l)  # Increase by 50%
```

| Pros | Cons |
|------|------|
| No a priori knowledge needed | Iterative (slower startup) |
| Adapts to problem difficulty | Condition number isn't the only stability indicator |
| Never over-samples unnecessarily | More complex implementation |

**Verdict:** Good for robustness when problem characteristics vary. Adds some overhead but guarantees stability.

### 4.3 Hybrid Variance-Regression Allocation

**Source:** Combining Giles (2015) with Cohen-Migliorati (2017)

**Idea:** Ensure both regression stability AND variance optimality by taking the maximum:

$$M_\ell = \max\left( C \cdot \dim(V_\ell) \log(\dim(V_\ell)), \quad \frac{2}{\varepsilon^2}\sqrt{\frac{V_\ell}{C_\ell}}\sum_j\sqrt{V_j C_j} \right)$$

The first term handles regression stability; the second handles variance optimality.

| Pros | Cons |
|------|------|
| Theoretically justified on both fronts | Requires variance estimation from pilot samples |
| Never under-samples | More complex formula |
| Optimal in both senses | May over-sample at some levels |

**Verdict:** The "proper" solution if we want theoretical guarantees for both constraints.

### 4.4 Christoffel Function Sampling

**Source:** Narayan et al. (2017), Cohen & Migliorati (2017)

**Idea:** Achieve the true $O(m \log m)$ bound by sampling from the optimal measure.

The **Christoffel function** for polynomial space $V_m$ is:

$$\kappa_m(y) = \sum_{j=1}^{m} |B_j(y)|^2$$

Sampling proportional to $\kappa_m(y)$ and using appropriate weights achieves optimal sample complexity.

| Pros | Cons |
|------|------|
| Achieves theoretical optimum | Non-trivial to implement |
| Well-established theory | Requires computing Christoffel function |
| Proven stability | Adds complexity to sampling step |

**Verdict:** More significant implementation effort, but could be a research contribution if we demonstrate it works for our problem.

---

## 5. Fundamentally Different Methodologies

These approaches would require **significant changes** to our existing framework — not just a different formula for $M_\ell$, but alterations to the overall MLMC structure.

### 5.1 Weighted MLMC

**Source:** arXiv:2405.03453 (2024)

**Key change:** Instead of the standard telescoping sum, use weighted combinations:

$$\hat{\mu} = \sum_{\ell=0}^{L} w_\ell \cdot \frac{1}{M_\ell}\sum_{n} P_\ell^{(n)}$$

where weights $w_\ell$ are optimised to minimise variance subject to unbiasedness.

**When useful:** When coarse level approximations are **poorly correlated** with fine levels. Standard MLMC assumes good correlation from coupling; if that fails (e.g., if optimal transport coupling underperforms), weighted MLMC can compensate.

**Impact on our code:** Would require:
- Restructuring the telescoping sum computation
- Adding weight optimisation step
- Changing how we combine level contributions

**Verdict:** Only worth pursuing if we observe poor correlation at coarse levels with current approach.

### 5.2 Multi-Fidelity Monte Carlo (MFMC)

**Source:** Peherstorfer et al. (2016, 2018)

**Key change:** Uses models of different **fidelity** (not just different discretisations) as control variates:

$$\hat{\mu}^{\text{MFMC}} = \hat{\mu}_0 + \sum_{\ell=1}^{L} \eta_\ell (\hat{\mu}_\ell^* - \hat{\mu}_\ell)$$

where $\eta_\ell$ are optimal control variate weights computed from model correlations.

**When useful:** When cheap surrogate models are available (e.g., reduced-dimension approximations, simplified physics).

**Impact on our code:** Would require:
- Multiple model implementations at different fidelities
- Correlation estimation between models
- Different sample allocation strategy

**Verdict:** Not directly applicable unless we develop surrogate models. Could be interesting future direction.

### 5.3 MLBLUE (Multilevel Best Linear Unbiased Estimator)

**Source:** Schaden et al. (2020), Croci et al. (2023)

**Key change:** A unified framework that includes MLMC, MFMC, and control variates as special cases. Solves:

$$\min_{\mathbf{w}} \text{Var}\left[\sum_\ell w_\ell \hat{\mu}_\ell\right] \quad \text{subject to unbiasedness constraints}$$

**When useful:** When you want provably optimal estimators within the class of linear unbiased estimators.

**Impact on our code:** Would require:
- Full covariance matrix estimation between levels
- Solving optimisation problem for weights
- Significant restructuring of estimation framework

**Verdict:** Theoretically elegant but probably overkill for our current needs. Worth knowing about for future reference.

### 5.4 Multi-Index Monte Carlo (MIMC)

**Source:** Haji-Ali et al. (2016), mentioned in Amélie's notes Section 11.1.5

**Key change:** Instead of a single index $\ell$ for refinement level, use a **multi-index** $\boldsymbol{\ell} = (\ell_1, \ell_2, \ldots)$ to refine in multiple directions (e.g., time discretisation AND spatial discretisation).

**When useful:** High-dimensional problems where refinement in multiple directions independently affects accuracy.

**Impact on our code:** Would require:
- Multi-index bookkeeping
- More complex level structure
- Different complexity analysis

**Verdict:** Interesting for future extensions but not immediately applicable to current problem.

---

## 6. Summary and Recommendations

### 6.1 Drop-In Replacements (Easy to Test)

| Strategy | Formula | Potential Savings | Implementation Effort |
|----------|---------|-------------------|----------------------|
| **Current** | $C \cdot m^2$ | Baseline | — |
| **Near-optimal** | $C \cdot m \log m$ | ~4× fewer samples | Trivial (one line) |
| **Adaptive** | Start small, increase if unstable | Variable | Moderate |
| **Hybrid** | $\max(\text{regression}, \text{variance})$ | Optimal balance | Moderate |

### 6.2 Fundamental Changes (Significant Effort)

| Methodology | When Useful | Implementation Effort |
|-------------|-------------|----------------------|
| Weighted MLMC | Poor level correlation | High |
| MFMC | Surrogate models available | High |
| MLBLUE | Maximum theoretical efficiency | Very high |
| MIMC | Multi-directional refinement | Very high |

### 6.3 My Recommendations

**Immediate (this week):**
1. Test the $m \log m$ scaling — just change one line and monitor condition numbers
2. If stable, we get ~4× speedup for free

**Short-term (next few weeks):**
1. Implement condition number monitoring as a diagnostic
2. Try adaptive allocation to see if we can reduce samples further
3. Tune the constant $C$ — experiment with 40, 60, 100 to find minimum stable value

**Future directions (potential research contributions):**
1. Implement Christoffel function sampling for true optimal complexity
2. If OT coupling underperforms, investigate weighted MLMC
3. Document any improvements for the paper

---

## References

1. Giles, M.B. (2008). "Multilevel Monte Carlo path simulation." *Operations Research*, 56(3):607-617.

2. Giles, M.B. (2015). "Multilevel Monte Carlo methods." *Acta Numerica*, 24:259-328.

3. Cohen, A. & Migliorati, G. (2017). "Optimal weighted least-squares methods." *SMAI Journal of Computational Mathematics*, 3:181-203.

4. Chkifa, A., Cohen, A., Migliorati, G., Nobile, F., & Tempone, R. (2015). "Discrete least squares polynomial approximation with random evaluations." *ESAIM: M2AN*, 49(3):815-837.

5. Migliorati, G., Nobile, F., von Schwerin, E., & Tempone, R. (2014). "Analysis of discrete $L^2$ projection on polynomial spaces with random evaluations." *Foundations of Computational Mathematics*, 14:419-456.

6. Narayan, A., Jakeman, J.D., & Zhou, T. (2017). "A Christoffel function weighted least squares algorithm for collocation approximations." *Mathematics of Computation*, 86(306):1913-1947.

7. Peherstorfer, B., Willcox, K., & Gunzburger, M. (2018). "Survey of multifidelity methods in uncertainty propagation, inference, and optimization." *SIAM Review*, 60(3):550-591.

8. Schaden, D. & Ullmann, E. (2020). "On multilevel best linear unbiased estimators." *SIAM/ASA Journal on Uncertainty Quantification*, 8(2):601-635.

9. Haji-Ali, A.-L., Nobile, F., & Tempone, R. (2016). "Multi-index Monte Carlo: when sparsity meets sampling." *Numerische Mathematik*, 132(4):767-806.
