# Can We Justify the $O(m \log m)$ Formula as "Optimal" for Our Problem?

**Author:** Wadoud Charbak  
**Date:** January 2026  
**Purpose:** Clarifying pointss discussed with Erik

---

## 1. What is the Cohen-Migliorati Problem Setup?

The paper "Optimal weighted least-squares methods" by Cohen & Migliorati (2017) addresses the following problem:

> "We consider the problem of reconstructing an unknown bounded function $u$ defined on a domain $X \subset \mathbb{R}^d$ from noiseless or noisy samples of $u$ at $n$ points $(x_i)_{i=1,\ldots,n}$. We measure the reconstruction error in a norm $L^2(X, d\rho)$ for some given probability measure $d\rho$."
>
> — Page 181, Abstract

In simpler terms: given $n$ random sample points, they want to fit a polynomial approximation to an unknown function and measure how accurate it is.

The key question they address is:

> "Given a space $V_m$ and a measure $d\rho$, how to best choose the samples $y_i$ and weights $w_i$ in order to ensure that the $L^2(X, d\rho)$ error $\|u - \tilde{u}\|$ is comparable to $e_m(u)$, with $n$ being as close as possible to $m$?"
>
> — Page 184

Here $V_m$ is a polynomial space of dimension $m$, and $e_m(u)$ is the best possible approximation error in that space.

### Their Key Result

The main contribution is **Corollary 2.2** (page 187), which states:

> **Corollary 2.2.** For any $r > 0$, if $m$ and $n$ are such that the condition
> $$m \leq \kappa \frac{n}{\ln n}, \quad \text{with } \kappa := \frac{1 - \ln 2}{2 + 2r}$$
> is satisfied, then the conclusions (i), (ii), (iii) and (iv) of Theorem 2.1 hold for weighted least squares with the choice of $w$ and $d\mu$ given by (2.6) and (2.7).

Rearranging the condition $m \leq \kappa \frac{n}{\ln n}$ gives:

$$n \geq \frac{m \ln n}{\kappa} \quad \Rightarrow \quad n = O(m \log m)$$

This is the origin of the $O(m \log m)$ bound.

---

## 2. What Conditions Does Their $O(m \log m)$ Bound Require?

The bound in Corollary 2.2 is **not universal**. It requires two specific conditions, defined by equations (2.6) and (2.7) in the paper.

### Condition 1: Optimal Sampling Measure (Equation 2.7)

The samples must be drawn from a specific probability distribution called the **Christoffel measure**:

$$d\mu := \frac{k_m}{m} d\rho$$

where $k_m(x)$ is the **Christoffel function** defined as:

$$k_m(x) := \sum_{j=1}^{m} |L_j(x)|^2$$

and $L_1, \ldots, L_m$ are an orthonormal basis for the polynomial space $V_m$.

The paper explicitly states this is required for optimality:

> "We thus obtain the following result as a consequence of Theorem 2, which shows that the above choice of $w$ and $d\mu$ allows us to obtain near-optimal estimates for the truncated weighted least-squares estimator, under the minimal condition that $n$ is at least of the order $m \ln(m)$."
>
> — Page 187

### Condition 2: Weighted Least Squares with Specific Weights (Equation 2.6)

The regression must use **weighted least squares** with weights given by:

$$w := \frac{m}{k_m} = \frac{m}{\sum_{j=1}^{m} |L_j|^2}$$

This is not ordinary least squares. The weight at each sample point depends on the Christoffel function evaluated at that point.

### What Happens Without These Conditions?

The paper explicitly addresses what happens when you use **standard least squares** (uniform sampling, equal weights) instead. From Section 1.3 (page 185):

> "If $V_m = \mathcal{P}_{m-1}$ on $X = [-1,1]$ and with $d\rho$ being the uniform probability measure. In this case, one choice for the $L_j$ are the Legendre polynomials with proper normalization $\|L_j\|_{L^\infty} = |L_j(1)| = \sqrt{1+2j}$ so that $K_m = m^2$, and therefore condition (1.11) imposes that $n$ is at least of order $m^2 \ln(m)$."
>
> — Page 185

This is crucial: **for Legendre polynomials with uniform/standard sampling, the required sample count is $O(m^2 \log m)$, not $O(m \log m)$**.

---

## 3. Does Our Problem Satisfy These Conditions?

Let us compare the Cohen-Migliorati setup with our MLMC regression problem.

### Direct Comparison

| Requirement | Cohen-Migliorati (for $O(m \log m)$) | Our MLMC Implementation |
|-------------|--------------------------------------|-------------------------|
| **Sampling measure** | Christoffel function $\frac{k_m}{m} d\rho$ | Samples from GBM paths (not controlled) |
| **Regression type** | Weighted least squares with $w = \frac{m}{k_m}$ | Ordinary least squares (equal weights) |
| **Sample points** | Independently chosen from optimal measure | Determined by simulated asset paths |
| **Basis functions** | General orthonormal polynomials | Normalised Legendre polynomials |
| **Domain** | Fixed domain $X \subset \mathbb{R}^d$ | Domain estimated from path statistics |

### The Fundamental Mismatch

In our problem, the sample points are **not under our control**. They come from simulating GBM paths:

- At each timestep, we observe the basket value $S_n = \vec{P} \cdot \vec{X}(t_n)$
- These values are determined by the stochastic dynamics, not by an optimal sampling strategy
- We cannot choose to sample from the Christoffel measure even if we wanted to

Additionally, we use **ordinary least squares**, not weighted least squares with Christoffel-dependent weights.

### Answer: No, Our Problem Does Not Satisfy the Conditions

We cannot claim that $O(m \log m)$ is "optimal" for our problem in the Cohen-Migliorati sense, because:

1. We do not sample from the Christoffel measure
2. We do not use weighted least squares
3. Our sample locations are dictated by stochastic simulation, not by choice

---

## 4. What Bound Actually Applies to Us?

### Worst-Case Theory: $O(m^2)$

Based on Section 1.3 of Cohen-Migliorati, when using Legendre polynomials without optimal sampling, the worst-case requirement is:

$$n = O(m^2 \log m)$$

Our current implementation uses $M_\ell = C \cdot m^2$ (with $C = 80$), which is consistent with this worst-case bound.

### Practical Behaviour: Often Much Better

However, the worst-case bound is often pessimistic. Cohen-Migliorati's own numerical experiments (Table 6.1, page 201) show interesting behaviour:

For **standard least squares** with $n = 26,599$ and $m = 200$ (so $n/m \approx 133$, far less than $m^2 = 40,000$):

| Dimension $d$ | Probability that $\text{cond}(G) \leq 3$ |
|---------------|------------------------------------------|
| $d = 1$ | 0 |
| $d = 2$ | 0 |
| $d = 5$ | 0.54 |
| $d = 10$ | **1** |
| $d = 50$ | **1** |
| $d = 100$ | **1** |

The paper comments on this:

> "For standard least squares with the uniform measure, the empirical probability that approximates (6.1) equals zero when $d = 1$ or $d = 2$, equals 0.54 when $d = 5$, and equals one when $d = 10$, $d = 50$ or $d = 100$."
>
> — Page 201

This suggests that **in higher dimensions, standard least squares often achieves stability with far fewer samples than the worst-case $m^2$ bound suggests**.

Since our problem involves $d \geq 2$ assets (and we've tested up to $d = 50$), this observation is relevant and encouraging.

### The Role of Empirical Testing

The gap between worst-case theory ($m^2$) and practical behaviour means that **empirical testing is the right approach** for determining what scaling works for our specific problem.

This is exactly what our sample allocation experiments are designed to do.

---

## Summary

| Question | Answer |
|----------|--------|
| What is the Cohen-Migliorati $O(m \log m)$ bound? | The minimum samples needed for stable polynomial regression **under optimal conditions** |
| What conditions are required? | (1) Sampling from the Christoffel measure, (2) Weighted least squares with specific weights |
| Does our problem satisfy these conditions? | **No** — our samples come from GBM paths, and we use ordinary least squares |
| Can we cite Cohen-Migliorati to justify $O(m \log m)$ as optimal? | **No** — the optimality result does not apply to our setup |
| What can we say? | Our $m^2$ scaling is theoretically justified for non-optimal sampling; $m \log m$ is a practical heuristic to test empirically |

---

## References

Cohen, A. & Migliorati, G. (2017). "Optimal weighted least-squares methods." *SMAI Journal of Computational Mathematics*, 3:181-203.  
Available at: https://smai-jcm.centre-mersenne.org/item/10.5802/smai-jcm.24.pdf
