# Error Quantification and Stability in Laplace Approximation Methods
## Analysis of Bayer, Häppölä, and Tempone (2017)

**Author:** Wadoud Charbak (KAUST Internship)  
**Date:** January 2025  
**Reference:** *Implied Stopping Rules for American Basket Options from Markovian Projection*, arXiv:1705.00558

---

## 1. Introduction

To support my comparison framework between MLMC and Laplace approximation methods for computing the projected volatility surface $\bar{b}^2(t, s)$, I went back to the original Bayer et al. (2017) paper to understand how they quantified errors and addressed numerical stability. This document summarises my findings.

The short answer: they didn't really formalise either.

---

## 2. The Laplace Approximation: A Quick Recap

The core computation involves evaluating a ratio of high-dimensional integrals (Paper Equation 39):

$$\bar{b}^2(t, s) = \frac{\mathbb{E}\left[(P_1 b b^\top P_1^\top)(t, \mathbf{X}(t)) \mid P_1 \cdot \mathbf{X}(t) = s\right]}{p_{P_1 \mathbf{X}(t)}(s)}$$

The Laplace method approximates integrals of the form $\int e^{f(\mathbf{z})} \, d\mathbf{z}$ by expanding around the mode $\mathbf{z}^*$:

$$\int e^{f(\mathbf{z})} \, d\mathbf{z} \approx e^{f(\mathbf{z}^*)} \cdot \sqrt{\frac{(2\pi)^{d-1}}{|\det(-H_f)|}}$$

where $H_f$ is the Hessian matrix evaluated at the mode.

The final formula for the volatility surface becomes (Paper Equation 41):

$$\bar{b}^2(t, s) = \exp(f^* - \tilde{f}^*) \cdot \sqrt{\frac{|\det(-H_{\tilde{f}})|}{|\det(-H_f)|}}$$

This involves finding two saddle points (for numerator and denominator), computing two Hessians, and taking their ratio. Plenty of places for things to go wrong, as I discovered.

---

## 3. How the Paper Quantifies Errors

### 3.1 The Four Error Sources (Section 3.4)

The paper explicitly identifies four sources of error in their pricing methodology:

| Error Source | Controllable? | Convergence Rate |
|--------------|---------------|------------------|
| Statistical error (finite $M$ paths) | ✅ Yes | $O(M^{-1/2})$ via CLT |
| Forward Euler bias | ✅ Yes | $O(N_t^{-1})$ |
| Backward solver discretisation | ✅ Yes | $O(N_t^{-1})$ |
| **Laplace approximation error** | ❌ No | Unknown |

The key quote from Section 3.4 is quite telling:

> "For the Laplace error $|A^\pm_{\infty,\infty,\infty,b^{(x_0)}} - A^\pm_{\infty,\infty,\infty,\tilde{b}_1}|$, there is no simple and practical way to control the error."

In other words, the fundamental error introduced by replacing exact conditional expectations with saddle-point approximations **cannot be rigorously bounded**. They essentially acknowledge this is a limitation and move on.

### 3.2 The Upper/Lower Bound Approach (or the "Sandwich" Approach if you prefer)

Instead of directly quantifying the Laplace error, the paper uses Rogers' duality framework to bracket the true option price:

- **Lower bound $A^-$**: Obtained from the implied stopping rule $\tau^\dagger$
- **Upper bound $A^+$**: Obtained from a near-optimal martingale $R^*$

The gap $|A^+ - A^-|$ gives an indication of approximation quality, but crucially, this gap measures the **overall methodology quality**, not the Laplace approximation error specifically.
They essentially say: "We can't prove the projection is good. But if we calculate the Lower Bound and the Upper Bound, and they happen to be close to each other, then we know we did a good job for that specific case."

### 3.3 What They Claim

From Section 3.5.2:

> "relative numerical accuracy in the approximation of around one percent"

And from the Abstract:

> "a few percent"

Looking at their Figure 4(b), the actual upper/lower bound gaps range from roughly 1% to 10% depending on moneyness. These are reasonable for practical purposes, but they're **empirical observations**, not theoretical guarantees.

Looking at Figure 3(b), they actually have an error plot where they say and I quote
> "The estimate for the uncertainty is achieved as a combination of the upper and lower bounds presented in 3(a), together with an estimate of the statistical error and bias for both."

without elaborating anywhere else how this was calculated. 

### 3.4 What They Didn't Do

The paper contains no:

1. Systematic convergence study for the Laplace approximation error as dimension increases
2. Rigorous theoretical error bounds (beyond the controllable discretisation errors)
3. Sensitivity analysis of parameter regimes where Laplace might fail
4. Error decomposition showing how much of the total error comes from the saddle-point approximation vs other sources

This isn't a criticism of the paper. They were focused on demonstrating a practical pricing method, not developing a complete error theory. But it does mean that if Bayer, Häppölä, and Tempone couldn't properly quantify this error after years of work on the method, there's absolutely no way I can do it in a quick 3 month internship, as much as I would love to.

---

## 4. How the Paper Addresses Stability

### 4.1 Explicit Mentions: Almost None

The word "stability" appears exactly twice in the paper, and both instances are in passing remarks rather than systematic analysis.

**Remark 2.2** mentions:

> "We extend artificially the domain... For $(\bar{b}^{(x_0)})^2$ we also set a lower bound to guarantee numerical stability and well-posedness."

This is a subtle admission that they needed to **clamp the volatility** to prevent negative or extremely small values. No details are given about:
- What parameters cause problems
- How the lower bound was chosen
- What happens when the "natural" volatility would be negative

**Remark 3.2** hints at density issues:

> "The projected volatility can only be reliably evaluated inside the area where the density for $P_1\mathbf{X}(t)$ is not negligible. At the most extreme case, at the initial time, the density of $P_1\mathbf{X}(0)$ focuses on a single point."

I believe this acknowledges that near $t = 0$ or in sparse density regions, the method becomes unreliable. But again, no quantitative stability analysis is provided.

### 4.2 The Newton Iteration (Equation 42)

The paper describes their mode-finding procedure:

$$\mathbf{z}^{(n+1)} = \mathbf{z}^{(n)} - (H_f(\mathbf{z}^{(n)}))^{-1} \nabla f(\mathbf{z}^{(n)})$$

They claim this is "very robust" and converges in "a few dozens of iterations". However, there's no discussion of:

- What happens when the Hessian is singular or nearly singular
- Parameter regimes where convergence fails
- Saddle points that might not exist for extreme parameter combinations
- Initial guess sensitivity

### 4.3 Complete Omissions

The paper provides no analysis of:

1. **Hessian conditioning**: What happens when $\det(-H)$ approaches zero or becomes negative?
2. **Saddle point existence**: Are there parameter regimes where the saddle point simply doesn't exist?
3. **Boundary effects**: How does the method behave near $t = 0$ where the distribution concentrates on a single point?
4. **Parameter sensitivity**: Which combinations of $\sigma$, $\rho$, $T$, $d$ cause numerical difficulties?

From my research across the internet thhese are other things that are considered for similar problems (unless there are any specific reasons why they are not discussed here?)

### 4.3 What I believe caused this collapse

The Laplace approximation relies on "Small Noise Expansion" or "Saddle Point Approximation." It makes a massive assumption: The integral is dominated entirely by a single sharp peak (the mode).

This works perfectly when:
1. Volatility is Low: The peak is sharp.
2. Maturity is Short: The diffusion hasn't spread out much.

In the case of the extreme cases of the `stress_test_exploration.ipynb`, The High Volatility / Long Maturity I used would cause the probability density function to become "flat" (diffuse). There is no longer a sharp peak. The Laplace assumption, that we can approximate the curve with a quadratic Gaussian at the peak, completely breaks down. The Hessian (curvature) becomes small, and the approximation divides by this small number, causing the numerical values to explode or collapse to zero/NaN.

---

## 5. What I Found in My Comparison Experiments

### 5.1 The Stress Test Results

In my `stress_test_exploration.ipynb`, I systematically pushed both methods to their limits. The results were quite illuminating.

**Moderate Case** ($d=10$, $\sigma=0.3$, $\rho=0.75$, $T=1.0$):
- Correlation: 0.8594
- $L^2$ disagreement: 0.3249
- Both surfaces look reasonable

**High-Vol Case** ($d=10$, $\sigma=0.4$, $\rho=0.75$, $T=1.0$):
- Correlation: 0.2513
- $L^2$ disagreement: 1.2487
- Laplace surface shows spikes and instabilities

**Aggressive Case** ($d=15$, $\sigma=0.4$, $\rho=0.85$, $T=1.5$):
- Correlation: 0.5258
- $L^2$ disagreement: 1.0144
- **Laplace surface completely fragmented with massive holes**
- MLMC surface remains smooth throughout

The visual evidence from the surface plots is striking. In the aggressive case, the Laplace surface looks like Swiss cheese while MLMC produces a perfectly smooth, continuous surface. This is a genuine robustness advantage, that can be a potentially very important thing to write about.

### 5.2 Why MLMC is More Robust

MLMC's superior stability in extreme regimes comes from its fundamentally different computational approach:

| Aspect | Laplace | MLMC |
|--------|---------|------|
| Nature | Analytical approximation | Statistical estimation |
| Failure mode | Hessian inversion fails | Polynomial regression always produces output |
| Saddle point requirement | Must exist and be unique | Not required |
| Density requirement | Must be non-negligible | Handles any sample distribution |

The polynomial regression at the heart of MLMC will always produce *some* output, even if the fit is suboptimal. Laplace can return NaN entirely when the Hessian isn't positive definite or when Newton iteration fails to converge.

### 5.3 Failure Modes in My Laplace Implementation

Looking at my `laplace_volatility.py`, the error handling reveals the fragility:

```python
try:
    det_H_star = np.linalg.det(-H_star)
    det_H_tilde = np.linalg.det(-H_tilde)
    
    if det_H_star <= 0 or det_H_tilde <= 0:
        warnings.warn(f"Non-positive definite Hessian at t={t}, s={s}")
        return np.nan
    
    log_b_squared = (f_star - f_tilde) + 0.5 * (np.log(det_H_tilde) - np.log(det_H_star))
    b_squared = np.exp(log_b_squared)
    
except np.linalg.LinAlgError:
    warnings.warn(f"Singular Hessian at t={t}, s={s}")
    return np.nan
```

When these conditions trigger, we get NaN values scattered across the volatility surface, which is exactly what we see in the stress test plots.

### 5.4 Parameter Boundaries I've Identified

From my systematic testing, rough stability boundaries for Laplace are:

| Parameter | Stable Region | Problematic Region |
|-----------|---------------|-------------------|
| Volatility $\sigma$ | $\sigma < 0.35$ | $\sigma > 0.40$ |
| Correlation $\rho$ | $\rho < 0.80$ | $\rho > 0.85$ (combined with high $\sigma$) |
| Maturity $T$ | $T < 1.0$ | $T > 1.5$ (with other extreme params) |
| Dimension $d$ | $d \leq 10$ | $d \geq 15$ (stress regime) |

These are empirical observations from my experiments, not theoretical guarantees. But they provide practical guidance that the paper never offers.

---

## 6. Implications for My Comparison Framework

### 6.1 Reframing "Error" as "Disagreement"

Given that neither method is ground truth, I've adopted the terminology of "method agreement" rather than "error". The metrics I compute are the following:

- **$L^2$ disagreement**: $\sqrt{\frac{1}{N}\sum_i (\bar{b}^2_{\text{MLMC},i} - \bar{b}^2_{\text{Laplace},i})^2} / \text{mean}(\bar{b}^2)$
- **Correlation**: Pearson correlation between the two surfaces
- **Valid grid fraction**: Proportion of grid points where both methods produce finite values

These measure how much the methods agree, not how close either is to the "true" answer.

### 6.2 What I Can Reasonably Conclude

From my analysis:

1. **Under benign parameters**, MLMC and Laplace agree to within 20-30% $L^2$ disagreement with correlation > 0.85. This is acceptable given the fundamentally different computational approaches.

2. **Under stress parameters**, Laplace can fail catastrophically while MLMC remains stable. This is a genuine practical advantage.

3. **The ~20% systematic disagreement** between methods in the baseline case is structural, arising from:
   - MLMC's polynomial regression compresses dynamic range
   - Laplace computes a different (analytical) approximation to the same underlying quantity
   - Neither is "correct". They're different approximations with different bias characteristics.

4. **Trying to quantify "Laplace error" is futile** without a ground truth. The paper acknowledges this, and my experiments confirm it.

### 6.3 Computational Trade-offs

| Aspect | MLMC | Laplace |
|--------|------|---------|
| Speed | 0.5-2s | 14-42s |
| Stability | Excellent | Poor under stress |
| Accuracy | Unknown (but consistent) | Unknown (but fragile) |
| Extensibility | Any simulatable model | Requires analytical density |

MLMC is roughly 20-30x faster and more robust, at the cost of some loss of dynamic range in the volatility surface. For practical pricing, this seems like a good trade-off.

---

## 7. Conclusions

### 7.1 What the Paper Does Well

- Introduces a clever analytical technique for dimension reduction
- Demonstrates practical pricing accuracy on well-behaved test cases
- Provides upper/lower bounds for validation

### 7.2 What the Paper Doesn't Address

- Rigorous error bounds for the Laplace approximation itself
- Stability analysis under extreme parameters
- Failure mode characterisation
- Parameter regime guidance

### 7.3 What I am Contributing with my Research

My stress testing has identified a genuine gap in the literature: **Laplace approximation methods can fail silently** in parameter regimes that aren't obviously pathological. The comparison framework I've built provides:

1. Empirical stability boundaries for both methods
2. Metrics for quantifying method agreement without assuming ground truth
3. Evidence that MLMC provides superior robustness with competitive accuracy and significantly lower costs

### 7.4 Final Thought

If Bayer, Häppölä, and Tempone, with all their expertise in this area, concluded that "there is no simple and practical way to control" the Laplace approximation error, then I'm certainly not going to crack that problem. What I *can* do is characterise the behaviour empirically and compare it to the new and improved MLMC method we have been working on. That's the realistic goal for this project. Now, I believe I have spent enough time on Laplace. 

---

## References

1. Bayer, C., Häppölä, J., & Tempone, R. (2017). *Implied Stopping Rules for American Basket Options from Markovian Projection*. Quantitative Finance, 19(3), 371-390. arXiv:1705.00558

2. Giles, M. B. (2015). *Multilevel Monte Carlo methods*. Acta Numerica, 24, 259-328.

3. Gyöngy, I. (1986). *Mimicking the one-dimensional marginal distributions of processes having an Itô differential*. Probability Theory and Related Fields, 71, 501-516.

---

