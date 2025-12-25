# Error Analysis Theory for MLMC and Laplace Methods in American Basket Option Pricing

**Estimating numerical error in high-dimensional option pricing requires a sophisticated framework combining telescoping sum decomposition, Richardson extrapolation, and primal-dual bounds**—essential tools when ground truth is unknowable. This report provides the mathematical formulations and practical implementation guidance needed for rigorous error quantification in Markovian projection methods with L² polynomial regression.

The core challenge in American basket option pricing is that **neither MLMC nor Laplace approximation provides exact solutions**, yet practitioners must quantify computational accuracy. Modern error analysis resolves this through three complementary approaches: the MLMC telescoping sum framework estimates bias from level differences without knowing truth; Laplace approximation error is controlled through primal-dual bounds; and cross-validation between methods establishes confidence when analytical benchmarks don't exist.

---

## The MLMC telescoping sum framework for error estimation

The foundational MLMC identity exploits a telescoping structure that enables error estimation without knowing the true value P:

$$\mathbb{E}[P_L] = \mathbb{E}[P_0] + \sum_{\ell=1}^{L} \mathbb{E}[P_\ell - P_{\ell-1}]$$

where $P_\ell$ denotes the level-$\ell$ numerical approximation with increasing accuracy as $\ell$ increases. The unbiased MLMC estimator becomes:

$$\hat{Y} = \sum_{\ell=0}^{L} \hat{Y}_\ell = N_0^{-1}\sum_{n=1}^{N_0} P_0^{(0,n)} + \sum_{\ell=1}^{L} \left\{ N_\ell^{-1} \sum_{n=1}^{N_\ell} \left(P_\ell^{(\ell,n)} - P_{\ell-1}^{(\ell,n)}\right) \right\}$$

The critical insight is that **$P_\ell^{(\ell,n)}$ and $P_{l-1}^{(\ell,n)}$ must use the same underlying stochastic sample** (same Brownian path)—this coupling creates the correlation that drives variance reduction. For the mean square error decomposition:

$$\text{MSE} = \underbrace{\sum_{\ell=0}^{L} N_\ell^{-1} V_\ell}_{\text{Variance (statistical error)}} + \underbrace{(\mathbb{E}[P_L] - \mathbb{E}[P])^2}_{\text{Bias}^2}$$

The key assumptions enabling tractable analysis are **geometric convergence rates**: bias decays as $|\mathbb{E}[P_\ell - P]| \leq c_1 \cdot 2^{-\alpha\ell}$ with weak convergence rate $\alpha$, while variance decays as $V_\ell \leq c_2 \cdot 2^{-\beta\ell}$ with rate $\beta$.

---

## Estimating bias without knowing truth via Richardson extrapolation

Since the true value P is unknown, the **level differences serve as a proxy for bias estimation**. The Richardson extrapolation-like technique proceeds from the assumption that weak error follows $\mathbb{E}[P_\ell - P] = c \cdot 2^{-\alpha\ell} + O(2^{-2\alpha\ell})$. This yields:

$$\mathbb{E}[P - P_L] = \sum_{\ell=L+1}^{\infty} \mathbb{E}[P_\ell - P_{\ell-1}] \approx \frac{\mathbb{E}[P_L - P_{L-1}]}{2^\alpha - 1}$$

The practical bias estimate therefore uses the observable quantity $|m_L| = |\mathbb{E}[P_L - P_{L-1}]|$ scaled by the factor $(2^\alpha - 1)^{-1}$. Giles' implementation extends this to a robust estimator using multiple previous levels:

$$\text{bias estimate} = \max_{\ell \in \{L-2, L-1, L\}} \frac{|m_\ell|}{2^{\alpha(L-\ell)}(2^\alpha - 1)}$$

The weak convergence rate $\alpha$ is estimated empirically via linear regression on $\log_2|m_\ell|$ versus level $\ell$, with a floor of $\alpha \geq 0.5$ for robustness. For Euler-Maruyama discretization, $\alpha = 1$ is typical; Milstein schemes achieve $\alpha = 2$ for smooth payoffs.

---

## Variance estimation distinguishes "square of difference" from "difference of squares"

A subtle but critical distinction exists between two variance-related quantities. The **correct MLMC variance estimator** computes the sample variance of the coupled difference $Y_\ell = P_\ell - P_{l-1}$:

$$\hat{V}_\ell = \frac{1}{N_\ell}\sum_{n=1}^{N_\ell}(Y^{(n)})^2 - \left(\frac{1}{N_\ell}\sum_{n=1}^{N_\ell}Y^{(n)}\right)^2$$

This equals $\mathbb{E}[(P_\ell - P_{\ell-1})^2] - (\mathbb{E}[P_\ell - P_{\ell-1}])^2$—the **variance of the difference**. The alternative quantity $\mathbb{E}[P_\ell^2] - \mathbb{E}[P_{\ell-1}^2]$ is the **difference of second moments**, which has fundamentally different meaning.

The crucial relationship involves the correlation $\rho$ between levels:

$$\text{Var}[P_\ell - P_{\ell-1}] = \text{Var}[P_\ell] + \text{Var}[P_{\ell-1}] - 2\rho\sqrt{\text{Var}[P_\ell]\text{Var}[P_{\ell-1}]}$$

Strong coupling (same Brownian path) makes $\rho \to 1$, so $V_\ell \ll \text{Var}[P_\ell] + \text{Var}[P_{\ell-1}]$. This high correlation is what enables MLMC's variance reduction—the variance reduction factor $\text{VRF}_\ell = (\text{Var}[P_\ell] + \text{Var}[P_{\ell-1}])/V_\ell$ should satisfy $\text{VRF} \gg 1$ (typically **$>10$** for effective coupling).

---

## L² and L∞ error norms require pathwise coupling

The L² error norm in MLMC measures **strong (pathwise) convergence**:

$$\|P_\ell - P\|_{L^2} = \sqrt{\mathbb{E}[(P_\ell - P)^2]} = \sqrt{\text{Var}[P_\ell - P] + (\mathbb{E}[P_\ell - P])^2}$$

For Lipschitz payoffs $P = f(S)$ with $|f(S_1) - f(S_2)| \leq K\|S_1 - S_2\|$, the variance bound becomes:

$$V[P_\ell - P_{\ell-1}] \leq K^2 \mathbb{E}[\|S_\ell - S_{\ell-1}\|^2]$$

This connects variance decay rate $\beta$ directly to strong convergence: **Euler-Maruyama achieves $\beta = 1$** (strong error $O(h^{1/2})$), while **Milstein achieves $\beta = 2$** (strong error $O(h)$). The complexity theorem states that for MSE $< \varepsilon^2$:

| Regime | Complexity |
|--------|------------|
| $\beta > \gamma$ (variance dominates) | $O(\varepsilon^{-2})$ |
| $\beta = \gamma$ | $O(\varepsilon^{-2}(\log\varepsilon)^2)$ |
| $\beta < \gamma$ (cost dominates) | $O(\varepsilon^{-2-(\gamma-\beta)/\alpha})$ |

The L∞ norm $\|P_\ell - P\|_\infty = \sup_\omega |P_\ell(\omega) - P(\omega)|$ is more stringent but rarely used in MLMC since pathwise supremum estimates are difficult to compute practically.

---

## Weak versus strong error exhibits distinct behavior in regression problems

**Weak error** measures distributional convergence: $|\mathbb{E}[P_\ell] - \mathbb{E}[P]|$—the difference of means. **Strong error** measures pathwise convergence: $\mathbb{E}[|P_\ell - P|^2]^{1/2}$—requiring the same underlying $\omega$.

For standard SDE discretization, weak convergence typically achieves one order higher than strong (Euler: weak $O(h)$, strong $O(h^{1/2})$). However, **regression problems in Longstaff-Schwartz style algorithms introduce complications**:

- The "truth" $P$ involves function approximation, not just discretization
- Strong convergence assumptions may not hold without explicit pathwise coupling
- Regression coefficients introduce additional bias-variance tradeoffs
- Level differences may not decay geometrically if regression basis changes across levels

For L² polynomial regression on volatility surfaces, the practical recommendation is to **use identical basis functions across all MLMC levels** and verify that $\log_2 V_\ell$ versus $\ell$ maintains linear decay with slope $\approx -\beta$.

---

## Laplace approximation error in Markovian projection

The Bayer-Häppölä-Tempone (2017) framework addresses American basket options by projecting a d-dimensional basket onto a 1D Markovian process via Gyöngy's mimicking theorem. For basket index $I_t = \sum_i w_i S_t^i$, the local volatility of the projected process requires computing:

$$\sigma^2(t,z) = \mathbb{E}[\delta_t^2 | I_t = z]$$

where $\delta_t$ is instantaneous volatility. The **Laplace approximation transforms this d-dimensional integral into evaluation at the saddle point**, achieving **linear complexity in dimension** rather than exponential.

The Laplace error structure is well-characterized asymptotically: **leading-order error is $O(t)$ for small $t$** in the short-time regime. For high-dimensional settings where dimension $p = o(n^{1/3})$, the error scales as $O(p^3/n)$ under mild regularity assumptions. The paper does not provide explicit error bounds for the Laplace component; instead, **total error is controlled through primal-dual bounds**.

---

## Primal-dual bounds quantify total methodology error

Following Rogers (2002), the duality framework provides rigorous bounds without requiring ground truth:

$$V_0(X_0) = \sup_\tau \mathbb{E}[h_\tau(X_\tau)] = \inf_M \mathbb{E}\left[\max_k (h_k(X_k) - M_k)\right]$$

Any stopping rule $\tau$ yields a **lower bound** (primal); any martingale $M$ with $M_0 = 0$ yields an **upper bound** (dual). The practical implementation uses:

- **Lower bound**: Apply the near-optimal exercise strategy from the projected problem to the full basket—directly simulable
- **Upper bound**: Solve only the low-dimensional optimal control problem and construct the dual martingale

**The bound gap quantifies total error** from all sources: Laplace approximation, Markovian projection, discretization, and Monte Carlo sampling. For baskets up to $d=50$ assets, the Bayer paper reports **relative price errors of only a few percent** with tight bound gaps.

---

## Optimal transport coupling for MLMC variance reduction

Beyond simple same-random-number coupling, optimal transport provides a principled framework. The Kantorovich formulation seeks:

$$\min_{\pi \in \Pi(\mu, \nu)} \int c(x,y) \, d\pi(x,y)$$

subject to marginal constraints. For MLMC, we want $(X_\ell, X_{\ell-1})$ to have correct marginals but maximal correlation. The **Giles-Szpruch antithetic coupling** achieves this for multi-dimensional SDEs without Lévy area simulation:

$$Y_\ell = N_\ell^{-1}\sum_i \left[\frac{1}{2}(P(X^f) + P(X^a)) - P(X^c)\right]$$

where $X^f$ (fine) and $X^a$ (antithetic) swap Brownian increments within coarse timesteps. This achieves **$\beta = 2$ for smooth payoffs** and $\beta = 3/2$ for piecewise smooth payoffs (vanilla calls/puts)—enabling $O(\varepsilon^{-2})$ complexity even with only $O(h^{1/2})$ strong convergence.

**Variance reduction verification** requires diagnostic plots:

| Diagnostic | Expected Behavior |
|------------|-------------------|
| $\log_2(V_\ell)$ vs $\ell$ | Linear with slope $-\beta$ |
| Correlation $\rho_\ell$ | $> 0.95$ (often $> 0.99$) |
| Variance reduction factor | $> 10$ |
| Kurtosis $\kappa_\ell$ | $< 100$ for reliability |

---

## Comparing MLMC and Laplace without ground truth

When neither method provides truth, **convergence studies combined with Bland-Altman analysis** establish confidence. Richardson extrapolation estimates individual method errors:

$$\text{Error}_h \approx \frac{f_h - f_{2h}}{2^p - 1}$$

for a method of order $p$ with refinement ratio 2. The **Grid Convergence Index** provides uncertainty quantification:

$$\text{GCI} = 1.25 \cdot |f_h - f_{2h}|/(2^p - 1)$$

For method comparison, the Bland-Altman limits of agreement are essential:

- **Mean difference (bias)**: $\bar{d} = \frac{1}{n}\sum_i (M_1^{(i)} - M_2^{(i)})$
- **95% Limits of Agreement**: $\bar{d} \pm 1.96 \cdot s_d$

Methods agree if the LOA falls within pre-specified practical tolerances. For volatility surfaces, the **L² norm over the surface** provides aggregate comparison:

$$\|\sigma_1 - \sigma_2\|_{L^2} = \sqrt{\int\int |\sigma_1(K,T) - \sigma_2(K,T)|^2 \, dK \, dT}$$

with discrete approximation using appropriate quadrature weights. Industry benchmarks suggest **IV-RMSE $< 1\%$** for liquid strikes/maturities is achievable with high-quality implementations.

---

## Practical implementation guidance for us at KAUST

For American basket option pricing using Markovian projection with L² polynomial regression:

**MLMC configuration**: Use antithetic coupling to achieve $\beta = 2$. Start with $L = 4$ levels, refining if bias estimate exceeds $\sqrt{\theta}\varepsilon$ where $\theta \in [0.25, 0.5]$ allocates error budget between bias and variance. Sample allocation follows:

$$N_\ell = 2\varepsilon^{-2}\sqrt{V_\ell/C_\ell} \sum_{\ell'=0}^{L}\sqrt{V_{\ell'}C_{\ell'}}$$

**Regression basis**: Use **5-20 weighted Laguerre or Hermite polynomials**—same basis across all levels for consistent strong convergence. Apply regression only to in-the-money paths following Longstaff-Schwartz. Verify fit quality through out-of-sample testing on independent paths.

**Error reporting protocol**: Report the Richardson-extrapolated value with GCI uncertainty bands for each method. Compare methods using Bland-Altman plots with bootstrap confidence intervals. Present RMSE stratified by moneyness (ITM/ATM/OTM) and maturity (short/medium/long). State all convergence orders—theoretical and observed.

**Diagnostic verification**: Confirm $\log_2 V_\ell$ versus $\ell$ maintains linear slope; correlation $\rho_\ell > 0.95$; kurtosis $\kappa_\ell < 100$; and consistency check $(a - b + c)/\sigma < 1$ where $a, b, c$ are level estimates with combined standard deviation $\sigma$.

---

## Conclusion

Error analysis for MLMC and Laplace methods in American basket option pricing rests on **three complementary pillars**: the telescoping sum framework that enables bias estimation from observable level differences rather than unknowable truth; the primal-dual bound methodology that quantifies total Laplace/projection error without explicit component decomposition; and the cross-validation toolkit (Richardson extrapolation, Bland-Altman analysis, convergence studies) that establishes confidence when comparing methods without analytical benchmarks.

The key implementation insight is that **variance of level differences**—not difference of variances—governs MLMC efficiency, with strong coupling creating the correlation that enables $O(\varepsilon^{-2})$ complexity. For Markovian projection, the bound gap serves as the practical error metric, implicitly capturing Laplace approximation error alongside all other numerical approximations. When ground truth is absent, self-convergence studies with Richardson extrapolation provide the most rigorous error estimates, while Bland-Altman analysis quantifies method agreement without privileging either as reference.

-----
# How We Apply Our Theory to Our Specific Problem

**The preceding sections establish general frameworks for MLMC error analysis and cross-method validation.** Here we ground these abstract tools in the concrete mathematical object we seek to compute: the projected volatility coefficient $\bar{b}^2(t, s)$ for Markovian projection of high-dimensional American basket options.

We emphasise from the outset: **the true projected volatility is unknown**, so we cannot directly measure our error against ground truth. This section explains what we are computing, where errors enter, and how we validate our results despite not knowing the exact answer.

---

## What we are trying to compute

Our goal is to reduce a $d$-dimensional basket option pricing problem to a tractable one-dimensional problem via Gyöngy's Lemma. The key quantity enabling this reduction is the **projected volatility coefficient**, defined through a conditional expectation:

$$\bar{b}^2(t, s) = \mathbb{E}\left[\vec{P}^{\,T} b(t, \vec{X}(t))\, b(t, \vec{X}(t))^T \vec{P} \;\Big|\; \vec{P} \cdot \vec{X}(t) = s,\; \vec{X}(0) = \vec{x}_0\right]$$

In words: given that the basket has value $s$ at time $t$, what is the expected instantaneous variance of the basket?

This conditional expectation cannot be computed analytically for general correlation structures. The best mean-square approximation property tells us that this conditional expectation is equivalent to solving a regression problem: find the function $h(t, s)$ that best predicts the instantaneous variance given only the basket value.

This yields the **theoretical regression formulation**:

$$\bar{b}^2(\cdot, \cdot) = \underset{h \in L^2}{\arg\min} \int_0^T \mathbb{E}\left[\left(\psi(t, \vec{X}(t)) - h(t, \vec{P} \cdot \vec{X}(t))\right)^2 \;\Big|\; \vec{X}(0) = \vec{x}_0\right] dt$$

where $\psi(t, \vec{X}) := \vec{P}^{\,T} b(t, \vec{X})\, b(t, \vec{X})^T \vec{P}$ is the instantaneous projected variance.

This is an infinite-dimensional optimisation problem. We cannot solve it exactly.

*Note: The theoretical regression formulation and its Monte Carlo discretisation correspond to equations (61) and (62) in Amelie's working notes.*

---

## What we actually compute

To make the problem tractable, we introduce several approximations:

**1. Restrict to polynomials:** Instead of searching over all possible functions, we search over polynomials up to some degree:

$$h(t, s) = \sum_p c_p \, P_{i_1}(\tilde{t}) \, P_{i_2}(\tilde{s})$$

where $P_n$ are Legendre polynomials and $\tilde{t}, \tilde{s}$ are rescaled to $[-1, 1]$.

**2. Replace the integral with a sum:** We evaluate at discrete timesteps $t_0, t_1, \ldots, t_{N-1}$.

**3. Replace the expectation with an average:** We simulate $M$ paths and average over them.

**4. Use numerical paths:** We simulate paths using Euler-Maruyama, not exact SDE solutions.

The **computational formulation** becomes:

$$\vec{c} = \underset{\vec{c}}{\arg\min} \frac{1}{M} \sum_{m=1}^{M} \frac{1}{N} \sum_{n=0}^{N-1} \left( \psi(t_n, \hat{\vec{X}}^{(m)}(t_n)) - \sum_{p} c_p\, \phi_p(t_n, S^{(m)}_n) \right)^2$$

where $S^{(m)}_n = \vec{P} \cdot \hat{\vec{X}}^{(m)}(t_n)$ is the basket value along path $m$ at time $t_n$.

This is a standard linear least-squares problem, solved via the normal equations.

---

## Where errors enter

Each approximation introduces error:

| Approximation | Error Source | What Controls It |
|---------------|--------------|------------------|
| Polynomial restriction | The true function may not be a polynomial | Polynomial degree |
| Time discretisation | We sample at discrete times only | Number of timesteps $N$ |
| Monte Carlo sampling | Finite samples introduce noise | Number of paths $M$ |
| Euler-Maruyama paths | Numerical paths differ from true SDE solutions | Timestep size $h$ |

---

## What we cannot measure

Here is the critical point: **we do not know $\bar{b}^2_{\text{true}}$**.

This means we cannot compute:
$$\|\bar{b}^2_{\text{computed}} - \bar{b}^2_{\text{true}}\|$$

regardless of what norm we use. Any error analysis that claims to measure this quantity directly is either:
- Using one method as a proxy for truth (which doens't help us here, especially since we cannot trust my crude way of regenerating the Laplace results from the paper), or
- Computing something else entirely


---

## What we can measure

Despite not knowing the true answer, we can still validate our methods through several approaches:

### Self-convergence

As we refine our computation (more levels, more samples, higher polynomial degree), the estimates should stabilise. If $\hat{b}^2_L$ denotes our estimate using $L$ MLMC levels, we monitor:

$$|\hat{b}^2_L - \hat{b}^2_{L-1}|$$

If this is small and decreasing geometrically with $L$, we have evidence that the method is converging to *something*. We cannot prove it is converging to the true answer, but divergence or erratic behaviour would indicate problems.

### Richardson extrapolation for bias estimation

If we assume the bias decays geometrically (a standard assumption for discretisation methods), then:

$$\mathbb{E}[\hat{b}^2_L] - \bar{b}^2_{\text{true}} \approx c \cdot 2^{-\alpha L}$$

for some constants $c$ and $\alpha$. The observable level differences $m_L = \mathbb{E}[\hat{b}^2_L - \hat{b}^2_{L-1}]$ allow us to estimate the remaining bias:

$$\text{Bias estimate} \approx \frac{m_L}{2^\alpha - 1}$$

This is an estimate, not exact knowledge, but it gives us a principled way to judge when we have refined enough.

### MLMC variance diagnostics

The variance of level corrections tells us whether our coupling is working:

$$V_\ell = \text{Var}[P_\ell - P_{\ell-1}]$$

We compute this as the sample variance of the differences (not the difference of sample variances). For effective coupling, we expect:
- $V_\ell$ to decay geometrically with level
- Correlation between $P_\ell$ and $P_{\ell-1}$ to be high (above 0.95)

If these diagnostics fail, something is wrong with the implementation, regardless of what the final numbers say.

### Cross-method comparison

The MLMC regression approach and the Laplace approximation approach target the same mathematical object through completely different computational routes. If both methods produce similar option prices, this provides confidence that both are working correctly.

We do not expect the volatility surfaces to match exactly. The methods make different approximations:
- MLMC fits polynomials to Monte Carlo samples
- Laplace evaluates analytical approximations at quadrature points

But the downstream quantities (option prices, exercise boundaries) should agree if both methods are valid.

### End-to-end validation

Ultimately, we care about option prices, not volatility surfaces. The volatility surface is an intermediate quantity. Two surfaces that look different can still produce nearly identical option prices if they agree in the regions that matter for the pricing PDE.

We can validate by:
1. Computing option prices using both MLMC and Laplace volatility surfaces
2. Comparing these prices
3. Checking against known benchmarks for simple cases (e.g., single-asset Black-Scholes)

Looking at what my comparison code does so far, this is exactly what we see. The two methods produce near identical option prices. They are only off by a few parts of a penny!

---

## The role of MLMC in this framework

MLMC does not magically eliminate error. What it does is **reduce computational cost** while maintaining accuracy.

The telescoping sum:
$$\hat{b}^2_L = \hat{b}^2_0 + \sum_{\ell=1}^{L} (\hat{b}^2_\ell - \hat{b}^2_{\ell-1})$$

allows us to:
- Use many cheap samples at coarse levels (where corrections are large but cheap)
- Use few expensive samples at fine levels (where corrections are small)

The variance of each correction term $V_\ell = \text{Var}[\hat{b}^2_\ell - \hat{b}^2_{\ell-1}]$ must be computed as the **variance of the coupled difference**, not as the difference of individual variances. This is because the coupling (using the same Brownian path) induces correlation that makes the difference small even when the individual terms have large variance.

---

## Implementation plan

Any code that computes the volatility surface via MLMC should incorporate the following diagnostics:

**1. Level-wise statistics (computed during MLMC):**
- Store $m_\ell = \frac{1}{N_\ell} \sum_n (P_\ell^{(n)} - P_{\ell-1}^{(n)})$ (mean correction at each level)
- Store $V_\ell = \text{Var}[P_\ell - P_{\ell-1}]$ (variance of corrections)
- Compute correlation $\rho_\ell$ between consecutive levels

**2. Convergence checks (post-computation):**
- Verify $|m_\ell|$ decays geometrically with $\ell$
- Verify $V_\ell$ decays geometrically with $\ell$
- Estimate bias via $|m_L| / (2^\alpha - 1)$
- Flag warnings if $\rho_\ell < 0.9$ (coupling may be ineffective)

**3. Cross-validation (when alternative method available):**
- Compute both surfaces on identical $(t, s)$ grid
- Report pointwise differences and summary statistics
- Compare downstream option prices, not just surfaces

**4. Self-consistency (multiple runs):**
- Run MLMC multiple times with different random seeds
- Check that results fall within expected statistical variation
- Verify $1/\sqrt{n}$ convergence of the ensemble mean

These diagnostics can be implemented as a lightweight module that wraps any MLMC volatility surface computation, logging the relevant quantities without modifying the core algorithm.

---

## Summary of our validation strategy

Since we cannot measure error against truth, we adopt a multi-pronged validation approach:

| Validation Method | What It Checks | Evidence of Success |
|-------------------|----------------|---------------------|
| Self-convergence | Method converges as refined | Level differences decay geometrically |
| Richardson extrapolation | Bias is under control | Estimated bias below tolerance |
| Variance diagnostics | Coupling is effective | $V_\ell$ decays, correlation high |
| Cross-method comparison | Methods agree | Option prices match within tolerance |
| Benchmark tests | Implementation is correct | Known cases reproduced exactly |

No single check is sufficient. Together, they provide confidence that our computed volatility surfaces, while not provably equal to the true answer, are fit for purpose in pricing American basket options.

---


