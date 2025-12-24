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