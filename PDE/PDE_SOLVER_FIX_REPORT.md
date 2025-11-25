# PDE Solver Bug Fix Report

## Overview

This report documents the debugging and fixing of the American basket option PDE solver. Two critical issues were identified and resolved:

1. **Missing S² factor** in the diffusion coefficient
2. **Numerical instability** requiring a switch from explicit to implicit time-stepping

---

## Bug 1: Missing S² in Diffusion Term

### Reminder of the Terms of the Black-Scholes PDE

The Black-Scholes PDE for option pricing is:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

Each term has a distinct financial and physical interpretation:

### 1. Time Decay Term (Theta)
$$\frac{\partial V}{\partial t}$$

Represents how the option value changes with the passage of time. This is typically negative for long options due to time decay eroding value as expiration approaches.

### 2. Diffusion Term (Gamma Effect)
$$\frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}$$

Represents the volatility or randomness in the stock price evolution. The $S^2$ factor is crucial—it ensures that diffusion scales with the stock price itself (larger stock prices have proportionally larger absolute price movements). The second derivative $\partial^2 V/\partial S^2$ is known as **gamma** in finance, measuring the convexity of the option value.

### 3. Drift Term (Delta Contribution)
$$rS\frac{\partial V}{\partial S}$$

Represents the deterministic growth of the stock price under the risk-neutral measure. The risk-free rate $r$ indicates the expected growth rate, whilst the $S$ factor ensures this growth is proportional to the current stock price (exponential growth behaviour). The first derivative $\partial V/\partial S$ is known as **delta** in finance, measuring the sensitivity of option value to stock price changes.

### 4. Discount Term
$$-rV$$

Represents the continuous discounting of the option value at the risk-free rate. This accounts for the time value of money—future payoffs must be discounted back to present value.

---

### The Problem

The Black-Scholes PDE for option pricing is:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

The original code in `finite_difference_operators.py` (or Amelie's `utils.py`) implemented the finite difference coefficients as:

```python
# INCORRECT - missing S²
A = (b_squared / (2 * dS_squared)) + (r * S_grid) / (2 * dS)
B = r + (b_squared / dS_squared)
C = (b_squared / (2 * dS_squared)) - (r * S_grid) / (2 * dS)
```

This corresponds to the PDE:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$$

**The S² multiplier on the diffusion term was missing.**

### Why It Wasn't Detected

The bug was subtle because:

1. The drift term `rS/(2ΔS)` dominated the coefficients (≈1.03 at S=250)
2. The incorrect diffusion term `b²/(2ΔS²)` was tiny (≈0.00007 at b=0.07)
3. The solver still produced "reasonable-looking" option values
4. Different volatility inputs produced nearly identical outputs (the problem I observed that led to me finding this)

### The Fix

```python
# CORRECT - includes S²
diffusion_coeff = (b ** 2) * (S_grid ** 2)
A = (diffusion_coeff / (2 * dS_squared)) + (r * S_grid) / (2 * dS)
B = r + (diffusion_coeff / dS_squared)
C = (diffusion_coeff / (2 * dS_squared)) - (r * S_grid) / (2 * dS)
```

---

## Bug 2: Numerical Instability with Explicit Scheme

### The Problem

After fixing the S² issue, the explicit Euler scheme became numerically unstable. Testing with the notebook's original parameters revealed:

| Volatility | N_timesteps=200 | N_timesteps=5000 |
|------------|-----------------|------------------|
| Low (7%)   | $14.37 ✓       | $14.37 ✓        |
| High (14%) | **$1.2×10⁶⁰** ✗ (just slightly higher than expected) | $20.06 ✓        |

The high volatility case **explodes** to astronomical values with the original 200 timesteps!

### Why It Explodes: The CFL Condition

Explicit finite difference schemes for parabolic PDEs have a **stability constraint** known as the CFL (Courant-Friedrichs-Lewy) condition:

$$\Delta t \leq \frac{(\Delta S)^2}{2 \cdot \max(\sigma^2 S^2)}$$

For our parameters:
- `ΔS = 2.01` (with 150 spatial points over [50, 350])
- `S_max = 350`
- `σ = 0.14` (high volatility case)

The maximum stable timestep is:

$$\Delta t_{\text{stable}} = \frac{(2.01)^2}{2 \times (0.14)^2 \times (350)^2} = \frac{4.04}{4802} \approx 0.00084$$

With T=1.0 year, this requires:

$$N_{\text{timesteps}} > \frac{1.0}{0.00084} \approx 1190$$

The notebook used only 200 timesteps, giving `Δt = 0.005`, which is **6× larger than the stability limit**.

### What Happens When Unstable

When the CFL condition is violated:
1. Small numerical errors get amplified at each timestep
2. The amplification factor is `> 1` instead of `< 1`
3. After ~200 iterations, errors grow exponentially: `(1.01)^200 ≈ 7.3`
4. The solution becomes meaningless (we observed 10⁶⁰!)

---

## Solution: Implicit Backward Euler

### How Explicit Euler Works

The explicit scheme updates the solution using known values from the previous timestep:

$$U^n = U^{n+1} + \Delta t \cdot L(U^{n+1})$$

where L is the spatial operator. This is simple to implement but **conditionally stable**.

### How Implicit Euler Works

The implicit scheme solves for the new solution using the operator at the current timestep:

$$(I - \Delta t \cdot L) U^n = U^{n+1}$$

This requires solving a linear system at each timestep, but is **unconditionally stable**.

### The Tridiagonal System

For a 1D PDE with central differences, the implicit scheme produces a **tridiagonal** system:

$$\begin{pmatrix}
1+\Delta t \beta_1 & -\Delta t \gamma_1 & 0 & \cdots \\
-\Delta t \alpha_2 & 1+\Delta t \beta_2 & -\Delta t \gamma_2 & \cdots \\
0 & -\Delta t \alpha_3 & 1+\Delta t \beta_3 & \cdots \\
\vdots & & & \ddots
\end{pmatrix}
\begin{pmatrix} U^n_1 \\ U^n_2 \\ U^n_3 \\ \vdots \end{pmatrix}
=
\begin{pmatrix} U^{n+1}_1 \\ U^{n+1}_2 \\ U^{n+1}_3 \\ \vdots \end{pmatrix}$$

where:
- $\alpha_i = \frac{\sigma^2 S_i^2}{2\Delta S^2} + \frac{rS_i}{2\Delta S}$ (lower diagonal)
- $\beta_i = r + \frac{\sigma^2 S_i^2}{\Delta S^2}$ (main diagonal)
- $\gamma_i = \frac{\sigma^2 S_i^2}{2\Delta S^2} - \frac{rS_i}{2\Delta S}$ (upper diagonal)

### Thomas Algorithm

Tridiagonal systems can be solved in **O(N)** time using the Thomas algorithm (a specialized form of Gaussian elimination):

1. **Forward elimination**: Transform to upper triangular form
2. **Back substitution**: Solve from bottom to top

This makes the implicit scheme only marginally more expensive per timestep than explicit, while being unconditionally stable.

---

## Comparison: Explicit vs Implicit

| Property | Explicit Euler | Implicit Euler |
|----------|----------------|----------------|
| **Stability** | Conditional (CFL) | Unconditional |
| **Timestep limit** | Δt < ΔS²/(2σ²S²) | None |
| **Cost per step** | O(N) | O(N) |
| **Implementation** | Simple | Tridiagonal solve |
| **Required timesteps** | ~5000+ | ~200 |
| **Total cost** | 25× higher | Baseline |

### When to Use Each

**Explicit schemes** are appropriate when:
- The diffusion coefficient is small
- Very fine spatial grids are used (small ΔS)
- Simplicity is prioritized over efficiency

**Implicit schemes** are preferred when:
- Large diffusion coefficients (high volatility, large S)
- Coarse grids are acceptable
- Computational efficiency matters
- Robustness is required (production code)

---

## Results After Fix

With the implicit scheme and S² correction:

| Volatility | Basket Vol | ATM Option Value |
|------------|------------|------------------|
| Low (10%, 8%, 5%) | 7.1% | $14.36 |
| High (14.1%) | 14.1% | $20.05 |
| **Difference** | | **$5.69** |

The option values now correctly respond to volatility changes, confirming:
1. Higher volatility → higher option value (vega effect) ✓
2. The relationship is approximately linear in σ√T ✓
3. Results are stable and physically meaningful ✓

---

## Files Modified

1. **`finite_difference_operators.py`**
   - Added S² factor to diffusion coefficient
   - Replaced explicit operator with implicit tridiagonal solver
   - Added Thomas algorithm implementation

2. **`american_option_pde_solver.py`**
   - Updated timestepping loop to use implicit scheme
   - Simplified early exercise enforcement

3. **`parameter_sensitivity_study.ipynb`**
   - Added `import copy` and `copy.deepcopy()` for config handling (best practice)

---

## Lessons Learned

1. **Always verify the discretized PDE matches the continuous PDE** - the S² bug was a transcription error that went unnoticed because outputs looked plausible.

2. **Test with extreme parameters** - the bug was only apparent when comparing different volatilities; a single test case wouldn't reveal it.

3. **Explicit schemes are fragile** - when diffusion coefficients scale with state variables (like S²), stability requirements can be severe.

4. **Implicit schemes are worth the complexity** - the Thomas algorithm adds minimal code complexity while providing unconditional stability.

5. **Computational efficiency matters** - 25× fewer timesteps means 25× faster experiments, enabling more thorough sensitivity analysis.
