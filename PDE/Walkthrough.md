# Walkthrough: American Basket Option Pricing with MLMC + Markovian Projection

## Overview

This codebase implements a sophisticated method for pricing American basket options by combining:
1. **Markovian Projection**: Reduces a high-dimensional basket problem to a 1D process
2. **Multi-Level Monte Carlo (MLMC)**: Efficiently estimates the projected volatility surface
3. **PDE Solving**: Uses backward Euler to solve the American option pricing equation

Think of it like your ARFF speedup work—we're achieving computational efficiency through smart dimensional reduction while maintaining accuracy.

---

## The Big Picture: What Problem Are We Solving?

### The Challenge

Pricing an American option on a basket of $d$ assets requires solving a $d$-dimensional PDE:

$$\frac{\partial V}{\partial t} + \sum_{i=1}^d r S_i \frac{\partial V}{\partial S_i} + \frac{1}{2}\sum_{i,j=1}^d \rho_{ij} \sigma_i \sigma_j S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} - rV = 0$$

For $d=3$ assets, this is already challenging. For $d>5$, it becomes computationally intractable (curse of dimensionality).

### The Solution: Markovian Projection

**Key insight from Gyöngy's Lemma**: If two processes have the same marginal distributions at all times, they're equivalent for pricing purposes.

We project the $d$-dimensional basket onto a 1D process:
$$B_t = \sum_{i=1}^d w_i S_i^{(t)}$$

where $w_i$ are basket weights. This 1D process has dynamics:

$$dB_t = r B_t dt + b(t, B_t) dW_t$$

The **projected volatility** $b(t, B_t)$ captures all the basket's statistical properties. We estimate this using MLMC and polynomial regression.

---

## File-by-File Walkthrough

### 1. `finite_difference_operators.py`

**Purpose**: Core PDE discretisation tools.

#### Key Function: `apply_pde_operator`

**Mathematics**: Discretises the operator:

$$\mathcal{L}U = \frac{1}{2}b^2(t,S)\frac{\partial^2 U}{\partial S^2} + rS\frac{\partial U}{\partial S} - rU$$

**Implementation**: Uses central differences with three-point stencil:

```python
# Lines 52-55
A = (b_squared / (2 * dS_squared)) + (r * S_grid) / (2 * dS)  # Backward term
B = r + (b_squared / dS_squared)                               # Diagonal term
C = (b_squared / (2 * dS_squared)) - (r * S_grid) / (2 * dS)  # Forward term
```

The stencil approximates:
- $\frac{\partial^2 U}{\partial S^2} \approx \frac{U_{i-1} - 2U_i + U_{i+1}}{\Delta S^2}$
- $\frac{\partial U}{\partial S} \approx \frac{U_{i+1} - U_{i-1}}{2\Delta S}$

**Where it's used**: Called in `american_option_pde_solver.py` line 115 during backward timestepping.

#### Function: `compute_payoff`

Simple payoff calculation:
- Put: $\max(K - S, 0)$
- Call: $\max(S - K, 0)$

---

### 2. `basket_simulation.py`

**Purpose**: Monte Carlo engine for generating basket paths and fitting volatility.

#### Key Function: `simulate_gbm_paths`

**Mathematics**: Simulates correlated Geometric Brownian Motion:

$$dS_i = r S_i dt + \sigma_i S_i dW_i$$

where $dW$ are correlated: $\mathbb{E}[dW_i dW_j] = \rho_{ij} dt$.

**Implementation**: Uses Cholesky decomposition (lines 45-46):

```python
chol_correlation = np.linalg.cholesky(cov_mat)  # G such that GG^T = Cov
dW = brownian_increments @ chol_correlation.T    # Correlated increments
```

This transforms independent Brownian increments into correlated ones.

#### Key Function: `construct_regression_system`

**Purpose**: Build the normal equations $D^T D \mathbf{c} = D^T \boldsymbol{\psi}$ for fitting $b(t,S)$.

**Mathematics**: The volatility surface is represented as:

$$b^2(t, S) = \sum_{p} c_p P_{i_1}(t) P_{i_2}(S)$$

where $P$ are orthonormalised Legendre polynomials on $[-1, 1]$.

**Implementation** (lines 165-178):
1. **Design matrix $D$**: Evaluate tensor product polynomials on basket paths
   ```python
   V_time = legvander(t_vals, degree)    # Legendre in time
   V_space = legvander(S_scaled, degree) # Legendre in space
   D[:, p] = V_time[:, i1] * V_space[:, i2]  # Tensor product
   ```

2. **Target vector $\boldsymbol{\psi}$**: Volatility differences $b_{\text{fine}}^2 - b_{\text{coarse}}^2$
   ```python
   # Lines 191-197: Compute basket volatility from individual assets
   sigma_fine = np.diag(vol * asset_prices_fine)
   b_fine_sq = (sigma_fine @ cov_mat @ sigma_fine).sum() / d**2
   ```

The basket volatility formula comes from:
$$b^2 = \frac{1}{d^2}\sum_{i,j} w_i w_j \sigma_i \sigma_j S_i S_j \rho_{ij}$$
for equal weights $w_i = 1/d$.

#### Function: `fit_volatility_coefficients`

**Implementation**: QR decomposition (lines 215-218):

```python
Q, R = np.linalg.qr(D, mode='reduced')
c = np.linalg.solve(R, Q.T @ psi)
```

This is more stable than solving normal equations directly. Think of it like using SVD in your Higgs SMEFT fits.

---

### 3. `mlmc_volatility_estimation.py`

**Purpose**: Implements the Multi-Level Monte Carlo hierarchy.

#### The MLMC Idea

**Standard Monte Carlo** cost for accuracy $\epsilon$: $\mathcal{O}(\epsilon^{-3})$

**MLMC** uses telescoping sums:

$$\mathbb{E}[Y_L] = \mathbb{E}[Y_0] + \sum_{l=1}^L \mathbb{E}[Y_l - Y_{l-1}]$$

Each level $l$ estimates the **difference** between fine ($h_l$) and coarse ($h_{l-1} = 2h_l$) timesteps. By coupling the paths (using the same random numbers), the variance of $Y_l - Y_{l-1}$ is much smaller than $Y_l$ alone.

**MLMC** cost: $\mathcal{O}(\epsilon^{-2} (\log \epsilon)^2)$ — much better!

#### Key Function: `generate_coupled_paths`

**Critical for variance reduction**: Coarse paths reuse fine path increments.

**Implementation** (lines 118-120):

```python
# Sum consecutive pairs of fine increments for coarse
Z_coarse = Z_fine.reshape(N_paths, N_coarse, 2, d).sum(axis=2)
```

This ensures $Y_{\text{fine}} - Y_{\text{coarse}}$ has low variance because both paths share the same randomness.

#### Key Function: `estimate_coefficients_at_level`

**Level-dependent features**:

1. **Timestep**: $h_l = h_0 \cdot 2^{-l}$ (line 234)
2. **Polynomial degree**: $\deg_l = \deg_{\max} - l$ (line 237) — fewer basis functions at finer levels
3. **Sample size**: $M_l \propto \dim(V_l)^2$ (line 243) — scales with basis dimension

**Why reduce polynomial degree?** At finer levels, we're estimating smaller corrections, so we need less expressive basis functions. This is the key to MLMC efficiency.

#### Function: `aggregate_mlmc_coefficients`

**Implementation** (lines 316-324):

```python
c_total = sum(
    estimate_coefficients_at_level(...)
    for level in range(max_degree + 1)
)
```

Simply sums contributions from all levels—the telescoping magic happens inside each level's estimation.

---

### 4. `american_option_pde_solver.py`

**Purpose**: Solves the pricing PDE with early exercise constraint.

#### The American Option PDE

**Mathematics**: The value $U(t, S)$ satisfies:

$$\frac{\partial U}{\partial t} + \mathcal{L}U = 0, \quad U(t,S) \geq g(S)$$

where $g(S)$ is the payoff. The inequality is the **early exercise constraint**.

**Backward Euler discretisation**:

$$U^n = U^{n+1} + \Delta t \cdot \mathcal{L}U^{n+1}$$

then enforce $U^n \leftarrow \max(U^n, g(S))$.

#### Key Function: `solve_american_option`

**Implementation** (lines 103-120):

```python
for n in reversed(range(N_timesteps)):
    U_next = U[n + 1, :]                # Value at t_{n+1}
    L_U = apply_pde_operator(U_next, b_current, S_grid, r, dS)
    U_continuation = U_next[1:-1] + dt * L_U[1:-1]  # Backward Euler
    U[n, 1:-1] = np.maximum(U_continuation, payoff_grid[1:-1])  # Early exercise
```

**Why backward in time?** We know the terminal condition $U(T, S) = g(S)$ and work backwards to find $U(0, S)$.

**Why backward Euler?** It's **unconditionally stable** for diffusion problems, unlike forward Euler which requires $\Delta t \leq \mathcal{O}(\Delta S^2)$.

#### Function: `compute_exercise_boundary`

Finds the **optimal exercise boundary** $S^*(t)$: the threshold below which (for puts) immediate exercise is optimal.

**Implementation** (lines 154-167): Identifies points where $U(t, S) \approx g(S)$.

For American puts, this boundary moves upward as $t \to T$ (less time value remaining).

---

### 5. `example_american_basket_pricing.py`

**Purpose**: End-to-end demonstration tying everything together.

#### The Complete Pipeline

**Step 1: Domain Estimation** (lines 65-72)
- Runs pilot simulation to find $[S_{\min}, S_{\max}]$ for basket values
- Uses 1st and 99th percentiles to avoid outliers

**Step 2: MLMC Volatility Estimation** (lines 80-87)
- Runs MLMC hierarchy with levels $l = 0, 1, \ldots, \text{max\_degree}$
- Each level estimates correction coefficients
- Aggregates into total coefficient vector $\mathbf{c}$

**Step 3: Build Volatility Surface** (lines 90-92)
- Constructs function $b(t, S) = \sqrt{\sum c_p P_{i_1}(t) P_{i_2}(S)}$

**Step 4: Solve PDE** (lines 100-105)
- Uses backward Euler with the projected volatility $b(t, S)$
- Enforces early exercise at each timestep

**Step 5: Analysis** (lines 113-118, 126-130)
- Extracts exercise boundary
- Computes time value (American value - intrinsic)
- Generates comprehensive visualisations

---

## Key Mathematical Connections

### Markovian Projection ↔ Quantum Mechanics

The projected volatility is analogous to an **effective potential** in many-body physics:
- High-dimensional basket → Effective 1D process
- Local volatility $b(t,S)$ → Effective Hamiltonian
- Captures all statistical properties while reducing dimensionality

### MLMC ↔ Richardson Extrapolation

MLMC's telescoping sum is similar to Richardson extrapolation you used in your neutrino oscillation work:
- Coarse estimate: cheap but inaccurate
- Fine estimate: expensive but accurate
- Difference: captures correction with low variance

### Backward Euler ↔ Implicit Integration

Like using implicit methods for stiff ODEs in your computational physics course:
- Unconditionally stable
- Allows larger timesteps
- Requires solving linear system (but only tridiagonal here)

---

## Computational Complexity Summary

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Standard MC | $\mathcal{O}(\epsilon^{-3})$ | For accuracy $\epsilon$ |
| MLMC | $\mathcal{O}(\epsilon^{-2} \log^2 \epsilon)$ | Via telescoping |
| Markovian Projection | Reduces $d$-dim to 1D | Factor of $d^2$ to $d^3$ savings |
| PDE Solver | $\mathcal{O}(N_t N_s)$ | Backward Euler on 1D grid |
| **Combined** | $\mathcal{O}(\epsilon^{-2} \log^2 \epsilon)$ | Overall computational cost |

Compare to your ARFF work: we're getting **115×-style speedups** through algorithmic improvements rather than GPU parallelisation (though GPU would amplify this further!).

---

## How the Pieces Fit Together

```
                    ┌─────────────────────┐
                    │  Basket Parameters  │
                    │  (S₀, r, σ, ρ, T)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ MLMC Hierarchy      │
                    │ (levels 0→L)        │
                    └──────────┬──────────┘
                               │ Coefficients c
                               ▼
                    ┌─────────────────────┐
                    │ Volatility Surface  │
                    │ b(t,S) = √(Σ cₚPₚ)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ PDE Solver          │
                    │ (Backward Euler)    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ American Option     │
                    │ Value U(0,S)        │
                    └─────────────────────┘
```

---

## Testing Strategy

Each module has comprehensive tests validating:

1. **Mathematical properties**: Terminal conditions, boundary conditions, monotonicity
2. **Numerical accuracy**: Convergence with grid refinement
3. **Economic constraints**: $U \geq g(S)$ for American options, put $\leq K$
4. **Edge cases**: Time-varying volatility, local volatility, coupled paths

The tests aren't just checking "does it run?"—they verify the underlying mathematics is correctly implemented.

---

## Common Pitfalls Avoided

1. **Path coupling**: Coarse paths MUST use the same random numbers as fine paths (line 118 in `mlmc_volatility_estimation.py`)
2. **Subsampling off-by-one**: When $N_{\text{fine}}$ is odd, `[::2]` gives one extra point (fixed in lines 175-178 of `basket_simulation.py`)
3. **Negative variance**: Polynomial fits can give $b^2 < 0$; we clamp to zero and warn (line 272 of `basket_simulation.py`)
4. **Early exercise**: Must check $U \geq g$ at EVERY interior point, not just boundaries (line 118 of `american_option_pde_solver.py`)

---

## Next Steps: Analysing Results

Now that you understand the code, the next phase is interpreting what the numbers mean:
- What does the volatility surface $b(t,S)$ tell us about the basket dynamics?
- How does the exercise boundary $S^*(t)$ behave?
- What's the time value vs intrinsic value split?
- How do the MLMC diagnostics (condition numbers, singular values) indicate fit quality?

Ready to dive into the results analysis!
