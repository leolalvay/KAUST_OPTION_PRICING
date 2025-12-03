# README: MLMC Folder

## Multi-Level Monte Carlo for Option Pricing

---

## 1. Introduction

### What Problem Does This Folder Solve?

This folder demonstrates the **Multi-Level Monte Carlo (MLMC)** method for efficiently pricing financial options. Standard Monte Carlo simulation requires enormous numbers of samples to achieve high accuracy, leading to prohibitive computational costs. MLMC overcomes this limitation through a clever variance reduction technique.

### Why Is This Important?

Monte Carlo methods are fundamental in computational finance for pricing derivatives, risk management, and portfolio optimisation. However, the computational cost of achieving high accuracy can be a major bottleneck. MLMC reduces this cost dramatically—by factors of 2× to 37× depending on the desired accuracy—making previously infeasible calculations practical.

### High-Level Approach

MLMC works by using a **telescoping sum**: instead of estimating a quantity directly with expensive high-resolution simulations, MLMC combines many cheap low-resolution simulations with corrections from progressively finer resolutions. The key insight is that the *difference* between consecutive resolution levels has much lower variance than the absolute value, so fewer expensive samples are needed.

The method also employs **optimal sample allocation** via Lagrange multipliers, ensuring that computational resources are distributed efficiently across resolution levels.

### Who Should Use This Code?

- **Researchers** studying Monte Carlo methods or variance reduction techniques
- **Quantitative analysts** needing efficient option pricing or risk calculations
- **Students** learning about computational finance or optimisation
- **Recruiters** evaluating expertise in numerical methods and algorithm optimisation

---

## 2. Quick Start

### Main Entry Points

There are two primary implementations:

1. **NumPy version** (CPU): [MLMC_adaptive.py](MLMC/MLMC_adaptive.py)
2. **JAX version** (GPU/TPU): [MLMC_adaptive_JAX.py](MLMC/MLMC_adaptive_JAX.py)

For validation and convergence rate measurement:
- **Diagnostics**: [MLMC_diagnostics.py](MLMC/MLMC_diagnostics.py)

### How to Run

```bash
# NumPy version (CPU)
python MLMC_adaptive.py

# JAX version (GPU/TPU - requires JAX installation)
python MLMC_adaptive_JAX.py

# Diagnostics (convergence validation)
python MLMC_diagnostics.py
```

### Expected Outputs

Each script generates:

- **Console output**:
  - Sample allocation per level
  - Computational costs for MLMC vs standard MC
  - Speedup factors
  - Final option price estimates with confidence intervals

- **PDF plots**:
  - Panel 1: Sample allocation N_l vs level l (shows exponential decay)
  - Panel 2: Complexity comparison (ε²C vs ε for MLMC vs MC)
  - Demonstrates MLMC's nearly flat cost vs MC's increasing cost

---

## 3. File Tree

```
MLMC/
├── Documentation
│   ├── walkthrough_adapt.md                    (581 lines)
│   └── Walkthrough_diag.md                     (469 lines)
│
├── Main MLMC Algorithms
│   ├── MLMC_adaptive.py                        (503 lines, NumPy)
│   ├── MLMC_adaptive_JAX.py                    (559 lines, JAX)
│   └── MLMC_diagnostics.py                     (250+ lines)
│
├── Baseline MC Package
│   └── Normal_MC/
│       ├── __init__.py
│       ├── BS_Analytic.py                      (Black-Scholes closed-form)
│       ├── MC_GBM.py                           (Basic single-level MC demo)
│       ├── MC_GBM_JAX.py                       (JAX single-level MC demo)
│       ├── MC_estimator.py                     (Reusable MC cost function)
│       ├── MC_estimator_JAX.py                 (JAX MC cost function)
│       ├── mc_cost.txt                         (Precomputed MC costs - NumPy)
│       └── mc_cost_jax.txt                     (Precomputed MC costs - JAX)
│
└── Output Plots (PDF)
    ├── MLMC_adaptive.pdf                       (NumPy version results)
    ├── MLMC_adaptive_JAX.pdf                   (JAX version results)
    └── mlmc_diagnostics.pdf                    (Convergence validation)
```

---

## 4. File Descriptions

### Documentation Files

#### [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) (581 lines)

**Purpose**: Complete pedagogical walkthrough of the adaptive MLMC algorithm.

**What It Covers**:
- Mathematical theory: MLMC estimator, telescoping sum, variance reduction
- Optimal allocation formula derived via Lagrange multipliers
- Bias estimation using Richardson extrapolation
- MLMC complexity theorem: O(ε⁻²) vs O(ε⁻³) for standard MC
- Detailed code annotations connecting theory to implementation
- Physics analogies for intuitive understanding
- Integration with American basket options project (Part 10)

**Recommended For**: Anyone wanting to understand MLMC theory and implementation in depth

---

#### [Walkthrough_diag.md](MLMC/Walkthrough_diag.md) (469 lines)

**Purpose**: Explains the diagnostics script and how to interpret convergence validation results.

**What It Covers**:
- Why MLMC works (telescoping sum, common-mode noise rejection)
- Convergence rate measurements: α (weak convergence) and β (variance decay)
- Expected vs observed rates and what deviations mean
- Kurtosis measurements and effects of payoff non-smoothness
- Interpreting diagnostic plots
- Physics analogies to detector systematics and noise rejection

**Recommended For**: Users wanting to validate MLMC performance or troubleshoot convergence issues

---

### Main MLMC Algorithms

#### [MLMC_adaptive.py](MLMC/MLMC_adaptive.py) (503 lines, NumPy)

**Purpose**: Full production-ready adaptive MLMC algorithm with optimal sample allocation.

**What It Does**:
- Starts with L=3 resolution levels and N0_pilot=100 samples each
- Iteratively runs simulations and estimates variances
- Computes optimal sample allocation via Lagrange multiplier formula
- Dynamically adds finer levels if bias exceeds tolerance
- Continues until convergence criteria met
- Tests 5 accuracy levels: ε ∈ {0.1, 0.05, 0.02, 0.01, 0.005}
- Compares MLMC cost to standard MC baseline
- Generates complexity analysis plots

**Key Functions**:
- `optimal_sample_allocation(V, C, eps)`: Computes N_l^opt from variances and costs using Lagrange multipliers
- `mlmc_level_simulation(l, N_l, ...)`: Runs N_l coupled simulations at level l

**Dependencies**: NumPy, Matplotlib, Normal_MC package

**Outputs**:
- Sample allocation table
- Total computational cost
- Speedup vs standard MC
- PDF plot: [MLMC_adaptive.pdf](MLMC/MLMC_adaptive.pdf)

**Algorithm Overview**:
1. Initialise with coarse levels
2. Run pilot samples, estimate variances
3. Compute optimal allocation (where to spend computational budget)
4. Run additional samples to reach optimal allocation
5. Check bias via Richardson extrapolation
6. Add finer level if bias too large
7. Repeat until converged

---

#### [MLMC_adaptive_JAX.py](MLMC/MLMC_adaptive_JAX.py) (559 lines, JAX)

**Purpose**: GPU/TPU-accelerated version of adaptive MLMC.

**What It Does**: Identical algorithm to NumPy version, but optimised for hardware acceleration.

**Key Features**:
- JIT compilation via `@jit` decorator for hot paths
- Vectorisation via `vmap` instead of Python loops
- Efficient sequential operations via `jax.lax.scan`
- Proper PRNG key management for reproducibility
- 64-bit precision enabled for financial calculations

**Dependencies**: JAX, Matplotlib, Normal_MC package

**Outputs**: Same as NumPy version, saved to [MLMC_adaptive_JAX.pdf](MLMC/MLMC_adaptive_JAX.pdf)

**When to Use**: GPU/TPU available, or larger problems where JIT compilation overhead is amortised

**Note**: This maintains identical mathematical structure to the NumPy version. See [MLMC_adaptive.py](MLMC/MLMC_adaptive.py) documentation above for algorithm details.

---

#### [MLMC_diagnostics.py](MLMC/MLMC_diagnostics.py) (250+ lines)

**Purpose**: Validation tool for MLMC convergence theory.

**What It Does**:
- Runs fixed pilot study with N_pilot=200,000 samples per level
- Measures convergence rates empirically:
  - **Variance decay (β)**: How fast variance decreases with level (expected ~2, observed ~1 due to payoff kinks)
  - **Weak convergence (α)**: How fast mean converges (expected ~1, observed ~1.24)
  - **Kurtosis**: Heavy-tailedness measure (6-12 at coarse levels due to bounded payoff)
- Fits power laws to empirical data
- Generates diagnostic plots showing rates vs level

**Key Insight**: Real-world convergence rates may differ from theory due to payoff non-smoothness (discontinuous derivatives at strike). This is expected and MLMC still provides substantial speedup.

**Dependencies**: NumPy, Matplotlib, Normal_MC package

**Outputs**: Diagnostic plots saved to [mlmc_diagnostics.pdf](MLMC/mlmc_diagnostics.pdf)

---

### Normal_MC Package (Baseline Methods)

The `Normal_MC/` package provides baseline single-level Monte Carlo implementations for comparison.

#### [Normal_MC/\_\_init\_\_.py](MLMC/Normal_MC/__init__.py)

**Purpose**: Package initialisation, exports key functions.

**Exports**:
- `BS_call`, `BS_put` (analytical solutions)
- `run_mc_estimation` (NumPy baseline)
- `run_mc_estimation_jax` (JAX baseline)

---

#### [Normal_MC/BS_Analytic.py](MLMC/Normal_MC/BS_Analytic.py) (110 lines)

**Purpose**: Black-Scholes closed-form pricing for European options.

**What It Does**:
- Implements analytical formulas for call and put options
- Uses error function for cumulative normal distribution
- Validates put-call parity

**Functions**:
- `BS_call(S0, K, T, r, sigma)`: European call option price
- `BS_put(S0, K, T, r, sigma)`: European put option price

**Use Case**: Provides ground truth for validating Monte Carlo estimates

---

#### [Normal_MC/MC_GBM.py](MLMC/Normal_MC/MC_GBM.py) (100+ lines)

**Purpose**: Basic single-level Monte Carlo demonstration.

**What It Does**:
- Shows standard MC without variance reduction
- Illustrates O(ε⁻³) complexity (cost grows rapidly as accuracy improves)
- Euler-Maruyama discretisation of geometric Brownian motion

**Use Case**: Educational baseline showing why MLMC is needed

---

#### [Normal_MC/MC_GBM_JAX.py](MLMC/Normal_MC/MC_GBM_JAX.py) (150+ lines)

**JAX Version**: GPU-accelerated single-level MC demo. Same functionality as [MC_GBM.py](MLMC/Normal_MC/MC_GBM.py), uses `vmap` for vectorised path simulation.

---

#### [Normal_MC/MC_estimator.py](MLMC/Normal_MC/MC_estimator.py) (100+ lines)

**Purpose**: Reusable function for standard MC cost estimation.

**What It Does**:
- Takes array of epsilon values
- For each ε: adaptively selects timestep dt ~ ε/√2
- Computes number of paths needed: N ~ ε⁻²
- Returns total costs and statistics

**Function**: `run_mc_estimation(eps_values, ...)`

**Used By**: [MLMC_adaptive.py](MLMC/MLMC_adaptive.py) for baseline comparison

---

#### [Normal_MC/MC_estimator_JAX.py](MLMC/Normal_MC/MC_estimator_JAX.py) (100+ lines)

**JAX Version**: GPU-accelerated MC cost estimation. Identical functionality to [MC_estimator.py](MLMC/Normal_MC/MC_estimator.py) with PRNG key management.

**Function**: `run_mc_estimation_jax(key, eps_values, ...)`

**Used By**: [MLMC_adaptive_JAX.py](MLMC/MLMC_adaptive_JAX.py) for baseline comparison

---

#### Data Files

**[mc_cost.txt](MLMC/Normal_MC/mc_cost.txt)** and **[mc_cost_jax.txt](MLMC/Normal_MC/mc_cost_jax.txt)**

**Purpose**: Precomputed standard MC costs for plotting.

**Format**: Each line contains `epsilon | total_cost`

**Use Case**: Allows quick plotting of MC baseline without re-running expensive simulations

**Example Data**: Shows approximately O(ε⁻³) scaling as expected from theory

---

## 5. How Programmes Link Together

### Execution Flow Diagram

```
┌─────────────────────────────────────────┐
│  MLMC_adaptive.py (or JAX version)      │ ← Main entry point
│  (defines financial parameters)         │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌──────────────┐
│ Import  │ │ Compute │ │ Run MLMC     │
│ BS Price│ │ MC Cost │ │ Adaptive     │
│         │ │ Baseline│ │ Algorithm    │
└────┬────┘ └────┬────┘ └──────┬───────┘
     │           │             │
     │           │             │
     ▼           ▼             ▼
┌─────────────────────────────────────────┐
│  Normal_MC/ Package                     │
│                                         │
│  ├─ BS_Analytic.py                     │
│  │   └─ BS_call(S0,K,T,r,σ)           │ ← Reference price
│  │                                     │
│  └─ MC_estimator.py (or _JAX.py)      │
│      └─ run_mc_estimation(eps, ...)   │ ← Baseline costs
│          ├─ For each ε:                │
│          │   ├─ dt = ε/√2              │
│          │   ├─ N = C/ε²               │
│          │   └─ Simulate N paths       │
│          └─ Return costs                │
└─────────────┬───────────────────────────┘
              │ Returns: MC costs array
              ▼
┌─────────────────────────────────────────┐
│  MLMC Adaptive Loop                     │
│                                         │
│  1. Initialise: L=3 levels              │
│     N0_pilot=100 samples/level          │
│                                         │
│  2. For l = 0, 1, ..., L:               │
│     ├─ mlmc_level_simulation(l, N_l)   │
│     │   ├─ Generate coupled paths:     │
│     │   │   Fine: dt_fine = h0*2^(-l)  │
│     │   │   Coarse: dt_c = 2*dt_fine   │
│     │   │   (reuse same random numbers)│
│     │   ├─ Compute Y_l = disc*(P_f-P_c)│
│     │   └─ Accumulate statistics       │
│     │                                   │
│     └─ Estimate mean_Y[l], var_Y[l]    │
│                                         │
│  3. optimal_sample_allocation(V, C, ε) │
│     └─ Lagrange formula:                │
│         N_l = (2/ε²)√(V_l/C_l) Σ√(V_j*C_j)
│                                         │
│  4. Run additional samples to reach     │
│     optimal allocation                  │
│                                         │
│  5. Bias check (Richardson extrap):     │
│     If |bias| > ε/√2: add level L+1    │
│                                         │
│  6. Convergence check:                  │
│     If not converged: repeat from 2     │
│                                         │
│  7. Final estimate:                     │
│     Y_MLMC = Σ mean_Y[l]               │
└─────────────┬───────────────────────────┘
              │ Returns: Y_MLMC, costs
              ▼
┌─────────────────────────────────────────┐
│  Comparison & Visualisation             │
│                                         │
│  ├─ Compare MLMC vs MC costs            │
│  ├─ Compute speedup factors             │
│  ├─ Generate plots:                     │
│  │   ├─ Sample allocation N_l vs l     │
│  │   └─ Complexity ε²C vs ε            │
│  └─ Save to PDF                         │
└─────────────────────────────────────────┘
```

### Dependency Chain

```
MLMC_adaptive.py (or _JAX.py)
├─ Normal_MC/__init__.py
│  ├─ BS_Analytic.py
│  │  └─ BS_call() for reference price
│  │
│  └─ MC_estimator.py (or _JAX.py)
│     └─ run_mc_estimation() for baseline
│
├─ numpy (or jax.numpy)
├─ matplotlib
└─ Internal functions:
   ├─ optimal_sample_allocation()
   └─ mlmc_level_simulation()

MLMC_diagnostics.py
├─ Normal_MC/BS_Analytic.py
├─ numpy
└─ matplotlib
```

### Order of Operations

**For MLMC_adaptive.py:**

1. **Setup**: Import analytical price from `BS_Analytic.py`
2. **Baseline**: Run `run_mc_estimation()` to get standard MC costs
3. **MLMC Initialisation**: Set L=3, N0=100
4. **Adaptive Loop** (repeat until converged):
   - Simulate coupled paths at each level
   - Estimate variances and means
   - Compute optimal allocation via Lagrange formula
   - Run additional samples to reach optimal allocation
   - Check bias via Richardson extrapolation
   - Add level if bias too large
5. **Comparison**: Compute MLMC vs MC cost ratios (speedup)
6. **Visualisation**: Generate and save plots

**For MLMC_diagnostics.py:**

1. **Setup**: Import analytical price
2. **Fixed Pilot**: Run N_pilot=200,000 samples at L=6 levels
3. **Convergence Rate Estimation**:
   - Fit power law to variance: V_l ~ 2^(-β*l)
   - Fit power law to mean: |μ_l| ~ 2^(-α*l)
   - Compute kurtosis at each level
4. **Visualisation**: Plot rates and diagnostic measures

---

## 6. Mathematical Concepts (Light Touch)

### Telescoping Sum

**Intuitive Explanation**: Imagine you want to measure the height of a tall building. Instead of using one very precise (expensive) ruler, you use many cheap rough rulers for most of the measurement, and only use the expensive ruler to measure the small *difference* between the rough and precise measurements. Because the difference is much smaller than the total height, you need far fewer expensive measurements.

**Technical**: The MLMC estimator decomposes E[Y_L] = E[Y_0] + Σ_{l=1}^L E[Y_l - Y_{l-1}], where each difference has variance ~100× smaller than |E[Y_l]|.

**See**: [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) Section 3 for mathematical details

---

### Optimal Sample Allocation

**Intuitive Explanation**: If you have a fixed budget to buy variance reduction, you should buy more where it's cheap and less where it's expensive. Optimal allocation is like smart shopping: spend your computational budget where it reduces error most cost-effectively.

**Technical**: Lagrange multipliers minimise total cost Σ N_l C_l subject to variance constraint Σ V_l/N_l ≤ ε². Solution: N_l ∝ √(V_l/C_l).

**See**: [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) Section 5 for derivation

---

### Variance Reduction

**Intuitive Explanation**: When you subtract two similar numbers, the result is smaller than either original number. MLMC exploits this: by correlating fine and coarse simulations (using the same random numbers), their difference has much lower variance than each alone.

**Technical**: Using coupled Brownian paths, Var[P_fine - P_coarse] << Var[P_fine] because the dominant randomness cancels in the difference.

**See**: [Walkthrough_diag.md](MLMC/Walkthrough_diag.md) Section 2 for variance analysis

---

### Convergence Rates

**Intuitive Explanation**: Convergence rates describe how fast errors decrease as you refine your simulation. Faster convergence (higher α, β) means you need fewer levels to achieve target accuracy, reducing computational cost.

**Technical**:
- Weak convergence rate α: |E[Y_l]| ~ 2^(-α*l), α ≈ 1 for Euler-Maruyama
- Variance decay rate β: Var[Y_l] ~ 2^(-β*l), β ≈ 2 for smooth payoffs, β ≈ 1 for kinked payoffs

**See**: [Walkthrough_diag.md](MLMC/Walkthrough_diag.md) Section 4 for convergence theory

---

### Brownian Path Coupling

**Intuitive Explanation**: To compare a high-resolution photo to a low-resolution version, you should blur the *same* original photo, not take two independent photos. Similarly, MLMC creates coarse paths by grouping fine timesteps, ensuring maximal correlation.

**Technical**: Coarse Brownian increments are sums of fine increments: dW_coarse[i] = dW_fine[2i] + dW_fine[2i+1]. This is the optimal coupling for Brownian motion.

**See**: [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) Section 4 for coupling details

---

### Richardson Extrapolation

**Intuitive Explanation**: If you know how fast errors decay with refinement, you can predict how large the next (uncomputed) correction would be. This lets you decide whether to add another level without actually computing it.

**Technical**: Fit |μ_l| ~ A·2^(-α*l) to last 3 levels, extrapolate to l=L+1 to estimate truncation bias.

**See**: [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) Section 6 for bias estimation

---

## 7. Key Results and Outputs

### Visualisations Produced

Each MLMC run generates a two-panel PDF plot:

**Panel 1: Sample Allocation**
- X-axis: Level l (0, 1, 2, ...)
- Y-axis: Number of samples N_l
- Shows exponential decay: N_l decreases as l increases
- Interpretation: Most samples at coarse levels (cheap), few at fine levels (expensive)

**Panel 2: Complexity Comparison**
- X-axis: Accuracy ε (0.1 down to 0.005)
- Y-axis: Computational cost ε²C (normalised)
- Two curves: MLMC (nearly flat) vs MC (increasing)
- Demonstrates MLMC's superior complexity

### Performance Metrics

**Speedup Factors** (from typical runs):

| Accuracy ε | MLMC Cost | MC Cost | Speedup |
|-----------|-----------|---------|---------|
| 0.100     | ~700      | ~1,400  | 2.0×    |
| 0.050     | ~700      | ~2,000  | 2.9×    |
| 0.020     | ~800      | ~3,900  | 5.0×    |
| 0.010     | ~750      | ~10,000 | 13.4×   |
| 0.005     | ~800      | ~28,600 | 36.6×   |

**Key Insight**: As accuracy requirement tightens (ε → 0), MLMC cost remains nearly constant whilst MC cost explodes. This is the O(ε⁻²) vs O(ε⁻³) difference.

### Sample Results

**Typical Output** for European call option (S0=100, K=100, T=1, r=0.05, σ=0.2):

- Analytical price (Black-Scholes): $10.45
- MLMC estimate at ε=0.01: $10.44 ± 0.01
- Number of levels: L=4
- Sample allocation: [10,000, 2,500, 625, 156, 39]
- Total cost: ~750 units
- MC cost for same accuracy: ~10,000 units
- Speedup: 13×

---

## 8. Documentation Guide

### Quick Reference

| Topic | Document | Section |
|-------|----------|---------|
| Complete MLMC theory | [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) | All sections |
| Optimal allocation derivation | [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) | Section 5 |
| Convergence validation | [Walkthrough_diag.md](MLMC/Walkthrough_diag.md) | All sections |
| Convergence rates (α, β) | [Walkthrough_diag.md](MLMC/Walkthrough_diag.md) | Section 4 |
| Kurtosis effects | [Walkthrough_diag.md](MLMC/Walkthrough_diag.md) | Section 5 |
| Complexity theorem | [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) | Section 7 |
| Bias estimation | [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) | Section 6 |
| Physics analogies | Both walkthrough files | Throughout |
| Integration with basket options | [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) | Section 10 |

### Recommended Reading Order for Newcomers

1. **Start here**: This README (you're reading it!)
2. **Run the code**: [MLMC_adaptive.py](MLMC/MLMC_adaptive.py) to see MLMC in action
3. **Visual understanding**: Examine the output plots to see sample allocation and speedup
4. **Theory introduction**: [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) Sections 1-3 (what is MLMC?)
5. **Validation**: Run [MLMC_diagnostics.py](MLMC/MLMC_diagnostics.py) and read [Walkthrough_diag.md](MLMC/Walkthrough_diag.md)
6. **Deep dive**: Complete [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) for full mathematical theory
7. **Advanced**: Experiment with different parameters and compare NumPy vs JAX versions

---

## 9. Common Use Cases

### Scenario 1: Understanding MLMC Fundamentals

**Goal**: See MLMC in action and understand the speedup

**Steps**:
1. Run NumPy version: `python MLMC_adaptive.py`
2. Examine console output showing level-by-level sample allocation
3. Open [MLMC_adaptive.pdf](MLMC/MLMC_adaptive.pdf)
4. Panel 1: Notice exponential decay in N_l (most samples at coarse levels)
5. Panel 2: Notice MLMC cost is nearly flat whilst MC cost increases
6. Read speedup factors in console (2× to 37× depending on ε)

---

### Scenario 2: Validating Convergence Rates

**Goal**: Verify that MLMC theory holds empirically

**Steps**:
1. Run diagnostics: `python MLMC_diagnostics.py`
2. Examine console output showing α and β estimates
3. Open [mlmc_diagnostics.pdf](MLMC/mlmc_diagnostics.pdf)
4. Check variance decay: β ≈ 1-2 (expect lower for kinked payoffs)
5. Check weak convergence: α ≈ 1-1.5 (Euler-Maruyama rate)
6. Kurtosis plots: Higher at coarse levels (heavy tails from bounded payoff)

**Interpretation**: Even if β < 2 (non-smooth payoff), MLMC still provides substantial speedup

---

### Scenario 3: Comparing NumPy vs JAX Performance

**Goal**: Benchmark CPU vs GPU implementation

**Steps**:
1. Run NumPy version: `python MLMC_adaptive.py`
2. Note total computational cost from console
3. Run JAX version: `python MLMC_adaptive_JAX.py`
4. Compare costs (should be nearly identical—algorithmic cost, not wall-clock time)
5. For wall-clock speedup: JAX benefits appear with larger problems or many repeated runs (JIT amortisation)

**When to Use JAX**:
- GPU/TPU hardware available
- Many MLMC runs needed (parameter sweeps)
- Very large number of paths (M ≥ 100,000 per level)
- After JIT compilation overhead is amortised

**When to Use NumPy**:
- CPU-only systems
- Quick one-off calculations
- Debugging (NumPy errors more readable)
- Learning the algorithm (NumPy syntax more familiar)

---

### Scenario 4: Integrating with Other Projects

**Goal**: Use MLMC for a different problem (e.g., volatility estimation)

**Steps**:
1. Study [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) Section 10 (integration with basket options)
2. Identify what plays the role of "payoff" in your problem
3. Implement coupled fine/coarse simulation for your problem
4. Adapt `mlmc_level_simulation()` function to your setting
5. Keep the optimal allocation and bias estimation machinery unchanged
6. See [PDE/mlmc_volatility_estimation.py](../PDE/mlmc_volatility_estimation.py) for a worked example

---

### Scenario 5: Parameter Sensitivity

**Goal**: Understand how financial parameters affect MLMC performance

**Steps**:
1. Modify parameters in [MLMC_adaptive.py](MLMC/MLMC_adaptive.py):
   - Volatility σ: Higher volatility → more levels needed
   - Maturity T: Longer maturity → more timesteps needed
   - Interest rate r: Usually minimal effect on MLMC structure
2. Re-run and compare sample allocations
3. Note: MLMC speedup is *problem-independent* (depends on ε, not on S0, K, etc.)

---

## 10. Next Steps

### How This Folder Relates to Other Folders

This MLMC folder provides foundational theory and simple examples. It connects to the broader project as follows:

1. **PDE Folder**: Applies MLMC to volatility surface estimation for American basket options. The MLMC framework here is extended to estimate regression coefficients across multiple levels rather than just option prices.

2. **L2_Regression Folder**: Uses MLMC within the polynomial regression framework. The multi-level regression combines MLMC variance reduction with L² regression for volatility fitting.

**Workflow Integration**:
```
MLMC/ → Demonstrates core MLMC theory (European call)
   ↓
PDE/mlmc_volatility_estimation.py → Applies MLMC to volatility estimation
   ↓
L2_Regression/Multi_Level/ → Combines MLMC with polynomial regression
```

### What to Explore Next

1. **For deeper theory**: Read [walkthrough_adapt.md](MLMC/walkthrough_adapt.md) and [Walkthrough_diag.md](MLMC/Walkthrough_diag.md) cover-to-cover

2. **For practical applications**: Explore [PDE/](../PDE/) folder to see MLMC applied to American basket options

3. **For regression techniques**: Explore [L2_Regression/](../L2_Regression/) folder to see MLMC combined with polynomial fitting

4. **For experimentation**:
   - Modify ε values in the main script to explore different accuracy/cost trade-offs
   - Try different payoff functions (digital options, power options, etc.)
   - Experiment with different convergence rates α, β

5. **For GPU acceleration**: If you have GPU access, benchmark JAX vs NumPy versions with large M

6. **For validation**: Run the diagnostics script and vary N_pilot to see how convergence rate estimates stabilise

---

## Summary

This folder provides a complete, pedagogically-designed implementation of adaptive Multi-Level Monte Carlo for option pricing. It demonstrates how MLMC achieves computational complexity O(ε⁻²) compared to O(ε⁻³) for standard Monte Carlo, leading to speedup factors of 2× to 37× depending on accuracy requirements.

The implementation includes both CPU (NumPy) and GPU (JAX) versions, comprehensive documentation with physics analogies, diagnostic tools for convergence validation, and baseline single-level MC for comparison. The code is production-ready and serves as a foundation for more advanced applications in the PDE and L2_Regression folders.

**Total Files**: 10 Python files, 2 comprehensive documentation files, 2 data files, 3 PDF outputs

**Key Result**: 2-37× computational speedup vs standard Monte Carlo, with speedup increasing as accuracy requirement tightens.

**Key Innovation**: Adaptive level addition with optimal sample allocation via Lagrange multipliers, achieving provably optimal computational complexity.
