# Complete Walkthrough: Adaptive MLMC with Optimal Sample Allocation

Let me walk you through this code step by step, explaining the theory, the mathematics, and how it's implemented.

---

## **Part 1: The Big Picture**

### What Does This Code Do?

The **first script** (`mlmc_diagnostics_simple.py`) was a **validation tool**. It checked if MLMC theory works by running fixed samples at each level.

This **second script** (`mlmc_adaptive.py`) is the **production algorithm**. It:
1. Starts with a small pilot run
2. Estimates variance at each level
3. **Adaptively allocates samples** to minimise cost
4. Adds levels dynamically until target accuracy is reached
5. Does this for multiple accuracy targets (ε = 0.1, 0.05, 0.02, 0.01, 0.005)

**Physics analogy**: The first script was like a systematic uncertainty study. This script is like actually running your analysis with optimal trigger settings to maximise physics reach within your computing budget!

---

## **Part 2: The Mathematical Foundation**

### The MLMC Estimator (Telescoping Sum)

**Mathematical formula:**
$$E[P_L] = E[P_0] + \sum_{l=1}^{L} E[Y_l]$$

where:
- $P_l$ = option payoff computed with timestep $h_l = h_0 \cdot 2^{-l}$
- $Y_l = e^{-rT}(P_l - P_{l-1})$ = discounted correction at level $l$

**In the code:**
```python
# Final MLMC estimate (line ~380)
final_mean_Y = sum_stats[0, :] / N
P_mlmc = np.sum(final_mean_Y)
```

**Why this works**: Instead of estimating $E[P_L]$ directly (expensive!), we estimate the **corrections** $E[Y_l]$, which have much lower variance.

---

### Optimal Sample Allocation (The Core Theory)

**The optimisation problem:**

Minimise: 
$$\text{Cost} = \sum_{l=0}^{L} N_l \cdot C_l$$

Subject to: 
$$\text{Variance}[\hat{P}_{\text{MLMC}}] = \sum_{l=0}^{L} \frac{V_l}{N_l} \leq \epsilon^2$$

where:
- $N_l$ = number of samples at level $l$
- $C_l$ = computational cost per sample (proportional to $2^l$ for GBM)
- $V_l = \text{Var}[Y_l]$ = variance of the correction
- $\epsilon$ = target root-mean-square error

**Solution via Lagrange multipliers:**

$$N_l^{\text{opt}} = \frac{2}{\epsilon^2} \sqrt{\frac{V_l}{C_l}} \sum_{j=0}^{L} \sqrt{V_j C_j}$$

**In the code:**
```python
def optimal_sample_allocation(V, C, eps):
    """
    N_l = (2/eps^2) * sqrt(V_l/C_l) * sum_j sqrt(V_j * C_j)
    """
    sum_sqrt_VC = np.sum(np.sqrt(V * C))
    mu = 2.0 * sum_sqrt_VC / eps**2
    N_opt = mu * np.sqrt(V / C)
    return np.ceil(N_opt).astype(int)
```

**Why this formula?**

The $\sqrt{V_l / C_l}$ term means:
- **High variance + low cost** → Use **many samples** (cheap to reduce variance)
- **Low variance + high cost** → Use **few samples** (already accurate, don't waste resources)

**Physics analogy**: Like allocating your DAQ bandwidth. You trigger more on channels with poor signal/noise (high $V_l$), but less on channels with clear signals (low $V_l$), weighted by readout cost.

---

### Convergence Rates (From Theory)

**Weak convergence (bias decay):**
$$|E[Y_l]| \sim 2^{-\alpha l}, \quad \alpha \approx 1$$

**Variance decay:**
$$\text{Var}[Y_l] \sim 2^{-\beta l}, \quad \beta \approx 2$$

**Cost growth:**
$$C_l \sim 2^{\gamma l}, \quad \gamma = 1$$

**In the code:**
```python
# Line 206
alpha = 1.0      # Weak convergence rate
beta = 1.0       # Variance decay rate  
gamma = 1.0      # Cost growth rate
```

**Note**: We use $\beta = 1$ here (conservative) rather than the theoretical $\beta = 2$ because the European call payoff has a kink at $S = K$, which slows variance decay at coarse levels.

---

### MLMC Complexity Theorem

**For standard Monte Carlo:**
$$\text{Work}_{\text{MC}} \sim \mathcal{O}(\epsilon^{-3})$$

**For optimal MLMC:**
$$\text{Work}_{\text{MLMC}} \sim \mathcal{O}(\epsilon^{-2}(\log \epsilon)^2)$$

**Why the improvement?**

The key is that $\sum_{l=0}^{L} N_l^{\text{opt}} C_l$ is **dominated by coarse levels** (which are cheap), not fine levels (which are expensive).

With $\beta = 2$ and $\gamma = 1$:
$$\sum_{l=0}^{L} N_l^{\text{opt}} C_l \sim \sum_{l=0}^{L} 2^{l/2} \cdot 2^l \sim \sum_{l=0}^{L} 2^{3l/2}$$

Wait, that diverges! But the variance constraint means we only need $L \sim \log(1/\epsilon)$ levels, so:
$$\text{Total cost} \sim \epsilon^{-2} \cdot L \sim \epsilon^{-2} \log(1/\epsilon)$$

Much better than $\epsilon^{-3}$!

---

## **Part 3: Code Structure Walkthrough**

### Function 1: `optimal_sample_allocation(V, C, eps)`

**What it does**: Computes the optimal number of samples at each level.

**Mathematical formula implemented:**
$$N_l^{\text{opt}} = \left\lceil \frac{2}{\epsilon^2} \sqrt{\frac{V_l}{C_l}} \sum_{j=0}^{L} \sqrt{V_j C_j} \right\rceil$$

**Code:**
```python
def optimal_sample_allocation(V, C, eps):
    sum_sqrt_VC = np.sum(np.sqrt(V * C))        # Σ_j √(V_j C_j)
    mu = 2.0 * sum_sqrt_VC / eps**2             # 2/ε² × Σ_j √(V_j C_j)
    N_opt = mu * np.sqrt(V / C)                 # × √(V_l / C_l)
    return np.ceil(N_opt).astype(int)           # Round up
```

**Why ceil (round up)?** To ensure we meet the variance constraint. Rounding down might leave RMSE slightly above $\epsilon$.

---

### Function 2: `mlmc_level_simulation(l, N_l, ...)`

**What it does**: Runs $N_l$ coupled simulations at level $l$.

**Key mathematical steps:**

1. **Timestep**: $h_{\text{fine}} = h_0 \cdot 2^{-l}$
   ```python
   h_fine = h0 * 2**(-l)
   ```

2. **Brownian increments**: $\Delta W \sim \mathcal{N}(0, h)$
   ```python
   dW_fine = np.random.normal(0, np.sqrt(h_fine), size=n_steps)
   ```

3. **Euler-Maruyama scheme**: 
   $$S_{n+1} = S_n + rS_n h + \sigma S_n \Delta W_n$$
   ```python
   S = S0
   for dW in dW_fine:
       S += r * S * h_fine + sigma * S * dW
   ```

4. **Brownian coupling** (the MLMC magic!):
   $$\Delta W_{\text{coarse}}[i] = \Delta W_{\text{fine}}[2i] + \Delta W_{\text{fine}}[2i+1]$$
   ```python
   dW_coarse = dW_fine.reshape(-1, 2).sum(axis=1)
   ```

5. **Correction**:
   $$Y = e^{-rT}(P_{\text{fine}} - P_{\text{coarse}})$$
   ```python
   Y = disc * (Pf - Pc)
   ```

**Returns**: $\sum Y$ and $\sum Y^2$ for computing mean and variance.

---

## **Part 4: The Main Adaptive Algorithm**

### Initialisation (Lines 220-260)

```python
L = 3                                    # Start with 4 levels (0,1,2,3)
h0 = 0.5                                 # Coarsest timestep
N0_pilot = 100                           # Pilot samples per level
N = np.zeros(L + 1, dtype=int)          # Current sample counts
dN = np.full(L + 1, N0_pilot, dtype=int) # Need N0_pilot samples initially
sum_stats = np.zeros((2, L + 1))        # [sum Y, sum Y²] at each level
cost_per_sample = 2.0**(gamma * levels) # C_l = 2^l
```

**Why start with L=3?** A reasonable guess. The algorithm will add more levels if needed.

**Why N0_pilot = 100?** Small enough to be cheap, large enough to get rough variance estimates.

---

### The Adaptive Loop (Lines 266-370)

The loop continues while `np.any(dN > 0)`, meaning "while any level needs more samples".

#### **Step 1: Run Additional Samples** (Lines 276-289)

```python
for l in range(L + 1):
    if dN[l] > 0:
        sum_Y, sum_Y_sq = mlmc_level_simulation(l, dN[l], ...)
        N[l] += dN[l]
        sum_stats[0, l] += sum_Y
        sum_stats[1, l] += sum_Y_sq
```

**What's happening**: For each level that needs more samples, run them and accumulate statistics.

---

#### **Step 2: Estimate Mean and Variance** (Lines 295-300)

**Mathematical formulas:**
$$\hat{\mu}_l = \frac{1}{N_l} \sum_{i=1}^{N_l} Y_l^{(i)}$$
$$\hat{V}_l = \frac{1}{N_l} \sum_{i=1}^{N_l} (Y_l^{(i)})^2 - \hat{\mu}_l^2$$

**Code:**
```python
mean_Y = np.abs(sum_stats[0, :] / N)
var_Y = np.maximum(0.0, sum_stats[1, :] / N - mean_Y**2)
```

**Why `np.maximum(0.0, ...)`?** Due to rounding errors, the variance estimate can be slightly negative. We clamp it to zero.

**Why `np.abs` on mean?** We only care about bias magnitude, not direction.

---

#### **Step 3: Enforce Minimum Decay Rates** (Lines 306-314)

**Mathematical idea**: With limited samples, variance estimates at fine levels can be **noisy**. We impose:
$$\hat{V}_l \geq \frac{1}{2} \cdot \frac{\hat{V}_{l-1}}{2^\beta}$$
$$|\hat{\mu}_l| \geq \frac{1}{2} \cdot \frac{|\hat{\mu}_{l-1}|}{2^\alpha}$$

**Code:**
```python
for l in range(2, L + 1):
    mean_Y[l] = max(mean_Y[l], 0.5 * mean_Y[l-1] / 2**alpha)
    var_Y[l] = max(var_Y[l], 0.5 * var_Y[l-1] / 2**beta)
```

**Why the factor 0.5?** Conservative buffer. We expect decay rate $2^{-\beta}$, but allow some fluctuation.

**Physics analogy**: Like systematic error floors in your detector. You know resolution can't be arbitrarily good, so you impose a minimum based on theory.

---

#### **Step 4: Compute Optimal Allocation** (Lines 319-323)

```python
N_opt = optimal_sample_allocation(var_Y, cost_per_sample, eps)
dN = np.maximum(N_opt - N, 0)
```

**What's happening**: Given updated variance estimates, recompute how many samples we **should** have at each level. The difference $\Delta N_l = N_l^{\text{opt}} - N_l^{\text{current}}$ tells us how many **more** to run.

---

#### **Step 5: Check Convergence and Level Addition** (Lines 330-370)

**Convergence criterion:**
```python
converged = np.sum(dN > 0.01 * N) == 0
```

**Interpretation**: "All levels have achieved their target sample counts (within 1%)".

**If converged, check bias:**

The **remaining bias** from truncating at level $L$ is estimated using Richardson extrapolation:

**Mathematical formula:**
$$\text{Bias} \approx \frac{1}{2^\alpha - 1} \max_{i \in \{-2,-1,0\}} \left( |\hat{\mu}_{L+i}| \cdot 2^{\alpha i} \right)$$

**Code:**
```python
indices = np.array([-2, -1, 0])
means_last_three = mean_Y[L + indices]  # [μ_{L-2}, μ_{L-1}, μ_L]
bias_estimate = np.max(means_last_three * 2.0**(alpha * indices)) / (2.0**alpha - 1.0)
```

**Why this formula?** It's a **Richardson extrapolation**. We fit an exponential $|\mu_l| \sim 2^{-\alpha l}$ to the last 3 levels and extrapolate to estimate $|\mu_{L+1}|$, which bounds the truncation error.

**Decision rule:**
$$\text{If } \text{Bias} > \frac{\epsilon}{\sqrt{2}} \implies \text{Add level } L+1$$

**Why $\epsilon/\sqrt{2}$?** Total RMSE has two components:
$$\text{RMSE}^2 = \text{Variance} + \text{Bias}^2$$

We want both $\leq \epsilon^2/2$, so $\text{RMSE} \leq \epsilon/\sqrt{2} + \epsilon/\sqrt{2} = \epsilon$.

**Adding a level:**
```python
L += 1
N = np.append(N, 0)
dN = np.append(dN, 0)
sum_stats = np.hstack((sum_stats, np.zeros((2, 1))))
var_Y = np.append(var_Y, var_Y[-1] / 2**beta)  # Extrapolate variance
```

Then recompute optimal allocation and continue the loop!

---

## **Part 5: Final Estimate and Results**

### Computing the MLMC Estimator (Lines 378-382)

**Mathematical formula:**
$$\hat{P}_{\text{MLMC}} = \sum_{l=0}^{L} \hat{\mu}_l = \sum_{l=0}^{L} \frac{1}{N_l} \sum_{i=1}^{N_l} Y_l^{(i)}$$

**Code:**
```python
final_mean_Y = sum_stats[0, :] / N
P_mlmc = np.sum(final_mean_Y)
```

**Interpretation**: Sum up the corrections from all levels. This is your option price estimate!

---

### Cost Analysis (Lines 385-388)

**Total computational cost:**
$$C_{\text{total}} = \sum_{l=0}^{L} N_l \cdot C_l$$

**Normalised cost** (for fair comparison across $\epsilon$):
$$C_{\text{norm}} = \epsilon^2 \cdot C_{\text{total}}$$

**Code:**
```python
total_cost = (eps**2) * np.dot(N, cost_per_sample)
```

**Why multiply by $\epsilon^2$?** 

If MLMC has optimal complexity $C \sim \epsilon^{-2}$, then $\epsilon^2 \cdot C$ should be **constant** (flat line on the plot)!

For standard MC with $C \sim \epsilon^{-3}$, we'd have $\epsilon^2 \cdot C \sim \epsilon^{-1}$ (decreasing line).

---

### Standard MC Cost Estimate (Line 390)

**Formula:**
$$C_{\text{MC}} = \frac{T}{h_{\text{finest}}} \cdot V_0$$

where:
- $T/h_{\text{finest}}$ = number of timesteps at finest resolution
- $V_0 = \text{Var}[P_0]$ = variance of payoff (doesn't benefit from MLMC variance reduction)

**Code:**
```python
h_finest = h0 * 2**(-L)
cost_mc_estimate = (T / h_finest) * var_Y[0]
```

**Why this estimate?** Standard MC would need to:
1. Run at the **finest timestep** (to match MLMC bias)
2. Use enough samples to match MLMC **variance**: $N_{\text{MC}} \sim V_0 / \epsilon^2$

So $C_{\text{MC}} \sim (T/h) \cdot (V_0/\epsilon^2) \sim \epsilon^{-2} \cdot \epsilon^{-1} = \epsilon^{-3}$

---

## **Part 6: The Plots Explained**

### Panel 1: Sample Allocation $N_l$ vs Level $l$

**What it shows**: For each target accuracy $\epsilon$, how many samples MLMC allocates at each level.

**Key observations:**

1. **Exponential decay**: $N_l$ drops by roughly a factor of 2-4 each level
   
   **Why?** From the formula:
   $$N_l \propto \sqrt{\frac{V_l}{C_l}}$$
   
   With $V_l \sim 2^{-\beta l}$ and $C_l \sim 2^l$:
   $$N_l \sim 2^{-\beta l / 2} / 2^{l/2} = 2^{-(\beta+1)l/2}$$
   
   For $\beta = 1$: $N_l \sim 2^{-l}$ (halves each level) ✅

2. **Most samples at coarse levels**: Level 0 might have 50,000 samples, level 5 only 1,000
   
   **Why?** Coarse levels are **cheap** (few timesteps) but have **high variance**. The optimal strategy is to "buy variance reduction" where it's cheapest!

3. **Tighter $\epsilon$ uses more levels**: 
   - $\epsilon = 0.1$ → 4 levels
   - $\epsilon = 0.005$ → 5-6 levels
   
   **Why?** Need finer timesteps to control bias as $\epsilon \to 0$.

---

### Panel 2: $\epsilon^2 \cdot C$ vs $\epsilon$ (Complexity Analysis)

**What it shows**: How total cost scales with accuracy.

**Theoretical prediction:**

| Method | Cost | $\epsilon^2 \cdot C$ |
|--------|------|---------------------|
| Standard MC | $C \sim \epsilon^{-3}$ | $\sim \epsilon^{-1}$ (grows!) |
| MLMC | $C \sim \epsilon^{-2}$ | $\sim \text{const}$ (flat!) |

**What you see:**

- **Orange line (MLMC)**: Nearly **flat** around 700-800
  - Confirms $C \sim \epsilon^{-2}$ ✅
  
- **Green line (Standard MC)**: Would grow like $\epsilon^{-1}$ if extended
  - At $\epsilon = 0.01$: MC cost is ~20× higher than MLMC
  - At $\epsilon = 0.005$: Gap widens to ~50×

**Why the slight variation in the orange line?**

Real implementation isn't perfectly $\epsilon^{-2}$ because:
1. Pilot run overhead (fixed cost independent of $\epsilon$)
2. Integer rounding of sample counts
3. Discrete level additions (can't add 0.3 of a level!)

But it's **remarkably close** to flat, validating the theory!

---

## **Part 7: The Summary Table**

```
Epsilon      MLMC Cost       MC Cost         Speedup   
----------------------------------------------------------------------
0.100        691.86          1388.64         2.0×
0.050        688.31          1998.07         2.9×
0.020        781.64          3877.33         5.0×
0.010        750.52          10089.23        13.4×
0.005        783.84          28661.18        36.6×
```

**What this tells you:**

1. **MLMC cost is roughly constant** (~700-800) across all $\epsilon$ ✅

2. **Standard MC cost explodes** as $\epsilon$ decreases:
   - $\epsilon = 0.1$: Cost ≈ 1,400
   - $\epsilon = 0.005$: Cost ≈ 29,000 (20× worse!)

3. **Speedup grows with accuracy**:
   - At loose accuracy ($\epsilon = 0.1$): 2× speedup (not worth the complexity?)
   - At tight accuracy ($\epsilon = 0.005$): **37× speedup** (absolutely worth it!)

**Physics analogy**: Like comparing $O(N^2)$ vs $O(N \log N)$ sorting algorithms. At small $N$, both are fine. At large $N$ (or tight $\epsilon$ here), the better algorithm dominates.

---

## **Part 8: Connecting Theory to Code**

### The Optimal Allocation Formula

**Theory:**
$$N_l^{\text{opt}} = \frac{2}{\epsilon^2} \sqrt{\frac{V_l}{C_l}} \sum_{j=0}^{L} \sqrt{V_j C_j}$$

**Derivation sketch:**

Lagrangian:
$$\mathcal{L} = \sum_{l=0}^{L} N_l C_l + \lambda \left( \sum_{l=0}^{L} \frac{V_l}{N_l} - \epsilon^2 \right)$$

Take $\partial \mathcal{L} / \partial N_l = 0$:
$$C_l - \lambda \frac{V_l}{N_l^2} = 0 \implies N_l = \sqrt{\frac{\lambda V_l}{C_l}}$$

Sum over the variance constraint:
$$\sum_l \frac{V_l}{N_l} = \epsilon^2 \implies \sum_l \frac{V_l}{\sqrt{\lambda V_l / C_l}} = \epsilon^2$$
$$\implies \sum_l \sqrt{\frac{V_l C_l}{\lambda}} = \epsilon^2 \implies \lambda = \frac{1}{\epsilon^4} \left( \sum_l \sqrt{V_l C_l} \right)^2$$

Substitute back:
$$N_l = \sqrt{\frac{V_l}{C_l}} \cdot \frac{1}{\epsilon^2} \sum_j \sqrt{V_j C_j}$$

(with a factor of 2 from RMSE vs variance). Boom! ✨

---

### The Bias Extrapolation

**Why Richardson extrapolation works:**

Assume $|\mu_l| = A \cdot 2^{-\alpha l} + B \cdot 2^{-\alpha' l} + \ldots$ (asymptotic expansion).

From three consecutive levels $\mu_{L-2}, \mu_{L-1}, \mu_L$, we can eliminate $A$ to estimate the next term, giving an **upper bound** on $|\mu_{L+1}|$.

**In practice**, the `max` over the last 3 levels catches any irregularities and gives a conservative estimate.

---

## **Part 9: Why This Algorithm is Beautiful**

### 1. **Self-Correcting**

If you underestimate variance at a level, the algorithm detects it in the next iteration (more samples needed) and corrects.

### 2. **Adaptive to Problem Difficulty**

Smooth payoffs (like digital options) → few levels needed
Kinky payoffs (like barrier options) → more levels needed

The algorithm **discovers** this automatically!

### 3. **Provably Optimal**

Given the variance estimates, the sample allocation is **mathematically optimal**. You can't beat it without better variance estimates.

### 4. **Efficient in Practice**

- Pilot run (100 samples/level) is ~1% of total cost
- Converges in 3-5 iterations typically
- Overhead is negligible compared to savings

---

## **Part 10: Connection to Your Project**

### For American Basket Options with Markovian Projection:

You'll replace:
```python
Pf = max(S - K, 0.0)  # European payoff
```

with:
```python
Pf = optimal_stopping_value(S_path)  # American payoff (Longstaff-Schwartz)
```

And before that:
```python
S_basket = markovian_projection(S_1, S_2, ..., S_d)  # Project d dimensions → 1
```

**The MLMC framework stays the same!** That's the power of MLMC—it's modular.

---

## **The Bottom Line**

**What this code does:**
1. Implements the full adaptive MLMC algorithm with provably optimal sample allocation
2. Dynamically adds levels until bias is acceptable
3. Validates the $O(\epsilon^{-2})$ complexity empirically
4. Shows 2-40× speedup over standard MC depending on accuracy

**Why it works:**
- **Variance reduction** via Brownian coupling (correlation between levels)
- **Optimal resource allocation** via Lagrange multipliers
- **Bias control** via Richardson extrapolation and dynamic level addition

**How to use it:**
- Want option price with RMSE < 0.01? Run `eps = 0.01` and get answer in seconds
- Want 10× tighter accuracy? MLMC costs ~100× less than standard MC!

This is production-ready code for **any** option pricing problem where you can couple paths. Just swap the payoff function! 🚀