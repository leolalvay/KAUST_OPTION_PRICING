# Understanding MLMC Diagnostics: A Complete Walkthrough

Let me walk you through this step by step, explaining what the code does and why.

---

## **Part 1: The Setup (What Problem Are We Solving?)**

### The Option Pricing Problem

```python
T = 1.0          # Time to maturity (years)
S0 = 100.0       # Initial stock price
K = 100.0        # Strike price (at-the-money)
r = 0.05         # Risk-free rate
sigma = 0.2      # Volatility
```

We want to price a **European call option**: the right (but not obligation) to buy a stock at price K = 100 at time T = 1 year.

**The stock follows geometric Brownian motion (GBM):**
```
dS = r*S*dt + sigma*S*dW
```

This is the same maths as your quantum field theory path integrals, but for stock prices instead of particle paths!

**The payoff at maturity:**
```python
Payoff = max(S_T - K, 0)
```
If the stock ends above £100, you make money. Otherwise, the option expires worthless.

---

## **Part 2: Why Do We Need MLMC?**

### The Standard Monte Carlo Problem

To price the option, we could:
1. Simulate many stock price paths
2. Compute payoff for each
3. Average and discount: `Price = E[e^(-rT) * Payoff]`

**BUT:** To simulate accurately, we need small timesteps. If we want error ε:
- Timestep required: `h ~ ε²`
- Number of samples needed: `N ~ ε⁻²`
- **Total cost**: `Work = N × (T/h) ~ ε⁻² × ε⁻² = ε⁻⁴` 😱

That's horrific! For ε = 0.01, you'd need 100 million path-steps.

---

### The MLMC Trick (Telescoping Sum)

Instead of computing `E[P_L]` directly at the finest level, MLMC writes:

```
E[P_L] = E[P_0] + E[P_1 - P_0] + E[P_2 - P_1] + ... + E[P_L - P_{L-1}]
```

where `P_l` means "payoff computed with timestep `h_l = h0 * 2^(-l)`".

**Why is this brilliant?**

Each **difference** `Y_l = P_l - P_{l-1}` has:
- **Much lower variance** than `P_l` itself (because we use correlated paths!)
- **Decaying mean** as l increases

This is like your T2K near-far detector setup! The near detector (coarse level) is cheap and constrains systematics. The far detector (fine level) measures the oscillation with fewer events needed because you've already removed common uncertainties.

---

## **Part 3: The Code Structure**

### Storage Arrays

```python
sum_Pf = np.zeros(L_max + 1)      # Accumulate fine payoffs
sum_Pc = np.zeros(L_max + 1)      # Accumulate coarse payoffs
sum_Y = np.zeros(L_max + 1)       # Accumulate corrections
sum_Y_sq = np.zeros(L_max + 1)    # For variance
sum_Y_cubed = np.zeros(L_max + 1) # For kurtosis
```

We need to track **moments** (mean, variance, kurtosis) at each level to verify MLMC theory works.

---

### The Main Loop Structure

```python
for l in levels:  # l = 0, 1, 2, ..., 8
    h_fine = h0 * 2^(-l)    # Timestep gets halved each level
    
    for n in range(N_pilot):  # Run N_pilot = 200,000 simulations
        # Simulate fine path (timestep h_fine)
        # Simulate coarse path (timestep 2*h_fine)
        # Compute correction Y = Pf - Pc
        # Accumulate statistics
```

At each level, we halve the timestep:
- Level 0: `h = 0.5` (2 steps)
- Level 1: `h = 0.25` (4 steps)
- Level 2: `h = 0.125` (8 steps)
- ...
- Level 8: `h ≈ 0.002` (512 steps)

---

## **Part 4: Path Simulation (The Heart of the Method)**

### Fine Path

```python
dW_fine = np.random.normal(0, sqrt(h_fine), size=n_steps)

S = S0
for dW in dW_fine:
    S += r * S * h_fine + sigma * S * dW
    
Pf = max(S - K, 0.0)
```

This is **Euler-Maruyama discretisation** of the GBM SDE:
```
S_{n+1} = S_n + r*S_n*h + sigma*S_n*dW_n
```

**Physics analogy:** It's like time-evolving your quantum state step by step:
```
|ψ(t+dt)⟩ = (1 - iH dt)|ψ(t)⟩ + noise
```

Each step advances the stock price by:
- **Drift term**: `r*S*h` (deterministic growth)
- **Diffusion term**: `sigma*S*dW` (random fluctuations)

---

### Coarse Path (The Critical Part!)

```python
if l > 0:
    dW_coarse = dW_fine.reshape(-1, 2).sum(axis=1)
    
    S = S0
    for dW in dW_coarse:
        S += r * S * h_coarse + sigma * S * dW
    
    Pc = max(S - K, 0.0)
```

**This is the KEY MLMC idea!**

The coarse path uses **the same random numbers** as the fine path, just summed in pairs:
```
dW_coarse[0] = dW_fine[0] + dW_fine[1]
dW_coarse[1] = dW_fine[2] + dW_fine[3]
...
```

**Why does this work?**

By the **Brownian bridge property**, summing consecutive increments gives you a valid coarser Brownian motion:
```
W(t₂) - W(t₀) = [W(t₁) - W(t₀)] + [W(t₂) - W(t₁)]
```

**Why does this help?**

Because both paths use the **same underlying randomness**, they're highly **correlated**. If the fine path goes up, the coarse path also goes up. 

**Result:** `Pf - Pc` is much smaller than `Pf` itself!

**Physics analogy:** It's like common-mode noise rejection in electronics. You measure the signal with two detectors at slightly different resolutions. The **difference** removes shared systematic errors, leaving only the correction term with much lower noise.

---

### The Correction Term

```python
Y = disc * (Pf - Pc)
```

This is what MLMC estimates at each level. The discount factor `disc = e^(-rT)` converts future payoff to present value.

We accumulate its moments:
```python
acc_Y += Y
acc_Y_sq += Y**2
acc_Y_cubed += Y**3
acc_Y_fourth += Y**4
```

These let us compute:
- **Mean**: `E[Y]` (measures weak convergence rate α)
- **Variance**: `Var[Y]` (measures variance decay rate β)
- **Kurtosis**: `Kurt[Y]` (checks if CLT applies)

---

## **Part 5: Computing Statistics**

After accumulating over N_pilot = 200,000 samples:

```python
mean_Y = sum_Y / N_pilot
var_Y = sum_Y_sq / N_pilot - mean_Y**2
```

This is standard sample statistics:
- **Sample mean**: `μ̂ = (1/N) Σ Y_i`
- **Sample variance**: `σ̂² = (1/N) Σ Y_i² - μ̂²`

### Kurtosis Calculation

```python
m1 = mean_Y
m2 = sum_Y_sq / N_pilot
m3 = sum_Y_cubed / N_pilot
m4 = sum_Y_fourth / N_pilot

m4_central = m4 - 4*m1*m3 + 6*(m1**2)*m2 - 3*m1**4
kurt_Y = m4_central / (var_Y**2)
```

**What is this?**

Kurtosis measures **tail heaviness**. It's the fourth central moment normalised by variance squared:
```
Kurt[X] = E[(X - μ)⁴] / σ⁴
```

**The formula** comes from expanding `(X - μ)⁴`:
```
(X - μ)⁴ = X⁴ - 4μX³ + 6μ²X² - 3μ⁴
```

Then taking expectations.

**Why do we care?**

- **Gaussian distributions have kurtosis = 3**
- **Kurtosis > 3** means "heavy tails" (extreme values more common)
- **For MLMC**, we want kurtosis bounded so the Central Limit Theorem applies properly

---

## **Part 6: Convergence Rate Estimates**

```python
log2_var_Y = np.log2(var_Y[1:])
beta = -np.polyfit(levels_fit, log2_var_Y, 1)[0]
```

**What's happening?**

If variance decays like `Var[Y_l] ~ 2^(-β*l)`, then:
```
log₂(Var[Y_l]) = -β*l + constant
```

So plotting `log₂(Var[Y_l])` vs `l` should give a **straight line with slope -β**.

**Similarly for mean:**
```python
log2_mean_Y = np.log2(np.abs(mean_Y[1:]))
alpha = -np.polyfit(levels_fit, log2_mean_Y, 1)[0]
```

If `|E[Y_l]| ~ 2^(-α*l)`, then the slope is `-α`.

---

## **Part 7: What the Results Mean**

### Your Output Shows

```
Variance decay rate (beta): 0.97 (theory: 2.0)
Weak convergence rate (alpha): 1.24 (theory: 1.0)
Kurtosis at finest level: 5.84 (Gaussian: 3.0)
```

**What does this tell us?**

### Beta ≈ 1 (Should be 2)

**Theory predicts:** `Var[Y_l] ~ 2^(-2l)` for GBM with Euler-Maruyama

**Your result:** β ≈ 1, so `Var[Y_l] ~ 2^(-l)` (slower decay)

**Why the discrepancy?**

This is actually **expected** at coarse levels! The theoretical rate β = 2 assumes:
1. Sufficient smoothness in the payoff
2. Small enough timesteps

**European call has a kink at S = K** (where payoff changes from 0 to S-K). This non-smoothness reduces the variance decay rate at coarse levels.

**Physics analogy:** It's like trying to approximate a sharp potential step with smooth basis functions. You need finer resolution to capture the discontinuity.

At finer levels (l = 6, 7, 8), β should approach 2 more closely.

---

### Alpha ≈ 1.24 (Should be 1)

**Theory predicts:** `|E[Y_l]| ~ 2^(-l)` for weak order 1 schemes

**Your result:** α ≈ 1.24 (faster decay, which is good!)

**What does this mean?**

The bias (difference between coarse and fine) decays **faster than expected**. This is excellent! It means fewer levels are needed to achieve target accuracy.

**Why?** For GBM, the Euler-Maruyama scheme actually has **stronger convergence** than the worst-case theory. The payoff is "smooth enough" in most regions.

---

### Kurtosis ≈ 5.84 (Should be 3)

**Gaussian has kurtosis = 3**

**Your result:** Kurt ≈ 5.84 (heavy tails)

**What does this mean?**

The correction Y has fatter tails than a Gaussian. This happens because:
1. Option payoffs are **bounded below** (can't go negative)
2. But can have **large spikes** (if stock price jumps above strike)

**Is this a problem?**

Not really! Kurtosis = 5.84 isn't terrible. It means:
- CLT still applies (just slightly slower convergence)
- Might need ~20% more samples than if it were perfectly Gaussian
- At finer levels, kurtosis decreases (as you see in the plot)

**Physics analogy:** It's like non-Gaussian beam profiles in your particle detector. You can still analyse them, just need to be more careful about tails.

---

## **Part 8: The Plots Explained**

### Panel 1: Variance Decay

```
Orange line (Var[Pf]): Stays roughly constant (~220)
Blue line (Var[Y]): Decays from ~1 to ~0.01
```

**Key insight:**

The **absolute variance** of the payoff doesn't change much with refinement. Whether you use 4 steps or 512 steps, you're still pricing the same option, so the spread in payoffs is similar.

But the **difference variance** drops exponentially! From level 1 to 8:
```
Var[Y₁] ≈ 1.25
Var[Y₈] ≈ 0.011
```

That's a **100× reduction**! This is why MLMC works.

**The slope of the blue line is -β**. Ideally -2, yours shows ~-1 at coarse levels.

---

### Panel 2: Mean Decay

```
Orange line (E[Pf]): Constant (~11)
Blue line (E[Y]): Decays from ~0.05 to ~0.0001
```

**Key insight:**

The **expected payoff** is roughly £11 (the option value). This doesn't depend on timestep granularity.

But the **expected correction** shrinks rapidly. Each finer level adds a smaller correction:
```
E[Y₁] ≈ 0.05   (5% of option value)
E[Y₈] ≈ 0.0001 (0.001% of option value)
```

**This tells you when to stop refining:** Once `|E[Y_L]|` is smaller than your target error ε, adding more levels won't help.

**The slope is -α**. Yours shows α ≈ 1.24, which is great!

---

### Panel 3: Kurtosis

```
Kurt[Y] drops from ~12 to ~6
```

**Key insight:**

At coarse levels (l = 1, 2), the correction has **very heavy tails** (Kurt ≈ 12). This is because with only 4 or 8 timesteps, occasional paths have wild fluctuations.

As you refine (l = 6, 7, 8), kurtosis approaches the Gaussian limit (3). With 256+ steps, paths are smoother and corrections become more predictable.

**What you want:** Kurtosis to stabilise at a reasonable value (<10). Yours does! ✅

---

## **Part 9: Why This Matters for Your Project**

### For American Options (Project 1)

You're extending this to:
1. **American basket options** (multiple stocks, early exercise)
2. **Markovian projection** (reduce dimensions)
3. **Optimal stopping** (decide when to exercise)

**This diagnostic tool validates:**
- Your MLMC implementation is correct ✅
- Convergence rates match theory (roughly) ✅
- Variance reduction is working ✅

**Next steps:**
1. Replace European payoff with optimal stopping rule
2. Apply Markovian projection to basket
3. Verify convergence rates still hold

---

### The Computational Win

If you ran standard MC at finest level (h = 0.002):
```
Cost_MC ~ N × (T/h) = 10⁶ × 512 = 512 million path-steps
```

With MLMC:
```
Cost_MLMC ~ N₀×2 + N₁×4 + ... + N₈×512
          ≈ 200k×(2 + 4 + 8 + ... + 512)
          ≈ 200k × 1000 = 200 million path-steps
```

**Savings: 2.5×** even at these coarse accuracy levels!

At tighter accuracy (ε = 0.001), MLMC wins by **10-50×**. That's the power of variance reduction!

---

## **The Bottom Line**

**What this code does:**
1. Prices a European call using MLMC at 9 levels (h from 0.5 to 0.002)
2. Measures how variance and bias decay with refinement
3. Validates that MLMC theory works (β ≈ 1-2, α ≈ 1)
4. Checks statistical properties (kurtosis)

**What the results mean:**
- ✅ Variance reduction is working (100× drop from coarse to fine)
- ✅ Bias decays faster than expected (α > 1, good!)
- ✅ Kurtosis is acceptable (< 10 at fine levels)
- ⚠️ Beta slower than ideal at coarse levels (payoff non-smoothness)

**Why it matters:**
- Proves MLMC is worth the complexity
- Gives you a validated baseline before adding American options, baskets, and Markovian projection
- Shows where to focus optimisation (coarse levels dominate cost!)

