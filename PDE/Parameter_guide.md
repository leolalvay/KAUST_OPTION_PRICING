# Parameter Guide: Controlling the American Basket Option Pricer

This guide explains **every parameter** you can control in `example_american_basket_pricing.py`, what it does, and how to change it effectively.

---

## 🎯 Quick Navigation

1. **Basket Configuration** (lines 40-49)
2. **Option Parameters** (lines 52-54)
3. **MLMC Parameters** (lines 57-58)
4. **PDE Solver Parameters** (lines 61-62)
5. **Domain Control** (manual override)

---

## 1. Basket Configuration Parameters

### `d` - Number of Assets (line 40)
**What it does**: Sets how many individual stocks/assets are in your basket.

**Current**: `d = 3`

**Effects**:
- **Computational cost**: Scales as O(d²) due to correlation matrix operations
- **Dimensionality**: Higher d makes Markovian projection more valuable (more reduction)
- **Typical values**: 2-10 (above 10 becomes very slow)

**Example changes**:
```python
d = 2   # Simple two-asset basket (fastest)
d = 5   # More realistic multi-asset basket
d = 10  # High-dimensional (slow, but shows MLMC power)
```

### `S0` - Initial Asset Prices (line 41)
**What it does**: Starting prices for each asset at t=0.

**Current**: `S0 = np.linspace(225, 275, num=d)` → [225, 250, 275] for d=3

**Effects**:
- **Scale**: Larger S0 → larger basket price → wider domain needed
- **Domain location**: Centers the basket around weighted average
- **Exercise boundary**: Affects where optimal exercise occurs

**Example changes**:
```python
S0 = np.array([100, 100, 100])[:, np.newaxis]  # Equal prices at 100
S0 = np.array([50, 200, 350])[:, np.newaxis]   # Very different assets
S0 = np.linspace(180, 220, num=d)[:, np.newaxis]  # Tighter range
```

**⚠️ Important**: Must reshape to column vector with `[:, np.newaxis]`

### `basket_weights` - Asset Weights (line 42)
**What it does**: How much of each asset is in the basket. Must sum to 1.

**Current**: `basket_weights = np.ones(d) / d` → Equal weights [1/3, 1/3, 1/3]

**Effects**:
- **Basket price**: $B = \sum_i w_i S_i$
- **Volatility**: Different weights change effective basket volatility
- **Skew**: Unequal weights can create asymmetric payoffs

**Example changes**:
```python
basket_weights = np.array([0.5, 0.3, 0.2])  # 50% in asset 1, 30% in 2, 20% in 3
basket_weights = np.array([0.6, 0.4])       # For d=2, heavy on first asset
basket_weights = np.array([0.1, 0.1, 0.8])  # Concentrated in asset 3
```

**Rule**: Must satisfy `np.sum(basket_weights) == 1.0`

### `vol` - Asset Volatilities (line 44)
**What it does**: Annual volatility (standard deviation) for each asset.

**Current**: `vol = np.array([0.2, 0.15, 0.1])` → 20%, 15%, 10%

**Effects**:
- **Option value**: Higher vol → higher option value (more optionality)
- **Basket vol**: Combined effect through correlation matrix
- **Time value**: Drives how much American > intrinsic

**Example changes**:
```python
vol = np.array([0.3, 0.3, 0.3])      # All highly volatile (tech stocks)
vol = np.array([0.1, 0.1, 0.1])      # All low vol (utilities)
vol = np.array([0.5, 0.2, 0.05])     # Mixed (crypto, equity, bond)
```

**Typical ranges**:
- Bonds: 0.02-0.05
- Equities: 0.15-0.30
- Tech stocks: 0.30-0.50
- Crypto: 0.50-1.50

---

## 2. Domain Control - **FIX FOR EXERCISE BOUNDARY!**

### The Problem
Your exercise boundary is flat at S_min because the **true boundary is below your computational domain**.

### The Solution: Manual Domain Override

Add these lines after line 62 in the example file:

```python
# =================================================================
# DOMAIN CONTROL - Set wider domain to capture exercise boundary
# =================================================================
USE_MANUAL_DOMAIN = True  # Set to False for automatic domain
S_min_override = 150      # Well below basket to capture boundary
S_max_override = 350      # Well above basket
```

Then modify the domain estimation section (around lines 77-86) to:

```python
print(f"\n{'-'*70}")
print("STEP 1: Domain Configuration")
print(f"{'-'*70}\n")

if USE_MANUAL_DOMAIN:
    S_min = S_min_override
    S_max = S_max_override
    print(f"Using MANUAL domain: [{S_min:.2f}, {S_max:.2f}]")
    print(f"  (Set to capture exercise boundary)")
else:
    S_min, S_max, basket_paths = estimate_basket_domain(
        S0, T, h0, r, cov_mat, vol, max_degree, basket_weights, N_pilot=10000
    )
    print(f"Using AUTOMATIC domain: [{S_min:.2f}, {S_max:.2f}]")
    print(f"  (From 1-99 percentiles of pilot paths)")
```

### Where Should the Boundary Be?

For American puts, approximate formula:
$$S^* \approx K \times \frac{r}{r + \sigma^2/2}$$

With your parameters (K=250, r=0.05, σ≈0.12):
$$S^* \approx 250 \times \frac{0.05}{0.05 + 0.0072} \approx 219$$

So the boundary is around **S≈220**, but your automatic domain started at S_min≈210!

**Setting S_min=150 will capture the boundary** ✓

---

## 3. Quick Parameter Combinations

### See Exercise Boundary (RECOMMENDED)
```python
USE_MANUAL_DOMAIN = True
S_min_override = 150
S_max_override = 350
N_spatial = 80  # More points for smoother boundary
```

### Fast Test (< 30 sec)
```python
d = 2
max_degree = 1
N_timesteps = 50
h0 = 0.1
```

### Publication Quality (5-10 min)
```python
d = 3
max_degree = 4
N_timesteps = 500
N_spatial = 100
h0 = 0.025
```

---

## 4. All Parameters Summary

| Parameter | Line | Type | Current | Range | Effect | Runtime Impact |
|-----------|------|------|---------|-------|--------|----------------|
| **Basket** |
| `d` | 40 | int | 3 | 2-10 | Assets in basket | O(d²) |
| `S0` | 41 | array | [225,250,275] | >0 | Initial prices | Minimal |
| `basket_weights` | 42 | array | [1/3,1/3,1/3] | Sum=1 | Basket composition | Minimal |
| `r` | 43 | float | 0.05 | 0-0.15 | Risk-free rate | Minimal |
| `vol` | 44 | array | [0.2,0.15,0.1] | 0.05-0.50 | Asset volatilities | Minimal |
| `cov_mat` | 45-49 | matrix | Mixed | [-1,1] | Correlations | Minimal |
| **Option** |
| `T` | 52 | float | 1.0 | 0.1-5.0 | Time to maturity | Linear |
| `K` | 53 | float | 250 | ~B₀ | Strike price | Minimal |
| `option_type` | 54 | str | "put" | put/call | Option type | Minimal |
| **MLMC** |
| `h0` | 57 | float | 0.05 | 0.01-0.2 | Coarsest timestep | O(1/h₀) |
| `max_degree` | 58 | int | 3 | 1-5 | MLMC levels | O(2^L) |
| **PDE** |
| `N_timesteps` | 61 | int | 200 | 50-1000 | Time grid points | Linear |
| `N_spatial` | 62 | int | 50 | 30-200 | Space grid points | Linear |

---

## 5. Recommended Experiments

### Experiment 1: See the Exercise Boundary
```python
USE_MANUAL_DOMAIN = True
S_min_override = 150  # Below expected boundary
S_max_override = 350
N_spatial = 100       # Fine grid for smooth boundary
```
**Expected**: Boundary visible around S≈220, moving toward K as t→T

### Experiment 2: Volatility Impact
```python
# Run three times with different vols:
vol = np.array([0.1, 0.1, 0.1])    # Low vol case
vol = np.array([0.2, 0.15, 0.1])   # Base case
vol = np.array([0.3, 0.3, 0.3])    # High vol case
```
**Expected**: Higher vol → higher option values, more time value

### Experiment 3: Correlation Effect
```python
# Run three times:
cov_mat = np.eye(d)                    # No correlation
cov_mat = current_matrix               # Base case
cov_mat = np.ones((d,d))*0.9 + np.eye(d)*0.1  # High correlation
```
**Expected**: Higher correlation → higher basket vol → higher option value

### Experiment 4: Moneyness Study
```python
# Run with K in [225, 237.5, 250, 262.5, 275]
```
**Expected**: ITM puts have high intrinsic, low time value %; OTM opposite

---

Want me to create a modified `example_american_basket_pricing.py` file with manual domain control built in so you can immediately see the exercise boundary?
