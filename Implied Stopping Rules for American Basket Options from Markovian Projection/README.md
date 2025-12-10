# American Basket Options from Markovian Projection

## Tutorial Notebook Based on Bayer, Häppölä & Tempone (2017)

This repository contains a comprehensive Jupyter Notebook tutorial on pricing American basket options using Markovian projection techniques.

### 📁 Contents

- `American_Basket_Options_Markovian_Projection.ipynb` - Main tutorial notebook (✅ COMPLETE - 68 cells)
- `Implied Stopping Rules for American Basket Options from Markovian Projection.pdf` - Original paper
- `environment.yml` - Conda environment specification
- `README.md` - This file

### 🚀 Getting Started

#### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate KAUST
```

#### 2. Launch Jupyter

```bash
jupyter notebook American_Basket_Options_Markovian_Projection.ipynb
```

Or use JupyterLab:
```bash
jupyter lab
```

### 📚 Notebook Structure

The notebook is organised into three parts for each major concept:

#### Part A: Standard Rigorous Explanation
- Complete mathematical formulations
- LaTeX equations from the paper
- Theoretical foundations

#### Part B: Code Examples
- Working Python implementations
- Numerical demonstrations
- Visualisations

#### Part C: 🎯 Wadoud-Tailored Intuition
- Physics-based analogies
- Connections to computational physics
- Metaphors from statistical mechanics, quantum mechanics, and ML

### 📖 Current Progress

**Status**: ✅ **COMPLETE** (100% - All 9 sections finished!)

#### ✅ COMPLETED: All Sections (68 cells total)

**Sections 1-6: Core Mathematical Theory** (39 cells)

1. **Introduction & Paper Overview**
   - Complete mathematical setup and motivation

2. **Section 2: Mathematical Foundations**
   - 2.1: Black-Scholes & Bachelier Models
   - 2.2: European vs American Options
   - 2.3: Hamilton-Jacobi-Bellman Equation

3. **Section 3: Markovian Projection Theory**
   - 3.1: Curse of Dimensionality
   - 3.2: Gyöngy's Lemma (with verification code)
   - 3.3: Projected Volatility Calculation

4. **Section 4: Laplace Approximation**
   - High-dimensional integration techniques
   - Newton iteration for critical points
   - Computational complexity analysis

5. **Section 5: Bounds for Option Prices**
   - 5.1: Lower Bound via Stopping Rules
   - 5.2: Upper Bound via Rogers' Duality
   - Gap quantification

6. **Section 6: Dimension Reduction Theory**
   - 6.1: Exact Reduction for Bachelier (Lemma 2.7)
   - 6.2: Approximate Reduction for Black-Scholes (Corollary 2.14)

**Sections 7-9: Numerical Examples & Applications** (29 cells)

7. **Section 7: Numerical Examples** ✅
   - 7.1: Complete pricing algorithm implementation
   - 7.2: 3D Black-Scholes basket (reproducing paper results)
   - 7.3: High-dimensional scalability (10D, 25D with runtime analysis)
   - 7.4: Bachelier vs Black-Scholes comparison (exact vs approximate)

8. **Section 8: Interactive Widgets** ✅
   - 8.1: Real-time parameter explorer with ipywidgets
   - 8.2: 4-panel visualisation dashboard with sliders
   - Live computation of bounds, stopping boundaries, and sensitivities

9. **Section 9: Conclusions** ✅
   - 9.1: Complete method summary with results table
   - 9.2: Comparison with existing methods (PDE, LSM)
   - 9.3: Physics connections map (complete analogy table)
   - 9.4: Practical implications and implementation checklist
   - 9.5: Further reading and references

### 🎯 Key Features

- **UK English** throughout
- **Physics-based intuition** connecting to:
  - Statistical mechanics (Langevin equations, mean-field theory)
  - Quantum mechanics (measurement theory, variational principles)
  - Computational methods (Monte Carlo, finite differences)
  - Machine learning (reinforcement learning, dynamic programming)

- **Comprehensive coverage** of paper concepts:
  - Markovian projection (Gyöngy's lemma)
  - Laplace approximation
  - Lower/upper bounds
  - Dimension reduction
  - Numerical methods

- **Reproducible results** from paper:
  - All figures
  - All numerical examples
  - Error analysis

### 📊 Paper Summary

**Problem**: Pricing American basket options in high dimensions suffers from curse of dimensionality.

**Solution**: Project high-dimensional dynamics onto low-dimensional Markovian process, solve HJB equation in reduced space.

**Results**:
- Exact for Bachelier model
- ~Few percent error for Black-Scholes
- Feasible up to 50 dimensions

**Key Innovation**: No basis function selection (unlike least-squares Monte Carlo) + rigorous error bounds.

### 🔧 Dependencies

Core packages:
- Python 3.11
- NumPy, SciPy, Matplotlib, Pandas
- JAX (for future extensions)
- Numba (JIT compilation)
- Jupyter, IPyWidgets

See `environment.yml` for complete list.

### 📖 References

**Main Paper:**
```
Bayer, C., Häppölä, J., & Tempone, R. (2017).
Implied stopping rules for American basket options from Markovian projection.
arXiv preprint arXiv:1705.00558v4.
```

**Related Work:**
- Gyöngy, I. (1986). Mimicking the one-dimensional marginal distributions
- Rogers, L. C. (2002). Monte Carlo valuation of American options
- Longstaff & Schwartz (2001). Valuing American options by simulation

### 📝 Notes

- Notebook designed for both learning and research
- Code emphasises clarity over performance (NumPy/SciPy first, JAX alternatives provided)
- All equation numbers reference original paper
- Interactive widgets for parameter exploration

### 🤝 Contributions

This is an educational tutorial. Suggestions and improvements welcome!

### 📧 Contact

For questions about the implementation, please refer to the paper or create an issue.

---

**Status**: ✅ **COMPLETE** (All 9 sections: 100% done, 68 cells, ready to use!)
**Last Updated**: 2025-11-06

---

## 🎓 What You'll Learn

This comprehensive tutorial teaches you:

- ✅ **Theory**: Complete mathematical formulation from the paper
  - Stochastic differential equations and option pricing
  - Hamilton-Jacobi-Bellman equations for optimal stopping
  - Gyöngy's lemma and Markovian projection
  - Laplace approximation for high-dimensional integrals
  - Rogers' duality and rigorous error bounds

- ✅ **Implementation**: Working Python code for all concepts
  - SDE simulation (Euler-Maruyama)
  - PDE solving (finite differences)
  - Monte Carlo methods
  - Numerical examples from paper

- ✅ **Intuition**: Physics-based understanding
  - Connections to Langevin dynamics
  - Path integrals and quantum mechanics
  - Mean-field theory and coarse-graining
  - Variational principles

- ✅ **Practice**: Interactive tools
  - Real-time parameter exploration widgets
  - 3D to 25D numerical examples
  - Scalability demonstrations
