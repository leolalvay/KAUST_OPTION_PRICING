# KAUST_OPTION_PRICING

---

## Project at a glance

The core objective is to reduce high-dimensional American basket option problems to lower-dimensional representations (Markovian projection), then evaluate numerical methods for estimating projected dynamics and pricing quantities.

Main themes:
- Markovian projection for dimensionality reduction
- MLMC estimators and sample allocation
- Polynomial/L2 regression (single-level and multi-level)
- PDE-based and comparative method studies
- Error and validation analysis

---

## Repository navigation 
> - **Local README**: whether that folder already contains deeper documentation

| Folder | Focus | Local README |
|---|---|---|
| `L2_Regression/` | Core regression workflows (simple polynomials, Legendre, single/multi-level), plus tutorial notebooks | ✅ (`L2_Regression/notebooks_seb/README.md`) |
| `Comparison_Between_Methods/` | MLMC vs Laplace comparison framework and experiments | ✅ |
| `MLMC/` | MLMC utilities/implementations and related scripts | ❌ |
| `PDE/` | PDE solvers and related plots/JAX variants | ❌ |
| `Laplace_Replication/` | Laplace-approximation related replication work | ❌ |
| `Error Analysis/` | Error-analysis material and reports | ❌ |
| `Estimating_Coefficients/` | Coefficient-estimation experiments/scripts | ❌ |
| `Sample_Allocation_MLMC_Investigation/` | Standalone MLMC sample allocation investigation | ✅ |
| `Transport_Map_Investigation/` | Transport-map related investigation area | ❌ |
| `Slides/` | Project reports and presentation material | ✅ |
| `THEORY Implied Stopping Rules for American Basket Options from Markovian Projection/` | Theory/tutorial notebook based on the reference paper | ✅ |
| `THEORY Multi-Level Monte Carlo Methods/` | Theory notes for MLMC methods | ❌ |

---

## Suggested reading path

If you are new to this repository:

1. Start from this root README (you are here)
2. Open `project_overview.md` for project context and outcomes
3. Go to `L2_Regression/notebooks_seb/` for guided notebooks
4. Move to `Comparison_Between_Methods/` for method-level benchmarking
5. Explore specialized folders (`MLMC/`, `PDE/`, investigations) as needed

---

## Documentation philosophy

This repository follows a **progressive documentation structure**:

- **Level 1 (root README):** global orientation and navigation
- **Level 2 (subfolder README):** folder-specific methods, setup, and usage
- **Level 3 (notebooks/scripts):** implementation and experiment details

So, as you navigate deeper, documentation becomes more detailed and technical.

---

## Top-level files (quick orientation)

- `project_overview.md` — extended project summary, methods, and outcomes
- `requirements.txt` — Python dependencies
- `timeline.png` — project timeline snapshot

---

## Notes

- Some folders intentionally do not yet include a local README.
- Existing subfolder READMEs should be treated as the authoritative source for folder-specific details.
- This root README is intentionally concise to avoid duplication.