# notebooks_seb — Reader Guide

This folder contains **guided notebooks** that explain and reproduce the main workflows in `L2_Regression` in a more readable, step-by-step format.

The purpose is to help readers understand the codebase by showing:
- what each method does,
- how it is implemented,
- and how results are validated (e.g., weak errors, volatility surfaces).

---

## Notebook → Codebase mapping (most important)

| Notebook | Main idea | Based on code/folders |
|---|---|---|
| `nb01_L2Regression_simple.ipynb` | Introductory \(L^2\)-regression intuition | `L2_Regression/Rough_programs/DM_org_refactored.py` |
| `nb02_L2Regression_single_level.ipynb` | Single-level regression workflow | `L2_Regression/Rough_programs/DesignMatrix_refactored.py` |
| `nb03_L2Regression_simple_polynomials.ipynb` | Simple-polynomial basis workflow (extended) | `L2_Regression/Simple_Polynomials` |
| `nb04_Legendre_single_level.ipynb` | Legendre **single-level** implementation | `L2_Regression/Legendre_Polynomials/Single_Level` |
| `nb05_Legendre_Multi_Level.ipynb` | Legendre **multi-level** implementation | `L2_Regression/Legendre_Polynomials/Multi_Level` (especially `Original_Programs`, with references to `Fast_Programs`) |

---

## Suggested reading order

If you are new to this folder:

1. `nb01` (quick intuition)  
2. `nb02` → `nb03` (single-level, simple polynomial basis)  
3. `nb04` (single-level Legendre)  
4. `nb05` (multi-level Legendre)

---

## Output files in this folder

Some notebooks save artifacts for analysis and reproducibility:
- CSV results (e.g., weak errors),
- figures (`.png`),
- PDF plots/reports (`.pdf`).

These outputs let you inspect results without rerunning every experiment.