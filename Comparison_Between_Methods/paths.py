"""
Path Configuration for the Comparison Module

This module adds the project root to the Python path to enable cross-folder
imports between PDE/, Laplace_Replication/, and Comparison/ modules.

Usage
-----
Import this module at the TOP of every script in the Comparison/ folder:

    import paths  # This modifies sys.path
    
    # Now imports work
    from PDE.mlmc_volatility_estimation import aggregate_mlmc_coefficients
    from Laplace_Replication.laplace_volatility import compute_volatility_surface

Notes
-----
This approach avoids code duplication and works immediately without
requiring package installation.

Author: Wadoud (KAUST Internship)
Based on: Bayer, Häppölä, Tempone (2017) comparison framework
"""

import sys
from pathlib import Path

# Get the project root (parent of Comparison/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add to Python path if not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Define key paths for convenience
PDE_PATH = PROJECT_ROOT / "PDE"
LAPLACE_PATH = PROJECT_ROOT / "Laplace_Replication"
COMPARISON_PATH = PROJECT_ROOT / "Comparison"


def verify_imports():
    """
    Test that all required modules can be imported.
    
    Returns
    -------
    bool
        True if all imports succeed, False otherwise.
    """
    success = True
    
    print("Verifying imports from project modules...")
    print(f"  Project root: {PROJECT_ROOT}")
    print()
    
    # Check PDE module imports
    print("PDE module:")
    try:
        from PDE.mlmc_volatility_estimation import aggregate_mlmc_coefficients
        print("  ✓ aggregate_mlmc_coefficients")
    except ImportError as e:
        print(f"  ✗ aggregate_mlmc_coefficients: {e}")
        success = False
    
    try:
        from PDE.mlmc_volatility_estimation import estimate_basket_domain
        print("  ✓ estimate_basket_domain")
    except ImportError as e:
        print(f"  ✗ estimate_basket_domain: {e}")
        success = False
    
    try:
        from PDE.basket_simulation import construct_volatility_surface
        print("  ✓ construct_volatility_surface")
    except ImportError as e:
        print(f"  ✗ construct_volatility_surface: {e}")
        success = False
    
    try:
        from PDE.basket_simulation import generate_polynomial_basis_pairs
        print("  ✓ generate_polynomial_basis_pairs")
    except ImportError as e:
        print(f"  ✗ generate_polynomial_basis_pairs: {e}")
        success = False
    
    try:
        from PDE.american_option_pde_solver import solve_american_option
        print("  ✓ solve_american_option")
    except ImportError as e:
        print(f"  ✗ solve_american_option: {e}")
        success = False
    
    print()
    
    # Check Laplace module imports
    print("Laplace_Replication module:")
    try:
        from Laplace_Replication.laplace_volatility import compute_volatility_surface
        print("  ✓ compute_volatility_surface")
    except ImportError as e:
        print(f"  ✗ compute_volatility_surface: {e}")
        success = False
    
    try:
        from Laplace_Replication.laplace_volatility import laplace_approximation_volatility_squared
        print("  ✓ laplace_approximation_volatility_squared")
    except ImportError as e:
        print(f"  ✗ laplace_approximation_volatility_squared: {e}")
        success = False
    
    try:
        from Laplace_Replication.pde_solver import solve_american_option_pde
        print("  ✓ solve_american_option_pde")
    except ImportError as e:
        print(f"  ✗ solve_american_option_pde: {e}")
        success = False
    
    print()
    
    if success:
        print("=" * 50)
        print("✓ All imports verified successfully!")
        print("=" * 50)
    else:
        print("=" * 50)
        print("✗ Some imports failed. Check module paths and __init__.py files.")
        print("=" * 50)
    
    return success


def get_results_path():
    """Return the path to the results directory, creating it if needed."""
    results_path = COMPARISON_PATH / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "figures").mkdir(exist_ok=True)
    (results_path / "tables").mkdir(exist_ok=True)
    return results_path


if __name__ == "__main__":
    verify_imports()
