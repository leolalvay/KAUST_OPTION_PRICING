"""
Volatility Estimation Methods Module

This module provides unified wrappers for both volatility estimation methods:
1. MLMC (Multi-Level Monte Carlo) + polynomial regression
2. Laplace approximation (analytical)

Both methods return a standardised VolatilitySurfaceResult object for
consistent comparison.

Author: Wadoud (KAUST Internship)
"""

from .common import (
    VolatilitySurfaceResult,
    compute_accuracy_metrics,
    format_metrics_table,
    save_metrics_markdown,
)
from .mlmc_ot_estimator import estimate_volatility_mlmc
from .laplace_wrapper import estimate_volatility_laplace

__all__ = [
    'VolatilitySurfaceResult',
    'compute_accuracy_metrics',
    'format_metrics_table',
    'save_metrics_markdown',
    'estimate_volatility_mlmc',
    'estimate_volatility_laplace',
]
