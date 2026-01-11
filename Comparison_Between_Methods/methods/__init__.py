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


# Import cost metrics utilities
from .cost_metrics import (
    CostMetrics,
    CostProfiler,
    profile_function,
    count_function_calls,
    single_threaded,
    set_threading,
    compare_methods,
    print_comparison_table,
    add_cost_metrics_to_result,
)

__all__ = [
    # Estimation methods
    'estimate_volatility_mlmc',
    'estimate_volatility_laplace',
    # Data structures
    'VolatilitySurfaceResult',
    # Agreement metrics
    'compute_method_agreement',
    'format_metrics_table',
    'save_metrics_markdown',
    # Cost metrics
    'CostMetrics',
    'CostProfiler',
    'profile_function',
    'count_function_calls',
    'single_threaded',
    'set_threading',
    'compare_methods',
    'print_comparison_table',
    'add_cost_metrics_to_result',
]
