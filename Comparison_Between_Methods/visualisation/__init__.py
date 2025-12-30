"""
Visualisation Module for Comparison Framework

This module provides plotting utilities for comparing MLMC and Laplace
volatility estimation methods.

Key Functions
-------------
- plot_surface_comparison: Side-by-side 3D surface plots
- plot_difference_heatmap: Coloured heatmap of differences
- plot_convergence_study: MLMC convergence curves

Author: Wadoud (KAUST Internship)
"""

from .surface_plots import (
    plot_surface_comparison,
    plot_surface_3d,
    plot_combined_comparison,
)
from .error_plots import (
    plot_difference_heatmap,
    plot_pointwise_error,
)
from .convergence_plots import (
    plot_mlmc_convergence,
    plot_error_vs_time,
)

__all__ = [
    'plot_surface_comparison',
    'plot_surface_3d',
    'plot_combined_comparison',
    'plot_difference_heatmap',
    'plot_pointwise_error',
    'plot_mlmc_convergence',
    'plot_error_vs_time',
]
