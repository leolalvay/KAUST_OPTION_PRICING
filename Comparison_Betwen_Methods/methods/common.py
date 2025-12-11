"""
Common Data Structures for Volatility Estimation Methods

This module defines the standardised return type for both MLMC and Laplace
volatility estimation methods, ensuring consistent comparison.

Author: Wadoud (KAUST Internship)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any


@dataclass
class VolatilitySurfaceResult:
    """
    Standardised result type for volatility estimation methods.
    
    Both MLMC and Laplace methods return this object to enable
    consistent comparison of their outputs.
    
    Attributes
    ----------
    b_surface : callable
        Function b(t, s) -> float that evaluates the volatility surface.
        t is time, s is basket value.
    b_squared_surface : callable
        Function b²(t, s) -> float for direct comparison.
        Note: Laplace naturally computes b², MLMC computes b.
    t_grid : np.ndarray
        Time grid points where surface was evaluated.
    s_grid : np.ndarray
        Basket value grid points where surface was evaluated.
    b_squared_values : np.ndarray
        Pre-computed b² values on the grid, shape (len(t_grid), len(s_grid)).
    method_name : str
        Either "MLMC" or "Laplace".
    computation_time : float
        Wall-clock time in seconds.
    parameters : dict
        All input parameters for reproducibility.
    coefficients : np.ndarray or None
        MLMC polynomial coefficients (None for Laplace).
    n_samples : int or None
        Number of Monte Carlo samples used (None for Laplace).
    mlmc_levels : int or None
        Number of MLMC levels (None for Laplace).
    domain_bounds : tuple or None
        (S_min, S_max) domain bounds used.
    """
    
    # Core surface representations
    b_surface: Callable[[float, float], float]
    b_squared_surface: Callable[[float, float], float]
    
    # Grid data for plotting
    t_grid: np.ndarray
    s_grid: np.ndarray
    b_squared_values: np.ndarray
    
    # Metadata
    method_name: str
    computation_time: float
    parameters: Dict[str, Any]
    
    # MLMC-specific (None for Laplace)
    coefficients: Optional[np.ndarray] = None
    n_samples: Optional[int] = None
    mlmc_levels: Optional[int] = None
    
    # Shared domain information
    domain_bounds: Optional[tuple] = None
    
    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"VolatilitySurfaceResult("
            f"method={self.method_name}, "
            f"grid=({len(self.t_grid)}, {len(self.s_grid)}), "
            f"time={self.computation_time:.2f}s)"
        )
    
    def summary(self) -> str:
        """Detailed summary of the result."""
        lines = [
            f"{'=' * 50}",
            f"Volatility Surface Result: {self.method_name}",
            f"{'=' * 50}",
            f"  Grid size: {len(self.t_grid)} × {len(self.s_grid)}",
            f"  t range: [{self.t_grid.min():.3f}, {self.t_grid.max():.3f}]",
            f"  s range: [{self.s_grid.min():.1f}, {self.s_grid.max():.1f}]",
            f"  Computation time: {self.computation_time:.3f} s",
            "",
            f"  b² statistics:",
            f"    Min: {np.nanmin(self.b_squared_values):.2f}",
            f"    Max: {np.nanmax(self.b_squared_values):.2f}",
            f"    Mean: {np.nanmean(self.b_squared_values):.2f}",
        ]
        
        if self.mlmc_levels is not None:
            lines.append(f"  MLMC levels: {self.mlmc_levels}")
        
        if self.domain_bounds is not None:
            lines.append(f"  Domain: [{self.domain_bounds[0]:.1f}, {self.domain_bounds[1]:.1f}]")
        
        lines.append(f"{'=' * 50}")
        return "\n".join(lines)


def compute_accuracy_metrics(
    b2_mlmc: np.ndarray,
    b2_laplace: np.ndarray
) -> Dict[str, Any]:
    """
    Compute accuracy metrics comparing two volatility surfaces.
    
    Neither method is treated as "ground truth" - we report symmetric
    measures of agreement/disagreement.
    
    Parameters
    ----------
    b2_mlmc : np.ndarray
        b² values from MLMC estimation.
    b2_laplace : np.ndarray
        b² values from Laplace approximation.
        
    Returns
    -------
    dict
        Dictionary containing:
        - l2_relative_error: ||b²_MLMC - b²_Laplace||_2 / ||b²_mean||_2
        - linf_error: max|b²_MLMC - b²_Laplace|
        - linf_relative_error: max relative error
        - mean_absolute_difference: mean|b²_MLMC - b²_Laplace|
        - mean_relative_difference: mean(2|diff| / |sum|) (symmetric)
        - bias: mean(b²_MLMC - b²_Laplace)
        - correlation: Pearson correlation coefficient
        - pointwise_relative_error: 2D array for heatmap
        - valid_fraction: fraction of grid points with finite values
        
    Notes
    -----
    Uses symmetric relative error 2|a-b|/(|a|+|b|) to avoid bias towards
    either method.
    """
    # Ensure numpy arrays
    b2_mlmc = np.asarray(b2_mlmc)
    b2_laplace = np.asarray(b2_laplace)
    
    # Handle shape mismatch by interpolation if needed
    if b2_mlmc.shape != b2_laplace.shape:
        raise ValueError(
            f"Shape mismatch: MLMC {b2_mlmc.shape} vs Laplace {b2_laplace.shape}. "
            "Both methods should use the same grid."
        )
    
    # Mask for valid (finite, positive) values
    valid = (
        np.isfinite(b2_mlmc) & 
        np.isfinite(b2_laplace) & 
        (b2_mlmc > 0) & 
        (b2_laplace > 0)
    )
    
    if not np.any(valid):
        return {
            "l2_relative_error": np.inf,
            "linf_error": np.inf,
            "mean_absolute_difference": np.inf,
            "mean_relative_difference": np.inf,
            "bias": np.nan,
            "correlation": np.nan,
            "pointwise_relative_error": np.full_like(b2_mlmc, np.nan),
            "valid_fraction": 0.0,
        }
    
    # Extract valid values
    mlmc_valid = b2_mlmc[valid]
    laplace_valid = b2_laplace[valid]
    diff = mlmc_valid - laplace_valid
    
    # L2 relative error (normalised by mean of both)
    mean_surface = (mlmc_valid + laplace_valid) / 2
    l2_error = np.sqrt(np.mean(diff ** 2))
    l2_norm = np.sqrt(np.mean(mean_surface ** 2))
    l2_relative = l2_error / l2_norm if l2_norm > 0 else np.inf
    
    # L-infinity error
    linf_error = np.max(np.abs(diff))
    
    # L-infinity relative error
    linf_relative = linf_error / l2_norm if l2_norm > 0 else np.inf
    
    # Mean absolute difference
    mean_abs_diff = np.mean(np.abs(diff))
    
    # Symmetric mean relative difference: 2|a-b|/(|a|+|b|)
    # This avoids bias towards either method
    sum_abs = np.abs(mlmc_valid) + np.abs(laplace_valid)
    sym_rel_diff = np.where(
        sum_abs > 0,
        2 * np.abs(diff) / sum_abs,
        0.0
    )
    mean_relative = np.mean(sym_rel_diff)
    
    # Bias (positive means MLMC > Laplace on average)
    bias = np.mean(diff)
    
    # Correlation
    if len(mlmc_valid) > 1:
        correlation = np.corrcoef(mlmc_valid, laplace_valid)[0, 1]
    else:
        correlation = np.nan
    
    # Pointwise symmetric relative error for heatmap
    pointwise = np.full_like(b2_mlmc, np.nan)
    sum_abs_full = np.abs(b2_mlmc) + np.abs(b2_laplace)
    np.divide(
        2 * np.abs(b2_mlmc - b2_laplace),
        sum_abs_full,
        out=pointwise,
        where=valid & (sum_abs_full > 0)
    )
    
    # Valid fraction
    valid_fraction = np.mean(valid)
    
    return {
        "l2_relative_error": l2_relative,
        "linf_error": linf_error,
        "linf_relative_error": linf_relative,
        "mean_absolute_difference": mean_abs_diff,
        "mean_relative_difference": mean_relative,
        "bias": bias,
        "correlation": correlation,
        "pointwise_relative_error": pointwise,
        "valid_fraction": valid_fraction,
    }


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format accuracy metrics as a readable table.
    
    Parameters
    ----------
    metrics : dict
        Output from compute_accuracy_metrics().
        
    Returns
    -------
    str
        Formatted table string.
    """
    lines = [
        "Comparison Metrics (MLMC vs Laplace)",
        "=" * 45,
        f"  L2 relative error:      {metrics['l2_relative_error']:.4f}",
        f"  L∞ error:               {metrics['linf_error']:.2f}",
        f"  Mean absolute diff:     {metrics['mean_absolute_difference']:.2f}",
        f"  Mean relative diff:     {metrics['mean_relative_difference']:.4f} ({metrics['mean_relative_difference']*100:.2f}%)",
        f"  Bias (MLMC - Laplace):  {metrics['bias']:.2f}",
        f"  Correlation:            {metrics['correlation']:.4f}",
        f"  Valid grid fraction:    {metrics['valid_fraction']:.2f} ({metrics['valid_fraction']*100:.1f}%)",
        "=" * 45
    ]
    return "\n".join(lines)


def save_metrics_markdown(
    metrics: Dict[str, Any], 
    filepath: str,
    params_summary: str = ""
) -> None:
    """
    Save metrics to a markdown file.
    
    Parameters
    ----------
    metrics : dict
        Output from compute_accuracy_metrics().
    filepath : str
        Path to save the markdown file.
    params_summary : str
        Optional parameter summary to include.
    """
    content = [
        "# Comparison Metrics: MLMC vs Laplace Approximation",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| L2 relative error | {metrics['l2_relative_error']:.6f} |",
        f"| L∞ error | {metrics['linf_error']:.4f} |",
        f"| Mean absolute difference | {metrics['mean_absolute_difference']:.4f} |",
        f"| Mean relative difference | {metrics['mean_relative_difference']*100:.2f}% |",
        f"| Bias (MLMC - Laplace) | {metrics['bias']:.4f} |",
        f"| Pearson correlation | {metrics['correlation']:.4f} |",
        f"| Valid grid fraction | {metrics['valid_fraction']*100:.1f}% |",
        "",
    ]
    
    if params_summary:
        content.extend([
            "## Parameters",
            "",
            "```",
            params_summary,
            "```",
            ""
        ])
    
    content.extend([
        "## Notes",
        "",
        "- Neither method is treated as ground truth",
        "- Relative differences use symmetric formula: 2|a-b|/(|a|+|b|)",
        "- L2 error normalised by mean surface magnitude",
        ""
    ])
    
    with open(filepath, 'w') as f:
        f.write("\n".join(content))
