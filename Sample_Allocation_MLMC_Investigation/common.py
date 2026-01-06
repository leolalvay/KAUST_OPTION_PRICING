"""
Common Data Structures for Volatility Estimation Methods

This module defines the standardised return type for both MLMC and Laplace
volatility estimation methods, ensuring consistent comparison.

Includes diagnostic functions for:
- Method agreement (MLMC vs Laplace) - NEITHER is ground truth
- MLMC level diagnostics (convergence rates, variance reduction)
- Across-run statistical uncertainty

Author: Wadoud (KAUST Internship)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any, TYPE_CHECKING
from scipy import stats as scipy_stats

if TYPE_CHECKING:
    from .mlmc_ot_estimator import LevelStats


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


def compute_method_agreement(
    b2_mlmc: np.ndarray,
    b2_laplace: np.ndarray
) -> Dict[str, Any]:
    """
    Compute agreement metrics comparing MLMC and Laplace volatility surfaces.

    IMPORTANT: Neither method is "ground truth" - these metrics measure
    DISAGREEMENT between methods, not ERROR. High disagreement may indicate:
    - Statistical noise in MLMC
    - Approximation error in Laplace
    - Insufficient resolution in either method

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
        - l2_disagreement: ||b²_MLMC - b²_Laplace||_2 / ||b²_mean||_2
        - linf_disagreement: max|b²_MLMC - b²_Laplace|
        - linf_relative_disagreement: max relative disagreement
        - mean_absolute_difference: mean|b²_MLMC - b²_Laplace|
        - mean_relative_difference: mean(2|diff| / |sum|) (symmetric)
        - bias: mean(b²_MLMC - b²_Laplace)
        - correlation: Pearson correlation coefficient
        - pointwise_relative_disagreement: 2D array for heatmap
        - valid_fraction: fraction of grid points with finite values

    Notes
    -----
    Uses symmetric relative difference 2|a-b|/(|a|+|b|) to avoid bias towards
    either method. This is NOT an error measure - neither method is ground truth.
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
            "l2_disagreement": np.inf,
            "linf_disagreement": np.inf,
            "linf_relative_disagreement": np.inf,
            "mean_absolute_difference": np.inf,
            "mean_relative_difference": np.inf,
            "bias": np.nan,
            "correlation": np.nan,
            "pointwise_relative_disagreement": np.full_like(b2_mlmc, np.nan),
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
        "l2_disagreement": l2_relative,
        "linf_disagreement": linf_error,
        "linf_relative_disagreement": linf_relative,
        "mean_absolute_difference": mean_abs_diff,
        "mean_relative_difference": mean_relative,
        "bias": bias,
        "correlation": correlation,
        "pointwise_relative_disagreement": pointwise,
        "valid_fraction": valid_fraction,
    }


# Backward compatibility alias
compute_accuracy_metrics = compute_method_agreement


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """
    Format agreement metrics as a readable table.

    Parameters
    ----------
    metrics : dict
        Output from compute_method_agreement().

    Returns
    -------
    str
        Formatted table string.
    """
    # Handle both old and new key names for backward compatibility
    l2_val = metrics.get('l2_disagreement', metrics.get('l2_relative_error', np.nan))
    linf_val = metrics.get('linf_disagreement', metrics.get('linf_error', np.nan))

    lines = [
        "Method Agreement (MLMC vs Laplace)",
        "NOTE: Neither method is ground truth",
        "=" * 45,
        f"  L2 disagreement:        {l2_val:.4f}",
        f"  L∞ disagreement:        {linf_val:.2f}",
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
        Output from compute_method_agreement().
    filepath : str
        Path to save the markdown file.
    params_summary : str
        Optional parameter summary to include.
    """
    # Handle both old and new key names
    l2_val = metrics.get('l2_disagreement', metrics.get('l2_relative_error', np.nan))
    linf_val = metrics.get('linf_disagreement', metrics.get('linf_error', np.nan))

    content = [
        "# Method Agreement: MLMC vs Laplace Approximation",
        "",
        "**IMPORTANT**: Neither method is ground truth. These metrics measure",
        "disagreement between methods, not error.",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| L² disagreement | {l2_val:.6f} |",
        f"| L∞ disagreement | {linf_val:.4f} |",
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
        "- **Neither method is treated as ground truth**",
        "- Relative differences use symmetric formula: 2|a-b|/(|a|+|b|)",
        "- L² disagreement normalised by mean surface magnitude",
        "- High correlation (> 0.95) indicates good agreement",
        ""
    ])

    with open(filepath, 'w') as f:
        f.write("\n".join(content))


# =============================================================================
# MLMC Level Diagnostics
# =============================================================================

def compute_mlmc_level_diagnostics(
    level_stats: Dict[int, 'LevelStats'],
    level_surfaces: Optional[Dict[int, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive MLMC diagnostics from level statistics.

    This function analyzes the convergence properties of MLMC following
    Giles (2015) methodology:
    - Weak convergence rate α from |E[P_ℓ - P_{ℓ-1}]| ~ 2^{-αℓ}
    - Variance decay rate β from V_ℓ ~ 2^{-βℓ}
    - Richardson extrapolation bias estimate

    Parameters
    ----------
    level_stats : Dict[int, LevelStats]
        Statistics for each level, from make_c with return_stats=True.
    level_surfaces : Dict[int, np.ndarray], optional
        Cumulative surfaces at each level for L² norm computation.

    Returns
    -------
    dict
        Diagnostic results containing:
        - alpha: weak convergence rate
        - beta: variance decay rate
        - richardson_bias: estimated bias from Richardson extrapolation
        - level_corrections: dict of ||b²_ℓ - b²_{ℓ-1}||_{L²}
        - variances: dict of V_ℓ per level
        - correlations: dict of ρ_ℓ per level
        - kurtosis_values: dict of κ_ℓ per level
        - variance_reduction_factors: dict of VRF_ℓ per level
        - convergence_quality: 'good' / 'acceptable' / 'poor'
        - warnings: list of diagnostic failures
    """
    warnings = []
    levels = sorted(level_stats.keys())
    L = max(levels)

    # Extract per-level statistics
    variances = {l: level_stats[l].variance for l in levels}
    mean_corrections = {l: level_stats[l].mean_correction for l in levels}
    correlations = {l: level_stats[l].correlation for l in levels}
    kurtosis_values = {l: level_stats[l].kurtosis for l in levels}
    vrfs = {l: level_stats[l].variance_reduction_factor for l in levels}

    # Compute L² level corrections if surfaces provided
    level_corrections = {}
    if level_surfaces is not None and len(level_surfaces) > 1:
        sorted_levels = sorted(level_surfaces.keys())
        for i, l in enumerate(sorted_levels):
            if i == 0:
                level_corrections[l] = np.sqrt(np.mean(level_surfaces[l] ** 2))
            else:
                prev_l = sorted_levels[i - 1]
                diff = level_surfaces[l] - level_surfaces[prev_l]
                level_corrections[l] = np.sqrt(np.mean(diff ** 2))

    # Fit convergence rates (need at least 2 levels > 0)
    higher_levels = [l for l in levels if l >= 1]
    alpha = np.nan
    beta = np.nan

    if len(higher_levels) >= 2:
        # Fit α from log₂|m_ℓ| vs ℓ
        log2_m = []
        lvls = []
        for l in higher_levels:
            m_l = np.abs(mean_corrections[l])
            if m_l > 1e-15:
                log2_m.append(np.log2(m_l))
                lvls.append(l)

        if len(lvls) >= 2:
            # Linear regression: log₂|m_ℓ| = c - α * ℓ
            coeffs = np.polyfit(lvls, log2_m, 1)
            alpha = -coeffs[0]

        # Fit β from log₂(V_ℓ) vs ℓ
        log2_v = []
        lvls_v = []
        for l in higher_levels:
            v_l = variances[l]
            if v_l > 1e-15:
                log2_v.append(np.log2(v_l))
                lvls_v.append(l)

        if len(lvls_v) >= 2:
            coeffs = np.polyfit(lvls_v, log2_v, 1)
            beta = -coeffs[0]

    # Richardson extrapolation bias estimate: bias ≈ m_L / (2^α - 1)
    richardson_bias = np.nan
    if not np.isnan(alpha) and L in mean_corrections:
        denom = 2 ** alpha - 1
        if denom > 0.1:  # Avoid division by small numbers
            richardson_bias = np.abs(mean_corrections[L]) / denom

    # Assess convergence quality
    convergence_quality = 'good'

    # Check correlations (should be > 0.95 for effective coupling)
    for l in higher_levels:
        if correlations[l] < 0.95:
            warnings.append(f"Level {l}: correlation {correlations[l]:.3f} < 0.95")
            if correlations[l] < 0.80:
                convergence_quality = 'poor'
            else:
                convergence_quality = 'acceptable' if convergence_quality == 'good' else convergence_quality

    # Check VRF (should be > 10 for effective variance reduction)
    for l in higher_levels:
        if vrfs[l] < 10:
            warnings.append(f"Level {l}: VRF {vrfs[l]:.1f} < 10")
            convergence_quality = 'acceptable' if convergence_quality == 'good' else convergence_quality

    # Check kurtosis (should be < 100 for reliable estimates)
    for l in levels:
        if np.abs(kurtosis_values[l]) > 100:
            warnings.append(f"Level {l}: kurtosis {kurtosis_values[l]:.1f} > 100")
            convergence_quality = 'poor'

    # Check α and β are reasonable
    if not np.isnan(alpha) and alpha < 0.5:
        warnings.append(f"Weak convergence rate α = {alpha:.2f} < 0.5 (slow convergence)")
        convergence_quality = 'acceptable' if convergence_quality == 'good' else convergence_quality

    if not np.isnan(beta) and beta < 1.0:
        warnings.append(f"Variance decay rate β = {beta:.2f} < 1.0 (insufficient variance reduction)")
        convergence_quality = 'acceptable' if convergence_quality == 'good' else convergence_quality

    return {
        'alpha': alpha,
        'beta': beta,
        'richardson_bias': richardson_bias,
        'level_corrections': level_corrections,
        'variances': variances,
        'mean_corrections': mean_corrections,
        'correlations': correlations,
        'kurtosis_values': kurtosis_values,
        'variance_reduction_factors': vrfs,
        'convergence_quality': convergence_quality,
        'warnings': warnings,
        'n_levels': len(levels),
    }


def compute_across_run_uncertainty(
    all_surfaces: np.ndarray,
    reference_surface: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute statistical uncertainty from multiple MLMC runs.

    Parameters
    ----------
    all_surfaces : np.ndarray
        Array of shape (n_runs, n_t, n_s) containing b² surfaces from
        multiple independent MLMC runs.
    reference_surface : np.ndarray, optional
        Reference surface (e.g., Laplace) for computing disagreement.
        If provided, also computes running average disagreement.

    Returns
    -------
    dict
        Statistical uncertainty metrics:
        - n_runs: number of independent runs
        - mean_surface: mean b² surface across runs
        - std_surface: pointwise standard deviation
        - relative_std: relative standard deviation (std / mean)
        - standard_error: standard error of the mean (std / sqrt(n))
        - relative_se: relative standard error
        - pointwise_cv: coefficient of variation (2D array)
        - running_std: std as function of n runs (convergence check)
    """
    n_runs = all_surfaces.shape[0]

    # Mean and std across runs
    mean_surface = np.mean(all_surfaces, axis=0)
    std_surface = np.std(all_surfaces, axis=0, ddof=1)

    # Standard error of the mean
    se_surface = std_surface / np.sqrt(n_runs)

    # Relative measures (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_std = np.where(
            np.abs(mean_surface) > 1e-10,
            std_surface / np.abs(mean_surface),
            np.nan
        )
        relative_se = np.where(
            np.abs(mean_surface) > 1e-10,
            se_surface / np.abs(mean_surface),
            np.nan
        )
        cv = np.where(
            np.abs(mean_surface) > 1e-10,
            std_surface / np.abs(mean_surface),
            np.nan
        )

    # Running statistics to verify 1/sqrt(n) convergence
    running_std = []
    for n in range(2, n_runs + 1):
        partial_mean = np.mean(all_surfaces[:n], axis=0)
        partial_std = np.std(all_surfaces[:n], axis=0, ddof=1)
        partial_se = partial_std / np.sqrt(n)
        # Report mean relative SE across the surface
        valid = np.abs(partial_mean) > 1e-10
        if np.any(valid):
            mean_rel_se = np.nanmean(partial_se[valid] / np.abs(partial_mean[valid]))
        else:
            mean_rel_se = np.nan
        running_std.append({'n': n, 'mean_relative_se': mean_rel_se})

    result = {
        'n_runs': n_runs,
        'mean_surface': mean_surface,
        'std_surface': std_surface,
        'standard_error': se_surface,
        'relative_std': np.nanmean(relative_std),
        'relative_se': np.nanmean(relative_se),
        'pointwise_cv': cv,
        'running_stats': running_std,
    }

    # If reference provided, compute disagreement with running average
    if reference_surface is not None:
        l2_disagreements = []
        for n in range(1, n_runs + 1):
            running_mean = np.mean(all_surfaces[:n], axis=0)
            diff = running_mean - reference_surface
            mean_ref = (running_mean + reference_surface) / 2
            l2_norm = np.sqrt(np.nanmean(mean_ref ** 2))
            l2_diff = np.sqrt(np.nanmean(diff ** 2))
            l2_disagreements.append(l2_diff / l2_norm if l2_norm > 0 else np.inf)
        result['l2_disagreement_vs_n'] = l2_disagreements

    return result
