"""
Convergence Plotting Utilities for MLMC Analysis

Provides visualisations for MLMC convergence studies and computational
cost comparisons.

Author: Wadoud (KAUST Internship)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.common import VolatilitySurfaceResult


def plot_mlmc_convergence(
    all_b_squared: np.ndarray,
    result_laplace: VolatilitySurfaceResult,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot MLMC convergence across multiple runs.
    
    Parameters
    ----------
    all_b_squared : np.ndarray
        All MLMC b² surfaces, shape (n_runs, len(t_grid), len(s_grid)).
    result_laplace : VolatilitySurfaceResult
        Laplace result for reference.
    t_grid : np.ndarray
        Time grid.
    s_grid : np.ndarray
        Basket value grid.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    fig : matplotlib Figure
    """
    n_runs = all_b_squared.shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Mean and std across runs
    mean_b_squared = np.mean(all_b_squared, axis=0)
    std_b_squared = np.std(all_b_squared, axis=0)
    
    # 1. Mean vs Laplace comparison at mid-time
    ax1 = axes[0]
    t_mid_idx = len(t_grid) // 2
    t_mid = t_grid[t_mid_idx]
    
    mean_slice = mean_b_squared[t_mid_idx, :]
    std_slice = std_b_squared[t_mid_idx, :]
    laplace_slice = result_laplace.b_squared_values[t_mid_idx, :]
    
    ax1.plot(s_grid, mean_slice, 'b-', linewidth=2, label='MLMC Mean')
    ax1.fill_between(
        s_grid,
        mean_slice - 2 * std_slice,
        mean_slice + 2 * std_slice,
        alpha=0.3, color='blue',
        label='MLMC ±2σ'
    )
    ax1.plot(s_grid, laplace_slice, 'r--', linewidth=2, label='Laplace')
    
    ax1.set_xlabel('Basket Value (s)', fontsize=11)
    ax1.set_ylabel(r'$\bar{b}^2$', fontsize=11)
    ax1.set_title(f'Slice at t = {t_mid:.3f}', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Standard deviation heatmap
    ax2 = axes[1]
    extent = [s_grid.min(), s_grid.max(), t_grid.min(), t_grid.max()]
    
    im2 = ax2.imshow(
        std_b_squared,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='viridis'
    )
    ax2.set_xlabel('Basket Value (s)', fontsize=11)
    ax2.set_ylabel('Time (t)', fontsize=11)
    ax2.set_title(f'MLMC Std Dev ({n_runs} runs)', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Std Dev', fontsize=10)
    
    # 3. Convergence analysis: decompose into bias and variance
    ax3 = axes[2]
    
    # Compute running average L² error to Laplace (total error)
    l2_errors = []
    # Compute running standard error of MLMC mean (stochastic component)
    std_errors = []
    
    for n in range(1, n_runs + 1):
        # Running mean surface
        running_mean = np.mean(all_b_squared[:n], axis=0)
        
        # Total L² error to Laplace
        diff = running_mean - result_laplace.b_squared_values
        valid = np.isfinite(diff)
        l2 = np.sqrt(np.mean(diff[valid] ** 2))
        l2_errors.append(l2)
        
        # Standard error of the mean (stochastic uncertainty)
        if n > 1:
            # Std across runs, then average over grid, divided by sqrt(n)
            std_across_runs = np.std(all_b_squared[:n], axis=0, ddof=1)
            mean_std = np.mean(std_across_runs[np.isfinite(std_across_runs)])
            std_errors.append(mean_std / np.sqrt(n))
        else:
            # For n=1, estimate from later runs
            std_across_runs = np.std(all_b_squared, axis=0, ddof=1)
            mean_std = np.mean(std_across_runs[np.isfinite(std_across_runs)])
            std_errors.append(mean_std)
    
    n_vals = np.arange(1, n_runs + 1)
    
    # Plot total error (bias-dominated, stays flat)
    ax3.plot(n_vals, l2_errors, 'b-o', markersize=5, linewidth=2,
             label=f'Total L² error (converges to {l2_errors[-1]:.1f})')
    
    # Plot stochastic component (follows 1/√n)
    ax3.plot(n_vals, std_errors, 'g-s', markersize=4, linewidth=1.5,
             alpha=0.8, label='Stochastic uncertainty (std err)')
    
    # Add reference line for 1/√n scaling (matched to stochastic component)
    ref_line = std_errors[0] / np.sqrt(n_vals)
    ax3.plot(n_vals, ref_line, 'g--', alpha=0.5, linewidth=1,
             label=r'Reference: $\propto 1/\sqrt{n}$')
    
    # Add horizontal line showing converged bias
    ax3.axhline(y=l2_errors[-1], color='b', linestyle=':', alpha=0.5,
                label=f'Systematic bias ≈ {l2_errors[-1]:.1f}')
    
    ax3.set_xlabel('Number of MLMC Runs', fontsize=11)
    ax3.set_ylabel('Error / Uncertainty', fontsize=11)
    ax3.set_title('Error Decomposition', fontsize=12)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=8, loc='upper right')
    
    plt.suptitle(f'MLMC Convergence Analysis ({n_runs} runs)', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_error_vs_time(
    errors: List[float],
    times: List[float],
    labels: List[str],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot L2 error vs computation time for different configurations.
    
    Parameters
    ----------
    errors : list of float
        L2 relative errors for each configuration.
    times : list of float
        Computation times in seconds.
    labels : list of str
        Labels for each configuration.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to arrays
    errors = np.array(errors)
    times = np.array(times)
    
    # Scatter plot
    scatter = ax.scatter(
        times, errors,
        s=100, c='steelblue',
        edgecolor='black', alpha=0.7
    )
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (times[i], errors[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel('Computation Time (s)', fontsize=12)
    ax.set_ylabel(r'$L^2$ Relative Error', fontsize=12)
    ax.set_title('Accuracy vs Computational Cost', fontsize=13)
    
    # Log scale if range is large
    if max(times) / min(times) > 10:
        ax.set_xscale('log')
    if max(errors) / min(errors) > 10:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_dimension_scaling(
    dimensions: List[int],
    mlmc_times: List[float],
    laplace_times: List[float],
    mlmc_errors: Optional[List[float]] = None,
    laplace_errors: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot computational scaling with dimension.
    
    Parameters
    ----------
    dimensions : list of int
        Number of assets for each test.
    mlmc_times : list of float
        MLMC computation times.
    laplace_times : list of float
        Laplace computation times.
    mlmc_errors : list of float, optional
        MLMC L2 errors (if available).
    laplace_errors : list of float, optional
        Laplace L2 errors (if available).
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    fig : matplotlib Figure
    """
    n_cols = 2 if mlmc_errors is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    if n_cols == 1:
        axes = [axes]
    
    # Timing plot
    ax1 = axes[0]
    ax1.semilogy(dimensions, mlmc_times, 'bo-', linewidth=2, markersize=8, label='MLMC')
    ax1.semilogy(dimensions, laplace_times, 'rs--', linewidth=2, markersize=8, label='Laplace')
    
    ax1.set_xlabel('Number of Assets (d)', fontsize=12)
    ax1.set_ylabel('Computation Time (s)', fontsize=12)
    ax1.set_title('Computational Scaling', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xticks(dimensions)
    
    # Error plot (if available)
    if mlmc_errors is not None and n_cols > 1:
        ax2 = axes[1]
        ax2.semilogy(dimensions, mlmc_errors, 'bo-', linewidth=2, markersize=8, label='MLMC')
        if laplace_errors is not None:
            ax2.semilogy(dimensions, laplace_errors, 'rs--', linewidth=2, markersize=8, label='Laplace')
        
        ax2.set_xlabel('Number of Assets (d)', fontsize=12)
        ax2.set_ylabel(r'$L^2$ Relative Error', fontsize=12)
        ax2.set_title('Accuracy Scaling', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.set_xticks(dimensions)
    
    plt.suptitle('Dimension Scaling Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_confidence_bands(
    result_mean: VolatilitySurfaceResult,
    all_b_squared: np.ndarray,
    result_laplace: VolatilitySurfaceResult,
    s_value: float,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot time evolution of b² with confidence bands.
    
    Parameters
    ----------
    result_mean : VolatilitySurfaceResult
        Mean MLMC result.
    all_b_squared : np.ndarray
        All MLMC runs, shape (n_runs, n_t, n_s).
    result_laplace : VolatilitySurfaceResult
        Laplace result.
    s_value : float
        Basket value to plot at.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find nearest s index
    s_idx = np.argmin(np.abs(result_mean.s_grid - s_value))
    actual_s = result_mean.s_grid[s_idx]
    
    t_grid = result_mean.t_grid
    
    # Extract time series for all runs
    all_series = all_b_squared[:, :, s_idx]  # Shape: (n_runs, n_t)
    
    # Compute statistics
    mean_series = np.mean(all_series, axis=0)
    std_series = np.std(all_series, axis=0)
    p5 = np.percentile(all_series, 5, axis=0)
    p95 = np.percentile(all_series, 95, axis=0)
    
    # Laplace series
    laplace_series = result_laplace.b_squared_values[:, s_idx]
    
    # Plot
    ax.plot(t_grid, mean_series, 'b-', linewidth=2, label='MLMC Mean')
    ax.fill_between(
        t_grid, p5, p95,
        alpha=0.2, color='blue',
        label='MLMC 5-95 percentile'
    )
    ax.fill_between(
        t_grid,
        mean_series - std_series,
        mean_series + std_series,
        alpha=0.3, color='blue',
        label='MLMC ±1σ'
    )
    ax.plot(t_grid, laplace_series, 'r--', linewidth=2, label='Laplace')
    
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel(r'$\bar{b}^2(t, s)$', fontsize=12)
    ax.set_title(f'Time Evolution at s = {actual_s:.1f}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig
