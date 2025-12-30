"""
Error Plotting Utilities for Volatility Comparison

Provides heatmaps and error distribution plots for comparing MLMC
and Laplace volatility estimation methods.

Author: Wadoud (KAUST Internship)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import Optional, Tuple, Dict
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.common import VolatilitySurfaceResult, compute_accuracy_metrics


def plot_difference_heatmap(
    result_mlmc: VolatilitySurfaceResult,
    result_laplace: VolatilitySurfaceResult,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot heatmap of the difference between MLMC and Laplace surfaces.
    
    Parameters
    ----------
    result_mlmc : VolatilitySurfaceResult
        MLMC estimation result.
    result_laplace : VolatilitySurfaceResult
        Laplace approximation result.
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
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Compute difference
    diff = result_mlmc.b_squared_values - result_laplace.b_squared_values
    
    # Absolute difference
    ax1 = axes[0]
    extent = [
        result_mlmc.s_grid.min(), result_mlmc.s_grid.max(),
        result_mlmc.t_grid.min(), result_mlmc.t_grid.max()
    ]
    
    # Use diverging colourmap centred at zero
    vmax = np.nanmax(np.abs(diff))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im1 = ax1.imshow(
        diff,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='RdBu_r',
        norm=norm
    )
    ax1.set_xlabel('Basket Value (s)', fontsize=11)
    ax1.set_ylabel('Time (t)', fontsize=11)
    ax1.set_title(r'Difference: $\bar{b}^2_{MLMC} - \bar{b}^2_{Laplace}$', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Difference', fontsize=10)
    
    # Relative difference
    ax2 = axes[1]
    mean_surface = (result_mlmc.b_squared_values + result_laplace.b_squared_values) / 2
    rel_diff = np.where(
        mean_surface > 0,
        diff / mean_surface,
        np.nan
    )
    
    vmax_rel = np.nanmax(np.abs(rel_diff))
    norm_rel = TwoSlopeNorm(vmin=-vmax_rel, vcenter=0, vmax=vmax_rel)
    
    im2 = ax2.imshow(
        rel_diff,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='RdBu_r',
        norm=norm_rel
    )
    ax2.set_xlabel('Basket Value (s)', fontsize=11)
    ax2.set_ylabel('Time (t)', fontsize=11)
    ax2.set_title(r'Relative Difference', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Relative Diff', fontsize=10)
    
    plt.suptitle('MLMC vs Laplace: Volatility² Surface Difference', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_pointwise_error(
    result_mlmc: VolatilitySurfaceResult,
    result_laplace: VolatilitySurfaceResult,
    metrics: Optional[Dict] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot symmetric relative error heatmap.
    
    Parameters
    ----------
    result_mlmc : VolatilitySurfaceResult
        MLMC estimation result.
    result_laplace : VolatilitySurfaceResult
        Laplace approximation result.
    metrics : dict, optional
        Pre-computed metrics from compute_accuracy_metrics().
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
    if metrics is None:
        metrics = compute_accuracy_metrics(result_mlmc, result_laplace)
    
    pointwise_error = metrics['pointwise_relative_error']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Heatmap of symmetric relative error
    ax1 = axes[0]
    extent = [
        result_mlmc.s_grid.min(), result_mlmc.s_grid.max(),
        result_mlmc.t_grid.min(), result_mlmc.t_grid.max()
    ]
    
    im1 = ax1.imshow(
        pointwise_error * 100,  # Convert to percentage
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='YlOrRd',
        vmin=0,
        vmax=np.nanpercentile(pointwise_error * 100, 95)
    )
    ax1.set_xlabel('Basket Value (s)', fontsize=11)
    ax1.set_ylabel('Time (t)', fontsize=11)
    ax1.set_title('Symmetric Relative Error (%)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Error (%)', fontsize=10)
    
    # Histogram of errors
    ax2 = axes[1]
    valid_errors = pointwise_error[np.isfinite(pointwise_error)] * 100
    
    ax2.hist(
        valid_errors,
        bins=50,
        density=True,
        alpha=0.7,
        color='steelblue',
        edgecolor='black'
    )
    ax2.axvline(
        np.median(valid_errors),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Median: {np.median(valid_errors):.2f}%'
    )
    ax2.axvline(
        np.mean(valid_errors),
        color='orange',
        linestyle=':',
        linewidth=2,
        label=f'Mean: {np.mean(valid_errors):.2f}%'
    )
    
    ax2.set_xlabel('Symmetric Relative Error (%)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(
        r'MLMC vs Laplace: Symmetric Error $2|a-b|/(|a|+|b|)$',
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_scatter_comparison(
    result_mlmc: VolatilitySurfaceResult,
    result_laplace: VolatilitySurfaceResult,
    metrics: Optional[Dict] = None,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Scatter plot of MLMC vs Laplace b² values.
    
    Perfect agreement would lie on the diagonal.
    
    Parameters
    ----------
    result_mlmc : VolatilitySurfaceResult
        MLMC estimation result.
    result_laplace : VolatilitySurfaceResult
        Laplace approximation result.
    metrics : dict, optional
        Pre-computed metrics from compute_accuracy_metrics().
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
    if metrics is None:
        metrics = compute_accuracy_metrics(result_mlmc, result_laplace)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten and filter valid values
    mlmc_flat = result_mlmc.b_squared_values.flatten()
    laplace_flat = result_laplace.b_squared_values.flatten()
    
    valid = np.isfinite(mlmc_flat) & np.isfinite(laplace_flat)
    mlmc_valid = mlmc_flat[valid]
    laplace_valid = laplace_flat[valid]
    
    # Subsample if too many points
    if len(mlmc_valid) > 5000:
        indices = np.random.choice(len(mlmc_valid), 5000, replace=False)
        mlmc_valid = mlmc_valid[indices]
        laplace_valid = laplace_valid[indices]
    
    # Scatter plot
    ax.scatter(
        laplace_valid, mlmc_valid,
        alpha=0.5, s=20, c='steelblue',
        edgecolor='none'
    )
    
    # Diagonal line (perfect agreement)
    lims = [
        min(mlmc_valid.min(), laplace_valid.min()),
        max(mlmc_valid.max(), laplace_valid.max())
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Agreement')
    
    # Add correlation annotation
    corr = metrics['correlation']
    ax.text(
        0.05, 0.95,
        f'Correlation: {corr:.4f}',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax.set_xlabel(r'Laplace $\bar{b}^2$', fontsize=12)
    ax.set_ylabel(r'MLMC $\bar{b}^2$', fontsize=12)
    ax.set_title('MLMC vs Laplace: Pointwise Comparison', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_summary_panel(
    result_mlmc: VolatilitySurfaceResult,
    result_laplace: VolatilitySurfaceResult,
    metrics: Optional[Dict] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary panel with multiple visualisations.
    
    Includes:
    - Both 3D surfaces
    - Difference heatmap
    - Error histogram
    - Scatter comparison
    
    Parameters
    ----------
    result_mlmc : VolatilitySurfaceResult
        MLMC estimation result.
    result_laplace : VolatilitySurfaceResult
        Laplace approximation result.
    metrics : dict, optional
        Pre-computed metrics.
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
    if metrics is None:
        metrics = compute_accuracy_metrics(result_mlmc, result_laplace)
    
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2 rows, 3 columns
    # Row 1: MLMC surface, Laplace surface, Wireframe overlay
    # Row 2: Difference heatmap, Error histogram, Scatter plot
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Row 1: 3D surfaces
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    T_mesh, S_mesh = np.meshgrid(result_mlmc.t_grid, result_mlmc.s_grid, indexing='ij')
    Z1 = np.nan_to_num(result_mlmc.b_squared_values, nan=0)
    ax1.plot_surface(S_mesh, T_mesh, Z1, cmap='viridis', alpha=0.9)
    ax1.set_title(f'MLMC ({result_mlmc.computation_time:.2f}s)', fontsize=10)
    ax1.set_xlabel('s', fontsize=9)
    ax1.set_ylabel('t', fontsize=9)
    ax1.view_init(elev=20, azim=-60)
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    Z2 = np.nan_to_num(result_laplace.b_squared_values, nan=0)
    ax2.plot_surface(S_mesh, T_mesh, Z2, cmap='viridis', alpha=0.9)
    ax2.set_title(f'Laplace ({result_laplace.computation_time:.2f}s)', fontsize=10)
    ax2.set_xlabel('s', fontsize=9)
    ax2.set_ylabel('t', fontsize=9)
    ax2.view_init(elev=20, azim=-60)
    
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    step = max(1, len(result_mlmc.t_grid) // 8)
    ax3.plot_wireframe(S_mesh[::step, ::step], T_mesh[::step, ::step], 
                       Z1[::step, ::step], color='blue', linewidth=0.5, alpha=0.7)
    ax3.plot_wireframe(S_mesh[::step, ::step], T_mesh[::step, ::step], 
                       Z2[::step, ::step], color='red', linewidth=0.5, alpha=0.7)
    ax3.set_title('Wireframe Overlay', fontsize=10)
    ax3.view_init(elev=20, azim=-60)
    
    # Row 2: Analysis plots
    ax4 = fig.add_subplot(2, 3, 4)
    diff = result_mlmc.b_squared_values - result_laplace.b_squared_values
    extent = [result_mlmc.s_grid.min(), result_mlmc.s_grid.max(),
              result_mlmc.t_grid.min(), result_mlmc.t_grid.max()]
    vmax = np.nanmax(np.abs(diff))
    im4 = ax4.imshow(diff, aspect='auto', origin='lower', extent=extent,
                     cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax4.set_title('Difference Heatmap', fontsize=10)
    ax4.set_xlabel('s', fontsize=9)
    ax4.set_ylabel('t', fontsize=9)
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    ax5 = fig.add_subplot(2, 3, 5)
    valid_errors = metrics.get('pointwise_relative_error', metrics.get('pointwise_relative_disagreement'))
    valid_errors = valid_errors[np.isfinite(valid_errors)] * 100
    ax5.hist(valid_errors, bins=40, density=True, alpha=0.7, color='steelblue')
    ax5.axvline(np.median(valid_errors), color='red', linestyle='--', 
                label=f'Median: {np.median(valid_errors):.2f}%')
    ax5.set_xlabel('Relative Error (%)', fontsize=9)
    ax5.set_ylabel('Density', fontsize=9)
    ax5.set_title('Error Distribution', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    mlmc_flat = result_mlmc.b_squared_values.flatten()
    laplace_flat = result_laplace.b_squared_values.flatten()
    valid = np.isfinite(mlmc_flat) & np.isfinite(laplace_flat)
    if np.sum(valid) > 2000:
        idx = np.random.choice(np.sum(valid), 2000, replace=False)
        mlmc_plot = mlmc_flat[valid][idx]
        laplace_plot = laplace_flat[valid][idx]
    else:
        mlmc_plot = mlmc_flat[valid]
        laplace_plot = laplace_flat[valid]
    ax6.scatter(laplace_plot, mlmc_plot, alpha=0.4, s=15)
    lims = [min(mlmc_plot.min(), laplace_plot.min()),
            max(mlmc_plot.max(), laplace_plot.max())]
    ax6.plot(lims, lims, 'r--', linewidth=1.5)
    ax6.set_xlabel('Laplace', fontsize=9)
    ax6.set_ylabel('MLMC', fontsize=9)
    ax6.set_title(f'Scatter (r={metrics["correlation"]:.3f})', fontsize=10)
    ax6.set_aspect('equal', adjustable='box')
    ax6.grid(True, alpha=0.3)
    
    l2_val = metrics.get('l2_disagreement', metrics.get('l2_relative_error', np.nan))
    plt.suptitle(
        f'MLMC vs Laplace Comparison\n'
        rf'$L^2$ Disagreement: {l2_val:.4f}, '
        f'Mean Rel. Diff: {metrics["mean_relative_difference"]*100:.2f}%',
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig
