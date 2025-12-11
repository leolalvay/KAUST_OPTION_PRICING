"""
Surface Plotting Utilities for Volatility Comparison

Provides 3D surface visualisations for comparing MLMC and Laplace
volatility surfaces.

Author: Wadoud (KAUST Internship)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.common import VolatilitySurfaceResult


def plot_surface_3d(
    result: VolatilitySurfaceResult,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    view_elev: float = 25,
    view_azim: float = -55,
    show_colorbar: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a 3D surface plot of b²(t, s).
    
    Parameters
    ----------
    result : VolatilitySurfaceResult
        Result object containing the surface data.
    ax : matplotlib Axes3D, optional
        Existing 3D axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title. Defaults to method name.
    cmap : str
        Colourmap name.
    view_elev : float
        Elevation angle for 3D view.
    view_azim : float
        Azimuth angle for 3D view.
    show_colorbar : bool
        Whether to show the colourbar.
        
    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes3D
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Create meshgrid
    T_mesh, S_mesh = np.meshgrid(result.t_grid, result.s_grid, indexing='ij')
    
    # Clean surface data for plotting
    Z = result.b_squared_values.copy()
    Z = np.nan_to_num(Z, nan=np.nanmean(Z))
    
    # Plot surface
    surf = ax.plot_surface(
        S_mesh, T_mesh, Z,
        cmap=cmap,
        edgecolor='none',
        alpha=0.9
    )
    
    # Labels
    ax.set_xlabel('Basket Value (s)', fontsize=11, labelpad=10)
    ax.set_ylabel('Time (t)', fontsize=11, labelpad=10)
    ax.set_zlabel(r'$\bar{b}^2(t, s)$', fontsize=11, labelpad=5)
    
    if title is None:
        title = f'{result.method_name}: Projected Volatility² Surface'
    ax.set_title(title, fontsize=12, pad=10)
    
    # Set view angle
    ax.view_init(elev=view_elev, azim=view_azim)
    
    # Colourbar
    if show_colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15)
        cbar.set_label(r'$\bar{b}^2$', fontsize=10)
    
    return fig, ax


def plot_surface_comparison(
    result_mlmc: VolatilitySurfaceResult,
    result_laplace: VolatilitySurfaceResult,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create side-by-side comparison of MLMC and Laplace surfaces.
    
    Parameters
    ----------
    result_mlmc : VolatilitySurfaceResult
        MLMC estimation result.
    result_laplace : VolatilitySurfaceResult
        Laplace approximation result.
    figsize : tuple
        Figure size (width, height).
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the figure.
        
    Returns
    -------
    fig : matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # MLMC surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_surface_3d(
        result_mlmc, ax=ax1,
        title=f'MLMC (time: {result_mlmc.computation_time:.2f}s)',
        show_colorbar=False
    )
    
    # Laplace surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_surface_3d(
        result_laplace, ax=ax2,
        title=f'Laplace (time: {result_laplace.computation_time:.2f}s)',
        show_colorbar=False
    )
    
    # Match z-axis limits
    z_min = min(
        np.nanmin(result_mlmc.b_squared_values),
        np.nanmin(result_laplace.b_squared_values)
    )
    z_max = max(
        np.nanmax(result_mlmc.b_squared_values),
        np.nanmax(result_laplace.b_squared_values)
    )
    ax1.set_zlim([z_min * 0.95, z_max * 1.05])
    ax2.set_zlim([z_min * 0.95, z_max * 1.05])
    
    plt.suptitle(
        r'Projected Volatility² Surface: $\bar{b}^2(t, s)$ Comparison',
        fontsize=14, y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_surface_slices(
    result_mlmc: VolatilitySurfaceResult,
    result_laplace: VolatilitySurfaceResult,
    t_values: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot 1D slices of the volatility surface at fixed times.
    
    Parameters
    ----------
    result_mlmc : VolatilitySurfaceResult
        MLMC estimation result.
    result_laplace : VolatilitySurfaceResult
        Laplace approximation result.
    t_values : array-like, optional
        Time values for slices. Defaults to evenly spaced.
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
    if t_values is None:
        # Choose 4 time slices
        t_grid = result_mlmc.t_grid
        indices = np.linspace(0, len(t_grid) - 1, 4, dtype=int)
        t_values = t_grid[indices]
    
    n_slices = len(t_values)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    s_grid = result_mlmc.s_grid
    
    for i, t in enumerate(t_values):
        ax = axes[i]
        
        # Find nearest time index
        t_idx_mlmc = np.argmin(np.abs(result_mlmc.t_grid - t))
        t_idx_laplace = np.argmin(np.abs(result_laplace.t_grid - t))
        
        # Get slices
        b2_mlmc = result_mlmc.b_squared_values[t_idx_mlmc, :]
        b2_laplace = result_laplace.b_squared_values[t_idx_laplace, :]
        
        # Plot
        ax.plot(s_grid, b2_mlmc, 'b-', linewidth=2, label='MLMC')
        ax.plot(s_grid, b2_laplace, 'r--', linewidth=2, label='Laplace')
        
        ax.set_xlabel('Basket Value (s)', fontsize=10)
        ax.set_ylabel(r'$\bar{b}^2(t, s)$', fontsize=10)
        ax.set_title(f't = {t:.3f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(
        r'Volatility² Slices: $\bar{b}^2(t, s)$ at Fixed Times',
        fontsize=13
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_wireframe_comparison(
    result_mlmc: VolatilitySurfaceResult,
    result_laplace: VolatilitySurfaceResult,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Overlay both surfaces as wireframes for direct comparison.
    
    This style matches the paper's Figure 1(b) / Figure 5(a).
    
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
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    T_mesh, S_mesh = np.meshgrid(result_mlmc.t_grid, result_mlmc.s_grid, indexing='ij')
    
    # Subsample for cleaner wireframe
    step_t = max(1, len(result_mlmc.t_grid) // 10)
    step_s = max(1, len(result_mlmc.s_grid) // 15)
    
    T_sub = T_mesh[::step_t, ::step_s]
    S_sub = S_mesh[::step_t, ::step_s]
    
    Z_mlmc = np.nan_to_num(
        result_mlmc.b_squared_values[::step_t, ::step_s],
        nan=0
    )
    Z_laplace = np.nan_to_num(
        result_laplace.b_squared_values[::step_t, ::step_s],
        nan=0
    )
    
    # Plot wireframes
    ax.plot_wireframe(
        S_sub, T_sub, Z_mlmc,
        color='blue', linewidth=0.6, alpha=0.8,
        label='MLMC'
    )
    ax.plot_wireframe(
        S_sub, T_sub, Z_laplace,
        color='red', linewidth=0.6, alpha=0.6,
        label='Laplace'
    )
    
    ax.set_xlabel('Basket Value (s)', fontsize=11, labelpad=10)
    ax.set_ylabel('Time (t)', fontsize=11, labelpad=10)
    ax.set_zlabel(r'$\bar{b}^2(t, s)$', fontsize=11, labelpad=5)
    ax.set_title(
        r'Wireframe Comparison: $\bar{b}^2(t, s)$',
        fontsize=12, pad=10
    )
    
    ax.legend(fontsize=10)
    ax.view_init(elev=25, azim=-55)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig
