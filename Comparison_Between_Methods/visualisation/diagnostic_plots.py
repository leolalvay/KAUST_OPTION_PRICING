"""
Diagnostic Plotting Utilities for MLMC Volatility Estimation

Provides comprehensive diagnostic plots for:
- MLMC convergence analysis (Giles-style)
- Statistical uncertainty from multiple runs
- Method agreement (MLMC vs Laplace)
- Additional diagnostics (VRF, kurtosis, GCI)

All plots use proper LaTeX-formatted mathematics.

Author: Wadoud (KAUST Internship)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import Optional, Tuple, Dict, Any
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable LaTeX-style rendering
plt.rcParams.update({
    'text.usetex': False,  # Use mathtext instead of full LaTeX
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
})


def plot_mlmc_convergence_diagnostics(
    level_diagnostics: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create Giles-style MLMC convergence diagnostic panel.

    Four subplots showing:
    1. Weak convergence: log₂|m_ℓ| vs ℓ with fitted slope -α
    2. Variance decay: log₂(V_ℓ) vs ℓ with fitted slope -β
    3. Level corrections bar chart
    4. Correlation ρ_ℓ vs level

    Parameters
    ----------
    level_diagnostics : dict
        Output from compute_mlmc_level_diagnostics().
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
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    variances = level_diagnostics['variances']
    mean_corrections = level_diagnostics['mean_corrections']
    correlations = level_diagnostics['correlations']
    alpha = level_diagnostics['alpha']
    beta = level_diagnostics['beta']
    richardson_bias = level_diagnostics['richardson_bias']

    levels = sorted(variances.keys())
    higher_levels = [l for l in levels if l >= 1]

    # Panel 1: Weak convergence (log₂|m_ℓ| vs ℓ)
    ax1 = axes[0, 0]
    log2_m = []
    lvls_m = []
    for l in levels:
        m_l = np.abs(mean_corrections[l])
        if m_l > 1e-15:
            log2_m.append(np.log2(m_l))
            lvls_m.append(l)

    if len(lvls_m) > 0:
        ax1.plot(lvls_m, log2_m, 'bo-', markersize=8, linewidth=2, label='Data')

        # Fitted line (for l >= 1)
        if not np.isnan(alpha) and len([l for l in lvls_m if l >= 1]) >= 2:
            l_fit = np.array([l for l in lvls_m if l >= 1])
            log2_m_fit = np.array([log2_m[i] for i, l in enumerate(lvls_m) if l >= 1])
            if len(l_fit) >= 2:
                # Extrapolate fit line
                l_line = np.linspace(min(l_fit), max(l_fit), 50)
                intercept = np.mean(log2_m_fit) + alpha * np.mean(l_fit)
                fit_line = intercept - alpha * l_line
                ax1.plot(l_line, fit_line, 'r--', linewidth=1.5,
                         label=rf'Fit: $\alpha = {alpha:.2f}$')

    ax1.set_xlabel(r'Level $\ell$')
    ax1.set_ylabel(r'$\log_2 |m_\ell|$')
    ax1.set_title(r'Weak Convergence: $|m_\ell| = \|b^2_\ell - b^2_{\ell-1}\|$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(levels)

    # Panel 2: Variance decay (log₂V_ℓ vs ℓ)
    ax2 = axes[0, 1]
    log2_v = []
    lvls_v = []
    for l in higher_levels:
        v_l = variances[l]
        if v_l > 1e-15:
            log2_v.append(np.log2(v_l))
            lvls_v.append(l)

    if len(lvls_v) > 0:
        ax2.plot(lvls_v, log2_v, 'go-', markersize=8, linewidth=2, label='Data')

        # Fitted line
        if not np.isnan(beta) and len(lvls_v) >= 2:
            l_line = np.linspace(min(lvls_v), max(lvls_v), 50)
            intercept = np.mean(log2_v) + beta * np.mean(lvls_v)
            fit_line = intercept - beta * l_line
            ax2.plot(l_line, fit_line, 'r--', linewidth=1.5,
                     label=rf'Fit: $\beta = {beta:.2f}$')

    ax2.set_xlabel(r'Level $\ell$')
    ax2.set_ylabel(r'$\log_2 V_\ell$')
    ax2.set_title(r'Variance Decay: $V_\ell = \mathrm{Var}[P_\ell - P_{\ell-1}]$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if len(lvls_v) > 0:
        ax2.set_xticks(lvls_v)

    # Panel 3: Level corrections bar chart
    ax3 = axes[1, 0]
    corrections = [mean_corrections[l] for l in levels]
    colors = ['C0' if c >= 0 else 'C3' for c in corrections]
    bars = ax3.bar(levels, corrections, color=colors, edgecolor='black', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel(r'Level $\ell$')
    ax3.set_ylabel(r'Mean correction $m_\ell$')
    ax3.set_title('Level Corrections (Telescoping Sum)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(levels)

    # Add Richardson bias annotation
    if not np.isnan(richardson_bias):
        ax3.annotate(
            rf'Richardson bias $\approx {richardson_bias:.2e}$',
            xy=(0.95, 0.95), xycoords='axes fraction',
            ha='right', va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    # Panel 4: Correlation ρ_ℓ vs level
    ax4 = axes[1, 1]
    corr_values = [correlations[l] for l in higher_levels]
    ax4.plot(higher_levels, corr_values, 'mo-', markersize=8, linewidth=2)
    ax4.axhline(y=0.95, color='r', linestyle='--', linewidth=1.5,
                label=r'Threshold $\rho = 0.95$')
    ax4.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax4.set_xlabel(r'Level $\ell$')
    ax4.set_ylabel(r'Correlation $\rho_\ell$')
    ax4.set_title(r'Coupling Correlation: $\rho_\ell = \mathrm{Corr}(P_\ell, P_{\ell-1})$')
    ax4.set_ylim([min(0.7, min(corr_values) - 0.05) if corr_values else 0.7, 1.02])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    if len(higher_levels) > 0:
        ax4.set_xticks(higher_levels)

    # Overall title with convergence quality
    quality = level_diagnostics['convergence_quality']
    quality_color = {'good': 'green', 'acceptable': 'orange', 'poor': 'red'}[quality]
    fig.suptitle(
        f'MLMC Convergence Diagnostics (Quality: {quality.upper()})',
        fontsize=13, fontweight='bold', color=quality_color
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_additional_diagnostics(
    level_diagnostics: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create summary plot of additional diagnostics from Error Analysis.md.

    Four subplots showing:
    1. VRF per level with threshold at 10
    2. Kurtosis per level with threshold at 100
    3. Summary table with pass/fail indicators
    4. Convergence rates summary

    Parameters
    ----------
    level_diagnostics : dict
        Output from compute_mlmc_level_diagnostics().
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
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    vrfs = level_diagnostics['variance_reduction_factors']
    kurtosis = level_diagnostics['kurtosis_values']
    correlations = level_diagnostics['correlations']
    alpha = level_diagnostics['alpha']
    beta = level_diagnostics['beta']
    warnings = level_diagnostics['warnings']

    levels = sorted(vrfs.keys())
    higher_levels = [l for l in levels if l >= 1]

    # Panel 1: VRF per level
    ax1 = axes[0, 0]
    vrf_values = [min(vrfs[l], 100) for l in higher_levels]  # Cap for display
    vrf_colors = ['green' if vrfs[l] >= 10 else 'red' for l in higher_levels]
    bars1 = ax1.bar(higher_levels, vrf_values, color=vrf_colors, edgecolor='black', alpha=0.7)
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Threshold (VRF = 10)')
    ax1.set_xlabel(r'Level $\ell$')
    ax1.set_ylabel('VRF')
    ax1.set_title(r'Variance Reduction Factor: $\mathrm{VRF}_\ell = \frac{\mathrm{Var}[P_\ell] + \mathrm{Var}[P_{\ell-1}]}{V_\ell}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    if len(higher_levels) > 0:
        ax1.set_xticks(higher_levels)

    # Add actual values as annotations
    for i, (l, v) in enumerate(zip(higher_levels, vrf_values)):
        actual_vrf = vrfs[l]
        label = f'{actual_vrf:.1f}' if actual_vrf < 1000 else f'{actual_vrf:.0e}'
        ax1.annotate(label, (l, v), ha='center', va='bottom', fontsize=8)

    # Panel 2: Kurtosis per level
    ax2 = axes[0, 1]
    kurt_values = [kurtosis[l] for l in levels]
    kurt_colors = ['green' if abs(k) < 100 else 'red' for k in kurt_values]
    bars2 = ax2.bar(levels, kurt_values, color=kurt_colors, edgecolor='black', alpha=0.7)
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label=r'Threshold ($\kappa = 100$)')
    ax2.axhline(y=-100, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xlabel(r'Level $\ell$')
    ax2.set_ylabel(r'Excess Kurtosis $\kappa_\ell$')
    ax2.set_title(r'Kurtosis: $\kappa_\ell$ (should be $< 100$)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(levels)

    # Panel 3: Summary table
    ax3 = axes[1, 0]
    ax3.axis('off')

    # Create summary data
    summary_data = []
    summary_data.append(['Metric', 'Value', 'Status'])
    summary_data.append([r'$\alpha$ (weak conv.)', f'{alpha:.2f}' if not np.isnan(alpha) else 'N/A',
                         'OK' if not np.isnan(alpha) and alpha >= 0.5 else 'WARN'])
    summary_data.append([r'$\beta$ (var. decay)', f'{beta:.2f}' if not np.isnan(beta) else 'N/A',
                         'OK' if not np.isnan(beta) and beta >= 1.0 else 'WARN'])

    for l in higher_levels:
        summary_data.append([f'VRF (level {l})', f'{vrfs[l]:.1f}',
                             'OK' if vrfs[l] >= 10 else 'FAIL'])
        summary_data.append([f'Corr (level {l})', f'{correlations[l]:.3f}',
                             'OK' if correlations[l] >= 0.95 else 'WARN'])

    for l in levels:
        kurt_status = 'OK' if abs(kurtosis[l]) < 100 else 'FAIL'
        summary_data.append([f'Kurtosis (level {l})', f'{kurtosis[l]:.1f}', kurt_status])

    # Create table
    table = ax3.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.3, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color code status column
    for i in range(1, len(summary_data)):
        status = summary_data[i][2]
        cell = table[(i, 2)]
        if status == 'OK':
            cell.set_facecolor('lightgreen')
        elif status == 'WARN':
            cell.set_facecolor('lightyellow')
        else:
            cell.set_facecolor('lightcoral')

    ax3.set_title('Diagnostic Summary', fontsize=11, fontweight='bold', pad=20)

    # Panel 4: Warnings and convergence quality
    ax4 = axes[1, 1]
    ax4.axis('off')

    quality = level_diagnostics['convergence_quality']
    quality_color = {'good': 'green', 'acceptable': 'orange', 'poor': 'red'}[quality]

    text_lines = [
        f"Convergence Quality: {quality.upper()}",
        "",
        "Warnings:" if warnings else "No warnings.",
    ]
    for w in warnings[:8]:  # Limit to 8 warnings
        text_lines.append(f"  - {w}")
    if len(warnings) > 8:
        text_lines.append(f"  ... and {len(warnings) - 8} more")

    text_lines.extend([
        "",
        "Thresholds:",
        r"  - VRF > 10 for effective coupling",
        r"  - $\rho_\ell > 0.95$ for good correlation",
        r"  - $|\kappa_\ell| < 100$ for reliability",
        r"  - $\alpha > 0.5$ for weak convergence",
        r"  - $\beta > 1.0$ for variance decay",
    ])

    ax4.text(0.05, 0.95, '\n'.join(text_lines),
             transform=ax4.transAxes,
             fontsize=9,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.set_title('Diagnostic Checks', fontsize=11, fontweight='bold', pad=20)

    fig.suptitle(
        'Additional MLMC Diagnostics (Error Analysis)',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_statistical_uncertainty(
    uncertainty_stats: Dict[str, Any],
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot statistical uncertainty from multiple MLMC runs.

    Two subplots showing:
    1. Standard error convergence vs number of runs (should follow 1/√n)
    2. Relative standard error heatmap on the surface

    Parameters
    ----------
    uncertainty_stats : dict
        Output from compute_across_run_uncertainty().
    t_grid, s_grid : np.ndarray
        Grid points for plotting.
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

    n_runs = uncertainty_stats['n_runs']
    running_stats = uncertainty_stats['running_stats']

    # Panel 1: SE convergence
    ax1 = axes[0]
    n_values = [s['n'] for s in running_stats]
    se_values = [s['mean_relative_se'] for s in running_stats]

    ax1.semilogy(n_values, se_values, 'bo-', markersize=6, linewidth=2, label='Observed SE')

    # Theoretical 1/sqrt(n) line
    if len(n_values) > 0 and len(se_values) > 0 and not np.isnan(se_values[-1]):
        n_theory = np.linspace(2, max(n_values), 50)
        # Fit scale from final point
        scale = se_values[-1] * np.sqrt(max(n_values))
        se_theory = scale / np.sqrt(n_theory)
        ax1.semilogy(n_theory, se_theory, 'r--', linewidth=1.5,
                     label=r'Theory: $\propto 1/\sqrt{n}$')

    ax1.set_xlabel('Number of runs $n$')
    ax1.set_ylabel('Mean relative SE')
    ax1.set_title(r'Statistical Uncertainty: $\mathrm{SE} = \sigma / \sqrt{n}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # Panel 2: Relative SE heatmap
    ax2 = axes[1]
    rel_se = uncertainty_stats.get('standard_error', np.zeros_like(uncertainty_stats['mean_surface']))
    mean_surf = uncertainty_stats['mean_surface']

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_se_pct = np.where(
            np.abs(mean_surf) > 1e-10,
            100 * rel_se / np.abs(mean_surf),
            np.nan
        )

    extent = [s_grid.min(), s_grid.max(), t_grid.min(), t_grid.max()]
    im = ax2.imshow(
        rel_se_pct,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='YlOrRd',
        vmin=0,
        vmax=np.nanpercentile(rel_se_pct, 95)
    )
    ax2.set_xlabel('Basket Value $s$')
    ax2.set_ylabel('Time $t$')
    ax2.set_title(f'Relative Standard Error (%) - {n_runs} runs')
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Relative SE (%)')

    fig.suptitle(
        'Across-Run Statistical Uncertainty Analysis',
        fontsize=12, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_method_agreement(
    b2_mlmc: np.ndarray,
    b2_laplace: np.ndarray,
    t_grid: np.ndarray,
    s_grid: np.ndarray,
    metrics: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot method agreement between MLMC and Laplace.

    IMPORTANT: This shows DISAGREEMENT, not ERROR. Neither method is ground truth.

    Two subplots showing:
    1. Scatter plot with correlation
    2. Difference heatmap

    Parameters
    ----------
    b2_mlmc, b2_laplace : np.ndarray
        b² surfaces from each method.
    t_grid, s_grid : np.ndarray
        Grid points.
    metrics : dict, optional
        Pre-computed metrics from compute_method_agreement().
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
    from methods.common import compute_method_agreement

    if metrics is None:
        metrics = compute_method_agreement(b2_mlmc, b2_laplace)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Scatter plot
    ax1 = axes[0]
    mlmc_flat = b2_mlmc.flatten()
    laplace_flat = b2_laplace.flatten()
    valid = np.isfinite(mlmc_flat) & np.isfinite(laplace_flat)
    mlmc_valid = mlmc_flat[valid]
    laplace_valid = laplace_flat[valid]

    # Subsample if too many points
    if len(mlmc_valid) > 3000:
        idx = np.random.choice(len(mlmc_valid), 3000, replace=False)
        mlmc_plot = mlmc_valid[idx]
        laplace_plot = laplace_valid[idx]
    else:
        mlmc_plot = mlmc_valid
        laplace_plot = laplace_valid

    ax1.scatter(laplace_plot, mlmc_plot, alpha=0.4, s=15, c='steelblue', edgecolor='none')

    # Diagonal line
    lims = [min(mlmc_plot.min(), laplace_plot.min()),
            max(mlmc_plot.max(), laplace_plot.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect Agreement')

    # Correlation annotation
    corr = metrics['correlation']
    l2_disagree = metrics.get('l2_disagreement', metrics.get('l2_relative_error', np.nan))
    ax1.text(
        0.05, 0.95,
        f'Correlation: {corr:.4f}\n$L^2$ disagreement: {l2_disagree:.4f}',
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax1.set_xlabel(r'Laplace $\bar{b}^2$')
    ax1.set_ylabel(r'MLMC $\bar{b}^2$')
    ax1.set_title('Pointwise Comparison')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Panel 2: Difference heatmap
    ax2 = axes[1]
    diff = b2_mlmc - b2_laplace
    extent = [s_grid.min(), s_grid.max(), t_grid.min(), t_grid.max()]

    vmax = np.nanmax(np.abs(diff))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax2.imshow(
        diff,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='RdBu_r',
        norm=norm
    )
    ax2.set_xlabel('Basket Value $s$')
    ax2.set_ylabel('Time $t$')
    ax2.set_title(r'Difference: $\bar{b}^2_{\mathrm{MLMC}} - \bar{b}^2_{\mathrm{Laplace}}$')
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Difference')

    fig.suptitle(
        'Method Agreement (MLMC vs Laplace)\n'
        r'$\mathbf{NOTE:}$ Neither method is ground truth',
        fontsize=12, fontweight='bold', y=1.05
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_l2_disagreement_vs_runs(
    uncertainty_stats: Dict[str, Any],
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot L² disagreement between running MLMC average and Laplace vs number of runs.

    This replaces the old "error" plot with proper terminology.

    Parameters
    ----------
    uncertainty_stats : dict
        Output from compute_across_run_uncertainty() with reference_surface provided.
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

    if 'l2_disagreement_vs_n' not in uncertainty_stats:
        ax.text(0.5, 0.5, 'No reference surface provided\nfor disagreement calculation',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig

    l2_disagree = uncertainty_stats['l2_disagreement_vs_n']
    n_runs = len(l2_disagree)
    n_values = list(range(1, n_runs + 1))

    ax.semilogy(n_values, l2_disagree, 'b-', linewidth=2, marker='o', markersize=5)

    ax.set_xlabel('Number of MLMC runs $n$')
    ax.set_ylabel(r'$L^2$ disagreement (MLMC mean vs Laplace)')
    ax.set_title(
        r'Running Average Disagreement: $\|\bar{b}^2_n - \bar{b}^2_{\mathrm{Laplace}}\|_{L^2}$'
        + '\n(Neither method is ground truth)'
    )
    ax.grid(True, alpha=0.3, which='both')

    # Annotate final value
    ax.annotate(
        f'Final: {l2_disagree[-1]:.4f}',
        xy=(n_runs, l2_disagree[-1]),
        xytext=(n_runs * 0.7, l2_disagree[-1] * 1.5),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig
