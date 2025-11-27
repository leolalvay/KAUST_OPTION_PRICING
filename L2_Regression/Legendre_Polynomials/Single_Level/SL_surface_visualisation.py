"""
This file is based on DM_SL_volsurf.py from Amelie's work.

3D visualisation of fitted local volatility surfaces using Legendre polynomial
regression. Demonstrates how surface quality varies with polynomial degree.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SL_legendre_utilities import (
    GBM_paths,
    scalings_l0,
    tot_degree_poly,
    normaleq_components_SL,
    fit_local_vol,
    make_b_bar
)


def plot_local_volatility(b_bar, basket, t, K=40, L=150, title=None, 
                          color='blue', save_path=None):
    """
    Create 3D wireframe plot of local volatility surface.
    
    Parameters
    ----------
    b_bar : callable
        Local volatility function from make_b_bar
    basket : ndarray, shape (M_0, N_t)
        Basket paths from pilot run (for percentile bounds)
    t : ndarray, shape (N_t,)
        Time grid
    K : int, optional
        Number of time grid points for plotting (default: 40)
    L : int, optional
        Number of space grid points per time slice (default: 150)
    title : str, optional
        Plot title (default: None)
    color : str, optional
        Wireframe colour (default: 'blue')
    save_path : str, optional
        If provided, save figure to this path (default: None)
    
    Notes
    -----
    Uses time-dependent domain bounds based on basket percentiles.
    This accounts for drift in the basket value over time.
    """
    # Compute time-dependent bounds from basket distribution
    s_min = np.percentile(basket, 1, axis=0)  # 1st percentile at each time
    s_max = np.percentile(basket, 99, axis=0)  # 99th percentile at each time
    
    # Select K time points for plotting
    t_plot = np.linspace(0, t[-1], K)
    idx = np.searchsorted(t, t_plot)  # Find nearest indices in original grid
    
    # Build 2D meshgrid for surface evaluation
    TT = np.zeros((L, K))
    SS = np.zeros((L, K))
    for j, ti in enumerate(idx):
        TT[:, j] = t_plot[j]
        SS[:, j] = np.linspace(s_min[ti], s_max[ti], L)
    
    # Evaluate local volatility on grid
    bbar = b_bar(TT, SS)
    
    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(TT, SS, bbar, 
                      rcount=40, ccount=40,
                      color=color,
                      linewidth=0.5)
    
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Basket $S$')
    ax.set_zlabel(r'$\bar{b}(t,S)$')
    
    if title:
        ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax


if __name__ == "__main__":
    # ========================================================================
    # Model Parameters
    # ========================================================================
    
    # Basket configuration
    d = 3  # Number of assets
    P1 = np.ones(d) / d  # Equal-weighted basket
    
    # Market parameters
    r = 0.05  # Risk-free rate
    x0 = np.linspace(225, 275, num=d)[:, np.newaxis]  # Initial asset values
    vol = np.array([0.2, 0.15, 0.1])  # Asset volatilities
    cov_mat = np.array([[1.0, 0.8, 0.3],
                        [0.8, 1.0, 0.1],
                        [0.3, 0.1, 1.0]])  # Correlation matrix
    
    # Time discretisation
    T = 1.0  # Time horizon
    dt = 0.005  # Time step
    N_t = int(T / dt)  # Number of time steps
    M_t = 400  # Number of training paths
    
    t = np.linspace(0, T, N_t)
    
    # ========================================================================
    # Comparative Study: Polynomial Degree Analysis
    # ========================================================================
    
    print("=" * 60)
    print("Volatility Surface Visualisation: Polynomial Degree Study")
    print("=" * 60)
    
    max_degrees = [3, 2, 1, 0]  # Test various polynomial degrees
    
    for max_deg in max_degrees:
        print(f"\nProcessing max_deg = {max_deg}...")
        
        # Generate basis function pairs
        pairs = tot_degree_poly(max_deg)
        print(f"  Number of basis functions: {len(pairs)}")
        
        # Pilot run for domain scaling
        s_min0, s_max0, basket0 = scalings_l0(x0, T, dt, r, cov_mat, vol, P1, 
                                              M_0=10000)
        
        # Generate training paths
        paths = GBM_paths(x0, r, vol, cov_mat, dt, N_t, M_t)
        
        # Build regression system
        D, psi = normaleq_components_SL(paths, P1, pairs, cov_mat, vol, 
                                        s_min0, s_max0, T)
        
        # Solve for coefficients
        c = fit_local_vol(D, psi)
        
        # Create callable volatility function
        b_bar = make_b_bar(c, pairs, s_min0, s_max0, T, max_deg)
        
        # Compute regression quality metrics
        psi_fit = D @ c.reshape(-1, 1)
        residual = np.linalg.norm(psi - psi_fit) / np.linalg.norm(psi)
        cond_D = np.linalg.cond(D)
        
        print(f"  Condition number: {cond_D:.2e}")
        print(f"  Relative residual: {residual * 100:.2f}%")
        
        # Create surface plot
        fig, ax = plot_local_volatility(
            b_bar, basket0, t,
            K=50, L=150,
            title=f'Local Volatility Surface (max degree = {max_deg})',
            color=f'C{max_deg}',
            save_path=f"VolSurf_maxdeg{max_deg}.pdf"
        )
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Visualisation complete!")
    print("Saved: VolSurf_maxdeg{0,1,2,3}.pdf")
    print("=" * 60)
