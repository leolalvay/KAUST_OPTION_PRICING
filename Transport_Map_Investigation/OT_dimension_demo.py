#!/usr/bin/env python3
"""
OT Dimension Demonstration Script
=================================

This script demonstrates the dimensional flow through:
1. d-dimensional GBM path generation
2. d-dimensional Gaussian-Brenier optimal transport coupling
3. d → 1 Markovian projection

The key insight: OT operates in full d-dimensional space BEFORE projection to 1D.

Author: Wadoud Charbak (KAUST Intern)
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable

np.random.seed(42)  # For reproducibility


# =============================================================================
# PART 1: GBM Path Generation (d-dimensional)
# =============================================================================

def generate_GBM_paths(
    x0: np.ndarray,      # Initial prices, shape (d,)
    r: float,            # Risk-free rate
    vol: np.ndarray,     # Volatilities, shape (d,)
    cov_mat: np.ndarray, # Correlation matrix, shape (d, d)
    dt: float,           # Timestep
    N_steps: int,        # Number of timesteps
    M_paths: int         # Number of paths
) -> np.ndarray:
    """
    Generate d-dimensional correlated GBM paths.
    
    Returns
    -------
    paths : ndarray, shape (M_paths, N_steps, d)
        Simulated asset price paths
    """
    d = len(x0)
    chol = np.linalg.cholesky(cov_mat)
    sqrt_dt = np.sqrt(dt)
    
    paths = np.zeros((M_paths, N_steps, d))
    paths[:, 0, :] = x0
    
    X = np.tile(x0, (M_paths, 1))  # Shape: (M, d)
    
    for n in range(1, N_steps):
        Z = np.random.randn(M_paths, d)
        dW = (Z @ chol.T) * sqrt_dt  # Correlated increments
        sigma = X * vol  # Diagonal volatility scaling
        X = X + r * X * dt + sigma * dW
        paths[:, n, :] = X
    
    return paths


def generate_coupled_paths(
    x0: np.ndarray,
    r: float,
    vol: np.ndarray,
    cov_mat: np.ndarray,
    dt_fine: float,
    N_fine: int,
    M_paths: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate coupled fine and coarse paths for MLMC.
    
    Coarse paths use dt_coarse = 2 * dt_fine with summed Brownian increments.
    
    Returns
    -------
    paths_fine : ndarray, shape (M_paths, N_fine, d)
    paths_coarse : ndarray, shape (M_paths, N_coarse, d)
    """
    d = len(x0)
    chol = np.linalg.cholesky(cov_mat)
    sqrt_dt_fine = np.sqrt(dt_fine)
    
    N_coarse = N_fine // 2
    dt_coarse = 2 * dt_fine
    
    # Generate all fine random increments
    Z_fine = np.random.randn(M_paths, N_fine, d)
    
    # Fine paths
    paths_fine = np.zeros((M_paths, N_fine, d))
    paths_fine[:, 0, :] = x0
    X_fine = np.tile(x0, (M_paths, 1))
    
    for n in range(1, N_fine):
        dW = (Z_fine[:, n-1, :] @ chol.T) * sqrt_dt_fine
        sigma = X_fine * vol
        X_fine = X_fine + r * X_fine * dt_fine + sigma * dW
        paths_fine[:, n, :] = X_fine
    
    # Coarse paths with COUPLED increments (sum consecutive pairs)
    Z_coarse = Z_fine.reshape(M_paths, N_coarse, 2, d).sum(axis=2)
    
    paths_coarse = np.zeros((M_paths, N_coarse, d))
    paths_coarse[:, 0, :] = x0
    X_coarse = np.tile(x0, (M_paths, 1))
    
    for n in range(1, N_coarse):
        # Note: sqrt(dt_fine) because Z_coarse has variance 2, not 1
        dW = (Z_coarse[:, n-1, :] @ chol.T) * sqrt_dt_fine
        sigma = X_coarse * vol
        X_coarse = X_coarse + r * X_coarse * dt_coarse + sigma * dW
        paths_coarse[:, n, :] = X_coarse
    
    return paths_fine, paths_coarse


# =============================================================================
# PART 2: Gaussian-Brenier Optimal Transport (d × d dimensional)
# =============================================================================

class GaussianBrenierMap:
    """
    Optimal transport map between two Gaussian distributions.
    
    Maps N(μ_f, C_f) to N(μ_c, C_c) via:
        T(x) = μ_c + A(x - μ_f)
    
    where A = C_f^{-1/2} (C_f^{1/2} C_c C_f^{1/2})^{1/2} C_f^{-1/2}
    
    KEY: All matrices are d × d, operating in full asset space!
    """
    
    def __init__(self, mu_f: np.ndarray, C_f: np.ndarray, 
                 mu_c: np.ndarray, C_c: np.ndarray):
        """
        Parameters
        ----------
        mu_f : ndarray, shape (d,)
            Mean of fine (source) distribution
        C_f : ndarray, shape (d, d)
            Covariance of fine distribution
        mu_c : ndarray, shape (d,)
            Mean of coarse (target) distribution
        C_c : ndarray, shape (d, d)
            Covariance of coarse distribution
        """
        self.mu_f = np.asarray(mu_f)
        self.mu_c = np.asarray(mu_c)
        self.C_f = np.asarray(C_f)
        self.C_c = np.asarray(C_c)
        
        # Store dimensions for inspection
        self.d = len(mu_f)
        
        # Compute A via eigendecomposition (numerically stable)
        eig_f, U_f = np.linalg.eigh(self.C_f)
        eig_f = np.maximum(eig_f, 1e-10)  # Regularise
        
        sqrt_C_f = U_f @ np.diag(np.sqrt(eig_f)) @ U_f.T
        invsqrt_C_f = U_f @ np.diag(1.0 / np.sqrt(eig_f)) @ U_f.T
        
        # Middle matrix M = C_f^{1/2} C_c C_f^{1/2}
        M = sqrt_C_f @ self.C_c @ sqrt_C_f
        
        eig_M, U_M = np.linalg.eigh(M)
        eig_M = np.maximum(eig_M, 1e-10)
        sqrt_M = U_M @ np.diag(np.sqrt(eig_M)) @ U_M.T
        
        # Brenier transport matrix
        self.A = invsqrt_C_f @ sqrt_M @ invsqrt_C_f
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the transport map.
        
        Parameters
        ----------
        x : ndarray, shape (M, d) or (d,)
            Points in d-dimensional space
            
        Returns
        -------
        y : ndarray, same shape as x
            Transported points (still d-dimensional!)
        """
        return self.mu_c + (x - self.mu_f) @ self.A.T
    
    def print_dimensions(self):
        """Print all dimensions for clarity."""
        print(f"\n{'='*60}")
        print("GAUSSIAN-BRENIER MAP DIMENSIONS")
        print(f"{'='*60}")
        print(f"  Asset dimension d = {self.d}")
        print(f"  μ_f shape: {self.mu_f.shape}  (d-dimensional mean)")
        print(f"  μ_c shape: {self.mu_c.shape}  (d-dimensional mean)")
        print(f"  C_f shape: {self.C_f.shape}  (d × d covariance)")
        print(f"  C_c shape: {self.C_c.shape}  (d × d covariance)")
        print(f"  A shape:   {self.A.shape}  (d × d transport matrix)")
        print(f"{'='*60}\n")


def identity_map(x: np.ndarray) -> np.ndarray:
    """Identity map for t=0 (no transport needed)."""
    return x


def compute_OT_maps(
    paths_fine: np.ndarray,    # Shape: (M, N_f, d)
    paths_coarse: np.ndarray,  # Shape: (M, N_c, d)
    verbose: bool = True
) -> List[Callable]:
    """
    Compute Gaussian-Brenier maps at each coarse timestep.
    
    Works in LOG-SPACE because log(GBM) is Gaussian.
    
    Returns
    -------
    maps : list of callable
        maps[n] transforms log(X_f) to estimated log(X_c) at timestep n
    """
    M, N_f, d = paths_fine.shape
    N_c = paths_coarse.shape[1]
    
    # Extract fine paths at coarse time intervals
    paths_fine_reduced = paths_fine[:, ::2, :]  # Shape: (M, N_c, d)
    
    # Transform to log-space
    log_fine = np.log(paths_fine_reduced)
    log_coarse = np.log(paths_coarse)
    
    if verbose:
        print(f"\n{'='*60}")
        print("COMPUTING OPTIMAL TRANSPORT MAPS")
        print(f"{'='*60}")
        print(f"  Fine paths shape:   {paths_fine.shape}")
        print(f"  Coarse paths shape: {paths_coarse.shape}")
        print(f"  Log-fine (reduced): {log_fine.shape}")
        print(f"  Log-coarse:         {log_coarse.shape}")
        print(f"  Asset dimension d = {d}")
        print(f"  Number of maps to compute: {N_c}")
    
    # Compute means and covariances at each timestep
    mu_fine = np.mean(log_fine, axis=0)    # Shape: (N_c, d)
    mu_coarse = np.mean(log_coarse, axis=0)  # Shape: (N_c, d)
    
    # Covariances at each timestep
    C_fine = np.array([np.cov(log_fine[:, n, :].T) for n in range(N_c)])
    C_coarse = np.array([np.cov(log_coarse[:, n, :].T) for n in range(N_c)])
    
    if verbose:
        print(f"\n  Statistics computed:")
        print(f"    μ_fine shape:  {mu_fine.shape}   (N_c × d)")
        print(f"    μ_coarse shape: {mu_coarse.shape}  (N_c × d)")
        print(f"    C_fine shape:  {C_fine.shape}  (N_c × d × d)")
        print(f"    C_coarse shape: {C_coarse.shape} (N_c × d × d)")
    
    # Construct maps
    maps = [identity_map]  # t=0: no transformation
    
    for n in range(1, N_c):
        brenier_map = GaussianBrenierMap(
            mu_fine[n], C_fine[n],
            mu_coarse[n], C_coarse[n]
        )
        maps.append(brenier_map)
    
    if verbose:
        print(f"\n  Constructed {len(maps)} transport maps")
        print(f"  Each map: T: ℝ^{d} → ℝ^{d}")
        print(f"{'='*60}\n")
    
    return maps


def apply_OT_maps(
    paths_fine: np.ndarray,  # Shape: (M, N_f, d)
    maps: List[Callable],
    verbose: bool = True
) -> np.ndarray:
    """
    Apply OT maps to transform fine paths to estimated coarse paths.
    
    Returns
    -------
    paths_coarse_est : ndarray, shape (M, N_c, d)
        Estimated coarse paths (still d-dimensional!)
    """
    M, N_f, d = paths_fine.shape
    N_c = len(maps)
    
    # Extract fine paths at coarse intervals
    paths_fine_reduced = paths_fine[:, ::2, :]
    
    # Transform to log-space
    log_fine = np.log(paths_fine_reduced)
    
    if verbose:
        print(f"\n{'='*60}")
        print("APPLYING OPTIMAL TRANSPORT MAPS")
        print(f"{'='*60}")
        print(f"  Input (fine paths):     shape {paths_fine.shape}")
        print(f"  Input (log, reduced):   shape {log_fine.shape}")
    
    # Apply maps at each timestep
    log_coarse_est = np.zeros_like(log_fine)
    
    for n in range(N_c):
        log_coarse_est[:, n, :] = maps[n](log_fine[:, n, :])
    
    # Transform back to price space
    paths_coarse_est = np.exp(log_coarse_est)
    
    if verbose:
        print(f"  After OT (log-space):   shape {log_coarse_est.shape}")
        print(f"  Output (price space):   shape {paths_coarse_est.shape}")
        print(f"\n  ⚠️  DIMENSION UNCHANGED: Still {d}-dimensional!")
        print(f"{'='*60}\n")
    
    return paths_coarse_est


# =============================================================================
# PART 3: Markovian Projection (d → 1)
# =============================================================================

def markovian_projection(
    paths: np.ndarray,        # Shape: (M, N, d)
    weights: np.ndarray,      # Shape: (d,)
    verbose: bool = True
) -> np.ndarray:
    """
    Project d-dimensional paths to 1D basket value.
    
    S(t) = weights · X(t) = Σᵢ wᵢ Xᵢ(t)
    
    THIS is where dimension reduction happens: d → 1
    
    Returns
    -------
    basket : ndarray, shape (M, N)
        1-dimensional basket value paths
    """
    M, N, d = paths.shape
    
    basket = paths @ weights  # (M, N, d) @ (d,) = (M, N)
    
    if verbose:
        print(f"\n{'='*60}")
        print("MARKOVIAN PROJECTION (d → 1)")
        print(f"{'='*60}")
        print(f"  Input paths:    shape {paths.shape}  (M × N × d)")
        print(f"  Basket weights: shape {weights.shape}  (d,)")
        print(f"  Output basket:  shape {basket.shape}  (M × N)")
        print(f"\n  ✓ DIMENSION REDUCED: {d}D → 1D")
        print(f"{'='*60}\n")
    
    return basket


# =============================================================================
# PART 4: Demonstration Script
# =============================================================================

def demonstrate_dimensions():
    """
    Full demonstration of dimensional flow through OT and Markovian projection.
    """
    print("\n" + "="*70)
    print(" OPTIMAL TRANSPORT DIMENSION DEMONSTRATION")
    print(" Showing: d-dim OT coupling BEFORE d→1 Markovian projection")
    print("="*70)
    
    # =========================================================================
    # Setup: d = 3 assets
    # =========================================================================
    d = 3
    x0 = np.array([100.0, 100.0, 100.0])
    r = 0.05
    vol = np.array([0.2, 0.15, 0.25])
    cov_mat = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    basket_weights = np.ones(d) / d  # Equal-weighted
    
    T = 1.0
    dt_fine = 0.01
    N_fine = int(T / dt_fine)
    M_paths = 5000
    
    print(f"\n📊 SETUP")
    print(f"   Asset dimension d = {d}")
    print(f"   Initial prices: {x0}")
    print(f"   Volatilities: {vol}")
    print(f"   Correlation matrix shape: {cov_mat.shape}")
    print(f"   Basket weights: {basket_weights}")
    print(f"   Fine timesteps: {N_fine}, Coarse timesteps: {N_fine//2}")
    print(f"   Number of paths: {M_paths}")
    
    # =========================================================================
    # Step 1: Generate coupled paths (d-dimensional)
    # =========================================================================
    print(f"\n\n{'─'*70}")
    print("STEP 1: Generate d-dimensional coupled paths")
    print(f"{'─'*70}")
    
    paths_fine, paths_coarse = generate_coupled_paths(
        x0, r, vol, cov_mat, dt_fine, N_fine, M_paths
    )
    
    print(f"  Fine paths:   {paths_fine.shape}   (M × N_f × d)")
    print(f"  Coarse paths: {paths_coarse.shape}  (M × N_c × d)")
    print(f"  → Both are {d}-dimensional in asset space")
    
    # =========================================================================
    # Step 2: Compute OT maps (d × d matrices)
    # =========================================================================
    print(f"\n\n{'─'*70}")
    print("STEP 2: Compute Gaussian-Brenier transport maps")
    print(f"{'─'*70}")
    
    maps = compute_OT_maps(paths_fine, paths_coarse, verbose=True)
    
    # Show details of one map
    if len(maps) > 1 and isinstance(maps[1], GaussianBrenierMap):
        maps[1].print_dimensions()
    
    # =========================================================================
    # Step 3: Apply OT maps (still d-dimensional!)
    # =========================================================================
    print(f"\n\n{'─'*70}")
    print("STEP 3: Apply OT maps to fine paths")
    print(f"{'─'*70}")
    
    paths_coarse_est = apply_OT_maps(paths_fine, maps, verbose=True)
    
    # =========================================================================
    # Step 4: NOW do Markovian projection (d → 1)
    # =========================================================================
    print(f"\n\n{'─'*70}")
    print("STEP 4: Markovian projection (THIS is where d → 1 happens)")
    print(f"{'─'*70}")
    
    basket_fine = markovian_projection(
        paths_fine[:, ::2, :], basket_weights, verbose=True
    )
    basket_coarse = markovian_projection(
        paths_coarse, basket_weights, verbose=False
    )
    basket_coarse_est = markovian_projection(
        paths_coarse_est, basket_weights, verbose=False
    )
    
    print(f"  Basket (fine):       shape {basket_fine.shape}")
    print(f"  Basket (coarse):     shape {basket_coarse.shape}")
    print(f"  Basket (OT est):     shape {basket_coarse_est.shape}")
    
    # =========================================================================
    # Step 5: Validate OT coupling quality
    # =========================================================================
    print(f"\n\n{'─'*70}")
    print("STEP 5: Validate OT coupling quality")
    print(f"{'─'*70}")
    
    # Compare variance of differences
    diff_standard = basket_fine - basket_coarse
    diff_OT = basket_fine - basket_coarse_est
    
    var_standard = np.var(diff_standard[:, -1])
    var_OT = np.var(diff_OT[:, -1])
    
    print(f"\n  Terminal variance of (fine - coarse):")
    print(f"    Standard coupling: {var_standard:.6f}")
    print(f"    OT coupling:       {var_OT:.6f}")
    print(f"    Variance ratio:    {var_OT/var_standard:.4f}")
    
    if var_OT < var_standard:
        print(f"\n  ✓ OT coupling reduces variance by {(1-var_OT/var_standard)*100:.1f}%")
    
    # =========================================================================
    # Step 6: Summary diagram
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("SUMMARY: DIMENSIONAL FLOW")
    print(f"{'='*70}")
    print("""
    ┌────────────────────────────────────────────────────────────┐
    │  paths_fine ∈ ℝ^{M × N_f × d}                              │
    │  (5000 × 100 × 3) = 3-dimensional asset space              │
    └───────────────────────┬────────────────────────────────────┘
                            │
                            ↓  log transform
    ┌────────────────────────────────────────────────────────────┐
    │  log(paths_fine) ∈ ℝ^{M × N_c × d}                         │
    │  (5000 × 50 × 3) = STILL 3-dimensional!                    │
    └───────────────────────┬────────────────────────────────────┘
                            │
                            ↓  Brenier map T: ℝ³ → ℝ³
                            │  (using 3×3 transport matrix A)
    ┌────────────────────────────────────────────────────────────┐
    │  T(log(paths_fine)) ∈ ℝ^{M × N_c × d}                      │
    │  (5000 × 50 × 3) = STILL 3-dimensional!                    │
    └───────────────────────┬────────────────────────────────────┘
                            │
                            ↓  exp transform
    ┌────────────────────────────────────────────────────────────┐
    │  paths_coarse_est ∈ ℝ^{M × N_c × d}                        │
    │  (5000 × 50 × 3) = STILL 3-dimensional!                    │
    └───────────────────────┬────────────────────────────────────┘
                            │
                            ↓  basket projection: w · X
                            │  (NOW dimension reduces: 3 → 1)
    ┌────────────────────────────────────────────────────────────┐
    │  basket_coarse_est ∈ ℝ^{M × N_c}                           │
    │  (5000 × 50) = NOW 1-dimensional!                          │
    └────────────────────────────────────────────────────────────┘
    """)
    
    # =========================================================================
    # Step 7: Visualisation
    # =========================================================================
    print(f"\n{'─'*70}")
    print("Generating visualisation...")
    print(f"{'─'*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sample d-dimensional paths (one path, all assets)
    ax1 = axes[0, 0]
    path_idx = 0
    time_fine = np.linspace(0, T, N_fine)
    time_coarse = np.linspace(0, T, N_fine//2)
    for i in range(d):
        ax1.plot(time_fine, paths_fine[path_idx, :, i], 
                 label=f'Asset {i+1} (fine)', alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.set_title(f'd-Dimensional Paths (d={d})\nOT operates HERE')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Transport matrix eigenvalues over time
    ax2 = axes[0, 1]
    eigenvalues_over_time = []
    for n in range(1, len(maps)):
        if isinstance(maps[n], GaussianBrenierMap):
            eigs = np.linalg.eigvalsh(maps[n].A)
            eigenvalues_over_time.append(eigs)
    eigenvalues_over_time = np.array(eigenvalues_over_time)
    
    for i in range(d):
        ax2.plot(time_coarse[1:], eigenvalues_over_time[:, i], 
                 label=f'λ_{i+1}', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_title(f'Transport Matrix A Eigenvalues\n(A is {d}×{d} matrix)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: 1D projected basket paths
    ax3 = axes[1, 0]
    for i in range(min(50, M_paths)):
        ax3.plot(time_coarse, basket_coarse[i, :], 'b-', alpha=0.1)
    ax3.plot(time_coarse, np.mean(basket_coarse, axis=0), 'b-', 
             linewidth=2, label='Mean basket')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Basket Value')
    ax3.set_title('1D Projected Basket Paths\nAfter Markovian projection (d→1)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Variance reduction comparison
    ax4 = axes[1, 1]
    var_standard_t = np.var(diff_standard, axis=0)
    var_OT_t = np.var(diff_OT, axis=0)
    
    ax4.plot(time_coarse, var_standard_t, 'r-', linewidth=2, 
             label='Standard coupling')
    ax4.plot(time_coarse, var_OT_t, 'g-', linewidth=2, 
             label='OT coupling')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Var[fine - coarse]')
    ax4.set_title('Variance Reduction from OT Coupling\n(Lower is better)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('OT_dimension_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualisation to 'OT_dimension_demo.png'")
    plt.show()
    
    # =========================================================================
    # Final message
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("KEY TAKEAWAY")
    print(f"{'='*70}")
    print(f"""
    The Gaussian-Brenier optimal transport map operates in the FULL 
    {d}-dimensional asset space:
    
        T: ℝ^{d} → ℝ^{d}
        
    using a {d}×{d} transport matrix A.
    
    The dimension reduction from {d}D to 1D happens AFTERWARDS via 
    Markovian projection:
    
        S = weights · X  :  ℝ^{d} → ℝ¹
    
    This ordering is important because:
    1. OT exploits the full {d}×{d} correlation structure
    2. The Gaussian assumption holds exactly in log-space for GBM
    3. We get provably optimal coupling BEFORE any information is lost
    """)
    print(f"{'='*70}\n")
    breakpoint()


if __name__ == "__main__":
    demonstrate_dimensions()
