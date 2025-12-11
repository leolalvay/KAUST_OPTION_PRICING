"""
Configuration Parameters for the Comparison Framework

This module defines the problem parameters for comparing MLMC and Laplace
approximation methods for American basket option pricing.

The default parameters are from Bayer, Häppölä, Tempone (2017), Equation 56:
    "Implied Stopping Rules for American Basket Options from Markovian Projection"

Key Parameters (3D Black-Scholes Test Case)
-------------------------------------------
- r = 0.05 (risk-free rate)
- σ = (0.2, 0.15, 0.1) (asset volatilities)
- Correlation matrix from Eq. 56
- P1 = [1, 1, 1] (equal weights)
- x0 = [100, 100, 100] (initial prices, basket = 300)
- T = 0.5 (maturity)
- K = 300 (at-the-money strike)

Author: Wadoud (KAUST Internship)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.linalg import cholesky


@dataclass
class ProblemParameters:
    """
    Parameters for American basket option pricing comparison.
    
    Default values reproduce the 3D Black-Scholes test case from
    Bayer et al. (2017), Section 3.5.2, Equation 56.
    
    Attributes
    ----------
    r : float
        Risk-free interest rate
    sigma : np.ndarray
        Asset volatilities, shape (d,)
    corr_matrix : np.ndarray
        Correlation matrix, shape (d, d)
    P1 : np.ndarray
        Basket weights, shape (d,)
    x0 : np.ndarray
        Initial asset prices, shape (d,)
    T : float
        Option maturity
    K : float
        Strike price
    option_type : str
        Either "put" or "call"
    N_t : int
        Number of time grid points for PDE solver
    N_s : int
        Number of spatial grid points for PDE solver
    max_degree : int
        Maximum polynomial degree for MLMC
    h0 : float
        Coarsest timestep for MLMC
    random_seed : int
        Random seed for reproducibility
    """
    
    # Model parameters
    r: float = 0.05
    sigma: Optional[np.ndarray] = None
    corr_matrix: Optional[np.ndarray] = None
    
    # Portfolio parameters
    P1: Optional[np.ndarray] = None
    x0: Optional[np.ndarray] = None
    
    # Option parameters
    T: float = 0.5
    K: float = 300.0
    option_type: str = "put"
    
    # Grid parameters for PDE solver
    N_t: int = 50
    N_s: int = 100
    
    # MLMC parameters
    max_degree: int = 3
    h0: float = 0.1
    
    # Reproducibility
    random_seed: int = 42
    
    def __post_init__(self):
        """Set default values for arrays after initialisation."""
        if self.sigma is None:
            self.sigma = np.array([0.2, 0.15, 0.1])
        
        if self.corr_matrix is None:
            # Correlation matrix from Equation 56
            self.corr_matrix = np.array([
                [1.0, 0.8, 0.3],
                [0.8, 1.0, 0.1],
                [0.3, 0.1, 1.0]
            ])
        
        if self.P1 is None:
            self.P1 = np.array([1.0, 1.0, 1.0])
        
        if self.x0 is None:
            self.x0 = np.array([100.0, 100.0, 100.0])
        
        # Ensure arrays
        self.sigma = np.asarray(self.sigma)
        self.corr_matrix = np.asarray(self.corr_matrix)
        self.P1 = np.asarray(self.P1)
        self.x0 = np.asarray(self.x0)
    
    @property
    def d(self) -> int:
        """Number of assets in the basket."""
        return len(self.x0)
    
    @property
    def S0(self) -> float:
        """Initial basket value."""
        return float(np.dot(self.P1, self.x0))
    
    @property
    def corr_chol(self) -> np.ndarray:
        """
        Cholesky factor of correlation matrix.
        
        Lower triangular L such that L @ L.T = corr_matrix.
        Used for generating correlated Brownian increments.
        """
        return cholesky(self.corr_matrix, lower=True)
    
    @property
    def cov_mat(self) -> np.ndarray:
        """
        Alias for correlation matrix (for compatibility with PDE code).
        
        Note: In the Black-Scholes model, the correlation matrix of the
        log-returns equals the covariance structure after normalisation
        by individual volatilities.
        """
        return self.corr_matrix
    
    def copy(self) -> 'ProblemParameters':
        """Create a copy of the parameters."""
        return ProblemParameters(
            r=self.r,
            sigma=self.sigma.copy(),
            corr_matrix=self.corr_matrix.copy(),
            P1=self.P1.copy(),
            x0=self.x0.copy(),
            T=self.T,
            K=self.K,
            option_type=self.option_type,
            N_t=self.N_t,
            N_s=self.N_s,
            max_degree=self.max_degree,
            h0=self.h0,
            random_seed=self.random_seed
        )
    
    def __str__(self) -> str:
        """Pretty print parameter summary."""
        lines = [
            "=" * 60,
            "Problem Parameters (Bayer et al. 2017, Eq. 56)",
            "=" * 60,
            f"  Number of assets: d = {self.d}",
            f"  Initial prices: x0 = {self.x0}",
            f"  Initial basket: S0 = {self.S0:.1f}",
            f"  Basket weights: P1 = {self.P1}",
            "",
            f"  Risk-free rate: r = {self.r}",
            f"  Volatilities: σ = {self.sigma}",
            f"  Correlation matrix:",
        ]
        for row in self.corr_matrix:
            lines.append(f"    {row}")
        lines.extend([
            "",
            f"  Option type: {self.option_type}",
            f"  Strike: K = {self.K:.1f}",
            f"  Maturity: T = {self.T}",
            "",
            f"  PDE grid: {self.N_t} × {self.N_s}",
            f"  MLMC max degree: {self.max_degree}",
            f"  MLMC coarsest timestep: h0 = {self.h0}",
            f"  Random seed: {self.random_seed}",
            "=" * 60
        ])
        return "\n".join(lines)


# Default parameters instance (paper's Eq. 56)
DEFAULT_PARAMS = ProblemParameters()


def create_2d_params(r=0.05, sigma=None, rho=0.5, T=0.5, K=200.0, 
                     seed=42) -> ProblemParameters:
    """
    Create parameters for a 2D Black-Scholes test case.
    
    Parameters
    ----------
    r : float
        Risk-free rate
    sigma : array-like or None
        Volatilities for the two assets. Default: [0.2, 0.15]
    rho : float
        Correlation between assets
    T : float
        Maturity
    K : float
        Strike price
    seed : int
        Random seed
        
    Returns
    -------
    ProblemParameters
        2D problem configuration
    """
    if sigma is None:
        sigma = np.array([0.2, 0.15])
    else:
        sigma = np.asarray(sigma)
    
    corr_matrix = np.array([
        [1.0, rho],
        [rho, 1.0]
    ])
    
    return ProblemParameters(
        r=r,
        sigma=sigma,
        corr_matrix=corr_matrix,
        P1=np.array([1.0, 1.0]),
        x0=np.array([100.0, 100.0]),
        T=T,
        K=K,
        random_seed=seed
    )


def create_5d_params(r=0.05, base_sigma=0.15, T=0.5, seed=42) -> ProblemParameters:
    """
    Create parameters for a 5D Black-Scholes test case.
    
    Parameters
    ----------
    r : float
        Risk-free rate
    base_sigma : float
        Base volatility (individual sigmas will vary around this)
    T : float
        Maturity
    seed : int
        Random seed
        
    Returns
    -------
    ProblemParameters
        5D problem configuration
    """
    d = 5
    
    # Varying volatilities
    sigma = base_sigma * np.array([1.2, 1.0, 0.9, 1.1, 0.8])
    
    # Positive definite correlation matrix (Toeplitz-like)
    corr_matrix = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            corr_matrix[i, j] = 0.5 ** (j - i)
            corr_matrix[j, i] = corr_matrix[i, j]
    
    return ProblemParameters(
        r=r,
        sigma=sigma,
        corr_matrix=corr_matrix,
        P1=np.ones(d),
        x0=100.0 * np.ones(d),
        T=T,
        K=500.0,  # d=5 means basket ≈ 500
        random_seed=seed
    )


def create_10d_params(r=0.05, base_sigma=0.125, T=0.5, seed=42) -> ProblemParameters:
    """
    Create parameters for a 10D Black-Scholes test case.
    
    Based on the paper's Section 3.5.3 (10-to-1 dimensional example).
    
    Returns
    -------
    ProblemParameters
        10D problem configuration
    """
    d = 10
    
    # Uniform volatilities
    sigma = base_sigma * np.ones(d)
    
    # Correlation matrix from paper's Equation 58
    corr_matrix = np.array([
        [1.0,   0.2,   0.2,   0.35,  0.2,   0.25,  0.2,   0.2,   0.3,   0.2  ],
        [0.2,   1.0,   0.2,   0.2,   0.2,   0.125, 0.45,  0.2,   0.2,   0.45 ],
        [0.2,   0.2,   1.0,   0.2,   0.2,   0.2,   0.2,   0.2,   0.45,  0.2  ],
        [0.35,  0.2,   0.2,   1.0,   0.2,   0.2,   0.2,   0.2,   0.425, 0.2  ],
        [0.25,  0.125, 0.2,   0.2,   1.0,   0.2,   0.2,   0.5,   0.35,  0.2  ],
        [0.2,   0.45,  0.2,   0.2,   0.2,   1.0,   0.2,   0.2,   0.2,   0.2  ],
        [0.2,   0.45,  0.2,   0.2,   0.2,   0.2,   1.0,   0.2,   0.2,   0.2  ],
        [0.2,   0.2,   0.2,   0.2,   0.2,   0.2,   0.2,   1.0,   0.2,  -0.1  ],
        [0.3,   0.2,   0.45,  0.425, 0.5,   0.35,  0.2,   0.2,   1.0,   0.2  ],
        [0.2,   0.45,  0.2,   0.2,   0.2,   0.2,   0.2,  -0.1,   0.2,   1.0  ]
    ])
    
    return ProblemParameters(
        r=r,
        sigma=sigma,
        corr_matrix=corr_matrix,
        P1=np.ones(d),
        x0=100.0 * np.ones(d),
        T=T,
        K=1000.0,  # d=10 means basket ≈ 1000
        random_seed=seed
    )


if __name__ == "__main__":
    # Test the configuration
    print(DEFAULT_PARAMS)
    
    print("\n2D Parameters:")
    params_2d = create_2d_params()
    print(f"  d={params_2d.d}, S0={params_2d.S0}, K={params_2d.K}")
    
    print("\n5D Parameters:")
    params_5d = create_5d_params()
    print(f"  d={params_5d.d}, S0={params_5d.S0}, K={params_5d.K}")
    
    print("\n10D Parameters:")
    params_10d = create_10d_params()
    print(f"  d={params_10d.d}, S0={params_10d.S0}, K={params_10d.K}")
