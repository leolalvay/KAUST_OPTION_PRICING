"""
Laplace Approximation for Projected Volatility

Replicating the methodology from Bayer, Häppölä, and Tempone (2017)
"Implied Stopping Rules for American Basket Options from Markovian Projection"

This implements Section 3.1.1 of the paper.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular
import warnings


def black_scholes_log_density(x, x0, t, r, sigma, corr_chol):
    """
    Log density of multivariate Black-Scholes (log-normal) at time t.
    
    Under Black-Scholes, log(X_i(t)/X_i(0)) ~ N((r - σ_i²/2)t, σ_i² t)
    with correlation structure from corr_chol.
    
    Parameters
    ----------
    x : array (d,)
        Asset prices at time t
    x0 : array (d,)
        Initial asset prices
    t : float
        Time
    r : float
        Risk-free rate
    sigma : array (d,)
        Volatilities for each asset
    corr_chol : array (d, d)
        Cholesky factor of correlation matrix
        
    Returns
    -------
    float
        Log density value
    """
    d = len(x)
    
    # Check for non-positive prices
    if np.any(x <= 0):
        return -np.inf
    
    # Log returns
    log_returns = np.log(x / x0)
    
    # Mean of log returns under risk-neutral measure
    mu = (r - 0.5 * sigma**2) * t
    
    # Covariance matrix of log returns
    # Cov[log X_i, log X_j] = σ_i σ_j ρ_ij t
    sigma_outer = np.outer(sigma, sigma)
    corr_matrix = corr_chol @ corr_chol.T
    cov_matrix = sigma_outer * corr_matrix * t
    
    # Standardised residuals
    centered = log_returns - mu
    
    # Log density of multivariate normal
    # -0.5 * (d*log(2π) + log|Σ| + z'Σ^{-1}z)
    try:
        L = cholesky(cov_matrix, lower=True)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # Solve L y = centered
        y = solve_triangular(L, centered, lower=True)
        quad_form = np.dot(y, y)
        
        # Jacobian: product of 1/x_i for transformation from normal to log-normal
        log_jacobian = -np.sum(np.log(x))
        
        log_density = -0.5 * (d * np.log(2 * np.pi) + log_det + quad_form) + log_jacobian
        
    except np.linalg.LinAlgError:
        return -np.inf
    
    return log_density


def projected_volatility_integrand_numerator(z, s, P1, x0, t, r, sigma, corr_chol):
    """
    Integrand for numerator of projected volatility (Equation 39 numerator).
    
    f(z) = φ(x(z); x0) * (P1 b b^T P1^T)(t, x(z))
    
    For Black-Scholes: b_i(t,x) = σ_i x_i, so
    (P1 b b^T P1^T) = Σ_ij P1_i P1_j σ_i σ_j ρ_ij x_i x_j
    """
    d = len(x0)
    
    # Reconstruct x from z (z are coordinates 2,...,d)
    # x_1 is determined by constraint P1 · x = s
    # x_1 = (s - Σ_{j>1} P1_j z_j) / P1_0
    
    x = np.zeros(d)
    x[1:] = z
    x[0] = (s - np.dot(P1[1:], z)) / P1[0]
    
    # Check positivity
    if x[0] <= 0 or np.any(z <= 0):
        return -np.inf, x
    
    # Log density
    log_phi = black_scholes_log_density(x, x0, t, r, sigma, corr_chol)
    
    if np.isinf(log_phi):
        return -np.inf, x
    
    # Compute (P1 b b^T P1^T) for Black-Scholes
    # b_ij = σ_i x_i G_ij where G is Cholesky of correlation
    # b b^T = diag(σ x) G G^T diag(σ x) = diag(σ x) Corr diag(σ x)
    # P1 b b^T P1^T = Σ_ij P1_i P1_j σ_i σ_j ρ_ij x_i x_j
    
    corr_matrix = corr_chol @ corr_chol.T
    
    # Efficient computation: (P1 .* σ .* x)^T Corr (P1 .* σ .* x)
    weighted = P1 * sigma * x
    vol_squared = weighted @ corr_matrix @ weighted
    
    log_f = log_phi + np.log(vol_squared)
    
    return log_f, x


def projected_volatility_integrand_denominator(z, s, P1, x0, t, r, sigma, corr_chol):
    """
    Integrand for denominator of projected volatility (Equation 39 denominator).
    
    f_tilde(z) = φ(x(z); x0)
    """
    d = len(x0)
    
    x = np.zeros(d)
    x[1:] = z
    x[0] = (s - np.dot(P1[1:], z)) / P1[0]
    
    if x[0] <= 0 or np.any(z <= 0):
        return -np.inf, x
    
    log_phi = black_scholes_log_density(x, x0, t, r, sigma, corr_chol)
    
    return log_phi, x


def find_laplace_mode(objective, z0, s, P1, x0, t, r, sigma, corr_chol):
    """
    Find the mode (maximum) of the integrand using Newton iteration.
    
    We maximise the log of the integrand.
    """
    def neg_objective(z):
        val, _ = objective(z, s, P1, x0, t, r, sigma, corr_chol)
        return -val if np.isfinite(val) else 1e10
    
    # Use scipy minimize (L-BFGS-B for bounded optimisation)
    d = len(x0)
    
    # Bounds: z_i > 0 (asset prices must be positive)
    bounds = [(1e-6, None) for _ in range(d - 1)]
    
    result = minimize(neg_objective, z0, method='L-BFGS-B', bounds=bounds)
    
    return result.x


def compute_hessian(objective, z_star, s, P1, x0, t, r, sigma, corr_chol, eps=1e-5):
    """
    Compute Hessian of log-integrand at mode using finite differences.
    """
    d = len(z_star)
    H = np.zeros((d, d))
    
    f0, _ = objective(z_star, s, P1, x0, t, r, sigma, corr_chol)
    
    for i in range(d):
        for j in range(i, d):
            z_pp = z_star.copy()
            z_pm = z_star.copy()
            z_mp = z_star.copy()
            z_mm = z_star.copy()
            
            z_pp[i] += eps
            z_pp[j] += eps
            z_pm[i] += eps
            z_pm[j] -= eps
            z_mp[i] -= eps
            z_mp[j] += eps
            z_mm[i] -= eps
            z_mm[j] -= eps
            
            f_pp, _ = objective(z_pp, s, P1, x0, t, r, sigma, corr_chol)
            f_pm, _ = objective(z_pm, s, P1, x0, t, r, sigma, corr_chol)
            f_mp, _ = objective(z_mp, s, P1, x0, t, r, sigma, corr_chol)
            f_mm, _ = objective(z_mm, s, P1, x0, t, r, sigma, corr_chol)
            
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
            H[j, i] = H[i, j]
    
    return H


def laplace_approximation_volatility(s, t, P1, x0, r, sigma, corr_chol):
    """
    Compute projected volatility using Laplace approximation (Equation 41).
    
    b̃²(t, s) = exp(f(z*) - f̃(z⋆)) * sqrt(det|H f̃(z⋆)| / det|H f(z*)|)
    
    Parameters
    ----------
    s : float
        Basket value (projection target)
    t : float
        Time
    P1 : array (d,)
        Portfolio weights
    x0 : array (d,)
        Initial asset prices
    r : float
        Risk-free rate
    sigma : array (d,)
        Asset volatilities
    corr_chol : array (d, d)
        Cholesky factor of correlation matrix
        
    Returns
    -------
    float
        Projected volatility squared, b̄²(t, s)
    """
    d = len(x0)
    
    if t < 1e-10:
        # At t=0, use limiting behaviour
        # The assets are at x0, so projected vol² = (P1 σ x0)^T Corr (P1 σ x0)
        corr_matrix = corr_chol @ corr_chol.T
        weighted = P1 * sigma * x0
        return weighted @ corr_matrix @ weighted
    
    # Initial guess: assets at their forward values scaled to match s
    forward = x0 * np.exp(r * t)
    scale = s / np.dot(P1, forward)
    z0 = forward[1:] * scale
    z0 = np.maximum(z0, 1e-6)  # Ensure positive
    
    # Find modes for numerator and denominator
    z_star = find_laplace_mode(
        projected_volatility_integrand_numerator, 
        z0, s, P1, x0, t, r, sigma, corr_chol
    )
    
    z_tilde = find_laplace_mode(
        projected_volatility_integrand_denominator,
        z0, s, P1, x0, t, r, sigma, corr_chol
    )
    
    # Evaluate log-integrands at modes
    f_star, _ = projected_volatility_integrand_numerator(
        z_star, s, P1, x0, t, r, sigma, corr_chol
    )
    f_tilde, _ = projected_volatility_integrand_denominator(
        z_tilde, s, P1, x0, t, r, sigma, corr_chol
    )
    
    if not (np.isfinite(f_star) and np.isfinite(f_tilde)):
        warnings.warn(f"Laplace approximation failed at t={t}, s={s}")
        return np.nan
    
    # Compute Hessians
    H_star = compute_hessian(
        projected_volatility_integrand_numerator,
        z_star, s, P1, x0, t, r, sigma, corr_chol
    )
    
    H_tilde = compute_hessian(
        projected_volatility_integrand_denominator,
        z_tilde, s, P1, x0, t, r, sigma, corr_chol
    )
    
    # Laplace formula: ratio of Gaussian integrals
    # ∫ exp(f) ≈ exp(f*) * sqrt((2π)^d / |det(-H)|)
    # Ratio: exp(f* - f̃*) * sqrt(det(-H̃) / det(-H))
    
    try:
        det_H_star = np.linalg.det(-H_star)
        det_H_tilde = np.linalg.det(-H_tilde)
        
        if det_H_star <= 0 or det_H_tilde <= 0:
            warnings.warn(f"Non-positive definite Hessian at t={t}, s={s}")
            return np.nan
        
        log_b_squared = (f_star - f_tilde) + 0.5 * (np.log(det_H_tilde) - np.log(det_H_star))
        b_squared = np.exp(log_b_squared)
        
    except np.linalg.LinAlgError:
        warnings.warn(f"Singular Hessian at t={t}, s={s}")
        return np.nan
    
    return b_squared


def compute_volatility_surface(t_grid, s_grid, P1, x0, r, sigma, corr_chol):
    """
    Compute projected volatility surface over a grid.
    
    Parameters
    ----------
    t_grid : array (N_t,)
        Time points
    s_grid : array (N_s,)
        Basket values
    P1, x0, r, sigma, corr_chol : model parameters
    
    Returns
    -------
    vol_surface : array (N_t, N_s)
        Projected volatility b̄(t, s) (not squared)
    """
    N_t = len(t_grid)
    N_s = len(s_grid)
    
    vol_surface = np.zeros((N_t, N_s))
    
    for i, t in enumerate(t_grid):
        for j, s in enumerate(s_grid):
            b_sq = laplace_approximation_volatility(s, t, P1, x0, r, sigma, corr_chol)
            vol_surface[i, j] = np.sqrt(b_sq) if np.isfinite(b_sq) and b_sq > 0 else np.nan
    
    return vol_surface


def interpolate_volatility_polynomial(t_grid, s_grid, vol_surface, degree=3):
    """
    Fit polynomial to volatility surface for each time slice.
    
    This follows the paper's approach of using polynomial interpolation
    to extend volatility evaluations to the full domain.
    
    Returns coefficients for each time slice.
    """
    N_t = len(t_grid)
    coefficients = []
    
    for i in range(N_t):
        vol_slice = vol_surface[i, :]
        valid = np.isfinite(vol_slice)
        
        if np.sum(valid) > degree:
            # Fit polynomial to valid points
            coef = np.polyfit(s_grid[valid], vol_slice[valid], degree)
            coefficients.append(coef)
        else:
            coefficients.append(None)
    
    return coefficients


if __name__ == "__main__":
    # Test with paper's 3D example (Equation 56)
    print("="*70)
    print("Testing Laplace Approximation - Bayer et al. (2017) 3D Example")
    print("="*70)
    
    # Parameters from Equation 56
    r = 0.05
    sigma = np.array([0.2, 0.15, 0.1])
    
    # Correlation matrix
    corr_matrix = np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.1],
        [0.3, 0.1, 1.0]
    ])
    corr_chol = cholesky(corr_matrix, lower=True)
    
    # Portfolio weights
    P1 = np.array([1.0, 1.0, 1.0])
    
    # Initial prices (implied from paper: basket starts at 300)
    x0 = np.array([100.0, 100.0, 100.0])
    
    # Test at a single point
    t_test = 0.25
    s_test = 300.0  # At-the-money
    
    b_sq = laplace_approximation_volatility(s_test, t_test, P1, x0, r, sigma, corr_chol)
    print(f"\nAt t={t_test}, s={s_test}:")
    print(f"  Projected volatility² = {b_sq:.4f}")
    print(f"  Projected volatility  = {np.sqrt(b_sq):.4f}")
    
    # Compute surface
    print("\nComputing volatility surface...")
    t_grid = np.linspace(0.01, 0.5, 10)
    s_grid = np.linspace(250, 350, 20)
    
    vol_surface = compute_volatility_surface(t_grid, s_grid, P1, x0, r, sigma, corr_chol)
    
    print(f"  Surface shape: {vol_surface.shape}")
    print(f"  Volatility range: [{np.nanmin(vol_surface):.2f}, {np.nanmax(vol_surface):.2f}]")
