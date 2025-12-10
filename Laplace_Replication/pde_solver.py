"""
American Option PDE Solver for Laplace Method Replication

Implements backward Euler scheme for 1D American option pricing
as described in Section 3.2 of Bayer et al. (2017).
"""

import numpy as np
from scipy.linalg import solve_banded


def solve_american_option_pde(vol_func, r, K, T, s_min, s_max, 
                               N_t=200, N_s=100, option_type='put'):
    """
    Solve 1D American option PDE using backward Euler.
    
    The PDE is (Equation 18):
    -∂_t ū_A = max(L̄ū_A, 0) where L̄ū_A ≥ 0 or ū_A = g(s)
    
    With:
    L̄u = (b̄²/2) ∂²u/∂s² + rs ∂u/∂s - ru
    
    Parameters
    ----------
    vol_func : callable
        Function vol_func(t, s) returning projected volatility b̄(t,s)
    r : float
        Risk-free rate
    K : float
        Strike price
    T : float
        Maturity
    s_min, s_max : float
        Spatial domain bounds
    N_t : int
        Number of time steps
    N_s : int
        Number of spatial points
    option_type : str
        'put' or 'call'
        
    Returns
    -------
    dict with:
        's_grid': spatial grid
        't_grid': time grid
        'values': option values V(t, s), shape (N_t+1, N_s)
        'exercise_boundary': early exercise boundary b(t)
    """
    # Grids
    dt = T / N_t
    ds = (s_max - s_min) / (N_s - 1)
    
    t_grid = np.linspace(0, T, N_t + 1)
    s_grid = np.linspace(s_min, s_max, N_s)
    
    # Payoff function
    if option_type == 'put':
        payoff = np.maximum(K - s_grid, 0)
    else:
        payoff = np.maximum(s_grid - K, 0)
    
    # Solution array
    V = np.zeros((N_t + 1, N_s))
    V[-1, :] = payoff  # Terminal condition
    
    # Backward timestepping
    for n in range(N_t - 1, -1, -1):
        t = t_grid[n]
        
        # Build tridiagonal system for implicit Euler
        # (I - dt * L) V^n = V^{n+1}
        
        # Get volatility at this time
        vol = np.array([vol_func(t, s) for s in s_grid])
        vol_sq = vol**2
        
        # Coefficients (Equation in Section 3.2)
        # L = (b²/2) ∂²/∂s² + rs ∂/∂s - r
        # Using central differences:
        # ∂²u/∂s² ≈ (u_{i-1} - 2u_i + u_{i+1}) / ds²
        # ∂u/∂s ≈ (u_{i+1} - u_{i-1}) / (2ds)
        
        # Coefficient of u_{i-1}
        alpha = vol_sq / (2 * ds**2) - r * s_grid / (2 * ds)
        
        # Coefficient of u_i
        beta = -vol_sq / ds**2 - r
        
        # Coefficient of u_{i+1}
        gamma = vol_sq / (2 * ds**2) + r * s_grid / (2 * ds)
        
        # Build matrix (I - dt * L) in banded form
        # Main diagonal: 1 - dt * beta
        # Lower diagonal: -dt * alpha
        # Upper diagonal: -dt * gamma
        
        main_diag = 1 - dt * beta
        lower_diag = -dt * alpha[1:]
        upper_diag = -dt * gamma[:-1]
        
        # Solve tridiagonal system
        # For boundary conditions, use Dirichlet: V = payoff at boundaries
        
        # RHS is V^{n+1}
        rhs = V[n + 1, :].copy()
        
        # Apply boundary conditions
        # At s_min: V = payoff (for put, this is K - s_min)
        # At s_max: V = payoff (for put, this is 0)
        rhs[0] = payoff[0]
        rhs[-1] = payoff[-1]
        
        # Set boundary rows to identity
        main_diag[0] = 1
        main_diag[-1] = 1
        lower_diag[0] = 0  # This affects row 1
        upper_diag[-1] = 0  # This affects row N_s-2
        
        # Modify RHS for boundary contribution from interior
        # Actually, let's use simpler boundary handling
        # Interior solve, then enforce boundary
        
        # Solve interior using Thomas algorithm
        V_new = thomas_algorithm(lower_diag, main_diag, upper_diag, rhs)
        
        # Apply early exercise constraint (American option)
        V[n, :] = np.maximum(V_new, payoff)
    
    # Compute exercise boundary
    exercise_boundary = np.zeros(N_t + 1)
    for n in range(N_t + 1):
        # Find where V(t, s) = payoff (exercise region)
        if option_type == 'put':
            # For put, exercise when s is small
            diff = V[n, :] - payoff
            exercise_idx = np.where(diff < 1e-6)[0]
            if len(exercise_idx) > 0:
                exercise_boundary[n] = s_grid[exercise_idx[-1]]
            else:
                exercise_boundary[n] = s_min
        else:
            diff = V[n, :] - payoff
            exercise_idx = np.where(diff < 1e-6)[0]
            if len(exercise_idx) > 0:
                exercise_boundary[n] = s_grid[exercise_idx[0]]
            else:
                exercise_boundary[n] = s_max
    
    return {
        's_grid': s_grid,
        't_grid': t_grid,
        'values': V,
        'exercise_boundary': exercise_boundary
    }


def thomas_algorithm(a, b, c, d):
    """
    Solve tridiagonal system using Thomas algorithm.
    
    a: lower diagonal (length n-1)
    b: main diagonal (length n)
    c: upper diagonal (length n-1)
    d: RHS (length n)
    """
    n = len(b)
    
    # Copy arrays to avoid modification
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    x = np.zeros(n)
    
    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n - 1):
        denom = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denom
    
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
    # Back substitution
    x[n-1] = d_prime[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x


def monte_carlo_lower_bound(x0, P1, r, sigma, corr_chol, T, K, 
                            exercise_boundary_func, N_paths=100000, N_steps=200):
    """
    Compute lower bound via Monte Carlo with implied stopping rule.
    
    This implements Equation 21:
    u_A(0, x0) ≥ E[exp(-rτ†) g(P1 X(τ†))]
    
    where τ† is the hitting time of the exercise boundary.
    """
    d = len(x0)
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)
    
    # Cholesky factor for correlated increments
    L = corr_chol
    
    # Simulate paths
    X = np.zeros((N_paths, d))
    X[:, :] = x0
    
    # Track stopping times and payoffs
    stopped = np.zeros(N_paths, dtype=bool)
    tau = np.ones(N_paths) * T
    payoff = np.zeros(N_paths)
    
    for n in range(N_steps):
        t = n * dt
        
        # Check stopping condition for non-stopped paths
        basket = X @ P1
        boundary = exercise_boundary_func(t)
        
        # For put: exercise when basket < boundary
        should_stop = (basket <= boundary) & (~stopped)
        
        if np.any(should_stop):
            tau[should_stop] = t
            payoff[should_stop] = np.maximum(K - basket[should_stop], 0)
            stopped[should_stop] = True
        
        # Evolve non-stopped paths
        active = ~stopped
        if not np.any(active):
            break
        
        # Generate correlated Brownian increments
        Z = np.random.randn(N_paths, d)
        dW = sqrt_dt * (Z @ L.T)
        
        # GBM evolution: dX = r X dt + σ X dW
        for i in range(d):
            X[active, i] *= np.exp((r - 0.5*sigma[i]**2)*dt + sigma[i]*dW[active, i])
    
    # Handle paths that didn't stop
    final_basket = X @ P1
    payoff[~stopped] = np.maximum(K - final_basket[~stopped], 0)
    
    # Discount payoffs
    discounted = np.exp(-r * tau) * payoff
    
    # Lower bound estimate
    lower_bound = np.mean(discounted)
    std_err = np.std(discounted) / np.sqrt(N_paths)
    
    return lower_bound, std_err


def monte_carlo_upper_bound(x0, P1, r, sigma, corr_chol, T, K,
                            value_func, value_grad_func,
                            N_paths=100000, N_steps=200):
    """
    Compute upper bound via dual martingale (Rogers).
    
    This implements Equation 25:
    u_A(0, x0) ≤ E[sup_{0≤t≤T} (Z̃(t) - R*(t))]
    
    where R* is the martingale from Equation 26.
    """
    d = len(x0)
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)
    
    L = corr_chol
    
    # Simulate paths
    X = np.zeros((N_paths, d))
    X[:, :] = x0
    
    # Martingale R
    R = np.zeros(N_paths)
    
    # Track supremum of Z̃ - R
    supremum = np.zeros(N_paths)
    
    for n in range(N_steps):
        t = n * dt
        
        # Compute Z̃(t) = exp(-rt) g(P1 X(t))
        basket = X @ P1
        Z_tilde = np.exp(-r * t) * np.maximum(K - basket, 0)
        
        # Update supremum
        supremum = np.maximum(supremum, Z_tilde - R)
        
        # Generate increments
        Z = np.random.randn(N_paths, d)
        dW = sqrt_dt * (Z @ L.T)
        
        # Update martingale R
        # dR = exp(-rt) (∇ū_A)^T P1 b dW
        # For simplicity, approximate using finite difference
        for i in range(N_paths):
            grad = value_grad_func(t, basket[i])
            vol_contribution = 0
            for j in range(d):
                vol_contribution += P1[j] * sigma[j] * X[i, j] * dW[i, j]
            R[i] += np.exp(-r * t) * grad * vol_contribution
        
        # Evolve X
        for j in range(d):
            X[:, j] *= np.exp((r - 0.5*sigma[j]**2)*dt + sigma[j]*dW[:, j])
    
    # Final time
    basket = X @ P1
    Z_tilde = np.exp(-r * T) * np.maximum(K - basket, 0)
    supremum = np.maximum(supremum, Z_tilde - R)
    
    upper_bound = np.mean(supremum)
    std_err = np.std(supremum) / np.sqrt(N_paths)
    
    return upper_bound, std_err


if __name__ == "__main__":
    print("PDE Solver Module - Test")
    
    # Simple test with constant volatility
    def const_vol(t, s):
        return 30.0  # Constant volatility
    
    result = solve_american_option_pde(
        const_vol, r=0.05, K=300, T=0.5,
        s_min=200, s_max=400, N_t=100, N_s=100
    )
    
    print(f"Option value at t=0, s=300: {result['values'][0, 50]:.4f}")
    print(f"Exercise boundary at t=0: {result['exercise_boundary'][0]:.2f}")
