import numpy as np
import math

# Market parameters
dt = 0.0625  # Time step
T = 1.0      # Time to maturity (years)
vol = 0.2    # Volatility (annualised)
r = 0.05     # Risk-free rate
K = 100      # Strike price
S0 = 100     # Initial spot price


def BS_call(S0, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    
    Parameters
    ----------
    S0 : float
        Initial spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (continuously compounded).
    sigma : float
        Volatility of the underlying asset (annualised).
    
    Returns
    -------
    float
        The theoretical call option price.
    
    Notes
    -----
    Uses the cumulative distribution function of the standard normal distribution
    approximated via the error function. The Black-Scholes formula is:
        C = S0 * N(d1) - K * exp(-rT) * N(d2)
    where d1 = [ln(S0/K) + (r + sigma²/2)T] / (sigma * sqrt(T))
          d2 = d1 - sigma * sqrt(T)
    """
    def N(x):
        """Standard normal cumulative distribution function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    C = S0 * N(d1) - K * math.exp(-r * T) * N(d2)
    return C


def BS_put(S0, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European put option.
    
    Parameters
    ----------
    S0 : float
        Initial spot price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (continuously compounded).
    sigma : float
        Volatility of the underlying asset (annualised).
    
    Returns
    -------
    float
        The theoretical put option price.
    
    Notes
    -----
    Uses the cumulative distribution function of the standard normal distribution
    approximated via the error function. The Black-Scholes formula is:
        P = K * exp(-rT) * N(-d2) - S0 * N(-d1)
    where d1 = [ln(S0/K) + (r + sigma²/2)T] / (sigma * sqrt(T))
          d2 = d1 - sigma * sqrt(T)
    
    Alternatively derived from put-call parity:
        P = C - S0 + K * exp(-rT)
    """
    def N(x):
        """Standard normal cumulative distribution function."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    P = K * math.exp(-r * T) * N(-d2) - S0 * N(-d1)
    return P


# Example usage
if __name__ == "__main__":
    BS_call_price = BS_call(S0, K, T, r, vol)
    BS_put_price = BS_put(S0, K, T, r, vol)
    
    print(f"BS analytic call price: {BS_call_price:.6f}")
    print(f"BS analytic put price:  {BS_put_price:.6f}")
    
    # Verify put-call parity: C - P = S0 - K*exp(-rT)
    parity_LHS = BS_call_price - BS_put_price
    parity_RHS = S0 - K * math.exp(-r * T)
    print(f"\nPut-call parity check:")
    print(f"C - P = {parity_LHS:.6f}")
    print(f"S0 - K*exp(-rT) = {parity_RHS:.6f}")
    print(f"Difference: {abs(parity_LHS - parity_RHS):.2e}")