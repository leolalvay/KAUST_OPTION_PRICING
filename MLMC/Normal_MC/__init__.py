"""
Normal_MC package - Standard Monte Carlo methods for option pricing.
"""

from .BS_Analytic import BS_call, BS_put
from .MC_estimator import run_mc_estimation
from .MC_estimator_JAX import run_mc_estimation_jax
