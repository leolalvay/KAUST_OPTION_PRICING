#!/usr/bin/env python
"""Simple test to debug JAX issues"""

import jax.numpy as jnp
from jax import random

print("Testing JAX imports...")

from JAX_basket_simulation import simulate_gbm_paths

print("Imports successful!")

# Test simple path generation
key = random.PRNGKey(42)
S0 = jnp.array([225., 250., 275.])[:, jnp.newaxis]
r = 0.05
vol = jnp.array([0.2, 0.15, 0.1])
cov_mat = jnp.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.1], [0.3, 0.1, 1.0]])

dt = 0.01
N_steps = 100
N_paths = 10

# Generate random increments
Z_random = random.normal(key, shape=(N_paths, N_steps - 1, 3))

print(f"Testing simulate_gbm_paths with N_paths={N_paths}, N_steps={N_steps}")
paths = simulate_gbm_paths(S0, r, vol, cov_mat, dt, N_steps, N_paths, Z_random)

print(f"Success! Paths shape: {paths.shape}")
print(f"First path, first step: {paths[0, 0, :]}")
print(f"First path, last step: {paths[0, -1, :]}")
