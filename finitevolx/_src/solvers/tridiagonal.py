"""Tridiagonal matrix solver (TDMA) for implicit vertical operations.

The tridiagonal (Thomas-algorithm) solve now lives in ``gaussx`` as part of the
shared solver substrate; this module re-exports it so the finitevolX public API
is unchanged. Both functions delegate to ``lineax``'s tridiagonal solver
(``jax.lax.linalg.tridiagonal_solve``, LAPACK / cuSPARSE) under the hood.

Primary use cases:
- Implicit vertical diffusion and friction
- Implicit vertical mixing (TKE closure)
- The implicit part of IMEX time integrators

Usage example
-------------
>>> import jax.numpy as jnp
>>> from finitevolx._src.solvers.tridiagonal import solve_tridiagonal
>>> a = jnp.array([1.0, 1.0, 1.0])  # lower diagonal (n-1,)
>>> b = jnp.array([4.0, 4.0, 4.0, 4.0])  # main diagonal  (n,)
>>> c = jnp.array([1.0, 1.0, 1.0])  # upper diagonal (n-1,)
>>> d = jnp.array([1.0, 2.0, 3.0, 4.0])  # right-hand side (n,)
>>> x = solve_tridiagonal(a, b, c, d)
>>> x.shape
(4,)
"""

from __future__ import annotations

from gaussx import solve_tridiagonal, solve_tridiagonal_batched

__all__ = ["solve_tridiagonal", "solve_tridiagonal_batched"]
