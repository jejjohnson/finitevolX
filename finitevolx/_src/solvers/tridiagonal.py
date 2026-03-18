"""Tridiagonal matrix solver (TDMA) for implicit vertical operations.

Provides a thin wrapper around :mod:`lineax`'s tridiagonal solver for
the classic Thomas-algorithm pattern used in ocean/atmosphere models:

    A x = d

where ``A`` is a tridiagonal matrix specified by its three diagonals
``(a, b, c)`` (lower, main, upper).

Primary use cases:
- Implicit vertical diffusion and friction
- Implicit vertical mixing (TKE closure)
- The implicit part of IMEX time integrators

The solver supports batched systems via :func:`eqx.filter_vmap`, making it
efficient for solving one tridiagonal system per horizontal column.

Usage example
-------------
>>> import jax.numpy as jnp
>>> from finitevolx._src.solvers.tridiagonal import solve_tridiagonal
>>> # Simple 4×4 system: diagonally dominant
>>> a = jnp.array([1.0, 1.0, 1.0])  # lower diagonal (n-1,)
>>> b = jnp.array([4.0, 4.0, 4.0, 4.0])  # main diagonal  (n,)
>>> c = jnp.array([1.0, 1.0, 1.0])  # upper diagonal (n-1,)
>>> d = jnp.array([1.0, 2.0, 3.0, 4.0])  # right-hand side (n,)
>>> x = solve_tridiagonal(a, b, c, d)
>>> x.shape
(4,)
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float
import lineax as lx


def solve_tridiagonal(
    lower: Float[Array, " n_minus_1"],
    diag: Float[Array, " n"],
    upper: Float[Array, " n_minus_1"],
    rhs: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Solve a tridiagonal linear system A x = d.

    Uses :class:`lineax.Tridiagonal`, which delegates to
    ``jax.lax.linalg.tridiagonal_solve`` (LAPACK/cuSPARSE under the hood).

    Parameters
    ----------
    lower : Float[Array, " n_minus_1"]
        Sub-diagonal of A, length ``n - 1``.
    diag : Float[Array, " n"]
        Main diagonal of A, length ``n``.
    upper : Float[Array, " n_minus_1"]
        Super-diagonal of A, length ``n - 1``.
    rhs : Float[Array, " n"]
        Right-hand side vector, length ``n``.

    Returns
    -------
    Float[Array, " n"]
        Solution vector x, length ``n``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx._src.solvers.tridiagonal import solve_tridiagonal
    >>> a = jnp.array([1.0, 1.0])
    >>> b = jnp.array([4.0, 4.0, 4.0])
    >>> c = jnp.array([1.0, 1.0])
    >>> d = jnp.array([6.0, 12.0, 14.0])
    >>> x = solve_tridiagonal(a, b, c, d)
    >>> x.shape
    (3,)
    """
    operator = lx.TridiagonalLinearOperator(diag, lower, upper)
    # throw=True (default) makes lineax raise on singular/ill-conditioned systems.
    sol = lx.linear_solve(operator, rhs, solver=lx.Tridiagonal())
    return sol.value


def solve_tridiagonal_batched(
    lower: Float[Array, "*batch n_minus_1"],
    diag: Float[Array, "*batch n"],
    upper: Float[Array, "*batch n_minus_1"],
    rhs: Float[Array, "*batch n"],
) -> Float[Array, "*batch n"]:
    """Solve independent tridiagonal systems over leading batch dimensions.

    This is a convenience wrapper that applies :func:`eqx.filter_vmap` over all
    leading dimensions of the input arrays.  Typical use: solve one vertical
    column per (j, i) horizontal grid point.

    Parameters
    ----------
    lower : Float[Array, "*batch n_minus_1"]
        Sub-diagonals, shape ``(*batch, n-1)``.
    diag : Float[Array, "*batch n"]
        Main diagonals, shape ``(*batch, n)``.
    upper : Float[Array, "*batch n_minus_1"]
        Super-diagonals, shape ``(*batch, n-1)``.
    rhs : Float[Array, "*batch n"]
        Right-hand sides, shape ``(*batch, n)``.

    Returns
    -------
    Float[Array, "*batch n"]
        Solutions, shape ``(*batch, n)``.

    Examples
    --------
    Solve 6×8 horizontal columns each with 10 vertical levels:

    >>> import jax, jax.numpy as jnp
    >>> key = jax.random.PRNGKey(0)
    >>> Ny, Nx, Nz = 6, 8, 10
    >>> b = 4.0 * jnp.ones((Ny, Nx, Nz))
    >>> a = jnp.ones((Ny, Nx, Nz - 1))
    >>> c = jnp.ones((Ny, Nx, Nz - 1))
    >>> d = jax.random.normal(key, (Ny, Nx, Nz))
    >>> x = solve_tridiagonal_batched(a, b, c, d)
    >>> x.shape
    (6, 8, 10)
    """
    n_batch = diag.ndim - 1
    fn = solve_tridiagonal
    for _ in range(n_batch):
        fn = eqx.filter_vmap(fn)
    return fn(lower, diag, upper, rhs)
