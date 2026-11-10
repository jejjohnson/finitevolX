"""Semi-Lagrangian advection: a pure step and its diffrax solver.

Traces characteristic curves backward in time and interpolates the old field
at the departure points.  Unlike Eulerian methods, the semi-Lagrangian scheme
is **unconditionally stable** — the CFL number can exceed 1.

Uses :func:`jax.scipy.ndimage.map_coordinates` for interpolation.

References
----------
- Staniforth & Côté (1991) — Semi-Lagrangian integration schemes.
- ECMWF IFS uses semi-Lagrangian as its primary advection method.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import diffrax as dfx
from diffrax import RESULTS
import equinox as eqx
import jax
import jax.numpy as jnp


def semi_lagrangian_step(
    field: jax.Array,
    u: jax.Array,
    v: jax.Array,
    dx: float,
    dy: float,
    dt: float,
    interp_order: int = 1,
    bc: str = "periodic",
) -> jax.Array:
    """Advect a 2D scalar field using semi-Lagrangian backtracking.

    Algorithm::

        1. Compute departure points: x_dep = x_i - u * dt,
                                      y_dep = y_j - v * dt
        2. Interpolate ``field`` at the departure points.
        3. Return interpolated values as the new field.

    Parameters
    ----------
    field : Array[Ny, Nx]
        Scalar field to advect.
    u, v : Array[Ny, Nx]
        Velocity components at the same grid points as ``field``, in
        **physical units** (m/s).
    dx, dy : float
        Grid spacing in x and y (m).
    dt : float
        Timestep (s).
    interp_order : int, optional
        Interpolation order passed to :func:`jax.scipy.ndimage.map_coordinates`.
        Currently JAX only supports ``order <= 1``.  Default 1.
    bc : str, optional
        Boundary handling: ``"periodic"`` (wrap) or ``"edge"``
        (Neumann-like clamp).  Default ``"periodic"``.

    Returns
    -------
    Array[Ny, Nx]
        Advected field.
    """
    if bc not in {"periodic", "edge"}:
        raise ValueError(f"bc must be 'periodic' or 'edge', got {bc!r}")
    if interp_order not in {0, 1}:
        raise ValueError(
            f"interp_order must be 0 or 1 (JAX limitation), got {interp_order}"
        )

    ny, nx = field.shape

    # Target grid coordinates (in index space)
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(ny, dtype=field.dtype),
        jnp.arange(nx, dtype=field.dtype),
        indexing="ij",
    )

    # Departure points in index space
    x_dep = x_coords - u * dt / dx
    y_dep = y_coords - v * dt / dy

    mode = "wrap" if bc == "periodic" else "nearest"

    return jax.scipy.ndimage.map_coordinates(
        field,
        [y_dep, x_dep],
        order=interp_order,
        mode=mode,
    )


class SemiLagrangianTerm(dfx.AbstractTerm):
    """Velocity term for :class:`SemiLagrangianSolver` in ``diffeqsolve``.

    ``dfx.ODETerm`` requires its vector field to return a pytree shaped like
    ``y0``, but semi-Lagrangian advection needs the velocity ``(u, v)`` of a
    scalar field ``y``.  This term carries that velocity function.

    Parameters
    ----------
    velocity : Callable[[float, Array, PyTree], tuple[Array, Array]]
        ``velocity(t, y, args) -> (u, v)``, each shaped like ``y``, in
        **grid index units per unit time** (physical velocity divided by
        the grid spacing).

    Examples
    --------
    >>> term = SemiLagrangianTerm(
    ...     lambda t, y, args: (jnp.ones_like(y), jnp.zeros_like(y))
    ... )
    >>> sol = dfx.diffeqsolve(term, SemiLagrangianSolver(), 0.0, 4.0, 1.0, y0)
    """

    velocity: Callable

    def vf(self, t, y, args):
        return self.velocity(t, y, args)

    def contr(self, t0, t1, **kwargs):
        del kwargs
        return t1 - t0

    def prod(self, vf, control):
        return jax.tree.map(lambda v: v * control, vf)


class SemiLagrangianSolver(dfx.AbstractSolver):
    """Semi-Lagrangian advection solver for diffrax.

    Traces characteristic curves backward in time and interpolates the old
    field at departure points.  Unconditionally stable (CFL > 1 allowed).

    The term's ``vf(t, y, args)`` must return ``(u, v)`` velocity components
    in **grid index units per second** (i.e. physical velocity divided by
    grid spacing); wrap the velocity function in :class:`SemiLagrangianTerm`
    to use the solver with ``diffeqsolve``.  The departure point is found
    with one backward Euler step along a velocity frozen at ``t0``, so the
    scheme is first order in time (:meth:`order` is 1); there is no error
    estimate, so it runs with fixed steps only.  Dense output (``SaveAt(ts=...)``)
    is linear between steps.

    Parameters
    ----------
    interpolation_order : int
        0 = nearest-neighbour, 1 = linear (diffusive, monotone).
        JAX currently only supports ``order <= 1``.
    """

    term_structure: ClassVar[Any] = dfx.AbstractTerm
    interpolation_cls: ClassVar[Any] = dfx.LocalLinearInterpolation
    interpolation_order: int = eqx.field(static=True, default=1)

    def order(self, terms):
        # first-order backward trajectory; the spatial interpolation order
        # (interpolation_order) is a separate, spatial accuracy
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        dt = t1 - t0

        # Velocity in grid-index units/second
        u, v = terms.vf(t0, y0, args)

        ny, nx = y0.shape
        y_coords, x_coords = jnp.meshgrid(
            jnp.arange(ny, dtype=y0.dtype),
            jnp.arange(nx, dtype=y0.dtype),
            indexing="ij",
        )

        x_dep = x_coords - u * dt
        y_dep = y_coords - v * dt

        y1 = jax.scipy.ndimage.map_coordinates(
            y0, [y_dep, x_dep], order=self.interpolation_order, mode="wrap"
        )

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
