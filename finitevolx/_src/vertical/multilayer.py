"""Convenience wrapper to lift 2D horizontal operators to multilayer fields.

Multilayer models (e.g., baroclinic shallow water, quasi-geostrophic with
multiple layers) treat the leading axis of a ``[nl, Ny, Nx]`` array as a
*batch* of independent 2D horizontal fields — one per layer or vertical mode.
There are no ghost cells in the layer dimension; every slice is a real,
physical 2D field.

This is fundamentally different from the :class:`~finitevolx.Difference3D`
(or :class:`~finitevolx.Interpolation3D`) operators, which treat the leading
axis as a *spatial* vertical dimension with ghost shells at ``k=0`` and
``k=Nz-1`` that must be filled by boundary conditions before use.  Only the
interior ``k=1..Nz-2`` slices are written by those operators.

Use :func:`multilayer` to lift any 2D callable (e.g., a bound method of
:class:`~finitevolx.Difference2D`) to operate on all layers in parallel:

    >>> import jax.numpy as jnp
    >>> import finitevolx as fvx
    >>> grid = fvx.CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> diff2d = fvx.Difference2D(grid=grid)
    >>> h = jnp.ones((4, grid.Ny, grid.Nx))  # 4 layers
    >>> dh_dx = fvx.multilayer(diff2d.diff_x_T_to_U)(h)  # shape [4, Ny, Nx]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx


def multilayer(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Lift a 2D horizontal operator to multilayer/multimode fields.

    Wraps ``fn`` with :func:`eqx.filter_vmap` so that it is applied
    independently to every layer (or mode) along the leading batch axis.
    The result is mathematically and numerically identical to calling ``fn``
    on each ``[Ny, Nx]`` slice in sequence, but is executed in a single
    vectorised pass via JAX.

    Parameters
    ----------
    fn : callable
        A callable that maps one or more ``[Ny, Nx]`` arrays to one or more
        ``[Ny, Nx]`` arrays (e.g., a bound method of
        :class:`~finitevolx.Difference2D`).

    Returns
    -------
    callable
        A new callable that accepts ``[nl, Ny, Nx]`` arrays and returns the
        same structure with shape ``[nl, Ny, Nx]``.  The leading axis is
        vmapped over; the inner ``[Ny, Nx]`` stencil is unchanged.

    Notes
    -----
    **Multilayer** (this function) vs. **true 3D**
    (:class:`~finitevolx.Difference3D`):

    * ``multilayer(fn)`` treats every layer as a real, independent 2D field.
      There are **no ghost layers**; the operator applies to **all** ``nl``
      slices, including ``k=0`` and ``k=nl-1``.
    * :class:`~finitevolx.Difference3D` treats the leading axis as a vertical
      *spatial* dimension with ghost shells at ``k=0`` and ``k=Nz-1`` that
      are kept at zero until filled by boundary conditions.  Only the interior
      ``k=1..Nz-2`` slices are written.

    These two approaches are **not interchangeable**.  Use ``multilayer`` for
    baroclinic or quasi-geostrophic layered models where every layer holds a
    real, independent horizontal field.  Use ``Difference3D`` for true 3D
    primitive-equation discretizations where vertical ghost layers are
    required at the top and bottom boundaries.

    Examples
    --------
    Apply :meth:`~finitevolx.Difference2D.diff_x_T_to_U` to 3 layers at once:

    >>> import jax.numpy as jnp
    >>> import finitevolx as fvx
    >>> grid = fvx.CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> diff2d = fvx.Difference2D(grid=grid)
    >>> h = jnp.ones((3, grid.Ny, grid.Nx))  # 3 layers
    >>> dh = fvx.multilayer(diff2d.diff_x_T_to_U)(h)
    >>> dh.shape
    (3, 10, 10)

    For operators that take multiple arguments (e.g., divergence), pass the
    bound method directly — :func:`eqx.filter_vmap` batches each positional
    argument over the leading axis independently:

    >>> u = jnp.ones((3, grid.Ny, grid.Nx))
    >>> v = jnp.ones((3, grid.Ny, grid.Nx))
    >>> div = fvx.multilayer(diff2d.divergence)(u, v)
    >>> div.shape
    (3, 10, 10)
    """
    return eqx.filter_vmap(fn)
