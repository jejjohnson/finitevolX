"""Base classes and helpers for the `finitevolx.pipekit` operator layer.

Two tiers, as in geotoolz::

    Tier A  finitevolX primitives        Difference2D(grid).diff_x_T_to_U(h)
            arrays in, arrays out;       eqx.Module / pure function
            no config, no carrier
                    │
                    ▼
    Tier B  pipekit Operators            Difference(grid=G, axis="x", src="T", dst="U")
            JSON config, eqx.Module      composes in pipekit Sequential / Graph,
            leaves, `|` composition      runs under jit / grad / vmap

A Tier-B operator stores its grid as a static JSON tuple (`GridSpec`) and
rebuilds the Tier-A grid from it at trace time, so the grid never crosses
``jit`` as a pytree of Python scalars (finitevolX #321) and the config
round-trips through ``op.state`` / ``Operator.from_state``.

Fields are ``[*batch, Ny, Nx]``: the 2-D Tier-A operators broadcast over
leading axes themselves (finitevolX #386), so Tier B calls them directly.
`over_leading` lifts a Tier-A function that is still 2-D only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from typing import Any

import equinox as eqx
import jax
from jaxtyping import Array
from pipekit_jax import JaxOperator

from finitevolx._src.grid.cartesian import CartesianGrid2D

#: ``(nx_interior, ny_interior, Lx, Ly)``: the arguments of
#: `CartesianGrid2D.from_interior`, as JSON primitives.
GridSpec = tuple[int, int, float, float]


def as_grid_spec(spec: Sequence[Any] | CartesianGrid2D) -> GridSpec:
    """Normalise a grid description to a `GridSpec` tuple.

    Parameters
    ----------
    spec : sequence of 4 numbers or CartesianGrid2D
        ``(nx_interior, ny_interior, Lx, Ly)`` (a list after a JSON round
        trip), or a grid to describe.

    Returns
    -------
    GridSpec
        ``(int, int, float, float)``, hashable, so it can be a static field.

    Raises
    ------
    ValueError
        If ``spec`` does not have four entries or a count is not positive.
    """
    if isinstance(spec, CartesianGrid2D):
        spec = (spec.Nx - 2, spec.Ny - 2, spec.Lx, spec.Ly)
    if len(spec) != 4:
        raise ValueError(
            f"grid spec must be (nx_interior, ny_interior, Lx, Ly), got {spec!r}"
        )
    nx, ny, lx, ly = spec
    if int(nx) < 1 or int(ny) < 1:
        raise ValueError(f"grid spec needs positive cell counts, got {spec!r}")
    return (int(nx), int(ny), float(lx), float(ly))


@functools.lru_cache(maxsize=64)
def build_grid(spec: GridSpec) -> CartesianGrid2D:
    """The `CartesianGrid2D` a `GridSpec` describes (cached per spec).

    Parameters
    ----------
    spec : GridSpec
        ``(nx_interior, ny_interior, Lx, Ly)``.

    Returns
    -------
    CartesianGrid2D
        ``CartesianGrid2D.from_interior(*spec)``.
    """
    nx, ny, lx, ly = as_grid_spec(spec)
    return CartesianGrid2D.from_interior(nx, ny, lx, ly)


def over_leading(fn: Callable[..., Any], core_ndim: int = 2) -> Callable[..., Any]:
    """Lift ``fn`` on ``[*core]`` arrays to ``[*batch, *core]`` arrays.

    Every input shares the same leading shape; it is flattened to one axis,
    ``jax.vmap(fn)`` maps over it, and every output array gets the leading
    shape back::

        x : [T, E, Ny, Nx] ─reshape─► [T·E, Ny, Nx] ─vmap fn─► [T·E, …] ─reshape─► [T, E, …]

    Only needed for Tier-A functions that are still 2-D only; the 2-D
    Cartesian operators take ``[*batch, Ny, Nx]`` natively.

    Parameters
    ----------
    fn : callable
        Function of positional arrays with ``core_ndim`` trailing axes each;
        it may return an array or a pytree of arrays.
    core_ndim : int
        Number of trailing axes ``fn`` consumes (2 for ``[Ny, Nx]``).

    Returns
    -------
    callable
        ``fn`` lifted over any number of leading axes (zero included).

    Raises
    ------
    ValueError
        If the inputs disagree on their leading shape.
    """

    @functools.wraps(fn)
    def lifted(*xs: Array) -> Any:
        leads = {tuple(x.shape[: x.ndim - core_ndim]) for x in xs}
        if len(leads) != 1:
            raise ValueError(f"inputs have different leading shapes: {sorted(leads)}")
        (lead,) = leads
        if not lead:
            return fn(*xs)
        flat = [x.reshape(-1, *x.shape[x.ndim - core_ndim :]) for x in xs]
        out = jax.vmap(fn)(*flat)
        return jax.tree.map(lambda y: y.reshape(*lead, *y.shape[1:]), out)

    return lifted


class FVXOperator(JaxOperator):
    """Base for finitevolX pipekit operators.

    A `pipekit_jax.JaxOperator` (pipekit ``Operator`` + ``eqx.Module``):
    config fields are ``eqx.field(static=True)``, learnable coefficients
    are leaves with ``converter=jnp.asarray``, and ``_apply`` maps fields
    ``[*batch, Ny, Nx]`` to fields.
    """


class GridOperator(FVXOperator):
    """An `FVXOperator` on a 2-D Cartesian C-grid.

    Attributes
    ----------
    grid : GridSpec
        ``(nx_interior, ny_interior, Lx, Ly)``, static; a list (after a
        JSON round trip) or a `CartesianGrid2D` is converted on
        construction.
    """

    grid: GridSpec = eqx.field(static=True, converter=as_grid_spec)

    @property
    def cartesian_grid(self) -> CartesianGrid2D:
        """The Tier-A grid, rebuilt from `grid` (cached per spec)."""
        return build_grid(self.grid)
