"""Rayleigh damping operator for Arakawa C-grids.

Relaxes a T-point field toward a reference state at a prescribed rate:

    dq[j, i] = -r[j, i] * (q[j, i] - q_ref[j, i])

The core math is :func:`~finitevolx.rayleigh_tendency`.  On a 3-D grid the
damping acts uniformly at every interior z-level.

References
----------
.. [1] Veros ocean model, ``veros/core/friction.py``, function
       ``rayleigh_friction``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.forcing._base import AbstractForcing
from finitevolx._src.forcing._utils import interior_or_scalar, mask_where
from finitevolx._src.forcing.functional import rayleigh_tendency
from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts


class RayleighDamping2D(AbstractForcing):
    """Rayleigh damping of a T-point field on a 2-D Arakawa C-grid.

        dq[j, i] = -r[j, i] * (q[j, i] - q_ref[j, i])

    Works on any scalar field at T-points (layer thickness, tracer, ...).
    With ``q_ref=None`` this is pure linear damping toward zero.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, ``dq`` is zeroed at dry
        T-cells (via ``jnp.where``, so NaN-filled land never leaks into the
        output).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, RayleighDamping2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> damp = RayleighDamping2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> dh = damp(h, r=1e-5, q_ref=0.5 * h)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None = None

    def __call__(
        self,
        q: Float[Array, "Ny Nx"],
        r: float | Float[Array, "Ny Nx"],
        q_ref: float | Float[Array, "Ny Nx"] | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """Rayleigh damping tendency at T-points.

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Field at T-points.
        r : float or Float[Array, "Ny Nx"]
            Damping rate [1/s] — a scalar or a T-point field.
        q_ref : float, Float[Array, "Ny Nx"] or None, optional
            Reference state — a scalar or a T-point field.  ``None``
            (default) damps toward zero.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Damping tendency at T-points, zero in the ghost ring.  When
            ``self.mask`` is set, dry cells are zeroed.
        """
        if self.mask is not None:
            # Zero dry cells first: NaN-filled land would otherwise give NaN
            # gradients through -r * (q - q_ref) even where dq is masked.
            q = mask_where(q, self.mask.h)
            if q_ref is not None and jnp.ndim(q_ref) == 2:
                q_ref = mask_where(jnp.asarray(q_ref), self.mask.h)

        q_ref_in = None if q_ref is None else interior_or_scalar(q_ref)
        # dq[j, i] = -r[j, i] * (q[j, i] - q_ref[j, i])
        dq = interior(
            rayleigh_tendency(q[1:-1, 1:-1], interior_or_scalar(r), q_ref_in), q
        )

        if self.mask is None:
            return dq
        return mask_where(dq, self.mask.h)


def _level_axis(x: Array) -> int | None:
    """vmap axis for a coefficient: per-level (``[Nz]`` / ``[Nz, Ny, Nx]``) or broadcast."""
    return 0 if x.ndim in (1, 3) else None


class RayleighDamping3D(AbstractForcing):
    """Rayleigh damping of a T-point field on a 3-D Arakawa C-grid.

    Applies the :class:`RayleighDamping2D` stencil uniformly at every
    z-level and zeroes the z-ghost slices:

        dq[k, j, i] = -r[k, j, i] * (q[k, j, i] - q_ref[k, j, i])

    ``r`` and ``q_ref`` may each be a scalar, a per-level profile ``[Nz]``,
    a horizontal field ``[Ny, Nx]`` (broadcast over depth), or a full
    ``[Nz, Ny, Nx]`` field.

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional land/ocean mask.  The inner :class:`RayleighDamping2D` is
        mask-free; dry cells of the 3-D result are zeroed with ``mask.h``
        via ``jnp.where``, so NaN-filled land never leaks into the output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid3D, RayleighDamping3D
    >>> grid = CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
    >>> damp = RayleighDamping3D(grid=grid)
    >>> q = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> r = jnp.linspace(0.0, 1e-5, grid.Nz)  # per-level damping rate
    >>> dq = damp(q, r)
    """

    grid: CartesianGrid3D
    mask: Mask3D | None
    _damp2d: RayleighDamping2D

    def __init__(
        self,
        grid: CartesianGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        # The inner 2-D op is mask-free; the 3-D wrapper owns the mask.
        self._damp2d = RayleighDamping2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        q: Float[Array, "Nz Ny Nx"],
        r: float
        | Float[Array, "Nz"]
        | Float[Array, "Ny Nx"]
        | Float[Array, "Nz Ny Nx"],
        q_ref: float
        | Float[Array, "Nz"]
        | Float[Array, "Ny Nx"]
        | Float[Array, "Nz Ny Nx"]
        | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Rayleigh damping tendency at T-points over all z-levels.

        Parameters
        ----------
        q : Float[Array, "Nz Ny Nx"]
            Field at T-points.
        r : float or Float[Array, "Nz"] or Float[Array, "Ny Nx"] or Float[Array, "Nz Ny Nx"]
            Damping rate [1/s].
        q_ref : same options as ``r``, or None, optional
            Reference state.  ``None`` (default) damps toward zero.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Damping tendency at T-points, zero in the ghost ring and the
            z-ghost slices.  When ``self.mask`` is set, dry cells are
            zeroed.
        """
        if self.mask is not None:
            # Zero dry cells first: NaN-filled land would otherwise give NaN
            # gradients through -r * (q - q_ref) even where dq is masked.
            q = mask_where(q, self.mask.h)
            if q_ref is not None and jnp.ndim(q_ref) == 3:
                q_ref = mask_where(jnp.asarray(q_ref), self.mask.h)

        r_arr = jnp.asarray(r)
        if q_ref is None:
            dq = eqx.filter_vmap(
                self._damp2d,
                in_axes=(0, _level_axis(r_arr)),
            )(q, r_arr)
        else:
            ref_arr = jnp.asarray(q_ref)
            dq = eqx.filter_vmap(
                self._damp2d,
                in_axes=(0, _level_axis(r_arr), _level_axis(ref_arr)),
            )(q, r_arr, ref_arr)

        # Zero the z-ghost slices to match the 3-D ghost-ring convention.
        dq = zero_z_ghosts(dq)

        if self.mask is None:
            return dq
        return mask_where(dq, self.mask.h)
