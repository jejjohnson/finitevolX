"""Rayleigh damping operator for Arakawa C-grids.

Relaxes a T-point field toward a reference state at a prescribed rate:

    dq[j, i] = -r[j, i] * (q[j, i] - q_ref[j, i])

The core math is :func:`~finitevolx.rayleigh_tendency`.

References
----------
.. [1] Veros ocean model, ``veros/core/friction.py``, function
       ``rayleigh_friction``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.forcing._base import AbstractForcing
from finitevolx._src.forcing._utils import interior_or_scalar, mask_where
from finitevolx._src.forcing.functional import rayleigh_tendency
from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.mask import Mask2D
from finitevolx._src.operators._ghost import interior


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
