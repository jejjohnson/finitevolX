"""Linear and quadratic bottom-drag operators for Arakawa C-grids.

Both operators act on the velocity components at their native faces:

    linear:     du_drag[j, i+1/2] = -r_on_u[j, i+1/2] * u[j, i+1/2]
    quadratic:  du_drag[j, i+1/2] = -Cd * |u|_on_u[j, i+1/2] * u[j, i+1/2]
                                    / h_bot_on_u[j, i+1/2]

(and likewise for ``v`` at V-points).  The core math is
:func:`~finitevolx.linear_drag_tendency` and
:func:`~finitevolx.quadratic_drag_tendency`.

On a 3-D grid (``k = 0`` bottom ghost, ``k = Nz - 1`` top ghost) the drag
acts only on the bottom interior z-level ``k = 1``.  For multilayer models
apply the 2-D operator to the bottom layer only
(``drag(u[-1], v[-1], ...)``), or use :func:`~finitevolx.multilayer` to
drag every layer.

References
----------
.. [1] Veros ocean model, ``veros/core/friction.py``, functions
       ``linear_bottom_friction`` and ``quadratic_bottom_friction``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.forcing._base import AbstractForcing
from finitevolx._src.forcing._utils import (
    mask_where,
    on_face_interior,
    safe_denominator,
    safe_speed,
)
from finitevolx._src.forcing.functional import (
    linear_drag_tendency,
    quadratic_drag_tendency,
)
from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators.interpolation import Interpolation2D


class LinearDrag2D(AbstractForcing):
    """Linear drag on a 2-D Arakawa C-grid.

        du_drag[j, i+1/2] = -r_on_u[j, i+1/2] * u[j, i+1/2]
        dv_drag[j+1/2, i] = -r_on_v[j+1/2, i] * v[j+1/2, i]

    A spatially varying coefficient ``r`` is given at T-points and
    interpolated to U/V-points with simple x/y averaging.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, ``du_drag`` is zeroed at
        dry U-faces and ``dv_drag`` at dry V-faces (via ``jnp.where``, so
        NaN-filled land never leaks into the output).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, LinearDrag2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> drag = LinearDrag2D(grid=grid)
    >>> u = jnp.ones((grid.Ny, grid.Nx))
    >>> v = jnp.ones((grid.Ny, grid.Nx))
    >>> du_drag, dv_drag = drag(u, v, r=1e-7)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None
    interp: Interpolation2D

    def __init__(
        self,
        grid: CartesianGrid2D,
        mask: Mask2D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.interp = Interpolation2D(grid=grid)

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        r: float | Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Linear drag tendencies (du_drag, dv_drag).

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        r : float or Float[Array, "Ny Nx"]
            Drag coefficient [1/s] — a scalar, or a T-point field (ghost
            ring filled) interpolated to U/V-points.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(du_drag, dv_drag)`` at U- and V-points, zero in the ghost
            ring.  When ``self.mask`` is set, dry faces are zeroed.
        """
        # r_on_u[j, i+1/2] = 1/2 * (r[j, i] + r[j, i+1])
        r_on_u = on_face_interior(r, self.interp.T_to_U)
        # r_on_v[j+1/2, i] = 1/2 * (r[j, i] + r[j+1, i])
        r_on_v = on_face_interior(r, self.interp.T_to_V)

        # du_drag[j, i+1/2] = -r_on_u * u
        du_drag = interior(linear_drag_tendency(u[1:-1, 1:-1], r_on_u), u)
        # dv_drag[j+1/2, i] = -r_on_v * v
        dv_drag = interior(linear_drag_tendency(v[1:-1, 1:-1], r_on_v), v)

        if self.mask is None:
            return du_drag, dv_drag
        return mask_where(du_drag, self.mask.u), mask_where(dv_drag, self.mask.v)


class QuadraticDrag2D(AbstractForcing):
    """Quadratic bottom drag on a 2-D Arakawa C-grid.

        du_drag[j, i+1/2] = -Cd * |u|_on_u * u[j, i+1/2] / h_bot_on_u
        dv_drag[j+1/2, i] = -Cd * |u|_on_v * v[j+1/2, i] / h_bot_on_v

    The flow speed is evaluated **at each face** from the native component
    and the 4-point average of the cross component:

        |u|_on_u[j, i+1/2] = sqrt(u[j, i+1/2]^2 + v_on_u[j, i+1/2]^2)
        |u|_on_v[j+1/2, i] = sqrt(v[j+1/2, i]^2 + u_on_v[j+1/2, i]^2)

    so boundary faces see the caller-filled ghost velocities rather than a
    zero ghost-ring speed.  The speed is gradient-safe at rest.  Spatially
    varying ``cd`` / ``h_bot`` are given at T-points and interpolated to
    U/V-points with simple x/y averaging.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, ``du_drag`` is zeroed at
        dry U-faces and ``dv_drag`` at dry V-faces (via ``jnp.where``, so
        NaN-filled land never leaks into the output).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, QuadraticDrag2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> drag = QuadraticDrag2D(grid=grid)
    >>> u = jnp.ones((grid.Ny, grid.Nx))
    >>> v = jnp.zeros((grid.Ny, grid.Nx))
    >>> du_drag, dv_drag = drag(u, v, cd=2.5e-3, h_bot=100.0)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None
    interp: Interpolation2D

    def __init__(
        self,
        grid: CartesianGrid2D,
        mask: Mask2D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.interp = Interpolation2D(grid=grid)

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        cd: float | Float[Array, "Ny Nx"],
        h_bot: float | Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Quadratic drag tendencies (du_drag, dv_drag).

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        cd : float or Float[Array, "Ny Nx"]
            Dimensionless drag coefficient — a scalar, or a T-point field
            (ghost ring filled) interpolated to U/V-points.
        h_bot : float or Float[Array, "Ny Nx"]
            Bottom-layer thickness [m] — a scalar, or a T-point field
            (ghost ring filled) interpolated to U/V-points.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(du_drag, dv_drag)`` at U- and V-points, zero in the ghost
            ring.  When ``self.mask`` is set, dry faces are zeroed.
        """
        mu = None if self.mask is None else self.mask.u
        mv = None if self.mask is None else self.mask.v
        return self._tendency(u, v, cd, h_bot, mu, mv)

    def _tendency(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        cd: float | Float[Array, "Ny Nx"],
        h_bot: float | Float[Array, "Ny Nx"],
        mu: Array | None,
        mv: Array | None,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Core stencil with explicit ``[Ny, Nx]`` face masks (or ``None``).

        Shared with :class:`QuadraticDrag3D`, which passes one z-level's masks.
        """
        # v_on_u[j, i+1/2] = 1/4*(v[j+1/2,i] + v[j-1/2,i] + v[j+1/2,i+1] + v[j-1/2,i+1])
        v_on_u = self.interp.V_to_U(v)[1:-1, 1:-1]
        # u_on_v[j+1/2, i] = 1/4*(u[j,i+1/2] + u[j+1,i+1/2] + u[j,i-1/2] + u[j+1,i-1/2])
        u_on_v = self.interp.U_to_V(u)[1:-1, 1:-1]
        u_in = u[1:-1, 1:-1]
        v_in = v[1:-1, 1:-1]

        # |u|_on_u[j, i+1/2] = sqrt(u^2 + v_on_u^2)
        speed_u = safe_speed(u_in, v_on_u)
        # |u|_on_v[j+1/2, i] = sqrt(v^2 + u_on_v^2)
        speed_v = safe_speed(v_in, u_on_v)

        cd_on_u = on_face_interior(cd, self.interp.T_to_U)
        cd_on_v = on_face_interior(cd, self.interp.T_to_V)
        # Swap in 1 at dry faces so a zero dry-cell thickness cannot give 0/0.
        h_on_u = safe_denominator(on_face_interior(h_bot, self.interp.T_to_U), mu)
        h_on_v = safe_denominator(on_face_interior(h_bot, self.interp.T_to_V), mv)

        # du_drag[j, i+1/2] = -Cd * |u|_on_u * u / h_bot_on_u
        du_drag = interior(quadratic_drag_tendency(u_in, speed_u, cd_on_u, h_on_u), u)
        # dv_drag[j+1/2, i] = -Cd * |u|_on_v * v / h_bot_on_v
        dv_drag = interior(quadratic_drag_tendency(v_in, speed_v, cd_on_v, h_on_v), v)

        return mask_where(du_drag, mu), mask_where(dv_drag, mv)


# Bottom interior z-level: k = 0 is the bottom ghost on ArakawaCGrid3D.
_K_BOT = 1


def _inject_bottom(
    du_2d: Float[Array, "Ny Nx"],
    dv_2d: Float[Array, "Ny Nx"],
    u: Float[Array, "Nz Ny Nx"],
    v: Float[Array, "Nz Ny Nx"],
) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
    """Embed 2-D bottom tendencies at level ``k = 1`` of zero 3-D arrays."""
    du = jnp.zeros(u.shape, dtype=jnp.result_type(u, du_2d)).at[_K_BOT].set(du_2d)
    dv = jnp.zeros(v.shape, dtype=jnp.result_type(v, dv_2d)).at[_K_BOT].set(dv_2d)
    return du, dv


class LinearDrag3D(AbstractForcing):
    """Linear bottom drag on a 3-D Arakawa C-grid.

    Applies the :class:`LinearDrag2D` stencil to the **bottom interior
    z-level only** and returns zero at every other level.  The 3-D grid
    stacks levels bottom-up (``k = 0`` bottom ghost, ``k = Nz - 1`` top
    ghost), so the bottom interior level is ``k_bot = 1``:

        du_drag[1, j, i+1/2] = -r_on_u[j, i+1/2] * u[1, j, i+1/2]
        dv_drag[1, j+1/2, i] = -r_on_v[j+1/2, i] * v[1, j+1/2, i]

    The bottom is the lowest interior level everywhere; a per-column
    sea-floor index (e.g. ``Mask3D.k_bottom``) is not used.

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional land/ocean mask.  The ``k_bot`` slices of ``mask.u`` /
        ``mask.v`` zero dry faces via ``jnp.where``, so NaN-filled land
        never leaks into the output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid3D, LinearDrag3D
    >>> grid = CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
    >>> drag = LinearDrag3D(grid=grid)
    >>> u = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> du_drag, dv_drag = drag(u, u, r=1e-7)
    """

    grid: CartesianGrid3D
    mask: Mask3D | None
    _drag2d: LinearDrag2D

    def __init__(
        self,
        grid: CartesianGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        # The inner 2-D op is mask-free; the 3-D wrapper passes per-level masks.
        self._drag2d = LinearDrag2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        r: float | Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Linear drag tendencies, non-zero only at the bottom interior level.

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.
        r : float or Float[Array, "Ny Nx"]
            Drag coefficient [1/s] — a scalar, or a T-point field (ghost
            ring filled) interpolated to U/V-points.

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(du_drag, dv_drag)`` at U- and V-points, zero outside
            ``k = 1`` and in the ghost ring.  When ``self.mask`` is
            set, dry faces are zeroed.
        """
        du_2d, dv_2d = self._drag2d(u[_K_BOT], v[_K_BOT], r)
        if self.mask is not None:
            du_2d = mask_where(du_2d, self.mask.u[_K_BOT])
            dv_2d = mask_where(dv_2d, self.mask.v[_K_BOT])
        return _inject_bottom(du_2d, dv_2d, u, v)


class QuadraticDrag3D(AbstractForcing):
    """Quadratic bottom drag on a 3-D Arakawa C-grid.

    Applies the :class:`QuadraticDrag2D` stencil to the **bottom interior
    z-level only** and returns zero at every other level.  The 3-D grid
    stacks levels bottom-up (``k = 0`` bottom ghost, ``k = Nz - 1`` top
    ghost), so the bottom interior level is ``k_bot = 1``:

        du_drag[1, j, i+1/2] = -Cd * |u|_on_u * u[1, j, i+1/2] / h_bot_on_u
        dv_drag[1, j+1/2, i] = -Cd * |u|_on_v * v[1, j+1/2, i] / h_bot_on_v

    The bottom is the lowest interior level everywhere; a per-column
    sea-floor index (e.g. ``Mask3D.k_bottom``) is not used.

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional land/ocean mask.  The ``k_bot`` slices of ``mask.u`` /
        ``mask.v`` zero dry faces via ``jnp.where``, so NaN-filled land
        never leaks into the output.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid3D, QuadraticDrag3D
    >>> grid = CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
    >>> drag = QuadraticDrag3D(grid=grid)
    >>> u = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> du_drag, dv_drag = drag(u, u, cd=2.5e-3, h_bot=grid.dz)
    """

    grid: CartesianGrid3D
    mask: Mask3D | None
    _drag2d: QuadraticDrag2D

    def __init__(
        self,
        grid: CartesianGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        # The inner 2-D op is mask-free; the 3-D wrapper passes per-level masks.
        self._drag2d = QuadraticDrag2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        cd: float | Float[Array, "Ny Nx"],
        h_bot: float | Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Quadratic drag tendencies, non-zero only at the bottom interior level.

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.
        cd : float or Float[Array, "Ny Nx"]
            Dimensionless drag coefficient — a scalar, or a T-point field
            (ghost ring filled) interpolated to U/V-points.
        h_bot : float or Float[Array, "Ny Nx"]
            Bottom-layer thickness [m] — a scalar, or a T-point field
            (ghost ring filled) interpolated to U/V-points.

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(du_drag, dv_drag)`` at U- and V-points, zero outside
            ``k = 1`` and in the ghost ring.  When ``self.mask`` is
            set, dry faces are zeroed.
        """
        mu = None if self.mask is None else self.mask.u[_K_BOT]
        mv = None if self.mask is None else self.mask.v[_K_BOT]
        du_2d, dv_2d = self._drag2d._tendency(u[_K_BOT], v[_K_BOT], cd, h_bot, mu, mv)
        return _inject_bottom(du_2d, dv_2d, u, v)
