"""Linear and quadratic bottom-drag operators for Arakawa C-grids.

Both operators act on the velocity components at their native faces:

    linear:     du_drag[j, i+1/2] = -r_on_u[j, i+1/2] * u[j, i+1/2]
    quadratic:  du_drag[j, i+1/2] = -Cd * |u|_on_u[j, i+1/2] * u[j, i+1/2]
                                    / h_bot_on_u[j, i+1/2]

(and likewise for ``v`` at V-points).  The core math is
:func:`~finitevolx.linear_drag_tendency` and
:func:`~finitevolx.quadratic_drag_tendency`.

For multilayer models apply the 2-D operator to the bottom layer only
(``drag(u[-1], v[-1], ...)``), or use :func:`~finitevolx.multilayer` to
drag every layer.

References
----------
.. [1] Veros ocean model, ``veros/core/friction.py``, functions
       ``linear_bottom_friction`` and ``quadratic_bottom_friction``.
"""

from __future__ import annotations

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
from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.mask import Mask2D
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
        if self.mask is not None:
            # Zero dry velocities first: NaN-filled land would otherwise give
            # NaN gradients through -r * u even where the output is masked.
            u = mask_where(u, self.mask.u)
            v = mask_where(v, self.mask.v)

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
        # Zero dry velocities first (land velocity is zero): the 4-point
        # cross-face average at a wet coastal face reads dry neighbours, and
        # NaN-filled land would otherwise make the speed there NaN.
        u = mask_where(u, mu)
        v = mask_where(v, mv)

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
