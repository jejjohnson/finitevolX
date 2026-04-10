"""Coriolis force operator for Arakawa C-grids.

Applies the Coriolis force to staggered velocity fields (u, v) on a C-grid,
producing tendencies du_cor and dv_cor at velocity points.

For the horizontal momentum equations on a C-grid:

    du/dt|cor[j, i+1/2] = +f_on_u[j, i+1/2] * v_on_u[j, i+1/2]
    dv/dt|cor[j+1/2, i] = -f_on_v[j+1/2, i] * u_on_v[j+1/2, i]

where f is the Coriolis parameter at T-points.  The Coriolis parameter is
interpolated to velocity points using simple averaging, and the cross-face
velocity averages are computed with 4-point bilinear interpolation.

References
----------
.. [1] Veros ocean model, ``veros/core/momentum.py``, function
       ``tend_coriolis``.
.. [2] Sadourny (1975) "The dynamics of finite-difference models of the
       shallow-water equations", J. Atmos. Sci., 32, 680–689.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators.interpolation import Interpolation2D


class Coriolis2D(eqx.Module):
    """Coriolis force operator for 2-D Arakawa C-grids.

    Computes the Coriolis tendency for both velocity components:

        du_cor[j, i+1/2] = +f_on_u[j, i+1/2] * v_on_u[j, i+1/2]
        dv_cor[j+1/2, i] = -f_on_v[j+1/2, i] * u_on_v[j+1/2, i]

    The Coriolis parameter f is interpolated from T-points to velocity points
    using simple x/y averaging.  The cross-face velocity averages are computed
    with 4-point bilinear interpolation (same as :class:`Interpolation2D`).

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, Coriolis2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> cor = Coriolis2D(grid=grid)
    >>> u = jnp.zeros((grid.Ny, grid.Nx))
    >>> v = jnp.ones((grid.Ny, grid.Nx))
    >>> f = jnp.ones((grid.Ny, grid.Nx))
    >>> du_cor, dv_cor = cor(u, v, f)
    """

    grid: CartesianGrid2D
    interp: Interpolation2D

    def __init__(self, grid: CartesianGrid2D) -> None:
        self.grid = grid
        self.interp = Interpolation2D(grid=grid)

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        f: Float[Array, "Ny Nx"],
        mask: Mask2D | None = None,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Coriolis tendencies (du_cor, dv_cor).

        du_cor[j, i+1/2] = +f_on_u * v_on_u
        dv_cor[j+1/2, i] = -f_on_v * u_on_v

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points (east faces).
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points (north faces).
        f : Float[Array, "Ny Nx"]
            Coriolis parameter at T-points.
        mask : Mask2D or None
            Optional land/ocean mask.  If provided, ``du_cor`` is multiplied
            by ``mask.u`` and ``dv_cor`` by ``mask.v``.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(du_cor, dv_cor)`` — Coriolis tendencies at U-points and
            V-points respectively, both zero in the ghost ring.
        """
        # Interpolate f from T-points to velocity points
        # f_on_u[j, i+1/2] = 1/2 * (f[j, i] + f[j, i+1])
        f_on_u = self.interp.T_to_U(f)
        # f_on_v[j+1/2, i] = 1/2 * (f[j, i] + f[j+1, i])
        f_on_v = self.interp.T_to_V(f)

        # Cross-face velocity averages (4-point bilinear)
        # v_on_u[j, i+1/2] = 1/4*(v[j+1/2,i] + v[j-1/2,i] + v[j+1/2,i+1] + v[j-1/2,i+1])
        v_on_u = self.interp.V_to_U(v)
        # u_on_v[j+1/2, i] = 1/4*(u[j,i+1/2] + u[j+1,i+1/2] + u[j,i-1/2] + u[j+1,i-1/2])
        u_on_v = self.interp.U_to_V(u)

        # du_cor[j, i+1/2] = +f_on_u * v_on_u
        du_cor = interior(f_on_u[1:-1, 1:-1] * v_on_u[1:-1, 1:-1], u)
        # dv_cor[j+1/2, i] = -f_on_v * u_on_v
        dv_cor = interior(-f_on_v[1:-1, 1:-1] * u_on_v[1:-1, 1:-1], v)

        if mask is not None:
            du_cor = du_cor * mask.u
            dv_cor = dv_cor * mask.v

        return du_cor, dv_cor


class Coriolis3D(eqx.Module):
    """Coriolis force operator for 3-D Arakawa C-grids.

    Applies the same horizontal Coriolis stencil as :class:`Coriolis2D`
    independently at each z-level.  The Coriolis parameter f is 2-D
    (depth-independent) and broadcast over all z-levels.

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid3D, Coriolis3D
    >>> grid = CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
    >>> cor = Coriolis3D(grid=grid)
    >>> u = jnp.zeros((grid.Nz, grid.Ny, grid.Nx))
    >>> v = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> f = jnp.ones((grid.Ny, grid.Nx))
    >>> du_cor, dv_cor = cor(u, v, f)
    """

    grid: CartesianGrid3D
    _cor2d: Coriolis2D

    def __init__(self, grid: CartesianGrid3D) -> None:
        self.grid = grid
        self._cor2d = Coriolis2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        f: Float[Array, "Ny Nx"],
        mask: Mask2D | None = None,
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Coriolis tendencies over all z-levels.

        du_cor[k, j, i+1/2] = +f_on_u[j, i+1/2] * v_on_u[k, j, i+1/2]
        dv_cor[k, j+1/2, i] = -f_on_v[j+1/2, i] * u_on_v[k, j+1/2, i]

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.
        f : Float[Array, "Ny Nx"]
            Coriolis parameter at T-points (depth-independent).
        mask : Mask2D or None
            Optional land/ocean mask.  If provided, ``du_cor`` is multiplied
            by ``mask.u`` and ``dv_cor`` by ``mask.v``.

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(du_cor, dv_cor)`` — Coriolis tendencies at U-points and
            V-points, both zero in the ghost ring.
        """
        # Vmap the 2D Coriolis operator over z-levels.
        # f is 2D (depth-independent) and broadcast to each z-slice.
        du_cor, dv_cor = eqx.filter_vmap(lambda u_k, v_k: self._cor2d(u_k, v_k, f))(
            u, v
        )

        # Zero the z-ghost slices to match the 3D ghost-ring convention.
        du_cor = zero_z_ghosts(du_cor)
        dv_cor = zero_z_ghosts(dv_cor)

        if mask is not None:
            du_cor = du_cor * mask.u
            dv_cor = dv_cor * mask.v

        return du_cor, dv_cor
