"""Coriolis force operator for Arakawa C-grids.

Applies the Coriolis force to staggered velocity fields (u, v) on a C-grid,
producing tendencies du_cor and dv_cor at velocity points.

For the horizontal momentum equations on a C-grid the Coriolis force is
the vortex force with the planetary vorticity f, in the energy-conserving
form of Sadourny (1975): f is taken to the X-points (corners), multiplied
there by the cross velocity averaged to the corner, and the product is
averaged back to the faces:

    du/dt|cor[j, i+1/2] = +avg_y( f_X * avg_x(v) )[j, i+1/2]
    dv/dt|cor[j+1/2, i] = -avg_x( f_X * avg_y(u) )[j+1/2, i]

with ``f_X[j+1/2, i+1/2]`` the 4-point average of f (given at T-points).
Because both tendencies are built from the same corner products,
``sum(u * du) + sum(v * dv) = 0`` exactly for any f (beta-plane included),
on a periodic domain or a closed basin.  For uniform f it reduces to the
familiar ``f * V_to_U(v)`` / ``-f * U_to_V(u)``.

References
----------
.. [1] Veros ocean model, ``veros/core/momentum.py``, function
       ``tend_coriolis``.
.. [2] Sadourny (1975) "The dynamics of finite-difference models of the
       shallow-water equations", J. Atmos. Sci., 32, 680–689.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts


class Coriolis2D(eqx.Module):
    """Coriolis force operator for 2-D Arakawa C-grids.

    Computes the Coriolis tendency for both velocity components in the
    energy-conserving corner-product form (Sadourny 1975):

        du_cor[j, i+1/2] = +1/2 (P[j+1/2, i+1/2] + P[j-1/2, i+1/2])
        dv_cor[j+1/2, i] = -1/2 (Q[j+1/2, i+1/2] + Q[j+1/2, i-1/2])

        P = f_X * avg_x(v),   Q = f_X * avg_y(u)   at X-points,

    with ``f_X`` the 4-point average of the T-point f.  The kinetic-energy
    tendency ``sum(u * du_cor) + sum(v * dv_cor)`` vanishes to round-off for
    any f, including a beta-plane (the f_on_u * V_to_U(v) form used before
    #289 conserves energy only for uniform f).  For uniform f the two forms
    coincide.

    The corner products are formed on the full arrays (X rows / columns
    ``0 .. N-2``), so the first interior row / column reads the south /
    west ghost corners built from the caller's ghost cells.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, ``u`` / ``v`` are zeroed
        on dry faces with ``jnp.where`` before the stencil (so NaN-filled
        land cannot leak) and the output on dry U / V faces.  Energy is
        still conserved, and for uniform f the result equals the face form
        on the masked velocities.  ``None`` (default) leaves the output
        untouched.

    Notes
    -----
    ``f`` must be filled on the whole array, ghost ring included (e.g. an
    analytic ``f0 + beta * y`` evaluated at every T-point): the corner
    average ``f_X`` on the first / last interior rows reads the ghost rows
    of ``f``.

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
    mask: Mask2D | None = None

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        f: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Coriolis tendencies (du_cor, dv_cor).

        du_cor[j, i+1/2] = +avg_y(f_X * avg_x(v))
        dv_cor[j+1/2, i] = -avg_x(f_X * avg_y(u))

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points (east faces).
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points (north faces).
        f : Float[Array, "Ny Nx"]
            Coriolis parameter at T-points, ghost ring filled.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(du_cor, dv_cor)`` — Coriolis tendencies at U-points and
            V-points respectively, both zero in the ghost ring.  When
            ``self.mask`` is set, dry U / V faces are zero.
        """
        if self.mask is not None:
            # zero dry faces before any stencil reads them (NaN-filled land)
            u = jnp.where(self.mask.u, u, 0.0)
            v = jnp.where(self.mask.v, v, 0.0)
        # f_X[j+1/2, i+1/2] = 1/4 (f[j, i] + f[j+1, i] + f[j, i+1] + f[j+1, i+1])
        #   X rows / cols 0 .. N-2 (the south / west ghost corners included)
        f_x = 0.25 * (f[:-1, :-1] + f[1:, :-1] + f[:-1, 1:] + f[1:, 1:])  # [Ny-1, Nx-1]
        # v_X[j+1/2, i+1/2] = 1/2 (v[j+1/2, i] + v[j+1/2, i+1])
        v_x = 0.5 * (v[:-1, :-1] + v[:-1, 1:])  # [Ny-1, Nx-1]
        # u_X[j+1/2, i+1/2] = 1/2 (u[j, i+1/2] + u[j+1, i+1/2])
        u_x = 0.5 * (u[:-1, :-1] + u[1:, :-1])  # [Ny-1, Nx-1]
        # corner products P = f_X v_X, Q = f_X u_X              [Ny-1, Nx-1]
        p = f_x * v_x
        q = f_x * u_x
        # du_cor[j, i+1/2] = +1/2 (P[j+1/2, i+1/2] + P[j-1/2, i+1/2])  [Ny-2, Nx-2]
        du_cor = interior(0.5 * (p[1:, 1:] + p[:-1, 1:]), u)
        # dv_cor[j+1/2, i] = -1/2 (Q[j+1/2, i+1/2] + Q[j+1/2, i-1/2])  [Ny-2, Nx-2]
        dv_cor = interior(-0.5 * (q[1:, 1:] + q[1:, :-1]), v)

        if self.mask is not None:
            du_cor = jnp.where(self.mask.u, du_cor, 0.0)
            dv_cor = jnp.where(self.mask.v, dv_cor, 0.0)

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
    mask : Mask3D or None, optional
        Optional land/ocean mask.  Pattern A (post-compute) — the
        inner :class:`Coriolis2D` is constructed ``mask=None`` and
        the 3-D result is post-multiplied by ``mask.u`` / ``mask.v``
        after the vmap.

        Takes a :class:`Mask3D` (not ``Mask2D``): even though ``f``
        is 2-D and depth-independent, the velocity mask is natively
        3-D on the Arakawa C-grid, and using ``Mask3D`` keeps the
        3-D operator suite type-uniform per #209.  Issue #209's Q4
        locked this in for Advection3D and this commit extends the
        same decision to Coriolis3D.

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
    mask: Mask3D | None
    _cor2d: Coriolis2D

    def __init__(
        self,
        grid: CartesianGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        # Pattern A: inner 2-D op is mask-free; the 3-D wrapper owns the mask.
        self._cor2d = Coriolis2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        f: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Coriolis tendencies over all z-levels.

        du_cor[k, j, i+1/2] = +avg_y(f_X * avg_x(v[k]))
        dv_cor[k, j+1/2, i] = -avg_x(f_X * avg_y(u[k]))

        (the energy-conserving form of :class:`Coriolis2D` at each level)

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.
        f : Float[Array, "Ny Nx"]
            Coriolis parameter at T-points (depth-independent).

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(du_cor, dv_cor)`` — Coriolis tendencies at U-points and
            V-points, both zero in the ghost ring.  When ``self.mask``
            is set, ``du_cor`` is zeroed at dry U-faces via
            ``* mask.u`` and ``dv_cor`` at dry V-faces via ``* mask.v``.
        """
        if self.mask is not None:
            # zero dry faces before the stencil, as Coriolis2D does
            u = jnp.where(self.mask.u, u, 0.0)
            v = jnp.where(self.mask.v, v, 0.0)
        # Vmap the 2D Coriolis operator over z-levels.
        # f is 2D (depth-independent) and broadcast to each z-slice.
        du_cor, dv_cor = eqx.filter_vmap(lambda u_k, v_k: self._cor2d(u_k, v_k, f))(
            u, v
        )

        # Zero the z-ghost slices to match the 3D ghost-ring convention.
        du_cor = zero_z_ghosts(du_cor)
        dv_cor = zero_z_ghosts(dv_cor)

        if self.mask is not None:
            du_cor = du_cor * self.mask.u
            dv_cor = dv_cor * self.mask.v

        return du_cor, dv_cor
