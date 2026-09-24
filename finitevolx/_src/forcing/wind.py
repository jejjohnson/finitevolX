"""Surface wind-stress forcing operators for Arakawa C-grids.

The wind stress ``(tau_x, tau_y)`` is supplied at T-points (cell centres) and
interpolated to the velocity faces, where it accelerates the top layer:

    du_wind[j, i+1/2] = tau_x_on_u[j, i+1/2] / (rho0 * dz_top_on_u[j, i+1/2])
    dv_wind[j+1/2, i] = tau_y_on_v[j+1/2, i] / (rho0 * dz_top_on_v[j+1/2, i])

The core math is :func:`~finitevolx.wind_stress_tendency`.  On a 3-D grid
(``k = 0`` bottom ghost, ``k = Nz - 1`` top ghost) the stress acts only on the
top interior z-level ``k = Nz - 2``.

References
----------
.. [1] Veros ocean model, ``veros/core/momentum.py``, function
       ``tend_windstress``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.forcing._base import AbstractForcing
from finitevolx._src.forcing._utils import (
    mask_where,
    on_face_interior,
    safe_denominator,
)
from finitevolx._src.forcing.functional import wind_stress_tendency
from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators.interpolation import Interpolation2D


class WindStress2D(AbstractForcing):
    """Surface wind-stress forcing on a 2-D Arakawa C-grid.

    Computes the momentum tendency of the top layer:

        du_wind[j, i+1/2] = tau_x_on_u[j, i+1/2] / (rho0 * dz_top_on_u[j, i+1/2])
        dv_wind[j+1/2, i] = tau_y_on_v[j+1/2, i] / (rho0 * dz_top_on_v[j+1/2, i])

    ``tau_x``, ``tau_y`` (and ``dz_top`` when it is a field) are given at
    T-points and interpolated to U/V-points with simple x/y averaging.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, ``du_wind`` is zeroed at
        dry U-faces and ``dv_wind`` at dry V-faces (via ``jnp.where``, so a
        zero dry-cell ``dz_top`` or NaN-filled land never leaks NaN).
    rho0 : float, optional
        Reference density [kg/m^3].  Default ``1025.0``.

    Examples
    --------
    Idealised double-gyre wind ``tau_x = -tau0 * cos(2 pi y / Ly)``:

    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid2D, WindStress2D
    >>> grid = CartesianGrid2D.from_interior(8, 8, 1.0e6, 1.0e6)
    >>> y = (jnp.arange(grid.Ny) - 0.5) * grid.dy
    >>> tau_x = -0.1 * jnp.cos(2 * jnp.pi * y / grid.Ly)[:, None] * jnp.ones(grid.Nx)
    >>> tau_y = jnp.zeros((grid.Ny, grid.Nx))
    >>> wind = WindStress2D(grid=grid)
    >>> du_wind, dv_wind = wind(tau_x, tau_y, dz_top=50.0)
    """

    grid: CartesianGrid2D
    mask: Mask2D | None
    rho0: float
    interp: Interpolation2D

    def __init__(
        self,
        grid: CartesianGrid2D,
        mask: Mask2D | None = None,
        rho0: float = 1025.0,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.rho0 = rho0
        self.interp = Interpolation2D(grid=grid)

    def __call__(
        self,
        tau_x: Float[Array, "Ny Nx"],
        tau_y: Float[Array, "Ny Nx"],
        dz_top: float | Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Wind-stress tendencies (du_wind, dv_wind).

        Parameters
        ----------
        tau_x : Float[Array, "Ny Nx"]
            Zonal wind stress at T-points [N/m^2].
        tau_y : Float[Array, "Ny Nx"]
            Meridional wind stress at T-points [N/m^2].
        dz_top : float or Float[Array, "Ny Nx"]
            Top-layer thickness [m] — a scalar, or a T-point field (ghost
            ring filled) interpolated to U/V-points.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(du_wind, dv_wind)`` at U- and V-points, zero in the ghost
            ring.  When ``self.mask`` is set, dry faces are zeroed.
        """
        mu = None if self.mask is None else self.mask.u
        mv = None if self.mask is None else self.mask.v
        return self._tendency(tau_x, tau_y, dz_top, mu, mv)

    def _tendency(
        self,
        tau_x: Float[Array, "Ny Nx"],
        tau_y: Float[Array, "Ny Nx"],
        dz_top: float | Float[Array, "Ny Nx"],
        mu: Array | None,
        mv: Array | None,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Core stencil with explicit ``[Ny, Nx]`` face masks (or ``None``).

        Shared with :class:`WindStress3D`, which passes one z-level's masks.
        """
        # tau_x_on_u[j, i+1/2] = 1/2 * (tau_x[j, i] + tau_x[j, i+1])
        tau_x_on_u = self.interp.T_to_U(tau_x)
        # tau_y_on_v[j+1/2, i] = 1/2 * (tau_y[j, i] + tau_y[j+1, i])
        tau_y_on_v = self.interp.T_to_V(tau_y)
        # Swap in 1 at dry faces so a zero dry-cell thickness cannot give 0/0.
        dz_on_u = safe_denominator(on_face_interior(dz_top, self.interp.T_to_U), mu)
        dz_on_v = safe_denominator(on_face_interior(dz_top, self.interp.T_to_V), mv)

        # du_wind[j, i+1/2] = tau_x_on_u / (rho0 * dz_top_on_u)
        du_wind = interior(
            wind_stress_tendency(tau_x_on_u[1:-1, 1:-1], self.rho0, dz_on_u), tau_x
        )
        # dv_wind[j+1/2, i] = tau_y_on_v / (rho0 * dz_top_on_v)
        dv_wind = interior(
            wind_stress_tendency(tau_y_on_v[1:-1, 1:-1], self.rho0, dz_on_v), tau_y
        )

        return mask_where(du_wind, mu), mask_where(dv_wind, mv)


class WindStress3D(AbstractForcing):
    """Surface wind-stress forcing on a 3-D Arakawa C-grid.

    Applies the :class:`WindStress2D` stencil to the **top interior
    z-level only** and returns zero at every other level.  The 3-D grid
    stacks levels bottom-up (``k = 0`` bottom ghost, ``k = Nz - 1`` top
    ghost), so the top interior level is ``k_top = Nz - 2``:

        du_wind[k_top, j, i+1/2] = tau_x_on_u[j, i+1/2] / (rho0 * dz_top_on_u[j, i+1/2])
        dv_wind[k_top, j+1/2, i] = tau_y_on_v[j+1/2, i] / (rho0 * dz_top_on_v[j+1/2, i])
        du_wind[k != k_top] = dv_wind[k != k_top] = 0

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional land/ocean mask.  The ``k_top`` slices of ``mask.u`` /
        ``mask.v`` zero dry faces (via ``jnp.where``) and guard the
        ``dz_top`` division there, so zero dry-cell thickness or NaN-filled
        land never leaks NaN.
    rho0 : float, optional
        Reference density [kg/m^3].  Default ``1025.0``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import CartesianGrid3D, WindStress3D
    >>> grid = CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
    >>> wind = WindStress3D(grid=grid)
    >>> tau = 0.1 * jnp.ones((grid.Ny, grid.Nx))
    >>> du_wind, dv_wind = wind(tau, tau, dz_top=grid.dz)
    >>> du_wind.shape
    (6, 8, 8)
    """

    grid: CartesianGrid3D
    mask: Mask3D | None
    _wind2d: WindStress2D

    def __init__(
        self,
        grid: CartesianGrid3D,
        mask: Mask3D | None = None,
        rho0: float = 1025.0,
    ) -> None:
        self.grid = grid
        self.mask = mask
        # The inner 2-D op is mask-free; the 3-D wrapper passes per-level masks.
        self._wind2d = WindStress2D(grid=grid.horizontal_grid(), rho0=rho0)

    def __call__(
        self,
        tau_x: Float[Array, "Ny Nx"],
        tau_y: Float[Array, "Ny Nx"],
        dz_top: float | Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Wind-stress tendencies, non-zero only at the top interior level.

        Parameters
        ----------
        tau_x : Float[Array, "Ny Nx"]
            Zonal wind stress at T-points [N/m^2].
        tau_y : Float[Array, "Ny Nx"]
            Meridional wind stress at T-points [N/m^2].
        dz_top : float or Float[Array, "Ny Nx"]
            Top-layer thickness [m] — a scalar, or a T-point field (ghost
            ring filled) interpolated to U/V-points.

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(du_wind, dv_wind)`` at U- and V-points, zero outside
            ``k = Nz - 2`` and in the ghost ring.  When ``self.mask`` is set,
            dry faces are zeroed.
        """
        k_top = self.grid.Nz - 2
        mu = None if self.mask is None else self.mask.u[k_top]
        mv = None if self.mask is None else self.mask.v[k_top]
        du_2d, dv_2d = self._wind2d._tendency(tau_x, tau_y, dz_top, mu, mv)

        # Inject at the top interior z-level k_top = Nz - 2.
        shape = (self.grid.Nz, self.grid.Ny, self.grid.Nx)
        du_wind = jnp.zeros(shape, dtype=du_2d.dtype).at[k_top].set(du_2d)
        dv_wind = jnp.zeros(shape, dtype=dv_2d.dtype).at[k_top].set(dv_2d)
        return du_wind, dv_wind
