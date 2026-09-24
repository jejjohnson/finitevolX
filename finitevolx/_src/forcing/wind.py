"""Surface wind-stress forcing operators for Arakawa C-grids.

The wind stress ``(tau_x, tau_y)`` is supplied at T-points (cell centres) and
interpolated to the velocity faces, where it accelerates the top layer:

    du_wind[j, i+1/2] = tau_x_on_u[j, i+1/2] / (rho0 * dz_top_on_u[j, i+1/2])
    dv_wind[j+1/2, i] = tau_y_on_v[j+1/2, i] / (rho0 * dz_top_on_v[j+1/2, i])

The core math is :func:`~finitevolx.wind_stress_tendency`.

References
----------
.. [1] Veros ocean model, ``veros/core/momentum.py``, function
       ``tend_windstress``.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from finitevolx._src.forcing._base import AbstractForcing
from finitevolx._src.forcing._utils import on_face_interior
from finitevolx._src.forcing.functional import wind_stress_tendency
from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.mask import Mask2D
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
        Optional land/ocean mask.  When provided, ``du_wind`` is
        post-multiplied by ``mask.u`` and ``dv_wind`` by ``mask.v``.
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
        # tau_x_on_u[j, i+1/2] = 1/2 * (tau_x[j, i] + tau_x[j, i+1])
        tau_x_on_u = self.interp.T_to_U(tau_x)
        # tau_y_on_v[j+1/2, i] = 1/2 * (tau_y[j, i] + tau_y[j+1, i])
        tau_y_on_v = self.interp.T_to_V(tau_y)
        dz_on_u = on_face_interior(dz_top, self.interp.T_to_U)
        dz_on_v = on_face_interior(dz_top, self.interp.T_to_V)

        # du_wind[j, i+1/2] = tau_x_on_u / (rho0 * dz_top_on_u)
        du_wind = interior(
            wind_stress_tendency(tau_x_on_u[1:-1, 1:-1], self.rho0, dz_on_u), tau_x
        )
        # dv_wind[j+1/2, i] = tau_y_on_v / (rho0 * dz_top_on_v)
        dv_wind = interior(
            wind_stress_tendency(tau_y_on_v[1:-1, 1:-1], self.rho0, dz_on_v), tau_y
        )

        if self.mask is not None:
            du_wind = du_wind * self.mask.u
            dv_wind = dv_wind * self.mask.v

        return du_wind, dv_wind
