"""Spherical finite-difference operators on Arakawa C-grids.

These operators apply spherical metric scaling (Layer 2) to the raw
stencils from :mod:`finitevolx._src.operators.stencils` (Layer 1).

Half-index notation
-------------------
Storage index [j, i] encodes:
  T[j, i]  at cell centre    (lon_i,        lat_j       )
  U[j, i]  at east face      (lon_{i+1/2},  lat_j       )
  V[j, i]  at north face     (lon_i,        lat_{j+1/2} )
  X[j, i]  at NE corner      (lon_{i+1/2},  lat_{j+1/2} )

Pole handling
-------------
Operators that divide by ``cos(lat)`` return NaN where
``|cos(lat)| < 1e-12``.
"""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, Float

from finitevolx._src.grid.spherical_grid import (
    SphericalArakawaCGrid2D,
    SphericalArakawaCGrid3D,
)
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators._utils import _safe_div_cos
from finitevolx._src.operators.stencils import (
    diff_x_bwd,
    diff_x_fwd,
    diff_y_bwd,
    diff_y_fwd,
)


class SphericalDifference2D(eqx.Module):
    """Spherical finite-difference operators on a 2-D Arakawa C-grid.

    Parameters
    ----------
    grid : SphericalArakawaCGrid2D
        The underlying 2-D spherical grid.
    """

    grid: SphericalArakawaCGrid2D

    # ------------------------------------------------------------------
    # Forward differences (centre/face → face/corner)
    # ------------------------------------------------------------------

    def diff_lon_T_to_U(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Zonal derivative T → U on a sphere.

        dh/dx[j, i+½] = (h[j, i+1] − h[j, i]) / (R · cos(lat_U) · dlon)

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Zonal derivative at U-points.
        """
        raw = diff_x_fwd(h)
        cos_U = self.grid.cos_lat_U[1:-1, 1:-1]
        out = interior(_safe_div_cos(raw, cos_U, self.grid.R * self.grid.dlon), h)
        return out

    def diff_lat_T_to_V(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Meridional derivative T → V on a sphere.

        dh/dy[j+½, i] = (h[j+1, i] − h[j, i]) / (R · dlat)

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Meridional derivative at V-points.
        """
        raw = diff_y_fwd(h)
        out = interior(raw / (self.grid.R * self.grid.dlat), h)
        return out

    def diff_lon_V_to_X(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Zonal derivative V → X (corner) on a sphere.

        dv/dx[j+½, i+½] = (v[j+½, i+1] − v[j+½, i]) / (R · cos(lat_X) · dlon)

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Zonal derivative at X-points.
        """
        raw = diff_x_fwd(v)
        cos_X = self.grid.cos_lat_X[1:-1, 1:-1]
        out = interior(_safe_div_cos(raw, cos_X, self.grid.R * self.grid.dlon), v)
        return out

    def diff_lat_U_to_X(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Meridional derivative U → X (corner) on a sphere.

        du/dy[j+½, i+½] = (u[j+1, i+½] − u[j, i+½]) / (R · dlat)

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity at U-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Meridional derivative at X-points.
        """
        raw = diff_y_fwd(u)
        out = interior(raw / (self.grid.R * self.grid.dlat), u)
        return out

    # ------------------------------------------------------------------
    # Backward differences (face/corner → centre/face)
    # ------------------------------------------------------------------

    def diff_lon_U_to_T(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Backward zonal derivative U → T on a sphere.

        du/dx[j, i] = (u[j, i+½] − u[j, i−½]) / (R · cos(lat_T) · dlon)

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity at U-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Zonal derivative at T-points.
        """
        raw = diff_x_bwd(u)
        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]
        out = interior(_safe_div_cos(raw, cos_T, self.grid.R * self.grid.dlon), u)
        return out

    def diff_lat_V_to_T(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Backward meridional derivative V → T on a sphere.

        dv/dy[j, i] = (v[j+½, i] − v[j−½, i]) / (R · dlat)

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Meridional derivative at T-points.
        """
        raw = diff_y_bwd(v)
        out = interior(raw / (self.grid.R * self.grid.dlat), v)
        return out

    # ------------------------------------------------------------------
    # Second-order derivatives
    # ------------------------------------------------------------------

    def diff2_lon(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Second zonal derivative at T-points on a sphere.

        d²h/dx² = 1/(R² cos²φ) · (h[j,i+1] − 2h[j,i] + h[j,i−1]) / dlon²

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Second zonal derivative at T-points.
        """
        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]
        d2h = (diff_x_fwd(h) - diff_x_bwd(h)) / self.grid.dlon**2
        out = interior(
            _safe_div_cos(d2h, cos_T, self.grid.R**2 * cos_T), h
        )
        return out

    def laplacian_merid(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Meridional term of the spherical Laplacian at T-points.

        1/(R² cosφ) · d/dφ(cosφ · dh/dφ)

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Meridional Laplacian term at T-points.
        """
        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]
        dlat = self.grid.dlat
        # dh/dlat at half-indices
        dh_N = (h[2:, 1:-1] - h[1:-1, 1:-1]) / dlat
        dh_S = (h[1:-1, 1:-1] - h[:-2, 1:-1]) / dlat
        # cos(lat) at V-point latitudes (half-cell north/south)
        cos_N = 0.5 * (cos_T + self.grid.cos_lat_T[2:, 1:-1])
        cos_S = 0.5 * (cos_T + self.grid.cos_lat_T[:-2, 1:-1])
        d_cos_dh = (cos_N * dh_N - cos_S * dh_S) / dlat
        out = interior(_safe_div_cos(d_cos_dh, cos_T, self.grid.R**2), h)
        return out


class SphericalDifference3D(eqx.Module):
    """Spherical finite-difference operators on a 3-D Arakawa C-grid.

    Applies 2-D spherical differences to each z-level via ``jax.vmap``.

    Parameters
    ----------
    grid : SphericalArakawaCGrid3D
        The underlying 3-D spherical grid.
    """

    grid: SphericalArakawaCGrid3D
    _diff2d: SphericalDifference2D

    def __init__(self, grid: SphericalArakawaCGrid3D):
        self.grid = grid
        self._diff2d = SphericalDifference2D(grid=grid.horizontal_grid())

    def diff_lon_T_to_U(
        self, h: Float[Array, "Nz Ny Nx"]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Zonal derivative T → U over all z-levels."""
        return jax.vmap(self._diff2d.diff_lon_T_to_U)(h)

    def diff_lat_T_to_V(
        self, h: Float[Array, "Nz Ny Nx"]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Meridional derivative T → V over all z-levels."""
        return jax.vmap(self._diff2d.diff_lat_T_to_V)(h)

    def diff_lon_V_to_X(
        self, v: Float[Array, "Nz Ny Nx"]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Zonal derivative V → X over all z-levels."""
        return jax.vmap(self._diff2d.diff_lon_V_to_X)(v)

    def diff_lat_U_to_X(
        self, u: Float[Array, "Nz Ny Nx"]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Meridional derivative U → X over all z-levels."""
        return jax.vmap(self._diff2d.diff_lat_U_to_X)(u)

    def diff_lon_U_to_T(
        self, u: Float[Array, "Nz Ny Nx"]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Backward zonal derivative U → T over all z-levels."""
        return jax.vmap(self._diff2d.diff_lon_U_to_T)(u)

    def diff_lat_V_to_T(
        self, v: Float[Array, "Nz Ny Nx"]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Backward meridional derivative V → T over all z-levels."""
        return jax.vmap(self._diff2d.diff_lat_V_to_T)(v)

    def diff2_lon(
        self, h: Float[Array, "Nz Ny Nx"]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Second zonal derivative at T-points over all z-levels."""
        return jax.vmap(self._diff2d.diff2_lon)(h)

    def laplacian_merid(
        self, h: Float[Array, "Nz Ny Nx"]
    ) -> Float[Array, "Nz Ny Nx"]:
        """Meridional Laplacian term at T-points over all z-levels."""
        return jax.vmap(self._diff2d.laplacian_merid)(h)
