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
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.spherical import (
    SphericalGrid2D,
    SphericalGrid3D,
)
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators._utils import _safe_div_cos
from finitevolx._src.operators.spherical_compound import _geostrophic_velocity_sphere
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
    grid : SphericalGrid2D
        The underlying 2-D spherical grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching its
        output stagger:

        * T-output → ``mask.h`` (``diff_lon_U_to_T``, ``diff_lat_V_to_T``,
          ``diff2_lon``, ``laplacian_merid``)
        * U-output → ``mask.u`` (``diff_lon_T_to_U``)
        * V-output → ``mask.v`` (``diff_lat_T_to_V``)
        * X-output → ``mask.xy_corner_strict`` (``diff_lon_V_to_X``,
          ``diff_lat_U_to_X``)
        * (U, V)-output → ``(mask.u, mask.v)`` (``geostrophic_velocity``)

        Per #209 Q2, spherical 2-D operators take a Cartesian ``Mask2D``
        rather than a dedicated ``SphericalMask2D`` — the mask geometry
        is coordinate-agnostic at the post-compute-multiply layer.
    """

    grid: SphericalGrid2D
    mask: Mask2D | None = None

    # ------------------------------------------------------------------
    # Forward differences (centre/face → face/corner)
    # ------------------------------------------------------------------

    def diff_lon_T_to_U(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Zonal derivative T → U on a sphere.

        dh/dx[j, i+½] = (h[j, i+1] − h[j, i]) / (R · cos(lat_U) · dlon)
        """
        raw = diff_x_fwd(h)
        cos_U = self.grid.cos_lat_U[1:-1, 1:-1]
        out = interior(_safe_div_cos(raw, cos_U, self.grid.R * self.grid.dlon), h)
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def diff_lat_T_to_V(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Meridional derivative T → V on a sphere.

        dh/dy[j+½, i] = (h[j+1, i] − h[j, i]) / (R · dlat)
        """
        raw = diff_y_fwd(h)
        out = interior(raw / (self.grid.R * self.grid.dlat), h)
        if self.mask is not None:
            out = out * self.mask.v
        return out

    def diff_lon_V_to_X(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Zonal derivative V → X (corner) on a sphere.

        dv/dx[j+½, i+½] = (v[j+½, i+1] − v[j+½, i]) / (R · cos(lat_X) · dlon)
        """
        raw = diff_x_fwd(v)
        cos_X = self.grid.cos_lat_X[1:-1, 1:-1]
        out = interior(_safe_div_cos(raw, cos_X, self.grid.R * self.grid.dlon), v)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def diff_lat_U_to_X(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Meridional derivative U → X (corner) on a sphere.

        du/dy[j+½, i+½] = (u[j+1, i+½] − u[j, i+½]) / (R · dlat)
        """
        raw = diff_y_fwd(u)
        out = interior(raw / (self.grid.R * self.grid.dlat), u)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    # ------------------------------------------------------------------
    # Backward differences (face/corner → centre/face)
    # ------------------------------------------------------------------

    def diff_lon_U_to_T(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Backward zonal derivative U → T on a sphere.

        du/dx[j, i] = (u[j, i+½] − u[j, i−½]) / (R · cos(lat_T) · dlon)
        """
        raw = diff_x_bwd(u)
        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]
        out = interior(_safe_div_cos(raw, cos_T, self.grid.R * self.grid.dlon), u)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def diff_lat_V_to_T(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Backward meridional derivative V → T on a sphere.

        dv/dy[j, i] = (v[j+½, i] − v[j−½, i]) / (R · dlat)
        """
        raw = diff_y_bwd(v)
        out = interior(raw / (self.grid.R * self.grid.dlat), v)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    # ------------------------------------------------------------------
    # Second-order derivatives
    # ------------------------------------------------------------------

    def diff2_lon(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Second zonal derivative at T-points on a sphere.

        d²h/dx² = 1/(R² cos²φ) · (h[j,i+1] − 2h[j,i] + h[j,i−1]) / dlon²
        """
        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]
        d2h = (diff_x_fwd(h) - diff_x_bwd(h)) / self.grid.dlon**2
        out = interior(_safe_div_cos(d2h, cos_T, self.grid.R**2 * cos_T), h)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def laplacian_merid(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Meridional term of the spherical Laplacian at T-points.

        1/(R² cosφ) · d/dφ(cosφ · dh/dφ)
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
        if self.mask is not None:
            out = out * self.mask.h
        return out

    # ------------------------------------------------------------------
    # Compound diagnostics
    # ------------------------------------------------------------------

    def geostrophic_velocity(
        self,
        h: Float[Array, "Ny Nx"],
        f: Float[Array, "Ny Nx"],
        gravity: float = 9.80665,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Geostrophic velocity from free-surface height on a sphere.

        u_g[j, i+1/2] = -g / (f_on_U[j, i+1/2] * R) * dh_dlat_U[j, i+1/2]
        v_g[j+1/2, i] =  g / (f_on_V[j+1/2, i] * R * cos_V[j+1/2, i])
                         * dh_dlon_V[j+1/2, i]

        with the compact 4-point stencils

        f_on_U[j, i+1/2]    = 1/2 * (f[j, i] + f[j, i+1])
        dh_dlat_U[j, i+1/2] = (h[j+1, i] + h[j+1, i+1]
                              - h[j-1, i] - h[j-1, i+1]) / (4 * dlat)
        f_on_V[j+1/2, i]    = 1/2 * (f[j, i] + f[j+1, i])
        cos_V[j+1/2, i]     = 1/2 * (cos_lat_T[j, i] + cos_lat_T[j+1, i])
        dh_dlon_V[j+1/2, i] = (h[j, i+1] + h[j+1, i+1]
                              - h[j, i-1] - h[j+1, i-1]) / (4 * dlon)

        Class form of :func:`~finitevolx.geostrophic_velocity_sphere`.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Free-surface height at T-points.
        f : Float[Array, "Ny Nx"]
            Coriolis parameter at T-points.
        gravity : float, optional
            Gravitational acceleration.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(u_g, v_g)`` at U- and V-points, zero in the ghost ring and,
            when ``self.mask`` is set, at dry U- and V-faces.

        Notes
        -----
        Under a mask, the face-averaged Coriolis parameter is replaced by
        ``1`` on every dry face *before* the division
        (``f_on_U = 1`` where ``mask.u`` is False, ``f_on_V = 1`` where
        ``mask.v`` is False), so no dry face divides by zero -- whatever
        ``f`` holds on land, and even where a wet neighbour's ``f`` would
        cancel it -- keeping both the forward value and its reverse-mode
        gradient finite.  Wet faces keep their real ``f_on_face``.  The
        output is then masked with ``jnp.where``.  ``h`` is not modified:
        the 4-point stencils of wet coastal faces read ``h`` on the
        neighbouring land cells, so land ``h`` must be finite.
        """
        if self.mask is None:
            return _geostrophic_velocity_sphere(h, f, self.grid, gravity)
        u_g, v_g = _geostrophic_velocity_sphere(
            h, f, self.grid, gravity, wet_u=self.mask.u, wet_v=self.mask.v
        )
        u_g = jnp.where(self.mask.u, u_g, 0.0)
        v_g = jnp.where(self.mask.v, v_g, 0.0)
        return u_g, v_g


class SphericalDifference3D(eqx.Module):
    """Spherical finite-difference operators on a 3-D Arakawa C-grid.

    Applies 2-D spherical differences to each z-level via ``eqx.filter_vmap``.

    Parameters
    ----------
    grid : SphericalGrid3D
        The underlying 3-D spherical grid.
    mask : Mask3D or None, optional
        Optional land/ocean mask.  Pattern A (post-compute) — the inner
        :class:`SphericalDifference2D` is built ``mask=None`` and the
        3-D result is post-multiplied by the mask field matching each
        method's output stagger:

        * T-output → ``mask.h`` (``diff_lon_U_to_T``, ``diff_lat_V_to_T``,
          ``diff2_lon``, ``laplacian_merid``)
        * U-output → ``mask.u`` (``diff_lon_T_to_U``)
        * V-output → ``mask.v`` (``diff_lat_T_to_V``)
        * X-output → ``mask.xy_corner_strict`` (``diff_lon_V_to_X``,
          ``diff_lat_U_to_X``)

        Per #209 Q3, the 3-D class takes the Cartesian ``Mask3D``
        rather than a dedicated ``SphericalMask3D``.
    """

    grid: SphericalGrid3D
    mask: Mask3D | None
    _diff2d: SphericalDifference2D

    def __init__(
        self,
        grid: SphericalGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        # Pattern A: inner 2-D op is mask-free; the 3-D wrapper owns the mask.
        self._diff2d = SphericalDifference2D(grid=grid.horizontal_grid())

    def diff_lon_T_to_U(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Zonal derivative T → U over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._diff2d.diff_lon_T_to_U)(h))
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def diff_lat_T_to_V(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Meridional derivative T → V over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._diff2d.diff_lat_T_to_V)(h))
        if self.mask is not None:
            out = out * self.mask.v
        return out

    def diff_lon_V_to_X(self, v: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Zonal derivative V → X over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._diff2d.diff_lon_V_to_X)(v))
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def diff_lat_U_to_X(self, u: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Meridional derivative U → X over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._diff2d.diff_lat_U_to_X)(u))
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def diff_lon_U_to_T(self, u: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Backward zonal derivative U → T over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._diff2d.diff_lon_U_to_T)(u))
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def diff_lat_V_to_T(self, v: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Backward meridional derivative V → T over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._diff2d.diff_lat_V_to_T)(v))
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def diff2_lon(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Second zonal derivative at T-points over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._diff2d.diff2_lon)(h))
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def laplacian_merid(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Meridional Laplacian term at T-points over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._diff2d.laplacian_merid)(h))
        if self.mask is not None:
            out = out * self.mask.h
        return out
