"""Compound spherical operators on Arakawa C-grids.

Divergence, vorticity (curl), Laplacian, geostrophic velocity, and
potential vorticity on the sphere.  These compose the primitive
spherical differences from :class:`SphericalDifference2D` with the
appropriate metric terms.

Coriolis is coordinate-independent (purely algebraic). Use
:class:`Coriolis2D` directly with a spherical grid.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.grid.spherical_grid import (
    SphericalArakawaCGrid2D,
    SphericalArakawaCGrid3D,
)
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators._utils import _safe_div_cos
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.operators.stencils import (
    diff_x_bwd,
    diff_x_fwd,
)


class SphericalDivergence2D(eqx.Module):
    """Horizontal divergence on a 2-D spherical Arakawa C-grid.

    div = 1/(R*cos(phi)) * [du/dlon + d(v*cos(phi))/dphi]   at T-points

    Coriolis is coordinate-independent — use ``Coriolis2D`` directly
    with a spherical grid.

    Parameters
    ----------
    grid : SphericalArakawaCGrid2D
        The underlying 2-D spherical grid.
    """

    grid: SphericalArakawaCGrid2D

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """Horizontal divergence at T-points.

        div[j, i] = 1/(R*cos(lat_T)) * [du/dlon + d(v*cos(lat_V))/dphi]

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Zonal velocity at U-points.
        v : Float[Array, "Ny Nx"]
            Meridional velocity at V-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the T-point output is
            multiplied by ``mask.h``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Divergence at T-points.
        """
        dlon = self.grid.dlon
        dlat = self.grid.dlat
        R = self.grid.R

        # du/dlon at T: backward diff U -> T
        du_dlon = diff_x_bwd(u) / dlon

        # d(v*cos(phi))/dphi at T: backward diff V -> T
        v_cos_N = v[1:-1, 1:-1] * self.grid.cos_lat_V[1:-1, 1:-1]
        v_cos_S = v[:-2, 1:-1] * self.grid.cos_lat_V[:-2, 1:-1]
        dvc_dlat = (v_cos_N - v_cos_S) / dlat

        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]
        out = interior(_safe_div_cos(du_dlon + dvc_dlat, cos_T, R), u)
        if mask is not None:
            out = out * mask.h
        return out


class SphericalVorticity2D(eqx.Module):
    """Vorticity and PV-flux operators on a 2-D spherical Arakawa C-grid.

    zeta = 1/(R*cos(phi)) * [dv/dlon - d(u*cos(phi))/dphi]   at X-points

    The PV flux methods (energy-conserving, enstrophy-conserving,
    Arakawa-Lamb) are coordinate-independent: they involve only
    interpolation and multiplication.  They are delegated to an
    internal ``Interpolation2D``.

    Coriolis is coordinate-independent — use ``Coriolis2D`` directly
    with a spherical grid.

    Parameters
    ----------
    grid : SphericalArakawaCGrid2D
        The underlying 2-D spherical grid.
    """

    grid: SphericalArakawaCGrid2D
    _interp: Interpolation2D

    def __init__(self, grid: SphericalArakawaCGrid2D) -> None:
        self.grid = grid
        self._interp = Interpolation2D(grid=grid)

    def relative_vorticity(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """Relative vorticity (curl) at X-points on a sphere.

        zeta[j+1/2, i+1/2] = 1/(R*cos(lat_X)) * [dv/dlon - d(u*cos(phi))/dphi]

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Zonal velocity at U-points.
        v : Float[Array, "Ny Nx"]
            Meridional velocity at V-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the X-point output
            is multiplied by ``mask.psi``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Relative vorticity at X-points.
        """
        dlon = self.grid.dlon
        dlat = self.grid.dlat
        R = self.grid.R

        # dv/dlon at X: forward diff V -> X
        dv_dlon = diff_x_fwd(v) / dlon

        # d(u*cos(phi))/dphi at X: forward diff U -> X
        u_cos_N = u[2:, 1:-1] * self.grid.cos_lat_U[2:, 1:-1]
        u_cos_S = u[1:-1, 1:-1] * self.grid.cos_lat_U[1:-1, 1:-1]
        duc_dlat = (u_cos_N - u_cos_S) / dlat

        cos_X = self.grid.cos_lat_X[1:-1, 1:-1]
        out = interior(_safe_div_cos(dv_dlon - duc_dlat, cos_X, R), u)
        if mask is not None:
            out = out * mask.psi
        return out

    def potential_vorticity(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        h: Float[Array, "Ny Nx"],
        f: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """Potential vorticity at X-points on a sphere.

        q = (zeta + f) / h

        ``mask`` is an optional :class:`ArakawaCGridMask`; if provided,
        the X-point output is multiplied by ``mask.psi``.
        """
        zeta = self.relative_vorticity(u, v)
        f_on_X = self._interp.T_to_X(f)
        h_on_X = self._interp.T_to_X(h)
        num = zeta[1:-1, 1:-1] + f_on_X[1:-1, 1:-1]
        den = h_on_X[1:-1, 1:-1]
        out = interior(jnp.where(den == 0, jnp.nan, num / den), u)
        if mask is not None:
            out = out * mask.psi
        return out

    def pv_flux_energy_conserving(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Energy-conserving PV flux (coordinate-independent).

        ``mask`` is an optional :class:`ArakawaCGridMask`; if provided,
        ``qu`` is multiplied by ``mask.u`` and ``qv`` by ``mask.v``.
        """
        q_on_u = self._interp.X_to_U(q)
        q_on_v = self._interp.X_to_V(q)
        qu = interior(q_on_u[1:-1, 1:-1] * u[1:-1, 1:-1], u)
        qv = interior(q_on_v[1:-1, 1:-1] * v[1:-1, 1:-1], v)
        if mask is not None:
            qu = qu * mask.u
            qv = qv * mask.v
        return qu, qv

    def pv_flux_enstrophy_conserving(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Enstrophy-conserving PV flux (coordinate-independent).

        ``mask`` is an optional :class:`ArakawaCGridMask`; if provided,
        ``qu`` is multiplied by ``mask.u`` and ``qv`` by ``mask.v``.
        """
        u_on_q = self._interp.U_to_X(u)
        v_on_q = self._interp.V_to_X(v)
        qu_at_q = interior(q[1:-1, 1:-1] * u_on_q[1:-1, 1:-1], q)
        qv_at_q = interior(q[1:-1, 1:-1] * v_on_q[1:-1, 1:-1], q)
        qu = self._interp.X_to_U(qu_at_q)
        qv = self._interp.X_to_V(qv_at_q)
        if mask is not None:
            qu = qu * mask.u
            qv = qv * mask.v
        return qu, qv

    def pv_flux_arakawa_lamb(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        alpha: float = 1.0 / 3.0,
        mask: ArakawaCGridMask | None = None,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Arakawa-Lamb PV flux: weighted blend of energy and enstrophy.

        ``mask`` is an optional :class:`ArakawaCGridMask`; if provided,
        ``qu`` is multiplied by ``mask.u`` and ``qv`` by ``mask.v``.
        """
        qu_e, qv_e = self.pv_flux_energy_conserving(q, u, v)
        qu_s, qv_s = self.pv_flux_enstrophy_conserving(q, u, v)
        qu = alpha * qu_e + (1.0 - alpha) * qu_s
        qv = alpha * qv_e + (1.0 - alpha) * qv_s
        if mask is not None:
            qu = qu * mask.u
            qv = qv * mask.v
        return qu, qv


class SphericalLaplacian2D(eqx.Module):
    """Spherical Laplacian on a 2-D Arakawa C-grid.

    nabla^2 h = 1/(R^2 cos^2 phi) * d^2h/dlon^2
              + 1/(R^2 cos phi) * d/dphi(cos phi * dh/dphi)

    Parameters
    ----------
    grid : SphericalArakawaCGrid2D
        The underlying 2-D spherical grid.
    """

    grid: SphericalArakawaCGrid2D

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """Laplacian at T-points on a sphere.

        ``mask`` is an optional :class:`ArakawaCGridMask`; if provided,
        the T-point output is multiplied by ``mask.h``.
        """
        R = self.grid.R
        dlon = self.grid.dlon
        dlat = self.grid.dlat
        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]

        # d^2h/dlon^2
        d2h_dlon2 = (diff_x_fwd(h) - diff_x_bwd(h)) / dlon**2

        # d/dphi(cos(phi) * dh/dphi)
        dh_N = (h[2:, 1:-1] - h[1:-1, 1:-1]) / dlat
        dh_S = (h[1:-1, 1:-1] - h[:-2, 1:-1]) / dlat
        cos_V_N = 0.5 * (cos_T + self.grid.cos_lat_T[2:, 1:-1])
        cos_V_S = 0.5 * (cos_T + self.grid.cos_lat_T[:-2, 1:-1])
        d_coslat_dh = (cos_V_N * dh_N - cos_V_S * dh_S) / dlat

        lon_term = _safe_div_cos(d2h_dlon2, cos_T, R**2 * cos_T)
        lat_term = _safe_div_cos(d_coslat_dh, cos_T, R**2)
        out = interior(lon_term + lat_term, h)
        if mask is not None:
            out = out * mask.h
        return out


def geostrophic_velocity_sphere(
    h: Float[Array, "Ny Nx"],
    f: Float[Array, "Ny Nx"],
    grid: SphericalArakawaCGrid2D,
    gravity: float = 9.80665,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Geostrophic velocity from free-surface height on a sphere.

    u_g = -g/(f*R) * dh/dphi       at U-points
    v_g =  g/(f*R*cos(phi)) * dh/dlon  at V-points

    This is a functional helper (no class wrapper); per the design
    decision functional ops do not take masks. Apply ``* mask.u`` and
    ``* mask.v`` after the call if you need masking.

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Free-surface height at T-points.
    f : Float[Array, "Ny Nx"]
        Coriolis parameter at T-points.
    grid : SphericalArakawaCGrid2D
        The spherical grid.
    gravity : float
        Gravitational acceleration.

    Returns
    -------
    tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
        (u_g at U-points, v_g at V-points).
    """
    R = grid.R
    dlon = grid.dlon
    dlat = grid.dlat

    # u_g at U-points: compact 4-point stencil
    f_on_U = 0.5 * (f[1:-1, 1:-1] + f[1:-1, 2:])
    dh_dlat_U = (h[2:, 1:-1] + h[2:, 2:] - h[:-2, 1:-1] - h[:-2, 2:]) / (4.0 * dlat)
    u_g = interior(-gravity / (f_on_U * R) * dh_dlat_U, h)

    # v_g at V-points: compact 4-point stencil
    f_on_V = 0.5 * (f[1:-1, 1:-1] + f[2:, 1:-1])
    cos_on_V = 0.5 * (grid.cos_lat_T[1:-1, 1:-1] + grid.cos_lat_T[2:, 1:-1])
    dh_dlon_V = (h[1:-1, 2:] + h[2:, 2:] - h[1:-1, :-2] - h[2:, :-2]) / (4.0 * dlon)
    v_g = interior(_safe_div_cos(gravity * dh_dlon_V, cos_on_V, f_on_V * R), h)

    return u_g, v_g


# ======================================================================
# 3-D wrappers (vmap over z-levels)
# ======================================================================


class SphericalDivergence3D(eqx.Module):
    """3-D spherical divergence (vmap over z-levels).

    Parameters
    ----------
    grid : SphericalArakawaCGrid3D
    """

    grid: SphericalArakawaCGrid3D
    _div2d: SphericalDivergence2D

    def __init__(self, grid: SphericalArakawaCGrid3D):
        self.grid = grid
        self._div2d = SphericalDivergence2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Horizontal divergence at T-points over all z-levels.

        ``mask`` is an optional 2-D :class:`ArakawaCGridMask` broadcast
        over all z-levels; if provided, the output is multiplied by
        ``mask.h``.
        """
        out = zero_z_ghosts(eqx.filter_vmap(self._div2d)(u, v))
        if mask is not None:
            out = out * mask.h
        return out


class SphericalVorticity3D(eqx.Module):
    """3-D spherical vorticity (vmap over z-levels).

    Parameters
    ----------
    grid : SphericalArakawaCGrid3D
    """

    grid: SphericalArakawaCGrid3D
    _vort2d: SphericalVorticity2D

    def __init__(self, grid: SphericalArakawaCGrid3D):
        self.grid = grid
        self._vort2d = SphericalVorticity2D(grid=grid.horizontal_grid())

    def relative_vorticity(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Relative vorticity at X-points over all z-levels.

        ``mask`` is an optional 2-D :class:`ArakawaCGridMask` broadcast
        over all z-levels; if provided, the output is multiplied by
        ``mask.psi``.
        """
        out = zero_z_ghosts(eqx.filter_vmap(self._vort2d.relative_vorticity)(u, v))
        if mask is not None:
            out = out * mask.psi
        return out


class SphericalLaplacian3D(eqx.Module):
    """3-D spherical Laplacian (vmap over z-levels).

    Parameters
    ----------
    grid : SphericalArakawaCGrid3D
    """

    grid: SphericalArakawaCGrid3D
    _lap2d: SphericalLaplacian2D

    def __init__(self, grid: SphericalArakawaCGrid3D):
        self.grid = grid
        self._lap2d = SphericalLaplacian2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Laplacian at T-points over all z-levels.

        ``mask`` is an optional 2-D :class:`ArakawaCGridMask` broadcast
        over all z-levels; if provided, the output is multiplied by
        ``mask.h``.
        """
        out = zero_z_ghosts(eqx.filter_vmap(self._lap2d)(h))
        if mask is not None:
            out = out * mask.h
        return out
