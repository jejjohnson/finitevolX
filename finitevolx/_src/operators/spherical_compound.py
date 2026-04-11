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

from finitevolx._src.grid.spherical import (
    SphericalGrid2D,
    SphericalGrid3D,
)
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators._utils import _safe_div_cos
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.operators.stencils import (
    diff_x_bwd,
    diff_x_fwd,
)


class SphericalDivergence2D(eqx.Module):
    """Horizontal divergence on a 2-D spherical Arakawa C-grid.

    div = 1/(R·cosφ) · [du/dλ + d(v·cosφ)/dφ]   at T-points

    Coriolis is coordinate-independent — use ``Coriolis2D`` directly
    with a spherical grid.

    Parameters
    ----------
    grid : SphericalGrid2D
        The underlying 2-D spherical grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  Post-compute pattern — the divergence
        output is multiplied by ``mask.h`` at the end.  Per #209 Q2,
        this is a Cartesian ``Mask2D`` pending a dedicated
        ``SphericalMask2D`` follow-up.
    """

    grid: SphericalGrid2D
    mask: Mask2D | None = None

    def __call__(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Horizontal divergence at T-points.

        div[j, i] = 1/(R·cos(lat_T)) · [du/dλ + d(v·cos(lat_V))/dφ]
        """
        dlon = self.grid.dlon
        dlat = self.grid.dlat
        R = self.grid.R

        # du/dλ at T: backward diff U → T
        du_dlon = diff_x_bwd(u) / dlon

        # d(v·cosφ)/dφ at T: backward diff V → T
        v_cos_N = v[1:-1, 1:-1] * self.grid.cos_lat_V[1:-1, 1:-1]
        v_cos_S = v[:-2, 1:-1] * self.grid.cos_lat_V[:-2, 1:-1]
        dvc_dlat = (v_cos_N - v_cos_S) / dlat

        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]
        out = interior(_safe_div_cos(du_dlon + dvc_dlat, cos_T, R), u)
        if self.mask is not None:
            out = out * self.mask.h
        return out


class SphericalVorticity2D(eqx.Module):
    """Vorticity and PV-flux operators on a 2-D spherical Arakawa C-grid.

    ζ = 1/(R·cosφ) · [dv/dλ − d(u·cosφ)/dφ]   at X-points

    The PV flux methods (energy-conserving, enstrophy-conserving,
    Arakawa-Lamb) are coordinate-independent: they involve only
    interpolation and multiplication.  They are delegated to an
    internal ``Interpolation2D``.

    Coriolis is coordinate-independent — use ``Coriolis2D`` directly
    with a spherical grid.

    Parameters
    ----------
    grid : SphericalGrid2D
        The underlying 2-D spherical grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  Pass-down pattern — the internal
        ``Interpolation2D`` is built with the same mask so the
        ``pv_flux_*`` methods pick up the post-compute zero
        automatically via the staggered-interp layer.  The sphere-
        specific ``relative_vorticity`` and ``potential_vorticity``
        compute via raw stencils and then post-multiply by
        ``mask.xy_corner_strict`` at the end.  ``potential_vorticity``
        additionally applies a ``jnp.where`` to replace the
        mask-induced NaNs at dry X-corners with exact zero — same
        fix as :class:`Vorticity2D.potential_vorticity`.

        Per #209 Q2, this is a Cartesian ``Mask2D`` pending a
        ``SphericalMask2D`` follow-up.
    """

    grid: SphericalGrid2D
    mask: Mask2D | None
    _interp: Interpolation2D

    def __init__(
        self,
        grid: SphericalGrid2D,
        mask: Mask2D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self._interp = Interpolation2D(grid=grid, mask=mask)

    def relative_vorticity(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Relative vorticity (curl) at X-points on a sphere.

        ζ[j+½, i+½] = 1/(R·cos(lat_X)) · [dv/dλ − d(u·cosφ)/dφ]
        """
        dlon = self.grid.dlon
        dlat = self.grid.dlat
        R = self.grid.R

        # dv/dλ at X: forward diff V → X
        dv_dlon = diff_x_fwd(v) / dlon

        # d(u·cosφ)/dφ at X: forward diff U → X
        u_cos_N = u[2:, 1:-1] * self.grid.cos_lat_U[2:, 1:-1]
        u_cos_S = u[1:-1, 1:-1] * self.grid.cos_lat_U[1:-1, 1:-1]
        duc_dlat = (u_cos_N - u_cos_S) / dlat

        cos_X = self.grid.cos_lat_X[1:-1, 1:-1]
        out = interior(_safe_div_cos(dv_dlon - duc_dlat, cos_X, R), u)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def potential_vorticity(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        h: Float[Array, "Ny Nx"],
        f: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Potential vorticity at X-points on a sphere.

        q = (ζ + f) / h
        """
        zeta = self.relative_vorticity(u, v)
        # f_on_X and h_on_X go through self._interp — pass-down masking applies.
        f_on_X = self._interp.T_to_X(f)
        h_on_X = self._interp.T_to_X(h)
        num = zeta[1:-1, 1:-1] + f_on_X[1:-1, 1:-1]
        den = h_on_X[1:-1, 1:-1]
        pv = jnp.where(den == 0, jnp.nan, num / den)
        out = interior(pv, u)
        if self.mask is not None:
            # Under pass-down masking, h_on_X is exactly zero at every dry
            # X-corner, so every dry corner hit the NaN branch.  Replace
            # those mask-induced NaNs with 0 while preserving the NaN
            # sentinel for genuine zero thickness at wet corners.
            out = jnp.where(self.mask.xy_corner_strict, out, 0.0)
        return out

    def pv_flux_energy_conserving(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Energy-conserving PV flux (coordinate-independent)."""
        q_on_u = self._interp.X_to_U(q)  # already * mask.u under pass-down
        q_on_v = self._interp.X_to_V(q)  # already * mask.v under pass-down
        qu = interior(q_on_u[1:-1, 1:-1] * u[1:-1, 1:-1], u)
        qv = interior(q_on_v[1:-1, 1:-1] * v[1:-1, 1:-1], v)
        return qu, qv

    def pv_flux_enstrophy_conserving(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Enstrophy-conserving PV flux (coordinate-independent)."""
        u_on_q = self._interp.U_to_X(u)  # * mask.xy_corner_strict
        v_on_q = self._interp.V_to_X(v)  # * mask.xy_corner_strict
        qu_at_q = interior(q[1:-1, 1:-1] * u_on_q[1:-1, 1:-1], q)
        qv_at_q = interior(q[1:-1, 1:-1] * v_on_q[1:-1, 1:-1], q)
        qu = self._interp.X_to_U(qu_at_q)  # * mask.u
        qv = self._interp.X_to_V(qv_at_q)  # * mask.v
        return qu, qv

    def pv_flux_arakawa_lamb(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        alpha: float = 1.0 / 3.0,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Arakawa-Lamb PV flux: weighted blend of energy and enstrophy."""
        qu_e, qv_e = self.pv_flux_energy_conserving(q, u, v)
        qu_s, qv_s = self.pv_flux_enstrophy_conserving(q, u, v)
        qu = alpha * qu_e + (1.0 - alpha) * qu_s
        qv = alpha * qv_e + (1.0 - alpha) * qv_s
        return qu, qv


class SphericalLaplacian2D(eqx.Module):
    """Spherical Laplacian on a 2-D Arakawa C-grid.

    ∇²h = 1/(R²cos²φ) · d²h/dλ² + 1/(R²cosφ) · d/dφ(cosφ · dh/dφ)

    Parameters
    ----------
    grid : SphericalGrid2D
        The underlying 2-D spherical grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  Post-compute pattern — the output
        is multiplied by ``mask.h`` at the end.  Per #209 Q2, this is
        a Cartesian ``Mask2D`` pending a ``SphericalMask2D`` follow-up.
    """

    grid: SphericalGrid2D
    mask: Mask2D | None = None

    def __call__(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Laplacian at T-points on a sphere."""
        R = self.grid.R
        dlon = self.grid.dlon
        dlat = self.grid.dlat
        cos_T = self.grid.cos_lat_T[1:-1, 1:-1]

        # d²h/dλ²
        d2h_dlon2 = (diff_x_fwd(h) - diff_x_bwd(h)) / dlon**2

        # d/dφ(cosφ · dh/dφ)
        dh_N = (h[2:, 1:-1] - h[1:-1, 1:-1]) / dlat
        dh_S = (h[1:-1, 1:-1] - h[:-2, 1:-1]) / dlat
        cos_V_N = 0.5 * (cos_T + self.grid.cos_lat_T[2:, 1:-1])
        cos_V_S = 0.5 * (cos_T + self.grid.cos_lat_T[:-2, 1:-1])
        d_coslat_dh = (cos_V_N * dh_N - cos_V_S * dh_S) / dlat

        lon_term = _safe_div_cos(d2h_dlon2, cos_T, R**2 * cos_T)
        lat_term = _safe_div_cos(d_coslat_dh, cos_T, R**2)
        out = interior(lon_term + lat_term, h)
        if self.mask is not None:
            out = out * self.mask.h
        return out


def geostrophic_velocity_sphere(
    h: Float[Array, "Ny Nx"],
    f: Float[Array, "Ny Nx"],
    grid: SphericalGrid2D,
    gravity: float = 9.80665,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Geostrophic velocity from free-surface height on a sphere.

    u_g = -g/(f·R) · dh/dφ       at U-points
    v_g =  g/(f·R·cosφ) · dh/dλ  at V-points

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Free-surface height at T-points.
    f : Float[Array, "Ny Nx"]
        Coriolis parameter at T-points.
    grid : SphericalGrid2D
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

    Pattern A (post-compute) — the inner 2-D op is mask-free and the
    3-D result is post-multiplied by ``mask.h``.

    Parameters
    ----------
    grid : SphericalGrid3D
    mask : Mask3D or None, optional
        Optional land/ocean mask (Cartesian ``Mask3D``; #209 Q3).
    """

    grid: SphericalGrid3D
    mask: Mask3D | None
    _div2d: SphericalDivergence2D

    def __init__(
        self,
        grid: SphericalGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self._div2d = SphericalDivergence2D(grid=grid.horizontal_grid())

    def __call__(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Horizontal divergence at T-points over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._div2d)(u, v))
        if self.mask is not None:
            out = out * self.mask.h
        return out


class SphericalVorticity3D(eqx.Module):
    """3-D spherical vorticity (vmap over z-levels).

    Pattern A (post-compute) — the inner 2-D op is mask-free and the
    3-D result is post-multiplied by ``mask.xy_corner_strict``.

    Parameters
    ----------
    grid : SphericalGrid3D
    mask : Mask3D or None, optional
        Optional land/ocean mask (Cartesian ``Mask3D``; #209 Q3).
    """

    grid: SphericalGrid3D
    mask: Mask3D | None
    _vort2d: SphericalVorticity2D

    def __init__(
        self,
        grid: SphericalGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self._vort2d = SphericalVorticity2D(grid=grid.horizontal_grid())

    def relative_vorticity(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Relative vorticity at X-points over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._vort2d.relative_vorticity)(u, v))
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out


class SphericalLaplacian3D(eqx.Module):
    """3-D spherical Laplacian (vmap over z-levels).

    Pattern A (post-compute) — the inner 2-D op is mask-free and the
    3-D result is post-multiplied by ``mask.h``.

    Parameters
    ----------
    grid : SphericalGrid3D
    mask : Mask3D or None, optional
        Optional land/ocean mask (Cartesian ``Mask3D``; #209 Q3).
    """

    grid: SphericalGrid3D
    mask: Mask3D | None
    _lap2d: SphericalLaplacian2D

    def __init__(
        self,
        grid: SphericalGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self._lap2d = SphericalLaplacian2D(grid=grid.horizontal_grid())

    def __call__(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Laplacian at T-points over all z-levels."""
        out = zero_z_ghosts(eqx.filter_vmap(self._lap2d)(h))
        if self.mask is not None:
            out = out * self.mask.h
        return out
