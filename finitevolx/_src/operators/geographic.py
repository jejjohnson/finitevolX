"""
Spherical-coordinate finite-difference operators on Arakawa C-grids.

All operators include the appropriate metric terms for a spherical
(lon, lat) grid.  Grid spacings ``dlon`` and ``dlat`` are in
**radians**, and the Earth radius ``R`` enters as a scale factor.

Grid point convention (same as Cartesian C-grid but with lon/lat):

    T[j, i]  cell centre    (lon_i,        lat_j       )
    U[j, i]  east face      (lon_{i+1/2},  lat_j       )
    V[j, i]  north face     (lon_i,        lat_{j+1/2} )
    X[j, i]  NE corner      (lon_{i+1/2},  lat_{j+1/2} )

Interior-point idiom: output has the same shape as input, ghost ring
is zero.

**Pole handling**: operators that divide by ``cos(lat)`` produce
``NaN`` where ``|cos(lat)| < eps`` (near the poles).  This makes
pole singularities explicit rather than silently returning ``Inf``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.constants import R_EARTH

_COS_EPS = 1e-12  # guard against division by cos(lat) ≈ 0 near poles


def _safe_div_cos(
    numerator: Float[Array, "..."],
    cos_val: Float[Array, "..."],
    scale: float | Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute ``numerator / (scale * cos_val)`` with pole guard.

    Returns NaN where ``|cos_val| < eps`` instead of Inf.
    """
    denom = scale * cos_val
    safe_denom = jnp.where(jnp.abs(cos_val) < _COS_EPS, 1.0, denom)
    result = numerator / safe_denom
    return jnp.where(jnp.abs(cos_val) < _COS_EPS, jnp.nan, result)


# ======================================================================
# Spherical differences  (Issue #7)
# ======================================================================


def diff_lon_T_to_U(
    h: Float[Array, "Ny Nx"],
    cos_lat_T: Float[Array, "Ny Nx"],
    dlon: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Zonal derivative T -> U on a sphere.

    dh/dx[j, i+1/2] = (h[j, i+1] - h[j, i]) / (R * cos(lat_j) * dlon)

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Scalar at T-points.
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points.
    dlon : float
        Longitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Zonal derivative at U-points.
    """
    out = jnp.zeros_like(h)
    cos_on_U = 0.5 * (cos_lat_T[1:-1, 1:-1] + cos_lat_T[1:-1, 2:])
    numer = h[1:-1, 2:] - h[1:-1, 1:-1]
    out = out.at[1:-1, 1:-1].set(_safe_div_cos(numer, cos_on_U, R * dlon))
    return out


def diff_lat_T_to_V(
    h: Float[Array, "Ny Nx"],
    dlat: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Meridional derivative T -> V on a sphere.

    dh/dy[j+1/2, i] = (h[j+1, i] - h[j, i]) / (R * dlat)

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Scalar at T-points.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Meridional derivative at V-points.
    """
    out = jnp.zeros_like(h)
    out = out.at[1:-1, 1:-1].set((h[2:, 1:-1] - h[1:-1, 1:-1]) / (R * dlat))
    return out


def diff_lon_V_to_X(
    v: Float[Array, "Ny Nx"],
    cos_lat_V: Float[Array, "Ny Nx"],
    dlon: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Zonal derivative V -> X (corner) on a sphere.

    dv/dx[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i])
                           / (R * cos(lat_{j+1/2}) * dlon)

    Parameters
    ----------
    v : Float[Array, "Ny Nx"]
        Velocity at V-points.
    cos_lat_V : Float[Array, "Ny Nx"]
        cos(latitude) at V-points.
    dlon : float
        Longitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Zonal derivative at X-points.
    """
    out = jnp.zeros_like(v)
    cos_on_X = 0.5 * (cos_lat_V[1:-1, 1:-1] + cos_lat_V[1:-1, 2:])
    numer = v[1:-1, 2:] - v[1:-1, 1:-1]
    out = out.at[1:-1, 1:-1].set(_safe_div_cos(numer, cos_on_X, R * dlon))
    return out


def diff_lat_U_to_X(
    u: Float[Array, "Ny Nx"],
    dlat: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Meridional derivative U -> X (corner) on a sphere.

    du/dy[j+1/2, i+1/2] = (u[j+1, i+1/2] - u[j, i+1/2]) / (R * dlat)

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Velocity at U-points.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Meridional derivative at X-points.
    """
    out = jnp.zeros_like(u)
    out = out.at[1:-1, 1:-1].set((u[2:, 1:-1] - u[1:-1, 1:-1]) / (R * dlat))
    return out


# ======================================================================
# Backward spherical differences (face -> centre)
# ======================================================================


def diff_lon_U_to_T(
    u: Float[Array, "Ny Nx"],
    cos_lat_T: Float[Array, "Ny Nx"],
    dlon: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Backward zonal derivative U -> T on a sphere.

    du/dx[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / (R * cos(lat_j) * dlon)

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Velocity at U-points.
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points.
    dlon : float
        Longitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Zonal derivative at T-points.
    """
    out = jnp.zeros_like(u)
    cos_T = cos_lat_T[1:-1, 1:-1]
    numer = u[1:-1, 1:-1] - u[1:-1, :-2]
    out = out.at[1:-1, 1:-1].set(_safe_div_cos(numer, cos_T, R * dlon))
    return out


def diff_lat_V_to_T(
    v: Float[Array, "Ny Nx"],
    dlat: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Backward meridional derivative V -> T on a sphere.

    dv/dy[j, i] = (v[j+1/2, i] - v[j-1/2, i]) / (R * dlat)

    Parameters
    ----------
    v : Float[Array, "Ny Nx"]
        Velocity at V-points.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Meridional derivative at T-points.
    """
    out = jnp.zeros_like(v)
    out = out.at[1:-1, 1:-1].set((v[1:-1, 1:-1] - v[:-2, 1:-1]) / (R * dlat))
    return out


# ======================================================================
# Second-order spherical differences  (Issue #7)
# ======================================================================


def diff2_lon_T(
    h: Float[Array, "Ny Nx"],
    cos_lat_T: Float[Array, "Ny Nx"],
    dlon: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Second zonal derivative at T-points on a sphere.

    d²h/dx² = 1/(R² cos²(lat)) * (h[j,i+1] - 2h[j,i] + h[j,i-1]) / dlon²

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Scalar at T-points.
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points.
    dlon : float
        Longitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Second zonal derivative at T-points.  Ghost ring is zero.
    """
    out = jnp.zeros_like(h)
    cos_T = cos_lat_T[1:-1, 1:-1]
    # (h[j,i+1] - 2*h[j,i] + h[j,i-1]) / dlon^2
    d2h = (h[1:-1, 2:] - 2.0 * h[1:-1, 1:-1] + h[1:-1, :-2]) / dlon**2
    # d²h/dx² = d2h / (R² cos²(lat));  NaN near poles
    out = out.at[1:-1, 1:-1].set(_safe_div_cos(d2h, cos_T, R**2 * cos_T))
    return out


def laplacian_merid_T(
    h: Float[Array, "Ny Nx"],
    cos_lat_T: Float[Array, "Ny Nx"],
    dlat: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Meridional term of the spherical Laplacian at T-points.

    1/(R² cos(lat)) * d/dlat(cos(lat) * dh/dlat)

    This is **not** a plain d²h/dy²; it includes the metric factor
    ``1/cos(lat) * d/dlat(cos * ...)`` that accounts for convergence
    of meridians.  Combined with :func:`diff2_lon_T` it gives the full
    spherical Laplacian (see :func:`laplacian_sphere`).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Scalar at T-points.
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Meridional Laplacian term at T-points.  Ghost ring is zero.
        NaN near the poles where ``|cos(lat)| < eps``.
    """
    out = jnp.zeros_like(h)
    cos_T = cos_lat_T[1:-1, 1:-1]
    # dh/dlat at half-indices
    dh_N = (h[2:, 1:-1] - h[1:-1, 1:-1]) / dlat
    dh_S = (h[1:-1, 1:-1] - h[:-2, 1:-1]) / dlat
    # cos(lat) at half-indices (V-point latitudes)
    cos_N = 0.5 * (cos_T + cos_lat_T[2:, 1:-1])
    cos_S = 0.5 * (cos_T + cos_lat_T[:-2, 1:-1])
    d_cos_dh = (cos_N * dh_N - cos_S * dh_S) / dlat
    out = out.at[1:-1, 1:-1].set(_safe_div_cos(d_cos_dh, cos_T, R**2))
    return out


# ======================================================================
# Compound spherical operators
# ======================================================================


def divergence_sphere(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    cos_lat_T: Float[Array, "Ny Nx"],
    cos_lat_V: Float[Array, "Ny Nx"],
    dlon: float,
    dlat: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Horizontal divergence on a sphere at T-points.

    div = 1/(R cos(lat)) * [ d(u)/dlon + d(v cos(lat))/dlat ]

    Using backward differences U->T and V->T with metric terms.

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Zonal velocity at U-points.
    v : Float[Array, "Ny Nx"]
        Meridional velocity at V-points.
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points.
    cos_lat_V : Float[Array, "Ny Nx"]
        cos(latitude) at V-points.
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Divergence at T-points.
    """
    out = jnp.zeros_like(u)
    # du/dlon at T: backward diff U -> T
    du_dlon = (u[1:-1, 1:-1] - u[1:-1, :-2]) / dlon
    # d(v*cos)/dlat at T: backward diff V -> T
    v_cos_N = v[1:-1, 1:-1] * cos_lat_V[1:-1, 1:-1]  # north face
    v_cos_S = v[:-2, 1:-1] * cos_lat_V[:-2, 1:-1]  # south face
    dvc_dlat = (v_cos_N - v_cos_S) / dlat
    cos_T = cos_lat_T[1:-1, 1:-1]
    out = out.at[1:-1, 1:-1].set(_safe_div_cos(du_dlon + dvc_dlat, cos_T, R))
    return out


def curl_sphere(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    cos_lat_U: Float[Array, "Ny Nx"],
    cos_lat_X: Float[Array, "Ny Nx"],
    dlon: float,
    dlat: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Relative vorticity (curl) on a sphere at X-points.

    zeta = 1/(R cos(lat)) * [ dv/dlon - d(u cos(lat))/dlat ]

    Using forward differences V->X and U->X with metric terms.

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Zonal velocity at U-points.
    v : Float[Array, "Ny Nx"]
        Meridional velocity at V-points.
    cos_lat_U : Float[Array, "Ny Nx"]
        cos(latitude) at U-points.
    cos_lat_X : Float[Array, "Ny Nx"]
        cos(latitude) at X-points.
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Relative vorticity at X-points.
    """
    out = jnp.zeros_like(u)
    # dv/dlon at X: forward diff V -> X
    dv_dlon = (v[1:-1, 2:] - v[1:-1, 1:-1]) / dlon
    # d(u*cos)/dlat at X: forward diff U -> X
    u_cos_N = u[2:, 1:-1] * cos_lat_U[2:, 1:-1]
    u_cos_S = u[1:-1, 1:-1] * cos_lat_U[1:-1, 1:-1]
    duc_dlat = (u_cos_N - u_cos_S) / dlat
    cos_X = cos_lat_X[1:-1, 1:-1]
    out = out.at[1:-1, 1:-1].set(_safe_div_cos(dv_dlon - duc_dlat, cos_X, R))
    return out


def laplacian_sphere(
    h: Float[Array, "Ny Nx"],
    cos_lat_T: Float[Array, "Ny Nx"],
    dlon: float,
    dlat: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Laplacian on a sphere at T-points.

    nabla^2 h = 1/(R^2 cos^2(lat)) * d^2h/dlon^2
              + 1/(R^2 cos(lat)) * d/dlat(cos(lat) * dh/dlat)

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Scalar at T-points.
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points.
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Laplacian at T-points.
    """
    out = jnp.zeros_like(h)
    cos_T = cos_lat_T[1:-1, 1:-1]

    # d^2h/dlon^2
    d2h_dlon2 = (h[1:-1, 2:] - 2.0 * h[1:-1, 1:-1] + h[1:-1, :-2]) / dlon**2

    # d/dlat(cos * dh/dlat):
    # dh/dlat at V-points (half-index j+1/2 and j-1/2)
    dh_dlat_N = (h[2:, 1:-1] - h[1:-1, 1:-1]) / dlat
    dh_dlat_S = (h[1:-1, 1:-1] - h[:-2, 1:-1]) / dlat
    # cos(lat) at V-point latitudes = midpoints
    cos_V_N = 0.5 * (cos_T + cos_lat_T[2:, 1:-1])
    cos_V_S = 0.5 * (cos_T + cos_lat_T[:-2, 1:-1])
    d_coslat_dh = (cos_V_N * dh_dlat_N - cos_V_S * dh_dlat_S) / dlat

    lon_term = _safe_div_cos(d2h_dlon2, cos_T, R**2 * cos_T)
    lat_term = _safe_div_cos(d_coslat_dh, cos_T, R**2)
    out = out.at[1:-1, 1:-1].set(lon_term + lat_term)
    return out


def geostrophic_velocity_sphere(
    h: Float[Array, "Ny Nx"],
    f: Float[Array, "Ny Nx"],
    cos_lat_T: Float[Array, "Ny Nx"],
    dlon: float,
    dlat: float,
    gravity: float = 9.80665,
    R: float = R_EARTH,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Geostrophic velocity from free-surface height on a sphere.

    u_g = -g / (f R) * dh/dlat
    v_g =  g / (f R cos(lat)) * dh/dlon

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Free-surface height (or pressure / f) at T-points.
    f : Float[Array, "Ny Nx"]
        Coriolis parameter at T-points.
    cos_lat_T : Float[Array, "Ny Nx"]
        cos(latitude) at T-points.
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    gravity : float
        Gravitational acceleration.
    R : float
        Radius of the sphere.

    Returns
    -------
    tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
        (u_g at U-points, v_g at V-points).
    """
    # u_g at U-points: -g/f * dh/dy  →  needs f on U, dh/dlat at U
    # v_g at V-points:  g/f * dh/dx  →  needs f on V, dh/dlon at V
    u_g = jnp.zeros_like(h)
    v_g = jnp.zeros_like(h)

    # dh/dlat at U-points: average T→U of dh/dlat
    # Use the compact 4-point stencil (like grad_perp)
    # u_g[j, i+1/2] = -g/(f_on_U * R) * (h[j+1,i]+h[j+1,i+1]-h[j-1,i]-h[j-1,i+1])/(4*dlat)
    f_on_U = 0.5 * (f[1:-1, 1:-1] + f[1:-1, 2:])
    dh_dlat_U = (h[2:, 1:-1] + h[2:, 2:] - h[:-2, 1:-1] - h[:-2, 2:]) / (4.0 * dlat)
    u_g = u_g.at[1:-1, 1:-1].set(-gravity / (f_on_U * R) * dh_dlat_U)

    # dh/dlon at V-points: average T→V of dh/dlon
    f_on_V = 0.5 * (f[1:-1, 1:-1] + f[2:, 1:-1])
    cos_on_V = 0.5 * (cos_lat_T[1:-1, 1:-1] + cos_lat_T[2:, 1:-1])
    dh_dlon_V = (h[1:-1, 2:] + h[2:, 2:] - h[1:-1, :-2] - h[2:, :-2]) / (4.0 * dlon)
    v_g = v_g.at[1:-1, 1:-1].set(
        _safe_div_cos(gravity * dh_dlon_V, cos_on_V, f_on_V * R)
    )

    return u_g, v_g


def potential_vorticity_sphere(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    h: Float[Array, "Ny Nx"],
    f: Float[Array, "Ny Nx"],
    cos_lat_U: Float[Array, "Ny Nx"],
    cos_lat_X: Float[Array, "Ny Nx"],
    dlon: float,
    dlat: float,
    R: float = R_EARTH,
) -> Float[Array, "Ny Nx"]:
    """Potential vorticity on a sphere at X-points.

    q = (zeta + f) / h

    where zeta is the relative vorticity computed with spherical metric
    terms and h is interpolated to X-points.

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Zonal velocity at U-points.
    v : Float[Array, "Ny Nx"]
        Meridional velocity at V-points.
    h : Float[Array, "Ny Nx"]
        Layer thickness at T-points.
    f : Float[Array, "Ny Nx"]
        Coriolis parameter at T-points.
    cos_lat_U : Float[Array, "Ny Nx"]
        cos(latitude) at U-points.
    cos_lat_X : Float[Array, "Ny Nx"]
        cos(latitude) at X-points.
    dlon : float
        Longitudinal grid spacing in radians.
    dlat : float
        Latitudinal grid spacing in radians.
    R : float
        Radius of the sphere.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Potential vorticity at X-points.
    """
    zeta = curl_sphere(u, v, cos_lat_U, cos_lat_X, dlon, dlat, R)
    # Interpolate f and h from T to X (bilinear average of 4 T-cells)
    f_on_X = 0.25 * (f[1:-1, 1:-1] + f[1:-1, 2:] + f[2:, 1:-1] + f[2:, 2:])
    h_on_X = 0.25 * (h[1:-1, 1:-1] + h[1:-1, 2:] + h[2:, 1:-1] + h[2:, 2:])

    out = jnp.zeros_like(u)
    num = zeta[1:-1, 1:-1] + f_on_X
    out = out.at[1:-1, 1:-1].set(jnp.where(h_on_X == 0, jnp.nan, num / h_on_X))
    return out
