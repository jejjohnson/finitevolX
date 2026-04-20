"""
Area- and volume-weighted reduction helpers for Arakawa C-grids.

Provides scalar totals and means of T-point fields that account for
the correct per-cell metric on Cartesian and spherical grids:

    Cartesian T-cell area  = dx · dy
    Spherical T-cell area  = R² · cos(lat_T) · dlon · dlat
    3-D T-cell volume      = area · dz

Only physical interior cells contribute — the ghost ring is excluded
by construction.  When a mask is supplied, dry cells are excluded from
both the numerator and the denominator of ``*_mean``.

The polymorphic dispatchers :func:`area_sum`, :func:`area_mean`,
:func:`volume_sum`, :func:`volume_mean` choose the right metric at
runtime based on the grid type, so user code can stay grid-agnostic.

Example
-------
>>> from finitevolx import area_sum, SphericalGrid2D
>>> grid = SphericalGrid2D.from_interior(64, 32, (0.0, 360.0), (-80.0, 80.0))
>>> h = jnp.ones((grid.Ny, grid.Nx))
>>> total = area_sum(h, grid)  # integrated tracer over the basin
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.base import (
    ArakawaCGrid2D,
    ArakawaCGrid3D,
)
from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D

# ----------------------------------------------------------------------
# Cell-area / cell-volume metrics
# ----------------------------------------------------------------------


def cartesian_area_weights(grid: CartesianGrid2D) -> Float[Array, "Ny Nx"]:
    """T-cell areas on a Cartesian grid.

    ``A[j, i] = dx · dy`` broadcast to the grid shape.

    Parameters
    ----------
    grid : CartesianGrid2D

    Returns
    -------
    Float[Array, "Ny Nx"]
        Per-cell area, constant everywhere.
    """
    return jnp.full((grid.Ny, grid.Nx), grid.dx * grid.dy)


def spherical_area_weights(grid: SphericalGrid2D) -> Float[Array, "Ny Nx"]:
    """T-cell areas on a spherical grid.

    ``A[j, i] = R² · cos(lat_T[j, i]) · dlon · dlat``.

    Near the poles (``|cos(lat_T)| < 1e-12``) the weight is taken as
    zero rather than a tiny positive number to avoid near-singular
    denominators in ``*_mean``.

    Parameters
    ----------
    grid : SphericalGrid2D

    Returns
    -------
    Float[Array, "Ny Nx"]
        Per-T-cell area.
    """
    cos_T = grid.cos_lat_T
    base = grid.R**2 * grid.dlon * grid.dlat
    weights = base * cos_T
    return jnp.where(jnp.abs(cos_T) < 1e-12, 0.0, weights)


def area_weights(grid: ArakawaCGrid2D) -> Float[Array, "Ny Nx"]:
    """Dispatch to :func:`cartesian_area_weights` or :func:`spherical_area_weights`.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        A concrete 2-D grid (Cartesian or spherical).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Per-T-cell area on the supplied grid.

    Raises
    ------
    TypeError
        If the grid type is not recognized.
    """
    if isinstance(grid, SphericalGrid2D):
        return spherical_area_weights(grid)
    if isinstance(grid, CartesianGrid2D):
        return cartesian_area_weights(grid)
    raise TypeError(
        f"area_weights: unsupported grid type {type(grid).__name__}. "
        "Use CartesianGrid2D or SphericalGrid2D."
    )


def cartesian_volume_weights(grid: CartesianGrid3D) -> Float[Array, "Nz Ny Nx"]:
    """T-cell volumes on a Cartesian 3-D grid: ``dx · dy · dz``."""
    return jnp.full((grid.Nz, grid.Ny, grid.Nx), grid.dx * grid.dy * grid.dz)


def spherical_volume_weights(grid: SphericalGrid3D) -> Float[Array, "Nz Ny Nx"]:
    """T-cell volumes on a spherical 3-D grid.

    ``V[k, j, i] = R² · cos(lat_T[j, i]) · dlon · dlat · dz`` — the
    horizontal area broadcast over z, times the uniform vertical
    thickness.
    """
    area_2d = spherical_area_weights(grid.horizontal_grid())
    return jnp.broadcast_to(area_2d * grid.dz, (grid.Nz, grid.Ny, grid.Nx))


def volume_weights(grid: ArakawaCGrid3D) -> Float[Array, "Nz Ny Nx"]:
    """Dispatch to Cartesian/spherical volume weights based on grid type."""
    if isinstance(grid, SphericalGrid3D):
        return spherical_volume_weights(grid)
    if isinstance(grid, CartesianGrid3D):
        return cartesian_volume_weights(grid)
    raise TypeError(
        f"volume_weights: unsupported grid type {type(grid).__name__}. "
        "Use CartesianGrid3D or SphericalGrid3D."
    )


# ----------------------------------------------------------------------
# 2-D interior reductions
# ----------------------------------------------------------------------


def _interior_2d(field: Float[Array, "Ny Nx"]) -> Float[Array, "Ny_i Nx_i"]:
    """Slice physical interior (exclude 1-cell ghost ring)."""
    return field[1:-1, 1:-1]


def _mask_factor_2d(
    mask: Mask2D | None,
    grid: ArakawaCGrid2D,
) -> Float[Array, "Ny Nx"]:
    """Return a float mask for T-points (1.0 wet, 0.0 dry) or all-ones."""
    if mask is None:
        return jnp.ones((grid.Ny, grid.Nx))
    return jnp.asarray(mask.h, dtype=jnp.float64)


def area_sum(
    field: Float[Array, "Ny Nx"],
    grid: ArakawaCGrid2D,
    mask: Mask2D | None = None,
) -> Float[Array, ""]:
    """Area-weighted sum over physical interior T-cells.

    Computes ``Σ_j Σ_i A[j, i] · m[j, i] · field[j, i]`` where ``A``
    is the per-cell area metric for the grid type (Cartesian or
    spherical), ``m`` is the T-point wet/dry mask (1.0 wet, 0.0 dry)
    when ``mask`` is supplied, and the sum is taken over the interior
    ``[1:-1, 1:-1]`` only.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        T-point field.
    grid : ArakawaCGrid2D
        Cartesian or spherical 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  Dry cells contribute zero.

    Returns
    -------
    Float[Array, ""]
        Scalar area-weighted sum.
    """
    w = area_weights(grid)
    m = _mask_factor_2d(mask, grid)
    return jnp.sum(_interior_2d(w * m * field))


def area_mean(
    field: Float[Array, "Ny Nx"],
    grid: ArakawaCGrid2D,
    mask: Mask2D | None = None,
) -> Float[Array, ""]:
    """Area-weighted mean over physical interior T-cells.

    Returns ``(Σ A·m·field) / (Σ A·m)``.  Dry cells are excluded from
    both sums.  When the total wet area is zero, returns NaN rather
    than ±inf.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        T-point field.
    grid : ArakawaCGrid2D
    mask : Mask2D or None, optional

    Returns
    -------
    Float[Array, ""]
        Scalar area-weighted mean, or NaN if the total wet area is 0.
    """
    w = area_weights(grid)
    m = _mask_factor_2d(mask, grid)
    wm = _interior_2d(w * m)
    num = jnp.sum(wm * _interior_2d(field))
    den = jnp.sum(wm)
    return jnp.where(den == 0.0, jnp.nan, num / jnp.where(den == 0.0, 1.0, den))


# ----------------------------------------------------------------------
# 3-D interior reductions
# ----------------------------------------------------------------------


def _interior_3d(
    field: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz_i Ny_i Nx_i"]:
    return field[1:-1, 1:-1, 1:-1]


def _mask_factor_3d(
    mask: Mask3D | None,
    grid: ArakawaCGrid3D,
) -> Float[Array, "Nz Ny Nx"]:
    if mask is None:
        return jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    return jnp.asarray(mask.h, dtype=jnp.float64)


def volume_sum(
    field: Float[Array, "Nz Ny Nx"],
    grid: ArakawaCGrid3D,
    mask: Mask3D | None = None,
) -> Float[Array, ""]:
    """Volume-weighted sum over physical interior T-cells (3-D).

    Same semantics as :func:`area_sum` but with the per-cell volume
    metric ``V[k, j, i] = A[j, i] · dz`` and interior
    ``[1:-1, 1:-1, 1:-1]``.

    Parameters
    ----------
    field : Float[Array, "Nz Ny Nx"]
        T-point field.
    grid : ArakawaCGrid3D
    mask : Mask3D or None, optional

    Returns
    -------
    Float[Array, ""]
    """
    w = volume_weights(grid)
    m = _mask_factor_3d(mask, grid)
    return jnp.sum(_interior_3d(w * m * field))


def volume_mean(
    field: Float[Array, "Nz Ny Nx"],
    grid: ArakawaCGrid3D,
    mask: Mask3D | None = None,
) -> Float[Array, ""]:
    """Volume-weighted mean over physical interior T-cells (3-D).

    Returns ``(Σ V·m·field) / (Σ V·m)``.  NaN when the total wet
    volume is zero.
    """
    w = volume_weights(grid)
    m = _mask_factor_3d(mask, grid)
    wm = _interior_3d(w * m)
    num = jnp.sum(wm * _interior_3d(field))
    den = jnp.sum(wm)
    return jnp.where(den == 0.0, jnp.nan, num / jnp.where(den == 0.0, 1.0, den))


# ----------------------------------------------------------------------
# Explicit per-coordinate convenience aliases
# ----------------------------------------------------------------------


def cartesian_area_sum(
    field: Float[Array, "Ny Nx"],
    grid: CartesianGrid2D,
    mask: Mask2D | None = None,
) -> Float[Array, ""]:
    """Cartesian area-weighted sum (:func:`area_sum` with Cartesian grid)."""
    return area_sum(field, grid, mask)


def cartesian_area_mean(
    field: Float[Array, "Ny Nx"],
    grid: CartesianGrid2D,
    mask: Mask2D | None = None,
) -> Float[Array, ""]:
    """Cartesian area-weighted mean."""
    return area_mean(field, grid, mask)


def spherical_area_sum(
    field: Float[Array, "Ny Nx"],
    grid: SphericalGrid2D,
    mask: Mask2D | None = None,
) -> Float[Array, ""]:
    """Spherical area-weighted sum."""
    return area_sum(field, grid, mask)


def spherical_area_mean(
    field: Float[Array, "Ny Nx"],
    grid: SphericalGrid2D,
    mask: Mask2D | None = None,
) -> Float[Array, ""]:
    """Spherical area-weighted mean."""
    return area_mean(field, grid, mask)


def cartesian_volume_sum(
    field: Float[Array, "Nz Ny Nx"],
    grid: CartesianGrid3D,
    mask: Mask3D | None = None,
) -> Float[Array, ""]:
    """Cartesian volume-weighted sum."""
    return volume_sum(field, grid, mask)


def cartesian_volume_mean(
    field: Float[Array, "Nz Ny Nx"],
    grid: CartesianGrid3D,
    mask: Mask3D | None = None,
) -> Float[Array, ""]:
    """Cartesian volume-weighted mean."""
    return volume_mean(field, grid, mask)


def spherical_volume_sum(
    field: Float[Array, "Nz Ny Nx"],
    grid: SphericalGrid3D,
    mask: Mask3D | None = None,
) -> Float[Array, ""]:
    """Spherical volume-weighted sum."""
    return volume_sum(field, grid, mask)


def spherical_volume_mean(
    field: Float[Array, "Nz Ny Nx"],
    grid: SphericalGrid3D,
    mask: Mask3D | None = None,
) -> Float[Array, ""]:
    """Spherical volume-weighted mean."""
    return volume_mean(field, grid, mask)
