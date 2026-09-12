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
>>> import jax.numpy as jnp
>>> from finitevolx import area_sum, SphericalGrid2D
>>> grid = SphericalGrid2D.from_interior(64, 32, (0.0, 360.0), (-80.0, 80.0))
>>> h = jnp.ones((grid.Ny, grid.Nx))
>>> total = area_sum(h, grid)  # integrated tracer over the basin
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

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


def _apply_mask_2d(
    field: Float[Array, "Ny Nx"],
    weights: Float[Array, "Ny Nx"],
    mask: Mask2D | None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Zero out dry cells in ``field`` and ``weights`` (2-D).

    Uses ``jnp.where(mask.h, x, 0.0)`` rather than a multiplicative
    float-cast mask, so that NaN/Inf values on dry T-cells — common
    for land cells in realistic data — cannot contaminate the wet-cell
    totals.  Returns the inputs unchanged when ``mask is None``.  The
    dtype of ``field`` and ``weights`` is preserved.
    """
    if mask is None:
        return field, weights
    m = mask.h  # Bool[Array, "Ny Nx"]
    # Use plain 0.0 (weak-typed) so the mask path does not introduce an
    # upcast to float64 on top of the input dtype; the actual dtype
    # comes from the field/weights arrays themselves.
    safe_field = jnp.where(m, field, 0.0)
    safe_weights = jnp.where(m, weights, 0.0)
    return safe_field, safe_weights


def area_sum(
    field: Float[Array, "Ny Nx"],
    grid: ArakawaCGrid2D,
    mask: Mask2D | None = None,
) -> Float[Array, ""]:
    """Area-weighted sum over physical interior T-cells.

    Computes ``Σ_j Σ_i A[j, i] · field[j, i]`` summed over the
    interior ``[1:-1, 1:-1]`` only.  When ``mask`` is supplied, dry
    T-cells are zeroed out *before* multiplication using
    ``jnp.where(mask.h, field, 0.0)`` so NaN/Inf sentinels stored on
    land cells cannot contaminate the result.

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
        Scalar area-weighted sum.  Dtype follows the promoted dtype of
        ``field`` and the area weights.
    """
    w = area_weights(grid)
    f, w = _apply_mask_2d(field, w, mask)
    return jnp.sum(_interior_2d(w * f))


def area_mean(
    field: Float[Array, "Ny Nx"],
    grid: ArakawaCGrid2D,
    mask: Mask2D | None = None,
) -> Float[Array, ""]:
    """Area-weighted mean over physical interior T-cells.

    Returns ``(Σ A·field) / (Σ A)`` over wet T-cells.  Dry cells are
    excluded from both numerator and denominator via
    ``jnp.where(mask.h, ..., 0.0)``, so NaN/Inf on land does not
    contaminate the mean.  When the total wet area is zero, returns
    NaN rather than ±inf.

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
    f, w = _apply_mask_2d(field, w, mask)
    wi = _interior_2d(w)
    num = jnp.sum(wi * _interior_2d(f))
    den = jnp.sum(wi)
    return jnp.where(den == 0.0, jnp.nan, num / jnp.where(den == 0.0, 1.0, den))


# ----------------------------------------------------------------------
# 3-D interior reductions
# ----------------------------------------------------------------------


def _interior_3d(
    field: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz_i Ny_i Nx_i"]:
    return field[1:-1, 1:-1, 1:-1]


def _apply_mask_3d(
    field: Float[Array, "Nz Ny Nx"],
    weights: Float[Array, "Nz Ny Nx"],
    mask: Mask3D | None,
) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
    """Zero out dry cells in ``field`` and ``weights`` (3-D).

    Same NaN/Inf-safe pattern as :func:`_apply_mask_2d`.
    """
    if mask is None:
        return field, weights
    m = mask.h  # Bool[Array, "Nz Ny Nx"]
    # See _apply_mask_2d for the weak-typed 0.0 rationale.
    safe_field = jnp.where(m, field, 0.0)
    safe_weights = jnp.where(m, weights, 0.0)
    return safe_field, safe_weights


def volume_sum(
    field: Float[Array, "Nz Ny Nx"],
    grid: ArakawaCGrid3D,
    mask: Mask3D | None = None,
) -> Float[Array, ""]:
    """Volume-weighted sum over physical interior T-cells (3-D).

    Same semantics as :func:`area_sum` but with the per-cell volume
    metric ``V[k, j, i] = A[j, i] · dz`` and interior
    ``[1:-1, 1:-1, 1:-1]``.  NaN/Inf-safe under masking — dry cells
    are zeroed via ``jnp.where(mask.h, field, 0.0)`` before
    multiplication.

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
    f, w = _apply_mask_3d(field, w, mask)
    return jnp.sum(_interior_3d(w * f))


def volume_mean(
    field: Float[Array, "Nz Ny Nx"],
    grid: ArakawaCGrid3D,
    mask: Mask3D | None = None,
) -> Float[Array, ""]:
    """Volume-weighted mean over physical interior T-cells (3-D).

    Returns ``(Σ V·field) / (Σ V)`` over wet T-cells.  NaN/Inf-safe
    under masking; NaN when the total wet volume is zero.
    """
    w = volume_weights(grid)
    f, w = _apply_mask_3d(field, w, mask)
    wi = _interior_3d(w)
    num = jnp.sum(wi * _interior_3d(f))
    den = jnp.sum(wi)
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


# ----------------------------------------------------------------------
# Masked sample statistics
# ----------------------------------------------------------------------

#: Staggering locations whose mask a sample statistic may be taken at.
#: ``"w"`` exists only on :class:`Mask3D`.
_MASK_LOCATIONS: tuple[str, ...] = ("h", "u", "v", "xy_corner", "w")


def _location_mask(
    mask: Mask2D | Mask3D,
    location: str,
) -> Bool[Array, "..."]:
    """Pick the boolean field of ``mask`` for a staggering location."""
    if location not in _MASK_LOCATIONS:
        raise ValueError(
            f"masked statistics: unknown location {location!r}. "
            f"Expected one of {_MASK_LOCATIONS}."
        )
    m = getattr(mask, location, None)
    if m is None:
        raise ValueError(
            f"masked statistics: {type(mask).__name__} has no {location!r} "
            "mask. (Only Mask3D carries a 'w' mask.)"
        )
    return m


def _sample_counts(
    samples: Float[Array, "N ..."],
    mask: Mask2D | Mask3D | None,
    location: str,
    axis: int | tuple[int, ...],
) -> tuple[Float[Array, "N ..."], Float[Array, "..."], Bool[Array, "..."] | None]:
    """Mask-safe samples, per-output sample counts, and the wet indicator.

    Returns ``(safe_samples, count, wet)`` where ``safe_samples`` has
    dry entries replaced by ``0.0`` (so NaN/Inf land sentinels cannot
    contaminate the reduction), ``count`` is the number of contributing
    entries per output element, and ``wet`` is the broadcast boolean
    mask (``None`` when no mask was supplied).
    """
    dtype = _accumulation_dtype(samples)
    if mask is None:
        # The count is the same everywhere, so take it from the shape
        # rather than reducing an indicator array: materialising
        # ``ones`` the size of an ``[N, Nz, Ny, Nx]`` stack can cost
        # gigabytes of device memory for a number known statically.
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        count = 1
        for reduced in axes:
            count *= samples.shape[reduced]
        return samples.astype(dtype), jnp.asarray(count, dtype=dtype), None

    m = _location_mask(mask, location)
    wet = jnp.broadcast_to(m, samples.shape)
    safe = jnp.where(wet, samples, 0.0).astype(dtype)
    count = jnp.sum(wet.astype(dtype), axis=axis)
    return safe, count, wet


def _accumulation_dtype(samples: Float[Array, "N ..."]) -> jnp.dtype:
    """Floating dtype wide enough to accumulate ``samples`` safely.

    At least float32.  A half-precision input is promoted rather than
    accumulated in place: ``float16`` saturates at 65504, so a wet-cell
    count over a 256x256 field overflows to ``inf`` and every mean it
    divides silently collapses to zero.  Its ``eps**2`` variance floor
    underflows to zero for the same reason, taking the documented
    safe-to-divide guarantee with it.

    Promoting is deliberate rather than a cast-back: at ``eps=1e-8``
    the floor itself is below ``float16``'s smallest subnormal, so
    there is no half-precision value that could carry the guarantee.
    ``float32`` and ``float64`` inputs are returned unchanged.
    """
    return jnp.promote_types(jnp.asarray(samples).dtype, jnp.float32)


def masked_mean(
    samples: Float[Array, "N ..."],
    mask: Mask2D | Mask3D | None = None,
    *,
    location: str = "h",
    axis: int | tuple[int, ...] = 0,
) -> Float[Array, "..."]:
    """Mean over ``axis``, excluding dry cells.

    With the default ``axis=0`` and a leading sample axis (time or
    ensemble member) this is the per-gridpoint mean — the ``loc`` of a
    standardising affine transform.  Pass ``axis=(0, -2, -1)`` for a
    single scalar per field, in which case only wet cells enter both
    the sum and the divisor.

    Dry cells are zeroed via ``jnp.where(mask, x, 0.0)`` *before* the
    reduction, so NaN/Inf sentinels stored on land cannot contaminate
    wet-cell statistics — the same guarantee :func:`area_mean` gives.

    Parameters
    ----------
    samples : Float[Array, "N ..."]
        Stacked samples; the reduced axes are given by ``axis``.
    mask : Mask2D or Mask3D or None, optional
        Land/ocean mask, broadcast against the trailing axes of
        ``samples``.  ``None`` means every cell is wet.
    location : str, optional
        Staggering location whose mask to use: ``"h"`` (default),
        ``"u"``, ``"v"``, ``"xy_corner"``, or ``"w"`` (3-D only).
    axis : int or tuple of int, optional
        Axis or axes to reduce over.  Default ``0``.

    Returns
    -------
    Float[Array, "..."]
        The mean, in the input dtype or ``float32``, whichever is
        wider — see :func:`_accumulation_dtype`.  ``0.0`` wherever no
        wet sample contributed.
        Zero — rather than NaN — so that the paired
        ``(x - loc) / scale`` leaves a dry cell at zero and the inverse
        map stays well defined.
    """
    safe, count, _ = _sample_counts(samples, mask, location, axis)
    total = jnp.sum(safe, axis=axis)
    empty = count == 0
    return jnp.where(empty, 0.0, total / jnp.where(empty, 1.0, count))


def masked_std(
    samples: Float[Array, "N ..."],
    mask: Mask2D | Mask3D | None = None,
    *,
    location: str = "h",
    axis: int | tuple[int, ...] = 0,
    eps: float = 1e-8,
    ddof: int = 0,
) -> Float[Array, "..."]:
    """Standard deviation over ``axis``, excluding dry cells, floored at ``eps``.

    Companion to :func:`masked_mean`: together they give the ``loc``
    and ``scale`` of a per-gridpoint standardising transform.

    The floor matters because a constant field — a spun-up layer
    thickness, or any cell the dynamics never touch — has exactly zero
    sample variance, and dividing by it would produce inf/NaN.  Dry
    cells return ``1.0`` (an identity scale) rather than ``eps``, so
    that a masked field round-trips unchanged through
    ``(x - loc) / scale``.

    Parameters
    ----------
    samples : Float[Array, "N ..."]
    mask : Mask2D or Mask3D or None, optional
    location : str, optional
        See :func:`masked_mean`.
    axis : int or tuple of int, optional
    eps : float, optional
        Lower bound on the returned standard deviation of a wet cell.
        Default ``1e-8``. Applied to the standard deviation itself, so
        any positive value is honoured exactly — including one whose
        square would underflow the accumulation dtype.
    ddof : int, optional
        Delta degrees of freedom; the divisor is ``count - ddof``.
        Default ``0``, matching :func:`jax.numpy.std`.

    Returns
    -------
    Float[Array, "..."]
        Standard deviation, at least ``eps`` on wet cells and exactly
        ``1.0`` on dry cells or where too few samples contributed to
        form the estimate.  In the input dtype or ``float32``,
        whichever is wider — see :func:`_accumulation_dtype`.
    """
    safe, count, wet = _sample_counts(samples, mask, location, axis)
    empty = count == 0

    mean = jnp.where(
        empty, 0.0, jnp.sum(safe, axis=axis) / jnp.where(empty, 1.0, count)
    )
    # ``jnp.expand_dims`` accepts a tuple, re-inserting every reduced
    # axis as a size-1 dimension so the mean broadcasts back.
    dev = safe - jnp.expand_dims(mean, axis)
    # Re-zero dry entries: ``safe - mean`` is ``-mean`` on land, which
    # would otherwise contribute to the variance of a spatial reduction.
    if wet is not None:
        dev = jnp.where(wet, dev, 0.0)

    denom = count - ddof
    degenerate = denom <= 0
    var = jnp.sum(dev**2, axis=axis) / jnp.where(degenerate, 1.0, denom)

    # Two separate jobs, done in two steps rather than one.
    #
    # ``sqrt`` has an infinite derivative at zero, so a constant field
    # — exactly the case ``eps`` exists for — would come back with a
    # NaN gradient. The doubled ``where`` is the standard remedy: the
    # inner one keeps the zero away from ``sqrt`` on the backward pass,
    # the outer one restores the exact zero on the forward pass. No
    # value is lifted, so nothing is perturbed.
    #
    # Clamping the variance instead would be simpler but wrong twice
    # over. ``eps**2`` underflows for a small floor — ``1e-30`` squares
    # to ``1e-60``, which is zero in float32 — and the dtype's smallest
    # normal imposes a second, invisible floor of ``sqrt(tiny)``, about
    # ``1.1e-19``, so a requested ``eps`` below it could never be
    # returned and real deviations beneath it would be inflated.
    #
    # The requested floor is applied to the standard deviation itself,
    # where it is representable whatever its size. Where it binds, its
    # gradient is zero.
    positive = var > 0.0
    std = jnp.where(positive, jnp.sqrt(jnp.where(positive, var, 1.0)), 0.0)
    std = jnp.maximum(std, eps)
    # ``empty`` as well as ``degenerate``: with a negative ddof a dry
    # cell has ``count == 0`` but ``denom > 0``, so it is not
    # degenerate and would be handed ``eps`` instead of the identity.
    return jnp.where(degenerate | empty, 1.0, std)


def masked_moments(
    samples: Float[Array, "N ..."],
    mask: Mask2D | Mask3D | None = None,
    *,
    location: str = "h",
    axis: int | tuple[int, ...] = 0,
    eps: float = 1e-8,
    ddof: int = 0,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """``(mean, std)`` in one call — see :func:`masked_mean`, :func:`masked_std`.

    Returns
    -------
    tuple of Float[Array, "..."]
        The mask-aware mean and the ``eps``-floored standard deviation,
        which are exactly what :class:`StateAffine`-style standardising
        transforms need as ``loc`` and ``scale``.
    """
    return (
        masked_mean(samples, mask, location=location, axis=axis),
        masked_std(samples, mask, location=location, axis=axis, eps=eps, ddof=ddof),
    )
