"""Shared utilities for spherical operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

_COS_EPS = 1e-12  # guard against division by cos(lat) ≈ 0 near poles


def _metric_like(
    metric: Float[Array, "..."], like: Float[Array, "..."]
) -> Float[Array, "..."]:
    """A grid metric array in the floating dtype of the field it acts on.

    Grid coordinate arrays are float64 NumPy arrays: multiplying one into a
    float32 field would promote the result to float64 (under x64 on JAX).
    The metric is a property of the grid, not a precision request, so it
    takes the field's dtype and namespace instead. A non-floating ``like``
    leaves the metric's dtype alone.

    Parameters
    ----------
    metric : array
        A grid metric, e.g. ``grid.cos_lat_T``.
    like : array
        The field the metric scales.

    Returns
    -------
    array
        ``metric`` as an array of ``like``'s namespace and floating dtype.
    """
    xp = array_namespace(like)
    dtype = xp.asarray(like).dtype
    if not xp.isdtype(dtype, "real floating"):
        return xp.asarray(metric)
    return xp.asarray(metric, dtype=dtype)


def _safe_div_cos(
    numerator: Float[Array, "..."],
    cos_val: Float[Array, "..."],
    scale: float | Float[Array, "..."],
) -> Float[Array, "..."]:
    """Compute ``numerator / (scale * cos_val)`` with pole guard.

    Returns NaN where ``|cos_val| < eps`` instead of Inf.

    Parameters
    ----------
    numerator : array
        Numerator of the division.
    cos_val : array
        cos(latitude) values; near-zero values trigger the pole guard.
    scale : float or array
        Additional scale factor in the denominator.

    Returns
    -------
    array
        Result with NaN at pole-adjacent points.
    """
    xp = array_namespace(numerator, cos_val)
    denom = scale * cos_val
    safe_denom = xp.where(xp.abs(cos_val) < _COS_EPS, 1.0, denom)
    result = numerator / safe_denom
    return xp.where(xp.abs(cos_val) < _COS_EPS, xp.nan, result)
