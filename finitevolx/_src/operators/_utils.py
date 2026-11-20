"""Shared utilities for spherical operators."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

_COS_EPS = 1e-12  # guard against division by cos(lat) ≈ 0 near poles


def _metric_like(
    metric: Float[Array, "..."], like: Float[Array, "..."]
) -> Float[Array, "..."]:
    """A grid metric array in the floating dtype of the field it acts on.

    Grid coordinate arrays are float64 NumPy arrays, which JAX treats as
    strongly typed: multiplying one into a float32 field would promote the
    result to float64 under x64. The metric is a property of the grid, not
    a precision request, so it takes the field's dtype instead (as the
    weakly typed ``jnp`` arrays the grids used to hold did). A non-floating
    ``like`` leaves the metric's dtype alone.

    Parameters
    ----------
    metric : array
        A grid metric, e.g. ``grid.cos_lat_T``.
    like : array
        The field the metric scales.

    Returns
    -------
    array
        ``metric`` as a JAX array of ``like``'s floating dtype.
    """
    dtype = jnp.result_type(like)
    if not jnp.issubdtype(dtype, jnp.floating):
        return jnp.asarray(metric)
    return jnp.asarray(metric, dtype=dtype)


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
    denom = scale * cos_val
    safe_denom = jnp.where(jnp.abs(cos_val) < _COS_EPS, 1.0, denom)
    result = numerator / safe_denom
    return jnp.where(jnp.abs(cos_val) < _COS_EPS, jnp.nan, result)
