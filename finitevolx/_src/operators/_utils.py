"""Shared utilities for spherical operators."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

_COS_EPS = 1e-12  # guard against division by cos(lat) ≈ 0 near poles


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
