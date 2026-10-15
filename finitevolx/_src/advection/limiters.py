"""Flux limiter functions for TVD (Total Variation Diminishing) reconstruction.

Each function takes the ratio of consecutive upwind differences

    r = Δ_upwind / Δ_downwind

and returns a limiter value φ(r) ∈ [0, 2] that blends between 1st-order
upwind (φ = 0) and an anti-diffusive correction.

TVD reconstruction at east face i+1/2:

    Positive flow (u ≥ 0):
        r   = (h[i] − h[i−1]) / (h[i+1] − h[i])
        h_L = h[i]   + ½ φ(r) (h[i+1] − h[i])

    Negative flow (u < 0):
        r   = (h[i+1] − h[i+2]) / (h[i] − h[i+1])
        h_R = h[i+1] + ½ φ(r) (h[i]   − h[i+1])

References
----------
LeVeque, R. J., "Finite Volume Methods for Hyperbolic Problems", Cambridge,
2002, §6.6.
Sweby, P. K., "High resolution schemes using flux limiters for hyperbolic
conservation laws", SIAM J. Numer. Anal. 21 (1984), 995-1011.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def minmod(r: Float[Array, ...]) -> Float[Array, ...]:
    """Minmod flux limiter.

    φ(r) = max(0, min(1, r))

    The most diffusive TVD limiter; uses only the minimum of the two
    consecutive slopes.  Equivalent to the most restrictive TVD limiter.

    Parameters
    ----------
    r : array
        Ratio of consecutive upwind differences.

    Returns
    -------
    array
        Limiter values φ(r) ∈ [0, 1].
    """
    return jnp.maximum(0.0, jnp.minimum(1.0, r))


def van_leer(r: Float[Array, ...]) -> Float[Array, ...]:
    """Van Leer flux limiter.

    φ(r) = (r + |r|) / (1 + |r|)

    A smooth limiter that is symmetric around r = 1.  Returns 0 for r ≤ 0
    and approaches 2 for r → ∞.  Evaluated as ``2r / (1 + r)`` for
    ``0 < r <= 1`` and ``2 / (1 + 1/r)`` for ``r > 1``, each branch on a
    guarded argument, so ``r = ±inf`` gives 2 / 0 (not ``inf / inf``) and the
    gradient is finite for every ``r``, tiny positive ``r`` included (no
    ``1 / r²``).

    Parameters
    ----------
    r : array
        Ratio of consecutive upwind differences.

    Returns
    -------
    array
        Limiter values φ(r) ∈ [0, 2).
    """
    low = (r > 0.0) & (r <= 1.0)
    high = r > 1.0
    r_low = jnp.where(low, r, 0.0)  # φ = 2r / (1 + r),   0 < r <= 1
    r_high = jnp.where(high, r, 2.0)  # φ = 2 / (1 + 1/r),  r > 1
    return jnp.where(
        low,
        2.0 * r_low / (1.0 + r_low),
        jnp.where(high, 2.0 / (1.0 + 1.0 / r_high), 0.0),
    )


def superbee(r: Float[Array, ...]) -> Float[Array, ...]:
    """Superbee flux limiter.

    φ(r) = max(0, max(min(2r, 1), min(r, 2)))

    The most compressive TVD limiter; produces the sharpest fronts but can
    introduce steepening artefacts on smooth solutions.

    Parameters
    ----------
    r : array
        Ratio of consecutive upwind differences.

    Returns
    -------
    array
        Limiter values φ(r) ∈ [0, 2].
    """
    return jnp.maximum(
        0.0,
        jnp.maximum(jnp.minimum(2.0 * r, 1.0), jnp.minimum(r, 2.0)),
    )


def mc(r: Float[Array, ...]) -> Float[Array, ...]:
    """Monotonized Central (MC) flux limiter.

    φ(r) = max(0, min((1 + r)/2, 2r, 2))

    A second-order accurate limiter that is less compressive than Superbee
    and smoother than Minmod.  Sometimes called the MC limiter or
    Monotonized Central-difference limiter.

    Parameters
    ----------
    r : array
        Ratio of consecutive upwind differences.

    Returns
    -------
    array
        Limiter values φ(r) ∈ [0, 2].
    """
    return jnp.maximum(
        0.0,
        jnp.minimum(jnp.minimum((1.0 + r) / 2.0, 2.0 * r), 2.0),
    )


# ----------------------------------------------------------------------
# Limited slopes: φ(Δ_up / Δ_down) · Δ_down without the division (#294)
# ----------------------------------------------------------------------
#
# The TVD reconstructions need only ``φ(r) · Δ_down`` with
# ``r = Δ_up / Δ_down``.  For the four limiters that product has a closed
# form in the two differences (``a = Δ_up``, ``b = Δ_down``): zero unless
# ``a`` and ``b`` have the same sign, otherwise ``sgn(b)`` times
#
#   minmod    min(|a|, |b|)
#   superbee  max(min(2|a|, |b|), min(|a|, 2|b|))
#   mc        min((|a| + |b|) / 2, 2|a|, 2|b|)
#   van Leer  2 |a| |b| / (|a| + |b|)
#
# These are exactly φ(a / b) · b (multiply each φ by |b| > 0), need no
# epsilon, are exactly amplitude-covariant (homogeneous of degree 1), and
# return exactly 0 — the upwind value — whenever the signs differ.  The sign
# test uses ``sgn(a) sgn(b)``, not ``a b`` (which underflows to 0 at small
# amplitude).


def _same_sign(a: Array, b: Array) -> Array:
    return jnp.sign(a) * jnp.sign(b) > 0.0


def _minmod_slope(a: Array, b: Array) -> Array:
    """φ_minmod(a/b) · b = sgn(b) · min(|a|, |b|) if ab > 0, else 0."""
    return jnp.where(
        _same_sign(a, b), jnp.sign(b) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0
    )


def _superbee_slope(a: Array, b: Array) -> Array:
    """φ_superbee(a/b) · b = sgn(b) · max(min(2|a|, |b|), min(|a|, 2|b|))."""
    abs_a, abs_b = jnp.abs(a), jnp.abs(b)
    mag = jnp.maximum(jnp.minimum(2.0 * abs_a, abs_b), jnp.minimum(abs_a, 2.0 * abs_b))
    return jnp.where(_same_sign(a, b), jnp.sign(b) * mag, 0.0)


def _mc_slope(a: Array, b: Array) -> Array:
    """φ_mc(a/b) · b = sgn(b) · min((|a| + |b|)/2, 2|a|, 2|b|)."""
    abs_a, abs_b = jnp.abs(a), jnp.abs(b)
    mag = jnp.minimum(0.5 * (abs_a + abs_b), 2.0 * jnp.minimum(abs_a, abs_b))
    return jnp.where(_same_sign(a, b), jnp.sign(b) * mag, 0.0)


def _van_leer_slope(a: Array, b: Array) -> Array:
    """φ_vanleer(a/b) · b = 2ab / (a + b) if ab > 0, else 0.

    Written ``2 a · (b k) / ((a + b) k)`` with ``k = 1 / max(|a|, |b|)``
    held out of the gradient (exact: the ratio does not depend on ``k``), so
    the divisor lies in ``[1, 2]`` and the gradient cannot overflow at small
    amplitude.
    """
    same = _same_sign(a, b)
    scale = jax.lax.stop_gradient(jnp.maximum(jnp.abs(a), jnp.abs(b)))
    k = 1.0 / jnp.where(scale > 0.0, scale, 1.0)
    den = jnp.where(same, (a + b) * k, 1.0)
    return jnp.where(same, 2.0 * a * (b * k) / den, 0.0)


#: ``limiter name -> (a, b) -> φ(a / b) · b``.
_LIMITED_SLOPES = {
    "minmod": _minmod_slope,
    "van_leer": _van_leer_slope,
    "superbee": _superbee_slope,
    "mc": _mc_slope,
}
