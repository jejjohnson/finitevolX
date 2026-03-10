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
    and approaches 2 for r → ∞.

    Parameters
    ----------
    r : array
        Ratio of consecutive upwind differences.

    Returns
    -------
    array
        Limiter values φ(r) ∈ [0, 2).
    """
    r_abs = jnp.abs(r)
    return (r + r_abs) / (1.0 + r_abs)


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
