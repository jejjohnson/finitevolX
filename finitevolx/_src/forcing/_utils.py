"""Private helpers shared by the forcing module operators."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float


def interior_or_scalar(
    x: float | Float[Array, "..."],
) -> Float[Array, "..."]:
    """Interior slice ``[..., 1:-1, 1:-1]`` of a field; scalars pass through.

    Parameters
    ----------
    x : float or Float[Array, "... Ny Nx"]
        Scalar coefficient or a full-grid field (ghost ring included).

    Returns
    -------
    Float[Array, "..."]
        The scalar unchanged, or the interior of the field.
    """
    x = jnp.asarray(x)
    if x.ndim == 0:
        return x
    return x[..., 1:-1, 1:-1]


def on_face_interior(
    x: float | Float[Array, "Ny Nx"],
    t_to_face: Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]],
) -> Float[Array, "..."]:
    """Interior face values of a T-point coefficient; scalars pass through.

    Parameters
    ----------
    x : float or Float[Array, "Ny Nx"]
        Scalar coefficient or a T-point field (ghost ring filled).
    t_to_face : Callable
        T -> face interpolation, e.g. ``Interpolation2D.T_to_U``.

    Returns
    -------
    Float[Array, "..."]
        The scalar unchanged, or ``t_to_face(x)[1:-1, 1:-1]``.
    """
    x = jnp.asarray(x)
    if x.ndim == 0:
        return x
    return t_to_face(x)[1:-1, 1:-1]


def safe_speed(
    a: Float[Array, "..."],
    b: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Exact ``sqrt(a**2 + b**2)`` with finite (zero) gradient at rest.

    The double-``where`` keeps ``jax.grad`` from evaluating the infinite
    derivative of ``sqrt`` at zero, so quadratic drag stays differentiable
    for fluid at rest.
    """
    s2 = a * a + b * b
    positive = s2 > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, s2, 1.0)), 0.0)
