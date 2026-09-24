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


def safe_denominator(
    x: Float[Array, "..."],
    face_mask: Array | None,
) -> Float[Array, "..."]:
    """Replace a face denominator by ``1`` at dry faces.

    Dry-cell thicknesses are often zero, and ``0 / 0 = NaN`` survives a
    later multiplicative mask (``NaN * 0 = NaN``), so masked operators swap
    in a harmless value before dividing.

    Parameters
    ----------
    x : Float[Array, "..."]
        Denominator on the face interior ``[Ny-2, Nx-2]`` (or a scalar).
    face_mask : Array or None
        Full-grid ``[Ny, Nx]`` face mask, or ``None`` for no masking.

    Returns
    -------
    Float[Array, "..."]
        ``x`` at wet faces and ``1`` at dry faces.
    """
    if face_mask is None:
        return x
    return jnp.where(face_mask[1:-1, 1:-1], x, 1.0)


def mask_where(arr: Float[Array, "..."], mask: Array | None) -> Float[Array, "..."]:
    """Zero ``arr`` where ``mask`` is false, even if ``arr`` is NaN/Inf there.

    Unlike ``arr * mask``, ``jnp.where`` does not propagate non-finite
    values from dry points (e.g. NaN-filled land in the input fields).
    """
    if mask is None:
        return arr
    return jnp.where(mask, arr, 0.0)
