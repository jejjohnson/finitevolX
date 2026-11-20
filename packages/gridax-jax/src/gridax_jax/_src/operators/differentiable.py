"""Differentiable surrogates for non-smooth operations.

Adjoints break wherever ``jnp.abs``, ``jnp.maximum``, or a hard clamp appears:
those have a zero or undefined gradient at the kink, which corrupts the
reverse-mode pass that differentiable data-assimilation (4DVar, BFN) relies on.

This module provides smooth, everywhere-differentiable surrogates:

* :func:`smooth_abs` — ``sqrt(x**2 + eps**2)``, a rounded ``|x|`` used for
  wave speeds (e.g. the Rusanov flux, :func:`finitevolx.rusanov_flux`) and
  reconstruction smoothness weights.
* :func:`smooth_clamp` — a softplus-based ``max(x, x_min)`` with a nonzero
  gradient everywhere, used to keep layer thicknesses positive.
* :func:`smooth_max` — the generic two-argument variant of the above.

All three are pure ``jax.numpy`` and compose with ``jit`` / ``vmap`` / ``grad``.
The ``eps`` / ``sharpness`` knobs trade approximation error against gradient
conditioning: smaller ``eps`` (larger ``sharpness``) is closer to the
non-smooth function but has a stiffer gradient near the kink.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def smooth_abs(
    x: Float[Array, "..."],
    eps: float = 1e-8,
) -> Float[Array, "..."]:
    r"""Smooth approximation of ``|x|`` with a nonzero gradient at the origin.

    Computes :math:`\sqrt{x^2 + \varepsilon^2}`. Unlike :func:`jax.numpy.abs`,
    whose gradient is undefined at ``x = 0``, this has the well-defined
    derivative :math:`x / \sqrt{x^2 + \varepsilon^2}` everywhere, so it is safe
    inside a differentiated RHS (e.g. the dissipation term of a Rusanov flux).

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array.
    eps : float, optional
        Rounding scale. As ``eps -> 0`` the result approaches ``|x|``; the
        maximum deviation is ``eps`` (attained at ``x = 0``). Default ``1e-8``.

    Returns
    -------
    Float[Array, "..."]
        ``sqrt(x**2 + eps**2)``, same shape as ``x``.
    """
    return jnp.sqrt(x * x + eps * eps)


def smooth_clamp(
    x: Float[Array, "..."],
    x_min: float | Float[Array, "..."],
    sharpness: float = 10.0,
) -> Float[Array, "..."]:
    r"""Smooth approximation of ``max(x, x_min)`` (a soft lower clamp).

    Computes :math:`x_{\min} + \mathrm{softplus}\big((x - x_{\min})\,s\big) / s`
    with sharpness ``s``. The gradient w.r.t. ``x`` is the logistic
    :math:`\sigma\big((x - x_{\min})\,s\big)`, which is strictly positive
    everywhere — so, unlike ``jnp.maximum(x, x_min)`` (zero gradient below the
    clamp) or a hard clip, this never zeroes the backward pass. Used to keep a
    layer thickness ``h`` strictly positive without killing its sensitivity.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input array.
    x_min : float or Float[Array, "..."]
        Lower bound. The result is always ``> x_min`` and approaches
        ``max(x, x_min)`` as ``sharpness -> inf``.
    sharpness : float, optional
        Transition sharpness ``s``. Larger is closer to the hard clamp but
        stiffer near ``x = x_min``. Default ``10.0``.

    Returns
    -------
    Float[Array, "..."]
        Smoothly lower-clamped ``x``.
    """
    return x_min + jax.nn.softplus((x - x_min) * sharpness) / sharpness


def smooth_max(
    x: Float[Array, "..."],
    y: float | Float[Array, "..."],
    sharpness: float = 10.0,
) -> Float[Array, "..."]:
    r"""Smooth approximation of the elementwise ``max(x, y)``.

    The two-argument generalisation of :func:`smooth_clamp`:
    :math:`y + \mathrm{softplus}\big((x - y)\,s\big) / s`. Symmetric up to the
    softplus identity ``softplus(z) = z + softplus(-z)``; both gradients are
    strictly positive logistics, so neither argument's sensitivity is lost.

    Parameters
    ----------
    x, y : Float[Array, "..."] or float
        Inputs (broadcast together).
    sharpness : float, optional
        Transition sharpness. Default ``10.0``.

    Returns
    -------
    Float[Array, "..."]
        Smooth elementwise maximum of ``x`` and ``y``.
    """
    return y + jax.nn.softplus((x - y) * sharpness) / sharpness
