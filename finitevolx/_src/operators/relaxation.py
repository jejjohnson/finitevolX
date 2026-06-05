"""
Linear drag and Rayleigh-sponge relaxation tendency operators.

Both operators express the same first-order linear-restoring pattern that
shows up across ocean models as bottom drag, sponge / sponge-layer damping,
boundary-condition nudging, and tracer restoring:

.. math::

    \\partial_t X \\mathrel{+}= -\\,\\text{coef}\\cdot W \\cdot (X - X_\\text{ref}),

with a spatial weight ``W`` in ``[0, 1]``.  :func:`linear_drag` is the
``X_ref = 0`` special case applied on a single (bottom) layer; the general
weighted form is :func:`rayleigh_relaxation`.

Following the finitevolX layering rule (#209), these are mask-free functional
helpers: spatial selectivity is expressed through the ``weight`` map (for
relaxation) or the ``layer`` index (for drag), not through a ``Mask2D``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def linear_drag(
    u: Float[Array, "..."],
    v: Float[Array, "..."],
    *,
    coef: float | Float[Array, "..."],
    layer: int = -1,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    r"""Rayleigh (linear) bottom-drag tendency :math:`-\text{coef}\cdot u`.

    Returns the drag contribution to the momentum tendency.  For a stacked
    multi-layer field (layer axis ``-3``, shape ``[..., Nz, Ny, Nx]``) the
    drag is applied **only** on the selected ``layer`` (the deepest layer by
    default); every other layer receives an exact zero.  For a single-layer
    2-D field (shape ``[Ny, Nx]``) the ``layer`` argument is ignored and the
    drag acts on the whole field.

    Parameters
    ----------
    u : Float[Array, "..."]
        x-velocity at U-points.
    v : Float[Array, "..."]
        y-velocity at V-points.
    coef : float or Float[Array, "..."]
        Linear drag coefficient ``r`` (units of inverse time).  Scalar or a
        field broadcastable to the per-layer slice.
    layer : int, optional
        Index of the layer the drag acts on along axis ``-3`` (multi-layer
        inputs only).  Default ``-1`` (deepest layer).

    Returns
    -------
    tuple[Float[Array, "..."], Float[Array, "..."]]
        ``(du, dv)`` drag tendencies, same shapes as ``u`` and ``v``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import linear_drag
    >>> u = jnp.ones((3, 8, 8))  # 3 layers
    >>> du, dv = linear_drag(u, u, coef=1e-2)
    >>> bool((du[0] == 0).all()), bool((du[-1] != 0).any())
    (True, True)
    """
    if u.ndim < 3:
        return -coef * u, -coef * v

    idx = (..., layer, slice(None), slice(None))
    du = jnp.zeros_like(u).at[idx].set(-coef * u[idx])
    dv = jnp.zeros_like(v).at[idx].set(-coef * v[idx])
    return du, dv


def rayleigh_relaxation(
    x: Float[Array, "..."],
    x_ref: Float[Array, "..."],
    *,
    coef: float | Float[Array, "..."],
    weight: float | Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Sponge / relaxation tendency :math:`-\text{coef}\cdot W\cdot(x - x_\text{ref})`.

    The single linear-restoring operator behind sponge layers, boundary-
    condition forcing, back-and-forth-nudging (BFN), and tracer restoring.
    The tendency relaxes ``x`` toward the reference ``x_ref`` at rate ``coef``,
    modulated by a spatial weight map ``weight`` (typically a smooth ``[0, 1]``
    taper that is one inside the sponge and zero in the interior).

    Parameters
    ----------
    x : Float[Array, "..."]
        Current field.
    x_ref : Float[Array, "..."]
        Reference / target field, broadcastable to ``x``.
    coef : float or Float[Array, "..."]
        Relaxation rate ``gamma`` (inverse time).
    weight : float or Float[Array, "..."]
        Spatial weight ``W`` in ``[0, 1]``, broadcastable to ``x``.  Use a
        full-domain ``1.0`` for uniform restoring, or a boundary taper for a
        sponge layer.

    Returns
    -------
    Float[Array, "..."]
        Relaxation tendency, same shape as ``x``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import rayleigh_relaxation
    >>> ssh = jnp.zeros((8, 8))
    >>> ssh_bc = jnp.ones((8, 8))
    >>> dssh = rayleigh_relaxation(ssh, ssh_bc, coef=1e-3, weight=1.0)
    >>> bool((dssh > 0).all())  # relaxes upward toward ssh_bc
    True
    """
    return -coef * weight * (x - x_ref)
