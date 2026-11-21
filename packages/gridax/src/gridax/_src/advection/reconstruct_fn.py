"""
Functional reconstruction dispatcher for upwind face-value computation.

Provides a stateless, pure-function interface for computing upwind face
fluxes on Arakawa C-grids.  This is the functional counterpart of the
:class:`~finitevolx.Reconstruction2D` class — it requires no grid object
and works on raw arrays with ghost-cell convention.

The axis-generic entry point is :func:`gridax.advection.reconstruct` (#464),
which also serves the old ``reconstruct(q, u, dim, method, num_pts)`` call
with a deprecation warning.

All arrays follow the finitevolX ghost-cell convention:

* 2-D:  shape ``[Ny, Nx]`` with a 1-cell ghost ring.
  Interior data lives at ``[1:-1, 1:-1]``.
* 1-D:  shape ``[Nx]`` with 1 ghost cell on each side.

References
----------
Shu, C.-W. (1998). Essentially Non-Oscillatory and Weighted Essentially
Non-Oscillatory Schemes for Hyperbolic Conservation Laws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.advection.linear import (
    linear_3pts_left,
    linear_3pts_right,
    linear_5pts_left,
    linear_5pts_right,
)
from gridax._src.advection.weno import (
    weno_3pts,
    weno_3pts_improved,
    weno_3pts_improved_right,
    weno_3pts_right,
    weno_5pts,
    weno_5pts_improved,
    weno_5pts_improved_right,
    weno_5pts_right,
)
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


def plusminus(
    u: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Split velocity into positive and negative parts.

    Parameters
    ----------
    u : array
        Velocity field.

    Returns
    -------
    u_pos, u_neg : tuple of arrays
        ``u_pos = max(u, 0)`` and ``u_neg = min(u, 0)``, so that
        ``u = u_pos + u_neg``.
    """
    xp = array_namespace(u)
    return xp.maximum(u, 0.0), xp.minimum(u, 0.0)


# ── Low-level upwind helpers (operate on interior slices) ─────────────────────


def upwind_1pt(q: Float[Array, "..."], u: Float[Array, "..."]) -> Float[Array, "..."]:
    """1st-order upwind face value at interior faces along the last axis.

    Parameters
    ----------
    q : array, shape (..., N)
        Scalar field along reconstruction axis (includes ghost cells).
    u : array, shape (..., N-2)
        Velocity at interior faces (between cell centres, excluding ghost faces).

    Returns
    -------
    array, shape (..., N-2)
        Face values ``h_face[i+1/2]``, **not** multiplied by velocity.
    """
    xp = array_namespace(q, u)
    # Interior faces: between cells 1..N-2 and 2..N-1
    return xp.where(u >= 0.0, q[..., 1:-1], q[..., 2:])


# Pointwise stencils: ``method`` -> (left-biased, right-biased) face value.
# Left-biased  (positive flow) at face i+1/2 reads cells i-1..i+1 (3pt) / i-2..i+2 (5pt).
# Right-biased (negative flow) at face i+1/2 reads cells i..i+2   (3pt) / i-1..i+3 (5pt).
_STENCILS_3PT = {
    "weno": (weno_3pts, weno_3pts_right),
    "wenoz": (weno_3pts_improved, weno_3pts_improved_right),
    "linear": (linear_3pts_left, linear_3pts_right),
}
_STENCILS_5PT = {
    "weno": (weno_5pts, weno_5pts_right),
    "wenoz": (weno_5pts_improved, weno_5pts_improved_right),
    "linear": (linear_5pts_left, linear_5pts_right),
}


def _stencil(table: dict, width: int, method: str, side: int):
    """Look up the (left, right) stencil function; ``side`` 0 = left, 1 = right."""
    if method not in table:
        raise ValueError(f"Unknown {width}pt method: {method!r}")
    return table[method][side]


def upwind_3pt(
    q: Float[Array, "..."],
    u: Float[Array, "..."],
    method: str = "weno",
) -> Float[Array, "..."]:
    """3-point upwind face value at i+1/2 along the last axis.

    Output ``k`` is the face between cells ``k+1`` and ``k+2`` of ``q``
    (the east face ``j+1/2`` of interior cell ``j = k+1``), for both flow
    signs:

    * ``u >= 0``: ``stencil_left(q[j-1], q[j], q[j+1])``
    * ``u <  0``: ``stencil_right(q[j], q[j+1], q[j+2])``, with 1st-order
      upwind ``q[N-1]`` at the last face where ``j+2`` does not exist.

    Parameters
    ----------
    q : array
        Scalar field along reconstruction axis (length N).
    u : array
        Velocity at faces (length N-2, interior faces only).
    method : {'weno', 'wenoz', 'linear'}
        Reconstruction method.

    Returns
    -------
    array, shape (..., N-2)
        Face values at interior faces.
    """
    xp = array_namespace(q, u)
    left = _stencil(_STENCILS_3PT, 3, method, 0)
    right = _stencil(_STENCILS_3PT, 3, method, 1)
    # j = 1..N-2 (N-2 faces):  h[j+1/2] = L(q[j-1], q[j], q[j+1])
    h_pos = left(q[..., :-2], q[..., 1:-1], q[..., 2:])
    # j = 1..N-3:              h[j+1/2] = R(q[j], q[j+1], q[j+2])
    h_neg_int = right(q[..., 1:-2], q[..., 2:-1], q[..., 3:])
    # j = N-2:                 h[j+1/2] = q[N-1]  (1st-order upwind fallback)
    h_neg = xp.concatenate([h_neg_int, q[..., -1:]], axis=-1)
    return xp.where(u >= 0.0, h_pos, h_neg)


def upwind_5pt(
    q: Float[Array, "..."],
    u: Float[Array, "..."],
    method: str = "weno",
) -> Float[Array, "..."]:
    """5-point upwind face value at i+1/2 along the last axis.

    Output ``k`` is the face between cells ``k+2`` and ``k+3`` of ``q``
    (the east face ``j+1/2`` of cell ``j = k+2``), for both flow signs:

    * ``u >= 0``: ``stencil_left(q[j-2], ..., q[j+2])``
    * ``u <  0``: ``stencil_right(q[j-1], ..., q[j+3])``, with the 3-point
      right-biased stencil ``(q[j], q[j+1], q[j+2])`` at the last face
      ``j = N-3`` where ``j+3`` does not exist.

    Parameters
    ----------
    q : array
        Scalar field along reconstruction axis (length N).
    u : array
        Velocity at faces (length N-4, deep interior faces only).
    method : {'weno', 'wenoz', 'linear'}
        Reconstruction method.

    Returns
    -------
    array, shape (..., N-4)
        Face values at deep interior faces (requires 2-cell padding on each side).
    """
    xp = array_namespace(q, u)
    left = _stencil(_STENCILS_5PT, 5, method, 0)
    right = _stencil(_STENCILS_5PT, 5, method, 1)
    right3 = _stencil(_STENCILS_3PT, 3, method, 1)
    # j = 2..N-3 (N-4 faces):  h[j+1/2] = L(q[j-2], ..., q[j+2])
    h_pos = left(q[..., :-4], q[..., 1:-3], q[..., 2:-2], q[..., 3:-1], q[..., 4:])
    # j = 2..N-4:              h[j+1/2] = R(q[j-1], ..., q[j+3])
    h_neg_int = right(
        q[..., 1:-4], q[..., 2:-3], q[..., 3:-2], q[..., 4:-1], q[..., 5:]
    )
    # j = N-3:                 h[j+1/2] = R3(q[j], q[j+1], q[j+2])
    h_neg_last = right3(q[..., -3:-2], q[..., -2:-1], q[..., -1:])
    h_neg = xp.concatenate([h_neg_int, h_neg_last], axis=-1)
    return xp.where(u >= 0.0, h_pos, h_neg)
