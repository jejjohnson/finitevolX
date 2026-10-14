"""
Functional reconstruction dispatcher for upwind face-value computation.

Provides a stateless, pure-function interface for computing upwind face
fluxes on Arakawa C-grids.  This is the functional counterpart of the
:class:`~finitevolx.Reconstruction2D` class — it requires no grid object
and works on raw arrays with ghost-cell convention.

The top-level entry point is :func:`reconstruct`, which dispatches to
lower-level ``upwind_*`` helpers based on ``method`` and ``num_pts``.

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

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.advection.linear import (
    linear_3pts_left,
    linear_3pts_right,
    linear_5pts_left,
    linear_5pts_right,
)
from finitevolx._src.advection.weno import (
    weno_3pts,
    weno_3pts_improved,
    weno_3pts_improved_right,
    weno_3pts_right,
    weno_5pts,
    weno_5pts_improved,
    weno_5pts_improved_right,
    weno_5pts_right,
)
from finitevolx._src.operators._ghost import interior


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
    return jnp.maximum(u, 0.0), jnp.minimum(u, 0.0)


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
    # Interior faces: between cells 1..N-2 and 2..N-1
    return jnp.where(u >= 0.0, q[..., 1:-1], q[..., 2:])


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
    left = _stencil(_STENCILS_3PT, 3, method, 0)
    right = _stencil(_STENCILS_3PT, 3, method, 1)
    # j = 1..N-2 (N-2 faces):  h[j+1/2] = L(q[j-1], q[j], q[j+1])
    h_pos = left(q[..., :-2], q[..., 1:-1], q[..., 2:])
    # j = 1..N-3:              h[j+1/2] = R(q[j], q[j+1], q[j+2])
    h_neg_int = right(q[..., 1:-2], q[..., 2:-1], q[..., 3:])
    # j = N-2:                 h[j+1/2] = q[N-1]  (1st-order upwind fallback)
    h_neg = jnp.concatenate([h_neg_int, q[..., -1:]], axis=-1)
    return jnp.where(u >= 0.0, h_pos, h_neg)


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
    h_neg = jnp.concatenate([h_neg_int, h_neg_last], axis=-1)
    return jnp.where(u >= 0.0, h_pos, h_neg)


# ── Top-level dispatcher ─────────────────────────────────────────────────────


def reconstruct(
    q: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    dim: int,
    method: str = "weno",
    num_pts: int = 5,
) -> Float[Array, "Ny Nx"]:
    """Compute upwind face flux along *dim* using the given reconstruction method.

    This is a stateless, pure-function interface for upwind reconstruction.
    It dispatches to :func:`upwind_1pt`, :func:`upwind_3pt`, or
    :func:`upwind_5pt` based on ``num_pts``, with automatic boundary
    fallbacks for higher-order stencils.

    Parameters
    ----------
    q : Float[Array, "Ny Nx"]
        Cell-centred scalar at T-points (includes ghost ring).
    u : Float[Array, "Ny Nx"]
        Face-centred transport velocity at U-points (``dim=1``) or
        V-points (``dim=0``).  Includes ghost ring.
    dim : {0, 1}
        Spatial dimension: ``1`` for x (east-face flux), ``0`` for y
        (north-face flux).
    method : {'weno', 'wenoz', 'linear'}
        Stencil flavour.  ``'weno'`` uses Jiang-Shu weights,
        ``'wenoz'`` uses WENO-Z (Borges et al.), ``'linear'`` uses the
        optimal polynomial weights (no nonlinear limiting).
    num_pts : {1, 3, 5}
        Stencil width (number of cells used per direction).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Face flux ``h_face * u`` with zero ghost ring, same shape as *q*.
        ``flux[j, i]`` represents the flux at the east face (``dim=1``)
        or north face (``dim=0``) of cell ``[j, i]``.

    Notes
    -----
    For ``num_pts >= 3``, boundary faces where the full stencil does not
    fit are reconstructed with a lower-order fallback (3-point falls back
    to 1st-order upwind; 5-point falls back to 3-point, then 1st-order).
    Both flow signs reconstruct the same face, the east (north) face of
    cell ``[j, i]`` at ``i+1/2`` (``j+1/2``); the right-biased stencil for
    ``u < 0`` is shifted one cell east of the left-biased one, exactly as in
    :class:`~finitevolx.Reconstruction2D`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import reconstruct
    >>> Ny, Nx = 10, 10
    >>> q = jnp.ones((Ny, Nx))
    >>> u = jnp.ones((Ny, Nx))
    >>> flux = reconstruct(q, u, dim=1, method="weno", num_pts=5)
    >>> flux.shape
    (10, 10)
    """
    _VALID_METHODS = ("weno", "wenoz", "linear")
    if dim not in (0, 1):
        raise ValueError(f"dim must be 0 (y) or 1 (x), got {dim!r}")
    if num_pts not in (1, 3, 5):
        raise ValueError(f"num_pts must be 1, 3, or 5, got {num_pts!r}")
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    if dim == 1:
        # x-direction: operate on interior rows along last axis
        # q[1:-1, :] has shape (Ny-2, Nx); reconstruct returns face fluxes
        # of shape (Ny-2, Nx-2) — interior values only
        flux_vals = _reconstruct_last_axis(q[1:-1, :], u[1:-1, :], method, num_pts)
        return interior(flux_vals, q)
    else:
        # y-direction: swap to operate along last axis, then swap back
        flux_vals_swapped = _reconstruct_last_axis(
            jnp.swapaxes(q[:, 1:-1], 0, 1),
            jnp.swapaxes(u[:, 1:-1], 0, 1),
            method,
            num_pts,
        )
        flux_vals = jnp.swapaxes(flux_vals_swapped, 0, 1)
        return interior(flux_vals, q)


def _reconstruct_last_axis(
    q: Float[Array, "..."],
    u: Float[Array, "..."],
    method: str,
    num_pts: int,
) -> Float[Array, "..."]:
    """Reconstruct face flux along the last axis.

    Parameters
    ----------
    q : array, shape (..., N)
        Scalar along reconstruction axis (includes ghost cells).
    u : array, shape (..., N)
        Velocity at faces (includes ghost cells).

    Returns
    -------
    array, shape (..., N-2)
        Interior face flux values (``h_face * u`` at each interior face).
    """
    vel = u[..., 1:-1]  # interior face velocities, shape (..., N-2)

    if num_pts == 1:
        h_face = upwind_1pt(q, vel)
        return h_face * vel

    if num_pts == 3:
        h_face = upwind_3pt(q, vel, method)
        return h_face * vel

    # num_pts == 5: need boundary fallbacks
    N = q.shape[-1]
    if N < 5:
        # Not enough cells for 5-point; fall back to 3-point
        h_face = upwind_3pt(q, vel, method)
        return h_face * vel

    # Faces are indexed by the cell j to their west: face j+1/2, j = 1..N-2.
    # 5-point interior (j = 2..N-3)
    h5 = upwind_5pt(q, vel[..., 1:-1], method)

    left3 = _stencil(_STENCILS_3PT, 3, method, 0)
    right5 = _stencil(_STENCILS_5PT, 5, method, 1)

    # First face (j = 1): positive flow 3pt left (q[0..2]); negative flow has
    # the full right-biased 5pt stencil (q[0..4]).
    h_first_pos = left3(q[..., 0:1], q[..., 1:2], q[..., 2:3])
    h_first_neg = right5(
        q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4], q[..., 4:5]
    )
    h3_first = jnp.where(vel[..., :1] >= 0.0, h_first_pos, h_first_neg)

    # Last face (j = N-2): positive flow 3pt left (q[N-3..N-1]); negative flow
    # has no cell j+2, so 1st-order upwind q[N-1].
    h_last_pos = left3(q[..., -3:-2], q[..., -2:-1], q[..., -1:])
    h3_last = jnp.where(vel[..., -1:] >= 0.0, h_last_pos, q[..., -1:])

    h_face = jnp.concatenate([h3_first, h5, h3_last], axis=-1)
    return h_face * vel
