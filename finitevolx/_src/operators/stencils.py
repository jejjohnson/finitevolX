"""Raw finite-difference and averaging stencils for Arakawa C-grids.

These functions compute the pure index arithmetic of C-grid stencils without
any metric scaling (no division by dx, dy, etc.) or ghost-ring padding.
They return arrays sized to the **interior** of the output grid location.

Use these as building blocks for custom operators on any coordinate system —
Cartesian, spherical, cylindrical, or curvilinear — by applying the
appropriate metric scale factors to the result.

All functions are pure, stateless, and compatible with ``jax.jit``,
``jax.vmap``, and ``jax.grad``.

Example
-------
>>> from finitevolx import diff_x_fwd, interior
>>> raw = diff_x_fwd(h)  # pure index arithmetic
>>> scaled = raw / my_custom_metric  # user applies their own scaling
>>> result = interior(scaled, h)  # pad back to full grid shape
"""

from typing import NamedTuple

import jax
from jaxtyping import Array, Float

# =====================================================================
# 1-D difference stencils
# =====================================================================


def diff_x_fwd_1d(h: Float[Array, "Nx"]) -> Float[Array, "Nx-2"]:
    """Forward difference in x (centre → east face), 1-D.

    Δx h[i+½] = h[i+1] − h[i]

    Maps T-points → U-points.

    Parameters
    ----------
    h : Float[Array, "Nx"]
        Field on T-points, including ghost cells.

    Returns
    -------
    Float[Array, "Nx-2"]
        Raw difference at interior U-points.

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dx``,
    ``R·cos(φ)·dλ``, or whatever metric is appropriate.
    """
    return h[2:] - h[1:-1]


def diff_x_bwd_1d(h: Float[Array, "Nx"]) -> Float[Array, "Nx-2"]:
    """Backward difference in x (east face → centre), 1-D.

    Δx h[i] = h[i+½] − h[i−½]

    Maps U-points → T-points.

    Parameters
    ----------
    h : Float[Array, "Nx"]
        Field on U-points, including ghost cells.

    Returns
    -------
    Float[Array, "Nx-2"]
        Raw difference at interior T-points.

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dx``,
    ``R·cos(φ)·dλ``, or whatever metric is appropriate.
    """
    return h[1:-1] - h[:-2]


# =====================================================================
# 2-D difference stencils
# =====================================================================


def diff_x_fwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Forward difference in x (centre → east face).

    Δx h[j, i+½] = h[j, i+1] − h[j, i]

    Maps T-points → U-points (or V-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (T or V), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Raw difference at interior destination points (U or X).

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dx``,
    ``R·cos(φ)·dλ``, or whatever metric is appropriate.
    """
    return h[1:-1, 2:] - h[1:-1, 1:-1]


def diff_y_fwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Forward difference in y (centre → north face).

    Δy h[j+½, i] = h[j+1, i] − h[j, i]

    Maps T-points → V-points (or U-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (T or U), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Raw difference at interior destination points (V or X).

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dy``,
    ``R·dφ``, or whatever metric is appropriate.
    """
    return h[2:, 1:-1] - h[1:-1, 1:-1]


def diff_x_bwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Backward difference in x (east face → centre).

    Δx h[j, i] = h[j, i+½] − h[j, i−½]

    Maps U-points → T-points (or X-points → V-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (U or X), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Raw difference at interior destination points (T or V).

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dx``,
    ``R·cos(φ)·dλ``, or whatever metric is appropriate.
    """
    return h[1:-1, 1:-1] - h[1:-1, :-2]


def diff_y_bwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Backward difference in y (north face → centre).

    Δy h[j, i] = h[j+½, i] − h[j−½, i]

    Maps V-points → T-points (or X-points → U-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (V or X), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Raw difference at interior destination points (T or U).

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dy``,
    ``R·dφ``, or whatever metric is appropriate.
    """
    return h[1:-1, 1:-1] - h[:-2, 1:-1]


# =====================================================================
# 3-D difference stencils (horizontal plane per z-level)
# =====================================================================


def diff_x_fwd_3d(
    h: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz-2 Ny-2 Nx-2"]:
    """Forward difference in x over all z-levels (centre → east face).

    Δx h[k, j, i+½] = h[k, j, i+1] − h[k, j, i]

    Maps T-points → U-points (or V-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Nz Ny Nx"]
        Field on source points (T or V), including ghost ring.

    Returns
    -------
    Float[Array, "Nz-2 Ny-2 Nx-2"]
        Raw difference at interior destination points (U or X).

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dx``,
    ``R·cos(φ)·dλ``, or whatever metric is appropriate.
    """
    return h[1:-1, 1:-1, 2:] - h[1:-1, 1:-1, 1:-1]


def diff_y_fwd_3d(
    h: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz-2 Ny-2 Nx-2"]:
    """Forward difference in y over all z-levels (centre → north face).

    Δy h[k, j+½, i] = h[k, j+1, i] − h[k, j, i]

    Maps T-points → V-points (or U-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Nz Ny Nx"]
        Field on source points (T or U), including ghost ring.

    Returns
    -------
    Float[Array, "Nz-2 Ny-2 Nx-2"]
        Raw difference at interior destination points (V or X).

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dy``,
    ``R·dφ``, or whatever metric is appropriate.
    """
    return h[1:-1, 2:, 1:-1] - h[1:-1, 1:-1, 1:-1]


def diff_x_bwd_3d(
    h: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz-2 Ny-2 Nx-2"]:
    """Backward difference in x over all z-levels (east face → centre).

    Δx h[k, j, i] = h[k, j, i+½] − h[k, j, i−½]

    Maps U-points → T-points (or X-points → V-points).

    Parameters
    ----------
    h : Float[Array, "Nz Ny Nx"]
        Field on source points (U or X), including ghost ring.

    Returns
    -------
    Float[Array, "Nz-2 Ny-2 Nx-2"]
        Raw difference at interior destination points (T or V).

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dx``,
    ``R·cos(φ)·dλ``, or whatever metric is appropriate.
    """
    return h[1:-1, 1:-1, 1:-1] - h[1:-1, 1:-1, :-2]


def diff_y_bwd_3d(
    h: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz-2 Ny-2 Nx-2"]:
    """Backward difference in y over all z-levels (north face → centre).

    Δy h[k, j, i] = h[k, j+½, i] − h[k, j−½, i]

    Maps V-points → T-points (or X-points → U-points).

    Parameters
    ----------
    h : Float[Array, "Nz Ny Nx"]
        Field on source points (V or X), including ghost ring.

    Returns
    -------
    Float[Array, "Nz-2 Ny-2 Nx-2"]
        Raw difference at interior destination points (T or U).

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dy``,
    ``R·dφ``, or whatever metric is appropriate.
    """
    return h[1:-1, 1:-1, 1:-1] - h[1:-1, :-2, 1:-1]


# =====================================================================
# 1-D averaging stencils
# =====================================================================


def avg_x_fwd_1d(h: Float[Array, "Nx"]) -> Float[Array, "Nx-2"]:
    """Forward average in x (centre → east face), 1-D.

    h̄[i+½] = ½ (h[i] + h[i+1])

    Maps T-points → U-points.

    Parameters
    ----------
    h : Float[Array, "Nx"]
        Field on T-points, including ghost cells.

    Returns
    -------
    Float[Array, "Nx-2"]
        Averaged values at interior U-points.
    """
    return 0.5 * (h[1:-1] + h[2:])


def avg_x_bwd_1d(h: Float[Array, "Nx"]) -> Float[Array, "Nx-2"]:
    """Backward average in x (east face → centre), 1-D.

    h̄[i] = ½ (h[i−½] + h[i+½])

    Maps U-points → T-points.

    Parameters
    ----------
    h : Float[Array, "Nx"]
        Field on U-points, including ghost cells.

    Returns
    -------
    Float[Array, "Nx-2"]
        Averaged values at interior T-points.
    """
    return 0.5 * (h[1:-1] + h[:-2])


# =====================================================================
# 2-D averaging stencils (2-point)
# =====================================================================


def avg_x_fwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Forward average in x (centre → east face).

    h̄[j, i+½] = ½ (h[j, i] + h[j, i+1])

    Maps T-points → U-points (or V-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (T or V), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior destination points (U or X).
    """
    return 0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:])


def avg_y_fwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Forward average in y (centre → north face).

    h̄[j+½, i] = ½ (h[j, i] + h[j+1, i])

    Maps T-points → V-points (or U-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (T or U), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior destination points (V or X).
    """
    return 0.5 * (h[1:-1, 1:-1] + h[2:, 1:-1])


def avg_x_bwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Backward average in x (east face → centre).

    h̄[j, i] = ½ (h[j, i−½] + h[j, i+½])

    Maps U-points → T-points (or X-points → V-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (U or X), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior destination points (T or V).
    """
    return 0.5 * (h[1:-1, 1:-1] + h[1:-1, :-2])


def avg_y_bwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Backward average in y (north face → centre).

    h̄[j, i] = ½ (h[j−½, i] + h[j+½, i])

    Maps V-points → T-points (or X-points → U-points).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on source points (V or X), including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior destination points (T or U).
    """
    return 0.5 * (h[1:-1, 1:-1] + h[:-2, 1:-1])


# =====================================================================
# 2-D averaging stencils (4-point bilinear)
# =====================================================================


def avg_xy_fwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Bilinear average to NE corner (forward in both x and y).

    h̄[j+½, i+½] = ¼ (h[j,i] + h[j,i+1] + h[j+1,i] + h[j+1,i+1])

    Maps T-points → X-points.

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on T-points, including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior X-points.
    """
    return 0.25 * (h[1:-1, 1:-1] + h[1:-1, 2:] + h[2:, 1:-1] + h[2:, 2:])


def avg_xy_bwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Bilinear average from corners to centre (backward in both x and y).

    h̄[j, i] = ¼ (h[j+½,i+½] + h[j−½,i+½] + h[j+½,i−½] + h[j−½,i−½])

    Maps X-points → T-points.

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on X-points, including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior T-points.
    """
    return 0.25 * (h[1:-1, 1:-1] + h[:-2, 1:-1] + h[1:-1, :-2] + h[:-2, :-2])


def avg_xbwd_yfwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Bilinear average: U-points → V-points (backward-x, forward-y).

    h̄[j+½, i] = ¼ (h[j,i+½] + h[j+1,i+½] + h[j,i−½] + h[j+1,i−½])

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on U-points, including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior V-points.
    """
    return 0.25 * (h[1:-1, 1:-1] + h[2:, 1:-1] + h[1:-1, :-2] + h[2:, :-2])


def avg_xfwd_ybwd(h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny-2 Nx-2"]:
    """Bilinear average: V-points → U-points (forward-x, backward-y).

    h̄[j, i+½] = ¼ (h[j+½,i] + h[j−½,i] + h[j+½,i+1] + h[j−½,i+1])

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Field on V-points, including ghost ring.

    Returns
    -------
    Float[Array, "Ny-2 Nx-2"]
        Averaged values at interior U-points.
    """
    return 0.25 * (h[1:-1, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:] + h[:-2, 2:])


# =====================================================================
# 3-D averaging stencils (horizontal plane per z-level)
# =====================================================================


def avg_x_fwd_3d(
    h: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz-2 Ny-2 Nx-2"]:
    """Forward average in x over all z-levels (centre → east face).

    h̄[k, j, i+½] = ½ (h[k, j, i] + h[k, j, i+1])

    Maps T-points → U-points (or V-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Nz Ny Nx"]
        Field on source points (T or V), including ghost ring.

    Returns
    -------
    Float[Array, "Nz-2 Ny-2 Nx-2"]
        Averaged values at interior destination points (U or X).
    """
    return 0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 1:-1, 2:])


def avg_y_fwd_3d(
    h: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz-2 Ny-2 Nx-2"]:
    """Forward average in y over all z-levels (centre → north face).

    h̄[k, j+½, i] = ½ (h[k, j, i] + h[k, j+1, i])

    Maps T-points → V-points (or U-points → X-points).

    Parameters
    ----------
    h : Float[Array, "Nz Ny Nx"]
        Field on source points (T or U), including ghost ring.

    Returns
    -------
    Float[Array, "Nz-2 Ny-2 Nx-2"]
        Averaged values at interior destination points (V or X).
    """
    return 0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 2:, 1:-1])


def avg_x_bwd_3d(
    h: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz-2 Ny-2 Nx-2"]:
    """Backward average in x over all z-levels (east face → centre).

    h̄[k, j, i] = ½ (h[k, j, i−½] + h[k, j, i+½])

    Maps U-points → T-points (or X-points → V-points).

    Parameters
    ----------
    h : Float[Array, "Nz Ny Nx"]
        Field on source points (U or X), including ghost ring.

    Returns
    -------
    Float[Array, "Nz-2 Ny-2 Nx-2"]
        Averaged values at interior destination points (T or V).
    """
    return 0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 1:-1, :-2])


def avg_y_bwd_3d(
    h: Float[Array, "Nz Ny Nx"],
) -> Float[Array, "Nz-2 Ny-2 Nx-2"]:
    """Backward average in y over all z-levels (north face → centre).

    h̄[k, j, i] = ½ (h[k, j−½, i] + h[k, j+½, i])

    Maps V-points → T-points (or X-points → U-points).

    Parameters
    ----------
    h : Float[Array, "Nz Ny Nx"]
        Field on source points (V or X), including ghost ring.

    Returns
    -------
    Float[Array, "Nz-2 Ny-2 Nx-2"]
        Averaged values at interior destination points (T or U).
    """
    return 0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, :-2, 1:-1])


# =====================================================================
# Named neighbour views (one static slice per offset)
# =====================================================================


class _Neighbours2D(NamedTuple):
    """The 3×3 neighbourhood of every interior point, one slice per offset.

    Each field is ``h[..., 1+dj : Ny-1+dj, 1+di : Nx-1+di]`` — the value at
    offset ``(dj, di)`` from interior point ``(j, i)`` — with shape
    ``(..., Ny-2, Nx-2)``, exactly the slices the raw stencils above use:

    ========  ================  ========================
    field     offset (dj, di)   slice
    ========  ================  ========================
    ``c``     ( 0,  0)          ``h[..., 1:-1, 1:-1]``
    ``e``     ( 0, +1)          ``h[..., 1:-1, 2:]``
    ``w``     ( 0, -1)          ``h[..., 1:-1, :-2]``
    ``n``     (+1,  0)          ``h[..., 2:, 1:-1]``
    ``s``     (-1,  0)          ``h[..., :-2, 1:-1]``
    ``ne``    (+1, +1)          ``h[..., 2:, 2:]``
    ``nw``    (+1, -1)          ``h[..., 2:, :-2]``
    ``se``    (-1, +1)          ``h[..., :-2, 2:]``
    ``sw``    (-1, -1)          ``h[..., :-2, :-2]``
    ========  ================  ========================

    Why: a stencil that slices the same offset twice (e.g. ``h[1:-1, 1:-1]``
    in both ``diff_x_fwd(h)`` and ``diff_x_bwd(h)``) computes the same forward
    values, but reverse-mode AD transposes *each* slice into its own ``pad``
    of the cotangent.  Slicing each offset once lets JAX sum the cotangents
    of that view first, so the VJP is one fused pass with one ``pad`` per
    offset.  Unused fields cost nothing (XLA drops dead slices).
    """

    c: Array
    e: Array
    w: Array
    n: Array
    s: Array
    ne: Array
    nw: Array
    se: Array
    sw: Array


def _neighbours(h: Float[Array, "... Ny Nx"]) -> _Neighbours2D:
    """Slice the nine 3×3-neighbourhood views of ``h`` once each.

    See :class:`_Neighbours2D` for the offset of each field.  All views have
    shape ``(..., Ny-2, Nx-2)``.
    """
    return _Neighbours2D(
        c=h[..., 1:-1, 1:-1],
        e=h[..., 1:-1, 2:],
        w=h[..., 1:-1, :-2],
        n=h[..., 2:, 1:-1],
        s=h[..., :-2, 1:-1],
        ne=h[..., 2:, 2:],
        nw=h[..., 2:, :-2],
        se=h[..., :-2, 2:],
        sw=h[..., :-2, :-2],
    )


def _materialize(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Keep ``x`` as its own buffer instead of fusing it into its consumers.

    Identity on values and derivatives (``jax.lax.optimization_barrier``).
    Use it on an intermediate field that the *next* stencil reads at several
    offsets, e.g. the first Laplacian of a biharmonic operator: without it
    XLA may fuse the producer into each shifted read and recompute it once
    per offset (measured ~1.5x slower forward on CPU for the biharmonic).
    """
    return jax.lax.optimization_barrier(x)
