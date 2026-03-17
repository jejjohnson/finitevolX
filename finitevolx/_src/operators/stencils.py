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

from jaxtyping import Array, Float

# =====================================================================
# 1-D difference stencils
# =====================================================================


def diff_x_fwd_1d(h: Float[Array, " Nx"]) -> Float[Array, " Nx-2"]:
    """Forward difference in x (centre → east face), 1-D.

    Δx h[i+½] = h[i+1] − h[i]

    Maps T-points → U-points.

    Parameters
    ----------
    h : Float[Array, " Nx"]
        Field on T-points, including ghost cells.

    Returns
    -------
    Float[Array, " Nx-2"]
        Raw difference at interior U-points.

    Notes
    -----
    No metric scaling is applied.  The caller divides by ``dx``,
    ``R·cos(φ)·dλ``, or whatever metric is appropriate.
    """
    return h[2:] - h[1:-1]


def diff_x_bwd_1d(h: Float[Array, " Nx"]) -> Float[Array, " Nx-2"]:
    """Backward difference in x (east face → centre), 1-D.

    Δx h[i] = h[i+½] − h[i−½]

    Maps U-points → T-points.

    Parameters
    ----------
    h : Float[Array, " Nx"]
        Field on U-points, including ghost cells.

    Returns
    -------
    Float[Array, " Nx-2"]
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
