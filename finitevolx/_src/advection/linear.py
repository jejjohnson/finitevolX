"""
Linear (polynomial) face-value reconstruction stencils.

These are the *optimal* linear interpolation weights for reconstructing a
face value at position i+1/2 from cell-average data.  They correspond to the
linear weights that WENO schemes reduce to in smooth regions (i.e., when the
nonlinear smoothness-indicator weighting has no effect).

All functions follow the half-index convention:

    ... q_{i-1} --- q_i --x-- q_{i+1} --- q_{i+2} ...

where ``x`` marks the face at i+1/2.  Left-biased stencils use more cells
to the left of the face; right-biased stencils use more cells to the right.

References
----------
Shu, C.-W. (1998). Essentially Non-Oscillatory and Weighted Essentially
Non-Oscillatory Schemes for Hyperbolic Conservation Laws. In *Advanced
Numerical Approximation of Nonlinear Hyperbolic Equations*, Lecture Notes
in Mathematics 1697, Springer, pp. 325--432.
"""

from __future__ import annotations

from jaxtyping import Array

# ── 2-point (linear) ─────────────────────────────────────────────────────────


def linear_2pts(q0: Array, qp: Array) -> Array:
    """2-point centred reconstruction at face i+1/2.

    q_{i+1/2} = (q_i + q_{i+1}) / 2

    q0--x--qp

    This is simple linear interpolation (2nd-order centred).
    """
    return 0.5 * (q0 + qp)


# ── 3-point ───────────────────────────────────────────────────────────────────


def linear_3pts_left(qm: Array, q0: Array, qp: Array) -> Array:
    """3-point left-biased reconstruction at face i+1/2.

    q_{i+1/2} = -1/6 q_{i-1} + 5/6 q_i + 1/3 q_{i+1}

    qm-----q0--x--qp

    This is the optimal 3rd-order polynomial using 2 cells left and 1 right
    of the face.  Equivalent to the highest-order WENO3 candidate.
    """
    return -1.0 / 6.0 * qm + 5.0 / 6.0 * q0 + 1.0 / 3.0 * qp


def linear_3pts_right(q0: Array, qp: Array, qpp: Array) -> Array:
    """3-point right-biased reconstruction at face i+1/2.

    q_{i+1/2} = 1/3 q_i + 5/6 q_{i+1} - 1/6 q_{i+2}

    q0--x--qp-----qpp

    Mirror of :func:`linear_3pts_left` for negative-velocity upwinding.
    """
    return 1.0 / 3.0 * q0 + 5.0 / 6.0 * qp - 1.0 / 6.0 * qpp


# ── 4-point ───────────────────────────────────────────────────────────────────


def linear_4pts(qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """4-point centred reconstruction at face i+1/2.

    q_{i+1/2} = -1/12 q_{i-1} + 7/12 q_i + 7/12 q_{i+1} - 1/12 q_{i+2}

    qm-----q0--x--qp-----qpp

    Symmetric 4th-order polynomial through the 4 surrounding cells.
    """
    return -1.0 / 12.0 * qm + 7.0 / 12.0 * q0 + 7.0 / 12.0 * qp - 1.0 / 12.0 * qpp


# ── 5-point ───────────────────────────────────────────────────────────────────


def linear_5pts_left(qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array) -> Array:
    """5-point left-biased reconstruction at face i+1/2.

    q_{i+1/2} = 1/30 q_{i-2} - 13/60 q_{i-1} + 47/60 q_i
                + 9/20 q_{i+1} - 1/20 q_{i+2}

    qmm----qm-----q0--x--qp----qpp

    This is the optimal 5th-order polynomial using 3 cells left and 2 right
    of the face — the linear weight target of WENO5.
    """
    return (
        1.0 / 30.0 * qmm
        - 13.0 / 60.0 * qm
        + 47.0 / 60.0 * q0
        + 9.0 / 20.0 * qp
        - 1.0 / 20.0 * qpp
    )


def linear_5pts_right(
    qm: Array, q0: Array, qp: Array, qpp: Array, qppp: Array
) -> Array:
    """5-point right-biased reconstruction at face i+1/2.

    q_{i+1/2} = -1/20 q_{i-1} + 9/20 q_i + 47/60 q_{i+1}
                - 13/60 q_{i+2} + 1/30 q_{i+3}

    qm-----q0--x--qp----qpp---qppp

    Mirror of :func:`linear_5pts_left` for negative-velocity upwinding.
    """
    return (
        -1.0 / 20.0 * qm
        + 9.0 / 20.0 * q0
        + 47.0 / 60.0 * qp
        - 13.0 / 60.0 * qpp
        + 1.0 / 30.0 * qppp
    )


# ── 6-point ───────────────────────────────────────────────────────────────────


def linear_6pts(
    qmm: Array, qm: Array, q0: Array, qp: Array, qpp: Array, qppp: Array
) -> Array:
    """6-point centred reconstruction at face i+1/2.

    q_{i+1/2} = 1/60 q_{i-2} - 2/15 q_{i-1} + 37/60 q_i
                + 37/60 q_{i+1} - 2/15 q_{i+2} + 1/60 q_{i+3}

    qmm----qm-----q0--x--qp----qpp---qppp

    Symmetric 6th-order polynomial through 6 surrounding cells.
    """
    return (
        1.0 / 60.0 * qmm
        - 2.0 / 15.0 * qm
        + 37.0 / 60.0 * q0
        + 37.0 / 60.0 * qp
        - 2.0 / 15.0 * qpp
        + 1.0 / 60.0 * qppp
    )
