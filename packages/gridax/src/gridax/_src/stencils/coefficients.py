"""Layer 0: stencil offsets and exact coefficients, from numbers alone.

The lowest layer of gridax's derivative stack. Nothing here touches an array
of field values: a stencil depends only on the derivative order, the accuracy
order and the method, so it is computed once on the host (and cached), and the
layers above apply it by slicing on any backend. Collocated stencils (the
result sits on an input point) use integer offsets; staggered stencils (the
result sits half a cell from every input, as on the faces of a C-grid) use
half-integer offsets.

Offset rules, for derivative order :math:`n` and accuracy :math:`p`, as
finitediffx's for the collocated methods:

- a *forward* stencil uses the offsets :math:`0, 1, \\dots, n + p - 1`;
- a *backward* stencil mirrors it, :math:`-(n + p - 1), \\dots, 0`;
- a *central* stencil uses the :math:`K` integers around 0, with :math:`K` the
  smallest odd number at least :math:`n + p` (so ``accuracy=1`` is the
  second-order three-point stencil, as in ``numpy.gradient``);
- a *staggered* stencil uses the :math:`K` half-integers
  :math:`\\pm\\frac12, \\pm\\frac32, \\dots, \\pm\\frac{K - 1}{2}`, with :math:`K` the
  smallest even number at least :math:`n + p`. It has order :math:`K` for
  :math:`n = 0` (an average) and :math:`n = 1`, and :math:`K = 2` gives the
  C-grid stencils: :math:`(-1, 1)` for ``diff_*`` and :math:`(\\frac12, \\frac12)`
  for ``avg_*``.

The coefficients :math:`c_k` of offsets :math:`s_k` solve the Taylor
conditions :math:`\\sum_k c_k s_k^m = n!\\,\\delta_{mn}` for
:math:`m = 0, \\dots, K - 1`, in exact rational arithmetic, and each is rounded
once to float64.

Example:
    >>> difference_offsets(derivative=1, accuracy=1, method="central")
    (-1, 0, 1)
    >>> difference_coeffs((-1, 0, 1), derivative=1)
    (-0.5, 0.0, 0.5)
    >>> difference_offsets(derivative=1, accuracy=1, method="staggered")
    (-0.5, 0.5)
    >>> difference_coeffs((-0.5, 0.5), derivative=1)
    (-1.0, 1.0)
"""

from __future__ import annotations

from fractions import Fraction
import functools
import math
from typing import Any, Literal

from gridax._src.pytree import array_namespace

__all__ = [
    "METHODS",
    "Method",
    "difference_coeffs",
    "difference_offsets",
    "nonuniform_coeffs",
]

Method = Literal["central", "forward", "backward", "staggered"]

#: The collocated methods, widest support first: the fallback order of
#: ``gridax.analysis.adaptive_difference``. ``"staggered"`` is not in it,
#: because a staggered stencil writes half a cell away from its inputs.
METHODS: tuple[Method, ...] = ("central", "forward", "backward")


@functools.lru_cache(maxsize=256)
def difference_offsets(
    *, derivative: int = 1, accuracy: int = 1, method: Method = "central"
) -> tuple[int, ...] | tuple[float, ...]:
    """The offsets of one stencil, in units of the grid step.

    Args:
        derivative: The derivative order: at least 1, or 0 (an average onto
            the faces) for ``method="staggered"``.
        accuracy: The accuracy order, at least 1. A central or staggered
            stencil gets the order its symmetry gives for free on top (see
            the module docstring).
        method: ``"central"``, ``"forward"``, ``"backward"`` or
            ``"staggered"``.

    Returns:
        The offsets, ascending: integers for the collocated methods,
        half-integers (as floats, exact in binary) for ``"staggered"``.

    Raises:
        ValueError: An order out of range, or an unknown method.

    Examples:
        >>> difference_offsets(derivative=1, accuracy=2, method="forward")
        (0, 1, 2)
        >>> difference_offsets(derivative=2, accuracy=1, method="central")
        (-1, 0, 1)
        >>> difference_offsets(derivative=1, accuracy=3, method="staggered")
        (-1.5, -0.5, 0.5, 1.5)
        >>> difference_offsets(derivative=0, accuracy=1, method="staggered")
        (-0.5, 0.5)
    """
    if accuracy < 1:
        raise ValueError(f"accuracy must be at least 1; got {accuracy}")
    lowest = 0 if method == "staggered" else 1
    if derivative < lowest:
        raise ValueError(
            f"derivative must be at least {lowest} for method {method!r}; "
            f"got {derivative}"
        )
    support = derivative + accuracy
    if method == "forward":
        return tuple(range(support))
    if method == "backward":
        return tuple(range(-(support - 1), 1))
    if method == "central":
        half = support // 2
        return tuple(range(-half, half + 1))
    if method == "staggered":
        half = (support + 1) // 2
        return tuple(k + 0.5 for k in range(-half, half))
    raise ValueError(f"method must be one of {(*METHODS, 'staggered')}; got {method!r}")


@functools.lru_cache(maxsize=256)
def difference_coeffs(
    offsets: tuple[int | float, ...], derivative: int = 1
) -> tuple[float, ...]:
    r"""The coefficients of a stencil on ``offsets``, exact, from the Taylor conditions.

    Solves :math:`\sum_k c_k s_k^m = n!\,\delta_{mn}` for
    :math:`m = 0, \dots, K - 1` in rational arithmetic (every float offset
    converts to a fraction exactly) and rounds each coefficient once to the
    nearest float64, so :math:`\frac{1}{12}` is the float nearest 1/12.

    Args:
        offsets: The stencil's offsets in units of the step, distinct:
            integers for a collocated stencil, half-integers for a staggered
            one, or any other floats.
        derivative: The derivative order, at least 0; ``derivative=0`` gives
            interpolation weights. Fewer than ``derivative + 1`` offsets
            cannot resolve it.

    Returns:
        One coefficient per offset; :math:`f^{(n)}(x) \approx h^{-n} \sum_k
        c_k f(x + s_k h)`.

    Raises:
        ValueError: A negative derivative, repeated offsets, or too few
            offsets for the derivative.

    Examples:
        >>> difference_coeffs((-1, 0, 1), derivative=2)
        (1.0, -2.0, 1.0)
        >>> difference_coeffs((0, 1, 2), derivative=1)
        (-1.5, 2.0, -0.5)
        >>> [c * 24 for c in difference_coeffs((-1.5, -0.5, 0.5, 1.5), 1)]
        [1.0, -27.0, 27.0, -1.0]
        >>> [c * 16 for c in difference_coeffs((-1.5, -0.5, 0.5, 1.5), 0)]
        [-1.0, 9.0, 9.0, -1.0]
    """
    n = len(offsets)
    if derivative < 0:
        raise ValueError(f"derivative must be at least 0; got {derivative}")
    if derivative >= n:
        raise ValueError(
            f"{n} offsets cannot resolve derivative {derivative}; need at least "
            f"{derivative + 1}"
        )
    points = [Fraction(s) for s in offsets]
    if len(set(points)) != n:
        raise ValueError(f"offsets must be distinct; got {offsets}")
    # V[m][k] = s_k^m, the transposed Vandermonde matrix of the Taylor conditions
    matrix = [[s**m for s in points] for m in range(n)]
    rhs = [
        Fraction(math.factorial(derivative)) if m == derivative else Fraction(0)
        for m in range(n)
    ]
    return tuple(float(c) for c in _solve_exact(matrix, rhs))


def _solve_exact(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Gauss-Jordan elimination in rational arithmetic; ``matrix`` is invertible."""
    n = len(rhs)
    rows = [[*row, b] for row, b in zip(matrix, rhs, strict=True)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if rows[r][col] != 0)
        rows[col], rows[pivot] = rows[pivot], rows[col]
        lead = rows[col][col]
        rows[col] = [x / lead for x in rows[col]]
        for r in range(n):
            factor = rows[r][col]
            if r != col and factor != 0:
                rows[r] = [
                    x - factor * y for x, y in zip(rows[r], rows[col], strict=True)
                ]
    return [row[n] for row in rows]


def nonuniform_coeffs(dz: Any, derivative: int = 1) -> Any:
    r"""Stencil coefficients for points at arbitrary distances, on any backend.

    The non-uniform form of :func:`difference_coeffs`: for each point, the
    weights :math:`c_k` with :math:`f^{(n)}(z) \approx \sum_k c_k f(z + \delta_k)`
    solve the Taylor conditions :math:`\sum_k c_k \delta_k^m / m! = \delta_{mn}`
    for :math:`m = 0, \dots, K - 1`, so the stencil is exact for polynomials of
    degree :math:`K - 1`. Each row is solved in units of its own widest
    distance, which keeps the system well conditioned for coordinates in
    metres or pascals. Written on the array namespace, so traced distances
    (a ``TimeAxis`` under ``jax.jit``) work too.

    Args:
        dz: ``(..., K)`` signed distances :math:`\delta_k` from each point to
            its stencil's points; one of them is ``0``.
        derivative: The derivative order :math:`n`, below ``K``.

    Returns:
        ``(..., K)`` coefficients, in units of ``dz`` to the power ``-n``.

    Raises:
        ValueError: Too few points for the derivative.

    Examples:
        >>> nonuniform_coeffs(np.array([-1.0, 0.0, 1.0]))  # the uniform stencil
        array([-0.5,  0. ,  0.5])
        >>> # uneven levels at 0, 1 and 3 m: still exact for a quadratic
        >>> c = nonuniform_coeffs(np.array([-1.0, 0.0, 2.0]), derivative=2)
        >>> float(c @ np.array([-1.0, 0.0, 2.0]) ** 2)
        2.0
    """

    xp = array_namespace(dz)
    n_points = dz.shape[-1]
    if derivative >= n_points:
        raise ValueError(
            f"{n_points} points cannot resolve derivative {derivative}; need at "
            f"least {derivative + 1}"
        )
    scale = xp.max(xp.abs(dz), axis=-1, keepdims=True)
    unit = dz / scale
    # A[..., m, k] = δ_k^m / m!, in units of the row's widest distance
    powers = xp.stack([unit**m / math.factorial(m) for m in range(n_points)], axis=-2)
    rhs = xp.zeros((*dz.shape[:-1], n_points), dtype=unit.dtype)
    rhs = xp.where(xp.arange(n_points) == derivative, 1.0, rhs)
    coeffs = xp.linalg.solve(powers, rhs[..., None])[..., 0]
    return coeffs / scale**derivative
