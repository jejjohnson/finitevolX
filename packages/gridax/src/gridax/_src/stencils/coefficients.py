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
    "Bias",
    "Data",
    "Method",
    "difference_coeffs",
    "difference_offsets",
    "nonuniform_coeffs",
    "optimal_weights",
    "smoothness_matrix",
    "weno_stencils",
]

Method = Literal["central", "forward", "backward", "staggered"]
#: What a sample is: the value at a point, or the average over its cell.
Data = Literal["point", "cell"]
#: Where a staggered stencil leans: centred, or one more cell to the left
#: (the upwind side for flow towards +x) or to the right (flow towards -x).
Bias = Literal["centred", "left", "right"]

#: The collocated methods, widest support first: the fallback order of
#: ``gridax.analysis.adaptive_difference``. ``"staggered"`` is not in it,
#: because a staggered stencil writes half a cell away from its inputs.
METHODS: tuple[Method, ...] = ("central", "forward", "backward")


@functools.lru_cache(maxsize=256)
def difference_offsets(
    *,
    derivative: int = 1,
    accuracy: int = 1,
    method: Method = "central",
    bias: Bias = "centred",
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
        bias: For ``method="staggered"`` only. ``"centred"`` takes the
            smallest even number of half-integers at least ``n + p``;
            ``"left"`` takes the smallest odd number, with one more on the
            left, and ``"right"`` mirrors it. The biased sets are the upwind
            reconstructions of advection: ``left`` for flow towards ``+x``.

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
        >>> difference_offsets(
        ...     derivative=0, accuracy=5, method="staggered", bias="left"
        ... )
        (-2.5, -1.5, -0.5, 0.5, 1.5)
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
    if bias != "centred" and method != "staggered":
        raise ValueError(f"bias={bias!r} applies to method='staggered' only")
    if method == "forward":
        return tuple(range(support))
    if method == "backward":
        return tuple(range(-(support - 1), 1))
    if method == "central":
        half = support // 2
        return tuple(range(-half, half + 1))
    if method == "staggered":
        if bias == "centred":
            half = (support + 1) // 2
            return tuple(k + 0.5 for k in range(-half, half))
        odd = support + 1 - support % 2
        left = tuple(k + 0.5 for k in range(-(odd + 1) // 2, (odd - 1) // 2))
        if bias == "left":
            return left
        if bias == "right":
            return tuple(-s for s in reversed(left))
        raise ValueError(f"bias must be 'centred', 'left' or 'right'; got {bias!r}")
    raise ValueError(f"method must be one of {(*METHODS, 'staggered')}; got {method!r}")


@functools.lru_cache(maxsize=256)
def difference_coeffs(
    offsets: tuple[int | float, ...], derivative: int = 1, *, data: Data = "point"
) -> tuple[float, ...]:
    r"""The coefficients of a stencil on ``offsets``, exact, from the Taylor conditions.

    Solves :math:`\sum_k c_k \mu_m(s_k) = n!\,\delta_{mn}` for
    :math:`m = 0, \dots, K - 1` in rational arithmetic (every float offset
    converts to a fraction exactly) and rounds each coefficient once to the
    nearest float64, so :math:`\frac{1}{12}` is the float nearest 1/12. The
    moment :math:`\mu_m(s)` is what sample :math:`s` sees of :math:`x^m`:
    :math:`s^m` for point values, and the cell average
    :math:`((s + \frac12)^{m+1} - (s - \frac12)^{m+1}) / (m + 1)` for cell
    averages. The two agree for two samples and differ from three on.

    Args:
        offsets: The stencil's offsets in units of the step, distinct:
            integers for a collocated stencil, half-integers for a staggered
            one, or any other floats.
        derivative: The derivative order, at least 0; ``derivative=0`` gives
            interpolation weights. Fewer than ``derivative + 1`` offsets
            cannot resolve it.
        data: ``"point"`` (the default): the samples are point values, as
            for differences and interpolation on a C-grid. ``"cell"``: they
            are cell averages over unit cells centred on the offsets, as a
            finite-volume tracer carries; ``derivative=0`` is then the
            reconstruction of a point value (a face value, for advection).

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
        >>> [c * 12 for c in difference_coeffs((-1.5, -0.5, 0.5, 1.5), 0, data="cell")]
        [-1.0, 7.0, 7.0, -1.0]
    """
    return tuple(float(c) for c in _exact_coeffs(tuple(offsets), derivative, data))


@functools.lru_cache(maxsize=256)
def _exact_coeffs(
    offsets: tuple[int | float, ...], derivative: int, data: Data
) -> tuple[Fraction, ...]:
    """The exact rational coefficients behind :func:`difference_coeffs`."""
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
    if data not in ("point", "cell"):
        raise ValueError(f"data must be 'point' or 'cell'; got {data!r}")
    # V[m][k] = μ_m(s_k): the transposed Vandermonde matrix for point values
    matrix = [[_moment(s, m, data) for s in points] for m in range(n)]
    rhs = [
        Fraction(math.factorial(derivative)) if m == derivative else Fraction(0)
        for m in range(n)
    ]
    return tuple(_solve_exact(matrix, rhs))


def _moment(s: Fraction, m: int, data: Data) -> Fraction:
    """What a sample at ``s`` sees of ``x**m``: its value, or its cell average."""
    if data == "point":
        return s**m
    half = Fraction(1, 2)
    return ((s + half) ** (m + 1) - (s - half) ** (m + 1)) / (m + 1)


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


@functools.lru_cache(maxsize=64)
def weno_stencils(
    order: int, *, bias: Literal["left", "right"] = "left"
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """The full stencil of WENO of ``order`` and its sub-stencils, as offsets.

    WENO of order :math:`2r - 1` reconstructs the face value at offset 0 from
    the :math:`2r - 1` cell averages of the biased stencil, by blending the
    :math:`r` sub-stencils of :math:`r` consecutive cells that it holds. The
    sub-stencils are listed from the left, as ``gridax``'s ``weno_*``
    functions take them.

    Args:
        order: The order :math:`2r - 1`: 3, 5, 7, 9, …
        bias: ``"left"`` (flow towards ``+x``: the upwind cell is at
            :math:`-\\frac12`) or ``"right"`` (flow towards ``-x``).

    Returns:
        ``(full, subs)``: the :math:`2r - 1` offsets of the full stencil, and
        the :math:`r` sub-stencils, each :math:`r` offsets.

    Raises:
        ValueError: An even order, or one below 3.

    Examples:
        >>> full, subs = weno_stencils(5)
        >>> full
        (-2.5, -1.5, -0.5, 0.5, 1.5)
        >>> subs
        ((-2.5, -1.5, -0.5), (-1.5, -0.5, 0.5), (-0.5, 0.5, 1.5))
    """
    if order < 3 or order % 2 == 0:
        raise ValueError(f"the WENO order must be odd and at least 3; got {order}")
    r = (order + 1) // 2
    full = difference_offsets(
        derivative=0, accuracy=order, method="staggered", bias=bias
    )
    subs = tuple(tuple(full[k : k + r]) for k in range(r))
    return tuple(full), subs


@functools.lru_cache(maxsize=64)
def optimal_weights(
    offsets: tuple[float, ...], sub_offsets: tuple[tuple[float, ...], ...]
) -> tuple[float, ...]:
    r"""The linear (optimal) weights that blend sub-stencils into the full one.

    With :math:`\hat c^{(k)}` the cell-average reconstruction coefficients of
    sub-stencil :math:`k` padded with zeros onto ``offsets``, and :math:`c`
    those of ``offsets`` itself, the weights solve
    :math:`\sum_k g_k \hat c^{(k)} = c` exactly. They are the weights WENO
    falls back to where the data are smooth.

    Args:
        offsets: The full stencil's offsets.
        sub_offsets: The sub-stencils, each a subset of ``offsets``.

    Returns:
        One weight per sub-stencil, each the float nearest the exact value.

    Raises:
        ValueError: A sub-stencil outside ``offsets``, or sub-stencils that
            cannot reproduce the full stencil.

    Examples:
        >>> optimal_weights(*weno_stencils(5))
        (0.1, 0.6, 0.3)
        >>> [w * 35 for w in optimal_weights(*weno_stencils(7))]
        [1.0, 12.0, 18.0, 4.0]
    """
    full = _exact_coeffs(tuple(offsets), 0, "cell")
    columns = []
    for sub in sub_offsets:
        if not set(sub) <= set(offsets):
            raise ValueError(f"sub-stencil {sub} is not inside {offsets}")
        coeffs = dict(zip(sub, _exact_coeffs(tuple(sub), 0, "cell"), strict=True))
        columns.append([coeffs.get(s, Fraction(0)) for s in offsets])
    # the normal equations of a consistent system give its exact solution
    gram = [
        [sum(a * b for a, b in zip(ci, cj, strict=True)) for cj in columns]
        for ci in columns
    ]
    rhs = [sum(a * b for a, b in zip(ci, full, strict=True)) for ci in columns]
    weights = _solve_exact(gram, rhs)
    for row, target in enumerate(full):
        if sum(w * col[row] for w, col in zip(weights, columns, strict=True)) != target:
            raise ValueError("the sub-stencils cannot reproduce the full stencil")
    return tuple(float(w) for w in weights)


@functools.lru_cache(maxsize=64)
def smoothness_matrix(
    sub_offsets: tuple[float, ...], *, cell: float = -0.5
) -> tuple[tuple[float, ...], ...]:
    r"""The Jiang–Shu smoothness indicator of one sub-stencil, as a matrix.

    With :math:`p` the polynomial of degree :math:`r - 1` whose averages over
    the :math:`r` unit cells of ``sub_offsets`` are the data :math:`\bar q`,

    .. math::

        \beta = \sum_{l=1}^{r-1} \int_{c - 1/2}^{c + 1/2}
        \left(\frac{d^l p}{dx^l}\right)^2 dx = \bar q^\top B \bar q,

    with :math:`c` the centre of the cell being reconstructed (the grid step
    is 1, so the factors :math:`h^{2l-1}` drop out). :math:`p`'s coefficients
    are linear in :math:`\bar q`, so :math:`B` is a symmetric rational matrix,
    computed exactly.

    Args:
        sub_offsets: The :math:`r` cell offsets of the sub-stencil.
        cell: The centre of the cell the indicator is measured on:
            :math:`-\frac12` (the default) for a left-biased reconstruction
            at the face 0, :math:`+\frac12` for a right-biased one.

    Returns:
        :math:`B`, ``r`` rows of ``r`` floats, each the float nearest the
        exact value.

    Examples:
        >>> full, subs = weno_stencils(3)
        >>> smoothness_matrix(subs[0])  # beta = (q0 - qm)^2
        ((1.0, -1.0), (-1.0, 1.0))
        >>> _, (first, *_) = weno_stencils(7)
        >>> top_row, *_ = smoothness_matrix(first)
        >>> top_row[0] * 240
        547.0
    """
    points = [Fraction(s) for s in sub_offsets]
    r = len(points)
    # moments[j][m]: the average of x^m over cell j, so moments @ a = q-bar
    moments = [[_moment(s, m, "cell") for m in range(r)] for s in points]
    # a = M^{-1} q-bar: column j of M^{-1} is what q_j contributes to a
    inverse = [
        _solve_exact(moments, [Fraction(int(i == j)) for i in range(r)])
        for j in range(r)
    ]
    lo, hi = Fraction(cell) - Fraction(1, 2), Fraction(cell) + Fraction(1, 2)

    def integral(power: int) -> Fraction:
        return (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)

    matrix = [[Fraction(0)] * r for _ in range(r)]
    for order in range(1, r):
        # d^l x^m / dx^l = m! / (m - l)! x^(m - l)
        falling = {
            m: Fraction(math.factorial(m), math.factorial(m - order))
            for m in range(order, r)
        }
        for i in range(r):
            for j in range(r):
                matrix[i][j] += sum(
                    inverse[i][m1]
                    * inverse[j][m2]
                    * falling[m1]
                    * falling[m2]
                    * integral(m1 + m2 - 2 * order)
                    for m1 in range(order, r)
                    for m2 in range(order, r)
                )
    return tuple(tuple(float(x) for x in row) for row in matrix)


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
