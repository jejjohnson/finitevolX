"""Layer 0: the stencil offsets and exact coefficients of `gridax.stencils` (#454).

The coefficients solve the Taylor conditions sum_k c_k s_k^m = n! delta_mn,
m = 0..K-1, in rational arithmetic and round once; these tests compare them
with an independent rational solve, pin the staggered offset rule and its
orders, and check that the old `gridax.analysis` names still resolve.
"""

from __future__ import annotations

from fractions import Fraction
import math
import warnings

import gridax.analysis
import gridax.stencils as gs
import numpy as np
import pytest


def _exact(offsets: tuple[float, ...], derivative: int) -> list[Fraction]:
    """The exact rational solution, checked to satisfy the Taylor conditions exactly."""
    from gridax._src.stencils.coefficients import _solve_exact

    points = [Fraction(s) for s in offsets]
    n = len(points)
    matrix = [[s**m for s in points] for m in range(n)]
    rhs = [
        Fraction(math.factorial(derivative)) if m == derivative else Fraction(0)
        for m in range(n)
    ]
    solution = _solve_exact(matrix, rhs)
    # a zero residual in exact arithmetic: this is the unique solution
    for row, b in zip(matrix, rhs, strict=True):
        assert sum(v * c for v, c in zip(row, solution, strict=True)) == b
    return solution


CASES = [
    (method, derivative, accuracy)
    for method in ("central", "forward", "backward", "staggered")
    for derivative in range(0 if method == "staggered" else 1, 5)
    for accuracy in range(1, 6)
    if len(
        gs.difference_offsets(derivative=derivative, accuracy=accuracy, method=method)
    )
    <= 9
]


@pytest.mark.parametrize(("method", "derivative", "accuracy"), CASES)
def test_coefficients_are_the_exact_solution_rounded_once(
    method: str, derivative: int, accuracy: int
) -> None:
    offsets = gs.difference_offsets(
        derivative=derivative, accuracy=accuracy, method=method
    )
    got = gs.difference_coeffs(offsets, derivative)
    want = tuple(float(c) for c in _exact(offsets, derivative))
    assert got == want


@pytest.mark.parametrize(
    ("derivative", "accuracy", "offsets"),
    [
        (1, 1, (-0.5, 0.5)),
        (0, 1, (-0.5, 0.5)),
        (1, 2, (-1.5, -0.5, 0.5, 1.5)),
        (1, 3, (-1.5, -0.5, 0.5, 1.5)),
        (0, 4, (-1.5, -0.5, 0.5, 1.5)),
        (1, 5, (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),
    ],
)
def test_staggered_offsets_are_the_smallest_even_set(
    derivative: int, accuracy: int, offsets: tuple[float, ...]
) -> None:
    got = gs.difference_offsets(
        derivative=derivative, accuracy=accuracy, method="staggered"
    )
    assert got == offsets


def test_the_two_point_staggered_stencils_are_the_c_grid_ones() -> None:
    assert gs.difference_coeffs((-0.5, 0.5), 1) == (-1.0, 1.0)
    assert gs.difference_coeffs((-0.5, 0.5), 0) == (0.5, 0.5)


def test_the_four_point_staggered_stencils() -> None:
    four = (-1.5, -0.5, 0.5, 1.5)
    assert gs.difference_coeffs(four, 1) == (1 / 24, -27 / 24, 27 / 24, -1 / 24)
    assert gs.difference_coeffs(four, 0) == (-1 / 16, 9 / 16, 9 / 16, -1 / 16)


def test_non_dyadic_coefficients_are_the_nearest_float() -> None:
    # the float64 solve rounded to 12 decimals used to leave 3.3e-13 on these
    assert gs.difference_coeffs((-2, -1, 0, 1, 2), 1)[0] == 1 / 12
    assert gs.difference_coeffs((-1.5, -0.5, 0.5, 1.5), 1)[0] == 1 / 24


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
@pytest.mark.parametrize("derivative", [1, 2])
def test_the_accuracy_one_stencils_are_dyadic_and_unchanged(
    method: str, derivative: int
) -> None:
    # the default gridax.analysis stencils: exact before and after the rational solve
    offsets = gs.difference_offsets(derivative=derivative, accuracy=1, method=method)
    for c in gs.difference_coeffs(offsets, derivative):
        assert Fraction(c).denominator in (1, 2)


@pytest.mark.parametrize("derivative", [0, 1])
@pytest.mark.parametrize("accuracy", [1, 3, 5])
def test_staggered_orders(derivative: int, accuracy: int) -> None:
    # observed order K on f = sin(x + 0.7) at x = 0, from h = 0.2 and h = 0.1
    offsets = gs.difference_offsets(
        derivative=derivative, accuracy=accuracy, method="staggered"
    )
    coeffs = np.asarray(gs.difference_coeffs(offsets, derivative))
    s = np.asarray(offsets)
    exact = np.sin(0.7) if derivative == 0 else np.cos(0.7)

    def error(h: float) -> float:
        return abs(coeffs @ np.sin(s * h + 0.7) / h**derivative - exact)

    order = np.log2(error(0.2) / error(0.1))
    assert order == pytest.approx(len(offsets), abs=0.05)


@pytest.mark.parametrize("seed", range(5))
def test_random_offsets_reproduce_polynomials(seed: int) -> None:
    rng = np.random.default_rng(seed)
    size = int(rng.integers(2, 8))
    halves = rng.choice(np.arange(-12, 13), size=size, replace=False) / 2
    offsets = tuple(float(x) for x in np.sort(halves))
    derivative = int(rng.integers(0, size))
    coeffs = np.asarray(gs.difference_coeffs(offsets, derivative))
    s = np.asarray(offsets)
    # the Taylor conditions themselves: sum_k c_k s_k^m = n! delta_mn
    for m in range(size):
        want = float(math.factorial(derivative)) if m == derivative else 0.0
        scale = np.abs(coeffs).sum() * max(1.0, np.abs(s).max()) ** m
        assert coeffs @ s**m == pytest.approx(want, abs=1e-12 * scale)


def test_derivative_zero_is_staggered_only() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        gs.difference_offsets(derivative=0, method="central")


def test_bad_input_raises() -> None:
    with pytest.raises(ValueError, match="cannot resolve"):
        gs.difference_coeffs((0, 1), 2)
    with pytest.raises(ValueError, match="distinct"):
        gs.difference_coeffs((0, 0, 1), 1)
    with pytest.raises(ValueError, match="method"):
        gs.difference_offsets(method="sideways")  # type: ignore[arg-type]


def test_collocated_difference_rejects_the_staggered_method() -> None:
    with pytest.raises(ValueError, match="half a cell"):
        gridax.analysis.difference(np.arange(5.0), method="staggered")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name",
    [
        "METHODS",
        "Method",
        "difference_coeffs",
        "difference_offsets",
        "nonuniform_coeffs",
    ],
)
def test_the_old_analysis_names_warn_and_resolve(name: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = getattr(gridax.analysis, name)
    assert old is getattr(gs, name)
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "gridax.stencils" in str(w.message)
        for w in caught
    )


def test_the_named_stencils_are_re_exported() -> None:
    import gridax

    for name in gs.__all__:
        if name.startswith(("diff_", "avg_")) or name == "interior":
            assert getattr(gs, name) is getattr(gridax, name)
    assert sum(name.startswith(("diff_", "avg_")) for name in gs.__all__) == 32
