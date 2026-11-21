"""Layer 0 for cell averages and WENO (#462).

A cell sample is the average of f over its unit cell, so the moment of sample
s is ((s + 1/2)^(m+1) - (s - 1/2)^(m+1)) / (m + 1) instead of s^m. These
tests regenerate every hand-typed table of the advection code (the seven
``linear_*`` reconstructions, and the WENO sub-stencil coefficients, optimal
weights and Jiang–Shu smoothness matrices) and compare them bit for bit.
"""

from __future__ import annotations

from fractions import Fraction

from gridax._src.advection import linear, weno
from gridax._src.stencils.coefficients import _exact_coeffs
import gridax.stencils as gs
import numpy as np
import pytest

# (function, first cell relative to the face's left cell, number of cells)
LINEAR = {
    "linear_2pts": (0, 2),
    "linear_3pts_left": (-1, 3),
    "linear_3pts_right": (0, 3),
    "linear_4pts": (-1, 4),
    "linear_5pts_left": (-2, 5),
    "linear_5pts_right": (-1, 5),
    "linear_6pts": (-2, 6),
}


def _cells(first: int, count: int) -> tuple[float, ...]:
    """Offsets of the cells q[i + first] … relative to the face i + 1/2."""
    return tuple(first + k - 0.5 for k in range(count))


@pytest.mark.parametrize("name", sorted(LINEAR))
def test_linear_reconstructions_are_the_cell_average_stencils(name: str) -> None:
    first, count = LINEAR[name]
    coeffs = gs.difference_coeffs(_cells(first, count), 0, data="cell")
    q = np.random.default_rng(count).standard_normal((count, 64))
    got = getattr(linear, name)(*q)
    want = sum(c * q[k] for k, c in enumerate(coeffs))
    np.testing.assert_array_equal(got, want)


def test_point_and_cell_weights_agree_for_two_samples_only() -> None:
    two, four = (-0.5, 0.5), (-1.5, -0.5, 0.5, 1.5)
    assert gs.difference_coeffs(two, 0) == gs.difference_coeffs(two, 0, data="cell")
    assert gs.difference_coeffs(two, 1) == gs.difference_coeffs(two, 1, data="cell")
    assert [c * 16 for c in gs.difference_coeffs(four, 0)] == [-1, 9, 9, -1]
    assert [c * 12 for c in gs.difference_coeffs(four, 0, data="cell")] == [
        -1,
        7,
        7,
        -1,
    ]
    assert [c * 12 for c in gs.difference_coeffs(four, 1, data="cell")] == [
        1,
        -15,
        15,
        -1,
    ]


@pytest.mark.parametrize(
    ("first", "count"), [(0, 2), (-1, 3), (-1, 4), (-2, 5), (-2, 6)]
)
def test_cell_average_reconstruction_orders(first: int, count: int) -> None:
    # observed order K on the cell averages of sin(x + 0.7), from h = 0.2 and 0.1
    offsets = _cells(first, count)
    coeffs = np.asarray(gs.difference_coeffs(offsets, 0, data="cell"))
    s = np.asarray(offsets)

    def error(h: float) -> float:
        averages = (np.cos((s - 0.5) * h + 0.7) - np.cos((s + 0.5) * h + 0.7)) / h
        return abs(coeffs @ averages - np.sin(0.7))

    assert np.log2(error(0.2) / error(0.1)) == pytest.approx(count, abs=0.1)


@pytest.mark.parametrize(
    ("accuracy", "bias", "offsets"),
    [
        (3, "left", (-1.5, -0.5, 0.5)),
        (3, "right", (-0.5, 0.5, 1.5)),
        (5, "left", (-2.5, -1.5, -0.5, 0.5, 1.5)),
        (4, "left", (-2.5, -1.5, -0.5, 0.5, 1.5)),
        (5, "right", (-1.5, -0.5, 0.5, 1.5, 2.5)),
    ],
)
def test_biased_offsets_are_the_smallest_odd_set(accuracy, bias, offsets) -> None:
    got = gs.difference_offsets(
        derivative=0, accuracy=accuracy, method="staggered", bias=bias
    )
    assert got == offsets


def test_bias_applies_to_the_staggered_method_only() -> None:
    with pytest.raises(ValueError, match="staggered"):
        gs.difference_offsets(method="central", bias="left")
    with pytest.raises(ValueError, match="data"):
        gs.difference_coeffs((-0.5, 0.5), 0, data="volume")  # type: ignore[arg-type]


def _table(order: int):
    tables = {
        7: (weno._WENO7_COEFFS, weno._WENO7_WEIGHTS, weno._WENO7_BETA_MATRICES),
        9: (weno._WENO9_COEFFS, weno._WENO9_WEIGHTS, weno._WENO9_BETA_MATRICES),
    }
    right = {
        7: (
            weno._WENO7_COEFFS_RIGHT,
            weno._WENO7_WEIGHTS_RIGHT,
            weno._WENO7_BETA_MATRICES_RIGHT,
        ),
        9: (
            weno._WENO9_COEFFS_RIGHT,
            weno._WENO9_WEIGHTS_RIGHT,
            weno._WENO9_BETA_MATRICES_RIGHT,
        ),
    }
    return tables[order], right[order]


@pytest.mark.parametrize("order", [7, 9])
@pytest.mark.parametrize("bias", ["left", "right"])
def test_weno_tables_are_layer0_bit_for_bit(order: int, bias: str) -> None:
    left_tables, right_tables = _table(order)
    coeffs, weights, betas = left_tables if bias == "left" else right_tables
    full, subs = gs.weno_stencils(order, bias=bias)
    cell = -0.5 if bias == "left" else 0.5
    assert tuple(gs.difference_coeffs(s, 0, data="cell") for s in subs) == tuple(coeffs)
    assert gs.optimal_weights(full, subs) == tuple(weights)
    for sub, table in zip(subs, betas, strict=True):
        np.testing.assert_array_equal(
            np.asarray(gs.smoothness_matrix(sub, cell=cell)), table
        )


@pytest.mark.parametrize(
    ("order", "weights"),
    [(3, (1 / 3, 2 / 3)), (5, (0.1, 0.6, 0.3))],
)
def test_weno3_and_weno5_optimal_weights(order: int, weights: tuple) -> None:
    assert gs.optimal_weights(*gs.weno_stencils(order)) == weights


@pytest.mark.parametrize("order", [3, 5, 7, 9])
def test_optimal_weights_are_positive_and_sum_to_one(order: int) -> None:
    weights = gs.optimal_weights(*gs.weno_stencils(order))
    assert all(w > 0 for w in weights)
    assert sum(Fraction(w) for w in weights) == pytest.approx(1, abs=1e-15)


@pytest.mark.parametrize("order", [3, 5, 7, 9])
def test_smoothness_matrices_are_symmetric_semidefinite_and_blind_to_constants(order):
    _, subs = gs.weno_stencils(order)
    for sub in subs:
        b = np.asarray(gs.smoothness_matrix(sub))
        np.testing.assert_array_equal(b, b.T)
        assert np.linalg.eigvalsh(b).min() > -1e-10 * np.abs(b).max()
        # beta = 0 on constant data
        assert abs(np.ones(len(sub)) @ b @ np.ones(len(sub))) < 1e-12 * np.abs(b).max()


def test_weno5_closed_form_smoothness_matches_the_matrices() -> None:
    _, subs = gs.weno_stencils(5)
    q = np.random.default_rng(7).standard_normal((5, 200))
    qmm, qm, q0, qp, qpp = q
    closed = [
        13 / 12 * (qmm - 2 * qm + q0) ** 2 + 0.25 * (qmm - 4 * qm + 3 * q0) ** 2,
        13 / 12 * (qm - 2 * q0 + qp) ** 2 + 0.25 * (qm - qp) ** 2,
        13 / 12 * (q0 - 2 * qp + qpp) ** 2 + 0.25 * (3 * q0 - 4 * qp + qpp) ** 2,
    ]
    for k, sub in enumerate(subs):
        b = np.asarray(gs.smoothness_matrix(sub))
        window = q[k : k + 3]
        quadratic = np.einsum("i...,ij,j...->...", window, b, window)
        np.testing.assert_allclose(quadratic, closed[k], rtol=1e-13, atol=1e-13)


def test_smoothness_matrix_against_quadrature() -> None:
    # beta for one random data vector, by Gauss-Legendre quadrature of (p')^2 + (p'')^2
    _, subs = gs.weno_stencils(5)
    sub = subs[0]
    q = np.random.default_rng(3).standard_normal(3)
    # the polynomial with these cell averages: solve the moment system in float64
    moments = np.array(
        [
            [
                float(
                    (
                        (Fraction(s) + Fraction(1, 2)) ** (m + 1)
                        - (Fraction(s) - Fraction(1, 2)) ** (m + 1)
                    )
                    / (m + 1)
                )
                for m in range(3)
            ]
            for s in sub
        ]
    )
    a = np.linalg.solve(moments, q)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    x = -0.5 + 0.5 * nodes  # the cell [-1, 0]
    d1 = a[1] + 2 * a[2] * x
    d2 = 2 * a[2] * np.ones_like(x)
    beta = 0.5 * weights @ (d1**2 + d2**2)
    assert q @ np.asarray(gs.smoothness_matrix(sub)) @ q == pytest.approx(
        beta, rel=1e-12
    )


def test_exact_coefficients_reproduce_cell_averages_of_powers() -> None:
    offsets = (-2.5, -1.5, -0.5, 0.5, 1.5)
    exact = _exact_coeffs(offsets, 0, "cell")
    half = Fraction(1, 2)
    for m in range(5):
        averages = [
            ((Fraction(s) + half) ** (m + 1) - (Fraction(s) - half) ** (m + 1))
            / (m + 1)
            for s in offsets
        ]
        # the reconstruction of x^m at the face 0 is exactly 0^m
        assert sum(c * a for c, a in zip(exact, averages, strict=True)) == (
            1 if m == 0 else 0
        )


def test_weno_stencils_reject_bad_orders() -> None:
    with pytest.raises(ValueError, match="odd"):
        gs.weno_stencils(4)
