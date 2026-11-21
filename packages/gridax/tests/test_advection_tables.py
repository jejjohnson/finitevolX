"""weno.py takes its WENO-7 and WENO-9 tables from Layer 0 (#463).

The tables used to be typed by hand (239 numbers); they are now built on first
use by ``gridax.stencils`` and must equal the hand-typed values, frozen in
``data/weno_tables_hand_typed.npz``, bit for bit. The closed forms that stay
(WENO-3 and WENO-5, and the ``linear_*`` reconstructions) are pinned to Layer 0.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from gridax._src.advection import weno
import numpy as np
import pytest

HAND_TYPED = np.load(Path(__file__).parent / "data" / "weno_tables_hand_typed.npz")


@pytest.mark.parametrize("order", [7, 9])
@pytest.mark.parametrize("bias", ["left", "right"])
def test_generated_tables_equal_the_hand_typed_ones(order: int, bias: str) -> None:
    tables = weno._weno_tables(order, bias)
    np.testing.assert_array_equal(
        np.asarray(tables.coeffs), HAND_TYPED[f"weno{order}_{bias}_coeffs"]
    )
    np.testing.assert_array_equal(
        np.asarray(tables.weights), HAND_TYPED[f"weno{order}_{bias}_weights"]
    )
    np.testing.assert_array_equal(
        np.stack(tables.betas), HAND_TYPED[f"weno{order}_{bias}_betas"]
    )


def test_right_tables_are_the_mirror_of_the_left_ones() -> None:
    for order in (7, 9):
        left, right = (
            weno._weno_tables(order, "left"),
            weno._weno_tables(order, "right"),
        )
        assert right.coeffs == tuple(tuple(reversed(c)) for c in reversed(left.coeffs))
        assert right.weights == tuple(reversed(left.weights))
        for b_right, b_left in zip(right.betas, reversed(left.betas), strict=True):
            np.testing.assert_array_equal(b_right, b_left[::-1, ::-1])


@pytest.mark.parametrize(
    ("closed_form", "order", "bias"),
    [
        (weno.weno_3pts, 3, "left"),
        (weno.weno_3pts_right, 3, "right"),
        (weno.weno_5pts, 5, "left"),
        (weno.weno_5pts_right, 5, "right"),
    ],
)
def test_weno3_and_weno5_closed_forms_match_layer0(closed_form, order, bias) -> None:
    # the hand-written forms are factorisations of q^T B q; they agree to rounding
    tables = weno._weno_tables(order, bias)
    q = list(np.random.default_rng(order).standard_normal((order, 500)))
    r = (order + 1) // 2
    stencils = tuple(tuple(q[k : k + r]) for k in range(r))
    generic = weno._weno_reconstruct(stencils, *tables)
    np.testing.assert_allclose(closed_form(*q), generic, rtol=1e-13, atol=1e-13)


def test_importing_gridax_builds_no_table() -> None:
    code = (
        "import gridax, gridax.stencils; from gridax._src.advection import weno; "
        "print(weno._weno_tables.cache_info().currsize)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "0"
