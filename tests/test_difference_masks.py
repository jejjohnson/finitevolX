"""Mask-aware regression tests for ``Difference2D`` and ``Difference3D``.

These tests pin three things for every public method:

1. **Unmasked golden** — the existing math hasn't changed.  ``mask=None``
   produces bit-identical output to the prior implementation.
2. **Masked golden** — the new ``mask: ArakawaCGridMask`` parameter
   produces a deterministic, committed reference output for the
   canonical coastal mask in :mod:`tests.fixtures.inputs`.
3. **No-op invariant** — passing
   ``ArakawaCGridMask.from_dimensions(NY, NX)`` (an all-ocean mask)
   gives the same answer as ``mask=None``.

See ``tests/fixtures/README.md`` for the regeneration workflow.
"""

from __future__ import annotations

import numpy as np
import pytest

from finitevolx._src.operators.difference import Difference2D, Difference3D
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_grid_2d,
    make_grid_3d,
    make_h_field_2d,
    make_h_field_3d,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_q_field_2d,
    make_u_field_2d,
    make_u_field_3d,
    make_v_field_2d,
    make_v_field_3d,
)

# ----------------------------------------------------------------------
# Difference2D
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def diff2d():
    return Difference2D(grid=make_grid_2d())


@pytest.fixture(scope="module")
def fields_2d():
    return {
        "h": make_h_field_2d(),
        "u": make_u_field_2d(),
        "v": make_v_field_2d(),
        "q": make_q_field_2d(),
    }


@pytest.fixture(scope="module")
def mask2d():
    return make_mask_2d()


# Each entry is (method_name, callable_taking(diff, fields, mask) -> output).
# Defining the table once and parametrizing the three test classes over it
# keeps unmasked / masked / all-ocean coverage in lock-step.
DIFF2D_CALLS = [
    ("diff_x_T_to_U", lambda d, f, m: d.diff_x_T_to_U(f["h"], mask=m)),
    ("diff_y_T_to_V", lambda d, f, m: d.diff_y_T_to_V(f["h"], mask=m)),
    ("diff_y_U_to_X", lambda d, f, m: d.diff_y_U_to_X(f["u"], mask=m)),
    ("diff_x_V_to_X", lambda d, f, m: d.diff_x_V_to_X(f["v"], mask=m)),
    ("diff_y_X_to_U", lambda d, f, m: d.diff_y_X_to_U(f["q"], mask=m)),
    ("diff_x_X_to_V", lambda d, f, m: d.diff_x_X_to_V(f["q"], mask=m)),
    ("diff_x_U_to_T", lambda d, f, m: d.diff_x_U_to_T(f["u"], mask=m)),
    ("diff_y_V_to_T", lambda d, f, m: d.diff_y_V_to_T(f["v"], mask=m)),
    ("divergence", lambda d, f, m: d.divergence(f["u"], f["v"], mask=m)),
    ("curl", lambda d, f, m: d.curl(f["u"], f["v"], mask=m)),
    ("laplacian", lambda d, f, m: d.laplacian(f["h"], mask=m)),
    ("grad_perp", lambda d, f, m: d.grad_perp(f["h"], mask=m)),
]


class TestDifference2DUnmaskedGolden:
    """``mask=None`` matches the committed unmasked golden bit-for-bit."""

    @pytest.mark.parametrize(("method", "call"), DIFF2D_CALLS)
    def test_unmasked_matches_golden(self, diff2d, fields_2d, method, call):
        out = call(diff2d, fields_2d, None)
        assert_matches_golden(out, "Difference2D", method, "unmasked")


class TestDifference2DMaskedGolden:
    """``mask=coastal_mask`` matches the committed masked golden bit-for-bit."""

    @pytest.mark.parametrize(("method", "call"), DIFF2D_CALLS)
    def test_masked_matches_golden(self, diff2d, fields_2d, mask2d, method, call):
        out = call(diff2d, fields_2d, mask2d)
        assert_matches_golden(out, "Difference2D", method, "masked")


class TestDifference2DAllOceanIsNoOp:
    """Passing an all-ocean ArakawaCGridMask must equal mask=None.

    The all-ocean mask is what every existing model gets when migrated to
    the masks-everywhere API but the user doesn't actually have a
    coastline — this guards against accidental dtype/shape regressions
    in the post-compute multiply path.
    """

    @pytest.mark.parametrize(("method", "call"), DIFF2D_CALLS)
    def test_all_ocean_equals_unmasked(self, diff2d, fields_2d, method, call):
        out_none = call(diff2d, fields_2d, None)
        out_all_ocean = call(diff2d, fields_2d, make_mask_2d_all_ocean())
        if isinstance(out_none, tuple):
            for a, b in zip(out_none, out_all_ocean, strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        else:
            np.testing.assert_array_equal(
                np.asarray(out_none), np.asarray(out_all_ocean)
            )


class TestDifference2DDryCellsAreZero:
    """For T-output methods, every dry interior cell must be exactly 0.

    The post-compute-zero convention guarantees this. The test catches
    regressions where someone accidentally drops the multiply.
    """

    @pytest.mark.parametrize(
        ("method", "call", "stagger_attr"),
        [
            ("diff_x_U_to_T", DIFF2D_CALLS[6][1], "h"),
            ("diff_y_V_to_T", DIFF2D_CALLS[7][1], "h"),
            ("divergence", DIFF2D_CALLS[8][1], "h"),
            ("laplacian", DIFF2D_CALLS[10][1], "h"),
            ("diff_x_T_to_U", DIFF2D_CALLS[0][1], "u"),
            ("diff_y_T_to_V", DIFF2D_CALLS[1][1], "v"),
            ("curl", DIFF2D_CALLS[9][1], "psi"),
        ],
    )
    def test_dry_cells_zero(
        self, diff2d, fields_2d, mask2d, method, call, stagger_attr
    ):
        out = call(diff2d, fields_2d, mask2d)
        stagger_mask = np.asarray(getattr(mask2d, stagger_attr))
        out_np = np.asarray(out)
        # Every cell where the relevant stagger mask is False (dry) must be 0.
        assert np.all(out_np[~stagger_mask] == 0.0), f"{method} non-zero in dry cells"


# ----------------------------------------------------------------------
# Difference3D
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def diff3d():
    return Difference3D(grid=make_grid_3d())


@pytest.fixture(scope="module")
def fields_3d():
    return {
        "h": make_h_field_3d(),
        "u": make_u_field_3d(),
        "v": make_v_field_3d(),
    }


DIFF3D_CALLS = [
    ("diff_x_T_to_U", lambda d, f, m: d.diff_x_T_to_U(f["h"], mask=m)),
    ("diff_y_T_to_V", lambda d, f, m: d.diff_y_T_to_V(f["h"], mask=m)),
    ("diff_x_U_to_T", lambda d, f, m: d.diff_x_U_to_T(f["u"], mask=m)),
    ("diff_y_V_to_T", lambda d, f, m: d.diff_y_V_to_T(f["v"], mask=m)),
    ("divergence", lambda d, f, m: d.divergence(f["u"], f["v"], mask=m)),
    ("laplacian", lambda d, f, m: d.laplacian(f["h"], mask=m)),
]


class TestDifference3DUnmaskedGolden:
    @pytest.mark.parametrize(("method", "call"), DIFF3D_CALLS)
    def test_unmasked_matches_golden(self, diff3d, fields_3d, method, call):
        out = call(diff3d, fields_3d, None)
        assert_matches_golden(out, "Difference3D", method, "unmasked")


class TestDifference3DMaskedGolden:
    @pytest.mark.parametrize(("method", "call"), DIFF3D_CALLS)
    def test_masked_matches_golden(self, diff3d, fields_3d, mask2d, method, call):
        out = call(diff3d, fields_3d, mask2d)
        assert_matches_golden(out, "Difference3D", method, "masked")


class TestDifference3DAllOceanIsNoOp:
    @pytest.mark.parametrize(("method", "call"), DIFF3D_CALLS)
    def test_all_ocean_equals_unmasked(self, diff3d, fields_3d, method, call):
        out_none = call(diff3d, fields_3d, None)
        out_all_ocean = call(diff3d, fields_3d, make_mask_2d_all_ocean())
        np.testing.assert_array_equal(np.asarray(out_none), np.asarray(out_all_ocean))


class TestDifference3DBroadcastsOver2DMask:
    """The 2-D mask must broadcast cleanly over all z-levels.

    Difference3D zeros the z-ghost ring (k=0 and k=-1), so we only
    compare interior z-levels [1:-1] against the per-slice 2D operator.
    """

    def test_z_slices_match_2d_operator(self, diff3d, mask2d):
        from finitevolx._src.operators.difference import Difference2D

        diff2d = Difference2D(grid=diff3d.grid.horizontal_grid())
        h3d = make_h_field_3d()
        out3d = diff3d.laplacian(h3d, mask=mask2d)
        for k in range(1, h3d.shape[0] - 1):  # interior z-levels only
            out2d_k = diff2d.laplacian(h3d[k], mask=mask2d)
            np.testing.assert_allclose(
                np.asarray(out3d[k]), np.asarray(out2d_k), rtol=1e-12, atol=0
            )

    def test_z_ghost_levels_are_zero(self, diff3d):
        """The 3D operator zeros the z=0 and z=Nz-1 ghost rings."""
        h3d = make_h_field_3d()
        out = diff3d.laplacian(h3d)
        np.testing.assert_array_equal(np.asarray(out[0]), 0.0)
        np.testing.assert_array_equal(np.asarray(out[-1]), 0.0)
