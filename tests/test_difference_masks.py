"""Golden-output regression tests for the Difference1D/2D/3D mask field.

Each test exercises one of three code paths for every method on each
class:

* **unmasked** — ``mask=None`` (the default).  Asserts the output
  matches the committed golden ``.npz`` — a bit-pattern regression on
  the original code path.
* **masked** — operator constructed with the canonical cross-shaped
  ``Mask{1,2,3}D``.  Asserts the post-compute-multiply output matches
  its own golden.
* **all_ocean** — operator constructed with an all-ocean mask of the
  same dimensions.  Asserts the output equals the ``unmasked`` output
  bit for bit — a sanity check that ``mask=AllOcean`` is a no-op.

A separate ``TestDryCellsAreZero`` class pins the semantic that every
dry T-cell (or dry U/V/X face) in the masked output is **exactly** 0.

See ``tests/fixtures/`` for the shared inputs and golden helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from finitevolx._src.operators.difference import (
    Difference1D,
    Difference2D,
    Difference3D,
)
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_grid_1d,
    make_grid_2d,
    make_grid_3d,
    make_h_field_1d,
    make_h_field_2d,
    make_h_field_3d,
    make_mask_1d,
    make_mask_1d_all_ocean,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_mask_3d,
    make_mask_3d_all_ocean,
    make_u_field_1d,
    make_u_field_2d,
    make_u_field_3d,
    make_v_field_2d,
    make_v_field_3d,
)

# ---------------------------------------------------------------------------
# Difference1D
# ---------------------------------------------------------------------------


class TestDifference1DMasks:
    @pytest.fixture
    def diff_unmasked(self):
        return Difference1D(grid=make_grid_1d())

    @pytest.fixture
    def diff_masked(self):
        return Difference1D(grid=make_grid_1d(), mask=make_mask_1d())

    @pytest.fixture
    def diff_all_ocean(self):
        return Difference1D(grid=make_grid_1d(), mask=make_mask_1d_all_ocean())

    @pytest.mark.parametrize(
        "method,field_fn",
        [
            ("diff_x_T_to_U", make_h_field_1d),
            ("diff_x_U_to_T", make_u_field_1d),
            ("laplacian", make_h_field_1d),
        ],
    )
    def test_unmasked_golden(self, diff_unmasked, method, field_fn):
        out = getattr(diff_unmasked, method)(field_fn())
        assert_matches_golden(out, "Difference1D", method, "unmasked")

    @pytest.mark.parametrize(
        "method,field_fn",
        [
            ("diff_x_T_to_U", make_h_field_1d),
            ("diff_x_U_to_T", make_u_field_1d),
            ("laplacian", make_h_field_1d),
        ],
    )
    def test_masked_golden(self, diff_masked, method, field_fn):
        out = getattr(diff_masked, method)(field_fn())
        assert_matches_golden(out, "Difference1D", method, "masked")

    @pytest.mark.parametrize(
        "method,field_fn",
        [
            ("diff_x_T_to_U", make_h_field_1d),
            ("diff_x_U_to_T", make_u_field_1d),
            ("laplacian", make_h_field_1d),
        ],
    )
    def test_all_ocean_matches_unmasked(
        self, diff_unmasked, diff_all_ocean, method, field_fn
    ):
        out_unmasked = getattr(diff_unmasked, method)(field_fn())
        out_all_ocean = getattr(diff_all_ocean, method)(field_fn())
        np.testing.assert_array_equal(out_all_ocean, out_unmasked)


# ---------------------------------------------------------------------------
# Difference2D
# ---------------------------------------------------------------------------


_D2_UNARY_SPECS = [
    ("diff_x_T_to_U", make_h_field_2d),
    ("diff_y_T_to_V", make_h_field_2d),
    ("diff_y_U_to_X", make_u_field_2d),
    ("diff_x_V_to_X", make_v_field_2d),
    ("diff_y_X_to_U", make_h_field_2d),
    ("diff_x_X_to_V", make_h_field_2d),
    ("diff_x_U_to_T", make_u_field_2d),
    ("diff_y_V_to_T", make_v_field_2d),
    ("laplacian", make_h_field_2d),
]


class TestDifference2DUnaryMasks:
    @pytest.fixture
    def diff_unmasked(self):
        return Difference2D(grid=make_grid_2d())

    @pytest.fixture
    def diff_masked(self):
        return Difference2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def diff_all_ocean(self):
        return Difference2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    @pytest.mark.parametrize("method,field_fn", _D2_UNARY_SPECS)
    def test_unmasked_golden(self, diff_unmasked, method, field_fn):
        out = getattr(diff_unmasked, method)(field_fn())
        assert_matches_golden(out, "Difference2D", method, "unmasked")

    @pytest.mark.parametrize("method,field_fn", _D2_UNARY_SPECS)
    def test_masked_golden(self, diff_masked, method, field_fn):
        out = getattr(diff_masked, method)(field_fn())
        assert_matches_golden(out, "Difference2D", method, "masked")

    @pytest.mark.parametrize("method,field_fn", _D2_UNARY_SPECS)
    def test_all_ocean_matches_unmasked(
        self, diff_unmasked, diff_all_ocean, method, field_fn
    ):
        out_unmasked = getattr(diff_unmasked, method)(field_fn())
        out_all_ocean = getattr(diff_all_ocean, method)(field_fn())
        np.testing.assert_array_equal(out_all_ocean, out_unmasked)


class TestDifference2DCompoundMasks:
    @pytest.fixture
    def diff_unmasked(self):
        return Difference2D(grid=make_grid_2d())

    @pytest.fixture
    def diff_masked(self):
        return Difference2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def diff_all_ocean(self):
        return Difference2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    def test_divergence_unmasked(self, diff_unmasked):
        out = diff_unmasked.divergence(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Difference2D", "divergence", "unmasked")

    def test_divergence_masked(self, diff_masked):
        out = diff_masked.divergence(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Difference2D", "divergence", "masked")

    def test_divergence_all_ocean(self, diff_unmasked, diff_all_ocean):
        u = make_u_field_2d()
        v = make_v_field_2d()
        np.testing.assert_array_equal(
            diff_all_ocean.divergence(u, v),
            diff_unmasked.divergence(u, v),
        )

    def test_curl_unmasked(self, diff_unmasked):
        out = diff_unmasked.curl(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Difference2D", "curl", "unmasked")

    def test_curl_masked(self, diff_masked):
        out = diff_masked.curl(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Difference2D", "curl", "masked")

    def test_curl_all_ocean(self, diff_unmasked, diff_all_ocean):
        u = make_u_field_2d()
        v = make_v_field_2d()
        np.testing.assert_array_equal(
            diff_all_ocean.curl(u, v),
            diff_unmasked.curl(u, v),
        )

    def test_grad_perp_unmasked(self, diff_unmasked):
        out = diff_unmasked.grad_perp(make_h_field_2d())
        assert_matches_golden(out, "Difference2D", "grad_perp", "unmasked")

    def test_grad_perp_masked(self, diff_masked):
        out = diff_masked.grad_perp(make_h_field_2d())
        assert_matches_golden(out, "Difference2D", "grad_perp", "masked")

    def test_grad_perp_all_ocean(self, diff_unmasked, diff_all_ocean):
        psi = make_h_field_2d()
        u1, v1 = diff_unmasked.grad_perp(psi)
        u2, v2 = diff_all_ocean.grad_perp(psi)
        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Difference3D
# ---------------------------------------------------------------------------


_D3_UNARY_SPECS = [
    ("diff_x_T_to_U", make_h_field_3d),
    ("diff_y_T_to_V", make_h_field_3d),
    ("diff_x_U_to_T", make_u_field_3d),
    ("diff_y_V_to_T", make_v_field_3d),
    ("laplacian", make_h_field_3d),
]


class TestDifference3DMasks:
    @pytest.fixture
    def diff_unmasked(self):
        return Difference3D(grid=make_grid_3d())

    @pytest.fixture
    def diff_masked(self):
        return Difference3D(grid=make_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def diff_all_ocean(self):
        return Difference3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())

    @pytest.mark.parametrize("method,field_fn", _D3_UNARY_SPECS)
    def test_unmasked_golden(self, diff_unmasked, method, field_fn):
        out = getattr(diff_unmasked, method)(field_fn())
        assert_matches_golden(out, "Difference3D", method, "unmasked")

    @pytest.mark.parametrize("method,field_fn", _D3_UNARY_SPECS)
    def test_masked_golden(self, diff_masked, method, field_fn):
        out = getattr(diff_masked, method)(field_fn())
        assert_matches_golden(out, "Difference3D", method, "masked")

    @pytest.mark.parametrize("method,field_fn", _D3_UNARY_SPECS)
    def test_all_ocean_matches_unmasked(
        self, diff_unmasked, diff_all_ocean, method, field_fn
    ):
        out_unmasked = getattr(diff_unmasked, method)(field_fn())
        out_all_ocean = getattr(diff_all_ocean, method)(field_fn())
        np.testing.assert_array_equal(out_all_ocean, out_unmasked)

    def test_divergence_unmasked(self, diff_unmasked):
        out = diff_unmasked.divergence(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "Difference3D", "divergence", "unmasked")

    def test_divergence_masked(self, diff_masked):
        out = diff_masked.divergence(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "Difference3D", "divergence", "masked")

    def test_divergence_all_ocean(self, diff_unmasked, diff_all_ocean):
        u = make_u_field_3d()
        v = make_v_field_3d()
        np.testing.assert_array_equal(
            diff_all_ocean.divergence(u, v),
            diff_unmasked.divergence(u, v),
        )


# ---------------------------------------------------------------------------
# Dry-cell-zero invariants (the key semantic the post-compute multiply pins)
# ---------------------------------------------------------------------------


class TestDryCellsAreZero:
    """Every dry interior cell of the operator's output stagger is *exactly* 0.

    One representative method per output stagger.
    """

    def test_t_output_laplacian_2d(self):
        mask = make_mask_2d()
        diff = Difference2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(diff.laplacian(make_h_field_2d()))
        dry = np.asarray(~mask.h)
        assert np.all(out[dry] == 0.0)

    def test_u_output_diff_x_T_to_U_2d(self):
        mask = make_mask_2d()
        diff = Difference2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(diff.diff_x_T_to_U(make_h_field_2d()))
        dry = np.asarray(~mask.u)
        assert np.all(out[dry] == 0.0)

    def test_v_output_diff_y_T_to_V_2d(self):
        mask = make_mask_2d()
        diff = Difference2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(diff.diff_y_T_to_V(make_h_field_2d()))
        dry = np.asarray(~mask.v)
        assert np.all(out[dry] == 0.0)

    def test_x_output_curl_2d(self):
        mask = make_mask_2d()
        diff = Difference2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(diff.curl(make_u_field_2d(), make_v_field_2d()))
        dry = np.asarray(~mask.xy_corner_strict)
        assert np.all(out[dry] == 0.0)

    def test_grad_perp_u_and_v_components_2d(self):
        mask = make_mask_2d()
        diff = Difference2D(grid=make_grid_2d(), mask=mask)
        u, v = diff.grad_perp(make_h_field_2d())
        u = np.asarray(u)
        v = np.asarray(v)
        assert np.all(u[np.asarray(~mask.u)] == 0.0)
        assert np.all(v[np.asarray(~mask.v)] == 0.0)

    def test_t_output_laplacian_1d(self):
        mask = make_mask_1d()
        diff = Difference1D(grid=make_grid_1d(), mask=mask)
        out = np.asarray(diff.laplacian(make_h_field_1d()))
        dry = np.asarray(~mask.h)
        assert np.all(out[dry] == 0.0)

    def test_u_output_diff_x_T_to_U_1d(self):
        mask = make_mask_1d()
        diff = Difference1D(grid=make_grid_1d(), mask=mask)
        out = np.asarray(diff.diff_x_T_to_U(make_h_field_1d()))
        dry = np.asarray(~mask.u)
        assert np.all(out[dry] == 0.0)

    def test_t_output_divergence_3d(self):
        mask = make_mask_3d()
        diff = Difference3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(diff.divergence(make_u_field_3d(), make_v_field_3d()))
        dry = np.asarray(~mask.h)
        assert np.all(out[dry] == 0.0)
