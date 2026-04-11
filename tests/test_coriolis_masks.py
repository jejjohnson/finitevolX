"""Golden-output regression tests for the Coriolis2D / Coriolis3D mask field.

Coriolis already took a per-call ``mask=`` kwarg before this branch; Phase 5
promotes it to a class attribute so the mask travels with the operator.
Coriolis3D additionally pivots from ``Mask2D`` to ``Mask3D`` for
type-uniformity with the rest of the 3-D suite (issue #209 Q4).

Each operator × method × 3 variants (unmasked golden, masked golden,
all-ocean invariant), plus a dry-cell-zero pin per velocity stagger.
"""

from __future__ import annotations

import numpy as np
import pytest

from finitevolx._src.operators.coriolis import Coriolis2D, Coriolis3D
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_f_field_2d,
    make_grid_2d,
    make_grid_3d,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_mask_3d,
    make_mask_3d_all_ocean,
    make_u_field_2d,
    make_u_field_3d,
    make_v_field_2d,
    make_v_field_3d,
)


class TestCoriolis2DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Coriolis2D(grid=make_grid_2d())

    @pytest.fixture
    def op_masked(self):
        return Coriolis2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def op_all_ocean(self):
        return Coriolis2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    def test_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_u_field_2d(), make_v_field_2d(), make_f_field_2d())
        assert_matches_golden(out, "Coriolis2D", "__call__", "unmasked")

    def test_masked_golden(self, op_masked):
        out = op_masked(make_u_field_2d(), make_v_field_2d(), make_f_field_2d())
        assert_matches_golden(out, "Coriolis2D", "__call__", "masked")

    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean):
        u = make_u_field_2d()
        v = make_v_field_2d()
        f = make_f_field_2d()
        du1, dv1 = op_unmasked(u, v, f)
        du2, dv2 = op_all_ocean(u, v, f)
        np.testing.assert_array_equal(du1, du2)
        np.testing.assert_array_equal(dv1, dv2)


class TestCoriolis3DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Coriolis3D(grid=make_grid_3d())

    @pytest.fixture
    def op_masked(self):
        return Coriolis3D(grid=make_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def op_all_ocean(self):
        return Coriolis3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())

    def test_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_u_field_3d(), make_v_field_3d(), make_f_field_2d())
        assert_matches_golden(out, "Coriolis3D", "__call__", "unmasked")

    def test_masked_golden(self, op_masked):
        out = op_masked(make_u_field_3d(), make_v_field_3d(), make_f_field_2d())
        assert_matches_golden(out, "Coriolis3D", "__call__", "masked")

    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean):
        u = make_u_field_3d()
        v = make_v_field_3d()
        f = make_f_field_2d()
        du1, dv1 = op_unmasked(u, v, f)
        du2, dv2 = op_all_ocean(u, v, f)
        np.testing.assert_array_equal(du1, du2)
        np.testing.assert_array_equal(dv1, dv2)


class TestDryCellsAreZero:
    """``du_cor`` is exact 0 at dry U-faces; ``dv_cor`` at dry V-faces."""

    def test_coriolis2d_dry_u_and_v(self):
        mask = make_mask_2d()
        op = Coriolis2D(grid=make_grid_2d(), mask=mask)
        du, dv = op(make_u_field_2d(), make_v_field_2d(), make_f_field_2d())
        du = np.asarray(du)
        dv = np.asarray(dv)
        assert np.all(du[np.asarray(~mask.u)] == 0.0)
        assert np.all(dv[np.asarray(~mask.v)] == 0.0)

    def test_coriolis3d_dry_u_and_v(self):
        mask = make_mask_3d()
        op = Coriolis3D(grid=make_grid_3d(), mask=mask)
        du, dv = op(make_u_field_3d(), make_v_field_3d(), make_f_field_2d())
        du = np.asarray(du)
        dv = np.asarray(dv)
        assert np.all(du[np.asarray(~mask.u)] == 0.0)
        assert np.all(dv[np.asarray(~mask.v)] == 0.0)
