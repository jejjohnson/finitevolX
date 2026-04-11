"""Golden-output regression tests for the Spherical operator mask fields.

Covers:

* SphericalDifference2D / SphericalDifference3D
* SphericalDivergence2D / SphericalDivergence3D
* SphericalVorticity2D / SphericalVorticity3D
* SphericalLaplacian2D / SphericalLaplacian3D

Each method of each operator has three golden checks:

* **unmasked** — ``mask=None`` matches the committed `.npz` bit pattern
* **masked** — output under the canonical cross-shaped Mask{2,3}D
  matches the committed `.npz` bit pattern
* **all_ocean** — ``Mask{2,3}D.from_dimensions(...)`` ≡ ``mask=None``

Plus a ``TestDryCellsAreZero`` class pinning the core semantic per
stagger, and a ``TestNoNaNFromMask`` spot-check for
SphericalVorticity2D.potential_vorticity (the NaN-safe division).

Per issue #209 Q2/Q3, the spherical operators accept a Cartesian
``Mask2D`` / ``Mask3D`` rather than a dedicated ``SphericalMask*``.
"""

from __future__ import annotations

import numpy as np
import pytest

from finitevolx._src.operators.spherical_compound import (
    SphericalDivergence2D,
    SphericalDivergence3D,
    SphericalLaplacian2D,
    SphericalLaplacian3D,
    SphericalVorticity2D,
    SphericalVorticity3D,
)
from finitevolx._src.operators.spherical_difference import (
    SphericalDifference2D,
    SphericalDifference3D,
)
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_f_field_2d,
    make_h_field_2d,
    make_h_field_3d,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_mask_3d,
    make_mask_3d_all_ocean,
    make_q_field_2d,
    make_spherical_grid_2d,
    make_spherical_grid_3d,
    make_u_field_2d,
    make_u_field_3d,
    make_v_field_2d,
    make_v_field_3d,
)

# ---------------------------------------------------------------------------
# SphericalDifference2D
# ---------------------------------------------------------------------------


_SD2_SPECS = [
    ("diff_lon_T_to_U", make_h_field_2d),
    ("diff_lat_T_to_V", make_h_field_2d),
    ("diff_lon_V_to_X", make_v_field_2d),
    ("diff_lat_U_to_X", make_u_field_2d),
    ("diff_lon_U_to_T", make_u_field_2d),
    ("diff_lat_V_to_T", make_v_field_2d),
    ("diff2_lon", make_h_field_2d),
    ("laplacian_merid", make_h_field_2d),
]


class TestSphericalDifference2DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return SphericalDifference2D(grid=make_spherical_grid_2d())

    @pytest.fixture
    def op_masked(self):
        return SphericalDifference2D(grid=make_spherical_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def op_all_ocean(self):
        return SphericalDifference2D(
            grid=make_spherical_grid_2d(), mask=make_mask_2d_all_ocean()
        )

    @pytest.mark.parametrize("method,field_fn", _SD2_SPECS)
    def test_unmasked_golden(self, op_unmasked, method, field_fn):
        out = getattr(op_unmasked, method)(field_fn())
        assert_matches_golden(out, "SphericalDifference2D", method, "unmasked")

    @pytest.mark.parametrize("method,field_fn", _SD2_SPECS)
    def test_masked_golden(self, op_masked, method, field_fn):
        out = getattr(op_masked, method)(field_fn())
        assert_matches_golden(out, "SphericalDifference2D", method, "masked")

    @pytest.mark.parametrize("method,field_fn", _SD2_SPECS)
    def test_all_ocean_matches_unmasked(
        self, op_unmasked, op_all_ocean, method, field_fn
    ):
        out_unmasked = getattr(op_unmasked, method)(field_fn())
        out_all_ocean = getattr(op_all_ocean, method)(field_fn())
        np.testing.assert_array_equal(out_all_ocean, out_unmasked)


# ---------------------------------------------------------------------------
# SphericalDifference3D
# ---------------------------------------------------------------------------


_SD3_SPECS = [
    ("diff_lon_T_to_U", make_h_field_3d),
    ("diff_lat_T_to_V", make_h_field_3d),
    ("diff_lon_V_to_X", make_v_field_3d),
    ("diff_lat_U_to_X", make_u_field_3d),
    ("diff_lon_U_to_T", make_u_field_3d),
    ("diff_lat_V_to_T", make_v_field_3d),
    ("diff2_lon", make_h_field_3d),
    ("laplacian_merid", make_h_field_3d),
]


class TestSphericalDifference3DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return SphericalDifference3D(grid=make_spherical_grid_3d())

    @pytest.fixture
    def op_masked(self):
        return SphericalDifference3D(grid=make_spherical_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def op_all_ocean(self):
        return SphericalDifference3D(
            grid=make_spherical_grid_3d(), mask=make_mask_3d_all_ocean()
        )

    @pytest.mark.parametrize("method,field_fn", _SD3_SPECS)
    def test_unmasked_golden(self, op_unmasked, method, field_fn):
        out = getattr(op_unmasked, method)(field_fn())
        assert_matches_golden(out, "SphericalDifference3D", method, "unmasked")

    @pytest.mark.parametrize("method,field_fn", _SD3_SPECS)
    def test_masked_golden(self, op_masked, method, field_fn):
        out = getattr(op_masked, method)(field_fn())
        assert_matches_golden(out, "SphericalDifference3D", method, "masked")

    @pytest.mark.parametrize("method,field_fn", _SD3_SPECS)
    def test_all_ocean_matches_unmasked(
        self, op_unmasked, op_all_ocean, method, field_fn
    ):
        out_unmasked = getattr(op_unmasked, method)(field_fn())
        out_all_ocean = getattr(op_all_ocean, method)(field_fn())
        np.testing.assert_array_equal(out_all_ocean, out_unmasked)


# ---------------------------------------------------------------------------
# SphericalDivergence2D / 3D
# ---------------------------------------------------------------------------


class TestSphericalDivergence2DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return SphericalDivergence2D(grid=make_spherical_grid_2d())

    @pytest.fixture
    def op_masked(self):
        return SphericalDivergence2D(grid=make_spherical_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def op_all_ocean(self):
        return SphericalDivergence2D(
            grid=make_spherical_grid_2d(), mask=make_mask_2d_all_ocean()
        )

    def test_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "SphericalDivergence2D", "__call__", "unmasked")

    def test_masked_golden(self, op_masked):
        out = op_masked(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "SphericalDivergence2D", "__call__", "masked")

    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean):
        u = make_u_field_2d()
        v = make_v_field_2d()
        np.testing.assert_array_equal(op_all_ocean(u, v), op_unmasked(u, v))


class TestSphericalDivergence3DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return SphericalDivergence3D(grid=make_spherical_grid_3d())

    @pytest.fixture
    def op_masked(self):
        return SphericalDivergence3D(grid=make_spherical_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def op_all_ocean(self):
        return SphericalDivergence3D(
            grid=make_spherical_grid_3d(), mask=make_mask_3d_all_ocean()
        )

    def test_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "SphericalDivergence3D", "__call__", "unmasked")

    def test_masked_golden(self, op_masked):
        out = op_masked(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "SphericalDivergence3D", "__call__", "masked")

    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean):
        u = make_u_field_3d()
        v = make_v_field_3d()
        np.testing.assert_array_equal(op_all_ocean(u, v), op_unmasked(u, v))


# ---------------------------------------------------------------------------
# SphericalVorticity2D
# ---------------------------------------------------------------------------


class TestSphericalVorticity2DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return SphericalVorticity2D(grid=make_spherical_grid_2d())

    @pytest.fixture
    def op_masked(self):
        return SphericalVorticity2D(grid=make_spherical_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def op_all_ocean(self):
        return SphericalVorticity2D(
            grid=make_spherical_grid_2d(), mask=make_mask_2d_all_ocean()
        )

    def test_relative_vorticity_unmasked(self, op_unmasked):
        out = op_unmasked.relative_vorticity(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(
            out, "SphericalVorticity2D", "relative_vorticity", "unmasked"
        )

    def test_relative_vorticity_masked(self, op_masked):
        out = op_masked.relative_vorticity(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(
            out, "SphericalVorticity2D", "relative_vorticity", "masked"
        )

    def test_relative_vorticity_all_ocean(self, op_unmasked, op_all_ocean):
        u = make_u_field_2d()
        v = make_v_field_2d()
        np.testing.assert_array_equal(
            op_all_ocean.relative_vorticity(u, v),
            op_unmasked.relative_vorticity(u, v),
        )

    def test_potential_vorticity_unmasked(self, op_unmasked):
        out = op_unmasked.potential_vorticity(
            make_u_field_2d(), make_v_field_2d(), make_h_field_2d(), make_f_field_2d()
        )
        assert_matches_golden(
            out, "SphericalVorticity2D", "potential_vorticity", "unmasked"
        )

    def test_potential_vorticity_masked(self, op_masked):
        out = op_masked.potential_vorticity(
            make_u_field_2d(), make_v_field_2d(), make_h_field_2d(), make_f_field_2d()
        )
        assert_matches_golden(
            out, "SphericalVorticity2D", "potential_vorticity", "masked"
        )

    def test_potential_vorticity_masked_has_no_nans_at_dry(self, op_masked):
        """NaN-sanitisation branch must succeed at dry X-corners."""
        out = np.asarray(
            op_masked.potential_vorticity(
                make_u_field_2d(),
                make_v_field_2d(),
                make_h_field_2d(),
                make_f_field_2d(),
            )
        )
        mask = make_mask_2d()
        dry = np.asarray(~mask.xy_corner_strict)
        assert not np.isnan(out[dry]).any()

    def test_pv_flux_energy_conserving_unmasked(self, op_unmasked):
        qu, qv = op_unmasked.pv_flux_energy_conserving(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "SphericalVorticity2D", "pv_flux_energy_conserving", "unmasked"
        )

    def test_pv_flux_energy_conserving_masked(self, op_masked):
        qu, qv = op_masked.pv_flux_energy_conserving(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "SphericalVorticity2D", "pv_flux_energy_conserving", "masked"
        )

    def test_pv_flux_enstrophy_conserving_unmasked(self, op_unmasked):
        qu, qv = op_unmasked.pv_flux_enstrophy_conserving(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv),
            "SphericalVorticity2D",
            "pv_flux_enstrophy_conserving",
            "unmasked",
        )

    def test_pv_flux_enstrophy_conserving_masked(self, op_masked):
        qu, qv = op_masked.pv_flux_enstrophy_conserving(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "SphericalVorticity2D", "pv_flux_enstrophy_conserving", "masked"
        )

    def test_pv_flux_arakawa_lamb_unmasked(self, op_unmasked):
        qu, qv = op_unmasked.pv_flux_arakawa_lamb(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "SphericalVorticity2D", "pv_flux_arakawa_lamb", "unmasked"
        )

    def test_pv_flux_arakawa_lamb_masked(self, op_masked):
        qu, qv = op_masked.pv_flux_arakawa_lamb(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "SphericalVorticity2D", "pv_flux_arakawa_lamb", "masked"
        )


class TestSphericalVorticity3DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return SphericalVorticity3D(grid=make_spherical_grid_3d())

    @pytest.fixture
    def op_masked(self):
        return SphericalVorticity3D(grid=make_spherical_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def op_all_ocean(self):
        return SphericalVorticity3D(
            grid=make_spherical_grid_3d(), mask=make_mask_3d_all_ocean()
        )

    def test_unmasked_golden(self, op_unmasked):
        out = op_unmasked.relative_vorticity(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(
            out, "SphericalVorticity3D", "relative_vorticity", "unmasked"
        )

    def test_masked_golden(self, op_masked):
        out = op_masked.relative_vorticity(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(
            out, "SphericalVorticity3D", "relative_vorticity", "masked"
        )

    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean):
        u = make_u_field_3d()
        v = make_v_field_3d()
        np.testing.assert_array_equal(
            op_all_ocean.relative_vorticity(u, v),
            op_unmasked.relative_vorticity(u, v),
        )


# ---------------------------------------------------------------------------
# SphericalLaplacian2D / 3D
# ---------------------------------------------------------------------------


class TestSphericalLaplacian2DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return SphericalLaplacian2D(grid=make_spherical_grid_2d())

    @pytest.fixture
    def op_masked(self):
        return SphericalLaplacian2D(grid=make_spherical_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def op_all_ocean(self):
        return SphericalLaplacian2D(
            grid=make_spherical_grid_2d(), mask=make_mask_2d_all_ocean()
        )

    def test_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_h_field_2d())
        assert_matches_golden(out, "SphericalLaplacian2D", "__call__", "unmasked")

    def test_masked_golden(self, op_masked):
        out = op_masked(make_h_field_2d())
        assert_matches_golden(out, "SphericalLaplacian2D", "__call__", "masked")

    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean):
        h = make_h_field_2d()
        np.testing.assert_array_equal(op_all_ocean(h), op_unmasked(h))


class TestSphericalLaplacian3DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return SphericalLaplacian3D(grid=make_spherical_grid_3d())

    @pytest.fixture
    def op_masked(self):
        return SphericalLaplacian3D(grid=make_spherical_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def op_all_ocean(self):
        return SphericalLaplacian3D(
            grid=make_spherical_grid_3d(), mask=make_mask_3d_all_ocean()
        )

    def test_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_h_field_3d())
        assert_matches_golden(out, "SphericalLaplacian3D", "__call__", "unmasked")

    def test_masked_golden(self, op_masked):
        out = op_masked(make_h_field_3d())
        assert_matches_golden(out, "SphericalLaplacian3D", "__call__", "masked")

    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean):
        h = make_h_field_3d()
        np.testing.assert_array_equal(op_all_ocean(h), op_unmasked(h))


# ---------------------------------------------------------------------------
# Dry-cell-zero invariants — one per output stagger per class
# ---------------------------------------------------------------------------


class TestDryCellsAreZero:
    def test_spherical_diff2d_u_output(self):
        mask = make_mask_2d()
        op = SphericalDifference2D(grid=make_spherical_grid_2d(), mask=mask)
        out = np.asarray(op.diff_lon_T_to_U(make_h_field_2d()))
        assert np.all(out[np.asarray(~mask.u)] == 0.0)

    def test_spherical_diff2d_v_output(self):
        mask = make_mask_2d()
        op = SphericalDifference2D(grid=make_spherical_grid_2d(), mask=mask)
        out = np.asarray(op.diff_lat_T_to_V(make_h_field_2d()))
        assert np.all(out[np.asarray(~mask.v)] == 0.0)

    def test_spherical_diff2d_t_output(self):
        mask = make_mask_2d()
        op = SphericalDifference2D(grid=make_spherical_grid_2d(), mask=mask)
        out = np.asarray(op.laplacian_merid(make_h_field_2d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_spherical_diff2d_x_output(self):
        mask = make_mask_2d()
        op = SphericalDifference2D(grid=make_spherical_grid_2d(), mask=mask)
        out = np.asarray(op.diff_lon_V_to_X(make_v_field_2d()))
        assert np.all(out[np.asarray(~mask.xy_corner_strict)] == 0.0)

    def test_spherical_diff3d_t_output(self):
        mask = make_mask_3d()
        op = SphericalDifference3D(grid=make_spherical_grid_3d(), mask=mask)
        out = np.asarray(op.laplacian_merid(make_h_field_3d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_spherical_div2d_t_output(self):
        mask = make_mask_2d()
        op = SphericalDivergence2D(grid=make_spherical_grid_2d(), mask=mask)
        out = np.asarray(op(make_u_field_2d(), make_v_field_2d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_spherical_div3d_t_output(self):
        mask = make_mask_3d()
        op = SphericalDivergence3D(grid=make_spherical_grid_3d(), mask=mask)
        out = np.asarray(op(make_u_field_3d(), make_v_field_3d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_spherical_vort2d_x_output(self):
        mask = make_mask_2d()
        op = SphericalVorticity2D(grid=make_spherical_grid_2d(), mask=mask)
        out = np.asarray(op.relative_vorticity(make_u_field_2d(), make_v_field_2d()))
        assert np.all(out[np.asarray(~mask.xy_corner_strict)] == 0.0)

    def test_spherical_vort2d_potential_x_output(self):
        mask = make_mask_2d()
        op = SphericalVorticity2D(grid=make_spherical_grid_2d(), mask=mask)
        out = np.asarray(
            op.potential_vorticity(
                make_u_field_2d(),
                make_v_field_2d(),
                make_h_field_2d(),
                make_f_field_2d(),
            )
        )
        assert np.all(out[np.asarray(~mask.xy_corner_strict)] == 0.0)

    def test_spherical_vort3d_x_output(self):
        mask = make_mask_3d()
        op = SphericalVorticity3D(grid=make_spherical_grid_3d(), mask=mask)
        out = np.asarray(op.relative_vorticity(make_u_field_3d(), make_v_field_3d()))
        assert np.all(out[np.asarray(~mask.xy_corner_strict)] == 0.0)

    def test_spherical_lap2d_t_output(self):
        mask = make_mask_2d()
        op = SphericalLaplacian2D(grid=make_spherical_grid_2d(), mask=mask)
        out = np.asarray(op(make_h_field_2d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_spherical_lap3d_t_output(self):
        mask = make_mask_3d()
        op = SphericalLaplacian3D(grid=make_spherical_grid_3d(), mask=mask)
        out = np.asarray(op(make_h_field_3d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)
