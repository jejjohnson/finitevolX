"""Golden-output regression tests for the Interpolation / Divergence /
Vorticity operator mask fields.

Each operator family gets three checks per method:

* **unmasked** — `mask=None` (the default).  Asserts the output
  matches the committed golden `.npz` — a bit-pattern regression on
  the original code path.
* **masked** — operator constructed with the canonical cross-shaped
  Mask{1,2,3}D.  Asserts the (post-compute-multiply or pass-down)
  output matches its own golden.
* **all_ocean** — operator constructed with an all-ocean mask of the
  same dimensions.  Asserts the output equals the `unmasked` output
  bit for bit — a sanity check that `mask=AllOcean` is a no-op.

A `TestDryCellsAreZero` class pins the core semantic that every dry
interior cell of the operator's output stagger is exactly zero.

A `TestCrossFaceAudit` class is the audit called out in issue #209 for
the cross-face Interpolation2D methods (U_to_V, V_to_U, U_to_X, V_to_X,
X_to_U, X_to_V).  The post-compute convention zeroes dry output cells
but leaves wet output cells with full contributions from whatever the
caller stored at dry input cells.  If this audit passes, the design
holds as written in #209; if any cross-face method produces NaN / Inf
at a wet output cell adjacent to a coast, we would need to escalate
to the intermediate-masking pattern used by Diffusion.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from finitevolx._src.operators.divergence import Divergence2D
from finitevolx._src.operators.interpolation import (
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
)
from finitevolx._src.operators.vorticity import Vorticity2D, Vorticity3D
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_f_field_2d,
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
    make_q_field_2d,
    make_u_field_1d,
    make_u_field_2d,
    make_u_field_3d,
    make_v_field_2d,
    make_v_field_3d,
)

# ---------------------------------------------------------------------------
# Interpolation1D
# ---------------------------------------------------------------------------


class TestInterpolation1DMasks:
    @pytest.fixture
    def i_unmasked(self):
        return Interpolation1D(grid=make_grid_1d())

    @pytest.fixture
    def i_masked(self):
        return Interpolation1D(grid=make_grid_1d(), mask=make_mask_1d())

    @pytest.fixture
    def i_all_ocean(self):
        return Interpolation1D(grid=make_grid_1d(), mask=make_mask_1d_all_ocean())

    @pytest.mark.parametrize(
        "method,field_fn",
        [("T_to_U", make_h_field_1d), ("U_to_T", make_u_field_1d)],
    )
    def test_unmasked_golden(self, i_unmasked, method, field_fn):
        out = getattr(i_unmasked, method)(field_fn())
        assert_matches_golden(out, "Interpolation1D", method, "unmasked")

    @pytest.mark.parametrize(
        "method,field_fn",
        [("T_to_U", make_h_field_1d), ("U_to_T", make_u_field_1d)],
    )
    def test_masked_golden(self, i_masked, method, field_fn):
        out = getattr(i_masked, method)(field_fn())
        assert_matches_golden(out, "Interpolation1D", method, "masked")

    @pytest.mark.parametrize(
        "method,field_fn",
        [("T_to_U", make_h_field_1d), ("U_to_T", make_u_field_1d)],
    )
    def test_all_ocean_matches_unmasked(
        self, i_unmasked, i_all_ocean, method, field_fn
    ):
        out_unmasked = getattr(i_unmasked, method)(field_fn())
        out_all_ocean = getattr(i_all_ocean, method)(field_fn())
        np.testing.assert_array_equal(out_all_ocean, out_unmasked)


# ---------------------------------------------------------------------------
# Interpolation2D
# ---------------------------------------------------------------------------


_I2_SPECS = [
    ("T_to_U", make_h_field_2d),
    ("T_to_V", make_h_field_2d),
    ("T_to_X", make_h_field_2d),
    ("X_to_U", make_q_field_2d),
    ("X_to_V", make_q_field_2d),
    ("U_to_T", make_u_field_2d),
    ("V_to_T", make_v_field_2d),
    ("X_to_T", make_q_field_2d),
    ("U_to_X", make_u_field_2d),
    ("V_to_X", make_v_field_2d),
    ("U_to_V", make_u_field_2d),
    ("V_to_U", make_v_field_2d),
]


class TestInterpolation2DMasks:
    @pytest.fixture
    def i_unmasked(self):
        return Interpolation2D(grid=make_grid_2d())

    @pytest.fixture
    def i_masked(self):
        return Interpolation2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def i_all_ocean(self):
        return Interpolation2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    @pytest.mark.parametrize("method,field_fn", _I2_SPECS)
    def test_unmasked_golden(self, i_unmasked, method, field_fn):
        out = getattr(i_unmasked, method)(field_fn())
        assert_matches_golden(out, "Interpolation2D", method, "unmasked")

    @pytest.mark.parametrize("method,field_fn", _I2_SPECS)
    def test_masked_golden(self, i_masked, method, field_fn):
        out = getattr(i_masked, method)(field_fn())
        assert_matches_golden(out, "Interpolation2D", method, "masked")

    @pytest.mark.parametrize("method,field_fn", _I2_SPECS)
    def test_all_ocean_matches_unmasked(
        self, i_unmasked, i_all_ocean, method, field_fn
    ):
        out_unmasked = getattr(i_unmasked, method)(field_fn())
        out_all_ocean = getattr(i_all_ocean, method)(field_fn())
        np.testing.assert_array_equal(out_all_ocean, out_unmasked)


# ---------------------------------------------------------------------------
# Interpolation3D
# ---------------------------------------------------------------------------


_I3_SPECS = [
    ("T_to_U", make_h_field_3d),
    ("T_to_V", make_h_field_3d),
    ("U_to_T", make_u_field_3d),
    ("V_to_T", make_v_field_3d),
]


class TestInterpolation3DMasks:
    @pytest.fixture
    def i_unmasked(self):
        return Interpolation3D(grid=make_grid_3d())

    @pytest.fixture
    def i_masked(self):
        return Interpolation3D(grid=make_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def i_all_ocean(self):
        return Interpolation3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())

    @pytest.mark.parametrize("method,field_fn", _I3_SPECS)
    def test_unmasked_golden(self, i_unmasked, method, field_fn):
        out = getattr(i_unmasked, method)(field_fn())
        assert_matches_golden(out, "Interpolation3D", method, "unmasked")

    @pytest.mark.parametrize("method,field_fn", _I3_SPECS)
    def test_masked_golden(self, i_masked, method, field_fn):
        out = getattr(i_masked, method)(field_fn())
        assert_matches_golden(out, "Interpolation3D", method, "masked")

    @pytest.mark.parametrize("method,field_fn", _I3_SPECS)
    def test_all_ocean_matches_unmasked(
        self, i_unmasked, i_all_ocean, method, field_fn
    ):
        out_unmasked = getattr(i_unmasked, method)(field_fn())
        out_all_ocean = getattr(i_all_ocean, method)(field_fn())
        np.testing.assert_array_equal(out_all_ocean, out_unmasked)


# ---------------------------------------------------------------------------
# Divergence2D
# ---------------------------------------------------------------------------


class TestDivergence2DMasks:
    @pytest.fixture
    def div_unmasked(self):
        return Divergence2D(grid=make_grid_2d())

    @pytest.fixture
    def div_masked(self):
        return Divergence2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def div_all_ocean(self):
        return Divergence2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    def test_call_unmasked(self, div_unmasked):
        out = div_unmasked(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Divergence2D", "__call__", "unmasked")

    def test_call_masked(self, div_masked):
        out = div_masked(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Divergence2D", "__call__", "masked")

    def test_call_all_ocean(self, div_unmasked, div_all_ocean):
        u = make_u_field_2d()
        v = make_v_field_2d()
        np.testing.assert_array_equal(div_all_ocean(u, v), div_unmasked(u, v))

    def test_noflux_unmasked(self, div_unmasked):
        out = div_unmasked.noflux(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Divergence2D", "noflux", "unmasked")

    def test_noflux_masked(self, div_masked):
        out = div_masked.noflux(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Divergence2D", "noflux", "masked")

    def test_noflux_all_ocean(self, div_unmasked, div_all_ocean):
        u = make_u_field_2d()
        v = make_v_field_2d()
        np.testing.assert_array_equal(
            div_all_ocean.noflux(u, v),
            div_unmasked.noflux(u, v),
        )


# ---------------------------------------------------------------------------
# Vorticity2D
# ---------------------------------------------------------------------------


class TestVorticity2DMasks:
    @pytest.fixture
    def vort_unmasked(self):
        return Vorticity2D(grid=make_grid_2d())

    @pytest.fixture
    def vort_masked(self):
        return Vorticity2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def vort_all_ocean(self):
        return Vorticity2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    def test_relative_vorticity_unmasked(self, vort_unmasked):
        out = vort_unmasked.relative_vorticity(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Vorticity2D", "relative_vorticity", "unmasked")

    def test_relative_vorticity_masked(self, vort_masked):
        out = vort_masked.relative_vorticity(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "Vorticity2D", "relative_vorticity", "masked")

    def test_relative_vorticity_all_ocean(self, vort_unmasked, vort_all_ocean):
        u = make_u_field_2d()
        v = make_v_field_2d()
        np.testing.assert_array_equal(
            vort_all_ocean.relative_vorticity(u, v),
            vort_unmasked.relative_vorticity(u, v),
        )

    def test_potential_vorticity_unmasked(self, vort_unmasked):
        out = vort_unmasked.potential_vorticity(
            make_u_field_2d(), make_v_field_2d(), make_h_field_2d(), make_f_field_2d()
        )
        assert_matches_golden(out, "Vorticity2D", "potential_vorticity", "unmasked")

    def test_potential_vorticity_masked(self, vort_masked):
        out = vort_masked.potential_vorticity(
            make_u_field_2d(), make_v_field_2d(), make_h_field_2d(), make_f_field_2d()
        )
        assert_matches_golden(out, "Vorticity2D", "potential_vorticity", "masked")

    def test_potential_vorticity_masked_has_no_nans_at_dry(self, vort_masked):
        """The NaN-sanitisation branch must succeed at dry X-corners."""
        out = np.asarray(
            vort_masked.potential_vorticity(
                make_u_field_2d(),
                make_v_field_2d(),
                make_h_field_2d(),
                make_f_field_2d(),
            )
        )
        mask = make_mask_2d()
        dry = np.asarray(~mask.xy_corner_strict)
        assert not np.isnan(out[dry]).any(), (
            "potential_vorticity produced NaN at dry X-corners — the NaN-safe "
            "branch in Vorticity2D.potential_vorticity is broken."
        )

    def test_pv_flux_energy_conserving_unmasked(self, vort_unmasked):
        qu, qv = vort_unmasked.pv_flux_energy_conserving(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "Vorticity2D", "pv_flux_energy_conserving", "unmasked"
        )

    def test_pv_flux_energy_conserving_masked(self, vort_masked):
        qu, qv = vort_masked.pv_flux_energy_conserving(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "Vorticity2D", "pv_flux_energy_conserving", "masked"
        )

    def test_pv_flux_enstrophy_conserving_unmasked(self, vort_unmasked):
        qu, qv = vort_unmasked.pv_flux_enstrophy_conserving(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "Vorticity2D", "pv_flux_enstrophy_conserving", "unmasked"
        )

    def test_pv_flux_enstrophy_conserving_masked(self, vort_masked):
        qu, qv = vort_masked.pv_flux_enstrophy_conserving(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "Vorticity2D", "pv_flux_enstrophy_conserving", "masked"
        )

    def test_pv_flux_arakawa_lamb_unmasked(self, vort_unmasked):
        qu, qv = vort_unmasked.pv_flux_arakawa_lamb(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden(
            (qu, qv), "Vorticity2D", "pv_flux_arakawa_lamb", "unmasked"
        )

    def test_pv_flux_arakawa_lamb_masked(self, vort_masked):
        qu, qv = vort_masked.pv_flux_arakawa_lamb(
            make_q_field_2d(), make_u_field_2d(), make_v_field_2d()
        )
        assert_matches_golden((qu, qv), "Vorticity2D", "pv_flux_arakawa_lamb", "masked")


# ---------------------------------------------------------------------------
# Vorticity3D
# ---------------------------------------------------------------------------


class TestVorticity3DMasks:
    @pytest.fixture
    def vort_unmasked(self):
        return Vorticity3D(grid=make_grid_3d())

    @pytest.fixture
    def vort_masked(self):
        return Vorticity3D(grid=make_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def vort_all_ocean(self):
        return Vorticity3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())

    def test_relative_vorticity_unmasked(self, vort_unmasked):
        out = vort_unmasked.relative_vorticity(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "Vorticity3D", "relative_vorticity", "unmasked")

    def test_relative_vorticity_masked(self, vort_masked):
        out = vort_masked.relative_vorticity(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "Vorticity3D", "relative_vorticity", "masked")

    def test_relative_vorticity_all_ocean(self, vort_unmasked, vort_all_ocean):
        u = make_u_field_3d()
        v = make_v_field_3d()
        np.testing.assert_array_equal(
            vort_all_ocean.relative_vorticity(u, v),
            vort_unmasked.relative_vorticity(u, v),
        )


# ---------------------------------------------------------------------------
# Dry-cell-zero invariants
# ---------------------------------------------------------------------------


class TestDryCellsAreZero:
    """Every dry interior cell of the operator's output stagger is
    **exactly** 0 in the masked path.  One representative method per
    output stagger for each family.
    """

    # Interpolation1D
    def test_interp1d_t_output(self):
        mask = make_mask_1d()
        op = Interpolation1D(grid=make_grid_1d(), mask=mask)
        out = np.asarray(op.U_to_T(make_u_field_1d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_interp1d_u_output(self):
        mask = make_mask_1d()
        op = Interpolation1D(grid=make_grid_1d(), mask=mask)
        out = np.asarray(op.T_to_U(make_h_field_1d()))
        assert np.all(out[np.asarray(~mask.u)] == 0.0)

    # Interpolation2D
    def test_interp2d_t_output(self):
        mask = make_mask_2d()
        op = Interpolation2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op.U_to_T(make_u_field_2d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_interp2d_u_output(self):
        mask = make_mask_2d()
        op = Interpolation2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op.T_to_U(make_h_field_2d()))
        assert np.all(out[np.asarray(~mask.u)] == 0.0)

    def test_interp2d_v_output(self):
        mask = make_mask_2d()
        op = Interpolation2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op.T_to_V(make_h_field_2d()))
        assert np.all(out[np.asarray(~mask.v)] == 0.0)

    def test_interp2d_x_output(self):
        mask = make_mask_2d()
        op = Interpolation2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op.T_to_X(make_h_field_2d()))
        assert np.all(out[np.asarray(~mask.xy_corner_strict)] == 0.0)

    # Interpolation3D
    def test_interp3d_t_output(self):
        mask = make_mask_3d()
        op = Interpolation3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(op.U_to_T(make_u_field_3d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_interp3d_u_output(self):
        mask = make_mask_3d()
        op = Interpolation3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(op.T_to_U(make_h_field_3d()))
        assert np.all(out[np.asarray(~mask.u)] == 0.0)

    # Divergence2D (T-output)
    def test_divergence2d_t_output(self):
        mask = make_mask_2d()
        op = Divergence2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op(make_u_field_2d(), make_v_field_2d()))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    # Vorticity2D (X-output)
    def test_vorticity2d_x_output(self):
        mask = make_mask_2d()
        op = Vorticity2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op.relative_vorticity(make_u_field_2d(), make_v_field_2d()))
        assert np.all(out[np.asarray(~mask.xy_corner_strict)] == 0.0)

    def test_vorticity2d_potential_x_output(self):
        mask = make_mask_2d()
        op = Vorticity2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(
            op.potential_vorticity(
                make_u_field_2d(),
                make_v_field_2d(),
                make_h_field_2d(),
                make_f_field_2d(),
            )
        )
        assert np.all(out[np.asarray(~mask.xy_corner_strict)] == 0.0)

    # Vorticity3D (X-output)
    def test_vorticity3d_x_output(self):
        mask = make_mask_3d()
        op = Vorticity3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(op.relative_vorticity(make_u_field_3d(), make_v_field_3d()))
        assert np.all(out[np.asarray(~mask.xy_corner_strict)] == 0.0)


# ---------------------------------------------------------------------------
# Cross-face audit (issue #209 Q9)
# ---------------------------------------------------------------------------


class TestCrossFaceAudit:
    """Audit Interpolation2D cross-face methods (U↔V, U↔X, V↔X, X↔U, X↔V).

    These methods read a bilinear 4-point stencil that crosses the
    cell-centre / corner / face staggering.  Near a coast the stencil
    includes both wet and dry input cells.  The post-compute convention
    means:

    * **Dry output cells**: exactly 0 (via ``* mask.<stagger>``) — the
      promise we advertise.
    * **Wet output cells**: a possibly-contaminated bilinear average
      that includes whatever the caller stored at the dry input cells.

    This audit pins both halves of the semantic.  If the first check
    fails, the post-compute multiply is broken.  If the second check
    fails (non-finite values at wet cells under smooth analytic input),
    we would need to escalate to the Diffusion-style intermediate
    masking pattern — the audit the user asked for in #209 Q9.
    """

    CROSS_FACE_SPECS: typing.ClassVar = [
        ("U_to_V", make_u_field_2d, "v"),
        ("V_to_U", make_v_field_2d, "u"),
        ("U_to_X", make_u_field_2d, "xy_corner_strict"),
        ("V_to_X", make_v_field_2d, "xy_corner_strict"),
        ("X_to_U", make_q_field_2d, "u"),
        ("X_to_V", make_q_field_2d, "v"),
    ]

    @pytest.mark.parametrize("method,field_fn,output_stagger", CROSS_FACE_SPECS)
    def test_dry_output_is_zero(self, method, field_fn, output_stagger):
        mask = make_mask_2d()
        op = Interpolation2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(getattr(op, method)(field_fn()))
        dry = np.asarray(~getattr(mask, output_stagger))
        assert np.all(out[dry] == 0.0), (
            f"Interpolation2D.{method} leaked non-zero values into dry "
            f"{output_stagger} cells under the post-compute convention."
        )

    @pytest.mark.parametrize("method,field_fn,output_stagger", CROSS_FACE_SPECS)
    def test_wet_output_is_finite(self, method, field_fn, output_stagger):
        """No NaN/Inf at wet output cells under smooth analytic input.

        If this fails for a smooth sinusoidal input, it means the
        cross-face stencil is reading a NaN/Inf from somewhere — either
        from a bug in the stencil or from the caller storing a bad
        value at a dry input cell.  In our canonical fixtures the input
        fields are fully finite analytic expressions, so a failure here
        would indicate a bug in the operator.
        """
        mask = make_mask_2d()
        op = Interpolation2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(getattr(op, method)(field_fn()))
        wet = np.asarray(getattr(mask, output_stagger))
        assert np.all(np.isfinite(out[wet])), (
            f"Interpolation2D.{method} produced non-finite values at wet "
            f"{output_stagger} cells."
        )
