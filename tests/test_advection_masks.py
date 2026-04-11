"""Golden-output regression tests for the Advection1D / 2D / 3D mask field.

Phase 6 promotes ``mask`` from a per-call kwarg to a class field on every
advection operator.  This adds:

* Mask1D support on Advection1D (previously unmasked-only).
* Mask2D-field promotion on Advection2D, with the ``(2, 4, 6)`` adaptive
  stencil hierarchy pre-built once in ``__init__``.
* Mask3D pivot on Advection3D (was ``Mask2D``) with native 3-D
  hierarchies pre-built in ``__init__``.

Tests use the canonical cross-shaped masks from ``tests/fixtures/inputs.py``
and pin two representative methods per dimension:

* ``upwind1`` — the non-mask-dispatchable code path: the output takes the
  unmasked reconstruction path and is then post-multiplied by ``mask.h``
  so dry T-cells are exactly 0.
* ``weno5`` — the mask-dispatchable code path: ``upwind_flux`` is used
  with the pre-built adaptive hierarchy (narrowed to the method's stencil
  sizes).

Plus a ``TestDryCellsAreZero`` class pinning the exact-zero semantic for
every dry T-cell across both paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from finitevolx._src.advection.advection import (
    Advection1D,
    Advection2D,
    Advection3D,
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
# Advection1D
# ---------------------------------------------------------------------------


class TestAdvection1DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Advection1D(grid=make_grid_1d())

    @pytest.fixture
    def op_masked(self):
        return Advection1D(grid=make_grid_1d(), mask=make_mask_1d())

    @pytest.fixture
    def op_all_ocean(self):
        return Advection1D(grid=make_grid_1d(), mask=make_mask_1d_all_ocean())

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_unmasked_golden(self, op_unmasked, method):
        out = op_unmasked(make_h_field_1d(), make_u_field_1d(), method=method)
        assert_matches_golden(out, "Advection1D", f"__call___{method}", "unmasked")

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_masked_golden(self, op_masked, method):
        out = op_masked(make_h_field_1d(), make_u_field_1d(), method=method)
        assert_matches_golden(out, "Advection1D", f"__call___{method}", "masked")

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean, method):
        h = make_h_field_1d()
        u = make_u_field_1d()
        np.testing.assert_array_equal(
            op_all_ocean(h, u, method=method),
            op_unmasked(h, u, method=method),
        )


# ---------------------------------------------------------------------------
# Advection2D
# ---------------------------------------------------------------------------


class TestAdvection2DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Advection2D(grid=make_grid_2d())

    @pytest.fixture
    def op_masked(self):
        return Advection2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def op_all_ocean(self):
        return Advection2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_unmasked_golden(self, op_unmasked, method):
        out = op_unmasked(
            make_h_field_2d(), make_u_field_2d(), make_v_field_2d(), method=method
        )
        assert_matches_golden(out, "Advection2D", f"__call___{method}", "unmasked")

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_masked_golden(self, op_masked, method):
        out = op_masked(
            make_h_field_2d(), make_u_field_2d(), make_v_field_2d(), method=method
        )
        assert_matches_golden(out, "Advection2D", f"__call___{method}", "masked")

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean, method):
        h = make_h_field_2d()
        u = make_u_field_2d()
        v = make_v_field_2d()
        np.testing.assert_array_equal(
            op_all_ocean(h, u, v, method=method),
            op_unmasked(h, u, v, method=method),
        )


# ---------------------------------------------------------------------------
# Advection3D
# ---------------------------------------------------------------------------


class TestAdvection3DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Advection3D(grid=make_grid_3d())

    @pytest.fixture
    def op_masked(self):
        return Advection3D(grid=make_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def op_all_ocean(self):
        return Advection3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_unmasked_golden(self, op_unmasked, method):
        out = op_unmasked(
            make_h_field_3d(), make_u_field_3d(), make_v_field_3d(), method=method
        )
        assert_matches_golden(out, "Advection3D", f"__call___{method}", "unmasked")

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_masked_golden(self, op_masked, method):
        out = op_masked(
            make_h_field_3d(), make_u_field_3d(), make_v_field_3d(), method=method
        )
        assert_matches_golden(out, "Advection3D", f"__call___{method}", "masked")

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_all_ocean_matches_unmasked(self, op_unmasked, op_all_ocean, method):
        h = make_h_field_3d()
        u = make_u_field_3d()
        v = make_v_field_3d()
        np.testing.assert_array_equal(
            op_all_ocean(h, u, v, method=method),
            op_unmasked(h, u, v, method=method),
        )


# ---------------------------------------------------------------------------
# Dry-cell-zero invariants — both dispatch paths
# ---------------------------------------------------------------------------


class TestDryCellsAreZero:
    """Every dry T-cell is exactly 0 in the masked tendency, for both the
    mask-dispatchable path (weno5) and the non-dispatchable post-compute
    path (upwind1).
    """

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_advection1d(self, method):
        mask = make_mask_1d()
        op = Advection1D(grid=make_grid_1d(), mask=mask)
        out = np.asarray(op(make_h_field_1d(), make_u_field_1d(), method=method))
        dry = np.asarray(~mask.h)
        assert np.all(out[dry] == 0.0)

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_advection2d(self, method):
        mask = make_mask_2d()
        op = Advection2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(
            op(make_h_field_2d(), make_u_field_2d(), make_v_field_2d(), method=method)
        )
        dry = np.asarray(~mask.h)
        assert np.all(out[dry] == 0.0)

    @pytest.mark.parametrize("method", ["upwind1", "weno5"])
    def test_advection3d(self, method):
        mask = make_mask_3d()
        op = Advection3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(
            op(make_h_field_3d(), make_u_field_3d(), make_v_field_3d(), method=method)
        )
        dry = np.asarray(~mask.h)
        assert np.all(out[dry] == 0.0)
