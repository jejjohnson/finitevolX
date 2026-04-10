"""Mask-aware regression tests for the post-compute-zero operator family.

This module covers the operator classes that all share the same simple
"compute as if all-ocean, then multiply the output by ``mask.<stagger>``"
convention:

* ``Divergence2D``
* ``Interpolation2D`` / ``Interpolation3D``
* ``Vorticity2D`` / ``Vorticity3D``
* ``SphericalDifference2D`` / ``SphericalDifference3D``
* ``SphericalDivergence2D`` / ``SphericalDivergence3D``
* ``SphericalLaplacian2D`` / ``SphericalLaplacian3D``
* ``SphericalVorticity2D`` / ``SphericalVorticity3D``

Each operator method gets three checks:

1. **Unmasked golden** — bit-identical to the prior implementation when
   ``mask=None``.
2. **Masked golden** — bit-identical to the committed reference output
   for the canonical coastal mask.
3. **All-ocean invariant** — passing
   ``ArakawaCGridMask.from_dimensions(NY, NX)`` (an all-ocean mask)
   produces the same result as ``mask=None``.

``Difference2D`` / ``Difference3D`` have their own dedicated test file
(``test_difference_masks.py``) — kept separate because the dry-cell
zero invariant suite there is structured differently.
"""

from __future__ import annotations

import numpy as np
import pytest

from finitevolx._src.grid.spherical_grid import (
    SphericalArakawaCGrid2D,
    SphericalArakawaCGrid3D,
)
from finitevolx._src.operators.divergence import Divergence2D
from finitevolx._src.operators.interpolation import Interpolation2D, Interpolation3D
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
from finitevolx._src.operators.vorticity import Vorticity2D, Vorticity3D
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    NX_INTERIOR,
    NY_INTERIOR,
    NZ_INTERIOR,
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
# Shared module-scoped fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def mask2d():
    return make_mask_2d()


@pytest.fixture(scope="module")
def fields_2d():
    return {
        "h": make_h_field_2d(),
        "u": make_u_field_2d(),
        "v": make_v_field_2d(),
        "q": make_q_field_2d(),
    }


@pytest.fixture(scope="module")
def fields_3d():
    return {
        "h": make_h_field_3d(),
        "u": make_u_field_3d(),
        "v": make_v_field_3d(),
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _equal(a, b) -> None:
    if isinstance(a, tuple):
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _check_three_variants(
    operator: str,
    method: str,
    call,
    *,
    extra_args=(),
    extra_kwargs=None,
):
    """Run unmasked-golden + masked-golden + all-ocean checks for one method.

    ``call(*args, mask=...)`` is the operator method bound to its
    instance. ``extra_args`` and ``extra_kwargs`` carry the input
    fields.  We invoke it three ways:

      1. with ``mask=None`` and assert against the unmasked golden;
      2. with the canonical coastal mask and assert against the masked
         golden;
      3. with the all-ocean mask and assert it equals the unmasked path.
    """
    if extra_kwargs is None:
        extra_kwargs = {}
    coastal = make_mask_2d()
    all_ocean = make_mask_2d_all_ocean()

    out_unmasked = call(*extra_args, mask=None, **extra_kwargs)
    assert_matches_golden(out_unmasked, operator, method, "unmasked")

    out_masked = call(*extra_args, mask=coastal, **extra_kwargs)
    assert_matches_golden(out_masked, operator, method, "masked")

    out_all_ocean = call(*extra_args, mask=all_ocean, **extra_kwargs)
    _equal(out_unmasked, out_all_ocean)


# ----------------------------------------------------------------------
# Divergence2D
# ----------------------------------------------------------------------


class TestDivergence2D:
    @pytest.fixture(scope="class")
    def op(self):
        return Divergence2D(grid=make_grid_2d())

    def test_call(self, op, fields_2d):
        _check_three_variants(
            "Divergence2D",
            "__call__",
            op.__call__,
            extra_args=(fields_2d["u"], fields_2d["v"]),
        )

    def test_noflux(self, op, fields_2d):
        _check_three_variants(
            "Divergence2D",
            "noflux",
            op.noflux,
            extra_args=(fields_2d["u"], fields_2d["v"]),
        )


# ----------------------------------------------------------------------
# Interpolation2D / Interpolation3D
# ----------------------------------------------------------------------


# (method_name, input_field_key)
INTERP2D_METHODS = [
    ("T_to_U", "h"),
    ("T_to_V", "h"),
    ("T_to_X", "h"),
    ("X_to_U", "q"),
    ("X_to_V", "q"),
    ("U_to_T", "u"),
    ("V_to_T", "v"),
    ("X_to_T", "q"),
    ("U_to_X", "u"),
    ("V_to_X", "v"),
    ("U_to_V", "u"),
    ("V_to_U", "v"),
]


class TestInterpolation2D:
    @pytest.fixture(scope="class")
    def op(self):
        return Interpolation2D(grid=make_grid_2d())

    @pytest.mark.parametrize(("method", "input_key"), INTERP2D_METHODS)
    def test_method(self, op, fields_2d, method, input_key):
        bound = getattr(op, method)
        _check_three_variants(
            "Interpolation2D",
            method,
            bound,
            extra_args=(fields_2d[input_key],),
        )


INTERP3D_METHODS = [
    ("T_to_U", "h"),
    ("T_to_V", "h"),
    ("U_to_T", "u"),
    ("V_to_T", "v"),
]


class TestInterpolation3D:
    @pytest.fixture(scope="class")
    def op(self):
        return Interpolation3D(grid=make_grid_3d())

    @pytest.mark.parametrize(("method", "input_key"), INTERP3D_METHODS)
    def test_method(self, op, fields_3d, method, input_key):
        bound = getattr(op, method)
        _check_three_variants(
            "Interpolation3D",
            method,
            bound,
            extra_args=(fields_3d[input_key],),
        )


# ----------------------------------------------------------------------
# Vorticity2D / Vorticity3D
# ----------------------------------------------------------------------


class TestVorticity2D:
    @pytest.fixture(scope="class")
    def op(self):
        return Vorticity2D(grid=make_grid_2d())

    @pytest.fixture(scope="class")
    def q_vort(self, op):
        return op.relative_vorticity(make_u_field_2d(), make_v_field_2d())

    def test_relative_vorticity(self, op, fields_2d):
        _check_three_variants(
            "Vorticity2D",
            "relative_vorticity",
            op.relative_vorticity,
            extra_args=(fields_2d["u"], fields_2d["v"]),
        )

    def test_potential_vorticity(self, op, fields_2d):
        f2d = make_h_field_2d() * 0.0 + 1.0  # constant Coriolis
        _check_three_variants(
            "Vorticity2D",
            "potential_vorticity",
            op.potential_vorticity,
            extra_args=(fields_2d["u"], fields_2d["v"], fields_2d["h"], f2d),
        )

    def test_pv_flux_energy_conserving(self, op, fields_2d, q_vort):
        _check_three_variants(
            "Vorticity2D",
            "pv_flux_energy_conserving",
            op.pv_flux_energy_conserving,
            extra_args=(q_vort, fields_2d["u"], fields_2d["v"]),
        )

    def test_pv_flux_enstrophy_conserving(self, op, fields_2d, q_vort):
        _check_three_variants(
            "Vorticity2D",
            "pv_flux_enstrophy_conserving",
            op.pv_flux_enstrophy_conserving,
            extra_args=(q_vort, fields_2d["u"], fields_2d["v"]),
        )

    def test_pv_flux_arakawa_lamb(self, op, fields_2d, q_vort):
        _check_three_variants(
            "Vorticity2D",
            "pv_flux_arakawa_lamb",
            op.pv_flux_arakawa_lamb,
            extra_args=(q_vort, fields_2d["u"], fields_2d["v"]),
        )


class TestVorticity3D:
    @pytest.fixture(scope="class")
    def op(self):
        return Vorticity3D(grid=make_grid_3d())

    def test_relative_vorticity(self, op, fields_3d):
        _check_three_variants(
            "Vorticity3D",
            "relative_vorticity",
            op.relative_vorticity,
            extra_args=(fields_3d["u"], fields_3d["v"]),
        )


# ----------------------------------------------------------------------
# Spherical operators
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def sgrid2():
    return SphericalArakawaCGrid2D.from_interior(
        NX_INTERIOR,
        NY_INTERIOR,
        lon_range=(0.0, 10.0),
        lat_range=(10.0, 20.0),
    )


@pytest.fixture(scope="module")
def sgrid3():
    return SphericalArakawaCGrid3D.from_interior(
        NX_INTERIOR,
        NY_INTERIOR,
        NZ_INTERIOR,
        lon_range=(0.0, 10.0),
        lat_range=(10.0, 20.0),
        Lz=1.0,
    )


SPHERICAL_DIFF2D_METHODS = [
    ("diff_lon_T_to_U", "h"),
    ("diff_lat_T_to_V", "h"),
    ("diff_lon_V_to_X", "v"),
    ("diff_lat_U_to_X", "u"),
    ("diff_lon_U_to_T", "u"),
    ("diff_lat_V_to_T", "v"),
    ("diff2_lon", "h"),
    ("laplacian_merid", "h"),
]


class TestSphericalDifference2D:
    @pytest.fixture(scope="class")
    def op(self, sgrid2):
        return SphericalDifference2D(grid=sgrid2)

    @pytest.mark.parametrize(("method", "input_key"), SPHERICAL_DIFF2D_METHODS)
    def test_method(self, op, fields_2d, method, input_key):
        bound = getattr(op, method)
        _check_three_variants(
            "SphericalDifference2D",
            method,
            bound,
            extra_args=(fields_2d[input_key],),
        )


SPHERICAL_DIFF3D_METHODS = [
    ("diff_lon_T_to_U", "h"),
    ("diff_lat_T_to_V", "h"),
    ("diff_lon_U_to_T", "u"),
    ("diff_lat_V_to_T", "v"),
    ("laplacian_merid", "h"),
]


class TestSphericalDifference3D:
    @pytest.fixture(scope="class")
    def op(self, sgrid3):
        return SphericalDifference3D(grid=sgrid3)

    @pytest.mark.parametrize(("method", "input_key"), SPHERICAL_DIFF3D_METHODS)
    def test_method(self, op, fields_3d, method, input_key):
        bound = getattr(op, method)
        _check_three_variants(
            "SphericalDifference3D",
            method,
            bound,
            extra_args=(fields_3d[input_key],),
        )


class TestSphericalDivergence:
    def test_2d(self, sgrid2, fields_2d):
        op = SphericalDivergence2D(grid=sgrid2)
        _check_three_variants(
            "SphericalDivergence2D",
            "__call__",
            op.__call__,
            extra_args=(fields_2d["u"], fields_2d["v"]),
        )

    def test_3d(self, sgrid3, fields_3d):
        op = SphericalDivergence3D(grid=sgrid3)
        _check_three_variants(
            "SphericalDivergence3D",
            "__call__",
            op.__call__,
            extra_args=(fields_3d["u"], fields_3d["v"]),
        )


class TestSphericalLaplacian:
    def test_2d(self, sgrid2, fields_2d):
        op = SphericalLaplacian2D(grid=sgrid2)
        _check_three_variants(
            "SphericalLaplacian2D",
            "__call__",
            op.__call__,
            extra_args=(fields_2d["h"],),
        )

    def test_3d(self, sgrid3, fields_3d):
        op = SphericalLaplacian3D(grid=sgrid3)
        _check_three_variants(
            "SphericalLaplacian3D",
            "__call__",
            op.__call__,
            extra_args=(fields_3d["h"],),
        )


class TestSphericalVorticity:
    def test_2d(self, sgrid2, fields_2d):
        op = SphericalVorticity2D(grid=sgrid2)
        _check_three_variants(
            "SphericalVorticity2D",
            "relative_vorticity",
            op.relative_vorticity,
            extra_args=(fields_2d["u"], fields_2d["v"]),
        )

    def test_3d(self, sgrid3, fields_3d):
        op = SphericalVorticity3D(grid=sgrid3)
        _check_three_variants(
            "SphericalVorticity3D",
            "relative_vorticity",
            op.relative_vorticity,
            extra_args=(fields_3d["u"], fields_3d["v"]),
        )


# ----------------------------------------------------------------------
# Dry-cell-zero invariants for the post-compute-zero family
# ----------------------------------------------------------------------


class TestDryCellsAreZero:
    """For each method whose output is at a defined stagger, every dry
    cell of that stagger must be exactly zero in the masked output.

    Spot-check a representative method per operator family — exhaustive
    coverage lives in the per-method "masked golden" test which would
    catch the same regression a different way.
    """

    def test_divergence_dry_cells(self, fields_2d, mask2d):
        op = Divergence2D(grid=make_grid_2d())
        out = np.asarray(op(fields_2d["u"], fields_2d["v"], mask=mask2d))
        h_mask = np.asarray(mask2d.h)
        assert np.all(out[~h_mask] == 0.0)

    def test_interp_T_to_U_dry_cells(self, fields_2d, mask2d):
        op = Interpolation2D(grid=make_grid_2d())
        out = np.asarray(op.T_to_U(fields_2d["h"], mask=mask2d))
        u_mask = np.asarray(mask2d.u)
        assert np.all(out[~u_mask] == 0.0)

    def test_vorticity_relative_vorticity_dry_cells(self, fields_2d, mask2d):
        op = Vorticity2D(grid=make_grid_2d())
        out = np.asarray(
            op.relative_vorticity(fields_2d["u"], fields_2d["v"], mask=mask2d)
        )
        psi_mask = np.asarray(mask2d.psi)
        assert np.all(out[~psi_mask] == 0.0)
