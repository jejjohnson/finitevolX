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

* ``upwind1`` — the one-cell tier of the adaptive hierarchy (#296 routed it
  through ``upwind_flux``; it used to read land values and post-multiply).
* ``weno5`` — the full ``(2, 4, 6)`` tier ladder of ``upwind_flux``, with the
  pre-built adaptive hierarchy narrowed to the method's stencil sizes.

In both, the face fluxes on dry faces are zeroed before the divergence
(#296), so the masked goldens hold the conservative, land-independent
tendency.

Plus a ``TestDryCellsAreZero`` class pinning the exact-zero semantic for
every dry T-cell, and ``TestMaskSafety`` (#296): no mass crosses the coast,
the wet tendencies do not depend on the land values (``1e10``, ``0`` or
``NaN``), for every method.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.advection import (
    Advection1D,
    Advection2D,
    Advection3D,
)
from finitevolx._src.advection.spherical_advection import SphericalAdvection2D
from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)
from finitevolx._src.grid.spherical import SphericalGrid2D
from finitevolx._src.mask import Mask1D, Mask2D, Mask3D
from finitevolx._src.operators.reductions import spherical_area_weights
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
    """Every dry T-cell is exactly 0 in the masked tendency (upwind1 and
    weno5 tiers).
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


# ---------------------------------------------------------------------------
# Mask safety (#296): no flux through the coast, land values never read
# ---------------------------------------------------------------------------

METHODS_2D = (
    "naive",
    "upwind1",
    "upwind2",
    "upwind3",
    "weno3",
    "weno5",
    "wenoz5",
    "weno7",
    "weno9",
    "minmod",
    "van_leer",
    "superbee",
    "mc",
)
# Reconstruction3D has no upwind2 / upwind3 stencils.
METHODS_3D = tuple(m for m in METHODS_2D if m not in ("upwind2", "upwind3"))

N = 16  # interior cells per horizontal direction


def _basin_2d() -> np.ndarray:
    """Closed 10x12 basin with a 2x3 island, at least two land cells from
    every edge so the closed-wall write region ``[2:-2, 2:-2]`` covers every
    wet cell (Σ tendency then telescopes over all wet faces)."""
    h = np.zeros((N + 2, N + 2), dtype=bool)
    h[4:14, 3:15] = True
    h[8:10, 7:10] = False
    h[5, 12] = False  # a one-cell notch: wet cells with a one-cell stencil
    return h


def _basin_1d() -> np.ndarray:
    h = np.zeros((N + 2,), dtype=bool)
    h[3:15] = True
    h[8] = False
    return h


def _basin_3d() -> np.ndarray:
    h = np.broadcast_to(_basin_2d(), (4, N + 2, N + 2)).copy()
    h[0] = False
    h[-1] = False
    return h


def _fields(shape, seed):
    rng = np.random.default_rng(seed)
    h = jnp.asarray(1.0 + 0.3 * rng.standard_normal(shape))
    u = jnp.asarray(rng.standard_normal(shape))
    v = jnp.asarray(rng.standard_normal(shape))
    return h, u, v


def _ops():
    """(name, operator, wet mask, arguments-builder) for every dimension."""
    g1 = CartesianGrid1D.from_interior(N, float(N))
    g2 = CartesianGrid2D.from_interior(N, N, float(N), float(N))
    g3 = CartesianGrid3D.from_interior(N, N, 2, float(N), float(N), 2.0)
    gs = SphericalGrid2D.from_interior(
        nx_interior=N, ny_interior=N, lon_range=(0.0, 60.0), lat_range=(-30.0, 30.0)
    )
    m1, m2, m3 = _basin_1d(), _basin_2d(), _basin_3d()
    return {
        "1d": (Advection1D(grid=g1, mask=Mask1D.from_mask(m1)), m1, METHODS_2D),
        "2d": (Advection2D(grid=g2, mask=Mask2D.from_mask(m2)), m2, METHODS_2D),
        "3d": (Advection3D(grid=g3, mask=Mask3D.from_mask(m3)), m3, METHODS_3D),
        "sph2d": (
            SphericalAdvection2D(grid=gs, mask=Mask2D.from_mask(m2)),
            m2,
            METHODS_2D,
        ),
    }


OPS = _ops()
CASES = [(name, m) for name, (_, _, methods) in OPS.items() for m in methods]


def _call(op, h, u, v, method):
    if h.ndim == 1:
        return op(h, u, method=method)
    return op(h, u, v, method=method)


class TestMaskSafety:
    @pytest.mark.parametrize(("dim", "method"), CASES)
    def test_no_leak_through_coast(self, dim, method):
        """Closed basin, constant h, u = v = +1 everywhere (not pre-masked):
        the wet→dry face fluxes are zeroed, so Σ (area ×) tendency is zero.

        Before #296 the dispatchable methods drained the basin through its
        east / north coasts (Σ = -0.33 per row for upwind1-tier outflow).
        Tolerance: the per-cell terms are O(1) (dx = 1, |u| = 1), so 1e-14
        allows the float64 roundoff of a ~100-term sum.
        """
        op, wet, _ = OPS[dim]
        h = jnp.ones(wet.shape)
        u = jnp.ones(wet.shape)
        out = _call(op, h, u, u, method)
        weight = spherical_area_weights(op.grid) if dim == "sph2d" else 1.0
        total = jnp.sum(weight * out)
        scale = jnp.sum(weight * jnp.abs(out))
        assert float(scale) > 0.0  # the coastal cells do see the flow
        assert abs(float(total)) < 1e-14 * float(scale)

    @pytest.mark.parametrize(("dim", "method"), CASES)
    def test_land_values_never_read(self, dim, method):
        """land = 1e10, 0 or NaN gives bit-identical wet tendencies (and dry
        tendencies exactly 0): every chosen stencil reads wet cells only and
        the dry inputs are zeroed with ``jnp.where`` before any stencil."""
        op, wet, _ = OPS[dim]
        h, u, v = _fields(wet.shape, seed=0)
        w = jnp.asarray(wet)
        outs = [
            np.asarray(_call(op, jnp.where(w, h, land), u, v, method))
            for land in (1e10, 0.0, jnp.nan)
        ]
        np.testing.assert_array_equal(outs[0], outs[1])
        np.testing.assert_array_equal(outs[0], outs[2])
        assert np.all(outs[0][~wet] == 0.0)

    @pytest.mark.parametrize(("dim", "method"), CASES)
    def test_constant_field_reconstructs_exactly(self, dim, method):
        """Free-stream check: a constant h is reconstructed exactly at every
        wet face, so the tendency equals ``naive`` (-h·div u) — no stencil
        reads a zeroed land cell.  Tolerance: the WENO7/9 linear weights sum
        to 1 only to a few ulp, giving ~1e-14 on O(1) tendencies."""
        op, wet, _ = OPS[dim]
        _, u, v = _fields(wet.shape, seed=1)
        h = jnp.ones(wet.shape)
        np.testing.assert_allclose(
            _call(op, h, u, v, method),
            _call(op, h, u, v, "naive"),
            rtol=0.0,
            atol=1e-13 * float(jnp.max(jnp.abs(_call(op, h, u, v, "naive")))),
        )

    @pytest.mark.parametrize(
        "method",
        [pytest.param("weno9", marks=pytest.mark.slow), "van_leer", "naive"],
    )
    def test_grad_finite_with_nan_land(self, method):
        """NaN on land leaves the gradient finite (inputs zeroed by where)."""
        op, wet, _ = OPS["2d"]
        h, u, v = _fields(wet.shape, seed=2)
        hn = jnp.where(jnp.asarray(wet), h, jnp.nan)
        g = jax.grad(lambda x: jnp.sum(op(x, u, v, method=method) ** 2))(hn)
        assert bool(jnp.all(jnp.isfinite(g)))


class TestWenoZ5AllDimensions:
    """``wenoz5`` is accepted by Advection1D / 3D (it was 2-D only)."""

    def test_advection1d(self):
        grid = make_grid_1d()
        out = Advection1D(grid=grid)(
            make_h_field_1d(), make_u_field_1d(), method="wenoz5"
        )
        ref = Advection1D(grid=grid)(
            make_h_field_1d(), make_u_field_1d(), method="weno5"
        )
        assert bool(jnp.all(jnp.isfinite(out)))
        # same 5-point stencils, different nonlinear weights: close, not equal
        np.testing.assert_allclose(out, ref, atol=0.1 * float(jnp.max(jnp.abs(ref))))

    @pytest.mark.slow
    def test_advection3d(self):
        grid = make_grid_3d()
        args = (make_h_field_3d(), make_u_field_3d(), make_v_field_3d())
        out = Advection3D(grid=grid)(*args, method="wenoz5")
        ref = Advection3D(grid=grid)(*args, method="weno5")
        assert bool(jnp.all(jnp.isfinite(out)))
        np.testing.assert_allclose(out, ref, atol=0.1 * float(jnp.max(jnp.abs(ref))))
        # matches the 2-D operator level by level
        g2 = make_grid_2d()
        out2 = Advection2D(grid=g2)(args[0][1], args[1][1], args[2][1], method="wenoz5")
        np.testing.assert_allclose(out[1], out2, rtol=1e-14, atol=1e-14)
