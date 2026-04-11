"""Golden-output regression tests for Diffusion / BiharmonicDiffusion /
MomentumAdvection mask fields (Phase 7).

Phase 7 is the BREAKING commit that unifies the diffusion/momentum mask
API on ``Mask2D`` / ``Mask3D`` class fields:

* ``diffusion_2d`` free function loses its ``mask_h``/``mask_u``/``mask_v``
  raw-array kwargs and becomes mask-free (Layer-2 functional per #209).
* ``Diffusion2D`` / ``Diffusion3D`` grow a ``mask`` class field; the
  three-step intermediate flux-masking pattern is inlined via
  ``_diffusion_2d_impl`` so ``Diffusion3D`` can vmap it per-z-slice.
* ``BiharmonicDiffusion2D`` / ``BiharmonicDiffusion3D`` grow a ``mask``
  class field but apply it **final-only** — the inner harmonic is
  deliberately mask-free to avoid corrupting the second Laplacian pass.
* ``MomentumAdvection2D`` / ``MomentumAdvection3D`` grow a ``mask``
  class field with pass-down into the internal Difference2D /
  Interpolation2D, plus a final post-multiply by ``mask.u`` / ``mask.v``.
"""

from __future__ import annotations

import numpy as np
import pytest

from finitevolx._src.diffusion.diffusion import (
    BiharmonicDiffusion2D,
    BiharmonicDiffusion3D,
    Diffusion2D,
    Diffusion3D,
)
from finitevolx._src.diffusion.momentum import (
    MomentumAdvection2D,
    MomentumAdvection3D,
)
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_grid_2d,
    make_grid_3d,
    make_h_field_2d,
    make_h_field_3d,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_mask_3d,
    make_mask_3d_all_ocean,
    make_u_field_2d,
    make_u_field_3d,
    make_v_field_2d,
    make_v_field_3d,
)

# ---------------------------------------------------------------------------
# Diffusion2D
# ---------------------------------------------------------------------------


class TestDiffusion2DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Diffusion2D(grid=make_grid_2d())

    @pytest.fixture
    def op_masked(self):
        return Diffusion2D(grid=make_grid_2d(), mask=make_mask_2d())

    @pytest.fixture
    def op_all_ocean(self):
        return Diffusion2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())

    def test_call_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_h_field_2d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion2D", "__call__", "unmasked")

    def test_call_masked_golden(self, op_masked):
        out = op_masked(make_h_field_2d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion2D", "__call__", "masked")

    def test_call_all_ocean(self, op_unmasked, op_all_ocean):
        h = make_h_field_2d()
        np.testing.assert_array_equal(
            op_all_ocean(h, kappa=1.0),
            op_unmasked(h, kappa=1.0),
        )

    def test_fluxes_unmasked_golden(self, op_unmasked):
        out = op_unmasked.fluxes(make_h_field_2d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion2D", "fluxes", "unmasked")

    def test_fluxes_masked_golden(self, op_masked):
        out = op_masked.fluxes(make_h_field_2d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion2D", "fluxes", "masked")

    def test_fluxes_all_ocean(self, op_unmasked, op_all_ocean):
        h = make_h_field_2d()
        fx1, fy1 = op_unmasked.fluxes(h, kappa=1.0)
        fx2, fy2 = op_all_ocean.fluxes(h, kappa=1.0)
        np.testing.assert_array_equal(fx1, fx2)
        np.testing.assert_array_equal(fy1, fy2)


# ---------------------------------------------------------------------------
# Diffusion3D
# ---------------------------------------------------------------------------


class TestDiffusion3DMasks:
    @pytest.fixture
    def op_unmasked(self):
        return Diffusion3D(grid=make_grid_3d())

    @pytest.fixture
    def op_masked(self):
        return Diffusion3D(grid=make_grid_3d(), mask=make_mask_3d())

    @pytest.fixture
    def op_all_ocean(self):
        return Diffusion3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())

    def test_call_unmasked_golden(self, op_unmasked):
        out = op_unmasked(make_h_field_3d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion3D", "__call__", "unmasked")

    def test_call_masked_golden(self, op_masked):
        out = op_masked(make_h_field_3d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion3D", "__call__", "masked")

    def test_call_all_ocean(self, op_unmasked, op_all_ocean):
        h = make_h_field_3d()
        np.testing.assert_array_equal(
            op_all_ocean(h, kappa=1.0),
            op_unmasked(h, kappa=1.0),
        )

    def test_fluxes_unmasked_golden(self, op_unmasked):
        out = op_unmasked.fluxes(make_h_field_3d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion3D", "fluxes", "unmasked")

    def test_fluxes_masked_golden(self, op_masked):
        out = op_masked.fluxes(make_h_field_3d(), kappa=1.0)
        assert_matches_golden(out, "Diffusion3D", "fluxes", "masked")

    def test_fluxes_all_ocean(self, op_unmasked, op_all_ocean):
        h = make_h_field_3d()
        fx1, fy1 = op_unmasked.fluxes(h, kappa=1.0)
        fx2, fy2 = op_all_ocean.fluxes(h, kappa=1.0)
        np.testing.assert_array_equal(fx1, fx2)
        np.testing.assert_array_equal(fy1, fy2)


# ---------------------------------------------------------------------------
# BiharmonicDiffusion2D / 3D
# ---------------------------------------------------------------------------


class TestBiharmonicDiffusionMasks:
    def test_2d_unmasked_golden(self):
        op = BiharmonicDiffusion2D(grid=make_grid_2d())
        out = op(make_h_field_2d(), kappa=1e-3)
        assert_matches_golden(out, "BiharmonicDiffusion2D", "__call__", "unmasked")

    def test_2d_masked_golden(self):
        op = BiharmonicDiffusion2D(grid=make_grid_2d(), mask=make_mask_2d())
        out = op(make_h_field_2d(), kappa=1e-3)
        assert_matches_golden(out, "BiharmonicDiffusion2D", "__call__", "masked")

    def test_2d_all_ocean(self):
        op_plain = BiharmonicDiffusion2D(grid=make_grid_2d())
        op_all = BiharmonicDiffusion2D(
            grid=make_grid_2d(), mask=make_mask_2d_all_ocean()
        )
        h = make_h_field_2d()
        np.testing.assert_array_equal(op_plain(h, kappa=1e-3), op_all(h, kappa=1e-3))

    def test_2d_inner_harm_is_mask_free(self):
        """Critical correctness pin: the inner ``_harm`` must be built
        ``mask=None`` even when the outer op has a mask.  If someone
        naively wires the outer mask into the inner harmonic, the second
        Laplacian pass sees ``lap1 == 0`` at dry cells which is a forced
        Dirichlet-0 BC that corrupts the ∇⁴ stencil at wet cells
        adjacent to land.
        """
        op = BiharmonicDiffusion2D(grid=make_grid_2d(), mask=make_mask_2d())
        assert op._harm.mask is None

    def test_3d_unmasked_golden(self):
        op = BiharmonicDiffusion3D(grid=make_grid_3d())
        out = op(make_h_field_3d(), kappa=1e-3)
        assert_matches_golden(out, "BiharmonicDiffusion3D", "__call__", "unmasked")

    def test_3d_masked_golden(self):
        op = BiharmonicDiffusion3D(grid=make_grid_3d(), mask=make_mask_3d())
        out = op(make_h_field_3d(), kappa=1e-3)
        assert_matches_golden(out, "BiharmonicDiffusion3D", "__call__", "masked")

    def test_3d_all_ocean(self):
        op_plain = BiharmonicDiffusion3D(grid=make_grid_3d())
        op_all = BiharmonicDiffusion3D(
            grid=make_grid_3d(), mask=make_mask_3d_all_ocean()
        )
        h = make_h_field_3d()
        np.testing.assert_array_equal(op_plain(h, kappa=1e-3), op_all(h, kappa=1e-3))

    def test_3d_inner_harm_is_mask_free(self):
        op = BiharmonicDiffusion3D(grid=make_grid_3d(), mask=make_mask_3d())
        assert op._harm.mask is None


# ---------------------------------------------------------------------------
# MomentumAdvection2D / 3D
# ---------------------------------------------------------------------------


class TestMomentumAdvectionMasks:
    def test_2d_unmasked_golden(self):
        op = MomentumAdvection2D(grid=make_grid_2d())
        out = op(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "MomentumAdvection2D", "__call__", "unmasked")

    def test_2d_masked_golden(self):
        op = MomentumAdvection2D(grid=make_grid_2d(), mask=make_mask_2d())
        out = op(make_u_field_2d(), make_v_field_2d())
        assert_matches_golden(out, "MomentumAdvection2D", "__call__", "masked")

    def test_2d_all_ocean(self):
        op_plain = MomentumAdvection2D(grid=make_grid_2d())
        op_all = MomentumAdvection2D(grid=make_grid_2d(), mask=make_mask_2d_all_ocean())
        u = make_u_field_2d()
        v = make_v_field_2d()
        du1, dv1 = op_plain(u, v)
        du2, dv2 = op_all(u, v)
        np.testing.assert_array_equal(du1, du2)
        np.testing.assert_array_equal(dv1, dv2)

    def test_3d_unmasked_golden(self):
        op = MomentumAdvection3D(grid=make_grid_3d())
        out = op(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "MomentumAdvection3D", "__call__", "unmasked")

    def test_3d_masked_golden(self):
        op = MomentumAdvection3D(grid=make_grid_3d(), mask=make_mask_3d())
        out = op(make_u_field_3d(), make_v_field_3d())
        assert_matches_golden(out, "MomentumAdvection3D", "__call__", "masked")

    def test_3d_all_ocean(self):
        op_plain = MomentumAdvection3D(grid=make_grid_3d())
        op_all = MomentumAdvection3D(grid=make_grid_3d(), mask=make_mask_3d_all_ocean())
        u = make_u_field_3d()
        v = make_v_field_3d()
        du1, dv1 = op_plain(u, v)
        du2, dv2 = op_all(u, v)
        np.testing.assert_array_equal(du1, du2)
        np.testing.assert_array_equal(dv1, dv2)


# ---------------------------------------------------------------------------
# Dry-cell-zero invariants
# ---------------------------------------------------------------------------


class TestDryCellsAreZero:
    """Every dry T-cell is exactly 0 in the diffusion tendency, and every
    dry U/V face is exactly 0 in the momentum tendencies.
    """

    def test_diffusion2d(self):
        mask = make_mask_2d()
        op = Diffusion2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op(make_h_field_2d(), kappa=1.0))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_diffusion3d(self):
        mask = make_mask_3d()
        op = Diffusion3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(op(make_h_field_3d(), kappa=1.0))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_biharmonic2d(self):
        mask = make_mask_2d()
        op = BiharmonicDiffusion2D(grid=make_grid_2d(), mask=mask)
        out = np.asarray(op(make_h_field_2d(), kappa=1e-3))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_biharmonic3d(self):
        mask = make_mask_3d()
        op = BiharmonicDiffusion3D(grid=make_grid_3d(), mask=mask)
        out = np.asarray(op(make_h_field_3d(), kappa=1e-3))
        assert np.all(out[np.asarray(~mask.h)] == 0.0)

    def test_momentum2d(self):
        mask = make_mask_2d()
        op = MomentumAdvection2D(grid=make_grid_2d(), mask=mask)
        du, dv = op(make_u_field_2d(), make_v_field_2d())
        du = np.asarray(du)
        dv = np.asarray(dv)
        assert np.all(du[np.asarray(~mask.u)] == 0.0)
        assert np.all(dv[np.asarray(~mask.v)] == 0.0)

    def test_momentum3d(self):
        mask = make_mask_3d()
        op = MomentumAdvection3D(grid=make_grid_3d(), mask=mask)
        du, dv = op(make_u_field_3d(), make_v_field_3d())
        du = np.asarray(du)
        dv = np.asarray(dv)
        assert np.all(du[np.asarray(~mask.u)] == 0.0)
        assert np.all(dv[np.asarray(~mask.v)] == 0.0)
