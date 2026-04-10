"""Mask-aware regression tests for ``Diffusion2D`` / ``Diffusion3D``
and the biharmonic variants.

Diffusion is one of the two operator families (the other is advection)
where masking happens at the *intermediate flux* step rather than as a
post-compute multiply on the output.  See ``docs/concepts/masking.md``
for the design rationale.

These tests pin three things for every public method:

1. **Unmasked golden** — bit-identical to the prior implementation when
   ``mask=None``.
2. **Masked golden** — bit-identical to the committed reference output
   for the canonical coastal mask, exercising the inline
   intermediate-masking code path.
3. **All-ocean invariant** — passing
   ``ArakawaCGridMask.from_dimensions(NY, NX)`` (an all-ocean mask)
   gives the same answer as ``mask=None``.

A small dry-cell-zero suite at the end pins the canonical "tendency
is exactly zero in dry interior cells" invariant.
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
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_grid_2d,
    make_grid_3d,
    make_h_field_2d,
    make_h_field_3d,
    make_mask_2d,
    make_mask_2d_all_ocean,
)

KAPPA = 1e-3  # must match the value used by tests/fixtures/_gen_golden.py


# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid2d():
    return make_grid_2d()


@pytest.fixture(scope="module")
def grid3d():
    return make_grid_3d()


@pytest.fixture(scope="module")
def h2d():
    return make_h_field_2d()


@pytest.fixture(scope="module")
def h3d():
    return make_h_field_3d()


@pytest.fixture(scope="module")
def mask2d():
    return make_mask_2d()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _equal(a, b) -> None:
    if isinstance(a, tuple):
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    else:
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ----------------------------------------------------------------------
# Diffusion2D
# ----------------------------------------------------------------------


class TestDiffusion2DCall:
    @pytest.fixture(scope="class")
    def op(self, grid2d):
        return Diffusion2D(grid=grid2d)

    def test_unmasked_matches_golden(self, op, h2d):
        out = op(h2d, kappa=KAPPA)
        assert_matches_golden(out, "Diffusion2D", "__call__", "unmasked")

    def test_masked_matches_golden(self, op, h2d, mask2d):
        out = op(h2d, kappa=KAPPA, mask=mask2d)
        assert_matches_golden(out, "Diffusion2D", "__call__", "masked")

    def test_all_ocean_equals_unmasked(self, op, h2d):
        out_none = op(h2d, kappa=KAPPA)
        out_all_ocean = op(h2d, kappa=KAPPA, mask=make_mask_2d_all_ocean())
        _equal(out_none, out_all_ocean)


class TestDiffusion2DFluxes:
    @pytest.fixture(scope="class")
    def op(self, grid2d):
        return Diffusion2D(grid=grid2d)

    def test_unmasked_matches_golden(self, op, h2d):
        out = op.fluxes(h2d, kappa=KAPPA)
        assert_matches_golden(out, "Diffusion2D", "fluxes", "unmasked")

    def test_masked_matches_golden(self, op, h2d, mask2d):
        out = op.fluxes(h2d, kappa=KAPPA, mask=mask2d)
        assert_matches_golden(out, "Diffusion2D", "fluxes", "masked")

    def test_all_ocean_equals_unmasked(self, op, h2d):
        out_none = op.fluxes(h2d, kappa=KAPPA)
        out_all_ocean = op.fluxes(h2d, kappa=KAPPA, mask=make_mask_2d_all_ocean())
        _equal(out_none, out_all_ocean)


# ----------------------------------------------------------------------
# Diffusion3D
# ----------------------------------------------------------------------


class TestDiffusion3DCall:
    @pytest.fixture(scope="class")
    def op(self, grid3d):
        return Diffusion3D(grid=grid3d)

    def test_unmasked_matches_golden(self, op, h3d):
        out = op(h3d, kappa=KAPPA)
        assert_matches_golden(out, "Diffusion3D", "__call__", "unmasked")

    def test_masked_matches_golden(self, op, h3d, mask2d):
        out = op(h3d, kappa=KAPPA, mask=mask2d)
        assert_matches_golden(out, "Diffusion3D", "__call__", "masked")

    def test_all_ocean_equals_unmasked(self, op, h3d):
        out_none = op(h3d, kappa=KAPPA)
        out_all_ocean = op(h3d, kappa=KAPPA, mask=make_mask_2d_all_ocean())
        _equal(out_none, out_all_ocean)


class TestDiffusion3DFluxes:
    @pytest.fixture(scope="class")
    def op(self, grid3d):
        return Diffusion3D(grid=grid3d)

    def test_unmasked_matches_golden(self, op, h3d):
        out = op.fluxes(h3d, kappa=KAPPA)
        assert_matches_golden(out, "Diffusion3D", "fluxes", "unmasked")

    def test_masked_matches_golden(self, op, h3d, mask2d):
        out = op.fluxes(h3d, kappa=KAPPA, mask=mask2d)
        assert_matches_golden(out, "Diffusion3D", "fluxes", "masked")

    def test_all_ocean_equals_unmasked(self, op, h3d):
        out_none = op.fluxes(h3d, kappa=KAPPA)
        out_all_ocean = op.fluxes(h3d, kappa=KAPPA, mask=make_mask_2d_all_ocean())
        _equal(out_none, out_all_ocean)


# ----------------------------------------------------------------------
# BiharmonicDiffusion2D / BiharmonicDiffusion3D
# ----------------------------------------------------------------------


class TestBiharmonicDiffusion2D:
    @pytest.fixture(scope="class")
    def op(self, grid2d):
        return BiharmonicDiffusion2D(grid=grid2d)

    def test_unmasked_matches_golden(self, op, h2d):
        out = op(h2d, kappa=KAPPA)
        assert_matches_golden(out, "BiharmonicDiffusion2D", "__call__", "unmasked")

    def test_masked_matches_golden(self, op, h2d, mask2d):
        out = op(h2d, kappa=KAPPA, mask=mask2d)
        assert_matches_golden(out, "BiharmonicDiffusion2D", "__call__", "masked")

    def test_all_ocean_equals_unmasked(self, op, h2d):
        out_none = op(h2d, kappa=KAPPA)
        out_all_ocean = op(h2d, kappa=KAPPA, mask=make_mask_2d_all_ocean())
        _equal(out_none, out_all_ocean)


class TestBiharmonicDiffusion3D:
    @pytest.fixture(scope="class")
    def op(self, grid3d):
        return BiharmonicDiffusion3D(grid=grid3d)

    def test_unmasked_matches_golden(self, op, h3d):
        out = op(h3d, kappa=KAPPA)
        assert_matches_golden(out, "BiharmonicDiffusion3D", "__call__", "unmasked")

    def test_masked_matches_golden(self, op, h3d, mask2d):
        out = op(h3d, kappa=KAPPA, mask=mask2d)
        assert_matches_golden(out, "BiharmonicDiffusion3D", "__call__", "masked")

    def test_all_ocean_equals_unmasked(self, op, h3d):
        out_none = op(h3d, kappa=KAPPA)
        out_all_ocean = op(h3d, kappa=KAPPA, mask=make_mask_2d_all_ocean())
        _equal(out_none, out_all_ocean)


# ----------------------------------------------------------------------
# Dry-cell-zero invariants
# ----------------------------------------------------------------------


class TestDryCellsAreZero:
    """Every dry interior T-cell must be exactly zero in the masked output."""

    def test_diffusion2d_dry_cells(self, grid2d, h2d, mask2d):
        op = Diffusion2D(grid=grid2d)
        out = np.asarray(op(h2d, kappa=KAPPA, mask=mask2d))
        h_mask = np.asarray(mask2d.h)
        assert np.all(out[~h_mask] == 0.0)

    def test_diffusion3d_dry_cells(self, grid3d, h3d, mask2d):
        op = Diffusion3D(grid=grid3d)
        out = np.asarray(op(h3d, kappa=KAPPA, mask=mask2d))
        h_mask_2d = np.asarray(mask2d.h)
        # Broadcast 2D mask over z; check every z-level.
        for k in range(out.shape[0]):
            assert np.all(out[k][~h_mask_2d] == 0.0), f"z={k}"

    def test_biharmonic2d_dry_cells(self, grid2d, h2d, mask2d):
        op = BiharmonicDiffusion2D(grid=grid2d)
        out = np.asarray(op(h2d, kappa=KAPPA, mask=mask2d))
        h_mask = np.asarray(mask2d.h)
        assert np.all(out[~h_mask] == 0.0)

    def test_biharmonic3d_dry_cells(self, grid3d, h3d, mask2d):
        op = BiharmonicDiffusion3D(grid=grid3d)
        out = np.asarray(op(h3d, kappa=KAPPA, mask=mask2d))
        h_mask_2d = np.asarray(mask2d.h)
        for k in range(out.shape[0]):
            assert np.all(out[k][~h_mask_2d] == 0.0), f"z={k}"
