"""Tests for ArakawaCGridMask and StencilCapability."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.cgrid_mask import (
    ArakawaCGridMask,
    StencilCapability,
    _count_contiguous,
    _dilate2d,
    _make_sponge,
    _pool2d_bool,
)

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rect_mask():
    """10×10 domain with a 1-cell land border."""
    mask = np.ones((10, 10), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return mask


@pytest.fixture
def island_mask():
    """20×20 domain with a 1-cell border and a small interior island.

    The domain is large enough that cells exist which are >2 hops from every
    land cell, giving non-empty open-ocean (classification == 3) cells.
    """
    mask = np.ones((20, 20), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    mask[8:12, 8:12] = False
    return mask


@pytest.fixture
def all_ocean():
    return ArakawaCGridMask.from_dimensions(8, 8)


@pytest.fixture
def rect_cgrid(rect_mask):
    return ArakawaCGridMask.from_mask(rect_mask)


@pytest.fixture
def island_cgrid(island_mask):
    return ArakawaCGridMask.from_mask(island_mask)


# ── _pool2d_bool ──────────────────────────────────────────────────────────────


class TestPool2dBool:
    def test_output_shape(self):
        h = np.ones((6, 8), dtype=np.float32)
        out = _pool2d_bool(h, 2, 1, 0.5)
        assert out.shape == (6, 8)

    def test_all_ones_kernel_2x1(self):
        # All-ones → mean = 1.0 for j≥1, but j=0 has h[j-1,i]=0 (padding)
        h = np.ones((5, 5), dtype=np.float32)
        out = _pool2d_bool(h, 2, 1, 0.75)
        # Row 0: (h[0]+0)/2 = 0.5, not > 0.75
        np.testing.assert_array_equal(out[0, :], False)
        np.testing.assert_array_equal(out[1:, :], True)

    def test_all_ones_kernel_2x2(self):
        # psi-style: needs all 4 → row 0 or col 0 is zero-padded → False
        h = np.ones((5, 5), dtype=np.float32)
        out = _pool2d_bool(h, 2, 2, 7.0 / 8.0)
        np.testing.assert_array_equal(out[0, :], False)
        np.testing.assert_array_equal(out[:, 0], False)
        np.testing.assert_array_equal(out[1:, 1:], True)

    def test_lenient_threshold_kernel_2x2(self):
        # w-style: needs ≥1 of 4 → isolated wet cell still passes
        h = np.zeros((4, 4), dtype=np.float32)
        h[2, 2] = 1.0
        out = _pool2d_bool(h, 2, 2, 1.0 / 8.0)
        # Cells (2,2), (2,3), (3,2), (3,3) can "see" h[2,2] → True
        assert out[2, 2]
        assert out[2, 3]
        assert out[3, 2]
        assert out[3, 3]


# ── _dilate2d ─────────────────────────────────────────────────────────────────


class TestDilate2d:
    def test_single_cell_4_connectivity(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        dilated = _dilate2d(mask)
        # Centre plus 4 neighbours
        assert dilated[2, 2]
        assert dilated[1, 2]
        assert dilated[3, 2]
        assert dilated[2, 1]
        assert dilated[2, 3]
        # Diagonals NOT dilated (4-connectivity)
        assert not dilated[1, 1]
        assert not dilated[3, 3]

    def test_no_wrap_around(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        dilated = _dilate2d(mask)
        assert not dilated[0, 5]
        assert not dilated[5, 0]


# ── _count_contiguous ─────────────────────────────────────────────────────────


class TestCountContiguous:
    def test_x_forward(self):
        # h = [1, 1, 0, 1, 1, 1]  →  x_pos = [2, 1, 0, 3, 2, 1]
        h = np.array([[True, True, False, True, True, True]])
        result = _count_contiguous(h, axis=1, forward=True)
        np.testing.assert_array_equal(result[0], [2, 1, 0, 3, 2, 1])

    def test_x_backward(self):
        # h = [1, 1, 0, 1, 1, 1]  →  x_neg = [1, 2, 0, 1, 2, 3]
        h = np.array([[True, True, False, True, True, True]])
        result = _count_contiguous(h, axis=1, forward=False)
        np.testing.assert_array_equal(result[0], [1, 2, 0, 1, 2, 3])

    def test_y_forward(self):
        # column [T, T, F, T, T, T]^T  →  y_pos = [2, 1, 0, 3, 2, 1]^T
        h = np.array([[True], [True], [False], [True], [True], [True]])
        result = _count_contiguous(h, axis=0, forward=True)
        np.testing.assert_array_equal(result[:, 0], [2, 1, 0, 3, 2, 1])

    def test_dry_cell_is_zero(self):
        h = np.array([[True, False, True]])
        assert _count_contiguous(h, axis=1, forward=True)[0, 1] == 0
        assert _count_contiguous(h, axis=1, forward=False)[0, 1] == 0


# ── _make_sponge ──────────────────────────────────────────────────────────────


class TestMakeSponge:
    def test_shape(self):
        s = _make_sponge(10, 12, 3)
        assert s.shape == (10, 12)

    def test_walls_are_zero(self):
        s = _make_sponge(10, 10, 3)
        np.testing.assert_array_equal(s[0, :], 0.0)
        np.testing.assert_array_equal(s[-1, :], 0.0)
        np.testing.assert_array_equal(s[:, 0], 0.0)
        np.testing.assert_array_equal(s[:, -1], 0.0)

    def test_interior_is_one(self):
        s = _make_sponge(20, 20, 4)
        np.testing.assert_allclose(s[10, 10], 1.0)


# ── StencilCapability ─────────────────────────────────────────────────────────


class TestStencilCapability:
    def test_shape(self, rect_mask):
        sc = StencilCapability.from_mask(rect_mask)
        assert sc.x_pos.shape == rect_mask.shape
        assert sc.x_neg.shape == rect_mask.shape
        assert sc.y_pos.shape == rect_mask.shape
        assert sc.y_neg.shape == rect_mask.shape

    def test_dry_cells_are_zero(self, rect_mask):
        sc = StencilCapability.from_mask(rect_mask)
        land = ~rect_mask
        np.testing.assert_array_equal(np.asarray(sc.x_pos)[land], 0)
        np.testing.assert_array_equal(np.asarray(sc.x_neg)[land], 0)

    def test_all_ocean_counts_positive(self, all_ocean):
        sc = all_ocean.stencil_capability
        assert bool(jnp.all(sc.x_pos >= 1))
        assert bool(jnp.all(sc.x_neg >= 1))


# ── ArakawaCGridMask construction ─────────────────────────────────────────────


class TestConstruction:
    def test_from_mask_shapes(self, rect_cgrid, rect_mask):
        m = rect_cgrid
        for arr in (m.h, m.u, m.v, m.w, m.psi):
            assert arr.shape == rect_mask.shape

    def test_from_dimensions_all_wet(self, all_ocean):
        assert bool(jnp.all(all_ocean.h))

    def test_from_ssh_nan_is_land(self):
        ssh = np.ones((6, 6), dtype=float)
        ssh[0, :] = np.nan
        ssh[:, 0] = np.nan
        m = ArakawaCGridMask.from_ssh(ssh)
        np.testing.assert_array_equal(np.asarray(m.h)[0, :], False)
        np.testing.assert_array_equal(np.asarray(m.h)[:, 0], False)

    def test_inverted_masks(self, rect_cgrid):
        m = rect_cgrid
        np.testing.assert_array_equal(np.asarray(m.h & m.not_h), False)
        np.testing.assert_array_equal(np.asarray(m.psi & m.not_psi), False)

    def test_top_level_import(self):
        from finitevolx import ArakawaCGridMask as ACM, StencilCapability as SC

        assert ACM is not None
        assert SC is not None


# ── staggered mask properties ─────────────────────────────────────────────────


class TestStaggeredMasks:
    def test_psi_subset_of_w(self, rect_cgrid):
        m = rect_cgrid
        # psi (strict) ⊆ w (lenient)
        assert bool(jnp.all(jnp.where(m.psi, m.w, True)))

    def test_u_requires_both_y_neighbours(self):
        # u[j, i] needs h[j, i] and h[j-1, i] both wet
        # With h[0, :] = False, u[1, :] must be False
        mask = np.ones((6, 6), dtype=bool)
        mask[0, :] = False
        m = ArakawaCGridMask.from_mask(mask)
        np.testing.assert_array_equal(np.asarray(m.u)[1, :], False)
        np.testing.assert_array_equal(np.asarray(m.u)[2, :], True)

    def test_psi_strict_threshold(self):
        # Remove one h-cell → psi for all four surrounding corners must be False
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        m = ArakawaCGridMask.from_mask(mask)
        # psi[2, 2] uses h[1,1], h[1,2], h[2,1], h[2,2] → h[2,2]=False → False
        assert not bool(m.psi[2, 2])
        assert not bool(m.psi[2, 3])
        assert not bool(m.psi[3, 2])
        assert not bool(m.psi[3, 3])


# ── vorticity boundary classification ────────────────────────────────────────


class TestVorticityBoundary:
    def test_w_valid_subset_of_w(self, rect_cgrid):
        m = rect_cgrid
        assert bool(jnp.all(jnp.where(m.w_valid, m.w, True)))

    def test_boundary_types_cover_all_wet_w(self, rect_cgrid):
        m = rect_cgrid
        covered = m.w_valid | m.w_vertical_bound | m.w_horizontal_bound
        assert bool(jnp.all(jnp.where(m.w, covered, True)))

    def test_cornerout_is_intersection(self, rect_cgrid):
        m = rect_cgrid
        expected = m.w_vertical_bound & m.w_horizontal_bound
        np.testing.assert_array_equal(
            np.asarray(m.w_cornerout_bound), np.asarray(expected)
        )


# ── irregular boundary indices ────────────────────────────────────────────────


class TestIrregularBoundary:
    def test_dtype_int32(self, rect_cgrid):
        m = rect_cgrid
        assert m.psi_irrbound_xids.dtype == jnp.int32
        assert m.psi_irrbound_yids.dtype == jnp.int32

    def test_paired_lengths(self, rect_cgrid):
        m = rect_cgrid
        assert len(m.psi_irrbound_xids) == len(m.psi_irrbound_yids)

    def test_within_interior_bounds(self, rect_cgrid):
        m = rect_cgrid
        Ny, Nx = m.h.shape
        assert bool(jnp.all(m.psi_irrbound_yids >= 1))
        assert bool(jnp.all(m.psi_irrbound_yids < Ny - 1))
        assert bool(jnp.all(m.psi_irrbound_xids >= 1))
        assert bool(jnp.all(m.psi_irrbound_xids < Nx - 1))


# ── land / coast classification ───────────────────────────────────────────────


class TestClassification:
    def test_land_cells_are_zero(self, rect_cgrid):
        m = rect_cgrid
        land_cls = np.asarray(m.classification)[~np.asarray(m.h)]
        np.testing.assert_array_equal(land_cls, 0)

    def test_wet_cells_are_positive(self, rect_cgrid):
        m = rect_cgrid
        wet_cls = np.asarray(m.classification)[np.asarray(m.h)]
        assert bool(np.all(wet_cls > 0))

    def test_accessor_consistency(self, rect_cgrid):
        m = rect_cgrid
        np.testing.assert_array_equal(
            np.asarray(m.ind_land), np.asarray(m.classification) == 0
        )
        np.testing.assert_array_equal(
            np.asarray(m.ind_coast), np.asarray(m.classification) == 1
        )
        np.testing.assert_array_equal(
            np.asarray(m.ind_near_coast), np.asarray(m.classification) == 2
        )
        np.testing.assert_array_equal(
            np.asarray(m.ind_ocean), np.asarray(m.classification) == 3
        )

    def test_all_ocean_no_land(self, all_ocean):
        assert bool(jnp.all(all_ocean.classification > 0))

    def test_open_ocean_exists_in_large_domain(self, island_cgrid):
        assert bool(jnp.any(island_cgrid.ind_ocean))

    def test_coast_cells_are_wet(self, rect_cgrid):
        m = rect_cgrid
        coast_cells = np.asarray(m.h)[np.asarray(m.ind_coast)]
        assert bool(np.all(coast_cells))

    def test_ind_boundary_outer_ring(self, rect_cgrid):
        m = rect_cgrid
        bnd = np.asarray(m.ind_boundary)
        np.testing.assert_array_equal(bnd[0, :], True)
        np.testing.assert_array_equal(bnd[-1, :], True)
        np.testing.assert_array_equal(bnd[:, 0], True)
        np.testing.assert_array_equal(bnd[:, -1], True)
        np.testing.assert_array_equal(bnd[1:-1, 1:-1], False)


# ── sponge layer ──────────────────────────────────────────────────────────────


class TestSponge:
    def test_default_is_ones(self, rect_cgrid):
        np.testing.assert_allclose(np.asarray(rect_cgrid.sponge), 1.0)

    def test_walls_are_zero(self, rect_mask):
        m = ArakawaCGridMask.from_mask(rect_mask, sponge_width=2)
        np.testing.assert_array_equal(np.asarray(m.sponge)[0, :], 0.0)
        np.testing.assert_array_equal(np.asarray(m.sponge)[:, 0], 0.0)

    def test_shape(self, rect_mask):
        m = ArakawaCGridMask.from_mask(rect_mask, sponge_width=3)
        assert m.sponge.shape == rect_mask.shape


# ── adaptive WENO stencil masks ───────────────────────────────────────────────


class TestAdaptiveMasks:
    def test_all_sizes_present(self, all_ocean):
        masks = all_ocean.get_adaptive_masks(direction="x", source="h")
        for s in (2, 4, 6, 8, 10):
            assert s in masks

    def test_mutually_exclusive(self, all_ocean):
        masks = all_ocean.get_adaptive_masks(direction="x", source="h")
        total = sum(np.asarray(m).astype(int) for m in masks.values())
        assert bool(np.all(total <= 1))

    def test_large_domain_large_stencil(self):
        m = ArakawaCGridMask.from_dimensions(20, 20)
        masks = m.get_adaptive_masks(direction="x", source="h")
        # Centre cell has ≥10 wet cells in each direction → stencil 10
        assert bool(masks[10][10, 10])

    def test_invalid_direction_raises(self, all_ocean):
        with pytest.raises(ValueError, match="direction"):
            all_ocean.get_adaptive_masks(direction="z")

    def test_invalid_source_raises(self, all_ocean):
        with pytest.raises(ValueError, match="source"):
            all_ocean.get_adaptive_masks(source="xyz")

    def test_near_border_limited_stencil(self):
        # Col=1 has only 1 wet cell to its left → max stencil in x is 2
        mask = np.ones((12, 12), dtype=bool)
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
        m = ArakawaCGridMask.from_mask(mask)
        masks = m.get_adaptive_masks(direction="x", source="h")
        assert bool(masks[2][5, 1])
        assert not bool(masks[4][5, 1])
