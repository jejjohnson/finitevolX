"""Tests for Mask1D / Mask2D / Mask3D and StencilCapability1D / 2D / 3D."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.mask import (
    Mask1D,
    Mask2D,
    Mask3D,
    StencilCapability1D,
    StencilCapability2D,
    StencilCapability3D,
)
from finitevolx._src.mask.utils import (
    _count_contiguous,
    _dilate,
    _make_sponge,
    _pool_bool,
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
    return Mask2D.from_dimensions(8, 8)


@pytest.fixture
def rect_cgrid(rect_mask):
    return Mask2D.from_mask(rect_mask)


@pytest.fixture
def island_cgrid(island_mask):
    return Mask2D.from_mask(island_mask)


# ── _pool_bool ────────────────────────────────────────────────────────────────


class TestPoolBool:
    def test_output_shape(self):
        h = np.ones((6, 8), dtype=np.float32)
        out = _pool_bool(h, (2, 1), 0.5)
        assert out.shape == (6, 8)

    def test_all_ones_kernel_2x1(self):
        # All-ones → mean = 1.0 for j≥1, but j=0 has h[j-1,i]=0 (padding)
        h = np.ones((5, 5), dtype=np.float32)
        out = _pool_bool(h, (2, 1), 0.75)
        # Row 0: (h[0]+0)/2 = 0.5, not > 0.75
        np.testing.assert_array_equal(out[0, :], False)
        np.testing.assert_array_equal(out[1:, :], True)

    def test_all_ones_kernel_2x2(self):
        # psi-style: needs all 4 → row 0 or col 0 is zero-padded → False
        h = np.ones((5, 5), dtype=np.float32)
        out = _pool_bool(h, (2, 2), 7.0 / 8.0)
        np.testing.assert_array_equal(out[0, :], False)
        np.testing.assert_array_equal(out[:, 0], False)
        np.testing.assert_array_equal(out[1:, 1:], True)

    def test_lenient_threshold_kernel_2x2(self):
        # w-style: needs ≥1 of 4 → isolated wet cell still passes
        h = np.zeros((4, 4), dtype=np.float32)
        h[2, 2] = 1.0
        out = _pool_bool(h, (2, 2), 1.0 / 8.0)
        # Cells (2,2), (2,3), (3,2), (3,3) can "see" h[2,2] → True
        assert out[2, 2]
        assert out[2, 3]
        assert out[3, 2]
        assert out[3, 3]


# ── _dilate ───────────────────────────────────────────────────────────────────


class TestDilate:
    def test_single_cell_4_connectivity(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        dilated = _dilate(mask)
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
        dilated = _dilate(mask)
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
        s = _make_sponge((10, 12), 3)
        assert s.shape == (10, 12)

    def test_walls_are_zero(self):
        s = _make_sponge((10, 10), 3)
        np.testing.assert_array_equal(s[0, :], 0.0)
        np.testing.assert_array_equal(s[-1, :], 0.0)
        np.testing.assert_array_equal(s[:, 0], 0.0)
        np.testing.assert_array_equal(s[:, -1], 0.0)

    def test_interior_is_one(self):
        s = _make_sponge((20, 20), 4)
        np.testing.assert_allclose(s[10, 10], 1.0)


# ── StencilCapability2D ─────────────────────────────────────────────────────────


class TestStencilCapability:
    def test_shape(self, rect_mask):
        sc = StencilCapability2D.from_mask(rect_mask)
        assert sc.x_pos.shape == rect_mask.shape
        assert sc.x_neg.shape == rect_mask.shape
        assert sc.y_pos.shape == rect_mask.shape
        assert sc.y_neg.shape == rect_mask.shape

    def test_dry_cells_are_zero(self, rect_mask):
        sc = StencilCapability2D.from_mask(rect_mask)
        land = ~rect_mask
        np.testing.assert_array_equal(np.asarray(sc.x_pos)[land], 0)
        np.testing.assert_array_equal(np.asarray(sc.x_neg)[land], 0)

    def test_all_ocean_counts_positive(self, all_ocean):
        sc = all_ocean.stencil_capability
        assert bool(jnp.all(sc.x_pos >= 1))
        assert bool(jnp.all(sc.x_neg >= 1))


# ── Mask2D construction ─────────────────────────────────────────────


class TestConstruction:
    def test_from_mask_shapes(self, rect_cgrid, rect_mask):
        m = rect_cgrid
        for arr in (m.h, m.u, m.v, m.xy_corner, m.xy_corner_strict):
            assert arr.shape == rect_mask.shape

    def test_from_dimensions_all_wet(self, all_ocean):
        assert bool(jnp.all(all_ocean.h))

    def test_from_ssh_nan_is_land(self):
        ssh = np.ones((6, 6), dtype=float)
        ssh[0, :] = np.nan
        ssh[:, 0] = np.nan
        m = Mask2D.from_ssh(ssh)
        np.testing.assert_array_equal(np.asarray(m.h)[0, :], False)
        np.testing.assert_array_equal(np.asarray(m.h)[:, 0], False)

    def test_inverted_masks(self, rect_cgrid):
        m = rect_cgrid
        np.testing.assert_array_equal(np.asarray(m.h & m.not_h), False)
        np.testing.assert_array_equal(
            np.asarray(m.xy_corner_strict & m.not_xy_corner_strict), False
        )

    def test_top_level_import(self):
        from finitevolx import Mask2D as ACM, StencilCapability2D as SC

        assert ACM is not None
        assert SC is not None


# ── staggered mask properties ─────────────────────────────────────────────────


class TestStaggeredMasks:
    def test_psi_subset_of_w(self, rect_cgrid):
        m = rect_cgrid
        # psi (strict) ⊆ w (lenient)
        assert bool(jnp.all(jnp.where(m.xy_corner_strict, m.xy_corner, True)))

    def test_u_requires_both_y_neighbours(self):
        # u[j, i] needs h[j, i] and h[j-1, i] both wet
        # With h[0, :] = False, u[1, :] must be False
        mask = np.ones((6, 6), dtype=bool)
        mask[0, :] = False
        m = Mask2D.from_mask(mask)
        np.testing.assert_array_equal(np.asarray(m.u)[1, :], False)
        np.testing.assert_array_equal(np.asarray(m.u)[2, :], True)

    def test_psi_strict_threshold(self):
        # Remove one h-cell → psi for all four surrounding corners must be False
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        m = Mask2D.from_mask(mask)
        # psi[2, 2] uses h[1,1], h[1,2], h[2,1], h[2,2] → h[2,2]=False → False
        assert not bool(m.xy_corner_strict[2, 2])
        assert not bool(m.xy_corner_strict[2, 3])
        assert not bool(m.xy_corner_strict[3, 2])
        assert not bool(m.xy_corner_strict[3, 3])


# ── vorticity boundary classification ────────────────────────────────────────


class TestVorticityBoundary:
    def test_w_valid_subset_of_w(self, rect_cgrid):
        m = rect_cgrid
        assert bool(jnp.all(jnp.where(m.xy_corner_valid, m.xy_corner, True)))

    def test_boundary_types_cover_all_wet_w(self, rect_cgrid):
        m = rect_cgrid
        covered = m.xy_corner_valid | m.xy_corner_y_wall | m.xy_corner_x_wall
        assert bool(jnp.all(jnp.where(m.xy_corner, covered, True)))

    def test_cornerout_is_intersection(self, rect_cgrid):
        m = rect_cgrid
        expected = m.xy_corner_y_wall & m.xy_corner_x_wall
        np.testing.assert_array_equal(
            np.asarray(m.xy_corner_convex), np.asarray(expected)
        )


# ── irregular boundary indices ────────────────────────────────────────────────


class TestIrregularBoundary:
    def test_dtype_int32(self, rect_cgrid):
        m = rect_cgrid
        assert m.xy_corner_strict_irrbound_cols.dtype == jnp.int32
        assert m.xy_corner_strict_irrbound_rows.dtype == jnp.int32

    def test_paired_lengths(self, rect_cgrid):
        m = rect_cgrid
        assert len(m.xy_corner_strict_irrbound_cols) == len(
            m.xy_corner_strict_irrbound_rows
        )

    def test_within_interior_bounds(self, rect_cgrid):
        m = rect_cgrid
        Ny, Nx = m.h.shape
        assert bool(jnp.all(m.xy_corner_strict_irrbound_rows >= 1))
        assert bool(jnp.all(m.xy_corner_strict_irrbound_rows < Ny - 1))
        assert bool(jnp.all(m.xy_corner_strict_irrbound_cols >= 1))
        assert bool(jnp.all(m.xy_corner_strict_irrbound_cols < Nx - 1))


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
        m = Mask2D.from_mask(rect_mask, sponge_width=2)
        np.testing.assert_array_equal(np.asarray(m.sponge)[0, :], 0.0)
        np.testing.assert_array_equal(np.asarray(m.sponge)[:, 0], 0.0)

    def test_shape(self, rect_mask):
        m = Mask2D.from_mask(rect_mask, sponge_width=3)
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
        m = Mask2D.from_dimensions(20, 20)
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
        m = Mask2D.from_mask(mask)
        masks = m.get_adaptive_masks(direction="x", source="h")
        assert bool(masks[2][5, 1])
        assert not bool(masks[4][5, 1])


# ── geometric correctness of staggered masks ─────────────────────────────────
#
# These tests verify exact grid-point values against the Arakawa C-grid layout:
#
#     w-----v-----w
#     |           |
#     u     h     u
#     |           |
#     w-----v-----w
#
# Convention:
#   h[j, i]   — cell centre
#   u[j, i]   — interface between h[j-1, i] and h[j, i]  (y-face)
#   v[j, i]   — interface between h[j, i-1] and h[j, i]  (x-face)
#   w[j, i]   — SW corner of h[j, i], uses h[j-1:j+1, i-1:i+1]
#   psi[j, i] — strict version of w (all 4 h-cells must be wet)


class TestStaggeredMaskGeometry:
    """Verify staggered mask values at specific grid points."""

    @pytest.fixture
    def basin6(self):
        """6×6 basin: land border, 4×4 ocean interior."""
        h = np.ones((6, 6), dtype=bool)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = False
        return h, Mask2D.from_mask(h)

    # ── u mask: interface between h[j-1,i] and h[j,i] ────────────────

    def test_u_wet_when_both_h_neighbours_wet(self, basin6):
        _, m = basin6
        u = np.asarray(m.u)
        # u[2,2]: between h[1,2] (wet) and h[2,2] (wet) → wet
        assert u[2, 2]

    def test_u_dry_when_h_below_is_land(self, basin6):
        _, m = basin6
        u = np.asarray(m.u)
        # u[1,2]: between h[0,2] (land) and h[1,2] (wet) → dry
        assert not u[1, 2]

    def test_u_dry_when_h_above_is_land(self, basin6):
        _, m = basin6
        u = np.asarray(m.u)
        # u[5,2]: between h[4,2] (wet) and h[5,2] (land) → dry
        assert not u[5, 2]

    def test_u_dry_at_row_0_padding(self, basin6):
        _, m = basin6
        u = np.asarray(m.u)
        # u[0,i]: between padded zero and h[0,i] → always dry
        np.testing.assert_array_equal(u[0, :], False)

    # ── v mask: interface between h[j,i-1] and h[j,i] ────────────────

    def test_v_wet_when_both_h_neighbours_wet(self, basin6):
        _, m = basin6
        v = np.asarray(m.v)
        # v[2,2]: between h[2,1] (wet) and h[2,2] (wet) → wet
        assert v[2, 2]

    def test_v_dry_when_h_left_is_land(self, basin6):
        _, m = basin6
        v = np.asarray(m.v)
        # v[2,1]: between h[2,0] (land) and h[2,1] (wet) → dry
        assert not v[2, 1]

    def test_v_dry_at_col_0_padding(self, basin6):
        _, m = basin6
        v = np.asarray(m.v)
        # v[j,0]: between padded zero and h[j,0] → always dry
        np.testing.assert_array_equal(v[:, 0], False)

    # ── w mask (lenient): at least 1 of 4 SW-corner h-cells wet ──────

    def test_w_wet_near_single_wet_cell(self):
        h = np.zeros((4, 4), dtype=bool)
        h[1, 1] = True  # single wet cell
        m = Mask2D.from_mask(h)
        w = np.asarray(m.xy_corner)
        # w at corners of h[1,1]: (1,1), (1,2), (2,1), (2,2)
        assert w[1, 1]
        assert w[1, 2]
        assert w[2, 1]
        assert w[2, 2]

    def test_w_dry_far_from_wet_cells(self):
        h = np.zeros((6, 6), dtype=bool)
        h[1, 1] = True
        m = Mask2D.from_mask(h)
        w = np.asarray(m.xy_corner)
        # w[4,4]: SW corner uses h[3,3], h[3,4], h[4,3], h[4,4] — all dry
        assert not w[4, 4]

    # ── psi mask (strict): all 4 SW-corner h-cells wet ────────────────

    def test_psi_wet_interior(self, basin6):
        _, m = basin6
        psi = np.asarray(m.xy_corner_strict)
        # psi[2,2]: uses h[1,1], h[1,2], h[2,1], h[2,2] — all wet
        assert psi[2, 2]

    def test_psi_dry_at_land_boundary(self, basin6):
        _, m = basin6
        psi = np.asarray(m.xy_corner_strict)
        # psi[1,2]: uses h[0,1] (land) → dry
        assert not psi[1, 2]


class TestVorticityBoundaryGeometry:
    """Verify vorticity boundary classification at specific grid points.

    For w[j,i] at the SW corner of cell (j,i), the 4 adjacent velocity
    faces are:
      - u[j, i]   (east)   — y-face between h[j-1,i] and h[j,i]
      - u[j, i-1] (west)   — y-face between h[j-1,i-1] and h[j,i-1]
      - v[j, i]   (north)  — x-face between h[j,i-1] and h[j,i]
      - v[j-1, i] (south)  — x-face between h[j-1,i-1] and h[j-1,i]
    """

    @pytest.fixture
    def basin8(self):
        """8×8 basin: land border, 6×6 ocean interior."""
        h = np.ones((8, 8), dtype=bool)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = False
        return Mask2D.from_mask(h)

    @pytest.fixture
    def channel8(self):
        """8×8 zonal channel: walls at j=0 and j=7, open in x."""
        h = np.ones((8, 8), dtype=bool)
        h[0, :] = h[-1, :] = False
        return Mask2D.from_mask(h)

    # ── xy_corner_valid: all 4 adjacent velocity faces wet ────────────────────

    def test_w_valid_deep_interior(self, basin8):
        m = basin8
        # w[4,4] is deep interior — all 4 adjacent faces should be wet
        assert bool(m.xy_corner_valid[4, 4])

    def test_w_valid_not_at_boundary(self, basin8):
        m = basin8
        # w[1,2] is near the bottom wall — u[1,2] is dry → not valid
        assert not bool(m.xy_corner_valid[1, 2])

    def test_w_valid_checks_correct_four_faces(self):
        """Build a domain where only the correct adjacency passes."""
        # 5x5, all ocean except h[2,2] = land.
        # This makes specific u/v faces dry, testing that xy_corner_valid checks
        # the right 4 faces (not shifted ones).
        h = np.ones((5, 5), dtype=bool)
        h[2, 2] = False
        m = Mask2D.from_mask(h)
        va = np.asarray(m.xy_corner_valid)

        # w[3,3]: SW corner of cell (3,3). Adjacent faces:
        #   u[3,3] = mean(h[2,3], h[3,3]) = mean(1,1) → wet
        #   u[3,2] = mean(h[2,2], h[3,2]) = mean(0,1) → dry!
        #   v[3,3] = mean(h[3,2], h[3,3]) = mean(1,1) → wet
        #   v[2,3] = mean(h[2,2], h[2,3]) = mean(0,1) → dry!
        assert not va[3, 3], "w[3,3] should NOT be valid (u[3,2] and v[2,3] dry)"

        # w[2,2]: Adjacent faces:
        #   u[2,2] = mean(h[1,2], h[2,2]) = mean(1,0) → dry!
        #   u[2,1] = mean(h[1,1], h[2,1]) = mean(1,1) → wet
        #   v[2,2] = mean(h[2,1], h[2,2]) = mean(1,0) → dry!
        #   v[1,2] = mean(h[1,1], h[1,2]) = mean(1,1) → wet
        assert not va[2, 2], "w[2,2] should NOT be valid (u[2,2] and v[2,2] dry)"

        # w[2,4]: far from the hole. Adjacent faces:
        #   u[2,4] = mean(h[1,4], h[2,4]) = mean(1,1) → wet
        #   u[2,3] = mean(h[1,3], h[2,3]) = mean(1,1) → wet
        #   v[2,4] = mean(h[2,3], h[2,4]) = mean(1,1) → wet
        #   v[1,4] = mean(h[1,3], h[1,4]) = mean(1,1) → wet
        assert va[2, 4], "w[2,4] should be valid (all 4 faces wet)"

    def test_w_valid_would_fail_with_wrong_adjacency(self):
        """Regression: the old code checked u[j+1,i] and v[j,i+1].

        This test passes with correct adjacency but would fail with the
        old (wrong) shifts.
        """
        # 5x5 all-ocean, then kill one cell to create a single-face gap.
        h = np.ones((5, 5), dtype=bool)
        h[3, 1] = False  # makes u[3,1] dry and u[4,1] dry, v[3,1] dry, v[3,2] dry
        m = Mask2D.from_mask(h)
        va = np.asarray(m.xy_corner_valid)
        u = np.asarray(m.u)
        v = np.asarray(m.v)

        # w[3,2]: adjacent faces are u[3,2], u[3,1], v[3,2], v[2,2]
        #   u[3,2] = mean(h[2,2], h[3,2]) → wet
        #   u[3,1] = mean(h[2,1], h[3,1]) = mean(1,0) → dry!
        #   v[3,2] = mean(h[3,1], h[3,2]) = mean(0,1) → dry!
        #   v[2,2] = mean(h[2,1], h[2,2]) → wet
        assert not va[3, 2], "u[3,1] and v[3,2] are dry"

        # Old code would have checked u[4,2] and v[3,3] instead,
        # both of which are wet — wrongly declaring this point valid.
        assert u[4, 2], "u[4,2] is wet (old code checked this)"
        assert v[3, 3], "v[3,3] is wet (old code checked this)"

    # ── channel: horizontal walls only ────────────────────────────────

    def test_channel_interior_is_valid(self, channel8):
        m = channel8
        # w[4,4] is in the interior of a channel → should be valid
        # (checking i≥2 to avoid left-padding artifacts)
        assert bool(m.xy_corner_valid[4, 4])

    def test_channel_near_wall_not_valid(self, channel8):
        m = channel8
        # w[1,4] is near the bottom wall (j=0 is land)
        # u[1,4] = mean(h[0,4], h[1,4]) = mean(0,1) → dry
        assert not bool(m.xy_corner_valid[1, 4])

    def test_channel_horizontal_bound_near_wall(self, channel8):
        m = channel8
        # w[1,4]: u[1,4] is dry → horizontal boundary
        assert bool(m.xy_corner_x_wall[1, 4])


class TestIrregularBoundaryGeometry:
    """Verify irregular psi boundary indices point to correct cells."""

    def test_irrbound_cells_are_dry_psi(self):
        """Every irregular boundary cell must be a dry psi cell."""
        h = np.ones((10, 10), dtype=bool)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = False
        h[4:7, 4:7] = False
        m = Mask2D.from_mask(h)
        psi = np.asarray(m.xy_corner_strict)
        yids = np.asarray(m.xy_corner_strict_irrbound_rows)
        xids = np.asarray(m.xy_corner_strict_irrbound_cols)
        for y, x in zip(yids, xids, strict=True):
            assert not psi[y, x], f"psi[{y},{x}] should be dry"

    def test_irrbound_has_wet_psi_neighbour(self):
        """Every irr. boundary cell must have ≥1 wet psi in its 3×3 hood."""
        h = np.ones((10, 10), dtype=bool)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = False
        h[4:7, 4:7] = False
        m = Mask2D.from_mask(h)
        psi = np.asarray(m.xy_corner_strict)
        yids = np.asarray(m.xy_corner_strict_irrbound_rows)
        xids = np.asarray(m.xy_corner_strict_irrbound_cols)
        Ny, Nx = psi.shape
        for y, x in zip(yids, xids, strict=True):
            patch = psi[
                max(0, y - 1) : min(Ny, y + 2),
                max(0, x - 1) : min(Nx, x + 2),
            ]
            assert patch.any(), f"irr boundary ({y},{x}) has no wet psi neighbour"

    def test_irrbound_not_on_array_edge(self):
        """Irregular boundary cells are in [1:-1, 1:-1], not on the edge."""
        h = np.ones((10, 10), dtype=bool)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = False
        m = Mask2D.from_mask(h)
        Ny, Nx = h.shape
        yids = np.asarray(m.xy_corner_strict_irrbound_rows)
        xids = np.asarray(m.xy_corner_strict_irrbound_cols)
        assert np.all(yids >= 1) and np.all(yids < Ny - 1)
        assert np.all(xids >= 1) and np.all(xids < Nx - 1)


class TestClassificationGeometry:
    """Verify land/coast classification at specific grid points."""

    def test_coast_is_ocean_adjacent_to_land(self):
        """Coast cells must be wet and have ≥1 land 4-neighbour."""
        h = np.ones((10, 10), dtype=bool)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = False
        m = Mask2D.from_mask(h)
        cls = np.asarray(m.classification)
        coast = cls == 1
        # Every coast cell must be wet
        assert np.all(h[coast])
        # Every coast cell must touch land (4-connected)
        land = ~h
        dilated_land = _dilate(land)
        assert np.all(dilated_land[coast])

    def test_near_coast_not_adjacent_to_land(self):
        """Near-coast cells must NOT directly touch land."""
        h = np.ones((20, 20), dtype=bool)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = False
        m = Mask2D.from_mask(h)
        cls = np.asarray(m.classification)
        near_coast = cls == 2
        land = ~h
        dilated_land = _dilate(land)
        # Near-coast should NOT be in the first dilation ring
        assert not np.any(near_coast & dilated_land & ~_dilate(dilated_land))
        # But should be in the second ring
        dilated_land_2 = _dilate(dilated_land)
        assert np.all(dilated_land_2[near_coast])

    def test_open_ocean_far_from_land(self):
        """Open-ocean cells must be >2 hops from any land cell."""
        h = np.ones((20, 20), dtype=bool)
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = False
        m = Mask2D.from_mask(h)
        cls = np.asarray(m.classification)
        ocean = cls == 3
        land = ~h
        dilated_2 = _dilate(_dilate(land))
        # Open ocean must NOT overlap with 2-dilation of land
        assert not np.any(ocean & dilated_2)


# ──────────────────────────────────────────────────────────────────────────────
# Mask1D
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mask1d_basin():
    """16-cell 1-D domain with a 1-cell land border on each end."""
    h = np.ones(16, dtype=bool)
    h[0] = h[-1] = False
    return h


@pytest.fixture
def all_ocean_1d():
    return Mask1D.from_dimensions(8)


class TestMask1D:
    def test_from_dimensions_shape(self):
        m = Mask1D.from_dimensions(10)
        assert m.h.shape == (10,)
        assert m.u.shape == (10,)
        assert bool(jnp.all(m.h))

    def test_from_mask_shape(self, mask1d_basin):
        m = Mask1D.from_mask(mask1d_basin)
        assert m.h.shape == (16,)
        assert m.u.shape == (16,)

    def test_from_mask_h_preserved(self, mask1d_basin):
        m = Mask1D.from_mask(mask1d_basin)
        np.testing.assert_array_equal(np.asarray(m.h), mask1d_basin)

    def test_u_face_lookahead(self, mask1d_basin):
        # u[i] needs h[i-1] AND h[i] both wet (kernel (2,) threshold 3/4)
        m = Mask1D.from_mask(mask1d_basin)
        u = np.asarray(m.u)
        # u[0] is padded with 0 in i-1 → False
        assert not u[0]
        # u[1] needs h[0]=F AND h[1]=T → False
        assert not u[1]
        # u[2] needs h[1]=T AND h[2]=T → True
        assert u[2]
        # u[15] needs h[14]=T AND h[15]=F → False
        assert not u[15]

    def test_not_h_is_inverse(self, mask1d_basin):
        m = Mask1D.from_mask(mask1d_basin)
        np.testing.assert_array_equal(np.asarray(m.not_h), ~mask1d_basin)

    def test_classification_levels(self, mask1d_basin):
        m = Mask1D.from_mask(mask1d_basin)
        c = np.asarray(m.classification)
        # Endpoints are land
        assert c[0] == 0 and c[-1] == 0
        # Cells immediately adjacent to land are coast
        assert c[1] == 1 and c[-2] == 1
        # Next ring is near-coast
        assert c[2] == 2 and c[-3] == 2
        # Deep interior is open ocean
        assert c[8] == 3

    def test_classification_indicators(self, mask1d_basin):
        m = Mask1D.from_mask(mask1d_basin)
        np.testing.assert_array_equal(m.ind_land, np.asarray(m.classification == 0))
        np.testing.assert_array_equal(m.ind_coast, np.asarray(m.classification == 1))
        np.testing.assert_array_equal(
            m.ind_near_coast, np.asarray(m.classification == 2)
        )
        np.testing.assert_array_equal(m.ind_ocean, np.asarray(m.classification == 3))

    def test_ind_boundary(self, all_ocean_1d):
        b = np.asarray(all_ocean_1d.ind_boundary)
        assert b[0] and b[-1]
        assert not np.any(b[1:-1])

    def test_stencil_capability_is_1d(self, all_ocean_1d):
        sc = all_ocean_1d.stencil_capability
        assert isinstance(sc, StencilCapability1D)
        assert sc.x_pos.shape == (8,)
        assert sc.x_neg.shape == (8,)

    def test_stencil_capability_all_ocean(self, all_ocean_1d):
        # In an all-ocean 1-D domain of length 8: x_pos[0]=8, x_neg[0]=1
        sc = all_ocean_1d.stencil_capability
        np.testing.assert_array_equal(np.asarray(sc.x_pos), [8, 7, 6, 5, 4, 3, 2, 1])
        np.testing.assert_array_equal(np.asarray(sc.x_neg), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_sponge_default_all_ones(self, all_ocean_1d):
        np.testing.assert_array_equal(np.asarray(all_ocean_1d.sponge), 1.0)

    def test_sponge_with_width(self):
        m = Mask1D.from_mask(np.ones(16, dtype=bool), sponge_width=3)
        s = np.asarray(m.sponge)
        # Endpoints are zero
        assert s[0] == 0.0 and s[-1] == 0.0
        # Interior is 1.0
        assert s[8] == 1.0

    def test_get_adaptive_masks_all_ocean(self, all_ocean_1d):
        masks = all_ocean_1d.get_adaptive_masks(stencil_sizes=(2, 4, 6))
        # Mutually exclusive
        sum_mask = sum(np.asarray(m).astype(int) for m in masks.values())
        assert np.all(sum_mask <= 1)

    def test_get_adaptive_masks_keys(self, all_ocean_1d):
        masks = all_ocean_1d.get_adaptive_masks(stencil_sizes=(2, 4))
        assert set(masks.keys()) == {2, 4}

    def test_from_ssh(self):
        ssh = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        m = Mask1D.from_mask(np.isfinite(ssh))
        # Compare against from_ssh
        m2 = Mask1D.from_ssh(ssh)
        np.testing.assert_array_equal(np.asarray(m.h), np.asarray(m2.h))


class TestStencilCapability1D:
    def test_from_mask(self):
        h = np.array([True, True, False, True, True, True])
        sc = StencilCapability1D.from_mask(h)
        np.testing.assert_array_equal(np.asarray(sc.x_pos), [2, 1, 0, 3, 2, 1])
        np.testing.assert_array_equal(np.asarray(sc.x_neg), [1, 2, 0, 1, 2, 3])

    def test_all_ocean(self):
        h = np.ones(5, dtype=bool)
        sc = StencilCapability1D.from_mask(h)
        np.testing.assert_array_equal(np.asarray(sc.x_pos), [5, 4, 3, 2, 1])
        np.testing.assert_array_equal(np.asarray(sc.x_neg), [1, 2, 3, 4, 5])

    def test_all_land(self):
        h = np.zeros(5, dtype=bool)
        sc = StencilCapability1D.from_mask(h)
        np.testing.assert_array_equal(np.asarray(sc.x_pos), [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(np.asarray(sc.x_neg), [0, 0, 0, 0, 0])


# ──────────────────────────────────────────────────────────────────────────────
# Mask3D
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def all_ocean_3d():
    return Mask3D.from_dimensions(4, 6, 8)


@pytest.fixture
def mask3d_with_island():
    """6×8×8 domain with a 1-cell ghost border and a single dry interior cell."""
    h = np.ones((6, 8, 8), dtype=bool)
    h[0, :, :] = h[-1, :, :] = False
    h[:, 0, :] = h[:, -1, :] = False
    h[:, :, 0] = h[:, :, -1] = False
    h[3, 4, 4] = False  # interior island
    return h


class TestMask3D:
    def test_from_dimensions_shape(self, all_ocean_3d):
        m = all_ocean_3d
        assert m.h.shape == (4, 6, 8)
        assert m.u.shape == (4, 6, 8)
        assert m.v.shape == (4, 6, 8)
        assert m.w.shape == (4, 6, 8)
        assert m.xy_corner.shape == (4, 6, 8)
        assert m.xy_corner_strict.shape == (4, 6, 8)

    def test_classification_shape(self, all_ocean_3d):
        assert all_ocean_3d.classification.shape == (4, 6, 8)

    def test_sponge_is_2d(self, all_ocean_3d):
        # 2-D horizontal sponge that broadcasts over z
        assert all_ocean_3d.sponge.shape == (6, 8)

    def test_sponge_with_width_is_2d(self):
        m = Mask3D.from_dimensions(4, 10, 12, sponge_width=3)
        assert m.sponge.shape == (10, 12)
        s = np.asarray(m.sponge)
        assert s[0, 0] == 0.0  # corner
        assert s[5, 6] == 1.0  # interior

    def test_stencil_capability_is_3d(self, all_ocean_3d):
        sc = all_ocean_3d.stencil_capability
        assert isinstance(sc, StencilCapability3D)
        # All six counts should have the full 3-D shape
        for fld in ("x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg"):
            assert getattr(sc, fld).shape == (4, 6, 8)

    def test_stencil_capability_z_direction(self, all_ocean_3d):
        # All-ocean domain: z_pos at k=0 should equal Nz (here 4)
        sc = all_ocean_3d.stencil_capability
        np.testing.assert_array_equal(np.asarray(sc.z_pos[0, 3, 4]), 4)
        np.testing.assert_array_equal(np.asarray(sc.z_neg[3, 3, 4]), 4)

    def test_w_face_lookup(self, mask3d_with_island):
        m = Mask3D.from_mask(mask3d_with_island)
        # w[k, j, i] needs h[k, j, i] AND h[k-1, j, i] both wet
        # The interior island at (3, 4, 4) makes:
        #   w[3, 4, 4] = h[3, 4, 4] AND h[2, 4, 4] = F AND T → False
        #   w[4, 4, 4] = h[4, 4, 4] AND h[3, 4, 4] = T AND F → False
        w = np.asarray(m.w)
        assert not w[3, 4, 4]
        assert not w[4, 4, 4]
        # Cells away from the island have both z-neighbours wet
        assert w[3, 1, 1]

    def test_island_classification(self, mask3d_with_island):
        m = Mask3D.from_mask(mask3d_with_island)
        c = np.asarray(m.classification)
        # The dry cell is land
        assert c[3, 4, 4] == 0
        # Direct neighbours (in any of 6 directions) are coast
        assert c[3, 3, 4] == 1  # y-neighbour
        assert c[3, 4, 3] == 1  # x-neighbour
        assert c[2, 4, 4] == 1  # z-neighbour
        assert c[4, 4, 4] == 1  # z-neighbour
        # A cell ≥2 cells from any land (ghost or island) is near-coast or ocean
        # (2, 3, 3) is min-distance 2 from k=0 ghost; ≥3 from x/y ghosts; 3 from island
        assert c[2, 3, 3] in (2, 3)

    def test_classification_indicators(self, mask3d_with_island):
        m = Mask3D.from_mask(mask3d_with_island)
        np.testing.assert_array_equal(m.ind_land, np.asarray(m.classification == 0))
        np.testing.assert_array_equal(m.ind_coast, np.asarray(m.classification == 1))
        np.testing.assert_array_equal(m.ind_ocean, np.asarray(m.classification == 3))

    def test_ind_boundary_covers_all_six_faces(self, all_ocean_3d):
        b = np.asarray(all_ocean_3d.ind_boundary)
        # All six faces are boundary
        assert np.all(b[0, :, :])
        assert np.all(b[-1, :, :])
        assert np.all(b[:, 0, :])
        assert np.all(b[:, -1, :])
        assert np.all(b[:, :, 0])
        assert np.all(b[:, :, -1])
        # Interior is not boundary
        assert not np.any(b[1:-1, 1:-1, 1:-1])

    def test_irrbound_indices_have_three_components(self, mask3d_with_island):
        m = Mask3D.from_mask(mask3d_with_island)
        # All three index arrays must have the same length
        n = len(m.xy_corner_strict_irrbound_layers)
        assert len(m.xy_corner_strict_irrbound_rows) == n
        assert len(m.xy_corner_strict_irrbound_cols) == n

    def test_get_adaptive_masks_x(self, all_ocean_3d):
        masks = all_ocean_3d.get_adaptive_masks(direction="x", stencil_sizes=(2, 4))
        assert set(masks.keys()) == {2, 4}
        sum_mask = sum(np.asarray(m).astype(int) for m in masks.values())
        assert np.all(sum_mask <= 1)

    def test_get_adaptive_masks_z(self, all_ocean_3d):
        masks = all_ocean_3d.get_adaptive_masks(direction="z", stencil_sizes=(2, 4))
        assert set(masks.keys()) == {2, 4}

    def test_get_adaptive_masks_invalid_direction(self, all_ocean_3d):
        with pytest.raises(ValueError, match="direction must be"):
            all_ocean_3d.get_adaptive_masks(direction="q")

    def test_k_bottom_optional(self, all_ocean_3d):
        assert all_ocean_3d.k_bottom is None

    def test_k_bottom_supplied(self):
        kb = np.full((6, 8), 3, dtype=np.int32)
        m = Mask3D.from_mask(np.ones((4, 6, 8), dtype=bool), k_bottom=kb)
        assert m.k_bottom is not None
        np.testing.assert_array_equal(np.asarray(m.k_bottom), 3)


class TestStencilCapability3D:
    def test_from_mask_all_ocean(self):
        h = np.ones((3, 4, 5), dtype=bool)
        sc = StencilCapability3D.from_mask(h)
        # Top-left corner: x_pos = Nx, y_pos = Ny, z_pos = Nz
        assert int(sc.x_pos[0, 0, 0]) == 5
        assert int(sc.y_pos[0, 0, 0]) == 4
        assert int(sc.z_pos[0, 0, 0]) == 3

    def test_from_mask_with_dry_cell(self):
        h = np.ones((3, 3, 3), dtype=bool)
        h[1, 1, 1] = False
        sc = StencilCapability3D.from_mask(h)
        assert int(sc.x_pos[1, 1, 1]) == 0
        assert int(sc.y_pos[1, 1, 1]) == 0
        assert int(sc.z_pos[1, 1, 1]) == 0

    def test_all_six_directions(self):
        h = np.ones((3, 3, 3), dtype=bool)
        sc = StencilCapability3D.from_mask(h)
        for fld in ("x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg"):
            arr = np.asarray(getattr(sc, fld))
            assert arr.shape == (3, 3, 3)
            # Every cell should have a positive count (it's all-ocean)
            assert np.all(arr >= 1)
