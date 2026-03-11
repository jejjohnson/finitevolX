"""Tests for the upwind_flux function and Advection mask integration."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.advection import Advection2D, Advection3D
from finitevolx._src.advection.flux import upwind_flux
from finitevolx._src.advection.reconstruction import Reconstruction2D
from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.grid.grid import ArakawaCGrid2D, ArakawaCGrid3D

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def grid2d():
    """10×10 grid (8×8 interior cells)."""
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def all_ocean_mask():
    """All-wet 10×10 mask (no land)."""
    return ArakawaCGridMask.from_dimensions(10, 10)


@pytest.fixture
def coastal_mask():
    """10×10 mask with a land column at x=4–5, forcing near-land fallback."""
    h = np.ones((10, 10), dtype=bool)
    h[:, 4:6] = False
    return ArakawaCGridMask.from_mask(h)


@pytest.fixture
def island_mask():
    """20×20 mask with a 1-cell land border and a 4×4 interior island."""
    h = np.ones((20, 20), dtype=bool)
    h[0, :] = False
    h[-1, :] = False
    h[:, 0] = False
    h[:, -1] = False
    h[8:12, 8:12] = False
    return ArakawaCGridMask.from_mask(h)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_rec_funcs_x(recon: Reconstruction2D) -> dict:
    """Return {2: upwind1_x, 4: weno3_x, 6: weno5_x}."""
    return {2: recon.upwind1_x, 4: recon.weno3_x, 6: recon.weno5_x}


def _make_rec_funcs_y(recon: Reconstruction2D) -> dict:
    """Return {2: upwind1_y, 4: weno3_y, 6: weno5_y}."""
    return {2: recon.upwind1_y, 4: recon.weno3_y, 6: recon.weno5_y}


# ── basic output properties ───────────────────────────────────────────────────


class TestUpwindFluxBasic:
    def test_output_shape_x(self, grid2d, all_ocean_mask):
        recon = Reconstruction2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="x", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, u, dim=1, rec_funcs=_make_rec_funcs_x(recon), mask_hierarchy=mask_hier
        )
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_output_shape_y(self, grid2d, all_ocean_mask):
        recon = Reconstruction2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="y", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, v, dim=0, rec_funcs=_make_rec_funcs_y(recon), mask_hierarchy=mask_hier
        )
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_is_zero_x(self, grid2d, all_ocean_mask):
        recon = Reconstruction2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="x", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, u, dim=1, rec_funcs=_make_rec_funcs_x(recon), mask_hierarchy=mask_hier
        )
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, -1], 0.0)

    def test_ghost_ring_is_zero_y(self, grid2d, all_ocean_mask):
        recon = Reconstruction2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="y", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, v, dim=0, rec_funcs=_make_rec_funcs_y(recon), mask_hierarchy=mask_hier
        )
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, -1], 0.0)

    def test_raises_on_empty_rec_funcs(self, grid2d, all_ocean_mask):
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="x", stencil_sizes=(2, 4, 6)
        )
        with pytest.raises(ValueError, match="rec_funcs"):
            upwind_flux(q, u, dim=1, rec_funcs={}, mask_hierarchy=mask_hier)

    def test_raises_on_invalid_dim(self, grid2d, all_ocean_mask):
        recon = Reconstruction2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="x", stencil_sizes=(2, 4, 6)
        )
        with pytest.raises(ValueError, match="dim"):
            upwind_flux(
                q,
                u,
                dim=2,
                rec_funcs=_make_rec_funcs_x(recon),
                mask_hierarchy=mask_hier,
            )

    def test_top_level_import(self):
        from finitevolx import upwind_flux as uf

        assert uf is not None


# ── rectangular domain (no boundary fallback needed) ─────────────────────────


class TestUpwindFluxRectangular:
    """On a rectangular all-ocean domain the masked flux must equal the pure
    single-stencil reconstruction."""

    def test_all_ocean_x_positive_constant(self, grid2d, all_ocean_mask):
        """Constant field, positive flow: flux must equal q everywhere."""
        recon = Reconstruction2D(grid=grid2d)
        q = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="x", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, u, dim=1, rec_funcs=_make_rec_funcs_x(recon), mask_hierarchy=mask_hier
        )
        np.testing.assert_allclose(result[1:-1, 1:-1], 3.0, rtol=1e-5)

    def test_all_ocean_x_negative_constant(self, grid2d, all_ocean_mask):
        """Constant field, negative flow: flux must equal -q everywhere."""
        recon = Reconstruction2D(grid=grid2d)
        q = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="x", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, u, dim=1, rec_funcs=_make_rec_funcs_x(recon), mask_hierarchy=mask_hier
        )
        np.testing.assert_allclose(result[1:-1, 1:-1], -3.0, rtol=1e-5)

    def test_all_ocean_y_positive_constant(self, grid2d, all_ocean_mask):
        recon = Reconstruction2D(grid=grid2d)
        q = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="y", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, v, dim=0, rec_funcs=_make_rec_funcs_y(recon), mask_hierarchy=mask_hier
        )
        np.testing.assert_allclose(result[1:-1, 1:-1], 4.0, rtol=1e-5)

    def test_all_ocean_y_negative_constant(self, grid2d, all_ocean_mask):
        recon = Reconstruction2D(grid=grid2d)
        q = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        mask_hier = all_ocean_mask.get_adaptive_masks(
            direction="y", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, v, dim=0, rec_funcs=_make_rec_funcs_y(recon), mask_hierarchy=mask_hier
        )
        np.testing.assert_allclose(result[1:-1, 1:-1], -4.0, rtol=1e-5)

    def test_all_ocean_x_matches_weno5_x(self):
        """On an all-ocean mask, upwind_flux must equal Reconstruction2D.weno5_x."""
        # Large enough grid for interior cells to qualify for WENO5 (stencil size 6)
        grid = ArakawaCGrid2D.from_interior(12, 12, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        mask = ArakawaCGridMask.from_dimensions(14, 14)
        q = jnp.broadcast_to(jnp.arange(14, dtype=float), (14, 14))
        u = jnp.ones((14, 14))
        mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        result = upwind_flux(
            q, u, dim=1, rec_funcs=_make_rec_funcs_x(recon), mask_hierarchy=mask_hier
        )
        expected = recon.weno5_x(q, u)
        # Interior cells (depth ≥ 3 from ghost) should match WENO5 exactly
        np.testing.assert_allclose(result[1:-1, 3:-3], expected[1:-1, 3:-3], rtol=1e-5)

    def test_all_ocean_y_matches_weno5_y(self):
        """On an all-ocean mask, upwind_flux must equal Reconstruction2D.weno5_y."""
        grid = ArakawaCGrid2D.from_interior(12, 12, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        mask = ArakawaCGridMask.from_dimensions(14, 14)
        q = jnp.broadcast_to(jnp.arange(14, dtype=float)[:, None], (14, 14))
        v = jnp.ones((14, 14))
        mask_hier = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        result = upwind_flux(
            q, v, dim=0, rec_funcs=_make_rec_funcs_y(recon), mask_hierarchy=mask_hier
        )
        expected = recon.weno5_y(q, v)
        np.testing.assert_allclose(result[3:-3, 1:-1], expected[3:-3, 1:-1], rtol=1e-5)


# ── irregular / masked domain: stencil fallback near boundaries ───────────────


class TestUpwindFluxMaskedDomain:
    """Verify that near-land cells receive lower-order stencils."""

    def test_coastal_x_finite_everywhere(self, coastal_mask):
        grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        q = jnp.broadcast_to(jnp.arange(10, dtype=float), (10, 10))
        u = jnp.ones((10, 10))
        mask_hier = coastal_mask.get_adaptive_masks(
            direction="x", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, u, dim=1, rec_funcs=_make_rec_funcs_x(recon), mask_hierarchy=mask_hier
        )
        assert jnp.all(jnp.isfinite(result)).item()

    def test_coastal_y_finite_everywhere(self, coastal_mask):
        grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        q = jnp.broadcast_to(jnp.arange(10, dtype=float)[:, None], (10, 10))
        v = jnp.ones((10, 10))
        mask_hier = coastal_mask.get_adaptive_masks(
            direction="y", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, v, dim=0, rec_funcs=_make_rec_funcs_y(recon), mask_hierarchy=mask_hier
        )
        assert jnp.all(jnp.isfinite(result)).item()

    def test_coastal_x_fallback_differs_from_weno5(self, coastal_mask):
        """Near-land cells must produce different values from pure WENO5.

        The land barrier at columns 4-5 forces upwind cells at cols 3 and 6
        into the 2-point or 4-point stencil tier.  With a non-constant field
        this produces different flux values than the unconstrained WENO5.
        """
        grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        x_idx = jnp.arange(10, dtype=float)
        q = jnp.broadcast_to(x_idx, (10, 10))
        for sign in (1.0, -1.0):
            u = sign * jnp.ones((10, 10))
            mask_hier = coastal_mask.get_adaptive_masks(
                direction="x", stencil_sizes=(2, 4, 6)
            )
            result = upwind_flux(
                q,
                u,
                dim=1,
                rec_funcs=_make_rec_funcs_x(recon),
                mask_hierarchy=mask_hier,
            )
            ref = recon.weno5_x(q, u)
            row = slice(2, 8)
            # At least one of the near-land columns must differ from WENO5
            diffs = jnp.any(result[row, 3] != ref[row, 3]) | jnp.any(
                result[row, 6] != ref[row, 6]
            )
            assert diffs.item(), f"Expected fallback for sign={sign}"

    def test_island_x_finite_everywhere(self, island_mask):
        grid = ArakawaCGrid2D.from_interior(18, 18, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        q = jnp.ones((20, 20))
        u = jnp.ones((20, 20))
        mask_hier = island_mask.get_adaptive_masks(
            direction="x", stencil_sizes=(2, 4, 6)
        )
        result = upwind_flux(
            q, u, dim=1, rec_funcs=_make_rec_funcs_x(recon), mask_hierarchy=mask_hier
        )
        assert jnp.all(jnp.isfinite(result)).item()


# ── conservation: total tracer preserved by divergence of upwind flux ─────────


class TestUpwindFluxConservation:
    """Verify discrete conservation properties of the upwind flux.

    Conservation is checked via the **telescoping cancellation** property:
    the sum of all cell divergences equals the net boundary flux.  With
    closed-basin no-flux BCs (zero velocity at the domain walls), both
    boundary fluxes are zero, so the total divergence is zero and the
    total tracer is preserved.

    In the ghost-ring convention used by finitevolX:
    * West/south walls: ghost ring is already zero (no-flux automatic).
    * East wall: the last interior U-face (column ``-2``) must be zeroed.
    * North wall: the last interior V-face (row ``-2``) must be zeroed.
    """

    def test_x_zero_total_divergence_with_no_flux_bc(self):
        """No-flux east wall: telescoping x-divergence sums to zero.

        Sum_i (fe[j,i] - fe[j,i-1]) = fe[east wall] - fe[west wall] = 0 - 0.
        """
        grid = ArakawaCGrid2D.from_interior(12, 12, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        mask = ArakawaCGridMask.from_dimensions(14, 14)
        q = 2.0 * jnp.ones((14, 14))
        # Uniform flow, but zero at the east-wall U-face so no flux exits
        u = jnp.ones((14, 14)).at[:, -2].set(0.0)
        mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
        fe = upwind_flux(
            q, u, dim=1, rec_funcs=_make_rec_funcs_x(recon), mask_hierarchy=mask_hier
        )
        # Telescoping: sum = fe[east wall] - fe[west wall ghost] = 0 - 0
        total_div = float(jnp.sum(fe[1:-1, 1:-1] - fe[1:-1, :-2]))
        assert abs(total_div) < 1e-5, f"Expected zero total divergence, got {total_div}"

    def test_y_zero_total_divergence_with_no_flux_bc(self):
        """No-flux north wall: telescoping y-divergence sums to zero.

        Sum_j (fn[j,i] - fn[j-1,i]) = fn[north wall] - fn[south wall] = 0 - 0.
        """
        grid = ArakawaCGrid2D.from_interior(12, 12, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        mask = ArakawaCGridMask.from_dimensions(14, 14)
        q = 5.0 * jnp.ones((14, 14))
        # Uniform flow, but zero at the north-wall V-face so no flux exits
        v = jnp.ones((14, 14)).at[-2, :].set(0.0)
        mask_hier = mask.get_adaptive_masks(direction="y", stencil_sizes=(2, 4, 6))
        fn = upwind_flux(
            q, v, dim=0, rec_funcs=_make_rec_funcs_y(recon), mask_hierarchy=mask_hier
        )
        # Telescoping: sum = fn[north wall] - fn[south wall ghost] = 0 - 0
        total_div = float(jnp.sum(fn[1:-1, 1:-1] - fn[:-2, 1:-1]))
        assert abs(total_div) < 1e-5, f"Expected zero total divergence, got {total_div}"


# ── Advection2D mask integration ─────────────────────────────────────────────


class TestAdvection2DMasked:
    """Verify Advection2D works with the mask parameter."""

    def test_rectangular_weno5_matches_unmasked(self):
        """On a rectangular domain, masked weno5 should match unmasked."""
        Ny, Nx = 14, 14
        grid = ArakawaCGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
        mask = ArakawaCGridMask.from_dimensions(Ny, Nx)
        adv = Advection2D(grid=grid)
        h = jnp.broadcast_to(jnp.arange(Nx, dtype=float), (Ny, Nx))
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        result_masked = adv(h, u, v, method="weno5", mask=mask)
        result_plain = adv(h, u, v, method="weno5")
        # Deep interior should match (boundary cells may differ due to
        # mask-based stencil selection at the domain edge)
        np.testing.assert_allclose(
            result_masked[4:-4, 4:-4], result_plain[4:-4, 4:-4], rtol=1e-5
        )

    def test_rectangular_weno3_runs(self):
        Ny, Nx = 10, 10
        grid = ArakawaCGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
        mask = ArakawaCGridMask.from_dimensions(Ny, Nx)
        adv = Advection2D(grid=grid)
        h = jnp.ones((Ny, Nx))
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        result = adv(h, u, v, method="weno3", mask=mask)
        assert result.shape == (Ny, Nx)
        assert jnp.all(jnp.isfinite(result)).item()

    def test_rectangular_tvd_runs(self):
        Ny, Nx = 10, 10
        grid = ArakawaCGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
        mask = ArakawaCGridMask.from_dimensions(Ny, Nx)
        adv = Advection2D(grid=grid)
        h = jnp.ones((Ny, Nx)) * 2.0
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        result = adv(h, u, v, method="minmod", mask=mask)
        assert result.shape == (Ny, Nx)
        assert jnp.all(jnp.isfinite(result)).item()

    def test_coastal_fallback_finite(self):
        """Masked domain with land barrier — result must be finite everywhere."""
        Ny, Nx = 10, 10
        h_mask = np.ones((Ny, Nx), dtype=bool)
        h_mask[:, 4:6] = False
        mask = ArakawaCGridMask.from_mask(h_mask)
        grid = ArakawaCGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
        adv = Advection2D(grid=grid)
        h = jnp.broadcast_to(jnp.arange(Nx, dtype=float), (Ny, Nx))
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        result = adv(h, u, v, method="weno5", mask=mask)
        assert jnp.all(jnp.isfinite(result)).item()

    def test_nan_on_land_does_not_corrupt(self):
        """NaN values on land cells must not propagate into wet-cell fluxes."""
        Ny, Nx = 12, 12
        h_mask = np.ones((Ny, Nx), dtype=bool)
        h_mask[:, 0] = False
        h_mask[:, -1] = False
        mask = ArakawaCGridMask.from_mask(h_mask)
        grid = ArakawaCGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
        adv = Advection2D(grid=grid)
        h = jnp.ones((Ny, Nx))
        # Put NaN on land columns
        h = h.at[:, 0].set(jnp.nan)
        h = h.at[:, -1].set(jnp.nan)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        result = adv(h, u, v, method="weno5", mask=mask)
        # Interior wet cells should be finite
        wet_interior = result[3:-3, 3:-3]
        assert jnp.all(jnp.isfinite(wet_interior)).item()

    def test_mask_none_is_noop(self):
        """mask=None should produce identical results to no mask."""
        Ny, Nx = 10, 10
        grid = ArakawaCGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
        adv = Advection2D(grid=grid)
        h = jnp.ones((Ny, Nx)) * 3.0
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        r1 = adv(h, u, v, method="weno5")
        r2 = adv(h, u, v, method="weno5", mask=None)
        np.testing.assert_array_equal(r1, r2)

    def test_non_dispatchable_method_ignores_mask(self):
        """Methods like weno7 should ignore the mask and use unmasked path."""
        Ny, Nx = 14, 14
        grid = ArakawaCGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
        mask = ArakawaCGridMask.from_dimensions(Ny, Nx)
        adv = Advection2D(grid=grid)
        h = jnp.ones((Ny, Nx)) * 2.0
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        r1 = adv(h, u, v, method="weno7")
        r2 = adv(h, u, v, method="weno7", mask=mask)
        np.testing.assert_array_equal(r1, r2)


# ── Advection3D mask integration ─────────────────────────────────────────────


class TestAdvection3DMasked:
    """Verify Advection3D works with the mask parameter."""

    def test_rectangular_weno3_runs(self):
        Nz, Ny, Nx = 3, 10, 10
        grid = ArakawaCGrid3D.from_interior(Nx - 2, Ny - 2, Nz - 2, 1.0, 1.0, 1.0)
        mask = ArakawaCGridMask.from_dimensions(Ny, Nx)
        adv = Advection3D(grid=grid)
        h = jnp.ones((Nz, Ny, Nx))
        u = jnp.ones((Nz, Ny, Nx))
        v = jnp.ones((Nz, Ny, Nx))
        result = adv(h, u, v, method="weno3", mask=mask)
        assert result.shape == (Nz, Ny, Nx)
        assert jnp.all(jnp.isfinite(result)).item()

    def test_rectangular_weno5_runs(self):
        Nz, Ny, Nx = 3, 14, 14
        grid = ArakawaCGrid3D.from_interior(Nx - 2, Ny - 2, Nz - 2, 1.0, 1.0, 1.0)
        mask = ArakawaCGridMask.from_dimensions(Ny, Nx)
        adv = Advection3D(grid=grid)
        h = jnp.ones((Nz, Ny, Nx))
        u = jnp.ones((Nz, Ny, Nx))
        v = jnp.ones((Nz, Ny, Nx))
        result = adv(h, u, v, method="weno5", mask=mask)
        assert result.shape == (Nz, Ny, Nx)
        assert jnp.all(jnp.isfinite(result)).item()

    def test_rectangular_tvd_runs(self):
        Nz, Ny, Nx = 3, 10, 10
        grid = ArakawaCGrid3D.from_interior(Nx - 2, Ny - 2, Nz - 2, 1.0, 1.0, 1.0)
        mask = ArakawaCGridMask.from_dimensions(Ny, Nx)
        adv = Advection3D(grid=grid)
        h = jnp.ones((Nz, Ny, Nx)) * 2.0
        u = jnp.ones((Nz, Ny, Nx))
        v = jnp.ones((Nz, Ny, Nx))
        result = adv(h, u, v, method="minmod", mask=mask)
        assert result.shape == (Nz, Ny, Nx)
        assert jnp.all(jnp.isfinite(result)).item()

    def test_coastal_fallback_finite(self):
        """3D masked domain with land barrier — result must be finite."""
        Nz, Ny, Nx = 3, 10, 10
        h_mask = np.ones((Ny, Nx), dtype=bool)
        h_mask[:, 4:6] = False
        mask = ArakawaCGridMask.from_mask(h_mask)
        grid = ArakawaCGrid3D.from_interior(Nx - 2, Ny - 2, Nz - 2, 1.0, 1.0, 1.0)
        adv = Advection3D(grid=grid)
        h = jnp.ones((Nz, Ny, Nx))
        u = jnp.ones((Nz, Ny, Nx))
        v = jnp.ones((Nz, Ny, Nx))
        result = adv(h, u, v, method="weno5", mask=mask)
        assert jnp.all(jnp.isfinite(result)).item()

    def test_mask_none_is_noop(self):
        """mask=None should produce identical results to no mask."""
        Nz, Ny, Nx = 3, 10, 10
        grid = ArakawaCGrid3D.from_interior(Nx - 2, Ny - 2, Nz - 2, 1.0, 1.0, 1.0)
        adv = Advection3D(grid=grid)
        h = jnp.ones((Nz, Ny, Nx)) * 3.0
        u = jnp.ones((Nz, Ny, Nx))
        v = jnp.ones((Nz, Ny, Nx))
        r1 = adv(h, u, v, method="weno5")
        r2 = adv(h, u, v, method="weno5", mask=None)
        np.testing.assert_array_equal(r1, r2)


# ── upwind_flux validation tests ─────────────────────────────────────────────


class TestUpwindFluxValidation:
    """Tests for upwind_flux input validation (RC1)."""

    def test_raises_on_missing_mask_keys(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        mask = ArakawaCGridMask.from_dimensions(grid2d.Ny, grid2d.Nx)
        # Only provide mask for size 2, but rec_funcs expects 2 and 6
        mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2,))
        with pytest.raises(ValueError, match="missing masks"):
            upwind_flux(
                q,
                u,
                dim=1,
                rec_funcs={2: recon.upwind1_x, 6: recon.weno5_x},
                mask_hierarchy=mask_hier,
            )

    def test_raises_on_empty_mask_hierarchy(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        with pytest.raises(ValueError, match="mask_hierarchy"):
            upwind_flux(
                q,
                u,
                dim=1,
                rec_funcs={2: recon.upwind1_x},
                mask_hierarchy={},
            )
