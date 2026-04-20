"""Tests for area/volume-weighted reduction helpers (Cartesian + spherical)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators.reductions import (
    area_mean,
    area_sum,
    area_weights,
    cartesian_area_mean,
    cartesian_area_sum,
    cartesian_volume_mean,
    cartesian_volume_sum,
    spherical_area_mean,
    spherical_area_sum,
    spherical_area_weights,
    spherical_volume_mean,
    spherical_volume_sum,
    volume_mean,
    volume_sum,
    volume_weights,
)

jax.config.update("jax_enable_x64", True)


R = 6.371e6


# ======================================================================
# Weights
# ======================================================================


class TestWeights:
    def test_cartesian_area_weights_uniform(self):
        grid = CartesianGrid2D.from_interior(
            nx_interior=10, ny_interior=8, Lx=20.0, Ly=16.0
        )
        w = area_weights(grid)
        np.testing.assert_allclose(w, grid.dx * grid.dy)

    def test_spherical_area_weights_cos_weighted(self):
        grid = SphericalGrid2D.from_interior(
            nx_interior=10,
            ny_interior=8,
            lon_range=(0.0, 360.0),
            lat_range=(-80.0, 80.0),
            R=R,
        )
        w = area_weights(grid)
        expected = grid.R**2 * grid.dlon * grid.dlat * grid.cos_lat_T
        np.testing.assert_allclose(w, expected, rtol=1e-12)

    def test_volume_weights_cartesian(self):
        grid = CartesianGrid3D.from_interior(
            nx_interior=8,
            ny_interior=8,
            nz_interior=4,
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
        )
        w = volume_weights(grid)
        np.testing.assert_allclose(w, grid.dx * grid.dy * grid.dz)

    def test_volume_weights_spherical(self):
        grid = SphericalGrid3D.from_interior(
            nx_interior=8,
            ny_interior=6,
            nz_interior=3,
            lon_range=(0.0, 360.0),
            lat_range=(-40.0, 40.0),
            Lz=100.0,
            R=R,
        )
        w = volume_weights(grid)
        expected_2d = spherical_area_weights(grid.horizontal_grid()) * grid.dz
        for k in range(grid.Nz):
            np.testing.assert_allclose(w[k], expected_2d, rtol=1e-12)

    def test_unsupported_grid_raises(self):
        class FakeGrid:
            Ny = 1
            Nx = 1

        with pytest.raises(TypeError, match="unsupported grid type"):
            area_weights(FakeGrid())


# ======================================================================
# Area sum / mean — Cartesian
# ======================================================================


class TestCartesianArea:
    @pytest.fixture
    def grid(self):
        return CartesianGrid2D.from_interior(
            nx_interior=10, ny_interior=8, Lx=10.0, Ly=8.0
        )

    def test_constant_field_area_sum(self, grid):
        h = 3.0 * jnp.ones((grid.Ny, grid.Nx))
        # Total wet area = interior_cells · dx · dy
        interior_cells = (grid.Ny - 2) * (grid.Nx - 2)
        expected = 3.0 * interior_cells * grid.dx * grid.dy
        np.testing.assert_allclose(area_sum(h, grid), expected, rtol=1e-12)

    def test_area_mean_constant_is_field_value(self, grid):
        c = 7.2
        h = c * jnp.ones((grid.Ny, grid.Nx))
        np.testing.assert_allclose(area_mean(h, grid), c, rtol=1e-12)

    def test_ghost_ring_ignored(self, grid):
        h = jnp.zeros((grid.Ny, grid.Nx))
        # Pollute ghost ring — it should be ignored.
        h = h.at[0, :].set(1e6).at[-1, :].set(1e6).at[:, 0].set(1e6).at[:, -1].set(1e6)
        np.testing.assert_allclose(area_sum(h, grid), 0.0, atol=1e-10)

    def test_alias_cartesian_area_sum(self, grid):
        h = jnp.arange(grid.Ny * grid.Nx, dtype=float).reshape(grid.Ny, grid.Nx)
        np.testing.assert_allclose(
            cartesian_area_sum(h, grid),
            area_sum(h, grid),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            cartesian_area_mean(h, grid),
            area_mean(h, grid),
            rtol=1e-12,
        )

    def test_all_dry_mean_is_nan(self, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        assert jnp.isnan(area_mean(h, grid, mask=all_dry))

    def test_mask_excludes_dry_cells(self, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        m = np.ones((grid.Ny, grid.Nx), dtype=bool)
        m[3, 4] = False
        area_mask = Mask2D.from_mask(m)
        s = area_sum(h, grid, mask=area_mask)
        # Total wet interior area is (Nx-2)(Ny-2) - 1 cells.
        expected = ((grid.Nx - 2) * (grid.Ny - 2) - 1) * grid.dx * grid.dy
        np.testing.assert_allclose(s, expected, rtol=1e-12)


# ======================================================================
# Area sum / mean — Spherical
# ======================================================================


class TestSphericalArea:
    def test_full_sphere_area_sum_approaches_4piR2(self):
        """Integrating 1 over the full sphere approximates 4πR²."""
        grid = SphericalGrid2D.from_interior(
            nx_interior=128,
            ny_interior=64,
            lon_range=(0.0, 360.0),
            lat_range=(-90.0, 90.0),
            R=R,
        )
        h = jnp.ones((grid.Ny, grid.Nx))
        total = area_sum(h, grid)
        expected = 4.0 * jnp.pi * R**2
        # Discretization introduces a ~1% error at this resolution; the
        # key property we care about is the right order of magnitude and
        # that refining the grid reduces the error.
        np.testing.assert_allclose(float(total), float(expected), rtol=0.02)

    def test_refining_grid_reduces_error(self):
        coarse = SphericalGrid2D.from_interior(
            32,
            16,
            lon_range=(0.0, 360.0),
            lat_range=(-80.0, 80.0),
            R=R,
        )
        fine = SphericalGrid2D.from_interior(
            128,
            64,
            lon_range=(0.0, 360.0),
            lat_range=(-80.0, 80.0),
            R=R,
        )
        h_c = jnp.ones((coarse.Ny, coarse.Nx))
        h_f = jnp.ones((fine.Ny, fine.Nx))

        # Analytical area of the lat band (-80, 80) on a sphere of radius R:
        #   A = 2π R² [sin(lat_max) - sin(lat_min)]
        lat_min_rad = jnp.deg2rad(-80.0)
        lat_max_rad = jnp.deg2rad(80.0)
        expected = 2.0 * jnp.pi * R**2 * (jnp.sin(lat_max_rad) - jnp.sin(lat_min_rad))

        err_c = abs(float(area_sum(h_c, coarse)) - float(expected)) / float(expected)
        err_f = abs(float(area_sum(h_f, fine)) - float(expected)) / float(expected)
        assert err_f < err_c

    def test_constant_mean_is_field_value(self):
        grid = SphericalGrid2D.from_interior(
            32,
            16,
            lon_range=(0.0, 360.0),
            lat_range=(-80.0, 80.0),
            R=R,
        )
        c = 5.0
        h = c * jnp.ones((grid.Ny, grid.Nx))
        np.testing.assert_allclose(area_mean(h, grid), c, rtol=1e-10)

    def test_equatorial_limit_matches_cartesian(self):
        """Narrow lat band around equator → spherical ≈ cartesian sum."""
        nx, ny = 32, 4
        sphere = SphericalGrid2D.from_interior(
            nx,
            ny,
            lon_range=(0.0, 10.0),
            lat_range=(-0.1, 0.1),
            R=R,
        )
        cart = CartesianGrid2D.from_interior(
            nx,
            ny,
            Lx=sphere.Lx,
            Ly=sphere.Ly,
        )
        key = jax.random.PRNGKey(7)
        h = jax.random.normal(key, (sphere.Ny, sphere.Nx))
        s_sph = area_sum(h, sphere)
        s_cart = area_sum(h, cart)
        np.testing.assert_allclose(float(s_sph), float(s_cart), rtol=1e-4)

    def test_alias_spherical_area_sum(self):
        grid = SphericalGrid2D.from_interior(
            16,
            10,
            lon_range=(0.0, 360.0),
            lat_range=(-40.0, 40.0),
            R=R,
        )
        key = jax.random.PRNGKey(8)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(
            spherical_area_sum(h, grid),
            area_sum(h, grid),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            spherical_area_mean(h, grid),
            area_mean(h, grid),
            rtol=1e-12,
        )

    def test_masked_all_dry_mean_nan(self):
        grid = SphericalGrid2D.from_interior(
            16,
            10,
            lon_range=(0.0, 360.0),
            lat_range=(-40.0, 40.0),
            R=R,
        )
        h = jnp.ones((grid.Ny, grid.Nx))
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        assert jnp.isnan(area_mean(h, grid, mask=all_dry))

    def test_jit_grad(self):
        grid = SphericalGrid2D.from_interior(
            16,
            10,
            lon_range=(0.0, 360.0),
            lat_range=(-40.0, 40.0),
            R=R,
        )
        h = jnp.ones((grid.Ny, grid.Nx))
        np.testing.assert_allclose(
            jax.jit(lambda x: area_sum(x, grid))(h),
            area_sum(h, grid),
        )
        g = jax.grad(lambda x: area_sum(x, grid))(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))


# ======================================================================
# Volume sum / mean
# ======================================================================


class TestVolume:
    @pytest.fixture
    def cart_grid(self):
        return CartesianGrid3D.from_interior(
            nx_interior=8,
            ny_interior=6,
            nz_interior=4,
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
        )

    @pytest.fixture
    def sph_grid(self):
        return SphericalGrid3D.from_interior(
            nx_interior=16,
            ny_interior=10,
            nz_interior=4,
            lon_range=(0.0, 360.0),
            lat_range=(-40.0, 40.0),
            Lz=100.0,
            R=R,
        )

    def test_cartesian_constant_volume_sum(self, cart_grid):
        h = 2.0 * jnp.ones((cart_grid.Nz, cart_grid.Ny, cart_grid.Nx))
        interior = (cart_grid.Nz - 2) * (cart_grid.Ny - 2) * (cart_grid.Nx - 2)
        expected = 2.0 * interior * cart_grid.dx * cart_grid.dy * cart_grid.dz
        np.testing.assert_allclose(volume_sum(h, cart_grid), expected, rtol=1e-12)

    def test_cartesian_volume_mean_constant(self, cart_grid):
        c = -1.5
        h = c * jnp.ones((cart_grid.Nz, cart_grid.Ny, cart_grid.Nx))
        np.testing.assert_allclose(volume_mean(h, cart_grid), c, rtol=1e-12)

    def test_spherical_volume_mean_constant(self, sph_grid):
        c = 3.14
        h = c * jnp.ones((sph_grid.Nz, sph_grid.Ny, sph_grid.Nx))
        np.testing.assert_allclose(volume_mean(h, sph_grid), c, rtol=1e-10)

    def test_spherical_volume_equals_area_times_dz_nz(self, sph_grid):
        h = jnp.ones((sph_grid.Nz, sph_grid.Ny, sph_grid.Nx))
        v = volume_sum(h, sph_grid)
        area = area_sum(
            jnp.ones((sph_grid.Ny, sph_grid.Nx)), sph_grid.horizontal_grid()
        )
        expected = area * sph_grid.dz * (sph_grid.Nz - 2)
        np.testing.assert_allclose(float(v), float(expected), rtol=1e-12)

    def test_aliases(self, cart_grid, sph_grid):
        h_c = jnp.ones((cart_grid.Nz, cart_grid.Ny, cart_grid.Nx))
        h_s = jnp.ones((sph_grid.Nz, sph_grid.Ny, sph_grid.Nx))
        np.testing.assert_allclose(
            cartesian_volume_sum(h_c, cart_grid),
            volume_sum(h_c, cart_grid),
        )
        np.testing.assert_allclose(
            cartesian_volume_mean(h_c, cart_grid),
            volume_mean(h_c, cart_grid),
        )
        np.testing.assert_allclose(
            spherical_volume_sum(h_s, sph_grid),
            volume_sum(h_s, sph_grid),
        )
        np.testing.assert_allclose(
            spherical_volume_mean(h_s, sph_grid),
            volume_mean(h_s, sph_grid),
        )

    def test_mask_excludes_dry_volume(self, cart_grid):
        h = jnp.ones((cart_grid.Nz, cart_grid.Ny, cart_grid.Nx))
        m = np.ones((cart_grid.Nz, cart_grid.Ny, cart_grid.Nx), dtype=bool)
        m[1, 2, 3] = False
        vmask = Mask3D.from_mask(m)
        cell_vol = cart_grid.dx * cart_grid.dy * cart_grid.dz
        total_interior = (cart_grid.Nz - 2) * (cart_grid.Ny - 2) * (cart_grid.Nx - 2)
        expected = (total_interior - 1) * cell_vol
        np.testing.assert_allclose(
            volume_sum(h, cart_grid, mask=vmask),
            expected,
            rtol=1e-12,
        )

    def test_all_dry_volume_mean_nan(self, cart_grid):
        h = jnp.ones((cart_grid.Nz, cart_grid.Ny, cart_grid.Nx))
        all_dry = Mask3D.from_mask(
            np.zeros((cart_grid.Nz, cart_grid.Ny, cart_grid.Nx), dtype=bool)
        )
        assert jnp.isnan(volume_mean(h, cart_grid, mask=all_dry))
