"""Tests for spherical diffusion (harmonic + biharmonic)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion2D, Diffusion2D
from finitevolx._src.diffusion.spherical_diffusion import (
    SphericalBiharmonicDiffusion2D,
    SphericalBiharmonicDiffusion3D,
    SphericalDiffusion2D,
    SphericalDiffusion3D,
)
from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D

jax.config.update("jax_enable_x64", True)


R = 6.371e6
NX_INT, NY_INT = 16, 10


@pytest.fixture
def grid():
    return SphericalGrid2D.from_interior(
        nx_interior=NX_INT,
        ny_interior=NY_INT,
        lon_range=(0.0, 360.0),
        lat_range=(-40.0, 40.0),
        R=R,
    )


@pytest.fixture
def grid3d():
    return SphericalGrid3D.from_interior(
        nx_interior=NX_INT,
        ny_interior=NY_INT,
        nz_interior=3,
        lon_range=(0.0, 360.0),
        lat_range=(-40.0, 40.0),
        Lz=100.0,
        R=R,
    )


@pytest.fixture
def diff_op(grid):
    return SphericalDiffusion2D(grid=grid)


@pytest.fixture
def diff_op3d(grid3d):
    return SphericalDiffusion3D(grid=grid3d)


# ======================================================================
# SphericalDiffusion2D
# ======================================================================


class TestSphericalDiffusion2D:
    def test_output_shape(self, diff_op, grid):
        h = jnp.zeros((grid.Ny, grid.Nx))
        out = diff_op(h, kappa=1.0)
        assert out.shape == (grid.Ny, grid.Nx)

    def test_ghost_ring_is_zero(self, diff_op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        out = diff_op(h, kappa=1.0)
        np.testing.assert_allclose(out[0, :], 0.0)
        np.testing.assert_allclose(out[-1, :], 0.0)
        np.testing.assert_allclose(out[:, 0], 0.0)
        np.testing.assert_allclose(out[:, -1], 0.0)

    def test_constant_tracer_zero_tendency(self, diff_op, grid):
        """Constant h → zero tendency everywhere (including non-constant cos)."""
        h = 3.5 * jnp.ones((grid.Ny, grid.Nx))
        out = diff_op(h, kappa=1.0)
        np.testing.assert_allclose(out[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_kappa_scales_linearly(self, diff_op, grid):
        """Doubling κ doubles the tendency."""
        key = jax.random.PRNGKey(0)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        t1 = diff_op(h, kappa=1.0)
        t2 = diff_op(h, kappa=2.0)
        np.testing.assert_allclose(t2[1:-1, 1:-1], 2.0 * t1[1:-1, 1:-1], rtol=1e-10)

    def test_zero_tendency_for_zero_kappa(self, diff_op, grid):
        key = jax.random.PRNGKey(1)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = diff_op(h, kappa=0.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_matches_cartesian_at_narrow_equatorial_band(self):
        """At the equator with a very thin lat band, cos(lat) ≈ 1 and the
        spherical operator reduces to the Cartesian flux-form diffusion
        with dx = R·dlon, dy = R·dlat."""
        nx, ny = 20, 6
        lon_range = (0.0, 20.0)  # small so dlon is small
        lat_range = (-0.5, 0.5)  # very narrow band around equator
        sphere = SphericalGrid2D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            lon_range=lon_range,
            lat_range=lat_range,
            R=R,
        )
        cart = CartesianGrid2D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            Lx=sphere.Lx,
            Ly=sphere.Ly,
        )
        key = jax.random.PRNGKey(3)
        h = jax.random.normal(key, (sphere.Ny, sphere.Nx))

        sph_op = SphericalDiffusion2D(grid=sphere)
        cart_op = Diffusion2D(grid=cart)

        t_sph = sph_op(h, kappa=1.0)
        t_cart = cart_op(h, kappa=1.0)

        # The two should agree to a few tolerance digits because cos(lat)
        # deviates from 1 by O(1e-4) across a 1-degree lat band.
        np.testing.assert_allclose(
            t_sph[1:-1, 1:-1],
            t_cart[1:-1, 1:-1],
            rtol=5e-4,
            atol=1e-4,
        )

    def test_fluxes_shape(self, diff_op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        assert fx.shape == (grid.Ny, grid.Nx)
        assert fy.shape == (grid.Ny, grid.Nx)

    def test_fluxes_constant_zero(self, diff_op, grid):
        h = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        # Constant tracer → zero flux everywhere.
        np.testing.assert_allclose(fx, 0.0, atol=1e-12)
        np.testing.assert_allclose(fy, 0.0, atol=1e-12)

    def test_fluxes_boundary_faces_zero(self, diff_op, grid):
        """East and north domain-wall faces are not written."""
        key = jax.random.PRNGKey(4)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        np.testing.assert_allclose(fx[:, -1], 0.0)  # east wall
        np.testing.assert_allclose(fx[:, -2], 0.0)  # east-wall face (not written)
        np.testing.assert_allclose(fx[0, :], 0.0)
        np.testing.assert_allclose(fx[-1, :], 0.0)
        np.testing.assert_allclose(fy[-1, :], 0.0)
        np.testing.assert_allclose(fy[-2, :], 0.0)

    def test_jit(self, diff_op, grid):
        key = jax.random.PRNGKey(5)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = jax.jit(diff_op.__call__)(h, 1.0)
        assert out.shape == (grid.Ny, grid.Nx)

    def test_grad(self, diff_op, grid):
        key = jax.random.PRNGKey(6)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        g = jax.grad(lambda x: diff_op(x, kappa=1.0).sum())(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))


# ======================================================================
# Masking
# ======================================================================


class TestSphericalDiffusion2DMasked:
    def test_all_dry_mask_zeros_tendency(self, grid):
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        op = SphericalDiffusion2D(grid=grid, mask=all_dry)
        key = jax.random.PRNGKey(7)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = op(h, kappa=1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_all_wet_mask_matches_unmasked(self, grid):
        all_wet = Mask2D.from_mask(np.ones((grid.Ny, grid.Nx), dtype=bool))
        op_masked = SphericalDiffusion2D(grid=grid, mask=all_wet)
        op_unmask = SphericalDiffusion2D(grid=grid)
        key = jax.random.PRNGKey(8)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(
            op_masked(h, kappa=1.0),
            op_unmask(h, kappa=1.0),
            atol=1e-12,
        )

    def test_dry_cell_zero(self, grid):
        """A dry T-cell has zero tendency in the masked tendency."""
        mask = np.ones((grid.Ny, grid.Nx), dtype=bool)
        mask[3, 5] = False  # make an interior cell dry
        op = SphericalDiffusion2D(grid=grid, mask=Mask2D.from_mask(mask))
        key = jax.random.PRNGKey(9)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = op(h, kappa=1.0)
        assert float(out[3, 5]) == 0.0


# ======================================================================
# SphericalDiffusion3D
# ======================================================================


class TestSphericalDiffusion3D:
    def test_output_shape(self, diff_op3d, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = diff_op3d(h, kappa=1.0)
        assert out.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_constant_tracer_zero(self, diff_op3d, grid3d):
        h = 2.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = diff_op3d(h, kappa=1.0)
        np.testing.assert_allclose(out[:, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_z_ghost_slices_zero(self, diff_op3d, grid3d):
        key = jax.random.PRNGKey(10)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = diff_op3d(h, kappa=1.0)
        np.testing.assert_allclose(out[0, :, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(out[-1, :, :], 0.0, atol=1e-10)

    def test_matches_2d_per_level(self, diff_op3d, grid3d):
        """Applying SphericalDiffusion3D on a z-broadcast field should
        equal the 2-D operator at each interior level."""
        op2d = SphericalDiffusion2D(grid=grid3d.horizontal_grid())
        key = jax.random.PRNGKey(11)
        h_2d = jax.random.normal(key, (grid3d.Ny, grid3d.Nx))
        h_3d = jnp.broadcast_to(h_2d, (grid3d.Nz, grid3d.Ny, grid3d.Nx))

        t_2d = op2d(h_2d, kappa=1.0)
        t_3d = diff_op3d(h_3d, kappa=1.0)

        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(t_3d[k], t_2d, atol=1e-12)

    def test_masked_all_dry_zero(self, grid3d):
        all_dry = Mask3D.from_mask(
            np.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        op = SphericalDiffusion3D(grid=grid3d, mask=all_dry)
        key = jax.random.PRNGKey(12)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = op(h, kappa=1.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_masked_all_wet_matches_unmasked(self, grid3d):
        all_wet = Mask3D.from_mask(
            np.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        op_m = SphericalDiffusion3D(grid=grid3d, mask=all_wet)
        op_u = SphericalDiffusion3D(grid=grid3d)
        key = jax.random.PRNGKey(13)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        np.testing.assert_allclose(op_m(h, 1.0), op_u(h, 1.0), atol=1e-12)

    def test_fluxes_shape(self, diff_op3d, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        fx, fy = diff_op3d.fluxes(h, kappa=1.0)
        assert fx.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert fy.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_jit_grad(self, diff_op3d, grid3d):
        key = jax.random.PRNGKey(14)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        jitted = jax.jit(diff_op3d.__call__)
        out = jitted(h, 1.0)
        assert out.shape == h.shape
        g = jax.grad(lambda x: diff_op3d(x, 1.0).sum())(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))


# ======================================================================
# SphericalBiharmonicDiffusion2D
# ======================================================================


class TestSphericalBiharmonicDiffusion2D:
    @pytest.fixture
    def op(self, grid):
        return SphericalBiharmonicDiffusion2D(grid=grid)

    def test_output_shape(self, op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        assert op(h, kappa=1.0).shape == (grid.Ny, grid.Nx)

    def test_ghost_ring_zero(self, op, grid):
        key = jax.random.PRNGKey(40)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = op(h, kappa=1.0)
        np.testing.assert_allclose(out[0, :], 0.0)
        np.testing.assert_allclose(out[-1, :], 0.0)
        np.testing.assert_allclose(out[:, 0], 0.0)
        np.testing.assert_allclose(out[:, -1], 0.0)

    def test_constant_field_zero(self, op, grid):
        h = 4.2 * jnp.ones((grid.Ny, grid.Nx))
        np.testing.assert_allclose(op(h, kappa=1.0)[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_kappa_scales_linearly(self, op, grid):
        key = jax.random.PRNGKey(41)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        t1 = op(h, kappa=1.0)
        t2 = op(h, kappa=2.5)
        np.testing.assert_allclose(t2[1:-1, 1:-1], 2.5 * t1[1:-1, 1:-1], rtol=1e-10)

    def test_zero_kappa(self, op, grid):
        key = jax.random.PRNGKey(42)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(op(h, kappa=0.0), 0.0, atol=1e-12)

    def test_sign_is_dissipative_for_small_scales(self, grid):
        """For a short-wave mode, positive kappa should dissipate.

        Evaluates -kappa * nabla^4 h on a sinusoidal field and checks
        that the sign of the tendency opposes the field at interior
        cells — the defining property of dissipative biharmonic mixing.
        """
        op = SphericalBiharmonicDiffusion2D(grid=grid)
        # Use grid indices to get an oscillating pattern.
        jj, ii = jnp.indices((grid.Ny, grid.Nx))
        h = jnp.cos(jnp.pi * ii / 3.0) * jnp.cos(jnp.pi * jj / 3.0)
        tend = op(h, kappa=1.0)
        # In the deep interior, sign(tend) * sign(h) should be <= 0
        # for a dissipative operator.
        deep_h = h[3:-3, 3:-3]
        deep_t = tend[3:-3, 3:-3]
        # Amplitudes where |h| is large — should be clearly opposite-signed.
        large = jnp.abs(deep_h) > 0.5
        assert jnp.all(jnp.where(large, deep_t * deep_h <= 0, True))

    def test_matches_cartesian_at_narrow_equatorial_band(self):
        nx, ny = 20, 6
        sphere = SphericalGrid2D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            lon_range=(0.0, 20.0),
            lat_range=(-0.5, 0.5),
            R=R,
        )
        cart = CartesianGrid2D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            Lx=sphere.Lx,
            Ly=sphere.Ly,
        )
        key = jax.random.PRNGKey(43)
        h = jax.random.normal(key, (sphere.Ny, sphere.Nx))
        t_s = SphericalBiharmonicDiffusion2D(grid=sphere)(h, kappa=1.0)
        t_c = BiharmonicDiffusion2D(grid=cart)(h, kappa=1.0)
        np.testing.assert_allclose(
            t_s[2:-2, 2:-2],
            t_c[2:-2, 2:-2],
            rtol=5e-3,
            atol=5e-3,
        )

    def test_jit_grad(self, op, grid):
        key = jax.random.PRNGKey(44)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = jax.jit(op.__call__)(h, 1.0)
        assert out.shape == h.shape
        g = jax.grad(lambda x: op(x, 1.0).sum())(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))

    def test_masked_all_dry_zero(self, grid):
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        op = SphericalBiharmonicDiffusion2D(grid=grid, mask=all_dry)
        key = jax.random.PRNGKey(45)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(op(h, kappa=1.0), 0.0, atol=1e-12)

    def test_masked_all_wet_matches_unmasked(self, grid):
        all_wet = Mask2D.from_mask(np.ones((grid.Ny, grid.Nx), dtype=bool))
        op_m = SphericalBiharmonicDiffusion2D(grid=grid, mask=all_wet)
        op_u = SphericalBiharmonicDiffusion2D(grid=grid)
        key = jax.random.PRNGKey(46)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        np.testing.assert_allclose(op_m(h, 1.0), op_u(h, 1.0), atol=1e-12)

    def test_dry_cell_zero(self, grid):
        m = np.ones((grid.Ny, grid.Nx), dtype=bool)
        m[4, 7] = False
        op = SphericalBiharmonicDiffusion2D(grid=grid, mask=Mask2D.from_mask(m))
        key = jax.random.PRNGKey(47)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        out = op(h, kappa=1.0)
        assert float(out[4, 7]) == 0.0


# ======================================================================
# SphericalBiharmonicDiffusion3D
# ======================================================================


class TestSphericalBiharmonicDiffusion3D:
    @pytest.fixture
    def op(self, grid3d):
        return SphericalBiharmonicDiffusion3D(grid=grid3d)

    def test_output_shape(self, op, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert op(h, kappa=1.0).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_constant_zero(self, op, grid3d):
        h = 1.7 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        np.testing.assert_allclose(op(h, kappa=1.0)[:, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_z_ghost_slices_zero(self, op, grid3d):
        key = jax.random.PRNGKey(48)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = op(h, kappa=1.0)
        np.testing.assert_allclose(out[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(out[-1], 0.0, atol=1e-10)

    def test_matches_2d_per_level(self, op, grid3d):
        op2d = SphericalBiharmonicDiffusion2D(grid=grid3d.horizontal_grid())
        key = jax.random.PRNGKey(49)
        h2 = jax.random.normal(key, (grid3d.Ny, grid3d.Nx))
        h3 = jnp.broadcast_to(h2, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        t2 = op2d(h2, kappa=1.0)
        t3 = op(h3, kappa=1.0)
        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(t3[k], t2, atol=1e-12)

    def test_masked_all_dry_zero(self, grid3d):
        all_dry = Mask3D.from_mask(
            np.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        op = SphericalBiharmonicDiffusion3D(grid=grid3d, mask=all_dry)
        key = jax.random.PRNGKey(50)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        np.testing.assert_allclose(op(h, kappa=1.0), 0.0, atol=1e-12)

    def test_jit_grad(self, op, grid3d):
        key = jax.random.PRNGKey(51)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        out = jax.jit(op.__call__)(h, 1.0)
        assert out.shape == h.shape
        g = jax.grad(lambda x: op(x, 1.0).sum())(h)
        assert g.shape == h.shape
        assert jnp.all(jnp.isfinite(g))
