"""Tests for SphericalMomentumAdvection2D and SphericalMomentumAdvection3D."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.diffusion.momentum import MomentumAdvection2D
from finitevolx._src.diffusion.spherical_momentum import (
    SphericalMomentumAdvection2D,
    SphericalMomentumAdvection3D,
)
from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D

jax.config.update("jax_enable_x64", True)


R = 6.371e6
NX_INT, NY_INT = 14, 10


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


# ======================================================================
# SphericalMomentumAdvection2D
# ======================================================================


class TestSphericalMomentumAdvection2D:
    @pytest.fixture
    def madv(self, grid):
        return SphericalMomentumAdvection2D(grid=grid)

    def test_output_shapes(self, madv, grid):
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        du, dv = madv(u, v)
        assert du.shape == (grid.Ny, grid.Nx)
        assert dv.shape == (grid.Ny, grid.Nx)

    def test_zero_velocity_zero_tendency(self, madv, grid):
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        du, dv = madv(u, v)
        np.testing.assert_allclose(du, 0.0, atol=1e-12)
        np.testing.assert_allclose(dv, 0.0, atol=1e-12)

    def test_outer_ghost_ring_zero(self, madv, grid):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        du, dv = madv(u, v)
        for field in (du, dv):
            np.testing.assert_allclose(field[:2, :], 0.0)
            np.testing.assert_allclose(field[-2:, :], 0.0)
            np.testing.assert_allclose(field[:, :2], 0.0)
            np.testing.assert_allclose(field[:, -2:], 0.0)

    @pytest.mark.parametrize("scheme", ["energy", "enstrophy", "al"])
    def test_schemes_produce_finite_output(self, madv, grid, scheme):
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        du, dv = madv(u, v, scheme=scheme)
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))

    def test_invalid_scheme_raises(self, madv, grid):
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        with pytest.raises(ValueError, match="Unknown scheme"):
            madv(u, v, scheme="bogus")

    @pytest.mark.parametrize("scheme", ["energy", "enstrophy", "al"])
    def test_matches_cartesian_at_narrow_equatorial_band(self, scheme):
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
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (sphere.Ny, sphere.Nx))
        v = jax.random.normal(k2, (sphere.Ny, sphere.Nx))

        madv_s = SphericalMomentumAdvection2D(grid=sphere)
        madv_c = MomentumAdvection2D(grid=cart)
        du_s, dv_s = madv_s(u, v, scheme=scheme)
        du_c, dv_c = madv_c(u, v, scheme=scheme)
        # 1-degree lat band → cos varies by O(1e-4); tolerances set
        # conservatively because the compound operator accumulates a
        # few reciprocal-cosine factors.
        np.testing.assert_allclose(
            du_s[2:-2, 2:-2],
            du_c[2:-2, 2:-2],
            rtol=5e-3,
            atol=5e-3,
        )
        np.testing.assert_allclose(
            dv_s[2:-2, 2:-2],
            dv_c[2:-2, 2:-2],
            rtol=5e-3,
            atol=5e-3,
        )

    def test_jit(self, madv, grid):
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        du, dv = jax.jit(lambda a, b: madv(a, b, scheme="al"))(u, v)
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))

    def test_grad(self, madv, grid):
        key = jax.random.PRNGKey(4)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))

        def loss(a, b):
            du, dv = madv(a, b, scheme="energy")
            return (du**2 + dv**2).sum()

        gu, gv = jax.grad(loss, argnums=(0, 1))(u, v)
        assert gu.shape == u.shape and gv.shape == v.shape
        assert jnp.all(jnp.isfinite(gu)) and jnp.all(jnp.isfinite(gv))


class TestSphericalMomentumAdvection2DMasked:
    def test_all_dry_zero(self, grid):
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        madv = SphericalMomentumAdvection2D(grid=grid, mask=all_dry)
        key = jax.random.PRNGKey(5)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        du, dv = madv(u, v, scheme="al")
        np.testing.assert_allclose(du, 0.0, atol=1e-12)
        np.testing.assert_allclose(dv, 0.0, atol=1e-12)

    def test_all_wet_matches_unmasked(self, grid):
        all_wet = Mask2D.from_mask(np.ones((grid.Ny, grid.Nx), dtype=bool))
        madv_m = SphericalMomentumAdvection2D(grid=grid, mask=all_wet)
        madv_u = SphericalMomentumAdvection2D(grid=grid)
        key = jax.random.PRNGKey(6)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        du_m, dv_m = madv_m(u, v, scheme="energy")
        du_u, dv_u = madv_u(u, v, scheme="energy")
        np.testing.assert_allclose(du_m, du_u, atol=1e-12)
        np.testing.assert_allclose(dv_m, dv_u, atol=1e-12)

    def test_dry_face_zero(self, grid):
        """A dry U-face and a dry V-face get zero tendency."""
        m = np.ones((grid.Ny, grid.Nx), dtype=bool)
        m[4, 5] = False
        madv = SphericalMomentumAdvection2D(grid=grid, mask=Mask2D.from_mask(m))
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        du, dv = madv(u, v, scheme="energy")
        # mask.u[j,i] is False wherever h[j,i] or h[j,i+1] is dry.
        mu = Mask2D.from_mask(m).u
        mv = Mask2D.from_mask(m).v
        # tendency must vanish exactly where the respective mask is False
        assert jnp.all(du[~mu.astype(bool)] == 0.0)
        assert jnp.all(dv[~mv.astype(bool)] == 0.0)


# ======================================================================
# SphericalMomentumAdvection3D
# ======================================================================


class TestSphericalMomentumAdvection3D:
    @pytest.fixture
    def madv(self, grid3d):
        return SphericalMomentumAdvection3D(grid=grid3d)

    def test_output_shapes(self, madv, grid3d):
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)
        du, dv = madv(u, v)
        assert du.shape == shape
        assert dv.shape == shape

    def test_zero_velocity_zero_tendency(self, madv, grid3d):
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)
        du, dv = madv(u, v, scheme="al")
        np.testing.assert_allclose(du, 0.0, atol=1e-12)
        np.testing.assert_allclose(dv, 0.0, atol=1e-12)

    def test_z_ghost_slices_zero(self, madv, grid3d):
        key = jax.random.PRNGKey(8)
        k1, k2 = jax.random.split(key)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        u = jax.random.normal(k1, shape)
        v = jax.random.normal(k2, shape)
        du, dv = madv(u, v, scheme="energy")
        np.testing.assert_allclose(du[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(du[-1], 0.0, atol=1e-10)
        np.testing.assert_allclose(dv[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(dv[-1], 0.0, atol=1e-10)

    def test_matches_2d_per_level(self, madv, grid3d):
        madv2 = SphericalMomentumAdvection2D(grid=grid3d.horizontal_grid())
        key = jax.random.PRNGKey(9)
        k1, k2 = jax.random.split(key)
        u2 = jax.random.normal(k1, (grid3d.Ny, grid3d.Nx))
        v2 = jax.random.normal(k2, (grid3d.Ny, grid3d.Nx))
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        u3 = jnp.broadcast_to(u2, shape)
        v3 = jnp.broadcast_to(v2, shape)
        du2, dv2 = madv2(u2, v2, scheme="al")
        du3, dv3 = madv(u3, v3, scheme="al")
        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(du3[k], du2, atol=1e-12)
            np.testing.assert_allclose(dv3[k], dv2, atol=1e-12)

    def test_masked_all_dry_zero(self, grid3d):
        all_dry = Mask3D.from_mask(
            np.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        madv = SphericalMomentumAdvection3D(grid=grid3d, mask=all_dry)
        key = jax.random.PRNGKey(10)
        k1, k2 = jax.random.split(key)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        u = jax.random.normal(k1, shape)
        v = jax.random.normal(k2, shape)
        du, dv = madv(u, v, scheme="energy")
        np.testing.assert_allclose(du, 0.0, atol=1e-12)
        np.testing.assert_allclose(dv, 0.0, atol=1e-12)

    def test_jit_grad(self, madv, grid3d):
        key = jax.random.PRNGKey(11)
        k1, k2 = jax.random.split(key)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        u = jax.random.normal(k1, shape)
        v = jax.random.normal(k2, shape)
        du, dv = jax.jit(lambda a, b: madv(a, b, scheme="energy"))(u, v)
        assert jnp.all(jnp.isfinite(du)) and jnp.all(jnp.isfinite(dv))

        def loss(a, b):
            du_, dv_ = madv(a, b, scheme="al")
            return (du_**2 + dv_**2).sum()

        gu, gv = jax.grad(loss, argnums=(0, 1))(u, v)
        assert jnp.all(jnp.isfinite(gu)) and jnp.all(jnp.isfinite(gv))
