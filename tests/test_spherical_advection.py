"""Tests for SphericalAdvection2D and SphericalAdvection3D."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.advection import Advection2D, Advection3D
from finitevolx._src.advection.spherical_advection import (
    SphericalAdvection2D,
    SphericalAdvection3D,
)
from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
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


# ======================================================================
# SphericalAdvection2D
# ======================================================================


class TestSphericalAdvection2D:
    @pytest.fixture
    def op(self, grid):
        return SphericalAdvection2D(grid=grid)

    def test_output_shape(self, op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        out = op(h, u, v)
        assert out.shape == (grid.Ny, grid.Nx)

    def test_zero_velocity_zero_tendency(self, op, grid):
        key = jax.random.PRNGKey(0)
        h = jax.random.normal(key, (grid.Ny, grid.Nx))
        u = jnp.zeros_like(h)
        v = jnp.zeros_like(h)
        out = op(h, u, v, method="upwind1")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "method",
        [
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
        ],
    )
    def test_methods_produce_finite_output(self, op, grid, method):
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = op(h, u, v, method=method)
        assert jnp.all(jnp.isfinite(out))

    def test_ghost_ring_zero(self, op, grid):
        key = jax.random.PRNGKey(2)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = op(h, u, v, method="upwind1")
        np.testing.assert_allclose(out[0, :], 0.0)
        np.testing.assert_allclose(out[-1, :], 0.0)
        np.testing.assert_allclose(out[:, 0], 0.0)
        np.testing.assert_allclose(out[:, -1], 0.0)

    def test_matches_cartesian_at_narrow_equatorial_band(self):
        """Spherical advection at the equator (cos≈1) should match
        Cartesian advection with dx=R·dlon, dy=R·dlat."""
        nx, ny = 24, 6
        lon_range = (0.0, 30.0)
        lat_range = (-0.5, 0.5)
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
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (sphere.Ny, sphere.Nx))
        u = jax.random.normal(k2, (sphere.Ny, sphere.Nx))
        v = jax.random.normal(k3, (sphere.Ny, sphere.Nx))

        sph_op = SphericalAdvection2D(grid=sphere)
        cart_op = Advection2D(grid=cart)

        for method in ("upwind1", "weno3", "weno5"):
            t_s = sph_op(h, u, v, method=method)
            t_c = cart_op(h, u, v, method=method)
            # Across 1-degree latitude, cos varies by O(1e-4).
            np.testing.assert_allclose(
                t_s[2:-2, 2:-2],
                t_c[2:-2, 2:-2],
                rtol=5e-3,
                atol=1e-3,
            )

    def test_jit(self, op, grid):
        key = jax.random.PRNGKey(4)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = jax.jit(lambda a, b, c: op(a, b, c, method="weno5"))(h, u, v)
        assert out.shape == h.shape
        assert jnp.all(jnp.isfinite(out))

    def test_grad(self, op, grid):
        key = jax.random.PRNGKey(5)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))

        def loss(h_, u_, v_):
            return op(h_, u_, v_, method="upwind1").sum()

        gh, gu, gv = jax.grad(loss, argnums=(0, 1, 2))(h, u, v)
        for g in (gh, gu, gv):
            assert g.shape == h.shape
            assert jnp.all(jnp.isfinite(g))

    def test_uniform_h_positive_u_zero_v(self, grid):
        """Uniform h with uniform u>0 → -div(h·u·cosφ) dependence.

        For h constant and v=0:
            dh/dt = -(1/(R·cosφ))·∂(h·u)/∂λ
        If additionally u is uniform in λ, this term is zero. Verify.
        """
        op = SphericalAdvection2D(grid=grid)
        h = 2.0 * jnp.ones((grid.Ny, grid.Nx))
        u = 1.5 * jnp.ones((grid.Ny, grid.Nx))
        v = jnp.zeros_like(h)
        out = op(h, u, v, method="upwind1")
        np.testing.assert_allclose(out[2:-2, 2:-2], 0.0, atol=1e-10)


# ======================================================================
# Masking
# ======================================================================


class TestSphericalAdvection2DMasked:
    def test_all_dry_mask_zero(self, grid):
        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        op = SphericalAdvection2D(grid=grid, mask=all_dry)
        key = jax.random.PRNGKey(6)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = op(h, u, v, method="weno3")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_all_wet_matches_unmasked_upwind1(self, grid):
        all_wet = Mask2D.from_mask(np.ones((grid.Ny, grid.Nx), dtype=bool))
        op_m = SphericalAdvection2D(grid=grid, mask=all_wet)
        op_u = SphericalAdvection2D(grid=grid)
        key = jax.random.PRNGKey(7)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        # Non-dispatchable method — should match bit-for-bit.
        np.testing.assert_allclose(
            op_m(h, u, v, method="upwind1"),
            op_u(h, u, v, method="upwind1"),
            atol=1e-12,
        )

    def test_dry_cell_tendency_zero(self, grid):
        mask = np.ones((grid.Ny, grid.Nx), dtype=bool)
        mask[4, 6] = False
        op = SphericalAdvection2D(grid=grid, mask=Mask2D.from_mask(mask))
        key = jax.random.PRNGKey(8)
        k1, k2, k3 = jax.random.split(key, 3)
        h = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))
        out = op(h, u, v, method="weno3")
        assert float(out[4, 6]) == 0.0


# ======================================================================
# SphericalAdvection3D
# ======================================================================


class TestSphericalAdvection3D:
    @pytest.fixture
    def op(self, grid3d):
        return SphericalAdvection3D(grid=grid3d)

    def test_output_shape(self, op, grid3d):
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jnp.ones(shape)
        u = jnp.zeros(shape)
        v = jnp.zeros(shape)
        out = op(h, u, v)
        assert out.shape == shape

    def test_zero_velocity_zero_tendency(self, op, grid3d):
        key = jax.random.PRNGKey(9)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.zeros_like(h)
        v = jnp.zeros_like(h)
        out = op(h, u, v, method="upwind1")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "method",
        [
            "naive",
            "upwind1",
            "weno3",
            "weno5",
            "weno7",
            "weno9",
            "minmod",
            "van_leer",
            "superbee",
            "mc",
        ],
    )
    def test_methods_produce_finite_output(self, op, grid3d, method):
        key = jax.random.PRNGKey(15)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)
        out = op(h, u, v, method=method)
        assert jnp.all(jnp.isfinite(out))

    def test_z_ghost_slices_zero(self, op, grid3d):
        key = jax.random.PRNGKey(10)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)
        out = op(h, u, v, method="weno3")
        np.testing.assert_allclose(out[0, :, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(out[-1, :, :], 0.0, atol=1e-10)

    def test_matches_2d_per_level(self, op, grid3d):
        """On a z-broadcast field, the 3-D spherical advection equals
        the 2-D spherical advection at each interior level."""
        op2d = SphericalAdvection2D(grid=grid3d.horizontal_grid())
        key = jax.random.PRNGKey(11)
        k1, k2, k3 = jax.random.split(key, 3)
        h_2d = jax.random.normal(k1, (grid3d.Ny, grid3d.Nx))
        u_2d = jax.random.normal(k2, (grid3d.Ny, grid3d.Nx))
        v_2d = jax.random.normal(k3, (grid3d.Ny, grid3d.Nx))
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h_3d = jnp.broadcast_to(h_2d, shape)
        u_3d = jnp.broadcast_to(u_2d, shape)
        v_3d = jnp.broadcast_to(v_2d, shape)

        t_2d = op2d(h_2d, u_2d, v_2d, method="weno3")
        t_3d = op(h_3d, u_3d, v_3d, method="weno3")

        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(t_3d[k], t_2d, atol=1e-12)

    def test_matches_cartesian_at_narrow_equatorial_band(self):
        """Narrow lat band around equator → spherical 3D ≈ Cartesian 3D."""
        nx, ny, nz = 16, 6, 3
        sphere = SphericalGrid3D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            nz_interior=nz,
            lon_range=(0.0, 20.0),
            lat_range=(-0.5, 0.5),
            Lz=100.0,
            R=R,
        )
        cart = CartesianGrid3D.from_interior(
            nx_interior=nx,
            ny_interior=ny,
            nz_interior=nz,
            Lx=sphere.Lx,
            Ly=sphere.Ly,
            Lz=100.0,
        )
        key = jax.random.PRNGKey(12)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (sphere.Nz, sphere.Ny, sphere.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)

        op_s = SphericalAdvection3D(grid=sphere)
        op_c = Advection3D(grid=cart)

        for method in ("upwind1", "weno3"):
            t_s = op_s(h, u, v, method=method)
            t_c = op_c(h, u, v, method=method)
            np.testing.assert_allclose(
                t_s[1:-1, 2:-2, 2:-2],
                t_c[1:-1, 2:-2, 2:-2],
                rtol=5e-3,
                atol=1e-3,
            )

    def test_masked_all_dry_zero(self, grid3d):
        all_dry = Mask3D.from_mask(
            np.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx), dtype=bool)
        )
        op = SphericalAdvection3D(grid=grid3d, mask=all_dry)
        key = jax.random.PRNGKey(13)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)
        out = op(h, u, v, method="weno3")
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_jit_grad(self, op, grid3d):
        key = jax.random.PRNGKey(14)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jax.random.normal(k1, shape)
        u = jax.random.normal(k2, shape)
        v = jax.random.normal(k3, shape)
        jitted = jax.jit(lambda a, b, c: op(a, b, c, method="weno3"))
        out = jitted(h, u, v)
        assert out.shape == shape
        g = jax.grad(lambda a: op(a, u, v, method="upwind1").sum())(h)
        assert g.shape == shape
        assert jnp.all(jnp.isfinite(g))
