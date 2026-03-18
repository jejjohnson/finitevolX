"""Tests for spherical compound operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.spherical_grid import (
    SphericalArakawaCGrid2D,
    SphericalArakawaCGrid3D,
)
from finitevolx._src.operators.geographic import (
    curl_sphere,
    divergence_sphere,
    geostrophic_velocity_sphere as geo_vel_old,
    laplacian_sphere,
    potential_vorticity_sphere,
)
from finitevolx._src.operators.spherical_compound import (
    SphericalDivergence2D,
    SphericalDivergence3D,
    SphericalLaplacian2D,
    SphericalLaplacian3D,
    SphericalVorticity2D,
    SphericalVorticity3D,
    geostrophic_velocity_sphere as geo_vel_new,
)
from finitevolx._src.operators.vorticity import Vorticity2D

R = 1.0
NX_INT, NY_INT = 10, 8


@pytest.fixture
def grid():
    return SphericalArakawaCGrid2D.from_interior(
        nx_interior=NX_INT,
        ny_interior=NY_INT,
        lon_range=(0.0, 360.0),
        lat_range=(-80.0, 80.0),
        R=R,
    )


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


# ======================================================================
# SphericalDivergence2D
# ======================================================================


class TestSphericalDivergence2D:
    @pytest.fixture
    def div_op(self, grid):
        return SphericalDivergence2D(grid=grid)

    def test_output_shape(self, div_op, grid):
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        assert div_op(u, v).shape == (grid.Ny, grid.Nx)

    def test_uniform_u_zero_v(self, div_op, grid):
        u = 3.0 * jnp.ones((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        result = div_op(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_cross_validate(self, div_op, grid, rng):
        k1, k2 = jax.random.split(rng)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        new = div_op(u, v)
        old = divergence_sphere(
            u, v, grid.cos_lat_T, grid.cos_lat_V, grid.dlon, grid.dlat, grid.R
        )
        np.testing.assert_allclose(new, old, atol=1e-6)


# ======================================================================
# SphericalVorticity2D
# ======================================================================


class TestSphericalVorticity2D:
    @pytest.fixture
    def vort_op(self, grid):
        return SphericalVorticity2D(grid=grid)

    def test_output_shape(self, vort_op, grid):
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        assert vort_op.relative_vorticity(u, v).shape == (grid.Ny, grid.Nx)

    def test_zero_velocity(self, vort_op, grid):
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        result = vort_op.relative_vorticity(u, v)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_cross_validate_curl(self, vort_op, grid, rng):
        k1, k2 = jax.random.split(rng)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        new = vort_op.relative_vorticity(u, v)
        old = curl_sphere(
            u, v, grid.cos_lat_U, grid.cos_lat_X, grid.dlon, grid.dlat, grid.R
        )
        np.testing.assert_allclose(new, old, atol=1e-6)

    def test_cross_validate_pv(self, vort_op, grid, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        h = 1.0 + 0.1 * jax.random.normal(k3, (grid.Ny, grid.Nx))
        f = 1e-4 * jnp.ones((grid.Ny, grid.Nx))
        new = vort_op.potential_vorticity(u, v, h, f)
        old = potential_vorticity_sphere(
            u, v, h, f, grid.cos_lat_U, grid.cos_lat_X, grid.dlon, grid.dlat, grid.R
        )
        np.testing.assert_allclose(new, old, atol=1e-5)

    def test_pv_flux_matches_cartesian(self, vort_op, grid, rng):
        """PV flux methods are coordinate-independent — should match Cartesian."""
        cart_vort = Vorticity2D(grid=grid)
        k1, k2, k3 = jax.random.split(rng, 3)
        q = jax.random.normal(k1, (grid.Ny, grid.Nx))
        u = jax.random.normal(k2, (grid.Ny, grid.Nx))
        v = jax.random.normal(k3, (grid.Ny, grid.Nx))

        # Energy-conserving
        qu_s, qv_s = vort_op.pv_flux_energy_conserving(q, u, v)
        qu_c, qv_c = cart_vort.pv_flux_energy_conserving(q, u, v)
        np.testing.assert_allclose(qu_s, qu_c, atol=1e-10)
        np.testing.assert_allclose(qv_s, qv_c, atol=1e-10)

        # Enstrophy-conserving
        qu_s, qv_s = vort_op.pv_flux_enstrophy_conserving(q, u, v)
        qu_c, qv_c = cart_vort.pv_flux_enstrophy_conserving(q, u, v)
        np.testing.assert_allclose(qu_s, qu_c, atol=1e-10)
        np.testing.assert_allclose(qv_s, qv_c, atol=1e-10)

        # Arakawa-Lamb
        qu_s, qv_s = vort_op.pv_flux_arakawa_lamb(q, u, v)
        qu_c, qv_c = cart_vort.pv_flux_arakawa_lamb(q, u, v)
        np.testing.assert_allclose(qu_s, qu_c, atol=1e-10)
        np.testing.assert_allclose(qv_s, qv_c, atol=1e-10)


# ======================================================================
# SphericalLaplacian2D
# ======================================================================


class TestSphericalLaplacian2D:
    @pytest.fixture
    def lap_op(self, grid):
        return SphericalLaplacian2D(grid=grid)

    def test_output_shape(self, lap_op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        assert lap_op(h).shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero(self, lap_op, grid):
        h = 7.0 * jnp.ones((grid.Ny, grid.Nx))
        result = lap_op(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_cross_validate(self, lap_op, grid, rng):
        h = jax.random.normal(rng, (grid.Ny, grid.Nx))
        new = lap_op(h)
        old = laplacian_sphere(h, grid.cos_lat_T, grid.dlon, grid.dlat, grid.R)
        np.testing.assert_allclose(new, old, atol=1e-5)


# ======================================================================
# Geostrophic velocity
# ======================================================================


class TestGeostrophicVelocity:
    def test_output_shapes(self, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        f = 1e-4 * jnp.ones((grid.Ny, grid.Nx))
        u_g, v_g = geo_vel_new(h, f, grid)
        assert u_g.shape == (grid.Ny, grid.Nx)
        assert v_g.shape == (grid.Ny, grid.Nx)

    def test_constant_h_zero(self, grid):
        h = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        f = 1e-4 * jnp.ones((grid.Ny, grid.Nx))
        u_g, v_g = geo_vel_new(h, f, grid)
        np.testing.assert_allclose(u_g[1:-1, 1:-1], 0.0, atol=1e-10)
        np.testing.assert_allclose(v_g[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_cross_validate(self, grid, rng):
        h = jax.random.normal(rng, (grid.Ny, grid.Nx))
        f = 1e-4 * jnp.ones((grid.Ny, grid.Nx))
        u_new, v_new = geo_vel_new(h, f, grid)
        u_old, v_old = geo_vel_old(h, f, grid.cos_lat_T, grid.dlon, grid.dlat, R=grid.R)
        np.testing.assert_allclose(u_new, u_old, atol=1e-6)
        np.testing.assert_allclose(v_new, v_old, atol=1e-6)


# ======================================================================
# 3D operators
# ======================================================================


class TestSpherical3DCompound:
    @pytest.fixture
    def grid3d(self):
        return SphericalArakawaCGrid3D.from_interior(
            nx_interior=NX_INT,
            ny_interior=NY_INT,
            nz_interior=4,
            lon_range=(0.0, 360.0),
            lat_range=(-80.0, 80.0),
            Lz=100.0,
            R=R,
        )

    def test_div3d_shape(self, grid3d):
        div3d = SphericalDivergence3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert div3d(u, v).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_vort3d_shape(self, grid3d):
        vort3d = SphericalVorticity3D(grid=grid3d)
        u = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert vort3d.relative_vorticity(u, v).shape == (
            grid3d.Nz,
            grid3d.Ny,
            grid3d.Nx,
        )

    def test_lap3d_shape(self, grid3d):
        lap3d = SphericalLaplacian3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert lap3d(h).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_lap3d_constant_zero(self, grid3d):
        lap3d = SphericalLaplacian3D(grid=grid3d)
        h = 7.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = lap3d(h)
        np.testing.assert_allclose(result[:, 1:-1, 1:-1], 0.0, atol=1e-10)


# ======================================================================
# JIT + grad
# ======================================================================


class TestJitGradCompound:
    def test_jit_divergence(self, grid, rng):
        div_op = SphericalDivergence2D(grid=grid)
        k1, k2 = jax.random.split(rng)
        u = jax.random.normal(k1, (grid.Ny, grid.Nx))
        v = jax.random.normal(k2, (grid.Ny, grid.Nx))
        result = jax.jit(lambda a, b: div_op(a, b))(u, v)  # noqa: PLW0108
        assert result.shape == (grid.Ny, grid.Nx)

    def test_grad_laplacian(self, grid, rng):
        lap_op = SphericalLaplacian2D(grid=grid)
        h = jax.random.normal(rng, (grid.Ny, grid.Nx))
        g = jax.grad(lambda x: lap_op(x).sum())(h)
        assert g.shape == h.shape
