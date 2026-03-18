"""Tests for SphericalDifference2D and SphericalDifference3D."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.spherical_grid import (
    SphericalArakawaCGrid2D,
    SphericalArakawaCGrid3D,
)
from finitevolx._src.operators.spherical_difference import (
    SphericalDifference2D,
    SphericalDifference3D,
)

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
def diff(grid):
    return SphericalDifference2D(grid=grid)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


# ======================================================================
# Forward differences
# ======================================================================


class TestDiffLonTtoU:
    def test_output_shape(self, diff, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        assert diff.diff_lon_T_to_U(h).shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero(self, diff, grid):
        h = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.diff_lon_T_to_U(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_ghost_ring_zero(self, diff, grid):
        h = jax.random.normal(jax.random.PRNGKey(0), (grid.Ny, grid.Nx))
        result = diff.diff_lon_T_to_U(h)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)

    def test_linear_longitude(self, diff, grid):
        """h = lon => dh/dx = 1/(R cos(lat)) at U-points."""
        h = grid.lon_T
        result = diff.diff_lon_T_to_U(h)
        expected = 1.0 / (R * grid.cos_lat_U[1:-1, 1:-1])
        np.testing.assert_allclose(result[1:-1, 1:-1], expected, rtol=1e-5)


class TestDiffLatTtoV:
    def test_constant_field_zero(self, diff, grid):
        h = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.diff_lat_T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_latitude(self, diff, grid):
        h = grid.lat_T
        result = diff.diff_lat_T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 1.0 / R, rtol=1e-5)


class TestDiffLonVtoX:
    def test_output_shape(self, diff, grid):
        v = jnp.ones((grid.Ny, grid.Nx))
        assert diff.diff_lon_V_to_X(v).shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero(self, diff, grid):
        v = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.diff_lon_V_to_X(v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_longitude(self, diff, grid):
        """v = lon => dv/dx = 1/(R cos(lat)) at X-points."""
        v = grid.lon_T
        result = diff.diff_lon_V_to_X(v)
        expected = 1.0 / (R * grid.cos_lat_X[1:-1, 1:-1])
        np.testing.assert_allclose(result[1:-1, 1:-1], expected, rtol=1e-5)


class TestDiffLatUtoX:
    def test_constant_field_zero(self, diff, grid):
        u = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.diff_lat_U_to_X(u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_latitude(self, diff, grid):
        u = grid.lat_T
        result = diff.diff_lat_U_to_X(u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 1.0 / R, rtol=1e-5)


# ======================================================================
# Backward differences
# ======================================================================


class TestDiffLonUtoT:
    def test_constant_field_zero(self, diff, grid):
        u = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.diff_lon_U_to_T(u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_longitude(self, diff, grid):
        """u = lon => du/dx = 1/(R cos(lat)) at T-points."""
        u = grid.lon_T
        result = diff.diff_lon_U_to_T(u)
        expected = 1.0 / (R * grid.cos_lat_T[1:-1, 1:-1])
        np.testing.assert_allclose(result[1:-1, 1:-1], expected, rtol=1e-5)


class TestDiffLatVtoT:
    def test_constant_field_zero(self, diff, grid):
        v = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.diff_lat_V_to_T(v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_latitude(self, diff, grid):
        """v = lat => dv/dy = 1/R at T-points."""
        v = grid.lat_T
        result = diff.diff_lat_V_to_T(v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 1.0 / R, rtol=1e-5)


# ======================================================================
# Second-order differences
# ======================================================================


class TestDiff2Lon:
    def test_output_shape(self, diff, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        assert diff.diff2_lon(h).shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero(self, diff, grid):
        h = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.diff2_lon(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_field_near_zero(self, diff, grid):
        """d²(lon)/dx² ≈ 0 for a linear zonal field."""
        h = grid.lon_T
        result = diff.diff2_lon(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-4)


class TestLaplacianMerid:
    def test_output_shape(self, diff, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        assert diff.laplacian_merid(h).shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero(self, diff, grid):
        h = 5.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.laplacian_merid(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_latitude_nonzero(self, diff, grid):
        """Meridional Laplacian of h=lat is non-trivial (not zero) due to metric."""
        h = grid.lat_T
        result = diff.laplacian_merid(h)
        assert not jnp.allclose(result[2:-2, 2:-2], 0.0, atol=1e-10)


# ======================================================================
# JIT and grad compatibility
# ======================================================================


class TestJitGrad:
    def test_jit(self, diff, grid, rng):
        h = jax.random.normal(rng, (grid.Ny, grid.Nx))
        result = jax.jit(lambda x: diff.diff_lon_T_to_U(x))(h)  # noqa: PLW0108
        assert result.shape == (grid.Ny, grid.Nx)

    def test_grad(self, diff, grid, rng):
        h = jax.random.normal(rng, (grid.Ny, grid.Nx))
        grad_fn = jax.grad(lambda x: diff.diff_lon_T_to_U(x).sum())
        g = grad_fn(h)
        assert g.shape == h.shape


# ======================================================================
# 3D operator
# ======================================================================


class TestSphericalDifference3D:
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

    @pytest.fixture
    def diff3d(self, grid3d):
        return SphericalDifference3D(grid=grid3d)

    def test_output_shape(self, diff3d, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff3d.diff_lon_T_to_U(h)
        assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_z_ghost_slices_zero(self, diff3d, grid3d):
        key = jax.random.PRNGKey(7)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff3d.diff_lon_T_to_U(h)
        np.testing.assert_allclose(result[0, :, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :, :], 0.0, atol=1e-10)

    def test_matches_per_level(self, diff3d, grid3d):
        key = jax.random.PRNGKey(0)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result_3d = diff3d.diff_lon_T_to_U(h)
        diff2d = diff3d._diff2d
        # Interior z-levels should match 2D; ghost slices (k=0, k=-1) are zeroed.
        for k in range(1, grid3d.Nz - 1):
            result_2d = diff2d.diff_lon_T_to_U(h[k])
            np.testing.assert_allclose(result_3d[k], result_2d, atol=1e-10)

    def test_all_methods_run(self, diff3d, grid3d):
        key = jax.random.PRNGKey(1)
        h = jax.random.normal(key, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for method_name in [
            "diff_lon_T_to_U",
            "diff_lat_T_to_V",
            "diff_lon_V_to_X",
            "diff_lat_U_to_X",
            "diff_lon_U_to_T",
            "diff_lat_V_to_T",
            "diff2_lon",
            "laplacian_merid",
        ]:
            method = getattr(diff3d, method_name)
            result = method(h)
            assert result.shape == h.shape, f"{method_name} shape mismatch"
