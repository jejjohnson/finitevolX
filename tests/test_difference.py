"""Tests for Difference1D, Difference2D, Difference3D."""
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.difference import Difference1D, Difference2D, Difference3D
from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D


@pytest.fixture
def grid1d():
    return ArakawaCGrid1D.from_interior(8, 1.0)


@pytest.fixture
def grid2d():
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return ArakawaCGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


class TestDifference1D:
    def test_diff_x_T_to_U_linear(self, grid1d):
        diff = Difference1D(grid=grid1d)
        # h = c * x  =>  dh/dx = c everywhere
        c = 2.0
        h = c * jnp.arange(grid1d.Nx, dtype=float) * grid1d.dx
        result = diff.diff_x_T_to_U(h)
        # interior should be ~c
        np.testing.assert_allclose(result[1:-1], c, rtol=1e-5)
        # ghost cells should be zero
        assert result[0] == 0.0
        assert result[-1] == 0.0

    def test_diff_x_U_to_T_constant(self, grid1d):
        diff = Difference1D(grid=grid1d)
        u = jnp.ones(grid1d.Nx)
        result = diff.diff_x_U_to_T(u)
        # constant field: backward diff = 0 at interior
        np.testing.assert_allclose(result[1:-1], 0.0, atol=1e-10)

    def test_laplacian_quadratic(self, grid1d):
        diff = Difference1D(grid=grid1d)
        # h = x^2  =>  d2h/dx2 = 2
        x = jnp.arange(grid1d.Nx, dtype=float) * grid1d.dx
        h = x**2
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1], 2.0, rtol=1e-5)

    def test_output_shape(self, grid1d):
        diff = Difference1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        assert diff.diff_x_T_to_U(h).shape == (grid1d.Nx,)
        assert diff.diff_x_U_to_T(h).shape == (grid1d.Nx,)
        assert diff.laplacian(h).shape == (grid1d.Nx,)


class TestDifference2D:
    def test_diff_x_T_to_U_linear(self, grid2d):
        diff = Difference2D(grid=grid2d)
        # h = c * x  =>  dh/dx = c
        c = 3.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = diff.diff_x_T_to_U(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_diff_y_T_to_V_linear(self, grid2d):
        diff = Difference2D(grid=grid2d)
        c = 2.5
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = jnp.broadcast_to(c * y[:, None], (grid2d.Ny, grid2d.Nx))
        result = diff.diff_y_T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_divergence_constant_flow(self, grid2d):
        diff = Difference2D(grid=grid2d)
        # uniform flow: divergence = 0
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = diff.divergence(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_curl_irrotational(self, grid2d):
        diff = Difference2D(grid=grid2d)
        # u = constant, v = constant => curl = 0
        u = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = diff.curl(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_laplacian_quadratic(self, grid2d):
        diff = Difference2D(grid=grid2d)
        # h = x^2 + y^2 => laplacian = 2 + 2 = 4
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = x[None, :] ** 2 + y[:, None] ** 2
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 4.0, rtol=1e-5)

    def test_output_shape(self, grid2d):
        diff = Difference2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert diff.diff_x_T_to_U(h).shape == (grid2d.Ny, grid2d.Nx)
        assert diff.laplacian(h).shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_is_zero(self, grid2d):
        diff = Difference2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = diff.diff_x_T_to_U(h)
        # boundary rows/cols must be zero
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, -1], 0.0)


class TestDifference3D:
    def test_diff_x_T_to_U_shape(self, grid3d):
        diff = Difference3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff.diff_x_T_to_U(h)
        assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_divergence_constant_flow(self, grid3d):
        diff = Difference3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff.divergence(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_laplacian_quadratic(self, grid3d):
        diff = Difference3D(grid=grid3d)
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        y = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        h2d = x[None, :] ** 2 + y[:, None] ** 2
        h = jnp.broadcast_to(h2d, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 4.0, rtol=1e-5)
