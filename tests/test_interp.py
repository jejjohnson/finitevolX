"""Tests for Interpolation1D, Interpolation2D, Interpolation3D."""
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.interpolation import Interpolation1D, Interpolation2D, Interpolation3D


@pytest.fixture
def grid1d():
    return ArakawaCGrid1D.from_interior(8, 1.0)


@pytest.fixture
def grid2d():
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return ArakawaCGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


class TestInterpolation1D:
    def test_T_to_U_constant(self, grid1d):
        interp = Interpolation1D(grid=grid1d)
        h = 3.0 * jnp.ones(grid1d.Nx)
        result = interp.T_to_U(h)
        np.testing.assert_allclose(result[1:-1], 3.0)

    def test_U_to_T_constant(self, grid1d):
        interp = Interpolation1D(grid=grid1d)
        u = 5.0 * jnp.ones(grid1d.Nx)
        result = interp.U_to_T(u)
        np.testing.assert_allclose(result[1:-1], 5.0)

    def test_T_to_U_linear(self, grid1d):
        interp = Interpolation1D(grid=grid1d)
        # h[i] = i  =>  h_on_u[i+1/2] = i + 0.5
        h = jnp.arange(grid1d.Nx, dtype=float)
        result = interp.T_to_U(h)
        expected = jnp.arange(grid1d.Nx, dtype=float) + 0.5
        np.testing.assert_allclose(result[1:-1], expected[1:-1], rtol=1e-6)

    def test_ghost_is_zero(self, grid1d):
        interp = Interpolation1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        result = interp.T_to_U(h)
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[-1], 0.0)


class TestInterpolation2D:
    def test_T_to_U_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.T_to_U(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0)

    def test_T_to_V_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 4.0)

    def test_T_to_X_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = 7.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.T_to_X(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 7.0)

    def test_X_to_U_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        q = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.X_to_U(q)
        np.testing.assert_allclose(result[1:-1, 1:-1], 3.0)

    def test_X_to_V_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        q = 6.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.X_to_V(q)
        np.testing.assert_allclose(result[1:-1, 1:-1], 6.0)

    def test_U_to_T_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        u = 9.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.U_to_T(u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 9.0)

    def test_V_to_T_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        v = 11.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.V_to_T(v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 11.0)

    def test_X_to_T_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        q = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.X_to_T(q)
        np.testing.assert_allclose(result[1:-1, 1:-1], 5.0)

    def test_U_to_X_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        u = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.U_to_X(u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0)

    def test_V_to_X_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        v = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.V_to_X(v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 4.0)

    def test_U_to_V_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        u = 8.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.U_to_V(u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 8.0)

    def test_V_to_U_constant(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        v = 10.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.V_to_U(v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 10.0)

    def test_output_shapes(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        for method in [
            interp.T_to_U, interp.T_to_V, interp.T_to_X,
            interp.U_to_T, interp.V_to_T, interp.X_to_T,
            interp.U_to_X, interp.V_to_X, interp.X_to_U, interp.X_to_V,
            interp.U_to_V, interp.V_to_U,
        ]:
            assert method(h).shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_zero(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.T_to_U(h)
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)


class TestInterpolation3D:
    def test_T_to_U_constant(self, grid3d):
        interp = Interpolation3D(grid=grid3d)
        h = 3.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = interp.T_to_U(h)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 3.0)

    def test_T_to_V_constant(self, grid3d):
        interp = Interpolation3D(grid=grid3d)
        h = 5.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = interp.T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 5.0)

    def test_output_shape(self, grid3d):
        interp = Interpolation3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert interp.T_to_U(h).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert interp.T_to_V(h).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert interp.U_to_T(h).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert interp.V_to_T(h).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
