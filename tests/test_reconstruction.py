"""Tests for Reconstruction1D, Reconstruction2D, Reconstruction3D."""
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)


@pytest.fixture
def grid1d():
    return ArakawaCGrid1D.from_interior(8, 1.0)


@pytest.fixture
def grid2d():
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return ArakawaCGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


class TestReconstruction1D:
    def test_naive_positive_flow(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.naive_x(h, u)
        np.testing.assert_allclose(result[1:-1], 1.0)

    def test_upwind1_positive_takes_upstream(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        # h[i] = i  =>  upwind with u>0 picks h[i]
        h = jnp.arange(grid1d.Nx, dtype=float)
        u = jnp.ones(grid1d.Nx)
        result = recon.upwind1_x(h, u)
        # fe[i+1/2] = h[i] * u = i * 1 = i
        np.testing.assert_allclose(result[1:-1], h[1:-1], rtol=1e-6)

    def test_upwind1_negative_takes_downstream(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.arange(grid1d.Nx, dtype=float)
        u = -jnp.ones(grid1d.Nx)
        result = recon.upwind1_x(h, u)
        # fe[i+1/2] = h[i+1] * u = (i+1) * (-1)
        np.testing.assert_allclose(result[1:-1], -h[2:], rtol=1e-6)

    def test_upwind2_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.upwind2_x(h, u).shape == (grid1d.Nx,)

    def test_upwind3_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.upwind3_x(h, u).shape == (grid1d.Nx,)

    def test_ghost_zero(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.naive_x(h, u)
        assert result[0] == 0.0
        assert result[-1] == 0.0


class TestReconstruction2D:
    def test_naive_x_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.naive_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0)

    def test_naive_y_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.naive_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 3.0)

    def test_upwind1_x_positive(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        # uniform h, positive u => flux = h * u
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.upwind1_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 8.0)

    def test_upwind1_y_negative(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.upwind1_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], -1.0)

    def test_upwind2_x_shape(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert recon.upwind2_x(h, u).shape == (grid2d.Ny, grid2d.Nx)

    def test_upwind3_x_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.upwind3_x(h, u)
        # constant h => all stencils give h * u = 5
        np.testing.assert_allclose(result[1:-1, 1:-1], 5.0, rtol=1e-5)

    def test_ghost_zero(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.naive_x(h, u)
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)


class TestReconstruction3D:
    def test_naive_x_shape(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert recon.naive_x(h, u).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_upwind1_x_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 3.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.upwind1_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 3.0)

    def test_upwind1_y_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 2.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.upwind1_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 2.0)
