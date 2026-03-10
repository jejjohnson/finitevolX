"""Tests for Advection1D, Advection2D, Advection3D."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection import Advection1D, Advection2D, Advection3D
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


class TestAdvection1D:
    def test_output_shape(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert adv(h, u).shape == (grid1d.Nx,)

    def test_constant_field_zero_tendency(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = adv(h, u, method="upwind1")
        # strictly interior (away from ghost edges): flux difference = 0
        np.testing.assert_allclose(result[2:-2], 0.0, atol=1e-10)

    def test_all_methods_run(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        for method in ["naive", "upwind1", "upwind2", "upwind3"]:
            result = adv(h, u, method=method)
            assert result.shape == (grid1d.Nx,)

    def test_tvd_methods_run(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, method=method)
            assert result.shape == (grid1d.Nx,)

    def test_tvd_constant_zero_tendency(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, method=method)
            np.testing.assert_allclose(result[2:-2], 0.0, atol=1e-10)

    def test_unknown_method_raises(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        with pytest.raises(ValueError, match="Unknown method"):
            adv(h, u, method="invalid")

    def test_ghost_zero(self, grid1d):
        adv = Advection1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = adv(h, u)
        # Ghost ring and boundary layer stay zero
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[1], 0.0)
        np.testing.assert_allclose(result[-2], 0.0)
        np.testing.assert_allclose(result[-1], 0.0)


class TestAdvection2D:
    def test_output_shape(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert adv(h, u, v).shape == (grid2d.Ny, grid2d.Nx)

    def test_constant_field_zero_tendency(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = adv(h, u, v, method="upwind1")
        # strictly interior (away from ghost edges): flux difference = 0
        np.testing.assert_allclose(result[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_all_methods_run(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        for method in ["naive", "upwind1", "upwind2", "upwind3"]:
            result = adv(h, u, v, method=method)
            assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_tvd_methods_run(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, v, method=method)
            assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_tvd_constant_zero_tendency(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, v, method=method)
            np.testing.assert_allclose(result[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_ghost_ring_zero(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = adv(h, u, v)
        # Ghost ring and boundary layers stay zero
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[1, :], 0.0)
        np.testing.assert_array_equal(result[-2, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, 1], 0.0)
        np.testing.assert_array_equal(result[:, -2], 0.0)
        np.testing.assert_array_equal(result[:, -1], 0.0)

    def test_unknown_method_raises(self, grid2d):
        adv = Advection2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        with pytest.raises(ValueError, match="Unknown method"):
            adv(h, u, v, method="bogus")


class TestAdvection3D:
    def test_output_shape(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert adv(h, u, v).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_constant_zero_tendency(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = adv(h, u, v, method="upwind1")
        # Valid region: all z-interior levels, deep horizontal interior
        # (avoids ghost-adjacent horizontal cells where flux ghosts are 0).
        np.testing.assert_allclose(result[1:-1, 2:-2, 2:-2], 0.0, atol=1e-10)

    def test_ghost_ring_zero(self, grid3d):
        """Ghost and boundary-adjacent rings must stay zero (no flux ghost set)."""
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = adv(h, u, v)
        # Outer ghost rows/cols
        np.testing.assert_array_equal(result[:, 0, :], 0.0)
        np.testing.assert_array_equal(result[:, -1, :], 0.0)
        np.testing.assert_array_equal(result[:, :, 0], 0.0)
        np.testing.assert_array_equal(result[:, :, -1], 0.0)
        # Second ring (boundary-adjacent horizontal cells, flux ghost not set)
        np.testing.assert_array_equal(result[:, 1, :], 0.0)
        np.testing.assert_array_equal(result[:, -2, :], 0.0)
        np.testing.assert_array_equal(result[:, :, 1], 0.0)
        np.testing.assert_array_equal(result[:, :, -2], 0.0)

    def test_tvd_methods_run(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, v, method=method)
            assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_tvd_constant_zero_tendency(self, grid3d):
        adv = Advection3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for method in ["minmod", "van_leer", "superbee", "mc"]:
            result = adv(h, u, v, method=method)
            np.testing.assert_allclose(result[1:-1, 2:-2, 2:-2], 0.0, atol=1e-10)
