"""Tests for SphericalArakawaCGrid2D and SphericalArakawaCGrid3D."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.grid import ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.grid.spherical_grid import (
    SphericalArakawaCGrid2D,
    SphericalArakawaCGrid3D,
)


@pytest.fixture
def grid2d():
    return SphericalArakawaCGrid2D.from_interior(
        nx_interior=10,
        ny_interior=8,
        lon_range=(0.0, 360.0),
        lat_range=(-80.0, 80.0),
        R=1.0,
    )


@pytest.fixture
def grid3d():
    return SphericalArakawaCGrid3D.from_interior(
        nx_interior=10,
        ny_interior=8,
        nz_interior=4,
        lon_range=(0.0, 360.0),
        lat_range=(-80.0, 80.0),
        Lz=100.0,
        R=1.0,
    )


class TestSphericalArakawaCGrid2D:
    def test_grid_sizes(self, grid2d):
        assert grid2d.Nx == 12
        assert grid2d.Ny == 10

    def test_dlon_dlat(self, grid2d):
        expected_dlon = 2.0 * jnp.pi / 10
        expected_dlat = 2.0 * jnp.deg2rad(80.0) / 8
        np.testing.assert_allclose(grid2d.dlon, expected_dlon, rtol=1e-6)
        np.testing.assert_allclose(grid2d.dlat, expected_dlat, rtol=1e-6)

    def test_dx_dy(self, grid2d):
        np.testing.assert_allclose(grid2d.dx, 1.0 * grid2d.dlon, rtol=1e-10)
        np.testing.assert_allclose(grid2d.dy, 1.0 * grid2d.dlat, rtol=1e-10)

    def test_cos_lat_T_shape(self, grid2d):
        assert grid2d.cos_lat_T.shape == (grid2d.Ny, grid2d.Nx)

    def test_cos_lat_T_values(self, grid2d):
        np.testing.assert_allclose(
            grid2d.cos_lat_T, jnp.cos(grid2d.lat_T), atol=1e-12
        )

    def test_cos_lat_U_equals_T(self, grid2d):
        np.testing.assert_allclose(grid2d.cos_lat_U, grid2d.cos_lat_T, atol=0)

    def test_cos_lat_V_offset(self, grid2d):
        expected = jnp.cos(grid2d.lat_T + 0.5 * grid2d.dlat)
        np.testing.assert_allclose(grid2d.cos_lat_V, expected, atol=1e-12)

    def test_cos_lat_X_equals_V(self, grid2d):
        np.testing.assert_allclose(grid2d.cos_lat_X, grid2d.cos_lat_V, atol=0)

    def test_lat_T_coordinates(self, grid2d):
        lat_min = jnp.deg2rad(-80.0)
        # j=1 is first interior cell at lat_min
        np.testing.assert_allclose(grid2d.lat_T[1, 0], lat_min, atol=1e-12)
        # j=0 is ghost cell one step south
        np.testing.assert_allclose(
            grid2d.lat_T[0, 0], lat_min - grid2d.dlat, atol=1e-12
        )

    def test_lon_T_coordinates(self, grid2d):
        lon_min = 0.0
        # i=1 is first interior cell at lon_min
        np.testing.assert_allclose(grid2d.lon_T[0, 1], lon_min, atol=1e-12)
        # i=0 is ghost cell one step west
        np.testing.assert_allclose(
            grid2d.lon_T[0, 0], lon_min - grid2d.dlon, atol=1e-12
        )

    def test_isinstance_arakawa2d(self, grid2d):
        assert isinstance(grid2d, ArakawaCGrid2D)

    def test_jit_compatible(self, grid2d):
        result = jax.jit(lambda g: g.dx)(grid2d)
        np.testing.assert_allclose(result, grid2d.dx, atol=0)


class TestSphericalArakawaCGrid3D:
    def test_grid_sizes(self, grid3d):
        assert grid3d.Nx == 12
        assert grid3d.Ny == 10
        assert grid3d.Nz == 6

    def test_dz(self, grid3d):
        np.testing.assert_allclose(grid3d.dz, 100.0 / 4, rtol=1e-10)

    def test_isinstance_arakawa3d(self, grid3d):
        assert isinstance(grid3d, ArakawaCGrid3D)

    def test_cos_lat_are_2d(self, grid3d):
        assert grid3d.cos_lat_T.ndim == 2
        assert grid3d.cos_lat_V.ndim == 2

    def test_horizontal_grid(self, grid3d):
        h_grid = grid3d.horizontal_grid()
        assert isinstance(h_grid, SphericalArakawaCGrid2D)
        assert h_grid.Nx == grid3d.Nx
        assert h_grid.Ny == grid3d.Ny
        np.testing.assert_allclose(h_grid.dlon, grid3d.dlon, atol=0)
        np.testing.assert_allclose(h_grid.dlat, grid3d.dlat, atol=0)
        np.testing.assert_allclose(h_grid.cos_lat_T, grid3d.cos_lat_T, atol=0)
