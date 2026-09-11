"""Tests for the spherical-grid physical cell-width helpers.

These properties back resolution and CFL guards in downstream models,
so the contract they must honour is: the widths are the *physical*
east-west / north-south extents of a lat-lon cell, and the reductions
(``min_cell_width``, ``max_aspect``) look only at the physical
interior, never the ghost ring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.utils.constants import R_EARTH


@pytest.fixture
def grid2d():
    """Unit-radius grid, symmetric about the equator with an equator row."""
    return SphericalGrid2D.from_interior(
        nx_interior=10,
        ny_interior=8,
        lon_range=(0.0, 360.0),
        lat_range=(-80.0, 80.0),
        R=1.0,
    )


@pytest.fixture
def grid2d_earth():
    return SphericalGrid2D.from_interior(
        nx_interior=12,
        ny_interior=6,
        lon_range=(-40.0, 20.0),
        lat_range=(20.0, 60.0),
        R=R_EARTH,
    )


@pytest.fixture
def grid3d():
    return SphericalGrid3D.from_interior(
        nx_interior=10,
        ny_interior=8,
        nz_interior=4,
        lon_range=(0.0, 360.0),
        lat_range=(-80.0, 80.0),
        Lz=100.0,
        R=1.0,
    )


def grid_with_cos(cos_column):
    """Build a grid with a hand-chosen ``cos(lat)`` profile.

    Lets the degenerate-pole cases be pinned exactly, independently of
    whether ``cos(pi/2)`` underflows to a small positive or small
    negative number in the active floating-point precision (it is
    ``-4.4e-08`` in float32 and ``+6.1e-17`` in float64, and other test
    modules enable x64 globally).
    """
    cos_column = jnp.asarray(cos_column, dtype=jnp.float32)
    ny = cos_column.shape[0]
    nx = 5
    cos_2d = jnp.broadcast_to(cos_column[:, None], (ny, nx))
    dlon = dlat = 0.1
    lat = jnp.broadcast_to(
        jnp.arccos(jnp.clip(cos_column, -1.0, 1.0))[:, None], (ny, nx)
    )
    lon = jnp.broadcast_to(jnp.arange(nx, dtype=jnp.float32) * dlon, (ny, nx))
    return SphericalGrid2D(
        Nx=nx,
        Ny=ny,
        Lx=nx * dlon,
        Ly=ny * dlat,
        dx=dlon,
        dy=dlat,
        dlon=dlon,
        dlat=dlat,
        R=1.0,
        cos_lat_T=cos_2d,
        cos_lat_U=cos_2d,
        cos_lat_V=cos_2d,
        cos_lat_X=cos_2d,
        lat_T=lat,
        lon_T=lon,
    )


class TestCellWidths2D:
    def test_dx_T_matches_metric_formula(self, grid2d_earth):
        expected = grid2d_earth.R * grid2d_earth.cos_lat_T * grid2d_earth.dlon
        np.testing.assert_allclose(grid2d_earth.dx_T, expected, rtol=1e-12)

    def test_dx_T_equals_uniform_dx_at_equator(self):
        """Where cos(lat) == 1 the true width collapses to the uniform ``dx``."""
        grid = SphericalGrid2D.from_interior(
            nx_interior=8,
            ny_interior=4,
            lon_range=(0.0, 80.0),
            lat_range=(-20.0, 20.0),
            R=R_EARTH,
        )
        lat = np.asarray(grid.lat_T)
        # The equator is a grid row for a symmetric range with even ny.
        j_eq = int(np.argmin(np.abs(lat[:, 0])))
        assert np.isclose(lat[j_eq, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(grid.dx_T)[j_eq, :], grid.dx, rtol=1e-6)

    def test_dx_T_shrinks_polewards(self, grid2d):
        """Width decreases with |latitude| (northern half, ties excluded)."""
        lat = np.asarray(grid2d.lat_T)[:, 0]
        dx = np.asarray(grid2d.dx_T)[:, 0]
        north = lat >= 0.0
        order = np.argsort(lat[north])
        assert np.all(np.diff(dx[north][order]) < 0.0)

    def test_dx_V_uses_the_V_point_latitude(self, grid2d):
        expected = grid2d.R * grid2d.cos_lat_V * grid2d.dlon
        np.testing.assert_allclose(grid2d.dx_V, expected, rtol=1e-12)
        # V sits half a cell north of T, so the widths genuinely differ.
        assert not np.allclose(grid2d.dx_V, grid2d.dx_T)

    def test_dy_T_is_uniform_and_matches_dy(self, grid2d_earth):
        assert grid2d_earth.dy_T == pytest.approx(grid2d_earth.R * grid2d_earth.dlat)
        assert grid2d_earth.dy_T == pytest.approx(grid2d_earth.dy)

    def test_unit_radius_grid_gives_arc_lengths(self, grid2d):
        """``R=1`` grids carry nondimensional arc lengths ``cos(lat)*dlon``."""
        expected = np.cos(np.asarray(grid2d.lat_T)) * grid2d.dlon
        np.testing.assert_allclose(grid2d.dx_T, expected, rtol=1e-12)


class TestMinCellWidth:
    def test_attained_at_highest_interior_latitude(self, grid2d):
        interior_lat = np.asarray(grid2d.lat_T)[1:-1, 1:-1]
        interior_dx = np.asarray(grid2d.dx_T)[1:-1, 1:-1]
        j_pole = np.unravel_index(np.argmax(np.abs(interior_lat)), interior_lat.shape)
        expected = min(float(interior_dx[j_pole]), grid2d.dy_T)
        assert float(grid2d.min_cell_width) == pytest.approx(expected, rel=1e-12)

    def test_excludes_ghost_rows(self):
        """A ghost row past the pole must not drive the minimum negative."""
        grid = SphericalGrid2D.from_interior(
            nx_interior=8,
            ny_interior=6,
            lon_range=(0.0, 360.0),
            lat_range=(-60.0, 60.0),
            R=1.0,
        )
        ghost_lat = np.asarray(grid.lat_T)[[0, -1], 0]
        # Sanity: the ghost ring sits outside the physical latitude band.
        assert np.max(np.abs(ghost_lat)) > np.deg2rad(60.0)
        interior_dx = np.asarray(grid.dx_T)[1:-1, 1:-1]
        assert float(grid.min_cell_width) == pytest.approx(
            min(interior_dx.min(), grid.dy_T), rel=1e-6
        )

    def test_clamps_a_negative_interior_width_to_zero(self):
        """A polar row whose cosine underflows negative must not go below 0."""
        grid = grid_with_cos([0.5, -4.371e-08, 0.5, 0.5])
        raw = np.asarray(grid.dx_T)[1:-1, 1:-1]
        assert raw.min() < 0.0  # the unclamped metric really does go negative
        assert float(grid.min_cell_width) == 0.0

    def test_is_never_negative_on_a_pole_to_pole_grid(self):
        """Whatever the precision, the reduction stays non-negative."""
        grid = SphericalGrid2D.from_interior(
            nx_interior=8,
            ny_interior=4,
            lon_range=(0.0, 360.0),
            lat_range=(-90.0, 90.0),
            R=1.0,
        )
        raw = np.asarray(grid.dx_T)[1:-1, 1:-1]
        expected = min(max(float(raw.min()), 0.0), grid.dy_T)
        assert float(grid.min_cell_width) >= 0.0
        assert float(grid.min_cell_width) == pytest.approx(expected, abs=1e-12)

    def test_takes_dy_when_meridional_spacing_is_finer(self):
        """The minimum is over both directions, not just the zonal one."""
        grid = SphericalGrid2D.from_interior(
            nx_interior=4,  # coarse in longitude
            ny_interior=180,  # fine in latitude
            lon_range=(0.0, 40.0),
            lat_range=(-10.0, 10.0),
            R=1.0,
        )
        assert float(grid.min_cell_width) == pytest.approx(grid.dy_T, rel=1e-12)


class TestMaxAspect:
    def test_matches_inverse_cosine_of_extreme_latitude(self):
        """For ``dlon == dlat`` the aspect is ``1 / cos(lat)``."""
        grid = SphericalGrid2D.from_interior(
            nx_interior=36,  # dlon = 10 degrees
            ny_interior=17,  # dlat = 10 degrees
            lon_range=(0.0, 360.0),
            lat_range=(-85.0, 85.0),
            R=1.0,
        )
        assert grid.dlon == pytest.approx(grid.dlat)
        interior_cos = np.asarray(grid.cos_lat_T)[1:-1, 1:-1]
        # The interior stops short of the poles, so no cell degenerates.
        assert interior_cos.min() > 0.0
        assert float(grid.max_aspect) == pytest.approx(
            1.0 / interior_cos.min(), rel=1e-5
        )

    def test_is_one_on_an_equatorial_isotropic_cell(self):
        grid = SphericalGrid2D.from_interior(
            nx_interior=4,
            ny_interior=2,
            lon_range=(0.0, 4.0),
            lat_range=(-1.0, 1.0),
            R=1.0,
        )
        # Cells straddle the equator, so the aspect stays very close to 1.
        assert float(grid.max_aspect) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("degenerate_cos", [0.0, -4.371e-08])
    def test_infinite_when_an_interior_cell_degenerates(self, degenerate_cos):
        """A zero or negative-width cell reports ``inf``, not a huge ratio."""
        grid = grid_with_cos([0.5, degenerate_cos, 0.5, 0.5])
        assert float(grid.min_cell_width) == 0.0
        assert np.isinf(float(grid.max_aspect))

    def test_finite_when_every_interior_cell_is_positive(self):
        grid = grid_with_cos([0.5, 0.25, 0.5, 0.5])
        assert float(grid.max_aspect) == pytest.approx(1.0 / 0.25, rel=1e-5)


class TestCellWidths3D:
    def test_delegates_to_horizontal_grid(self, grid3d):
        h = grid3d.horizontal_grid()
        np.testing.assert_allclose(grid3d.dx_T, h.dx_T, rtol=1e-12)
        np.testing.assert_allclose(grid3d.dx_V, h.dx_V, rtol=1e-12)
        assert grid3d.dy_T == pytest.approx(h.dy_T)
        assert float(grid3d.min_cell_width) == pytest.approx(float(h.min_cell_width))
        assert float(grid3d.max_aspect) == pytest.approx(float(h.max_aspect))

    def test_widths_are_two_dimensional(self, grid3d):
        assert grid3d.dx_T.shape == (grid3d.Ny, grid3d.Nx)

    def test_min_cell_width_ignores_dz(self, grid3d):
        """A very thin vertical layer must not change the horizontal minimum."""
        thin = SphericalGrid3D.from_interior(
            nx_interior=10,
            ny_interior=8,
            nz_interior=4,
            lon_range=(0.0, 360.0),
            lat_range=(-80.0, 80.0),
            Lz=1e-6,
            R=1.0,
        )
        assert float(thin.min_cell_width) == pytest.approx(float(grid3d.min_cell_width))


class TestJitCompatibility:
    def test_properties_usable_inside_jit(self, grid2d):
        @jax.jit
        def smallest(grid):
            return grid.min_cell_width

        assert float(smallest(grid2d)) == pytest.approx(float(grid2d.min_cell_width))

    def test_widths_usable_inside_jit(self, grid2d):
        @jax.jit
        def cfl_dt(grid, speed):
            return grid.min_cell_width / speed

        got = cfl_dt(grid2d, jnp.asarray(2.0))
        assert float(got) == pytest.approx(float(grid2d.min_cell_width) / 2.0)

    def test_max_aspect_usable_inside_jit(self, grid2d):
        assert float(jax.jit(lambda g: g.max_aspect)(grid2d)) == pytest.approx(
            float(grid2d.max_aspect)
        )
