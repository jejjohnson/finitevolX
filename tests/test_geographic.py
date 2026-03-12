"""Tests for spherical-coordinate operators in geographic.py."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.operators.geographic import (
    curl_sphere,
    diff2_lon_T,
    diff_lat_T_to_V,
    diff_lat_U_to_X,
    diff_lat_V_to_T,
    diff_lon_T_to_U,
    diff_lon_U_to_T,
    diff_lon_V_to_X,
    divergence_sphere,
    geostrophic_velocity_sphere,
    laplacian_merid_T,
    laplacian_sphere,
    potential_vorticity_sphere,
)

# Use a small test sphere for numerical convenience.
# The latitude range avoids the poles (±80° instead of ±90°) so that
# cos(lat) is never near zero, which would trigger NaN from the pole
# guard in the operators.
R = 1.0
Ny, Nx = 12, 12  # including ghost cells  (10x10 interior)
dlon = 2.0 * jnp.pi / (Nx - 2)  # radians
_LAT_MAX = jnp.deg2rad(80.0)
dlat = 2.0 * _LAT_MAX / (Ny - 2)  # span [-80°, +80°]


@pytest.fixture
def lat_lon_arrays():
    """Return (lat, lon, cos_lat_T, cos_lat_V, cos_lat_U, cos_lat_X) for a uniform sphere."""
    lon = jnp.arange(Nx, dtype=float) * dlon
    lat = -_LAT_MAX + jnp.arange(Ny, dtype=float) * dlat
    LON, LAT = jnp.meshgrid(lon, lat)  # [Ny, Nx]
    cos_T = jnp.cos(LAT)
    # V at j+1/2
    lat_V = -_LAT_MAX + (jnp.arange(Ny, dtype=float) + 0.5) * dlat
    _, LAT_V = jnp.meshgrid(lon, lat_V)
    cos_V = jnp.cos(LAT_V)
    # U at i+1/2 — same latitude as T
    cos_U = cos_T
    # X at j+1/2, i+1/2
    cos_X = cos_V
    return LON, LAT, cos_T, cos_V, cos_U, cos_X


# ======================================================================
# Primitive differences
# ======================================================================


class TestDiffLonTtoU:
    def test_output_shape(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = jnp.ones((Ny, Nx))
        result = diff_lon_T_to_U(h, cos_T, dlon, R)
        assert result.shape == (Ny, Nx)

    def test_constant_field(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = 5.0 * jnp.ones((Ny, Nx))
        result = diff_lon_T_to_U(h, cos_T, dlon, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_ghost_ring_zero(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = jnp.ones((Ny, Nx))
        result = diff_lon_T_to_U(h, cos_T, dlon, R)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)


class TestDiffLatTtoV:
    def test_constant_field(self):
        h = 5.0 * jnp.ones((Ny, Nx))
        result = diff_lat_T_to_V(h, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_latitude(self, lat_lon_arrays):
        """h = lat => dh/dlat = 1 => dh/dy = 1/(R*dlat)*dlat = 1/R."""
        _, LAT, *_ = lat_lon_arrays
        h = LAT
        result = diff_lat_T_to_V(h, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 1.0 / R, rtol=1e-5)


class TestDiffLonVtoX:
    def test_output_shape(self, lat_lon_arrays):
        _, _, _, cos_V, *_ = lat_lon_arrays
        v = jnp.ones((Ny, Nx))
        result = diff_lon_V_to_X(v, cos_V, dlon, R)
        assert result.shape == (Ny, Nx)

    def test_constant_field(self, lat_lon_arrays):
        _, _, _, cos_V, *_ = lat_lon_arrays
        v = 5.0 * jnp.ones((Ny, Nx))
        result = diff_lon_V_to_X(v, cos_V, dlon, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_longitude(self, lat_lon_arrays):
        """v = lon => dv/dlon = 1 => dv/dx = 1/(R cos(lat) dlon) * dlon = 1/(R cos(lat))."""
        LON, _, _, cos_V, _, cos_X = lat_lon_arrays
        v = LON
        result = diff_lon_V_to_X(v, cos_V, dlon, R)
        # Expected: 1 / (R * cos(lat_X)) at interior X-points
        expected = 1.0 / (R * cos_X[1:-1, 1:-1])
        np.testing.assert_allclose(result[1:-1, 1:-1], expected, rtol=1e-5)


class TestDiffLatUtoX:
    def test_output_shape(self):
        u = jnp.ones((Ny, Nx))
        result = diff_lat_U_to_X(u, dlat, R)
        assert result.shape == (Ny, Nx)

    def test_constant_field(self):
        u = 5.0 * jnp.ones((Ny, Nx))
        result = diff_lat_U_to_X(u, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_latitude(self, lat_lon_arrays):
        """u = lat => du/dlat = 1 => du/dy = 1/R."""
        _, LAT, *_ = lat_lon_arrays
        u = LAT
        result = diff_lat_U_to_X(u, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 1.0 / R, rtol=1e-5)


# ======================================================================
# Backward differences (face -> centre)
# ======================================================================


class TestDiffLonUtoT:
    def test_output_shape(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        u = jnp.ones((Ny, Nx))
        result = diff_lon_U_to_T(u, cos_T, dlon, R)
        assert result.shape == (Ny, Nx)

    def test_constant_field(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        u = 5.0 * jnp.ones((Ny, Nx))
        result = diff_lon_U_to_T(u, cos_T, dlon, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)


class TestDiffLatVtoT:
    def test_constant_field(self):
        v = 5.0 * jnp.ones((Ny, Nx))
        result = diff_lat_V_to_T(v, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_latitude(self, lat_lon_arrays):
        """v = lat => dv/dlat = 1 => dv/dy = 1/R."""
        _, LAT, *_ = lat_lon_arrays
        v = LAT
        result = diff_lat_V_to_T(v, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 1.0 / R, rtol=1e-5)


# ======================================================================
# Second-order differences
# ======================================================================


class TestDiff2LonT:
    def test_output_shape(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = jnp.ones((Ny, Nx))
        result = diff2_lon_T(h, cos_T, dlon, R)
        assert result.shape == (Ny, Nx)

    def test_constant_field(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = 5.0 * jnp.ones((Ny, Nx))
        result = diff2_lon_T(h, cos_T, dlon, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_field(self, lat_lon_arrays):
        """d²(lon)/dlon² = 0 => second derivative of linear field is zero."""
        LON, _, cos_T, *_ = lat_lon_arrays
        h = LON
        result = diff2_lon_T(h, cos_T, dlon, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)


class TestLaplacianMeridT:
    def test_output_shape(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = jnp.ones((Ny, Nx))
        result = laplacian_merid_T(h, cos_T, dlat, R)
        assert result.shape == (Ny, Nx)

    def test_constant_field(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = 5.0 * jnp.ones((Ny, Nx))
        result = laplacian_merid_T(h, cos_T, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)


# ======================================================================
# Compound operators
# ======================================================================


class TestDivergenceSphere:
    def test_output_shape(self, lat_lon_arrays):
        _, _, cos_T, cos_V, *_ = lat_lon_arrays
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        result = divergence_sphere(u, v, cos_T, cos_V, dlon, dlat, R)
        assert result.shape == (Ny, Nx)

    def test_uniform_field_zero_u(self, lat_lon_arrays):
        """Uniform u, zero v => du/dlon = 0 => div = 0."""
        _, _, cos_T, cos_V, *_ = lat_lon_arrays
        u = 3.0 * jnp.ones((Ny, Nx))
        v = jnp.zeros((Ny, Nx))
        result = divergence_sphere(u, v, cos_T, cos_V, dlon, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)


class TestCurlSphere:
    def test_output_shape(self, lat_lon_arrays):
        _, _, _, _, cos_U, cos_X = lat_lon_arrays
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        result = curl_sphere(u, v, cos_U, cos_X, dlon, dlat, R)
        assert result.shape == (Ny, Nx)

    def test_ghost_ring_zero(self, lat_lon_arrays):
        _, _, _, _, cos_U, cos_X = lat_lon_arrays
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        result = curl_sphere(u, v, cos_U, cos_X, dlon, dlat, R)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)

    def test_irrotational_field(self, lat_lon_arrays):
        """Zero velocity => zero curl."""
        _, _, _, _, cos_U, cos_X = lat_lon_arrays
        u = jnp.zeros((Ny, Nx))
        v = jnp.zeros((Ny, Nx))
        result = curl_sphere(u, v, cos_U, cos_X, dlon, dlat, R)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)


class TestLaplacianSphere:
    def test_output_shape(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = jnp.ones((Ny, Nx))
        result = laplacian_sphere(h, cos_T, dlon, dlat, R)
        assert result.shape == (Ny, Nx)

    def test_constant_field(self, lat_lon_arrays):
        """Laplacian of a constant is zero."""
        _, _, cos_T, *_ = lat_lon_arrays
        h = 7.0 * jnp.ones((Ny, Nx))
        result = laplacian_sphere(h, cos_T, dlon, dlat, R)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)


class TestGeostrophicVelocitySphere:
    def test_output_shapes(self, lat_lon_arrays):
        _, _, cos_T, *_ = lat_lon_arrays
        h = jnp.ones((Ny, Nx))
        f = 1e-4 * jnp.ones((Ny, Nx))
        u_g, v_g = geostrophic_velocity_sphere(h, f, cos_T, dlon, dlat, R=R)
        assert u_g.shape == (Ny, Nx)
        assert v_g.shape == (Ny, Nx)

    def test_constant_h_zero_velocity(self, lat_lon_arrays):
        """Constant height => zero geostrophic velocity."""
        _, _, cos_T, *_ = lat_lon_arrays
        h = 5.0 * jnp.ones((Ny, Nx))
        f = 1e-4 * jnp.ones((Ny, Nx))
        u_g, v_g = geostrophic_velocity_sphere(h, f, cos_T, dlon, dlat, R=R)
        np.testing.assert_allclose(u_g[1:-1, 1:-1], 0.0, atol=1e-10)
        np.testing.assert_allclose(v_g[1:-1, 1:-1], 0.0, atol=1e-10)


class TestPotentialVorticitySphere:
    def test_output_shape(self, lat_lon_arrays):
        _, _, _, _, cos_U, cos_X = lat_lon_arrays
        u = jnp.zeros((Ny, Nx))
        v = jnp.zeros((Ny, Nx))
        h = jnp.ones((Ny, Nx))
        f = 1e-4 * jnp.ones((Ny, Nx))
        result = potential_vorticity_sphere(u, v, h, f, cos_U, cos_X, dlon, dlat, R)
        assert result.shape == (Ny, Nx)

    def test_zero_velocity_pv_equals_f_over_h(self, lat_lon_arrays):
        """Zero velocity => PV = f/h at X-points."""
        _, _, _, _, cos_U, cos_X = lat_lon_arrays
        u = jnp.zeros((Ny, Nx))
        v = jnp.zeros((Ny, Nx))
        h = 2.0 * jnp.ones((Ny, Nx))
        f = 1e-4 * jnp.ones((Ny, Nx))
        result = potential_vorticity_sphere(u, v, h, f, cos_U, cos_X, dlon, dlat, R)
        # f_on_X / h_on_X = 1e-4 / 2 = 5e-5
        np.testing.assert_allclose(result[1:-1, 1:-1], 5e-5, rtol=1e-5)
