"""Tests for Vorticity2D and Vorticity3D."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.operators.vorticity import Vorticity2D, Vorticity3D


@pytest.fixture
def grid2d():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


class TestVorticity2D:
    def test_relative_vorticity_irrotational(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = vort.relative_vorticity(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_relative_vorticity_output_shape(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert vort.relative_vorticity(u, v).shape == (grid2d.Ny, grid2d.Nx)

    def test_potential_vorticity_output_shape(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = vort.potential_vorticity(u, v, h, f)
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_pv_positive_f_positive_h(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        f = 1.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = vort.potential_vorticity(u, v, h, f)
        # q = (0 + f) / h = 0.5
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.5, rtol=1e-5)

    def test_pv_flux_energy_output_shapes(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        qu, qv = vort.pv_flux_energy_conserving(q, u, v)
        assert qu.shape == (grid2d.Ny, grid2d.Nx)
        assert qv.shape == (grid2d.Ny, grid2d.Nx)

    def test_pv_flux_enstrophy_output_shapes(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        qu, qv = vort.pv_flux_enstrophy_conserving(q, u, v)
        assert qu.shape == (grid2d.Ny, grid2d.Nx)
        assert qv.shape == (grid2d.Ny, grid2d.Nx)

    def test_pv_flux_arakawa_lamb_output_shapes(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        qu, qv = vort.pv_flux_arakawa_lamb(q, u, v)
        assert qu.shape == (grid2d.Ny, grid2d.Nx)
        assert qv.shape == (grid2d.Ny, grid2d.Nx)

    def test_pv_flux_arakawa_lamb_is_blend(self, grid2d):
        """AL flux = 1/3 * energy + 2/3 * enstrophy."""
        vort = Vorticity2D(grid=grid2d)
        q = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        qu_e, _qv_e = vort.pv_flux_energy_conserving(q, u, v)
        qu_s, _qv_s = vort.pv_flux_enstrophy_conserving(q, u, v)
        qu_al, _qv_al = vort.pv_flux_arakawa_lamb(q, u, v, alpha=1.0 / 3.0)
        np.testing.assert_allclose(
            qu_al, 1.0 / 3.0 * qu_e + 2.0 / 3.0 * qu_s, rtol=1e-5
        )

    def test_relative_vorticity_solid_body_rotation(self, grid2d):
        """Solid-body rotation u = -c*y, v = c*x gives vorticity = 2*c at X-points.

        For linear fields the first-order finite-difference stencils are exact:
        dv/dx = c  and  du/dy = -c  →  zeta = c - (-c) = 2*c.
        """
        vort = Vorticity2D(grid=grid2d)
        c = 1.5
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-c * y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = vort.relative_vorticity(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c, rtol=1e-5)

    def test_relative_vorticity_sign_convention(self, grid2d):
        """Counter-clockwise rotation (positive omega) yields positive vorticity."""
        vort = Vorticity2D(grid=grid2d)
        # pure counter-clockwise: u < 0 in upper half, v > 0 in right half
        # Use simple solid-body rotation with c > 0 → expect positive vorticity
        c = 1.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-c * y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = vort.relative_vorticity(u, v)
        assert jnp.all(result[1:-1, 1:-1] > 0), (
            "Expected positive vorticity for CCW rotation"
        )

    def test_relative_vorticity_no_nan(self, grid2d):
        """Relative vorticity must not produce NaN for well-defined inputs."""
        vort = Vorticity2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = vort.relative_vorticity(u, v)
        assert jnp.all(jnp.isfinite(result)), "Vorticity contains NaN or Inf"

    def test_potential_vorticity_no_nan_positive_h(self, grid2d):
        """PV must not produce NaN when h > 0."""
        vort = Vorticity2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = vort.potential_vorticity(u, v, h, f)
        np.testing.assert_array_equal(jnp.isnan(result[1:-1, 1:-1]), False)


class TestVorticity3D:
    def test_relative_vorticity_irrotational(self, grid3d):
        vort = Vorticity3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = vort.relative_vorticity(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_output_shape(self, grid3d):
        vort = Vorticity3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert vort.relative_vorticity(u, v).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_relative_vorticity_solid_body_rotation(self, grid3d):
        """3D solid-body rotation gives vorticity = 2*c at each z-level.

        For linear fields the first-order stencils produce the correct constant curl.
        """
        vort = Vorticity3D(grid=grid3d)
        c = 2.0
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        y = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        u2d = jnp.broadcast_to(-c * y[:, None], (grid3d.Ny, grid3d.Nx))
        v2d = jnp.broadcast_to(c * x, (grid3d.Ny, grid3d.Nx))
        u = jnp.broadcast_to(u2d[None, :, :], (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.broadcast_to(v2d[None, :, :], (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = vort.relative_vorticity(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 2.0 * c, rtol=1e-5)

    def test_no_nan_output(self, grid3d):
        """3D relative vorticity must not produce NaN for well-defined inputs."""
        vort = Vorticity3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = vort.relative_vorticity(u, v)
        assert jnp.all(jnp.isfinite(result)), "3D vorticity contains NaN or Inf"
