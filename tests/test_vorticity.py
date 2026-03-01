"""Tests for Vorticity2D and Vorticity3D."""
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid import ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.vorticity import Vorticity2D, Vorticity3D


@pytest.fixture
def grid2d():
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return ArakawaCGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


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
        qu_e, qv_e = vort.pv_flux_energy_conserving(q, u, v)
        qu_s, qv_s = vort.pv_flux_enstrophy_conserving(q, u, v)
        qu_al, qv_al = vort.pv_flux_arakawa_lamb(q, u, v, alpha=1.0 / 3.0)
        np.testing.assert_allclose(
            qu_al, 1.0 / 3.0 * qu_e + 2.0 / 3.0 * qu_s, rtol=1e-5
        )


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
