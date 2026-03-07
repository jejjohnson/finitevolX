"""Tests for Interpolation1D, Interpolation2D, Interpolation3D."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.interpolation import (
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
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
        # h[i] = i  =>  h_on_u[i+1/2] = (h[i] + h[i+1]) / 2 = i + 0.5
        h = jnp.arange(grid1d.Nx, dtype=float)
        result = interp.T_to_U(h)
        expected = jnp.arange(grid1d.Nx, dtype=float) + 0.5
        np.testing.assert_allclose(result[1:-1], expected[1:-1], rtol=1e-6)

    def test_U_to_T_linear(self, grid1d):
        interp = Interpolation1D(grid=grid1d)
        # u[i] = i  (U-point at i+1/2)
        # u_on_T[i] = (u[i] + u[i-1]) / 2 = (i + (i-1)) / 2 = i - 0.5
        u = jnp.arange(grid1d.Nx, dtype=float)
        result = interp.U_to_T(u)
        expected = jnp.arange(grid1d.Nx, dtype=float) - 0.5
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
            interp.T_to_U,
            interp.T_to_V,
            interp.T_to_X,
            interp.U_to_T,
            interp.V_to_T,
            interp.X_to_T,
            interp.U_to_X,
            interp.V_to_X,
            interp.X_to_U,
            interp.X_to_V,
            interp.U_to_V,
            interp.V_to_U,
        ]:
            assert method(h).shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_zero(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = interp.T_to_U(h)
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)


class TestInterpolation2DNonConstant:
    """Non-constant field tests that catch axis mix-ups and off-by-one errors.

    All stencils are verified against their half-index formulae:
      T[j, i]  cell centre  (j,     i    )
      U[j, i]  east face    (j,     i+1/2)
      V[j, i]  north face   (j+1/2, i    )
      X[j, i]  NE corner    (j+1/2, i+1/2)
    """

    def test_T_to_U_linear_x(self, grid2d):
        """h[j,i]=i → h_on_u[j,i+1/2] = (h[j,i]+h[j,i+1])/2 = i+0.5"""
        interp = Interpolation2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float)
        h = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        result = interp.T_to_U(h)
        # out[j, i] = 0.5*(h[j,i]+h[j,i+1]) = i+0.5  for i in interior
        # h is constant in y → compare one representative interior row
        expected = ix + 0.5
        np.testing.assert_allclose(result[1, 1:-1], expected[1:-1], rtol=1e-6)

    def test_T_to_V_linear_y(self, grid2d):
        """h[j,i]=j → h_on_v[j+1/2,i] = (h[j,i]+h[j+1,i])/2 = j+0.5"""
        interp = Interpolation2D(grid=grid2d)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        h = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))
        result = interp.T_to_V(h)
        # out[j, i] = 0.5*(h[j,i]+h[j+1,i]) = j+0.5  for j in interior
        # h is constant in x → compare one representative interior column
        expected = jy + 0.5
        np.testing.assert_allclose(result[1:-1, 1], expected[1:-1], rtol=1e-6)

    def test_T_to_X_bilinear(self, grid2d):
        """h[j,i]=i+j → h_on_q[j+1/2,i+1/2] = (i+j)+(i+1+j)+(i+j+1)+(i+1+j+1))/4
        = i+j+1 → out[j,i] = i+j+1."""
        interp = Interpolation2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        h = ix[None, :] + jy[:, None]
        result = interp.T_to_X(h)
        # out[j,i] = 0.25*(h[j,i]+h[j,i+1]+h[j+1,i]+h[j+1,i+1]) = i+j+1
        expected = (ix[None, :] + jy[:, None]) + 1.0
        np.testing.assert_allclose(result[1:-1, 1:-1], expected[1:-1, 1:-1], rtol=1e-6)

    def test_U_to_T_linear_x(self, grid2d):
        """u[j,i]=i (U-point at i+1/2) → u_on_T[j,i]=(u[j,i]+u[j,i-1])/2=i-0.5"""
        interp = Interpolation2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float)
        u = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        result = interp.U_to_T(u)
        # u is constant in y → compare one representative interior row
        expected = ix - 0.5
        np.testing.assert_allclose(result[1, 1:-1], expected[1:-1], rtol=1e-6)

    def test_V_to_T_linear_y(self, grid2d):
        """v[j,i]=j (V-point at j+1/2) → v_on_T[j,i]=(v[j,i]+v[j-1,i])/2=j-0.5"""
        interp = Interpolation2D(grid=grid2d)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        v = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))
        result = interp.V_to_T(v)
        # v is constant in x → compare one representative interior column
        expected = jy - 0.5
        np.testing.assert_allclose(result[1:-1, 1], expected[1:-1], rtol=1e-6)

    def test_U_to_X_linear_y(self, grid2d):
        """u[j,i]=j (U-point) → u_on_q[j+1/2,i+1/2]=(u[j,i]+u[j+1,i])/2=j+0.5"""
        interp = Interpolation2D(grid=grid2d)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        u = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))
        result = interp.U_to_X(u)
        # u is constant in x → compare one representative interior column
        expected = jy + 0.5
        np.testing.assert_allclose(result[1:-1, 1], expected[1:-1], rtol=1e-6)

    def test_V_to_X_linear_x(self, grid2d):
        """v[j,i]=i (V-point) → v_on_q[j+1/2,i+1/2]=(v[j,i]+v[j,i+1])/2=i+0.5"""
        interp = Interpolation2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float)
        v = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        result = interp.V_to_X(v)
        # v is constant in y → compare one representative interior row
        expected = ix + 0.5
        np.testing.assert_allclose(result[1, 1:-1], expected[1:-1], rtol=1e-6)

    def test_X_to_U_linear_y(self, grid2d):
        """q[j,i]=j (X-point at j+1/2) → q_on_u[j,i+1/2]=(q[j,i]+q[j-1,i])/2=j-0.5"""
        interp = Interpolation2D(grid=grid2d)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        q = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))
        result = interp.X_to_U(q)
        # q is constant in x → compare one representative interior column
        expected = jy - 0.5
        np.testing.assert_allclose(result[1:-1, 1], expected[1:-1], rtol=1e-6)

    def test_X_to_V_linear_x(self, grid2d):
        """q[j,i]=i (X-point at i+1/2) → q_on_v[j+1/2,i]=(q[j,i]+q[j,i-1])/2=i-0.5"""
        interp = Interpolation2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float)
        q = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        result = interp.X_to_V(q)
        # q is constant in y → compare one representative interior row
        expected = ix - 0.5
        np.testing.assert_allclose(result[1, 1:-1], expected[1:-1], rtol=1e-6)

    def test_X_to_T_bilinear(self, grid2d):
        """q[j,i]=i+j → q_on_T[j,i]=(q[j,i]+q[j-1,i]+q[j,i-1]+q[j-1,i-1])/4
        = ((i+j)+(i+j-1)+(i-1+j)+(i-1+j-1))/4 = i+j-1 → out[j,i]=i+j-1."""
        interp = Interpolation2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        q = ix[None, :] + jy[:, None]
        result = interp.X_to_T(q)
        expected = (ix[None, :] + jy[:, None]) - 1.0
        np.testing.assert_allclose(result[1:-1, 1:-1], expected[1:-1, 1:-1], rtol=1e-6)

    def test_U_to_V_linear_x(self, grid2d):
        """u[j,i]=i (U-point) → u_on_v[j+1/2,i]=(u[j,i]+u[j+1,i]+u[j,i-1]+u[j+1,i-1])/4
        = (i+i+(i-1)+(i-1))/4 = i-0.5."""
        interp = Interpolation2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float)
        u = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        result = interp.U_to_V(u)
        # u is constant in y → compare one representative interior row
        expected = ix - 0.5
        np.testing.assert_allclose(result[1, 1:-1], expected[1:-1], rtol=1e-6)

    def test_V_to_U_linear_y(self, grid2d):
        """v[j,i]=j (V-point) → v_on_u[j,i+1/2]=(v[j,i]+v[j-1,i]+v[j,i+1]+v[j-1,i+1])/4
        = (j+(j-1)+j+(j-1))/4 = j-0.5."""
        interp = Interpolation2D(grid=grid2d)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        v = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))
        result = interp.V_to_U(v)
        # v is constant in x → compare one representative interior column
        expected = jy - 0.5
        np.testing.assert_allclose(result[1:-1, 1], expected[1:-1], rtol=1e-6)

    def test_T_to_U_does_not_mix_axes(self, grid2d):
        """T_to_U averages in x; a y-only field must be preserved exactly."""
        interp = Interpolation2D(grid=grid2d)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        h = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))
        result = interp.T_to_U(h)
        # h is constant in x → x-average leaves it unchanged
        np.testing.assert_allclose(result[1:-1, 1:-1], h[1:-1, 1:-1], rtol=1e-6)

    def test_T_to_V_does_not_mix_axes(self, grid2d):
        """T_to_V averages in y; an x-only field must be preserved exactly."""
        interp = Interpolation2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float)
        h = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        result = interp.T_to_V(h)
        # h is constant in y → y-average leaves it unchanged
        np.testing.assert_allclose(result[1:-1, 1:-1], h[1:-1, 1:-1], rtol=1e-6)


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
