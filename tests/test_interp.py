"""Tests for Interpolation1D, Interpolation2D, Interpolation3D."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.operators.interpolation import (
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
        """h[j,i]=i+j → h_on_q[j+1/2,i+1/2] = ((i+j)+(i+1+j)+(i+j+1)+(i+1+j+1))/4
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


# ---------------------------------------------------------------------------
# Ghost-cell interaction tests for all four staggered types
# ---------------------------------------------------------------------------


class TestGhostCellInteractions2D:
    """Verify that each operator reads/writes the correct ghost cells.

    Arakawa C-grid ghost-cell map (same-size [Ny, Nx] arrays):

    Type  | Written by forward op  | Ghost (BC-owned, index)
    ------+------------------------+---------------------------------------------
    T     | stays ghost            | row 0/Ny-1, col 0/Nx-1
    U     | [1:-1, 1:-1]           | west face U[j,0]; outside U[j,Nx-1]
    V     | [1:-1, 1:-1]           | south face V[0,i]; outside V[Ny-1,i]
    X     | [1:-1, 1:-1]           | SW corner X[0,0] etc.

    Forward operators (T→U, T→V, T→X) read EAST/NORTH/NE ghost T-cells for
    the last interior face.  Backward operators (U→T, V→T, X→T) read the
    BC-owned west/south ghost face for the first interior T-cell.
    Cross operators (U→V, V→U, U→X, V→X, X→U, X→V) read the BC-owned
    outside ghost on the "far" side of the stencil.
    """

    # -- T→U: last interior U-col reads east ghost T-cell --------------------

    def test_T_to_U_last_col_reads_east_ghost_T(self, grid2d):
        """T_to_U at last interior U-col uses T[j, Nx-1] (east ghost T)."""
        interp = Interpolation2D(grid=grid2d)
        h = jnp.zeros((grid2d.Ny, grid2d.Nx))
        h = h.at[1:-1, grid2d.Nx - 1].set(8.0)  # east ghost T = 8, rest 0
        result = interp.T_to_U(h)
        # out[j, Nx-2] = 0.5*(h[j,Nx-2] + h[j,Nx-1]) = 0.5*(0+8) = 4
        np.testing.assert_allclose(result[1:-1, grid2d.Nx - 2], 4.0, rtol=1e-6)

    # -- U→T: first interior T-col reads west ghost U-face -------------------

    def test_U_to_T_first_col_reads_west_ghost_U(self, grid2d):
        """U_to_T at first interior T-col uses U[j, 0] (west boundary U-face)."""
        interp = Interpolation2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        u = u.at[1:-1, 0].set(6.0)  # west ghost U-face = 6, rest 0
        result = interp.U_to_T(u)
        # out[j, 1] = 0.5*(u[j,1] + u[j,0]) = 0.5*(0+6) = 3
        np.testing.assert_allclose(result[1:-1, 1], 3.0, rtol=1e-6)

    # -- T→V: last interior V-row reads north ghost T-cell -------------------

    def test_T_to_V_last_row_reads_north_ghost_T(self, grid2d):
        """T_to_V at last interior V-row uses T[Ny-1, i] (north ghost T)."""
        interp = Interpolation2D(grid=grid2d)
        h = jnp.zeros((grid2d.Ny, grid2d.Nx))
        h = h.at[grid2d.Ny - 1, 1:-1].set(8.0)  # north ghost T = 8, rest 0
        result = interp.T_to_V(h)
        # out[Ny-2, i] = 0.5*(h[Ny-2,i] + h[Ny-1,i]) = 0.5*(0+8) = 4
        np.testing.assert_allclose(result[grid2d.Ny - 2, 1:-1], 4.0, rtol=1e-6)

    # -- V→T: first interior T-row reads south ghost V-face ------------------

    def test_V_to_T_first_row_reads_south_ghost_V(self, grid2d):
        """V_to_T at first interior T-row uses V[0, i] (south boundary V-face)."""
        interp = Interpolation2D(grid=grid2d)
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = v.at[0, 1:-1].set(6.0)  # south ghost V-face = 6, rest 0
        result = interp.V_to_T(v)
        # out[1, i] = 0.5*(v[1,i] + v[0,i]) = 0.5*(0+6) = 3
        np.testing.assert_allclose(result[1, 1:-1], 3.0, rtol=1e-6)

    # -- U→V: last interior V-row reads north ghost U-row --------------------

    def test_U_to_V_last_row_reads_north_ghost_U(self, grid2d):
        """U_to_V at last V-row uses U[Ny-1, i] (north ghost U-row)."""
        interp = Interpolation2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        u = u.at[grid2d.Ny - 1, 1:-1].set(4.0)  # north ghost U-row = 4
        u = u.at[grid2d.Ny - 2, 1:-1].set(2.0)  # last interior U-row = 2
        result = interp.U_to_V(u)
        # out[Ny-2,i]=0.25*(u[Ny-2,i]+u[Ny-1,i]+u[Ny-2,i-1]+u[Ny-1,i-1])
        # At i=2..Nx-3 (all set): 0.25*(2+4+2+4) = 3
        np.testing.assert_allclose(result[grid2d.Ny - 2, 2:-2], 3.0, rtol=1e-6)

    # -- V→U: last interior U-col reads east ghost V-col ---------------------

    def test_V_to_U_last_col_reads_east_ghost_V(self, grid2d):
        """V_to_U at last U-col uses V[j, Nx-1] (east ghost V-col)."""
        interp = Interpolation2D(grid=grid2d)
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = v.at[1:-1, grid2d.Nx - 1].set(4.0)  # east ghost V-col = 4
        v = v.at[1:-1, grid2d.Nx - 2].set(2.0)  # last interior V-col = 2
        result = interp.V_to_U(v)
        # out[j,Nx-2]=0.25*(v[j,Nx-2]+v[j-1,Nx-2]+v[j,Nx-1]+v[j-1,Nx-1])
        # At j=2..Ny-3 (all set): 0.25*(2+2+4+4) = 3
        np.testing.assert_allclose(result[2:-2, grid2d.Nx - 2], 3.0, rtol=1e-6)

    # -- X→T: first interior T reads SW ghost X-corner -----------------------

    def test_X_to_T_reads_sw_ghost_corner(self, grid2d):
        """X_to_T at first interior T-cell uses X[0,0], X[1,0], X[0,1]."""
        interp = Interpolation2D(grid=grid2d)
        q = jnp.zeros((grid2d.Ny, grid2d.Nx))
        q = q.at[0, 0].set(4.0)  # SW ghost corner
        q = q.at[1, 0].set(2.0)  # west ghost X-col
        q = q.at[0, 1].set(2.0)  # south ghost X-row
        q = q.at[1, 1].set(1.0)  # first interior X
        result = interp.X_to_T(q)
        # out[1,1] = 0.25*(q[1,1]+q[0,1]+q[1,0]+q[0,0]) = 0.25*(1+2+2+4)=2.25
        np.testing.assert_allclose(result[1, 1], 2.25, rtol=1e-6)

    # -- X→U: first interior U-row reads south ghost X-row ------------------

    def test_X_to_U_first_row_reads_south_ghost_X(self, grid2d):
        """X_to_U at first U-row uses X[0, i] (south ghost X-row)."""
        interp = Interpolation2D(grid=grid2d)
        q = jnp.zeros((grid2d.Ny, grid2d.Nx))
        q = q.at[0, 1:-1].set(4.0)  # south ghost X-row = 4
        q = q.at[1, 1:-1].set(2.0)  # first interior X-row = 2
        result = interp.X_to_U(q)
        # out[1, i] = 0.5*(q[1,i] + q[0,i]) = 0.5*(2+4) = 3
        np.testing.assert_allclose(result[1, 1:-1], 3.0, rtol=1e-6)

    # -- X→V: first interior V-col reads west ghost X-col -------------------

    def test_X_to_V_first_col_reads_west_ghost_X(self, grid2d):
        """X_to_V at first V-col uses X[j, 0] (west ghost X-col)."""
        interp = Interpolation2D(grid=grid2d)
        q = jnp.zeros((grid2d.Ny, grid2d.Nx))
        q = q.at[1:-1, 0].set(4.0)  # west ghost X-col = 4
        q = q.at[1:-1, 1].set(2.0)  # first interior X-col = 2
        result = interp.X_to_V(q)
        # out[j, 1] = 0.5*(q[j,1] + q[j,0]) = 0.5*(2+4) = 3
        np.testing.assert_allclose(result[1:-1, 1], 3.0, rtol=1e-6)


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
