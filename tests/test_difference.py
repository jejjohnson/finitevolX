"""Tests for Difference1D, Difference2D, Difference3D."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.difference import Difference1D, Difference2D, Difference3D
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


class TestDifference1D:
    def test_diff_x_T_to_U_linear(self, grid1d):
        diff = Difference1D(grid=grid1d)
        # h = c * x  =>  dh/dx = c everywhere
        c = 2.0
        h = c * jnp.arange(grid1d.Nx, dtype=float) * grid1d.dx
        result = diff.diff_x_T_to_U(h)
        # interior should be ~c
        np.testing.assert_allclose(result[1:-1], c, rtol=1e-5)
        # ghost cells should be zero
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[-1], 0.0)

    def test_diff_x_U_to_T_constant(self, grid1d):
        diff = Difference1D(grid=grid1d)
        u = jnp.ones(grid1d.Nx)
        result = diff.diff_x_U_to_T(u)
        # constant field: backward diff = 0 at interior
        np.testing.assert_allclose(result[1:-1], 0.0, atol=1e-10)

    def test_laplacian_quadratic(self, grid1d):
        diff = Difference1D(grid=grid1d)
        # h = x^2  =>  d2h/dx2 = 2
        x = jnp.arange(grid1d.Nx, dtype=float) * grid1d.dx
        h = x**2
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1], 2.0, rtol=1e-5)

    def test_output_shape(self, grid1d):
        diff = Difference1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        assert diff.diff_x_T_to_U(h).shape == (grid1d.Nx,)
        assert diff.diff_x_U_to_T(h).shape == (grid1d.Nx,)
        assert diff.laplacian(h).shape == (grid1d.Nx,)


class TestDifference2D:
    def test_diff_x_T_to_U_linear(self, grid2d):
        diff = Difference2D(grid=grid2d)
        # h = c * x  =>  dh/dx = c
        c = 3.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = diff.diff_x_T_to_U(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_diff_y_T_to_V_linear(self, grid2d):
        diff = Difference2D(grid=grid2d)
        c = 2.5
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = jnp.broadcast_to(c * y[:, None], (grid2d.Ny, grid2d.Nx))
        result = diff.diff_y_T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_diff_y_X_to_U_linear(self, grid2d):
        diff = Difference2D(grid=grid2d)
        c = 1.75
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        q = jnp.broadcast_to(c * y[:, None], (grid2d.Ny, grid2d.Nx))
        result = diff.diff_y_X_to_U(q)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_diff_x_X_to_V_linear(self, grid2d):
        diff = Difference2D(grid=grid2d)
        c = 1.25
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        q = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = diff.diff_x_X_to_V(q)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_divergence_constant_flow(self, grid2d):
        diff = Difference2D(grid=grid2d)
        # uniform flow: divergence = 0
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = diff.divergence(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_curl_irrotational(self, grid2d):
        diff = Difference2D(grid=grid2d)
        # u = constant, v = constant => curl = 0
        u = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = diff.curl(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_laplacian_quadratic(self, grid2d):
        diff = Difference2D(grid=grid2d)
        # h = x^2 + y^2 => laplacian = 2 + 2 = 4
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = x[None, :] ** 2 + y[:, None] ** 2
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 4.0, rtol=1e-5)

    def test_corner_streamfunction_velocity_is_discretely_nondivergent(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float)
        y = jnp.arange(grid2d.Ny, dtype=float)
        psi_x = y[:, None] * x[None, :]

        u = -diff.diff_y_X_to_U(psi_x)
        v = diff.diff_x_X_to_V(psi_x)
        divergence = diff.divergence(u, v)

        np.testing.assert_allclose(divergence[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_output_shape(self, grid2d):
        diff = Difference2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert diff.diff_x_T_to_U(h).shape == (grid2d.Ny, grid2d.Nx)
        assert diff.laplacian(h).shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_is_zero(self, grid2d):
        diff = Difference2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = diff.diff_x_T_to_U(h)
        # boundary rows/cols must be zero
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, -1], 0.0)

    def test_diff_x_U_to_T_linear(self, grid2d):
        """Backward x-difference on a linear u field: du/dx = c.

        u[j, i] = c * i * dx  (U-point at i+1/2)
        du_dx[j, i] = (u[j,i] - u[j,i-1]) / dx = c
        """
        diff = Difference2D(grid=grid2d)
        c = 2.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        u = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = diff.diff_x_U_to_T(u)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_diff_y_V_to_T_linear(self, grid2d):
        """Backward y-difference on a linear v field: dv/dy = c.

        v[j, i] = c * j * dy  (V-point at j+1/2)
        dv_dy[j, i] = (v[j,i] - v[j-1,i]) / dy = c
        """
        diff = Difference2D(grid=grid2d)
        c = 1.5
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        v = jnp.broadcast_to(c * y[:, None], (grid2d.Ny, grid2d.Nx))
        result = diff.diff_y_V_to_T(v)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_diff_y_U_to_X_linear(self, grid2d):
        """Forward y-difference on a linear u field: du/dy = c.

        u[j, i] = c * j * dy  (U-point varying in y)
        du_dy[j+1/2, i+1/2] = (u[j+1,i+1/2] - u[j,i+1/2]) / dy = c
        """
        diff = Difference2D(grid=grid2d)
        c = 1.2
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(c * y[:, None], (grid2d.Ny, grid2d.Nx))
        result = diff.diff_y_U_to_X(u)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_diff_x_V_to_X_linear(self, grid2d):
        """Forward x-difference on a linear v field: dv/dx = c.

        v[j, i] = c * i * dx  (V-point varying in x)
        dv_dx[j+1/2, i+1/2] = (v[j+1/2,i+1] - v[j+1/2,i]) / dx = c
        """
        diff = Difference2D(grid=grid2d)
        c = 0.8
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        v = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = diff.diff_x_V_to_X(v)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_curl_solid_body_rotation(self, grid2d):
        """Solid-body rotation u=-c*y, v=c*x gives curl = 2c at X-points.

        zeta = dv/dx - du/dy = c - (-c) = 2c
        """
        diff = Difference2D(grid=grid2d)
        c = 1.5
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-c * y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        result = diff.curl(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c, rtol=1e-5)

    def test_divergence_linear_flow(self, grid2d):
        """u=c*x, v=c*y gives divergence = 2c at T-points.

        delta = du/dx + dv/dy = c + c = 2c
        """
        diff = Difference2D(grid=grid2d)
        c = 2.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(c * y[:, None], (grid2d.Ny, grid2d.Nx))
        result = diff.divergence(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c, rtol=1e-5)

    def test_second_derivative_x_linear_is_zero(self, grid2d):
        """Second x-derivative of a linear field is zero at deep interior.

        diff_x_U_to_T(diff_x_T_to_U(h)) = d²h/dx² = 0 for h = c*x.

        Why column i=1 is excluded:
        diff_x_T_to_U writes only to interior U-points [1:-1, 1:-1].  The
        ghost U-face at column 0 (the west boundary face U[j, 1/2]) is NOT
        filled by this operator — its value is owned by the boundary-condition
        layer, not by the interior stencil.  When diff_x_U_to_T then reads
        dh_u[j, 0]=0 at i=1, it produces (c-0)/dx = c/dx instead of 0.
        This is correct operator behaviour: ghost cells always require explicit
        BC treatment before chained operator calls.  Columns i=2..Nx-3 use
        only interior dh_u values and correctly recover d²h/dx² = 0.
        """
        diff = Difference2D(grid=grid2d)
        c = 3.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        dh_u = diff.diff_x_T_to_U(h)
        result = diff.diff_x_U_to_T(dh_u)
        # At i=1 dh_u[j,0]=0 (ghost, no BC applied) → c/dx ≠ 0; skip that column.
        # At i=2..Nx-3 both dh_u neighbours are interior → second deriv = 0.
        np.testing.assert_allclose(result[1:-1, 2:-1], 0.0, atol=1e-10)

        # Also verify i=1 gives the EXPECTED non-zero value from zero ghost,
        # confirming the implementation is deterministic, not silently wrong.
        expected_i1 = c / grid2d.dx  # (c - 0) / dx
        np.testing.assert_allclose(result[1:-1, 1], expected_i1, rtol=1e-6)

    def test_second_derivative_y_linear_is_zero(self, grid2d):
        """Second y-derivative of a linear field is zero at deep interior.

        diff_y_V_to_T(diff_y_T_to_V(h)) = d²h/dy² = 0 for h = c*y.

        The ghost V-face at row 0 (south boundary face V[1/2, i]) is NOT
        filled by diff_y_T_to_V.  At j=1, diff_y_V_to_T reads dv_v[0, i]=0
        and produces (c-0)/dy instead of 0.  Rows j=2..Ny-3 are unaffected.
        """
        diff = Difference2D(grid=grid2d)
        c = 2.5
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = jnp.broadcast_to(c * y[:, None], (grid2d.Ny, grid2d.Nx))
        dh_v = diff.diff_y_T_to_V(h)
        result = diff.diff_y_V_to_T(dh_v)
        # At j=2..Ny-3 both dh_v neighbours are interior → second deriv = 0.
        np.testing.assert_allclose(result[2:-1, 1:-1], 0.0, atol=1e-10)

        # Verify j=1 gives the expected non-zero value due to zero south ghost.
        expected_j1 = c / grid2d.dy  # (c - 0) / dy
        np.testing.assert_allclose(result[1, 1:-1], expected_j1, rtol=1e-6)


class TestDifference3D:
    def test_diff_x_T_to_U_shape(self, grid3d):
        diff = Difference3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff.diff_x_T_to_U(h)
        assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_divergence_constant_flow(self, grid3d):
        diff = Difference3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff.divergence(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_laplacian_quadratic(self, grid3d):
        diff = Difference3D(grid=grid3d)
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        y = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        h2d = x[None, :] ** 2 + y[:, None] ** 2
        h = jnp.broadcast_to(h2d, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 4.0, rtol=1e-5)
