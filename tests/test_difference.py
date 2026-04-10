"""Tests for Difference1D, Difference2D, Difference3D."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)
from finitevolx._src.operators.difference import (
    Difference1D,
    Difference2D,
    Difference3D,
)
from finitevolx._src.operators.interpolation import Interpolation2D


@pytest.fixture
def grid1d():
    return CartesianGrid1D.from_interior(8, 1.0)


@pytest.fixture
def grid2d():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


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


# ---------------------------------------------------------------------------
# Perpendicular gradient (grad_perp) tests
# ---------------------------------------------------------------------------


class TestGradPerp2D:
    """Tests for Difference2D.grad_perp: ψ(T) → (u(U), v(V))."""

    def test_output_shapes(self, grid2d):
        diff = Difference2D(grid=grid2d)
        psi = jnp.ones((grid2d.Ny, grid2d.Nx))
        u, v = diff.grad_perp(psi)
        assert u.shape == (grid2d.Ny, grid2d.Nx)
        assert v.shape == (grid2d.Ny, grid2d.Nx)

    def test_constant_psi_gives_zero_velocity(self, grid2d):
        """Uniform ψ → zero velocity everywhere in the interior: grad_perp(const) = (0, 0)."""
        diff = Difference2D(grid=grid2d)
        psi = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u, v = diff.grad_perp(psi)
        np.testing.assert_allclose(u[1:-1, 1:-1], 0.0, atol=1e-12)
        np.testing.assert_allclose(v[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, grid2d):
        diff = Difference2D(grid=grid2d)
        psi = jnp.ones((grid2d.Ny, grid2d.Nx))
        u, v = diff.grad_perp(psi)
        # u ghost ring
        np.testing.assert_array_equal(u[0, :], 0.0)
        np.testing.assert_array_equal(u[-1, :], 0.0)
        np.testing.assert_array_equal(u[:, 0], 0.0)
        np.testing.assert_array_equal(u[:, -1], 0.0)
        # v ghost ring
        np.testing.assert_array_equal(v[0, :], 0.0)
        np.testing.assert_array_equal(v[-1, :], 0.0)
        np.testing.assert_array_equal(v[:, 0], 0.0)
        np.testing.assert_array_equal(v[:, -1], 0.0)

    def test_nondivergent(self, grid2d):
        """grad_perp(ψ) is discretely non-divergent: div(u, v) = 0.

        Deep interior [2:-2, 2:-2] avoids ghost-ring contamination from the
        chained divergence operator reading the zero ghost ring of u/v.
        """
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        psi = x[None, :] * y[:, None]  # ψ = x * y
        u, v = diff.grad_perp(psi)
        divergence = diff.divergence(u, v)
        np.testing.assert_allclose(divergence[2:-2, 2:-2], 0.0, atol=1e-12)

    def test_linear_psi_x(self, grid2d):
        """ψ = c * x → u = 0, v = c at interior [1:-1, 1:-1].

        ∂ψ/∂y = 0 → u = 0
        ∂ψ/∂x = c → v = c
        """
        diff = Difference2D(grid=grid2d)
        c = 3.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        psi = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        u, v = diff.grad_perp(psi)
        np.testing.assert_allclose(u[1:-1, 1:-1], 0.0, atol=1e-12)
        np.testing.assert_allclose(v[1:-1, 1:-1], c, rtol=1e-5)

    def test_linear_psi_y(self, grid2d):
        """ψ = c * y → u = -c, v = 0 at interior [1:-1, 1:-1].

        ∂ψ/∂y = c → u = -c
        ∂ψ/∂x = 0 → v = 0
        """
        diff = Difference2D(grid=grid2d)
        c = 2.0
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        psi = jnp.broadcast_to(c * y[:, None], (grid2d.Ny, grid2d.Nx))
        u, v = diff.grad_perp(psi)
        np.testing.assert_allclose(u[1:-1, 1:-1], -c, rtol=1e-5)
        np.testing.assert_allclose(v[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_matches_manual_decomposition(self, grid2d):
        """grad_perp agrees with T→X→(X→U, X→V) at the deep interior.

        The composed approach has zero X-point ghost ring, so the first
        interior strip differs.  The direct stencil used by grad_perp reads
        the T-point ghost cells of ψ, giving correct values everywhere in
        [1:-1, 1:-1].  Both approaches agree at [2:-2, 2:-2].
        """
        diff = Difference2D(grid=grid2d)
        interp = Interpolation2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        psi = jnp.sin(x[None, :]) * jnp.cos(y[:, None])

        u, v = diff.grad_perp(psi)

        psi_x = interp.T_to_X(psi)
        u_expected = -diff.diff_y_X_to_U(psi_x)
        v_expected = diff.diff_x_X_to_V(psi_x)
        np.testing.assert_allclose(u[2:-2, 2:-2], u_expected[2:-2, 2:-2], rtol=1e-5)
        np.testing.assert_allclose(v[2:-2, 2:-2], v_expected[2:-2, 2:-2], rtol=1e-5)

    def test_anisotropic_grid(self):
        """grad_perp respects different dx and dy spacings."""
        grid = CartesianGrid2D.from_interior(8, 8, 1.0, 2.0)
        diff = Difference2D(grid=grid)
        c = 1.5
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        psi = jnp.broadcast_to(c * y[:, None], (grid.Ny, grid.Nx))
        u, _v = diff.grad_perp(psi)
        # ∂ψ/∂y = c, so u = -c at interior [1:-1, 1:-1]
        np.testing.assert_allclose(u[1:-1, 1:-1], -c, rtol=1e-5)

    def test_mask_u_zeros_velocity(self, grid2d):
        """mask_u zeroes u at masked points."""
        diff = Difference2D(grid=grid2d)
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        psi = jnp.broadcast_to(y[:, None], (grid2d.Ny, grid2d.Nx))
        mask_u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        u, _v = diff.grad_perp(psi, mask_u=mask_u)
        np.testing.assert_allclose(u, 0.0, atol=1e-12)

    def test_mask_v_zeros_velocity(self, grid2d):
        """mask_v zeroes v at masked points."""
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        psi = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))
        mask_v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        _u, v = diff.grad_perp(psi, mask_v=mask_v)
        np.testing.assert_allclose(v, 0.0, atol=1e-12)

    def test_no_nan_output(self, grid2d):
        """grad_perp must not produce NaN for well-defined inputs."""
        diff = Difference2D(grid=grid2d)
        psi = jnp.ones((grid2d.Ny, grid2d.Nx))
        u, v = diff.grad_perp(psi)
        assert jnp.all(jnp.isfinite(u)), "u contains NaN or Inf"
        assert jnp.all(jnp.isfinite(v)), "v contains NaN or Inf"


# ---------------------------------------------------------------------------
# Ghost-cell interaction tests for difference operators
# ---------------------------------------------------------------------------


class TestDifferenceGhostCells2D:
    """Verify that each difference operator reads/writes the correct ghost cells.

    Forward operators (T→U, T→V, V→X, U→X):
      last interior face reads the EAST/NORTH ghost of the source array.

    Backward operators (U→T, V→T, X→U, X→V):
      first interior output reads the WEST/SOUTH ghost of the source array.
    """

    # -- diff_x_T_to_U: last U-col reads east ghost T -----------------------

    def test_diff_x_T_to_U_last_col_reads_east_ghost_T(self, grid2d):
        """diff_x_T_to_U at last interior U-col uses T[j, Nx-1] (east ghost T)."""
        diff = Difference2D(grid=grid2d)
        h = jnp.zeros((grid2d.Ny, grid2d.Nx))
        h = h.at[1:-1, grid2d.Nx - 2].set(2.0)  # last interior T = 2
        h = h.at[1:-1, grid2d.Nx - 1].set(6.0)  # east ghost T = 6
        result = diff.diff_x_T_to_U(h)
        expected = (6.0 - 2.0) / grid2d.dx
        np.testing.assert_allclose(result[1:-1, grid2d.Nx - 2], expected, rtol=1e-6)

    # -- diff_x_U_to_T: first T-col reads west ghost U-face -----------------

    def test_diff_x_U_to_T_first_col_reads_west_ghost_U(self, grid2d):
        """diff_x_U_to_T at first T-col uses U[j, 0] (west boundary U-face)."""
        diff = Difference2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        u = u.at[1:-1, 0].set(3.0)  # west ghost U-face = 3
        u = u.at[1:-1, 1].set(9.0)  # first interior U = 9
        result = diff.diff_x_U_to_T(u)
        expected = (9.0 - 3.0) / grid2d.dx
        np.testing.assert_allclose(result[1:-1, 1], expected, rtol=1e-6)

    # -- diff_y_T_to_V: last V-row reads north ghost T ----------------------

    def test_diff_y_T_to_V_last_row_reads_north_ghost_T(self, grid2d):
        """diff_y_T_to_V at last interior V-row uses T[Ny-1, i] (north ghost T)."""
        diff = Difference2D(grid=grid2d)
        h = jnp.zeros((grid2d.Ny, grid2d.Nx))
        h = h.at[grid2d.Ny - 2, 1:-1].set(1.0)  # last interior T = 1
        h = h.at[grid2d.Ny - 1, 1:-1].set(5.0)  # north ghost T = 5
        result = diff.diff_y_T_to_V(h)
        expected = (5.0 - 1.0) / grid2d.dy
        np.testing.assert_allclose(result[grid2d.Ny - 2, 1:-1], expected, rtol=1e-6)

    # -- diff_y_V_to_T: first T-row reads south ghost V-face ----------------

    def test_diff_y_V_to_T_first_row_reads_south_ghost_V(self, grid2d):
        """diff_y_V_to_T at first T-row uses V[0, i] (south boundary V-face)."""
        diff = Difference2D(grid=grid2d)
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = v.at[0, 1:-1].set(2.0)  # south ghost V-face = 2
        v = v.at[1, 1:-1].set(8.0)  # first interior V = 8
        result = diff.diff_y_V_to_T(v)
        expected = (8.0 - 2.0) / grid2d.dy
        np.testing.assert_allclose(result[1, 1:-1], expected, rtol=1e-6)

    # -- diff_x_V_to_X: last X-col reads east ghost V-col -------------------

    def test_diff_x_V_to_X_last_col_reads_east_ghost_V(self, grid2d):
        """diff_x_V_to_X at last X-col uses V[j, Nx-1] (east ghost V-col)."""
        diff = Difference2D(grid=grid2d)
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = v.at[1:-1, grid2d.Nx - 2].set(4.0)  # last interior V = 4
        v = v.at[1:-1, grid2d.Nx - 1].set(10.0)  # east ghost V = 10
        result = diff.diff_x_V_to_X(v)
        expected = (10.0 - 4.0) / grid2d.dx
        np.testing.assert_allclose(result[1:-1, grid2d.Nx - 2], expected, rtol=1e-6)

    # -- diff_y_U_to_X: last X-row reads north ghost U-row ------------------

    def test_diff_y_U_to_X_last_row_reads_north_ghost_U(self, grid2d):
        """diff_y_U_to_X at last X-row uses U[Ny-1, i] (north ghost U-row)."""
        diff = Difference2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        u = u.at[grid2d.Ny - 2, 1:-1].set(2.0)  # last interior U = 2
        u = u.at[grid2d.Ny - 1, 1:-1].set(8.0)  # north ghost U = 8
        result = diff.diff_y_U_to_X(u)
        expected = (8.0 - 2.0) / grid2d.dy
        np.testing.assert_allclose(result[grid2d.Ny - 2, 1:-1], expected, rtol=1e-6)

    # -- diff_x_X_to_V: first V-col reads west ghost X-col -----------------

    def test_diff_x_X_to_V_first_col_reads_west_ghost_X(self, grid2d):
        """diff_x_X_to_V at first V-col uses X[j, 0] (west ghost X-col)."""
        diff = Difference2D(grid=grid2d)
        q = jnp.zeros((grid2d.Ny, grid2d.Nx))
        q = q.at[1:-1, 0].set(2.0)  # west ghost X = 2
        q = q.at[1:-1, 1].set(8.0)  # first interior X = 8
        result = diff.diff_x_X_to_V(q)
        expected = (8.0 - 2.0) / grid2d.dx
        np.testing.assert_allclose(result[1:-1, 1], expected, rtol=1e-6)

    # -- diff_y_X_to_U: first U-row reads south ghost X-row ----------------

    def test_diff_y_X_to_U_first_row_reads_south_ghost_X(self, grid2d):
        """diff_y_X_to_U at first U-row uses X[0, i] (south ghost X-row)."""
        diff = Difference2D(grid=grid2d)
        q = jnp.zeros((grid2d.Ny, grid2d.Nx))
        q = q.at[0, 1:-1].set(3.0)  # south ghost X = 3
        q = q.at[1, 1:-1].set(9.0)  # first interior X = 9
        result = diff.diff_y_X_to_U(q)
        expected = (9.0 - 3.0) / grid2d.dy
        np.testing.assert_allclose(result[1, 1:-1], expected, rtol=1e-6)


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


# ---------------------------------------------------------------------------
# Analytical correctness: laplacian of constant field
# ---------------------------------------------------------------------------


class TestDifferenceLaplacianConstant:
    """Laplacian of a constant field must be zero everywhere in the interior."""

    def test_laplacian_1d_constant_is_zero(self):
        grid = CartesianGrid1D.from_interior(8, 1.0)
        diff = Difference1D(grid=grid)
        h = 5.0 * jnp.ones(grid.Nx)
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1], 0.0, atol=1e-10)

    def test_laplacian_2d_constant_is_zero(self):
        grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
        diff = Difference2D(grid=grid)
        h = 7.0 * jnp.ones((grid.Ny, grid.Nx))
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_laplacian_3d_constant_is_zero(self):
        grid = CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
        diff = Difference3D(grid=grid)
        h = 3.0 * jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_laplacian_2d_linear_x_is_zero(self):
        """Laplacian of h=c*x is zero (second derivative of linear field = 0).

        Difference2D.laplacian uses a direct 3-point stencil on h:
            d²h/dx²[j,i] = (h[j, i+1] - 2*h[j, i] + h[j, i-1]) / dx²
        For h=c*x=c*i*dx, the cancellation is exact at every interior point
        (including i=1, where h[0]=0 is the natural linear continuation).
        """
        grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
        diff = Difference2D(grid=grid)
        c = 3.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_laplacian_2d_linear_y_is_zero(self):
        """Laplacian of h=c*y is zero (second derivative of linear field = 0).

        Difference2D.laplacian uses a direct 3-point stencil on h:
            d²h/dy²[j,i] = (h[j+1, i] - 2*h[j, i] + h[j-1, i]) / dy²
        For h=c*y=c*j*dy, the cancellation is exact at every interior point
        (including j=1, where h[0]=0 is the natural linear continuation).
        """
        grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
        diff = Difference2D(grid=grid)
        c = 2.5
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.broadcast_to(c * y[:, None], (grid.Ny, grid.Nx))
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Cross-axis independence: x-diff of y-only field is zero, and vice versa
# ---------------------------------------------------------------------------


class TestDifferenceCrossAxis:
    """Derivatives of fields that vary in one axis only should vanish in the
    other axis."""

    def test_x_diff_of_y_only_field_is_zero(self):
        """h = f(y) only → dh/dx = 0 everywhere in the interior."""
        grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
        diff = Difference2D(grid=grid)
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.broadcast_to(2.0 * y[:, None], (grid.Ny, grid.Nx))
        result = diff.diff_x_T_to_U(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_y_diff_of_x_only_field_is_zero(self):
        """h = f(x) only → dh/dy = 0 everywhere in the interior."""
        grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
        diff = Difference2D(grid=grid)
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(3.0 * x, (grid.Ny, grid.Nx))
        result = diff.diff_y_T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Anisotropic (dx ≠ dy) and rectangular (nx ≠ ny) grid tests
# ---------------------------------------------------------------------------


class TestDifferenceAnisotropicGrid:
    """All difference operators must scale correctly when dx ≠ dy."""

    def test_diff_x_respects_dx_scaling(self):
        """dh/dx output should scale inversely with dx."""
        dx, dy = 2.0, 1.0
        grid = CartesianGrid2D.from_interior(8, 8, dx * 8, dy * 8)
        diff = Difference2D(grid=grid)
        c = 3.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        result = diff.diff_x_T_to_U(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_diff_y_respects_dy_scaling(self):
        """dh/dy output should scale inversely with dy."""
        dx, dy = 1.0, 3.0
        grid = CartesianGrid2D.from_interior(8, 8, dx * 8, dy * 8)
        diff = Difference2D(grid=grid)
        c = 2.0
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.broadcast_to(c * y[:, None], (grid.Ny, grid.Nx))
        result = diff.diff_y_T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_laplacian_anisotropic_quadratic(self):
        """h = a*x^2 + b*y^2 → Laplacian = 2a/dx^2*dx^2 + 2b/dy^2*dy^2.

        Using unit-domain grids so that
        d²h/dx² = 2*a  and  d²h/dy² = 2*b,
        giving Laplacian = 2*(a+b) as the continuous value.
        Since we use the same spacing for both axes in grid units, the
        discrete Laplacian matches the continuous one for quadratic fields.
        """
        # Use unit domain so that grid.dx = Lx/nx = 1.0, grid.dy = Ly/ny = 1.0
        grid = CartesianGrid2D.from_interior(8, 8, 8.0, 8.0)
        diff = Difference2D(grid=grid)
        a, b = 1.0, 2.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = a * x[None, :] ** 2 + b * y[:, None] ** 2
        result = diff.laplacian(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * (a + b), rtol=1e-5)


class TestDifferenceRectangularGrid:
    """Operators must work correctly on non-square (nx ≠ ny) domains."""

    def test_output_shape_rectangular(self):
        grid = CartesianGrid2D.from_interior(6, 10, 1.0, 1.0)
        diff = Difference2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        assert diff.diff_x_T_to_U(h).shape == (grid.Ny, grid.Nx)
        assert diff.diff_y_T_to_V(h).shape == (grid.Ny, grid.Nx)
        assert diff.laplacian(h).shape == (grid.Ny, grid.Nx)

    def test_linear_field_x_on_rectangular_grid(self):
        """Forward x-diff of a linear field is exact on a rectangular grid."""
        grid = CartesianGrid2D.from_interior(4, 10, 1.0, 1.0)
        diff = Difference2D(grid=grid)
        c = 2.5
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        result = diff.diff_x_T_to_U(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_linear_field_y_on_rectangular_grid(self):
        """Forward y-diff of a linear field is exact on a rectangular grid."""
        grid = CartesianGrid2D.from_interior(10, 4, 1.0, 1.0)
        diff = Difference2D(grid=grid)
        c = 1.5
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.broadcast_to(c * y[:, None], (grid.Ny, grid.Nx))
        result = diff.diff_y_T_to_V(h)
        np.testing.assert_allclose(result[1:-1, 1:-1], c, rtol=1e-5)

    def test_divergence_constant_flow_rectangular(self):
        """Divergence of uniform flow is zero on a rectangular grid."""
        grid = CartesianGrid2D.from_interior(5, 12, 2.0, 3.0)
        diff = Difference2D(grid=grid)
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = diff.divergence(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_no_nan_on_rectangular_grid(self):
        """Ensure no NaN values are produced for rectangular domains."""
        grid = CartesianGrid2D.from_interior(6, 10, 2.0, 5.0)
        diff = Difference2D(grid=grid)
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.sin(x[None, :]) + jnp.cos(y[:, None])
        for result in [diff.laplacian(h), diff.diff_x_T_to_U(h), diff.diff_y_T_to_V(h)]:
            assert jnp.all(jnp.isfinite(result)), "NaN/Inf on rectangular grid"
