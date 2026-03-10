"""Tests for Reconstruction1D, Reconstruction2D, Reconstruction3D."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.masks.cgrid_mask import ArakawaCGridMask
from finitevolx._src.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
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


class TestReconstruction1D:
    def test_naive_positive_flow(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.naive_x(h, u)
        np.testing.assert_allclose(result[1:-1], 1.0)

    def test_upwind1_positive_takes_upstream(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        # h[i] = i  =>  upwind with u>0 picks h[i]
        h = jnp.arange(grid1d.Nx, dtype=float)
        u = jnp.ones(grid1d.Nx)
        result = recon.upwind1_x(h, u)
        # fe[i+1/2] = h[i] * u = i * 1 = i
        np.testing.assert_allclose(result[1:-1], h[1:-1], rtol=1e-6)

    def test_upwind1_negative_takes_downstream(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.arange(grid1d.Nx, dtype=float)
        u = -jnp.ones(grid1d.Nx)
        result = recon.upwind1_x(h, u)
        # fe[i+1/2] = h[i+1] * u = (i+1) * (-1)
        np.testing.assert_allclose(result[1:-1], -h[2:], rtol=1e-6)

    def test_upwind2_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.upwind2_x(h, u).shape == (grid1d.Nx,)

    def test_upwind2_positive_flow(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        # Linear field h = i  =>  upwind2 positive should give h_face*u
        h = jnp.arange(grid1d.Nx, dtype=float)
        u = jnp.ones(grid1d.Nx)
        result = recon.upwind2_x(h, u)
        # For positive flow, fe[i+1/2] = (3/2*h[i] - 1/2*h[i-1]) * u[i+1/2]
        # result is indexed at [1:-1], corresponding to interior faces
        # expected_interior computes h_face values for faces [2:-1]
        expected = (1.5 * h[2:-1] - 0.5 * h[1:-2]) * u[2:-1]
        np.testing.assert_allclose(result[2:-1], expected, rtol=1e-5)

    def test_upwind2_negative_flow(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.arange(grid1d.Nx, dtype=float)
        u = -jnp.ones(grid1d.Nx)
        result = recon.upwind2_x(h, u)
        # For negative flow, fe[i+1/2] = (3/2*h[i+1] - 1/2*h[i+2]) * u[i+1/2]
        # Interior faces (except last) should use 2nd-order
        # Check a few interior values manually
        # result[1] should be (1.5*h[2] - 0.5*h[3]) * u[1] = (1.5*2 - 0.5*3) * (-1) = -1.5
        np.testing.assert_allclose(result[1], -1.5, rtol=1e-5)
        # Last interior face uses 1st-order fallback: h_face = h[i+1]
        expected_boundary = h[-1] * u[-2]
        np.testing.assert_allclose(result[-2], expected_boundary, rtol=1e-5)

    def test_upwind3_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.upwind3_x(h, u).shape == (grid1d.Nx,)

    def test_upwind3_positive_flow(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        # Quadratic field to test 3rd-order accuracy
        x = jnp.arange(grid1d.Nx, dtype=float)
        h = x**2
        u = jnp.ones(grid1d.Nx)
        result = recon.upwind3_x(h, u)
        # For positive flow, h_face = -1/6*h[i-1] + 5/6*h[i] + 1/3*h[i+1]
        expected = (-1.0 / 6.0 * h[:-2] + 5.0 / 6.0 * h[1:-1] + 1.0 / 3.0 * h[2:]) * u[
            1:-1
        ]
        np.testing.assert_allclose(result[1:-1], expected, rtol=1e-5)

    def test_upwind3_negative_flow(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        x = jnp.arange(grid1d.Nx, dtype=float)
        h = x**2
        u = -jnp.ones(grid1d.Nx)
        result = recon.upwind3_x(h, u)
        # For negative flow, interior uses 3rd-order, boundary uses 1st-order
        # Interior (except last): h_face = 1/3*h[i] + 5/6*h[i+1] - 1/6*h[i+2]
        expected_interior = (
            1.0 / 3.0 * h[1:-2] + 5.0 / 6.0 * h[2:-1] - 1.0 / 6.0 * h[3:]
        ) * u[1:-2]
        np.testing.assert_allclose(result[1:-2], expected_interior, rtol=1e-5)
        # Last interior face uses 1st-order fallback: h_face = h[i+1]
        expected_boundary = h[-1] * u[-2]
        np.testing.assert_allclose(result[-2], expected_boundary, rtol=1e-5)

    def test_ghost_zero(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.naive_x(h, u)
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[-1], 0.0)

    # --- WENO-3 tests ---

    def test_weno3_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.weno3_x(h, u).shape == (grid1d.Nx,)

    def test_weno3_constant_field(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = 3.0 * jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.weno3_x(h, u)
        # Constant field => WENO collapses to h*u = 3
        np.testing.assert_allclose(result[1:-1], 3.0, rtol=1e-5)

    def test_weno3_ghost_zero(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.weno3_x(h, u)
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[-1], 0.0)

    def test_weno3_both_flow_directions(self, grid1d):
        # Constant field: WENO-3 must give h*u exactly for both positive and negative flow
        recon = Reconstruction1D(grid=grid1d)
        h = 3.0 * jnp.ones(grid1d.Nx)
        for vel in (jnp.ones(grid1d.Nx), -jnp.ones(grid1d.Nx)):
            weno = recon.weno3_x(h, vel)
            upw3 = recon.upwind3_x(h, vel)
            np.testing.assert_allclose(weno[1:-1], upw3[1:-1], rtol=1e-5)

    # --- WENO-Z-3 tests ---

    def test_wenoz3_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.wenoz3_x(h, u).shape == (grid1d.Nx,)

    def test_wenoz3_constant_field(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = 5.0 * jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.wenoz3_x(h, u)
        np.testing.assert_allclose(result[1:-1], 5.0, rtol=1e-5)

    def test_wenoz3_negative_flow_constant(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = 5.0 * jnp.ones(grid1d.Nx)
        u = -jnp.ones(grid1d.Nx)
        result = recon.wenoz3_x(h, u)
        np.testing.assert_allclose(result[1:-1], -5.0, rtol=1e-5)

    # --- WENO-5 tests ---

    def test_weno5_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.weno5_x(h, u).shape == (grid1d.Nx,)

    def test_weno5_constant_field(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = 2.0 * jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.weno5_x(h, u)
        np.testing.assert_allclose(result[1:-1], 2.0, rtol=1e-5)

    def test_weno5_ghost_zero(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.weno5_x(h, u)
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[-1], 0.0)

    def test_weno5_negative_flow_constant(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = 4.0 * jnp.ones(grid1d.Nx)
        u = -jnp.ones(grid1d.Nx)
        result = recon.weno5_x(h, u)
        np.testing.assert_allclose(result[1:-1], -4.0, rtol=1e-5)

    # --- WENO-Z-5 tests ---

    def test_wenoz5_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.wenoz5_x(h, u).shape == (grid1d.Nx,)

    def test_wenoz5_constant_field(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = 7.0 * jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.wenoz5_x(h, u)
        np.testing.assert_allclose(result[1:-1], 7.0, rtol=1e-5)

    def test_wenoz5_negative_flow_constant(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = 7.0 * jnp.ones(grid1d.Nx)
        u = -jnp.ones(grid1d.Nx)
        result = recon.wenoz5_x(h, u)
        np.testing.assert_allclose(result[1:-1], -7.0, rtol=1e-5)

    @pytest.mark.parametrize("method_name", ["weno7_x", "weno9_x"])
    def test_higher_order_weno_constant_field(self, grid1d, method_name):
        recon = Reconstruction1D(grid=grid1d)
        h = 6.0 * jnp.ones(grid1d.Nx)
        for sign in (1.0, -1.0):
            u = sign * jnp.ones(grid1d.Nx)
            result = getattr(recon, method_name)(h, u)
            np.testing.assert_allclose(result[1:-1], 6.0 * sign, rtol=1e-5)

    def test_weno7_linear_field_large_grid(self):
        grid = ArakawaCGrid1D.from_interior(12, 1.0)
        recon = Reconstruction1D(grid=grid)
        h = jnp.arange(grid.Nx, dtype=float)
        u = jnp.ones(grid.Nx)
        result = recon.weno7_x(h, u)
        expected = 0.5 * (h[1:-1] + h[2:])
        np.testing.assert_allclose(result[3:-3], expected[2:-2], rtol=1e-5)

    def test_weno9_linear_field_large_grid(self):
        grid = ArakawaCGrid1D.from_interior(14, 1.0)
        recon = Reconstruction1D(grid=grid)
        h = jnp.arange(grid.Nx, dtype=float)
        u = jnp.ones(grid.Nx)
        result = recon.weno9_x(h, u)
        expected = 0.5 * (h[1:-1] + h[2:])
        np.testing.assert_allclose(result[4:-4], expected[3:-3], rtol=1e-5)

    # --- TVD flux-limiter tests ---

    def test_tvd_output_shape(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        assert recon.tvd_x(h, u).shape == (grid1d.Nx,)

    def test_tvd_constant_field_all_limiters(self, grid1d):
        """Constant field: all limiters must give h*u exactly."""
        recon = Reconstruction1D(grid=grid1d)
        h = 3.0 * jnp.ones(grid1d.Nx)
        for sign in (1.0, -1.0):
            u = sign * jnp.ones(grid1d.Nx)
            for lim in ("minmod", "van_leer", "superbee", "mc"):
                result = recon.tvd_x(h, u, limiter=lim)
                np.testing.assert_allclose(result[1:-1], 3.0 * sign, rtol=1e-5)

    def test_tvd_ghost_zero(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        result = recon.tvd_x(h, u)
        np.testing.assert_allclose(result[0], 0.0)
        np.testing.assert_allclose(result[-1], 0.0)

    def test_tvd_positive_flow_linear_field(self, grid1d):
        """Linear h: TVD limiter φ(1)=1 gives 2nd-order result (h[i]+h[i+1])/2."""
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.arange(grid1d.Nx, dtype=float)
        u = jnp.ones(grid1d.Nx)
        for lim in ("minmod", "van_leer", "superbee", "mc"):
            result = recon.tvd_x(h, u, limiter=lim)
            # Interior (not boundary): r=1 so phi=1, h_face = h[i] + 0.5*(h[i+1]-h[i])
            # = (h[i]+h[i+1])/2 which is centred
            expected = 0.5 * (h[1:-1] + h[2:]) * u[1:-1]
            np.testing.assert_allclose(result[1:-1], expected, rtol=1e-5)

    def test_tvd_unknown_limiter_raises(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)
        with pytest.raises(ValueError, match="Unknown flux limiter"):
            recon.tvd_x(h, u, limiter="bogus")


class TestReconstruction2D:
    def test_naive_x_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.naive_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0)

    def test_naive_y_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.naive_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 3.0)

    def test_upwind1_x_positive(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        # uniform h, positive u => flux = h * u
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.upwind1_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 8.0)

    def test_upwind1_y_negative(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.upwind1_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], -1.0)

    def test_upwind2_x_shape(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert recon.upwind2_x(h, u).shape == (grid2d.Ny, grid2d.Nx)

    def test_upwind3_x_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.upwind3_x(h, u)
        # constant h => all stencils give h * u = 5
        np.testing.assert_allclose(result[1:-1, 1:-1], 5.0, rtol=1e-5)

    def test_ghost_zero(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.naive_x(h, u)
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)

    # --- WENO-3 tests ---

    def test_weno3_x_output_shape(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert recon.weno3_x(h, u).shape == (grid2d.Ny, grid2d.Nx)

    def test_weno3_x_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno3_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 3.0, rtol=1e-5)

    def test_weno3_x_negative_flow_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno3_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], -3.0, rtol=1e-5)

    def test_weno3_y_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno3_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 4.0, rtol=1e-5)

    def test_weno3_y_negative_flow_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno3_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], -4.0, rtol=1e-5)

    def test_weno3_x_ghost_zero(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno3_x(h, u)
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)

    # --- WENO-Z-3 tests ---

    def test_wenoz3_x_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 6.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz3_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 6.0, rtol=1e-5)

    def test_wenoz3_x_negative_flow_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 6.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz3_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], -6.0, rtol=1e-5)

    def test_wenoz3_y_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 7.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz3_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 7.0, rtol=1e-5)

    def test_wenoz3_y_negative_flow_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 7.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz3_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], -7.0, rtol=1e-5)

    # --- WENO-5 tests ---

    def test_weno5_x_output_shape(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert recon.weno5_x(h, u).shape == (grid2d.Ny, grid2d.Nx)

    def test_weno5_x_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0, rtol=1e-5)

    def test_weno5_x_negative_flow_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], -2.0, rtol=1e-5)

    def test_weno5_y_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 5.0, rtol=1e-5)

    def test_weno5_y_negative_flow_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], -5.0, rtol=1e-5)

    # --- WENO-Z-5 tests ---

    def test_wenoz5_x_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 8.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz5_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 8.0, rtol=1e-5)

    def test_wenoz5_x_negative_flow_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 8.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz5_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1], -8.0, rtol=1e-5)

    def test_wenoz5_y_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 9.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz5_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 9.0, rtol=1e-5)

    def test_wenoz5_y_negative_flow_constant(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 9.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz5_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], -9.0, rtol=1e-5)

    @pytest.mark.parametrize(
        "method_name", ["weno7_x", "weno7_y", "weno9_x", "weno9_y"]
    )
    def test_higher_order_weno_constant(self, grid2d, method_name):
        recon = Reconstruction2D(grid=grid2d)
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        velocity = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = getattr(recon, method_name)(h, velocity)
        np.testing.assert_allclose(result[1:-1, 1:-1], 4.0, rtol=1e-5)
        negative = getattr(recon, method_name)(h, -velocity)
        np.testing.assert_allclose(negative[1:-1, 1:-1], -4.0, rtol=1e-5)

    @pytest.mark.parametrize(
        ("method_name", "depth"),
        [("weno7_x", 3), ("weno9_x", 4)],
    )
    def test_higher_order_weno_x_linear_ramp_is_centered(self, method_name, depth):
        grid = ArakawaCGrid2D.from_interior(12, 12, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        x_idx = jnp.arange(grid.Nx, dtype=float)
        h = jnp.broadcast_to(x_idx, (grid.Ny, grid.Nx))
        u = jnp.ones((grid.Ny, grid.Nx))
        result = getattr(recon, method_name)(h, u)
        expected = 0.5 * (h[1:-1, depth:-depth] + h[1:-1, depth + 1 : -(depth - 1)])
        np.testing.assert_allclose(result[1:-1, depth:-depth], expected, rtol=1e-5)

    @pytest.mark.parametrize(
        ("method_name", "depth"),
        [("weno7_y", 3), ("weno9_y", 4)],
    )
    def test_higher_order_weno_y_linear_ramp_is_centered(self, method_name, depth):
        grid = ArakawaCGrid2D.from_interior(12, 12, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        y_idx = jnp.arange(grid.Ny, dtype=float)
        h = jnp.broadcast_to(y_idx[:, None], (grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = getattr(recon, method_name)(h, v)
        expected = 0.5 * (h[depth:-depth, 1:-1] + h[depth + 1 : -(depth - 1), 1:-1])
        np.testing.assert_allclose(result[depth:-depth, 1:-1], expected, rtol=1e-5)

    # --- TVD flux-limiter tests ---

    def test_tvd_x_output_shape(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        assert recon.tvd_x(h, u).shape == (grid2d.Ny, grid2d.Nx)

    def test_tvd_x_constant_field_all_limiters(self, grid2d):
        """Constant field: all limiters must give h*u exactly."""
        recon = Reconstruction2D(grid=grid2d)
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        for sign in (1.0, -1.0):
            u = sign * jnp.ones((grid2d.Ny, grid2d.Nx))
            for lim in ("minmod", "van_leer", "superbee", "mc"):
                result = recon.tvd_x(h, u, limiter=lim)
                np.testing.assert_allclose(result[1:-1, 1:-1], 4.0 * sign, rtol=1e-5)

    def test_tvd_y_constant_field_all_limiters(self, grid2d):
        """Constant field: all limiters must give h*v exactly."""
        recon = Reconstruction2D(grid=grid2d)
        h = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        for sign in (1.0, -1.0):
            v = sign * jnp.ones((grid2d.Ny, grid2d.Nx))
            for lim in ("minmod", "van_leer", "superbee", "mc"):
                result = recon.tvd_y(h, v, limiter=lim)
                np.testing.assert_allclose(result[1:-1, 1:-1], 5.0 * sign, rtol=1e-5)

    def test_tvd_x_ghost_ring_zero(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.tvd_x(h, u)
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, -1], 0.0)

    def test_tvd_x_linear_field_is_centered(self, grid2d):
        """Linear h along x: TVD with φ(1)=1 must recover the centred average."""
        recon = Reconstruction2D(grid=grid2d)
        x_idx = jnp.arange(grid2d.Nx, dtype=float)
        h = jnp.broadcast_to(x_idx, (grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        for lim in ("minmod", "van_leer", "superbee", "mc"):
            result = recon.tvd_x(h, u, limiter=lim)
            # For a linear ramp, r=1, phi(1)=1 for all TVD limiters,
            # so h_face = (h[i]+h[i+1])/2.
            expected = 0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:]) * u[1:-1, 1:-1]
            np.testing.assert_allclose(result[1:-1, 1:-1], expected, rtol=1e-5)


class TestReconstruction3D:
    def test_naive_x_shape(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert recon.naive_x(h, u).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_upwind1_x_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 3.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.upwind1_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 3.0)

    def test_upwind1_y_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 2.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.upwind1_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 2.0)

    # --- WENO-3 tests ---

    def test_weno3_x_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 4.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno3_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 4.0, rtol=1e-5)

    def test_weno3_x_negative_flow_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 4.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno3_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -4.0, rtol=1e-5)

    def test_weno3_y_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 5.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno3_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 5.0, rtol=1e-5)

    def test_weno3_y_negative_flow_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 5.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno3_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -5.0, rtol=1e-5)

    def test_weno3_x_shape(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert recon.weno3_x(h, u).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    # --- WENO-Z-3 tests ---

    def test_wenoz3_x_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 6.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.wenoz3_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 6.0, rtol=1e-5)

    def test_wenoz3_x_negative_flow_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 6.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.wenoz3_x(h, u)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -6.0, rtol=1e-5)

    def test_wenoz3_y_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 7.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.wenoz3_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 7.0, rtol=1e-5)

    def test_wenoz3_y_negative_flow_constant(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 7.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.wenoz3_y(h, v)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -7.0, rtol=1e-5)

    @pytest.mark.parametrize(
        "method_name",
        ["weno7_x", "weno7_y", "weno9_x", "weno9_y"],
    )
    def test_higher_order_weno_constant(self, grid3d, method_name):
        recon = Reconstruction3D(grid=grid3d)
        h = 4.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        velocity = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = getattr(recon, method_name)(h, velocity)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 4.0, rtol=1e-5)
        negative = getattr(recon, method_name)(h, -velocity)
        np.testing.assert_allclose(negative[1:-1, 1:-1, 1:-1], -4.0, rtol=1e-5)

    @pytest.mark.parametrize(
        ("method_name", "depth"),
        [("weno7_x", 3), ("weno9_x", 4)],
    )
    def test_higher_order_weno_x_linear_ramp_is_centered(self, method_name, depth):
        grid = ArakawaCGrid3D.from_interior(12, 12, 4, 1.0, 1.0, 1.0)
        recon = Reconstruction3D(grid=grid)
        x_idx = jnp.arange(grid.Nx, dtype=float)
        h = jnp.broadcast_to(x_idx[None, None, :], (grid.Nz, grid.Ny, grid.Nx))
        u = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        result = getattr(recon, method_name)(h, u)
        expected = 0.5 * (
            h[1:-1, 1:-1, depth:-depth] + h[1:-1, 1:-1, depth + 1 : -(depth - 1)]
        )
        np.testing.assert_allclose(
            result[1:-1, 1:-1, depth:-depth], expected, rtol=1e-5
        )

    @pytest.mark.parametrize(
        ("method_name", "depth"),
        [("weno7_y", 3), ("weno9_y", 4)],
    )
    def test_higher_order_weno_y_linear_ramp_is_centered(self, method_name, depth):
        grid = ArakawaCGrid3D.from_interior(12, 12, 4, 1.0, 1.0, 1.0)
        recon = Reconstruction3D(grid=grid)
        y_idx = jnp.arange(grid.Ny, dtype=float)
        h = jnp.broadcast_to(y_idx[None, :, None], (grid.Nz, grid.Ny, grid.Nx))
        v = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
        result = getattr(recon, method_name)(h, v)
        expected = 0.5 * (
            h[1:-1, depth:-depth, 1:-1] + h[1:-1, depth + 1 : -(depth - 1), 1:-1]
        )
        np.testing.assert_allclose(
            result[1:-1, depth:-depth, 1:-1], expected, rtol=1e-5
        )

    # --- TVD flux-limiter tests ---

    def test_tvd_x_output_shape(self, grid3d):
        recon = Reconstruction3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert recon.tvd_x(h, u).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_tvd_x_constant_field_all_limiters(self, grid3d):
        """Constant field: all limiters must give h*u exactly."""
        recon = Reconstruction3D(grid=grid3d)
        h = 4.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for sign in (1.0, -1.0):
            u = sign * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
            for lim in ("minmod", "van_leer", "superbee", "mc"):
                result = recon.tvd_x(h, u, limiter=lim)
                np.testing.assert_allclose(
                    result[1:-1, 1:-1, 1:-1], 4.0 * sign, rtol=1e-5
                )

    def test_tvd_y_constant_field_all_limiters(self, grid3d):
        """Constant field: all limiters must give h*v exactly."""
        recon = Reconstruction3D(grid=grid3d)
        h = 5.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        for sign in (1.0, -1.0):
            v = sign * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
            for lim in ("minmod", "van_leer", "superbee", "mc"):
                result = recon.tvd_y(h, v, limiter=lim)
                np.testing.assert_allclose(
                    result[1:-1, 1:-1, 1:-1], 5.0 * sign, rtol=1e-5
                )


# ──────────────────────────────────────────────────────────────────────────────
# Mask-aware reconstruction tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def all_ocean_mask_2d():
    """All-ocean (no coastlines) 10x10 mask — masked methods match non-masked."""
    return ArakawaCGridMask.from_dimensions(10, 10)


@pytest.fixture
def coastal_mask_2d():
    """10x10 mask with a 2-cell-wide land block in the centre.

    The land block forces near-boundary cells to use lower-order stencils.
    """
    import numpy as np

    h = np.ones((10, 10), dtype=bool)
    # Place land at interior columns 4-5 (all rows) to create a barrier
    h[:, 4:6] = False
    return ArakawaCGridMask.from_mask(h)


@pytest.fixture
def all_ocean_mask_3d():
    """All-ocean mask for 3D tests (8x8 horizontal grid)."""
    return ArakawaCGridMask.from_dimensions(8, 8)


class TestReconstruction2DMasked:
    # --- weno5_x_masked ---

    def test_weno5_x_masked_output_shape(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_x_masked(h, u, all_ocean_mask_2d)
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_weno5_x_masked_all_ocean_matches_weno5(self, grid2d, all_ocean_mask_2d):
        """All-ocean mask: masked method must equal unmasked weno5_x."""
        recon = Reconstruction2D(grid=grid2d)
        h = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        masked = recon.weno5_x_masked(h, u, all_ocean_mask_2d)
        unmasked = recon.weno5_x(h, u)
        np.testing.assert_allclose(masked[1:-1, 1:-1], unmasked[1:-1, 1:-1], rtol=1e-5)

    def test_weno5_x_masked_constant_field(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_x_masked(h, u, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0, rtol=1e-5)

    def test_weno5_x_masked_negative_flow_constant(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_x_masked(h, u, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], -2.0, rtol=1e-5)

    def test_weno5_x_masked_ghost_zero(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_x_masked(h, u, all_ocean_mask_2d)
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)

    def test_weno5_x_masked_coastal_fallback(self, coastal_mask_2d):
        """Coastal mask: near-land cells use lower-order stencil with non-constant field.

        Verifies that (1) the masked reconstruction produces finite values everywhere,
        and (2) cells immediately adjacent to the land block produce different flux
        values from the unmasked WENO-5 for both flow signs, confirming that stencil
        fallback is actually being applied near the coastline.
        """
        grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)

        # Non-constant field varying in x so different stencils produce distinct values.
        x_indices = jnp.arange(grid.Nx, dtype=float)
        h = jnp.broadcast_to(x_indices, (grid.Ny, grid.Nx))

        for sign in (1.0, -1.0):
            u = sign * jnp.ones((grid.Ny, grid.Nx))

            ref = recon.weno5_x(h, u)
            masked = recon.weno5_x_masked(h, u, coastal_mask_2d)

            assert jnp.all(jnp.isfinite(masked)).item()

            # Columns 3 and 6 (0-based) are immediately adjacent to the land block
            # (land at cols 4-5) and lack sufficient stencil for even WENO3, so the
            # mask forces 1st-order upwind. For a non-constant h this differs from
            # the WENO-5 result produced by the unmasked method.
            row_slice = slice(2, 8)
            diffs_found = jnp.any(masked[row_slice, 3] != ref[row_slice, 3]) | jnp.any(
                masked[row_slice, 6] != ref[row_slice, 6]
            )
            assert diffs_found.item()

    # --- weno5_y_masked ---

    def test_weno5_y_masked_constant_field(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_y_masked(h, v, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], 5.0, rtol=1e-5)

    def test_weno5_y_masked_negative_flow_constant(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 5.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_y_masked(h, v, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], -5.0, rtol=1e-5)

    def test_weno5_y_masked_all_ocean_matches_weno5(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        masked = recon.weno5_y_masked(h, v, all_ocean_mask_2d)
        unmasked = recon.weno5_y(h, v)
        np.testing.assert_allclose(masked[1:-1, 1:-1], unmasked[1:-1, 1:-1], rtol=1e-5)

    # --- wenoz5_x_masked ---

    def test_wenoz5_x_masked_constant_field(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 8.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz5_x_masked(h, u, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], 8.0, rtol=1e-5)

    def test_wenoz5_x_masked_negative_flow_constant(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 8.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz5_x_masked(h, u, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], -8.0, rtol=1e-5)

    def test_wenoz5_x_masked_all_ocean_matches_wenoz5(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 7.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        masked = recon.wenoz5_x_masked(h, u, all_ocean_mask_2d)
        unmasked = recon.wenoz5_x(h, u)
        np.testing.assert_allclose(masked[1:-1, 1:-1], unmasked[1:-1, 1:-1], rtol=1e-5)

    # --- wenoz5_y_masked ---

    def test_wenoz5_y_masked_constant_field(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 9.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz5_y_masked(h, v, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], 9.0, rtol=1e-5)

    def test_wenoz5_y_masked_negative_flow_constant(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 9.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.wenoz5_y_masked(h, v, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], -9.0, rtol=1e-5)

    def test_wenoz5_y_masked_coastal_fallback(self, coastal_mask_2d):
        grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = recon.wenoz5_y_masked(h, v, coastal_mask_2d)
        assert jnp.all(jnp.isfinite(result)).item()

    # --- tvd_x_masked and tvd_y_masked ---

    def test_tvd_x_masked_output_shape(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.tvd_x_masked(h, u, all_ocean_mask_2d)
        assert result.shape == (grid2d.Ny, grid2d.Nx)

    def test_tvd_x_masked_constant_field(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.tvd_x_masked(h, u, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], 3.0, rtol=1e-5)

    def test_tvd_x_masked_negative_flow_constant(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.tvd_x_masked(h, u, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], -3.0, rtol=1e-5)

    def test_tvd_x_masked_all_ocean_matches_tvd(self, grid2d, all_ocean_mask_2d):
        """All-ocean mask: masked TVD must equal unmasked tvd_x."""
        recon = Reconstruction2D(grid=grid2d)
        h = 3.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        masked = recon.tvd_x_masked(h, u, all_ocean_mask_2d)
        unmasked = recon.tvd_x(h, u)
        np.testing.assert_allclose(masked[1:-1, 1:-1], unmasked[1:-1, 1:-1], rtol=1e-5)

    def test_tvd_y_masked_constant_field(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.tvd_y_masked(h, v, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], 4.0, rtol=1e-5)

    def test_tvd_y_masked_negative_flow_constant(self, grid2d, all_ocean_mask_2d):
        recon = Reconstruction2D(grid=grid2d)
        h = 4.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.tvd_y_masked(h, v, all_ocean_mask_2d)
        np.testing.assert_allclose(result[1:-1, 1:-1], -4.0, rtol=1e-5)

    def test_tvd_x_masked_coastal_fallback(self, coastal_mask_2d):
        """Near-land cells fall back to upwind1 for TVD masked."""
        grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
        recon = Reconstruction2D(grid=grid)
        x_indices = jnp.arange(grid.Nx, dtype=float)
        h = jnp.broadcast_to(x_indices, (grid.Ny, grid.Nx))
        for sign in (1.0, -1.0):
            u = sign * jnp.ones((grid.Ny, grid.Nx))
            masked = recon.tvd_x_masked(h, u, coastal_mask_2d)
            assert jnp.all(jnp.isfinite(masked)).item()


class TestReconstruction3DMasked:
    # --- weno5_x_masked ---

    def test_weno5_x_masked_output_shape(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno5_x_masked(h, u, all_ocean_mask_3d)
        assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_weno5_x_masked_constant_field(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 4.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno5_x_masked(h, u, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 4.0, rtol=1e-5)

    def test_weno5_x_masked_negative_flow_constant(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 4.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno5_x_masked(h, u, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -4.0, rtol=1e-5)

    # --- weno5_y_masked ---

    def test_weno5_y_masked_constant_field(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 5.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno5_y_masked(h, v, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 5.0, rtol=1e-5)

    def test_weno5_y_masked_negative_flow_constant(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 5.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.weno5_y_masked(h, v, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -5.0, rtol=1e-5)

    # --- wenoz5_x_masked ---

    def test_wenoz5_x_masked_constant_field(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 6.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.wenoz5_x_masked(h, u, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 6.0, rtol=1e-5)

    def test_wenoz5_x_masked_negative_flow_constant(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 6.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.wenoz5_x_masked(h, u, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -6.0, rtol=1e-5)

    # --- wenoz5_y_masked ---

    def test_wenoz5_y_masked_constant_field(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 7.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.wenoz5_y_masked(h, v, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 7.0, rtol=1e-5)

    def test_wenoz5_y_masked_negative_flow_constant(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 7.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.wenoz5_y_masked(h, v, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -7.0, rtol=1e-5)

    # --- tvd_x_masked and tvd_y_masked ---

    def test_tvd_x_masked_output_shape(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.tvd_x_masked(h, u, all_ocean_mask_3d)
        assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_tvd_x_masked_constant_field(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 4.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.tvd_x_masked(h, u, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 4.0, rtol=1e-5)

    def test_tvd_x_masked_negative_flow_constant(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 4.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        u = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.tvd_x_masked(h, u, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -4.0, rtol=1e-5)

    def test_tvd_y_masked_constant_field(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 5.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.tvd_y_masked(h, v, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 5.0, rtol=1e-5)

    def test_tvd_y_masked_negative_flow_constant(self, grid3d, all_ocean_mask_3d):
        recon = Reconstruction3D(grid=grid3d)
        h = 5.0 * jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = -jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = recon.tvd_y_masked(h, v, all_ocean_mask_3d)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], -5.0, rtol=1e-5)
