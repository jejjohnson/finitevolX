"""Tests for the pure functional semi-Lagrangian advection step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import semi_lagrangian_step

jax.config.update("jax_enable_x64", True)


class TestSemiLagrangianStep:
    """Tests for semi_lagrangian_step."""

    @pytest.fixture()
    def gaussian_field(self):
        """A Gaussian bump centred at (Ny/2, Nx/2)."""
        nx, ny = 64, 64
        x = jnp.arange(nx, dtype=jnp.float64)
        y = jnp.arange(ny, dtype=jnp.float64)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        # Note: meshgrid with "ij" gives shape (nx, ny); we want (ny, nx)
        X, Y = X.T, Y.T
        cx, cy = nx / 2.0, ny / 2.0
        sigma = 5.0
        return jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))

    def test_uniform_translation(self, gaussian_field):
        """Uniform velocity shifts the field; result should resemble shifted original."""
        ny, nx = gaussian_field.shape
        dx = dy = 1.0
        u_phys = 1.0  # 1 grid cell per second in x
        v_phys = 0.0
        dt = 5.0  # shift 5 cells

        u = jnp.full((ny, nx), u_phys)
        v = jnp.full((ny, nx), v_phys)

        result = semi_lagrangian_step(
            gaussian_field, u, v, dx, dy, dt, interp_order=1, bc="periodic"
        )

        # u > 0 advects the field to the right: new[i,j] = old[i, j - shift]
        shifted = jnp.roll(gaussian_field, 5, axis=1)
        # Integer-cell shift with linear interp on grid-aligned points is exact
        np.testing.assert_allclose(result, shifted, atol=1e-10)

    def test_uniform_translation_negative_velocity(self, gaussian_field):
        """Negative velocity shifts the field to the left."""
        ny, nx = gaussian_field.shape
        u = jnp.full((ny, nx), -3.0)
        v = jnp.zeros((ny, nx))

        result = semi_lagrangian_step(
            gaussian_field, u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
        )
        shifted = jnp.roll(gaussian_field, -3, axis=1)
        np.testing.assert_allclose(result, shifted, atol=1e-10)

    def test_diagonal_translation(self, gaussian_field):
        """Diagonal (u, v) both nonzero, integer shift."""
        ny, nx = gaussian_field.shape
        u = jnp.full((ny, nx), 2.0)
        v = jnp.full((ny, nx), 3.0)

        result = semi_lagrangian_step(
            gaussian_field, u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
        )
        shifted = jnp.roll(jnp.roll(gaussian_field, 2, axis=1), 3, axis=0)
        np.testing.assert_allclose(result, shifted, atol=1e-10)

    def test_zero_velocity_preserves_field(self, gaussian_field):
        """Zero velocity should return the same field."""
        ny, nx = gaussian_field.shape
        u = jnp.zeros((ny, nx))
        v = jnp.zeros((ny, nx))
        result = semi_lagrangian_step(
            gaussian_field, u, v, 1.0, 1.0, 1.0, interp_order=1
        )
        np.testing.assert_allclose(result, gaussian_field, atol=1e-10)

    def test_cfl_greater_than_one(self, gaussian_field):
        """Semi-Lagrangian should remain stable even with CFL > 1."""
        ny, nx = gaussian_field.shape
        dx = dy = 1.0
        u_phys = 5.0  # CFL = 5
        dt = 1.0

        u = jnp.full((ny, nx), u_phys)
        v = jnp.zeros((ny, nx))

        result = semi_lagrangian_step(
            gaussian_field, u, v, dx, dy, dt, interp_order=1, bc="periodic"
        )
        # Should not blow up
        assert jnp.all(jnp.isfinite(result))
        # Maximum should not exceed original (monotone for order=1)
        assert float(jnp.max(result)) <= float(jnp.max(gaussian_field)) + 1e-10

    def test_fractional_shift_has_diffusion(self, gaussian_field):
        """Half-cell shift with linear interp should cause some diffusion."""
        ny, nx = gaussian_field.shape
        u = jnp.full((ny, nx), 0.5)  # half-cell shift
        v = jnp.zeros((ny, nx))

        result = semi_lagrangian_step(
            gaussian_field, u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
        )
        # Peak should be lower due to diffusive interpolation
        assert float(jnp.max(result)) < float(jnp.max(gaussian_field))
        # But field should still be non-negative (monotone property)
        assert float(jnp.min(result)) >= -1e-10

    def test_edge_boundary_mode(self, gaussian_field):
        """bc='edge' should clamp rather than wrap."""
        ny, nx = gaussian_field.shape
        # Large shift pushes departure points outside domain
        u = jnp.full((ny, nx), 100.0)
        v = jnp.zeros((ny, nx))

        result_periodic = semi_lagrangian_step(
            gaussian_field, u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
        )
        result_edge = semi_lagrangian_step(
            gaussian_field, u, v, 1.0, 1.0, 1.0, interp_order=1, bc="edge"
        )
        # The two BC modes should produce different results
        assert not jnp.allclose(result_periodic, result_edge)
        # Both should be finite
        assert jnp.all(jnp.isfinite(result_periodic))
        assert jnp.all(jnp.isfinite(result_edge))

    def test_convergence_fractional_shift(self):
        """Error from fractional shift should decrease as field becomes smoother."""
        ny, nx = 64, 64
        x = jnp.arange(nx, dtype=jnp.float64)
        y = jnp.arange(ny, dtype=jnp.float64)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T

        # Very smooth field (wide Gaussian): less interpolation error
        field_smooth = jnp.exp(-((X - 32) ** 2 + (Y - 32) ** 2) / 200.0)
        # Less smooth (narrow Gaussian): more interpolation error
        field_narrow = jnp.exp(-((X - 32) ** 2 + (Y - 32) ** 2) / 10.0)

        u = jnp.full((ny, nx), 0.5)
        v = jnp.zeros((ny, nx))

        # Two forward + backward steps should approximate identity
        for field in [field_smooth, field_narrow]:
            result = semi_lagrangian_step(
                field, u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
            )
            result_back = semi_lagrangian_step(
                result, -u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
            )
            assert jnp.all(jnp.isfinite(result_back))

        # Smooth field should have less round-trip error
        rt_smooth = semi_lagrangian_step(
            field_smooth, u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
        )
        rt_smooth = semi_lagrangian_step(
            rt_smooth, -u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
        )
        rt_narrow = semi_lagrangian_step(
            field_narrow, u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
        )
        rt_narrow = semi_lagrangian_step(
            rt_narrow, -u, v, 1.0, 1.0, 1.0, interp_order=1, bc="periodic"
        )

        err_smooth = float(jnp.max(jnp.abs(rt_smooth - field_smooth)))
        err_narrow = float(jnp.max(jnp.abs(rt_narrow - field_narrow)))
        assert err_smooth < err_narrow

    def test_jit_compatible(self, gaussian_field):
        ny, nx = gaussian_field.shape
        u = jnp.zeros((ny, nx))
        v = jnp.zeros((ny, nx))
        step_jit = jax.jit(lambda f: semi_lagrangian_step(f, u, v, 1.0, 1.0, 0.1))
        result = step_jit(gaussian_field)
        np.testing.assert_allclose(result, gaussian_field, atol=1e-10)

    def test_output_shape(self, gaussian_field):
        """Output has the same shape as input."""
        ny, nx = gaussian_field.shape
        u = jnp.ones((ny, nx))
        v = jnp.zeros((ny, nx))
        result = semi_lagrangian_step(
            gaussian_field, u, v, 1.0, 1.0, 0.5, interp_order=1
        )
        assert result.shape == gaussian_field.shape

    def test_order_zero_nearest_neighbor(self, gaussian_field):
        """interp_order=0 should do nearest-neighbor interpolation."""
        ny, nx = gaussian_field.shape
        # Integer shift: should be exact even with order=0
        u = jnp.full((ny, nx), 3.0)
        v = jnp.zeros((ny, nx))

        result = semi_lagrangian_step(
            gaussian_field, u, v, 1.0, 1.0, 1.0, interp_order=0, bc="periodic"
        )
        shifted = jnp.roll(gaussian_field, 3, axis=1)
        np.testing.assert_allclose(result, shifted, atol=1e-10)
