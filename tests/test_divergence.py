"""Tests for Divergence2D and divergence_2d."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.difference import Difference2D
from finitevolx._src.divergence import Divergence2D, divergence_2d
from finitevolx._src.grid import ArakawaCGrid2D

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def grid():
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def div_op(grid):
    return Divergence2D(grid=grid)


# ---------------------------------------------------------------------------
# Functional API: divergence_2d
# ---------------------------------------------------------------------------


class TestDivergence2DFunctional:
    def test_output_shape(self, grid):
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        result = divergence_2d(u, v, dx=grid.dx, dy=grid.dy)
        assert result.shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero_divergence(self, grid):
        """Constant velocity field has zero divergence."""
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = divergence_2d(u, v, dx=grid.dx, dy=grid.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_field_exact_divergence(self, grid):
        """For u=c*x, v=c*y, divergence = 2c exactly at any resolution."""
        c = 2.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        u = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        v = jnp.broadcast_to(c * y[:, None], (grid.Ny, grid.Nx))
        result = divergence_2d(u, v, dx=grid.dx, dy=grid.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c, atol=1e-8)

    def test_ghost_ring_is_zero(self, grid):
        """Ghost ring in the output is always zero."""
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = divergence_2d(u, v, dx=grid.dx, dy=grid.dy)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[:, -1], 0.0, atol=1e-10)

    def test_anisotropic_divergence(self, grid):
        """u varies in x, v is zero → divergence = c_x exactly."""
        c_x = 3.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        u = jnp.broadcast_to(c_x * x, (grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        result = divergence_2d(u, v, dx=grid.dx, dy=grid.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], c_x, atol=1e-8)


# ---------------------------------------------------------------------------
# Class-based API: Divergence2D
# ---------------------------------------------------------------------------


class TestDivergence2DClass:
    def test_output_shape(self, div_op, grid):
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        assert div_op(u, v).shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero_divergence(self, div_op, grid):
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = div_op(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_field_exact_divergence(self, div_op, grid):
        """For u=c*x, v=c*y, divergence = 2c exactly."""
        c = 1.5
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        u = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        v = jnp.broadcast_to(c * y[:, None], (grid.Ny, grid.Nx))
        result = div_op(u, v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c, atol=1e-8)

    def test_matches_difference2d_divergence(self, div_op, grid):
        """Divergence2D.__call__ must match Difference2D.divergence."""
        diff = Difference2D(grid=grid)
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        u = jnp.broadcast_to(x, (grid.Ny, grid.Nx))
        v = jnp.broadcast_to(y[:, None], (grid.Ny, grid.Nx))
        np.testing.assert_allclose(div_op(u, v), diff.divergence(u, v), atol=1e-12)

    def test_matches_functional_api(self, div_op, grid):
        """Divergence2D.__call__ must match divergence_2d functional form."""
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        np.testing.assert_allclose(
            div_op(u, v),
            divergence_2d(u, v, dx=grid.dx, dy=grid.dy),
            atol=1e-12,
        )

    def test_geostrophic_velocity_nondivergent(self):
        """div(grad_perp(psi)) = 0 for any streamfunction psi.

        Geostrophic velocities derived from a corner-point streamfunction via
            u = -diff_y_X_to_U(psi)
            v =  diff_x_X_to_V(psi)
        are exactly non-divergent at interior T-points.
        """
        grid = ArakawaCGrid2D.from_interior(16, 16, 2.0 * np.pi, 2.0 * np.pi)
        diff = Difference2D(grid=grid)
        div_op = Divergence2D(grid=grid)

        # Arbitrary streamfunction at X-points (corners)
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        psi = jnp.sin(x[None, :]) * jnp.cos(y[:, None])

        u = -diff.diff_y_X_to_U(psi)  # u = -dpsi/dy  at U-points
        v = diff.diff_x_X_to_V(psi)   # v =  dpsi/dx  at V-points
        div = div_op(u, v)

        np.testing.assert_allclose(div[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_no_nan_output(self, div_op, grid):
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = div_op(u, v)
        assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# No-flux BC variant
# ---------------------------------------------------------------------------


class TestDivergence2DNoFlux:
    def test_output_shape(self, div_op, grid):
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        assert div_op.noflux(u, v).shape == (grid.Ny, grid.Nx)

    def test_zero_ghost_cells_same_as_standard(self, div_op, grid):
        """When ghost cells are already zero, noflux and __call__ agree."""
        # zeros_like gives zero ghost cells
        u = jnp.ones((grid.Ny, grid.Nx)).at[:, 0].set(0.0)
        v = jnp.ones((grid.Ny, grid.Nx)).at[0, :].set(0.0)
        np.testing.assert_allclose(
            div_op.noflux(u, v), div_op(u, v), atol=1e-12
        )

    def test_noflux_zeros_west_ghost(self, div_op, grid):
        """noflux makes result independent of the west ghost of u."""
        u_base = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        # Two inputs that differ only in the west ghost
        u_ghost_a = u_base.at[:, 0].set(0.0)
        u_ghost_b = u_base.at[:, 0].set(999.0)
        # noflux zeros the west ghost in both cases → identical result
        np.testing.assert_allclose(
            div_op.noflux(u_ghost_a, v),
            div_op.noflux(u_ghost_b, v),
            atol=1e-12,
        )

    def test_noflux_zeros_south_ghost(self, div_op, grid):
        """noflux makes result independent of the south ghost of v."""
        u = jnp.zeros((grid.Ny, grid.Nx))
        v_base = jnp.ones((grid.Ny, grid.Nx))
        # Two inputs that differ only in the south ghost
        v_ghost_a = v_base.at[0, :].set(0.0)
        v_ghost_b = v_base.at[0, :].set(999.0)
        # noflux zeros the south ghost in both cases → identical result
        np.testing.assert_allclose(
            div_op.noflux(u, v_ghost_a),
            div_op.noflux(u, v_ghost_b),
            atol=1e-12,
        )

    def test_noflux_no_nan(self, div_op, grid):
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        result = div_op.noflux(u, v)
        assert jnp.all(jnp.isfinite(result))
