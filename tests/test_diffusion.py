"""Tests for Diffusion2D, Diffusion3D, and diffusion_2d."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.diffusion import Diffusion2D, Diffusion3D, diffusion_2d
from finitevolx._src.grid import ArakawaCGrid2D, ArakawaCGrid3D

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def grid():
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return ArakawaCGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)


@pytest.fixture
def diff_op(grid):
    return Diffusion2D(grid=grid)


@pytest.fixture
def diff_op3d(grid3d):
    return Diffusion3D(grid=grid3d)


# ---------------------------------------------------------------------------
# Functional API: diffusion_2d
# ---------------------------------------------------------------------------


class TestDiffusion2DFunctional:
    def test_output_shape(self, grid):
        h = jnp.zeros((grid.Ny, grid.Nx))
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy)
        assert result.shape == (grid.Ny, grid.Nx)

    def test_ghost_ring_is_zero(self, grid):
        """Ghost ring in the output is always zero."""
        h = jnp.ones((grid.Ny, grid.Nx))
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy)
        np.testing.assert_allclose(result[0, :], 0.0)
        np.testing.assert_allclose(result[-1, :], 0.0)
        np.testing.assert_allclose(result[:, 0], 0.0)
        np.testing.assert_allclose(result[:, -1], 0.0)

    def test_constant_tracer_zero_tendency(self, grid):
        """Constant tracer has zero diffusion tendency everywhere."""
        h = jnp.ones((grid.Ny, grid.Nx))
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_quadratic_x_gives_constant_tendency(self, grid):
        """For h = c * x^2, ∇²h = 2c/dx^2 exactly (Laplacian = const)."""
        c = 3.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x**2, (grid.Ny, grid.Nx))
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy)
        # ∂²h/∂x² = 2c everywhere, ∂²h/∂y² = 0  →  tendency = 2c
        # Interior avoids boundary ghost-cell pollution: skip first/last column
        np.testing.assert_allclose(result[1:-1, 2:-2], 2.0 * c, rtol=1e-8)

    def test_quadratic_y_gives_constant_tendency(self, grid):
        """For h = c * y^2, ∇²h = 2c/dy^2 exactly."""
        c = 2.5
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.broadcast_to(c * y[:, None] ** 2, (grid.Ny, grid.Nx))
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy)
        # ∂²h/∂y² = 2c everywhere, ∂²h/∂x² = 0  →  tendency = 2c
        np.testing.assert_allclose(result[2:-2, 1:-1], 2.0 * c, rtol=1e-8)

    def test_kappa_scales_tendency(self, grid):
        """Doubling κ doubles the tendency."""
        c = 1.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x**2, (grid.Ny, grid.Nx))
        r1 = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy)
        r2 = diffusion_2d(h, kappa=2.0, dx=grid.dx, dy=grid.dy)
        np.testing.assert_allclose(r2[1:-1, 1:-1], 2.0 * r1[1:-1, 1:-1], rtol=1e-10)

    def test_mask_u_zeros_east_flux(self, grid):
        """mask_u = 0 everywhere zeroes all east-face fluxes → zero tendency."""
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(x**2, (grid.Ny, grid.Nx))
        mask_u = jnp.zeros_like(h)
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy, mask_u=mask_u)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_mask_v_zeros_north_flux(self, grid):
        """mask_v = 0 everywhere zeroes all north-face fluxes → zero tendency."""
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.broadcast_to(y[:, None] ** 2, (grid.Ny, grid.Nx))
        mask_v = jnp.zeros_like(h)
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy, mask_v=mask_v)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_mask_h_zeros_land_tendency(self, grid):
        """mask_h = 0 at a cell forces its tendency to zero."""
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(x**2, (grid.Ny, grid.Nx))
        mask_h = jnp.ones_like(h)
        # Mark the interior cell (3, 3) as land
        mask_h = mask_h.at[3, 3].set(0.0)
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy, mask_h=mask_h)
        assert float(result[3, 3]) == pytest.approx(0.0, abs=1e-12)

    def test_no_nan_output(self, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        result = diffusion_2d(h, kappa=1.0, dx=grid.dx, dy=grid.dy)
        assert jnp.all(jnp.isfinite(result))


# ---------------------------------------------------------------------------
# Class-based API: Diffusion2D
# ---------------------------------------------------------------------------


class TestDiffusion2DClass:
    def test_output_shape(self, diff_op, grid):
        h = jnp.zeros((grid.Ny, grid.Nx))
        assert diff_op(h, kappa=1.0).shape == (grid.Ny, grid.Nx)

    def test_constant_tracer_zero_tendency(self, diff_op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        result = diff_op(h, kappa=1.0)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_matches_functional_api(self, diff_op, grid):
        """Diffusion2D.__call__ must match diffusion_2d functional form."""
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(x**2, (grid.Ny, grid.Nx))
        kappa = 0.5
        np.testing.assert_allclose(
            diff_op(h, kappa=kappa),
            diffusion_2d(h, kappa=kappa, dx=grid.dx, dy=grid.dy),
            atol=1e-12,
        )

    def test_matches_functional_api_with_masks(self, diff_op, grid):
        """Class API with masks must match functional API with same masks."""
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(x**2, (grid.Ny, grid.Nx))
        mask_h = jnp.ones_like(h).at[2:4, 2:4].set(0.0)
        mask_u = jnp.ones_like(h).at[:, 0].set(0.0)
        mask_v = jnp.ones_like(h).at[0, :].set(0.0)
        kappa = 0.7
        np.testing.assert_allclose(
            diff_op(h, kappa=kappa, mask_h=mask_h, mask_u=mask_u, mask_v=mask_v),
            diffusion_2d(
                h,
                kappa=kappa,
                dx=grid.dx,
                dy=grid.dy,
                mask_h=mask_h,
                mask_u=mask_u,
                mask_v=mask_v,
            ),
            atol=1e-12,
        )

    def test_spatially_varying_kappa(self, diff_op, grid):
        """Spatially varying kappa array is accepted and gives finite results."""
        h = jnp.ones((grid.Ny, grid.Nx))
        kappa_field = jnp.full((grid.Ny, grid.Nx), 1e-2)
        result = diff_op(h, kappa=kappa_field)
        assert result.shape == (grid.Ny, grid.Nx)
        assert jnp.all(jnp.isfinite(result))

    def test_ghost_ring_is_zero(self, diff_op, grid):
        h = jnp.ones((grid.Ny, grid.Nx))
        result = diff_op(h, kappa=1.0)
        np.testing.assert_allclose(result[0, :], 0.0)
        np.testing.assert_allclose(result[-1, :], 0.0)
        np.testing.assert_allclose(result[:, 0], 0.0)
        np.testing.assert_allclose(result[:, -1], 0.0)


# ---------------------------------------------------------------------------
# Fluxes method: Diffusion2D.fluxes
# ---------------------------------------------------------------------------


class TestDiffusion2DFluxes:
    def test_output_shapes(self, diff_op, grid):
        h = jnp.zeros((grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        assert fx.shape == (grid.Ny, grid.Nx)
        assert fy.shape == (grid.Ny, grid.Nx)

    def test_constant_field_zero_fluxes(self, diff_op, grid):
        """Constant tracer → zero gradient → zero fluxes."""
        h = jnp.ones((grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        np.testing.assert_allclose(fx, 0.0, atol=1e-12)
        np.testing.assert_allclose(fy, 0.0, atol=1e-12)

    def test_linear_x_flux_is_constant(self, diff_op, grid):
        """For h = c*x, east-face flux = κ*c everywhere (constant gradient)."""
        c = 2.0
        kappa = 1.5
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=kappa)
        # flux_x = κ * c (constant forward difference of linear field)
        np.testing.assert_allclose(fx[1:-1, 1:-1], kappa * c, rtol=1e-8)
        # flux_y = 0 (no y-variation)
        np.testing.assert_allclose(fy[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_flux_divergence_equals_tendency(self, diff_op, grid):
        """∇·(flux_x, flux_y) must equal tendency from __call__."""
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.sin(x[None, :]) + jnp.cos(y[:, None])
        kappa = 0.3

        tendency = diff_op(h, kappa=kappa)
        fx, fy = diff_op.fluxes(h, kappa=kappa)

        # Manually compute divergence of fluxes
        dx, dy = grid.dx, grid.dy
        div_flux = jnp.zeros_like(h)
        div_flux = div_flux.at[1:-1, 1:-1].set(
            (fx[1:-1, 1:-1] - fx[1:-1, :-2]) / dx
            + (fy[1:-1, 1:-1] - fy[:-2, 1:-1]) / dy
        )
        np.testing.assert_allclose(tendency, div_flux, atol=1e-12)

    def test_mask_u_zeros_east_flux(self, diff_op, grid):
        """mask_u = 0 zeros east-face fluxes; north fluxes unaffected."""
        c = 1.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        mask_u = jnp.zeros_like(h)
        fx, _fy = diff_op.fluxes(h, kappa=1.0, mask_u=mask_u)
        np.testing.assert_allclose(fx, 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, diff_op, grid):
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(x**2, (grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=1.0)
        # Ghost ring of flux arrays is always zero
        np.testing.assert_allclose(fx[0, :], 0.0)
        np.testing.assert_allclose(fx[-1, :], 0.0)
        np.testing.assert_allclose(fx[:, 0], 0.0)
        np.testing.assert_allclose(fx[:, -1], 0.0)
        np.testing.assert_allclose(fy[0, :], 0.0)
        np.testing.assert_allclose(fy[-1, :], 0.0)
        np.testing.assert_allclose(fy[:, 0], 0.0)
        np.testing.assert_allclose(fy[:, -1], 0.0)


# ---------------------------------------------------------------------------
# Class-based API: Diffusion3D
# ---------------------------------------------------------------------------


class TestDiffusion3DClass:
    def test_output_shape(self, diff_op3d, grid3d):
        h = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff_op3d(h, kappa=1.0)
        assert result.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_constant_tracer_zero_tendency(self, diff_op3d, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff_op3d(h, kappa=1.0)
        np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, diff_op3d, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff_op3d(h, kappa=1.0)
        np.testing.assert_allclose(result[0, :, :], 0.0)
        np.testing.assert_allclose(result[-1, :, :], 0.0)
        np.testing.assert_allclose(result[:, 0, :], 0.0)
        np.testing.assert_allclose(result[:, -1, :], 0.0)
        np.testing.assert_allclose(result[:, :, 0], 0.0)
        np.testing.assert_allclose(result[:, :, -1], 0.0)

    def test_quadratic_x_gives_constant_tendency(self, diff_op3d, grid3d):
        """For h = c*x^2, tendency = 2c at interior cells (all z-levels)."""
        c = 2.0
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        h = jnp.broadcast_to(c * x**2, (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff_op3d(h, kappa=1.0)
        # Skip boundary z-levels and boundary polluted columns
        np.testing.assert_allclose(result[1:-1, 1:-1, 2:-2], 2.0 * c, rtol=1e-8)

    def test_fluxes_output_shapes(self, diff_op3d, grid3d):
        h = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        fx, fy = diff_op3d.fluxes(h, kappa=1.0)
        assert fx.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert fy.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_no_nan_output(self, diff_op3d, grid3d):
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        result = diff_op3d(h, kappa=1.0)
        assert jnp.all(jnp.isfinite(result))
