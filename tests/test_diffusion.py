"""Tests for Diffusion2D, Diffusion3D, and diffusion_2d."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.diffusion.diffusion import Diffusion2D, Diffusion3D, diffusion_2d
from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D


@pytest.fixture
def grid():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return CartesianGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)


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

    def test_all_dry_mask_zeros_tendency_via_flux_masks(self, grid):
        """An all-dry Mask2D zeros every face flux → zero tendency.

        Under the new class-field API, the functional ``diffusion_2d`` is
        mask-free; this test builds an all-dry ``Mask2D`` and asks the
        ``Diffusion2D`` class operator — which applies the intermediate
        flux-masking pattern internally — to produce zero output.
        """
        from finitevolx._src.mask import Mask2D

        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        diff = Diffusion2D(grid=grid, mask=all_dry)
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(x**2, (grid.Ny, grid.Nx))
        result = diff(h, kappa=1.0)
        np.testing.assert_allclose(result[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_interior_land_cell_tendency_is_zero(self, grid):
        """A dry T-cell carries exactly zero tendency under the class API.

        This is the operator-attribute equivalent of the old
        ``mask_h`` zeroing: construct a ``Mask2D`` with one dry interior
        cell at ``(3, 3)`` and verify the resulting tendency is 0 there.
        """
        from finitevolx._src.mask import Mask2D

        h_mask = np.ones((grid.Ny, grid.Nx), dtype=bool)
        h_mask[3, 3] = False
        mask = Mask2D.from_mask(h_mask)
        diff = Diffusion2D(grid=grid, mask=mask)
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(x**2, (grid.Ny, grid.Nx))
        result = diff(h, kappa=1.0)
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

    def test_masked_class_zeros_dry_interior_block(self, grid):
        """Class API with a Mask2D: wet-interior tendencies match the
        unmasked class API bit-for-bit (since mask.h is 1 at wet cells),
        and the dry 2x2 interior block carries exactly zero tendency.
        """
        from finitevolx._src.mask import Mask2D

        h_mask = np.ones((grid.Ny, grid.Nx), dtype=bool)
        h_mask[2:4, 2:4] = False  # dry 2x2 interior block
        mask = Mask2D.from_mask(h_mask)
        diff_unmasked = Diffusion2D(grid=grid)
        diff_masked = Diffusion2D(grid=grid, mask=mask)

        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(x**2, (grid.Ny, grid.Nx))
        out_masked = np.asarray(diff_masked(h, kappa=0.7))
        # Dry cells → exact zero via the mask.h final multiply
        assert np.all(out_masked[2:4, 2:4] == 0.0)
        # Unmasked path stays bit-identical at mask.none
        np.testing.assert_array_equal(
            np.asarray(diff_unmasked(h, kappa=0.7)),
            np.asarray(Diffusion2D(grid=grid, mask=None)(h, kappa=0.7)),
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
        """For h = c*x, east-face flux = κ*c for all interior-interior faces."""
        c = 2.0
        kappa = 1.5
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        fx, fy = diff_op.fluxes(h, kappa=kappa)
        # flux_x = κ * c for faces i=1 ... Nx-3 (east boundary face i=Nx-2 is 0)
        np.testing.assert_allclose(fx[1:-1, 1:-2], kappa * c, rtol=1e-8)
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

    def test_all_dry_mask_zeros_east_flux(self, grid):
        """An all-dry Mask2D (mask.u = 0 everywhere) zeros east-face fluxes."""
        from finitevolx._src.mask import Mask2D

        all_dry = Mask2D.from_mask(np.zeros((grid.Ny, grid.Nx), dtype=bool))
        diff_masked = Diffusion2D(grid=grid, mask=all_dry)
        c = 1.0
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
        fx, _fy = diff_masked.fluxes(h, kappa=1.0)
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
    @pytest.mark.slow
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


# ---------------------------------------------------------------------------
# BiharmonicDiffusion2D
# ---------------------------------------------------------------------------


class TestBiharmonicDiffusion2D:
    def test_constant_field_zero_tendency(self, grid):
        """Constant h -> nabla^4 h = 0 -> zero tendency everywhere."""
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion2D

        op = BiharmonicDiffusion2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_allclose(tend[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, grid):
        """Ghost cells must remain zero (interior-point idiom)."""
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion2D

        op = BiharmonicDiffusion2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_array_equal(tend[0, :], 0.0)
        np.testing.assert_array_equal(tend[-1, :], 0.0)
        np.testing.assert_array_equal(tend[:, 0], 0.0)
        np.testing.assert_array_equal(tend[:, -1], 0.0)

    def test_output_shape(self, grid):
        """Output shape must equal input shape."""
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion2D

        op = BiharmonicDiffusion2D(grid=grid)
        h = jnp.zeros((grid.Ny, grid.Nx))
        assert op(h, kappa=1.0).shape == (grid.Ny, grid.Nx)

    def test_kappa_scales_linearly(self, grid):
        """Doubling kappa doubles the magnitude of the tendency."""
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion2D

        op = BiharmonicDiffusion2D(grid=grid)
        ix = jnp.arange(grid.Nx, dtype=float) * grid.dx
        jy = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = ix[None, :] ** 2 + jy[:, None] ** 2
        tend1 = op(h, kappa=1.0)
        tend2 = op(h, kappa=2.0)
        np.testing.assert_allclose(
            tend2[2:-2, 2:-2], 2.0 * tend1[2:-2, 2:-2], rtol=1e-6
        )

    def test_negative_sign_convention(self, grid):
        """Biharmonic tendency is -kappa * nabla^4 h; the sign must be negative.

        Uses h = sin(x) * sin(y) which has a nonzero biharmonic.  The
        field is periodic so ghost cells are filled before calling the
        operator; results are compared in the deep interior [2:-2, 2:-2].
        """
        from finitevolx._src.boundary.boundary import enforce_periodic
        from finitevolx._src.diffusion.diffusion import (
            BiharmonicDiffusion2D,
            Diffusion2D,
        )

        op = BiharmonicDiffusion2D(grid=grid)
        harm = Diffusion2D(grid=grid)

        ix = jnp.arange(grid.Nx, dtype=float) * grid.dx
        jy = jnp.arange(grid.Ny, dtype=float) * grid.dy
        # sin(x)*sin(y): nabla^2 h = -2*sin(x)*sin(y) != const, so nabla^4 h != 0
        h = jnp.sin(ix[None, :]) * jnp.sin(jy[:, None])
        h = enforce_periodic(h)

        # nabla^4 h = nabla^2(nabla^2 h); compute independently
        lap1 = harm(h, kappa=1.0)
        lap2 = harm(lap1, kappa=1.0)

        kappa = 3.0
        tend = op(h, kappa=kappa)

        # Verify deep interior: tend = -kappa * nabla^4 h = -kappa * lap2
        # Also verify that lap2 is not identically zero (sign test is meaningful)
        assert float(jnp.max(jnp.abs(lap2[2:-2, 2:-2]))) > 0, (
            "lap2 is zero — test is vacuous"
        )
        np.testing.assert_allclose(
            tend[2:-2, 2:-2], -kappa * lap2[2:-2, 2:-2], rtol=1e-6
        )

    @pytest.mark.slow
    def test_scale_selective_damping(self):
        """Biharmonic tendency for sin(k*x) matches the discrete k^4 eigenvalue.

        For h = sin(k*x) with periodic BCs the discrete Laplacian gives:
            Lap(sin(k*x)) = mu_k * sin(k*x),   mu_k = -4 sin^2(k*dx/2) / dx^2

        The biharmonic tendency is then:
            -kappa * mu_k^2 * sin(k*x)

        The eigenvalue ratio mu_k4^2 / mu_k1^2 approx (k4/k1)^4 for well-resolved
        modes (k*dx << 1), demonstrating scale-selective dissipation.
        """
        from finitevolx._src.boundary.boundary import enforce_periodic
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion2D

        N_int = 64
        Lx = 2.0 * float(jnp.pi)
        Ly = 2.0 * float(jnp.pi)
        grid = CartesianGrid2D.from_interior(N_int, N_int, Lx, Ly)
        op = BiharmonicDiffusion2D(grid=grid)

        k1, k4 = 1, 4
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        kappa = 1.0

        def disc_lap_eigenvalue(k: int) -> float:
            # Eigenvalue of the discrete Laplacian for wavenumber k
            return float(-4.0 * jnp.sin(k * grid.dx / 2) ** 2 / grid.dx**2)

        for k in [k1, k4]:
            h = jnp.tile(jnp.sin(k * x), (grid.Ny, 1))
            h = enforce_periodic(h)

            tend = op(h, kappa=kappa)

            mu_k = disc_lap_eigenvalue(k)
            # Expected: -kappa * mu_k^2 * sin(k*x) = -kappa * mu_k^2 * h
            expected = -kappa * mu_k**2 * h

            # Deep interior to avoid ghost-cell contamination on all four sides
            # Use atol for near-zero crossings of sin where rtol is ill-conditioned
            max_val = float(jnp.max(jnp.abs(expected[4:-4, 4:-4])))
            np.testing.assert_allclose(
                tend[4:-4, 4:-4], expected[4:-4, 4:-4], rtol=0.01, atol=max_val * 1e-8
            )

        # Eigenvalue ratio approx (k4/k1)^4 for well-resolved modes
        mu_k1 = disc_lap_eigenvalue(k1)
        mu_k4 = disc_lap_eigenvalue(k4)
        eigenvalue_ratio = mu_k4**2 / mu_k1**2
        np.testing.assert_allclose(eigenvalue_ratio, float(k4**4 / k1**4), rtol=0.05)

    def test_biharmonic_stronger_than_harmonic_for_high_k(self):
        """For short-wave modes, biharmonic damps more than harmonic.

        The discrete Laplacian eigenvalue for wavenumber k is:
            mu_k = -4 sin^2(k*dx/2) / dx^2

        Harmonic damps as kappa_h |mu_k|, biharmonic as kappa_bi mu_k^2.
        For k*dx << 1: |mu_k| approx k^2 and mu_k^2 approx k^4, so with
        kappa_h = kappa_bi and k > 1/dx, biharmonic damps more (k^4 > k^2).
        """
        from finitevolx._src.boundary.boundary import enforce_periodic
        from finitevolx._src.diffusion.diffusion import (
            BiharmonicDiffusion2D,
            Diffusion2D,
        )

        N_int = 64
        Lx = 2.0 * float(jnp.pi)
        Ly = 2.0 * float(jnp.pi)
        grid = CartesianGrid2D.from_interior(N_int, N_int, Lx, Ly)

        harm_op = Diffusion2D(grid=grid)
        biharm_op = BiharmonicDiffusion2D(grid=grid)

        # k = 8: well-resolved, k*dx = 8 * (2*pi/64) = pi/4, |mu_k|^2 > |mu_k|
        k = 8
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        h = jnp.tile(jnp.sin(k * x), (grid.Ny, 1))
        h = enforce_periodic(h)

        tend_harm = harm_op(h, kappa=1.0)
        tend_biharm = biharm_op(h, kappa=1.0)

        # Mask out near-zero crossings of sin(k*x) to avoid division artefacts.
        # Use float multiplication instead of boolean indexing (JAX doesn't
        # support non-concrete boolean advanced indexing).
        mask = (jnp.abs(jnp.sin(k * x[4:-4])) > 0.5).astype(float)
        abs_harm = jnp.abs(tend_harm[4:-4, 4:-4])
        abs_biharm = jnp.abs(tend_biharm[4:-4, 4:-4])

        # biharmonic amplitude > harmonic amplitude at all masked (non-zero) cells
        assert jnp.all((abs_biharm > abs_harm) | (mask[None, :] == 0))


# ---------------------------------------------------------------------------
# BiharmonicDiffusion3D
# ---------------------------------------------------------------------------


class TestBiharmonicDiffusion3D:
    def test_constant_field_zero_tendency(self, grid3d):
        """Constant h -> nabla^4 h = 0 -> zero tendency."""
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion3D

        op = BiharmonicDiffusion3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_allclose(tend[1:-1, 1:-1, 1:-1], 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, grid3d):
        """Ghost cells must remain zero."""
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion3D

        op = BiharmonicDiffusion3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_array_equal(tend[0, :, :], 0.0)
        np.testing.assert_array_equal(tend[-1, :, :], 0.0)
        np.testing.assert_array_equal(tend[:, 0, :], 0.0)
        np.testing.assert_array_equal(tend[:, -1, :], 0.0)
        np.testing.assert_array_equal(tend[:, :, 0], 0.0)
        np.testing.assert_array_equal(tend[:, :, -1], 0.0)

    def test_output_shape(self, grid3d):
        """Output shape must equal input shape."""
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion3D

        op = BiharmonicDiffusion3D(grid=grid3d)
        h = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert op(h, kappa=1.0).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_kappa_scales_linearly(self, grid3d):
        """Doubling kappa doubles the magnitude of the tendency."""
        from finitevolx._src.diffusion.diffusion import BiharmonicDiffusion3D

        op = BiharmonicDiffusion3D(grid=grid3d)
        ix = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        jy = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        h = jnp.broadcast_to(
            ix[None, None, :] + jy[None, :, None] ** 2,
            (grid3d.Nz, grid3d.Ny, grid3d.Nx),
        )
        tend1 = op(h, kappa=1.0)
        tend2 = op(h, kappa=2.0)
        np.testing.assert_allclose(
            tend2[2:-2, 2:-2, 2:-2], 2.0 * tend1[2:-2, 2:-2, 2:-2], rtol=1e-6
        )


# ---------------------------------------------------------------------------
# #295: variable kappa is averaged to the faces (second order)
# ---------------------------------------------------------------------------


def _kappa_q_2d(x, y, Lx, Ly):
    """κ(x, y) and q(x, y) on [0, Lx] x [0, Ly]; ∂q/∂n = 0 on the walls."""
    kap = 1.0 + 0.5 * jnp.sin(jnp.pi * x / Lx) * jnp.cos(0.5 * jnp.pi * y / Ly)
    q = jnp.cos(jnp.pi * x / Lx) * jnp.cos(jnp.pi * y / Ly)
    return kap, q


def _exact_div_kappa_grad_2d(x, y, Lx, Ly):
    """∇·(κ∇q) = ∂x(κ ∂x q) + ∂y(κ ∂y q) by automatic differentiation."""
    import jax

    def fx(a, b):  # κ ∂q/∂x
        kap, _ = _kappa_q_2d(a, b, Lx, Ly)
        return kap * jax.grad(lambda s: _kappa_q_2d(s, b, Lx, Ly)[1])(a)

    def fy(a, b):  # κ ∂q/∂y
        kap, _ = _kappa_q_2d(a, b, Lx, Ly)
        return kap * jax.grad(lambda s: _kappa_q_2d(a, s, Lx, Ly)[1])(b)

    def div(a, b):
        return jax.grad(fx, 0)(a, b) + jax.grad(fy, 1)(a, b)

    return jax.vmap(jax.vmap(div))(x, y)


def _variable_kappa_error_2d(n: int) -> float:
    import jax

    grid = CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
    # T[j, i] sits at (i dx, j dy); the closed walls are the faces at dx/2 and
    # (n + 1/2) dx, so shift the coordinates to put the walls at 0 and L
    ii = (jnp.arange(grid.Nx) - 0.5) * grid.dx
    jj = (jnp.arange(grid.Ny) - 0.5) * grid.dy
    y, x = jnp.meshgrid(jj, ii, indexing="ij")
    kap, q = _kappa_q_2d(x, y, 1.0, 1.0)
    exact = jax.jit(_exact_div_kappa_grad_2d, static_argnums=(2, 3))(x, y, 1.0, 1.0)
    out = Diffusion2D(grid=grid)(q, kappa=kap)
    return float(jnp.abs(out - exact)[1:-1, 1:-1].max())


class TestVariableKappa:
    @pytest.mark.slow
    def test_convergence_second_order(self):
        """Manufactured solution with κ varying in x and y: second order.

        With the source-cell κ of the pre-#295 operator the error was
        first order (rate 1.00); the face mean gives rate 2.  ∂q/∂n = 0 on
        the walls, so the closed walls (zero flux) are exact.
        """
        errs = [_variable_kappa_error_2d(n) for n in (8, 16, 32)]
        rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
        assert np.all(rates >= 1.9), (errs, rates)

    @pytest.mark.parametrize("wall", ["closed", "open"])
    def test_constant_field_bit_identical_to_scalar(self, grid, wall):
        """A constant κ field equals the scalar-κ result bit for bit (the
        face mean (κ + κ) / 2 is exact), tendency and fluxes, 2-D and 3-D."""
        rng = np.random.default_rng(3)
        h = jnp.asarray(rng.normal(size=(grid.Ny, grid.Nx)))
        k = jnp.full((grid.Ny, grid.Nx), 0.37)
        op = Diffusion2D(grid=grid)
        np.testing.assert_array_equal(op(h, k, wall=wall), op(h, 0.37, wall=wall))
        for a, b in zip(
            op.fluxes(h, k, wall=wall), op.fluxes(h, 0.37, wall=wall), strict=True
        ):
            np.testing.assert_array_equal(a, b)

    def test_constant_field_bit_identical_3d(self, grid3d):
        rng = np.random.default_rng(4)
        shape = (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        h = jnp.asarray(rng.normal(size=shape))
        op = Diffusion3D(grid=grid3d)  # eager: jit may reassociate a scalar κ
        np.testing.assert_array_equal(op(h, jnp.full(shape, 0.37)), op(h, 0.37))

    def test_face_kappa_is_mean(self, grid):
        """flux_x[j, i+1/2] = (κ[j, i] + κ[j, i+1]) / 2 · (h[j, i+1] - h[j, i]) / dx."""
        rng = np.random.default_rng(5)
        h = jnp.asarray(rng.normal(size=(grid.Ny, grid.Nx)))
        k = jnp.asarray(rng.uniform(0.5, 1.5, size=(grid.Ny, grid.Nx)))
        fx, fy = Diffusion2D(grid=grid).fluxes(h, k)
        ex = 0.5 * (k[1:-1, 1:-2] + k[1:-1, 2:-1]) * (h[1:-1, 2:-1] - h[1:-1, 1:-2])
        ey = 0.5 * (k[1:-2, 1:-1] + k[2:-1, 1:-1]) * (h[2:-1, 1:-1] - h[1:-2, 1:-1])
        np.testing.assert_allclose(fx[1:-1, 1:-2], ex / grid.dx, rtol=1e-14)
        np.testing.assert_allclose(fy[1:-2, 1:-1], ey / grid.dy, rtol=1e-14)

    def test_open_wall_face_reads_kappa_ghost(self, grid):
        """wall="open": the wall-face κ is (κ_edge + κ_ghost) / 2, so a
        zero-gradient κ halo gives the wall face the edge-cell κ."""
        from finitevolx._src.boundary.boundary import zero_gradient_boundaries

        rng = np.random.default_rng(6)
        h = jnp.asarray(rng.normal(size=(grid.Ny, grid.Nx)))
        k = zero_gradient_boundaries(
            jnp.asarray(rng.uniform(0.5, 1.5, size=(grid.Ny, grid.Nx)))
        )
        fx, fy = Diffusion2D(grid=grid).fluxes(h, k, wall="open")
        # west wall face (j, 1/2) and east wall face (j, Nx-3/2)
        np.testing.assert_allclose(
            fx[1:-1, 0], k[1:-1, 1] * (h[1:-1, 1] - h[1:-1, 0]) / grid.dx, rtol=1e-14
        )
        np.testing.assert_allclose(
            fx[1:-1, -2],
            k[1:-1, -2] * (h[1:-1, -1] - h[1:-1, -2]) / grid.dx,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            fy[0, 1:-1], k[1, 1:-1] * (h[1, 1:-1] - h[0, 1:-1]) / grid.dy, rtol=1e-14
        )

    @pytest.mark.parametrize("masked", [False, True])
    def test_symmetric_and_conservative(self, masked):
        """The variable-κ operator stays symmetric with zero column sums."""
        import jax

        from finitevolx._src.mask import Mask2D

        g = CartesianGrid2D.from_interior(6, 6, 1.0, 1.0)
        m = np.ones((g.Ny, g.Nx), dtype=bool)
        m[[0, -1], :] = False
        m[:, [0, -1]] = False
        if masked:
            m[3, 3:5] = False  # an island (unmasked: the interior is all wet)
        mask = Mask2D.from_mask(m) if masked else None
        k = jnp.asarray(np.random.default_rng(7).uniform(0.5, 1.5, size=m.shape))
        op = Diffusion2D(grid=g, mask=mask)
        n = m.size
        M = np.asarray(
            jax.jit(jax.jacfwd(lambda x: op(x.reshape(m.shape), k).ravel()))(
                jnp.zeros(n)
            )
        )
        wv = m.ravel()
        A = M[np.ix_(wv, wv)]
        np.testing.assert_allclose(A, A.T, atol=1e-12 * np.abs(A).max())
        np.testing.assert_allclose(A.sum(axis=0), 0.0, atol=1e-11 * np.abs(A).max())
