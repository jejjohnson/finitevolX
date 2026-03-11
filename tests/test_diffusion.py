"""Tests for HarmonicDiffusion2D, BiharmonicDiffusion2D, and their 3-D counterparts.

All arrays share the same shape [Ny, Nx] (2-D) or [Nz, Ny, Nx] (3-D) with
one ghost-cell ring on each side.  Operators write only to the interior
[1:-1, 1:-1] / [1:-1, 1:-1, 1:-1]; the ghost ring is zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.boundary import enforce_periodic
from finitevolx._src.diffusion import (
    BiharmonicDiffusion2D,
    BiharmonicDiffusion3D,
    HarmonicDiffusion2D,
    HarmonicDiffusion3D,
)
from finitevolx._src.grid import ArakawaCGrid2D, ArakawaCGrid3D

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def grid2d():
    # 8 interior cells → total shape [10, 10]
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture()
def grid3d():
    # 4 z-levels, 8 interior cells in x/y → shape [6, 10, 10]
    return ArakawaCGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# HarmonicDiffusion2D
# ---------------------------------------------------------------------------


class TestHarmonicDiffusion2D:
    def test_constant_field_zero_tendency(self, grid2d):
        """Constant h → ∇²h = 0 → zero tendency everywhere."""
        op = HarmonicDiffusion2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_allclose(tend[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, grid2d):
        """Ghost cells must remain zero (interior-point idiom)."""
        op = HarmonicDiffusion2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_array_equal(tend[0, :], 0.0)
        np.testing.assert_array_equal(tend[-1, :], 0.0)
        np.testing.assert_array_equal(tend[:, 0], 0.0)
        np.testing.assert_array_equal(tend[:, -1], 0.0)

    def test_output_shape(self, grid2d):
        """Output shape must equal input shape."""
        op = HarmonicDiffusion2D(grid=grid2d)
        h = jnp.zeros((grid2d.Ny, grid2d.Nx))
        assert op(h, kappa=1.0).shape == (grid2d.Ny, grid2d.Nx)

    def test_kappa_scales_linearly(self, grid2d):
        """Doubling κ doubles the tendency."""
        op = HarmonicDiffusion2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        jy = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = ix[None, :] ** 2 + jy[:, None] ** 2
        tend1 = op(h, kappa=1.0)
        tend2 = op(h, kappa=2.0)
        np.testing.assert_allclose(
            tend2[1:-1, 1:-1], 2.0 * tend1[1:-1, 1:-1], rtol=1e-6
        )

    def test_quadratic_field_constant_laplacian(self, grid2d):
        """h = x² + y² → ∇²h = 2 + 2 = 4 everywhere.

        The discrete second-order scheme is exact for quadratic polynomials:
        (h[i+1] - 2h[i] + h[i-1]) / dx² = 2 for h[i] = (i·dx)².
        """
        op = HarmonicDiffusion2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        jy = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = ix[None, :] ** 2 + jy[:, None] ** 2
        kappa = 1.0
        tend = op(h, kappa=kappa)
        # ∇²(x² + y²) = 2 + 2 = 4; discrete scheme is exact for quadratics
        expected = kappa * 4.0
        np.testing.assert_allclose(tend[1:-1, 1:-1], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# BiharmonicDiffusion2D
# ---------------------------------------------------------------------------


class TestBiharmonicDiffusion2D:
    def test_constant_field_zero_tendency(self, grid2d):
        """Constant h → ∇⁴h = 0 → zero tendency everywhere."""
        op = BiharmonicDiffusion2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_allclose(tend[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, grid2d):
        """Ghost cells must remain zero (interior-point idiom)."""
        op = BiharmonicDiffusion2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_array_equal(tend[0, :], 0.0)
        np.testing.assert_array_equal(tend[-1, :], 0.0)
        np.testing.assert_array_equal(tend[:, 0], 0.0)
        np.testing.assert_array_equal(tend[:, -1], 0.0)

    def test_output_shape(self, grid2d):
        """Output shape must equal input shape."""
        op = BiharmonicDiffusion2D(grid=grid2d)
        h = jnp.zeros((grid2d.Ny, grid2d.Nx))
        assert op(h, kappa=1.0).shape == (grid2d.Ny, grid2d.Nx)

    def test_kappa_scales_linearly(self, grid2d):
        """Doubling κ doubles the magnitude of the tendency."""
        op = BiharmonicDiffusion2D(grid=grid2d)
        ix = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        jy = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = ix[None, :] ** 2 + jy[:, None] ** 2
        tend1 = op(h, kappa=1.0)
        tend2 = op(h, kappa=2.0)
        np.testing.assert_allclose(
            tend2[2:-2, 2:-2], 2.0 * tend1[2:-2, 2:-2], rtol=1e-6
        )

    def test_negative_sign_convention(self, grid2d):
        """Biharmonic tendency is −κ · ∇⁴h; the sign must be negative.

        Uses h = sin(x) · sin(y) which has a nonzero biharmonic.  The
        field is periodic so ghost cells are filled before calling the
        operator; results are compared in the deep interior [2:-2, 2:-2].
        """
        from finitevolx._src.difference import Difference2D

        op = BiharmonicDiffusion2D(grid=grid2d)
        diff = Difference2D(grid=grid2d)

        ix = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        jy = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        # sin(x)*sin(y): ∇²h = -2*sin(x)*sin(y) ≠ const, so ∇⁴h ≠ 0
        h = jnp.sin(ix[None, :]) * jnp.sin(jy[:, None])
        h = enforce_periodic(h)

        # ∇⁴h = ∇²(∇²h); compute independently using the same operator
        lap1 = diff.laplacian(h)
        lap2 = diff.laplacian(lap1)

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

    def test_scale_selective_damping(self):
        """Biharmonic tendency for sin(k·x) matches the discrete k⁴ eigenvalue.

        For h = sin(k·x) with periodic BCs the discrete Laplacian gives:
            Lap(sin(k·x)) = μ_k · sin(k·x),   μ_k = −4 sin²(k·dx/2) / dx²

        The biharmonic tendency is then:
            −κ · μ_k² · sin(k·x)

        The eigenvalue ratio μ_k4² / μ_k1² ≈ (k4/k1)⁴ for well-resolved modes
        (k·dx ≪ 1), demonstrating scale-selective dissipation.
        """
        N_int = 64
        Lx = 2.0 * float(jnp.pi)
        Ly = 2.0 * float(jnp.pi)
        grid = ArakawaCGrid2D.from_interior(N_int, N_int, Lx, Ly)
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
            # Expected: -kappa * mu_k² * sin(k*x) = -kappa * mu_k² * h
            expected = -kappa * mu_k**2 * h

            # Deep interior to avoid ghost-cell contamination on all four sides
            # Use atol for near-zero crossings of sin where rtol is ill-conditioned
            max_val = float(jnp.max(jnp.abs(expected[4:-4, 4:-4])))
            np.testing.assert_allclose(
                tend[4:-4, 4:-4], expected[4:-4, 4:-4], rtol=0.01, atol=max_val * 1e-8
            )

        # Eigenvalue ratio ≈ (k4/k1)⁴ for well-resolved modes
        mu_k1 = disc_lap_eigenvalue(k1)
        mu_k4 = disc_lap_eigenvalue(k4)
        eigenvalue_ratio = mu_k4**2 / mu_k1**2
        np.testing.assert_allclose(eigenvalue_ratio, float(k4**4 / k1**4), rtol=0.05)

    def test_biharmonic_stronger_than_harmonic_for_high_k(self):
        """For short-wave modes, biharmonic damps more than harmonic.

        The discrete Laplacian eigenvalue for wavenumber k is:
            μ_k = −4 sin²(k·dx/2) / dx²

        Harmonic damps as κ_h |μ_k|, biharmonic as κ_bi μ_k².
        For k·dx ≪ 1: |μ_k| ≈ k² and μ_k² ≈ k⁴, so with kappa_h = kappa_bi
        and k > 1/dx, biharmonic damps more (k⁴ > k²).
        """
        N_int = 64
        Lx = 2.0 * float(jnp.pi)
        Ly = 2.0 * float(jnp.pi)
        grid = ArakawaCGrid2D.from_interior(N_int, N_int, Lx, Ly)

        harm_op = HarmonicDiffusion2D(grid=grid)
        biharm_op = BiharmonicDiffusion2D(grid=grid)

        # k = 8: well-resolved, k·dx = 8 * (2π/64) = π/4, |μ_k|² > |μ_k|
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
# HarmonicDiffusion3D
# ---------------------------------------------------------------------------


class TestHarmonicDiffusion3D:
    def test_constant_field_zero_tendency(self, grid3d):
        """Constant h → ∇²h = 0 → zero tendency."""
        op = HarmonicDiffusion3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_allclose(tend[1:-1, 1:-1, 1:-1], 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, grid3d):
        """Ghost cells must remain zero."""
        op = HarmonicDiffusion3D(grid=grid3d)
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
        op = HarmonicDiffusion3D(grid=grid3d)
        h = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert op(h, kappa=1.0).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_kappa_scales_linearly(self, grid3d):
        """Doubling κ doubles the tendency."""
        op = HarmonicDiffusion3D(grid=grid3d)
        ix = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        jy = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        h = jnp.broadcast_to(
            ix[None, None, :] + jy[None, :, None] ** 2,
            (grid3d.Nz, grid3d.Ny, grid3d.Nx),
        )
        tend1 = op(h, kappa=1.0)
        tend2 = op(h, kappa=2.0)
        np.testing.assert_allclose(
            tend2[1:-1, 1:-1, 1:-1], 2.0 * tend1[1:-1, 1:-1, 1:-1], rtol=1e-6
        )


# ---------------------------------------------------------------------------
# BiharmonicDiffusion3D
# ---------------------------------------------------------------------------


class TestBiharmonicDiffusion3D:
    def test_constant_field_zero_tendency(self, grid3d):
        """Constant h → ∇⁴h = 0 → zero tendency."""
        op = BiharmonicDiffusion3D(grid=grid3d)
        h = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        tend = op(h, kappa=1.0)
        np.testing.assert_allclose(tend[1:-1, 1:-1, 1:-1], 0.0, atol=1e-12)

    def test_ghost_ring_is_zero(self, grid3d):
        """Ghost cells must remain zero."""
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
        op = BiharmonicDiffusion3D(grid=grid3d)
        h = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        assert op(h, kappa=1.0).shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_kappa_scales_linearly(self, grid3d):
        """Doubling κ doubles the magnitude of the tendency."""
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
