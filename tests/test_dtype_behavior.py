"""Tests for dtype consistency and numerical robustness.

Every public operator must:
- Produce finite outputs on well-defined inputs (no silent NaN/Inf).
- Agree between float32 and float64 within a generous tolerance.
- Not produce NaN when fed NaN-free but numerically extreme inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.reconstruction import Reconstruction1D, Reconstruction2D
from finitevolx._src.diffusion.diffusion import Diffusion2D, diffusion_2d
from finitevolx._src.grid.cartesian import CartesianGrid1D, CartesianGrid2D
from finitevolx._src.operators.difference import Difference1D, Difference2D
from finitevolx._src.operators.divergence import Divergence2D, divergence_2d
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.operators.vorticity import Vorticity2D

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid1d():
    return CartesianGrid1D.from_interior(8, 1.0)


@pytest.fixture
def grid2d():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


# ---------------------------------------------------------------------------
# float32 vs float64 consistency
# ---------------------------------------------------------------------------


class TestFloat32Float64Agreement:
    """float32 and float64 results must agree within a generous tolerance."""

    def test_difference_laplacian_f32_f64_agree(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h64 = (x[None, :] ** 2 + y[:, None] ** 2).astype(jnp.float64)
        h32 = h64.astype(jnp.float32)

        res64 = diff.laplacian(h64)
        res32 = diff.laplacian(h32)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
            err_msg="float32 and float64 Laplacian differ beyond tolerance",
        )

    def test_difference_diff_x_f32_f64_agree(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h64 = jnp.broadcast_to(2.0 * x, (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        h32 = h64.astype(jnp.float32)

        res64 = diff.diff_x_T_to_U(h64)
        res32 = diff.diff_x_T_to_U(h32)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )

    def test_interpolation_T_to_U_f32_f64_agree(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h64 = jnp.arange(grid2d.Ny * grid2d.Nx, dtype=jnp.float64).reshape(
            grid2d.Ny, grid2d.Nx
        )
        h32 = h64.astype(jnp.float32)

        res64 = interp.T_to_U(h64)
        res32 = interp.T_to_U(h32)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )

    def test_reconstruction_weno3_f32_f64_agree(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h64 = jnp.ones((grid2d.Ny, grid2d.Nx), dtype=jnp.float64)
        u64 = jnp.ones((grid2d.Ny, grid2d.Nx), dtype=jnp.float64)
        h32 = h64.astype(jnp.float32)
        u32 = u64.astype(jnp.float32)

        res64 = recon.weno3_x(h64, u64)
        res32 = recon.weno3_x(h32, u32)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )

    def test_reconstruction_weno5_f32_f64_agree(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h64 = jnp.ones((grid2d.Ny, grid2d.Nx), dtype=jnp.float64)
        u64 = jnp.ones((grid2d.Ny, grid2d.Nx), dtype=jnp.float64)
        h32 = h64.astype(jnp.float32)
        u32 = u64.astype(jnp.float32)

        res64 = recon.weno5_x(h64, u64)
        res32 = recon.weno5_x(h32, u32)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )

    def test_vorticity_f32_f64_agree(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u64 = jnp.broadcast_to(-y[:, None], (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        v64 = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        u32 = u64.astype(jnp.float32)
        v32 = v64.astype(jnp.float32)

        res64 = vort.relative_vorticity(u64, v64)
        res32 = vort.relative_vorticity(u32, v32)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )

    def test_divergence_2d_f32_f64_agree(self, grid2d):
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u64 = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        v64 = jnp.broadcast_to(y[:, None], (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        u32 = u64.astype(jnp.float32)
        v32 = v64.astype(jnp.float32)

        res64 = divergence_2d(u64, v64, dx=grid2d.dx, dy=grid2d.dy)
        res32 = divergence_2d(u32, v32, dx=grid2d.dx, dy=grid2d.dy)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )

    def test_divergence2d_class_f32_f64_agree(self, grid2d):
        div_op = Divergence2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u64 = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        v64 = jnp.broadcast_to(y[:, None], (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        u32 = u64.astype(jnp.float32)
        v32 = v64.astype(jnp.float32)

        res64 = div_op(u64, v64)
        res32 = div_op(u32, v32)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )

    def test_diffusion_2d_functional_f32_f64_agree(self, grid2d):
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h64 = jnp.broadcast_to(x**2, (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        h32 = h64.astype(jnp.float32)

        res64 = diffusion_2d(h64, kappa=1.0, dx=grid2d.dx, dy=grid2d.dy)
        res32 = diffusion_2d(h32, kappa=1.0, dx=grid2d.dx, dy=grid2d.dy)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )

    def test_diffusion2d_class_f32_f64_agree(self, grid2d):
        diff_op = Diffusion2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h64 = jnp.broadcast_to(x**2, (grid2d.Ny, grid2d.Nx)).astype(jnp.float64)
        h32 = h64.astype(jnp.float32)

        res64 = diff_op(h64, kappa=1.0)
        res32 = diff_op(h32, kappa=1.0)

        np.testing.assert_allclose(
            res32[1:-1, 1:-1].astype(float),
            res64[1:-1, 1:-1].astype(float),
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# No-NaN / no-Inf checks for all operators on valid inputs
# ---------------------------------------------------------------------------


class TestNoNanOnValidInputs:
    """All operators must produce finite outputs on well-defined finite inputs."""

    def test_difference_1d_no_nan(self, grid1d):
        diff = Difference1D(grid=grid1d)
        x = jnp.arange(grid1d.Nx, dtype=float) * grid1d.dx
        h = jnp.sin(x)
        for result in [diff.diff_x_T_to_U(h), diff.diff_x_U_to_T(h), diff.laplacian(h)]:
            assert jnp.all(jnp.isfinite(result)), "NaN or Inf in 1D difference output"

    def test_difference_2d_no_nan(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = jnp.sin(x[None, :]) * jnp.cos(y[:, None])
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        results = [
            diff.diff_x_T_to_U(h),
            diff.diff_y_T_to_V(h),
            diff.laplacian(h),
            diff.divergence(u, v),
            diff.curl(u, v),
        ]
        for result in results:
            assert jnp.all(jnp.isfinite(result)), "NaN or Inf in 2D difference output"

    def test_interpolation_2d_no_nan(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = jnp.sin(x[None, :]) + jnp.cos(y[:, None])
        for result in [interp.T_to_U(h), interp.T_to_V(h), interp.T_to_X(h)]:
            assert jnp.all(jnp.isfinite(result)), "NaN or Inf in interpolation output"

    def test_reconstruction_weno3_no_nan(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float)
        h = jnp.broadcast_to(jnp.sin(x), (grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno3_x(h, u)
        assert jnp.all(jnp.isfinite(result)), "NaN or Inf in weno3 reconstruction"

    def test_reconstruction_weno5_no_nan(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float)
        h = jnp.broadcast_to(jnp.sin(x), (grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_x(h, u)
        assert jnp.all(jnp.isfinite(result)), "NaN or Inf in weno5 reconstruction"

    def test_reconstruction_weno5_y_no_nan(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        y = jnp.arange(grid2d.Ny, dtype=float)
        h = jnp.broadcast_to(jnp.cos(y[:, None]), (grid2d.Ny, grid2d.Nx))
        v = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_y(h, v)
        assert jnp.all(jnp.isfinite(result)), "NaN or Inf in weno5_y reconstruction"

    def test_vorticity_no_nan(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = vort.relative_vorticity(u, v)
        assert jnp.all(jnp.isfinite(result)), "NaN or Inf in vorticity output"

    def test_divergence2d_no_nan(self, grid2d):
        div_op = Divergence2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(jnp.sin(x), (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(jnp.cos(y[:, None]), (grid2d.Ny, grid2d.Nx))
        for result in [div_op(u, v), div_op.noflux(u, v)]:
            assert jnp.all(jnp.isfinite(result)), "NaN or Inf in Divergence2D output"

    def test_divergence_2d_functional_no_nan(self, grid2d):
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(jnp.sin(x), (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(jnp.cos(y[:, None]), (grid2d.Ny, grid2d.Nx))
        result = divergence_2d(u, v, dx=grid2d.dx, dy=grid2d.dy)
        assert jnp.all(jnp.isfinite(result)), "NaN or Inf in divergence_2d output"

    def test_diffusion2d_no_nan(self, grid2d):
        diff_op = Diffusion2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = jnp.sin(x[None, :]) * jnp.cos(y[:, None])
        result = diff_op(h, kappa=1e-2)
        assert jnp.all(jnp.isfinite(result)), "NaN or Inf in Diffusion2D output"

    def test_diffusion_2d_functional_no_nan(self, grid2d):
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = jnp.sin(x[None, :]) * jnp.cos(y[:, None])
        result = diffusion_2d(h, kappa=1e-2, dx=grid2d.dx, dy=grid2d.dy)
        assert jnp.all(jnp.isfinite(result)), "NaN or Inf in diffusion_2d output"


# ---------------------------------------------------------------------------
# Reconstruction stability near discontinuities (no NaN regression tests)
# ---------------------------------------------------------------------------


class TestReconstructionNaNRegression:
    """Reconstruction schemes must not produce NaN near sharp discontinuities."""

    def _step_field(self, grid: CartesianGrid2D) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Create a step function and positive velocity field."""
        h = jnp.ones((grid.Ny, grid.Nx))
        # Introduce a step discontinuity in the middle of the domain
        mid = grid.Nx // 2
        h = h.at[:, mid:].set(5.0)
        u = jnp.ones((grid.Ny, grid.Nx))
        return h, u

    def test_weno3_x_no_nan_near_step(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h, u = self._step_field(grid2d)
        result = recon.weno3_x(h, u)
        assert jnp.all(jnp.isfinite(result[1:-1, 1:-1])), (
            "weno3_x produced NaN near step discontinuity"
        )

    def test_weno5_x_no_nan_near_step(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h, u = self._step_field(grid2d)
        result = recon.weno5_x(h, u)
        assert jnp.all(jnp.isfinite(result[1:-1, 1:-1])), (
            "weno5_x produced NaN near step discontinuity"
        )

    def test_wenoz3_x_no_nan_near_step(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h, u = self._step_field(grid2d)
        result = recon.wenoz3_x(h, u)
        assert jnp.all(jnp.isfinite(result[1:-1, 1:-1])), (
            "wenoz3_x produced NaN near step discontinuity"
        )

    def test_wenoz5_x_no_nan_near_step(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h, u = self._step_field(grid2d)
        result = recon.wenoz5_x(h, u)
        assert jnp.all(jnp.isfinite(result[1:-1, 1:-1])), (
            "wenoz5_x produced NaN near step discontinuity"
        )

    def test_weno5_x_no_nan_near_step_negative_flow(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h, _ = self._step_field(grid2d)
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_x(h, u)
        assert jnp.all(jnp.isfinite(result[1:-1, 1:-1])), (
            "weno5_x produced NaN near step discontinuity with negative flow"
        )

    def test_weno5_y_no_nan_near_step(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        mid = grid2d.Ny // 2
        h = h.at[mid:, :].set(5.0)
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        result = recon.weno5_y(h, v)
        assert jnp.all(jnp.isfinite(result[1:-1, 1:-1])), (
            "weno5_y produced NaN near step discontinuity"
        )

    def test_weno3_1d_no_nan_near_step(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        mid = grid1d.Nx // 2
        h = h.at[mid:].set(5.0)
        u = jnp.ones(grid1d.Nx)
        result = recon.weno3_x(h, u)
        assert jnp.all(jnp.isfinite(result[1:-1])), (
            "1D weno3 produced NaN near step discontinuity"
        )

    def test_weno5_1d_no_nan_near_step(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        mid = grid1d.Nx // 2
        h = h.at[mid:].set(5.0)
        u = jnp.ones(grid1d.Nx)
        result = recon.weno5_x(h, u)
        assert jnp.all(jnp.isfinite(result[1:-1])), (
            "1D weno5 produced NaN near step discontinuity"
        )


# ---------------------------------------------------------------------------
# Small dx stress test: operators must not overflow or break on fine grids
# ---------------------------------------------------------------------------


class TestSmallSpacingRobustness:
    """Operators must produce finite results on grids with very fine spacing."""

    def test_difference_small_dx_is_finite(self):
        dx = 1e-4
        grid = CartesianGrid2D.from_interior(8, 8, dx * 8, dx * 8)
        diff = Difference2D(grid=grid)
        x = jnp.arange(grid.Nx, dtype=float) * grid.dx
        y = jnp.arange(grid.Ny, dtype=float) * grid.dy
        h = jnp.sin(x[None, :]) * jnp.cos(y[:, None])
        result = diff.laplacian(h)
        assert jnp.all(jnp.isfinite(result)), "Laplacian has NaN/Inf on fine grid"

    def test_interpolation_small_dx_is_finite(self):
        dx = 1e-4
        grid = CartesianGrid2D.from_interior(8, 8, dx * 8, dx * 8)
        interp = Interpolation2D(grid=grid)
        h = jnp.ones((grid.Ny, grid.Nx))
        result = interp.T_to_U(h)
        assert jnp.all(jnp.isfinite(result)), "Interpolation has NaN/Inf on fine grid"
