"""Tests for JAX-transform compatibility: jit and vmap.

Every public operator must produce identical results whether run in
eager mode or under jax.jit / jax.vmap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.difference import Difference1D, Difference2D, Difference3D
from finitevolx._src.diffusion import Diffusion2D, diffusion_2d
from finitevolx._src.divergence import Divergence2D, divergence_2d
from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.interpolation import (
    Interpolation1D,
    Interpolation2D,
)
from finitevolx._src.reconstruction import Reconstruction1D, Reconstruction2D
from finitevolx._src.vorticity import Vorticity2D

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid1d():
    return ArakawaCGrid1D.from_interior(8, 1.0)


@pytest.fixture
def grid2d():
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return ArakawaCGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Difference operators under jit
# ---------------------------------------------------------------------------


class TestJitDifference:
    """jit-compiled difference operators must match eager execution."""

    def test_jit_diff_x_T_to_U_matches_eager(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(2.0 * x, (grid2d.Ny, grid2d.Nx))

        eager = diff.diff_x_T_to_U(h)
        jitted = jax.jit(diff.diff_x_T_to_U)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_diff_y_T_to_V_matches_eager(self, grid2d):
        diff = Difference2D(grid=grid2d)
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = jnp.broadcast_to(3.0 * y[:, None], (grid2d.Ny, grid2d.Nx))

        eager = diff.diff_y_T_to_V(h)
        jitted = jax.jit(diff.diff_y_T_to_V)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_laplacian_matches_eager(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = x[None, :] ** 2 + y[:, None] ** 2

        eager = diff.laplacian(h)
        jitted = jax.jit(diff.laplacian)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_divergence_matches_eager(self, grid2d):
        diff = Difference2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))

        eager = diff.divergence(u, v)
        jitted = jax.jit(diff.divergence)(u, v)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_curl_matches_eager(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))

        eager = diff.curl(u, v)
        jitted = jax.jit(diff.curl)(u, v)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_1d_laplacian_matches_eager(self, grid1d):
        diff = Difference1D(grid=grid1d)
        x = jnp.arange(grid1d.Nx, dtype=float) * grid1d.dx
        h = x**2

        eager = diff.laplacian(h)
        jitted = jax.jit(diff.laplacian)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_3d_laplacian_matches_eager(self, grid3d):
        diff = Difference3D(grid=grid3d)
        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        y = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        h2d = x[None, :] ** 2 + y[:, None] ** 2
        h = jnp.broadcast_to(h2d, (grid3d.Nz, grid3d.Ny, grid3d.Nx))

        eager = diff.laplacian(h)
        jitted = jax.jit(diff.laplacian)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)


# ---------------------------------------------------------------------------
# Interpolation operators under jit
# ---------------------------------------------------------------------------


class TestJitInterpolation:
    """jit-compiled interpolation operators must match eager execution."""

    def test_jit_T_to_U_matches_eager(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = jnp.arange(grid2d.Ny * grid2d.Nx, dtype=float).reshape(grid2d.Ny, grid2d.Nx)

        eager = interp.T_to_U(h)
        jitted = jax.jit(interp.T_to_U)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_T_to_V_matches_eager(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = jnp.arange(grid2d.Ny * grid2d.Nx, dtype=float).reshape(grid2d.Ny, grid2d.Nx)

        eager = interp.T_to_V(h)
        jitted = jax.jit(interp.T_to_V)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_T_to_X_matches_eager(self, grid2d):
        interp = Interpolation2D(grid=grid2d)
        h = jnp.arange(grid2d.Ny * grid2d.Nx, dtype=float).reshape(grid2d.Ny, grid2d.Nx)

        eager = interp.T_to_X(h)
        jitted = jax.jit(interp.T_to_X)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_1d_T_to_U_matches_eager(self, grid1d):
        interp = Interpolation1D(grid=grid1d)
        h = jnp.arange(grid1d.Nx, dtype=float)

        eager = interp.T_to_U(h)
        jitted = jax.jit(interp.T_to_U)(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)


# ---------------------------------------------------------------------------
# Reconstruction operators under jit
# ---------------------------------------------------------------------------


class TestJitReconstruction:
    """jit-compiled reconstruction operators must match eager execution."""

    def test_jit_weno3_x_matches_eager(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))

        eager = recon.weno3_x(h, u)
        jitted = jax.jit(recon.weno3_x)(h, u)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_weno5_x_matches_eager(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = -jnp.ones((grid2d.Ny, grid2d.Nx))

        eager = recon.weno5_x(h, u)
        jitted = jax.jit(recon.weno5_x)(h, u)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_upwind1_x_matches_eager(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float)
        h = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))

        eager = recon.upwind1_x(h, u)
        jitted = jax.jit(recon.upwind1_x)(h, u)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_1d_weno3_matches_eager(self, grid1d):
        recon = Reconstruction1D(grid=grid1d)
        h = jnp.ones(grid1d.Nx)
        u = jnp.ones(grid1d.Nx)

        eager = recon.weno3_x(h, u)
        jitted = jax.jit(recon.weno3_x)(h, u)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)


# ---------------------------------------------------------------------------
# Vorticity operators under jit
# ---------------------------------------------------------------------------


class TestJitVorticity:
    """jit-compiled vorticity operators must match eager execution."""

    def test_jit_relative_vorticity_matches_eager(self, grid2d):
        vort = Vorticity2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        u = jnp.broadcast_to(-y[:, None], (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))

        eager = vort.relative_vorticity(u, v)
        jitted = jax.jit(vort.relative_vorticity)(u, v)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)


# ---------------------------------------------------------------------------
# Divergence operators under jit
# ---------------------------------------------------------------------------


class TestJitDivergence:
    """jit-compiled divergence operators must match eager execution."""

    def test_jit_divergence2d_matches_eager(self, grid2d):
        div_op = Divergence2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))

        eager = div_op(u, v)
        jitted = jax.jit(div_op)(u, v)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_divergence2d_noflux_matches_eager(self, grid2d):
        div_op = Divergence2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))

        eager = div_op.noflux(u, v)
        jitted = jax.jit(div_op.noflux)(u, v)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_divergence_2d_functional_matches_eager(self, grid2d):
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))

        eager = divergence_2d(u, v, dx=grid2d.dx, dy=grid2d.dy)
        jitted = jax.jit(divergence_2d, static_argnums=(2, 3))(
            u, v, grid2d.dx, grid2d.dy
        )

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)


# ---------------------------------------------------------------------------
# Diffusion operators under jit
# ---------------------------------------------------------------------------


class TestJitDiffusion:
    """jit-compiled diffusion operators must match eager execution."""

    def test_jit_diffusion2d_matches_eager(self, grid2d):
        diff_op = Diffusion2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(x**2, (grid2d.Ny, grid2d.Nx))

        eager = diff_op(h, kappa=1e-2)
        jitted = jax.jit(lambda h: diff_op(h, kappa=1e-2))(h)

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)

    def test_jit_diffusion2d_fluxes_matches_eager(self, grid2d):
        diff_op = Diffusion2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(x**2, (grid2d.Ny, grid2d.Nx))

        fx_eager, fy_eager = diff_op.fluxes(h, kappa=0.5)
        fx_jit, fy_jit = jax.jit(lambda h: diff_op.fluxes(h, kappa=0.5))(h)

        np.testing.assert_allclose(fx_jit, fx_eager, rtol=1e-7)
        np.testing.assert_allclose(fy_jit, fy_eager, rtol=1e-7)

    def test_jit_diffusion_2d_functional_matches_eager(self, grid2d):
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(x**2, (grid2d.Ny, grid2d.Nx))

        eager = diffusion_2d(h, kappa=1.0, dx=grid2d.dx, dy=grid2d.dy)
        jitted = jax.jit(diffusion_2d, static_argnums=(2, 3))(
            h, 1.0, grid2d.dx, grid2d.dy
        )

        np.testing.assert_allclose(jitted, eager, rtol=1e-7)


# ---------------------------------------------------------------------------
# Repeated calls are pure and deterministic
# ---------------------------------------------------------------------------


class TestPurityAndDeterminism:
    """Operators must not have hidden state; repeated calls must agree."""

    def test_difference_repeated_calls_identical(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))

        r1 = diff.laplacian(h)
        r2 = diff.laplacian(h)

        np.testing.assert_array_equal(r1, r2)

    def test_jit_difference_repeated_calls_identical(self, grid2d):
        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        h = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))
        jitted = jax.jit(diff.laplacian)

        r1 = jitted(h)
        r2 = jitted(h)

        np.testing.assert_array_equal(r1, r2)

    def test_reconstruction_repeated_calls_identical(self, grid2d):
        recon = Reconstruction2D(grid=grid2d)
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.ones((grid2d.Ny, grid2d.Nx))

        r1 = recon.weno5_x(h, u)
        r2 = recon.weno5_x(h, u)

        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# vmap: batching over independent fields
# ---------------------------------------------------------------------------


class TestVmapOperators:
    """vmap over a batch of independent fields must agree with a loop."""

    def test_vmap_difference_laplacian_matches_loop(self, grid2d):
        """Laplacian of each field in a batch must match per-field eager calls."""
        diff = Difference2D(grid=grid2d)
        rng = jax.random.PRNGKey(0)
        batch = jax.random.normal(rng, (4, grid2d.Ny, grid2d.Nx))

        # Loop reference
        loop_results = jnp.stack([diff.laplacian(batch[i]) for i in range(4)])
        # vmap result
        vmap_result = jax.vmap(diff.laplacian)(batch)

        np.testing.assert_allclose(vmap_result, loop_results, rtol=1e-7)

    def test_vmap_interpolation_matches_loop(self, grid2d):
        """Interpolation of each field in a batch must match per-field eager calls."""
        interp = Interpolation2D(grid=grid2d)
        rng = jax.random.PRNGKey(1)
        batch = jax.random.normal(rng, (3, grid2d.Ny, grid2d.Nx))

        loop_results = jnp.stack([interp.T_to_U(batch[i]) for i in range(3)])
        vmap_result = jax.vmap(interp.T_to_U)(batch)

        np.testing.assert_allclose(vmap_result, loop_results, rtol=1e-7)

    def test_vmap_divergence2d_matches_loop(self, grid2d):
        """Divergence2D vmapped over a batch must match per-field eager calls."""
        div_op = Divergence2D(grid=grid2d)
        rng = jax.random.PRNGKey(2)
        u_batch = jax.random.normal(rng, (4, grid2d.Ny, grid2d.Nx))
        v_batch = jax.random.normal(jax.random.PRNGKey(3), (4, grid2d.Ny, grid2d.Nx))

        loop_results = jnp.stack([div_op(u_batch[i], v_batch[i]) for i in range(4)])
        vmap_result = jax.vmap(div_op)(u_batch, v_batch)

        np.testing.assert_allclose(vmap_result, loop_results, rtol=1e-7)

    def test_vmap_diffusion2d_matches_loop(self, grid2d):
        """Diffusion2D vmapped over a batch must match per-field eager calls."""
        diff_op = Diffusion2D(grid=grid2d)
        rng = jax.random.PRNGKey(4)
        batch = jax.random.normal(rng, (4, grid2d.Ny, grid2d.Nx))
        kappa = 1e-2

        loop_results = jnp.stack([diff_op(batch[i], kappa=kappa) for i in range(4)])
        vmap_result = jax.vmap(lambda h: diff_op(h, kappa=kappa))(batch)

        np.testing.assert_allclose(vmap_result, loop_results, rtol=1e-7)
