"""Spatial convergence order tests for finite-difference operators.

For each operator we measure the max-norm error at two grid resolutions
(N and 2N) and verify the error ratio is approximately 2^order (order=2 for
second-order stencils).

Only interior-point values are compared so that ghost-cell contamination
does not pollute the convergence measurement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx._src.grid.cartesian import CartesianGrid1D, CartesianGrid2D
from finitevolx._src.operators.difference import Difference1D, Difference2D

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_error(result: jnp.ndarray, exact: jnp.ndarray) -> float:
    """Max absolute error over the physical interior."""
    return float(jnp.max(jnp.abs(result - exact)))


# ---------------------------------------------------------------------------
# 1-D convergence
# ---------------------------------------------------------------------------


class TestConvergence1D:
    """The Difference1D operators must converge at the expected order."""

    def test_diff_x_T_to_U_convergence_linear_is_exact(self):
        """For linear h=c*x the first-order difference is machine-exact at any N.

        We just verify the interior error is near machine precision for two
        different grid sizes (convergence is trivially satisfied for linear
        fields because the stencil is exact).
        """
        c = 3.0
        for n in [16, 32]:
            grid = CartesianGrid1D.from_interior(n, 1.0)
            diff = Difference1D(grid=grid)
            x = jnp.arange(grid.Nx, dtype=float) * grid.dx
            h = c * x
            result = diff.diff_x_T_to_U(h)
            np.testing.assert_allclose(result[1:-1], c, atol=1e-10)

    def test_laplacian_convergence_quadratic(self):
        """Laplacian of h=x^2 is exactly 2 at any resolution (exact for quadratics)."""
        for n in [16, 32, 64]:
            grid = CartesianGrid1D.from_interior(n, 1.0)
            diff = Difference1D(grid=grid)
            x = jnp.arange(grid.Nx, dtype=float) * grid.dx
            h = x**2
            result = diff.laplacian(h)
            np.testing.assert_allclose(result[1:-1], 2.0, atol=1e-8)

    def test_second_order_convergence_sin(self):
        """Laplacian of h=sin(2πx/L) should converge at O(dx²).

        We test the error ratio between N=16 and N=32 is ≥ 3.5 (close to 4).
        """
        L = 1.0
        k = 2.0 * np.pi / L

        errors = []
        for n in [16, 32]:
            grid = CartesianGrid1D.from_interior(n, L)
            diff = Difference1D(grid=grid)
            x = (jnp.arange(grid.Nx, dtype=float) + 0.5) * grid.dx
            h = jnp.sin(k * x)
            result = diff.laplacian(h)
            exact = -(k**2) * jnp.sin(k * x)
            # Compare only deep interior to avoid ghost-cell edge effects
            errors.append(_max_error(result[2:-2], exact[2:-2]))

        ratio = errors[0] / errors[1]
        assert ratio >= 3.5, (
            f"1D Laplacian convergence ratio {ratio:.2f} < 3.5 (expected ~4 for O(dx²))"
        )


# ---------------------------------------------------------------------------
# 2-D convergence
# ---------------------------------------------------------------------------


class TestConvergence2D:
    """Difference2D operators must converge at the expected spatial order."""

    def test_diff_x_T_to_U_convergence_linear_exact(self):
        """For linear h=c*x the forward x-difference is exact for any N."""
        c = 2.5
        for n in [8, 16]:
            grid = CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
            diff = Difference2D(grid=grid)
            x = jnp.arange(grid.Nx, dtype=float) * grid.dx
            h = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
            result = diff.diff_x_T_to_U(h)
            np.testing.assert_allclose(result[1:-1, 1:-1], c, atol=1e-10)

    def test_diff_y_T_to_V_convergence_linear_exact(self):
        c = 1.7
        for n in [8, 16]:
            grid = CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
            diff = Difference2D(grid=grid)
            y = jnp.arange(grid.Ny, dtype=float) * grid.dy
            h = jnp.broadcast_to(c * y[:, None], (grid.Ny, grid.Nx))
            result = diff.diff_y_T_to_V(h)
            np.testing.assert_allclose(result[1:-1, 1:-1], c, atol=1e-10)

    def test_laplacian_convergence_quadratic_exact(self):
        """Laplacian of h = x^2 + y^2 is exactly 4 at any resolution."""
        for n in [8, 16, 32]:
            grid = CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
            diff = Difference2D(grid=grid)
            x = jnp.arange(grid.Nx, dtype=float) * grid.dx
            y = jnp.arange(grid.Ny, dtype=float) * grid.dy
            h = x[None, :] ** 2 + y[:, None] ** 2
            result = diff.laplacian(h)
            np.testing.assert_allclose(result[1:-1, 1:-1], 4.0, atol=1e-8)

    def test_laplacian_second_order_convergence_sin(self):
        """Laplacian of h=sin(kx)sin(ky) converges at O(dx²).

        For h = sin(kx)sin(ky), ∇²h = -2k²·h.
        Error at N=32 should be ≥ 3.5× smaller than at N=16.
        """
        L = 2.0 * np.pi
        k = 1.0

        errors = []
        for n in [16, 32]:
            grid = CartesianGrid2D.from_interior(n, n, L, L)
            diff = Difference2D(grid=grid)
            x = (jnp.arange(grid.Nx, dtype=float) + 0.5) * grid.dx
            y = (jnp.arange(grid.Ny, dtype=float) + 0.5) * grid.dy
            h = jnp.sin(k * x[None, :]) * jnp.sin(k * y[:, None])
            result = diff.laplacian(h)
            exact = -2.0 * k**2 * h
            # Compare deep interior to avoid stencil boundary effects
            errors.append(_max_error(result[2:-2, 2:-2], exact[2:-2, 2:-2]))

        ratio = errors[0] / errors[1]
        assert ratio >= 3.5, (
            f"2D Laplacian convergence ratio {ratio:.2f} < 3.5 (expected ~4 for O(dx²))"
        )

    def test_divergence_convergence_linear_exact(self):
        """For u=c*x, v=c*y divergence = 2c exactly at any N."""
        c = 2.0
        for n in [8, 16]:
            grid = CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
            diff = Difference2D(grid=grid)
            x = jnp.arange(grid.Nx, dtype=float) * grid.dx
            y = jnp.arange(grid.Ny, dtype=float) * grid.dy
            u = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
            v = jnp.broadcast_to(c * y[:, None], (grid.Ny, grid.Nx))
            result = diff.divergence(u, v)
            np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c, atol=1e-8)

    def test_curl_convergence_solid_body_exact(self):
        """For solid-body rotation u=-c*y, v=c*x, curl = 2c exactly at any N."""
        c = 1.5
        for n in [8, 16]:
            grid = CartesianGrid2D.from_interior(n, n, 1.0, 1.0)
            diff = Difference2D(grid=grid)
            x = jnp.arange(grid.Nx, dtype=float) * grid.dx
            y = jnp.arange(grid.Ny, dtype=float) * grid.dy
            u = jnp.broadcast_to(-c * y[:, None], (grid.Ny, grid.Nx))
            v = jnp.broadcast_to(c * x, (grid.Ny, grid.Nx))
            result = diff.curl(u, v)
            np.testing.assert_allclose(result[1:-1, 1:-1], 2.0 * c, rtol=1e-5)

    def test_diff_x_first_order_convergence_sin(self):
        """∂h/∂x for h=sin(kx) converges at O(dx) (first-order accurate).

        diff_x_T_to_U is a forward difference (first-order stencil):
            dh/dx[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx

        When N doubles (dx halves), the error should roughly halve (ratio ≈ 2).
        We require ratio ≥ 1.8 as a conservative lower bound.
        """
        L = 2.0 * np.pi
        k = 1.0

        errors = []
        for n in [16, 32]:
            grid = CartesianGrid2D.from_interior(n, n, L, L)
            diff = Difference2D(grid=grid)
            x = (jnp.arange(grid.Nx, dtype=float) + 0.5) * grid.dx
            h = jnp.broadcast_to(jnp.sin(k * x), (grid.Ny, grid.Nx))
            result = diff.diff_x_T_to_U(h)
            exact = jnp.broadcast_to(k * jnp.cos(k * x), (grid.Ny, grid.Nx))
            errors.append(_max_error(result[2:-2, 2:-2], exact[2:-2, 2:-2]))

        ratio = errors[0] / errors[1]
        assert ratio >= 1.8, (
            f"diff_x convergence ratio {ratio:.2f} < 1.8 (expected ~2 for O(dx) first-order)"
        )
