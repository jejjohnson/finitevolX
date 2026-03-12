"""Tests for tridiagonal (TDMA) solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx._src.solvers.tridiagonal import (
    solve_tridiagonal,
    solve_tridiagonal_batched,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# solve_tridiagonal (single system)
# ---------------------------------------------------------------------------


class TestSolveTridiagonal:
    def test_known_solution(self):
        """Diagonally dominant 4×4 system with known solution."""
        # System:  4 1 0 0   x   7     x   1
        #          1 4 1 0 * x = 14  → x = 2
        #          0 1 4 1   x   21    x = 3
        #          0 0 1 4   x   16    x = 4  (but non-trivial due to coupling)
        b = jnp.array([4.0, 4.0, 4.0, 4.0])
        a = jnp.array([1.0, 1.0, 1.0])
        c = jnp.array([1.0, 1.0, 1.0])
        x_expect = jnp.array([1.0, 2.0, 3.0, 4.0])
        d = jnp.array(
            [
                4 * 1 + 1 * 2,
                1 * 1 + 4 * 2 + 1 * 3,
                1 * 2 + 4 * 3 + 1 * 4,
                1 * 3 + 4 * 4,
            ],
            dtype=jnp.float64,
        )
        x = solve_tridiagonal(a, b, c, d)
        np.testing.assert_allclose(np.array(x), np.array(x_expect), atol=1e-12)

    def test_diagonal_system(self):
        """Pure diagonal matrix (a=0, c=0) reduces to element-wise division."""
        n = 5
        b = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        a = jnp.zeros(n - 1)
        c = jnp.zeros(n - 1)
        d = jnp.array([4.0, 9.0, 16.0, 25.0, 36.0])
        x = solve_tridiagonal(a, b, c, d)
        np.testing.assert_allclose(np.array(x), np.array(d / b), atol=1e-14)

    def test_output_shape(self):
        n = 7
        a = jnp.ones(n - 1)
        b = 4.0 * jnp.ones(n)
        c = jnp.ones(n - 1)
        d = jnp.ones(n)
        x = solve_tridiagonal(a, b, c, d)
        assert x.shape == (n,)

    def test_roundtrip_random(self):
        """Solve A x = d, then verify A @ x ≈ d for a random system."""
        key = jax.random.PRNGKey(42)
        n = 20
        k1, k2, k3, k4 = jax.random.split(key, 4)
        a = jax.random.normal(k1, (n - 1,))
        b = 5.0 + jax.random.normal(k2, (n,))  # diag-dominant
        c = jax.random.normal(k3, (n - 1,))
        d = jax.random.normal(k4, (n,))

        x = solve_tridiagonal(a, b, c, d)

        # Reconstruct A @ x
        Ax = b * x
        Ax = Ax.at[1:].add(a * x[:-1])
        Ax = Ax.at[:-1].add(c * x[1:])
        np.testing.assert_allclose(np.array(Ax), np.array(d), atol=1e-10)

    def test_size_2(self):
        """Minimal non-trivial system (2×2)."""
        b = jnp.array([3.0, 3.0])
        a = jnp.array([1.0])
        c = jnp.array([1.0])
        # [3 1] [x0]   [4]     x0 = 1
        # [1 3] [x1] = [4]  →  x1 = 1
        d = jnp.array([4.0, 4.0])
        x = solve_tridiagonal(a, b, c, d)
        np.testing.assert_allclose(np.array(x), np.array([1.0, 1.0]), atol=1e-12)

    def test_jit_compatible(self):
        """Solver works under jax.jit."""

        @jax.jit
        def solve(a, b, c, d):
            return solve_tridiagonal(a, b, c, d)

        b = jnp.array([4.0, 4.0, 4.0])
        a = jnp.ones(2)
        c = jnp.ones(2)
        d = jnp.array([5.0, 6.0, 5.0])
        x = solve(a, b, c, d)
        assert x.shape == (3,)

    def test_vertical_diffusion_pattern(self):
        """Implicit vertical diffusion: -κ u_{k-1} + (1+2κ) u_k - κ u_{k+1} = rhs."""
        nz = 10
        kappa = 0.25
        b = (1.0 + 2.0 * kappa) * jnp.ones(nz)
        a = -kappa * jnp.ones(nz - 1)
        c = -kappa * jnp.ones(nz - 1)
        # Neumann BC at top/bottom: adjust first/last diagonal
        b = b.at[0].set(1.0 + kappa)
        b = b.at[-1].set(1.0 + kappa)

        rhs = jnp.ones(nz)
        x = solve_tridiagonal(a, b, c, rhs)

        # Verify the system: A @ x ≈ rhs
        Ax = b * x
        Ax = Ax.at[1:].add(a * x[:-1])
        Ax = Ax.at[:-1].add(c * x[1:])
        np.testing.assert_allclose(np.array(Ax), np.array(rhs), atol=1e-12)


# ---------------------------------------------------------------------------
# solve_tridiagonal_batched
# ---------------------------------------------------------------------------


class TestSolveTridiagonalBatched:
    def test_single_batch_dim(self):
        """Batched solve over one leading dimension."""
        n_cols = 5
        nz = 8
        b = 4.0 * jnp.ones((n_cols, nz))
        a = jnp.ones((n_cols, nz - 1))
        c = jnp.ones((n_cols, nz - 1))
        d = jnp.ones((n_cols, nz))

        x = solve_tridiagonal_batched(a, b, c, d)
        assert x.shape == (n_cols, nz)

        # Verify each column
        for i in range(n_cols):
            x_i = solve_tridiagonal(a[i], b[i], c[i], d[i])
            np.testing.assert_allclose(np.array(x[i]), np.array(x_i), atol=1e-14)

    def test_two_batch_dims(self):
        """Batched solve over (Ny, Nx) horizontal grid."""
        Ny, Nx, Nz = 3, 4, 6
        b = 4.0 * jnp.ones((Ny, Nx, Nz))
        a = jnp.ones((Ny, Nx, Nz - 1))
        c = jnp.ones((Ny, Nx, Nz - 1))
        d = jnp.ones((Ny, Nx, Nz))

        x = solve_tridiagonal_batched(a, b, c, d)
        assert x.shape == (Ny, Nx, Nz)

    def test_batched_matches_loop(self):
        """Batched result matches looping over columns."""
        key = jax.random.PRNGKey(7)
        Ny, Nx, Nz = 2, 3, 5
        k1, k2, k3, k4 = jax.random.split(key, 4)
        a = jax.random.normal(k1, (Ny, Nx, Nz - 1))
        b = 5.0 + jax.random.uniform(k2, (Ny, Nx, Nz))
        c = jax.random.normal(k3, (Ny, Nx, Nz - 1))
        d = jax.random.normal(k4, (Ny, Nx, Nz))

        x_batched = solve_tridiagonal_batched(a, b, c, d)

        for j in range(Ny):
            for i in range(Nx):
                x_ji = solve_tridiagonal(a[j, i], b[j, i], c[j, i], d[j, i])
                np.testing.assert_allclose(
                    np.array(x_batched[j, i]),
                    np.array(x_ji),
                    atol=1e-12,
                )

    def test_no_batch_dims(self):
        """With 1-D inputs (no batch), behaves like solve_tridiagonal."""
        b = jnp.array([4.0, 4.0, 4.0])
        a = jnp.ones(2)
        c = jnp.ones(2)
        d = jnp.array([5.0, 6.0, 5.0])
        x_single = solve_tridiagonal(a, b, c, d)
        x_batched = solve_tridiagonal_batched(a, b, c, d)
        np.testing.assert_allclose(np.array(x_batched), np.array(x_single), atol=1e-14)

    def test_jit_compatible(self):
        """Batched solver works under jax.jit."""

        @jax.jit
        def solve(a, b, c, d):
            return solve_tridiagonal_batched(a, b, c, d)

        Ny, Nx, Nz = 4, 6, 10
        b = 4.0 * jnp.ones((Ny, Nx, Nz))
        a = jnp.ones((Ny, Nx, Nz - 1))
        c = jnp.ones((Ny, Nx, Nz - 1))
        d = jnp.ones((Ny, Nx, Nz))
        x = solve(a, b, c, d)
        assert x.shape == (Ny, Nx, Nz)
