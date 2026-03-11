"""Tests for the Arakawa (1966) Jacobian operator."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.jacobian import arakawa_jacobian


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _periodic_field(Ny: int, Nx: int, func) -> jnp.ndarray:
    """Return a field on [Ny, Nx] with periodic ghost cells set from *func*.

    *func(J, I)* is evaluated on the physical interior rows j=1..Ny-2 and
    columns i=1..Nx-2 using normalised coordinates J/(Ny-2) and I/(Nx-2).
    Ghost cells are filled so that the field is exactly periodic over the
    interior with periods (Ny-2) in j and (Nx-2) in i.
    """
    Nyi = float(Ny - 2)
    Nxi = float(Nx - 2)
    j = jnp.arange(Ny, dtype=float)
    i = jnp.arange(Nx, dtype=float)
    I, J = jnp.meshgrid(i, j)
    f = func(J / Nyi, I / Nxi)
    # Periodic ghost rows: south ghost = last interior row, north ghost = first
    f = f.at[0, :].set(f[Ny - 2, :])
    f = f.at[Ny - 1, :].set(f[1, :])
    # Periodic ghost cols: west ghost = last interior col, east ghost = first
    f = f.at[:, 0].set(f[:, Nx - 2])
    f = f.at[:, Nx - 1].set(f[:, 1])
    return f


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_2d():
    """Return a 12x10 regular grid with halo, dx=1, dy=1."""
    Ny, Nx = 12, 10
    dx, dy = 1.0, 1.0
    x = jnp.arange(Nx, dtype=float)
    y = jnp.arange(Ny, dtype=float)
    X, Y = jnp.meshgrid(x, y)
    return X, Y, dx, dy


@pytest.fixture
def grid_nonsquare():
    """Return a 14x12 regular grid with dx=0.5, dy=1.0."""
    Ny, Nx = 14, 12
    dx, dy = 0.5, 1.0
    x = jnp.arange(Nx, dtype=float) * dx
    y = jnp.arange(Ny, dtype=float) * dy
    X, Y = jnp.meshgrid(x, y)
    return X, Y, dx, dy


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestArakawaJacobianShape:
    def test_output_shape_2d(self, grid_2d):
        X, Y, dx, dy = grid_2d
        Ny, Nx = X.shape
        J = arakawa_jacobian(X, Y, dx, dy)
        assert J.shape == (Ny - 2, Nx - 2)

    def test_output_shape_nonsquare(self, grid_nonsquare):
        X, Y, dx, dy = grid_nonsquare
        Ny, Nx = X.shape
        J = arakawa_jacobian(X, Y, dx, dy)
        assert J.shape == (Ny - 2, Nx - 2)

    def test_output_shape_batched(self):
        """Batch dimension is passed through unchanged."""
        Nz, Ny, Nx = 4, 12, 10
        f = jnp.ones((Nz, Ny, Nx))
        g = jnp.ones((Nz, Ny, Nx))
        J = arakawa_jacobian(f, g, 1.0, 1.0)
        assert J.shape == (Nz, Ny - 2, Nx - 2)


class TestArakawaJacobianSymmetry:
    def test_antisymmetry(self, grid_2d):
        """J(f, g) = -J(g, f) up to float32 rounding."""
        X, Y, dx, dy = grid_2d
        f = jnp.sin(X) * jnp.cos(Y)
        g = jnp.cos(X) * jnp.sin(Y)
        Jfg = arakawa_jacobian(f, g, dx, dy)
        Jgf = arakawa_jacobian(g, f, dx, dy)
        # Algebraically exact; tolerance covers float32 rounding (~1 ULP ~1e-7)
        np.testing.assert_allclose(Jfg, -Jgf, atol=1e-7)

    def test_self_jacobian_zero(self, grid_2d):
        """J(f, f) = 0 for any f (algebraic identity)."""
        X, Y, dx, dy = grid_2d
        f = jnp.sin(X + Y)
        J = arakawa_jacobian(f, f, dx, dy)
        # Exactly zero in exact arithmetic; float32 rounding ~1e-8
        np.testing.assert_allclose(J, 0.0, atol=1e-7)

    def test_self_jacobian_zero_nonconst(self, grid_nonsquare):
        """J(f, f) = 0 on a non-square grid."""
        X, Y, dx, dy = grid_nonsquare
        f = X**2 + Y**2
        J = arakawa_jacobian(f, f, dx, dy)
        np.testing.assert_allclose(J, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Conservation properties (require periodic boundary conditions)
# ---------------------------------------------------------------------------


class TestArakawaJacobianConservation:
    """Conservation proofs require periodic BCs; ghost cells are set accordingly."""

    def test_integral_vanishes(self):
        """∫∫ J(f, g) dA = 0 with periodic BCs."""
        Ny, Nx = 12, 10
        f = _periodic_field(Ny, Nx, lambda j, i: jnp.sin(2 * jnp.pi * j) * jnp.cos(2 * jnp.pi * i))
        g = _periodic_field(Ny, Nx, lambda j, i: jnp.cos(2 * jnp.pi * j) + jnp.sin(2 * jnp.pi * i))
        J = arakawa_jacobian(f, g, 1.0, 1.0)
        # Algebraically zero for periodic BCs; float32 precision ~1e-6
        np.testing.assert_allclose(float(jnp.sum(J)), 0.0, atol=1e-6)

    def test_energy_conservation(self):
        """∫∫ f · J(f, g) dA = 0 with periodic BCs."""
        Ny, Nx = 12, 10
        f = _periodic_field(Ny, Nx, lambda j, i: jnp.sin(2 * jnp.pi * j) * jnp.cos(2 * jnp.pi * i))
        g = _periodic_field(Ny, Nx, lambda j, i: jnp.cos(2 * jnp.pi * j) + jnp.sin(2 * jnp.pi * i))
        J = arakawa_jacobian(f, g, 1.0, 1.0)
        f_int = f[1:-1, 1:-1]
        np.testing.assert_allclose(float(jnp.sum(f_int * J)), 0.0, atol=1e-6)

    def test_enstrophy_conservation(self):
        """∫∫ g · J(f, g) dA = 0 with periodic BCs."""
        Ny, Nx = 12, 10
        f = _periodic_field(Ny, Nx, lambda j, i: jnp.sin(2 * jnp.pi * j) * jnp.cos(2 * jnp.pi * i))
        g = _periodic_field(Ny, Nx, lambda j, i: jnp.cos(2 * jnp.pi * j) + jnp.sin(2 * jnp.pi * i))
        J = arakawa_jacobian(f, g, 1.0, 1.0)
        g_int = g[1:-1, 1:-1]
        np.testing.assert_allclose(float(jnp.sum(g_int * J)), 0.0, atol=1e-6)

    def test_integral_vanishes_nonsquare(self):
        """∫∫ J(f, g) dA = 0 holds for dx ≠ dy with periodic BCs."""
        Ny, Nx = 14, 12
        f = _periodic_field(Ny, Nx, lambda j, i: jnp.sin(2 * jnp.pi * j) * jnp.cos(2 * jnp.pi * i))
        g = _periodic_field(Ny, Nx, lambda j, i: jnp.cos(2 * jnp.pi * j) + jnp.sin(2 * jnp.pi * i))
        J = arakawa_jacobian(f, g, 0.5, 1.0)
        # Slightly wider tolerance for non-square grids due to dx scaling in float32
        np.testing.assert_allclose(float(jnp.sum(J)), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Analytical checks
# ---------------------------------------------------------------------------


class TestArakawaJacobianAnalytical:
    def test_linear_fields_known_jacobian(self):
        """J(x, y) = 1 analytically; Arakawa scheme is exact for linear f, g."""
        Ny, Nx = 20, 20
        dx, dy = 1.0, 1.0
        x = jnp.arange(Nx, dtype=float)
        y = jnp.arange(Ny, dtype=float)
        X, Y = jnp.meshgrid(x, y)
        # J(x, y) = ∂x/∂x * ∂y/∂y - ∂x/∂y * ∂y/∂x = 1*1 - 0*0 = 1
        J = arakawa_jacobian(X, Y, dx, dy)
        np.testing.assert_allclose(J, 1.0, atol=1e-6)

    def test_quadratic_known_jacobian(self):
        """J(x*y, x^2) = -2x^2 analytically; check against expected values."""
        Ny, Nx = 30, 30
        dx, dy = 0.1, 0.1
        x = jnp.arange(Nx, dtype=float) * dx
        y = jnp.arange(Ny, dtype=float) * dy
        X, Y = jnp.meshgrid(x, y)
        # f = x*y,  g = x^2
        # J(f,g) = ∂(xy)/∂x * ∂(x²)/∂y - ∂(xy)/∂y * ∂(x²)/∂x
        #        = y * 0 - x * 2x = -2x²
        f = X * Y
        g = X**2
        J = arakawa_jacobian(f, g, dx, dy)
        X_int = X[1:-1, 1:-1]
        expected = -2.0 * X_int**2
        # Second-order accurate; O(h²) truncation error
        np.testing.assert_allclose(J, expected, atol=dx**2 * 5)

    def test_constant_g_zero_jacobian(self):
        """J(f, c) = 0 when g is constant."""
        Ny, Nx = 12, 10
        f = jnp.arange(Ny * Nx, dtype=float).reshape(Ny, Nx)
        g = jnp.full((Ny, Nx), 3.14)
        J = arakawa_jacobian(f, g, 1.0, 1.0)
        np.testing.assert_allclose(J, 0.0, atol=1e-6)

    def test_constant_f_zero_jacobian(self):
        """J(c, g) = 0 when f is constant."""
        Ny, Nx = 12, 10
        f = jnp.full((Ny, Nx), 2.71)
        g = jnp.arange(Ny * Nx, dtype=float).reshape(Ny, Nx)
        J = arakawa_jacobian(f, g, 1.0, 1.0)
        np.testing.assert_allclose(J, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# JAX compatibility
# ---------------------------------------------------------------------------


class TestArakawaJacobianJAX:
    def test_jit_compatible(self, grid_2d):
        """Function can be compiled with jax.jit."""
        import jax

        X, Y, dx, dy = grid_2d
        f = jnp.sin(X) * jnp.cos(Y)
        g = jnp.cos(X) * jnp.sin(Y)
        J_eager = arakawa_jacobian(f, g, dx, dy)
        J_jit = jax.jit(arakawa_jacobian, static_argnums=(2, 3))(f, g, dx, dy)
        # JIT and eager produce bit-identical results in practice;
        # 1e-7 covers any reordering differences in float32
        np.testing.assert_allclose(J_jit, J_eager, atol=1e-7)

    def test_batched_matches_loop(self):
        """Batched call matches looping over the batch dimension."""
        Nz, Ny, Nx = 3, 12, 10
        f = jnp.stack([jnp.ones((Ny, Nx)) * (k + 1) for k in range(Nz)])
        x = jnp.arange(Nx, dtype=float)
        y = jnp.arange(Ny, dtype=float)
        X, Y = jnp.meshgrid(x, y)
        g = jnp.broadcast_to(jnp.sin(X) * jnp.cos(Y), (Nz, Ny, Nx))

        J_batched = arakawa_jacobian(f, g, 1.0, 1.0)
        J_loop = jnp.stack(
            [arakawa_jacobian(f[k], g[k], 1.0, 1.0) for k in range(Nz)]
        )
        np.testing.assert_allclose(J_batched, J_loop, atol=1e-6)
