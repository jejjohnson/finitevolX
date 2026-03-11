"""Tests for kinetic_energy and bernoulli_potential on Arakawa C-grids.

All arrays share the same shape [Ny, Nx] with one ghost-cell ring on each
side.  U-points live at east faces (j, i+1/2), V-points at north faces
(j+1/2, i), and T-points at cell centres (j, i).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.grid.constants import GRAVITY
from finitevolx._src.grid.grid import ArakawaCGrid2D
from finitevolx._src.operators.operators import (
    bernoulli_potential,
    kinetic_energy,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def grid2d():
    # 8 interior cells in each direction → total shape [10, 10]
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


# ---------------------------------------------------------------------------
# kinetic_energy
# ---------------------------------------------------------------------------


class TestKineticEnergy2D:
    def test_constant_unity_field(self, grid2d):
        """Constant u=v=1 everywhere → ke = 1.0 at interior."""
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        ke = kinetic_energy(u=u, v=v)

        assert ke.shape == (grid2d.Ny, grid2d.Nx)
        # u²_on_T = 0.5*(1² + 1²) = 1, v²_on_T = 0.5*(1² + 1²) = 1
        # ke = 0.5*(u²_on_T + v²_on_T) = 0.5*(1 + 1) = 1.0
        np.testing.assert_allclose(ke[1:-1, 1:-1], 1.0, rtol=1e-6)

    def test_ghost_ring_is_zero(self, grid2d):
        """Ghost cells must remain zero (interior-point idiom)."""
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        ke = kinetic_energy(u=u, v=v)

        np.testing.assert_array_equal(ke[0, :], 0.0)
        np.testing.assert_array_equal(ke[-1, :], 0.0)
        np.testing.assert_array_equal(ke[:, 0], 0.0)
        np.testing.assert_array_equal(ke[:, -1], 0.0)

    def test_output_shape(self, grid2d):
        """ke must have the same shape as u and v."""
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        ke = kinetic_energy(u=u, v=v)
        assert ke.shape == (grid2d.Ny, grid2d.Nx)

    def test_nonconst_u_linear_in_x(self, grid2d):
        """u varies linearly in x; v=0 → checks U→T averaging along x-axis.

        u[j, i] = i  (U-point at east face i+1/2)
        u²_on_T[j, i] = 0.5*(u[j,i]² + u[j,i-1]²) = 0.5*(i² + (i-1)²)
        ke[j, i] = 0.5 * u²_on_T[j, i]
        """
        ix = jnp.arange(grid2d.Nx, dtype=float)
        u = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))

        ke = kinetic_energy(u=u, v=v)

        # expected for interior columns i=1..Nx-2
        u2 = ix**2
        # u²_on_T[i] = 0.5*(u2[i] + u2[i-1])
        u2_on_T = 0.5 * (u2[1:-1] + u2[:-2])
        expected = 0.5 * u2_on_T  # ke = 0.5 * u²_on_T (v=0)

        np.testing.assert_allclose(ke[1, 1:-1], expected, rtol=1e-6)

    def test_nonconst_v_linear_in_y(self, grid2d):
        """v varies linearly in y; u=0 → checks V→T averaging along y-axis.

        v[j, i] = j  (V-point at north face j+1/2)
        v²_on_T[j, i] = 0.5*(v[j,i]² + v[j-1,i]²) = 0.5*(j² + (j-1)²)
        ke[j, i] = 0.5 * v²_on_T[j, i]
        """
        jy = jnp.arange(grid2d.Ny, dtype=float)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))

        ke = kinetic_energy(u=u, v=v)

        # expected for interior rows j=1..Ny-2
        v2 = jy**2
        # v²_on_T[j] = 0.5*(v2[j] + v2[j-1])
        v2_on_T = 0.5 * (v2[1:-1] + v2[:-2])
        expected = 0.5 * v2_on_T  # ke = 0.5 * v²_on_T (u=0)

        # v is constant in x → all interior columns are identical; compare one column
        np.testing.assert_allclose(ke[1:-1, 1], expected, rtol=1e-6)

    def test_nonconst_uv_combined(self, grid2d):
        """u and v both vary; verifies combined averaging is independent per axis."""
        ix = jnp.arange(grid2d.Nx, dtype=float)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        u = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))

        ke = kinetic_energy(u=u, v=v)

        u2 = ix**2
        u2_on_T = 0.5 * (u2[1:-1] + u2[:-2])  # shape [Nx-2]
        v2 = jy**2
        v2_on_T = 0.5 * (v2[1:-1] + v2[:-2])  # shape [Ny-2]
        expected = 0.5 * (u2_on_T[None, :] + v2_on_T[:, None])

        np.testing.assert_allclose(ke[1:-1, 1:-1], expected, rtol=1e-6)

    def test_integer_inputs_preserve_fractional_output(self, grid2d):
        """Integer inputs should still produce floating kinetic energy."""
        u = jnp.ones((grid2d.Ny, grid2d.Nx), dtype=jnp.int32)
        v = jnp.zeros((grid2d.Ny, grid2d.Nx), dtype=jnp.int32)

        ke = kinetic_energy(u=u, v=v)

        assert jnp.issubdtype(ke.dtype, jnp.floating)
        np.testing.assert_allclose(ke[1:-1, 1:-1], 0.5, rtol=1e-6)


# ---------------------------------------------------------------------------
# bernoulli_potential
# ---------------------------------------------------------------------------


class TestBernoulliPotential2D:
    def test_constant_unity_fields(self, grid2d):
        """u=v=h=1 everywhere → p = ke + g*h = 1.0 + g at interior."""
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        h = jnp.ones((grid2d.Ny, grid2d.Nx))

        p = bernoulli_potential(h=h, u=u, v=v)

        assert p.shape == (grid2d.Ny, grid2d.Nx)
        np.testing.assert_allclose(p[1:-1, 1:-1], 1.0 + GRAVITY * 1.0, rtol=1e-6)

    def test_ghost_ring_is_zero(self, grid2d):
        """Ghost cells must remain zero (interior-point idiom)."""
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        h = jnp.ones((grid2d.Ny, grid2d.Nx))
        p = bernoulli_potential(h=h, u=u, v=v)

        np.testing.assert_array_equal(p[0, :], 0.0)
        np.testing.assert_array_equal(p[-1, :], 0.0)
        np.testing.assert_array_equal(p[:, 0], 0.0)
        np.testing.assert_array_equal(p[:, -1], 0.0)

    def test_output_shape(self, grid2d):
        """p must have the same shape as h, u, v."""
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        h = jnp.zeros((grid2d.Ny, grid2d.Nx))
        p = bernoulli_potential(h=h, u=u, v=v)
        assert p.shape == (grid2d.Ny, grid2d.Nx)

    def test_equals_ke_plus_gh(self, grid2d):
        """p == ke(u,v) + g*h elementwise at interior, with spatially varying fields."""
        ix = jnp.arange(grid2d.Nx, dtype=float)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        u = jnp.broadcast_to(ix, (grid2d.Ny, grid2d.Nx))
        v = jnp.broadcast_to(jy[:, None], (grid2d.Ny, grid2d.Nx))
        # h: distinct values to avoid accidental cancellation
        h = (ix[None, :] + jy[:, None]) * 0.1

        p = bernoulli_potential(h=h, u=u, v=v)
        ke = kinetic_energy(u=u, v=v)

        expected = ke[1:-1, 1:-1] + GRAVITY * h[1:-1, 1:-1]
        np.testing.assert_allclose(p[1:-1, 1:-1], expected, rtol=1e-6)

    def test_zero_velocity_gives_gravity_term_only(self, grid2d):
        """u=v=0 → ke=0 → p = g*h at interior."""
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        ix = jnp.arange(grid2d.Nx, dtype=float)
        jy = jnp.arange(grid2d.Ny, dtype=float)
        h = ix[None, :] + jy[:, None]

        p = bernoulli_potential(h=h, u=u, v=v)
        np.testing.assert_allclose(p[1:-1, 1:-1], GRAVITY * h[1:-1, 1:-1], rtol=1e-6)

    def test_integer_inputs_preserve_fractional_output(self, grid2d):
        """Integer h/u/v should still produce floating Bernoulli potential."""
        u = jnp.ones((grid2d.Ny, grid2d.Nx), dtype=jnp.int32)
        v = jnp.zeros((grid2d.Ny, grid2d.Nx), dtype=jnp.int32)
        h = jnp.ones((grid2d.Ny, grid2d.Nx), dtype=jnp.int32)

        p = bernoulli_potential(h=h, u=u, v=v)

        assert jnp.issubdtype(p.dtype, jnp.floating)
        np.testing.assert_allclose(p[1:-1, 1:-1], GRAVITY + 0.5, rtol=1e-6)


# ---------------------------------------------------------------------------
# Operator identity / compatibility tests
# ---------------------------------------------------------------------------


class TestOperatorIdentities:
    """Discrete identities that must hold for any physically consistent
    implementation of divergence, curl, and gradient operators."""

    def test_divergence_of_curl_is_zero(self, grid2d):
        """div(curl(ψ)) = 0 for any streamfunction ψ.

        u = -∂ψ/∂y  (from X-point via diff_y_X_to_U)
        v =  ∂ψ/∂x  (from X-point via diff_x_X_to_V)
        div(u, v) = ∂u/∂x + ∂v/∂y = -∂²ψ/∂x∂y + ∂²ψ/∂x∂y = 0
        """
        from finitevolx._src.operators.difference import Difference2D

        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float)
        y = jnp.arange(grid2d.Ny, dtype=float)
        psi = y[:, None] * x[None, :]  # ψ = x*y

        u = -diff.diff_y_X_to_U(psi)
        v = diff.diff_x_X_to_V(psi)
        divergence = diff.divergence(u, v)

        np.testing.assert_allclose(divergence[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_curl_of_gradient_is_zero(self, grid2d):
        """curl(grad(ϕ)) = 0 for any potential ϕ.

        u = ∂ϕ/∂x (T→U forward)
        v = ∂ϕ/∂y (T→V forward)
        curl(u, v) = ∂v/∂x - ∂u/∂y = ∂²ϕ/∂x∂y - ∂²ϕ/∂y∂x = 0

        Note: The identity holds strictly in the deep interior [2:-2, 2:-2].
        The first and last interior row/column are contaminated by zero ghost
        boundaries of the intermediate u and v fields (they are only set in
        [1:-1, 1:-1], so reading their ghost ring gives 0 instead of the
        physical value).
        """
        from finitevolx._src.operators.difference import Difference2D

        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        phi = x[None, :] ** 2 + y[:, None] ** 2  # ϕ = x² + y²

        u = diff.diff_x_T_to_U(phi)  # ∂ϕ/∂x at U-points
        v = diff.diff_y_T_to_V(phi)  # ∂ϕ/∂y at V-points
        curl = diff.curl(u, v)

        # Deep interior only: [2:-2, 2:-2] avoids ghost boundary contamination
        # on all four sides of the intermediate u and v fields.
        np.testing.assert_allclose(curl[2:-2, 2:-2], 0.0, atol=1e-10)

    def test_laplacian_equals_div_grad(self, grid2d):
        """∇²ϕ = div(grad(ϕ)) at deep interior points.

        laplacian = diff_x_U_to_T(diff_x_T_to_U(h)) + diff_y_V_to_T(diff_y_T_to_V(h))
        This must match calling diff.laplacian(h) directly at the deep interior.
        """
        from finitevolx._src.operators.difference import Difference2D

        diff = Difference2D(grid=grid2d)
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        y = jnp.arange(grid2d.Ny, dtype=float) * grid2d.dy
        h = x[None, :] ** 2 + y[:, None] ** 2

        lap = diff.laplacian(h)

        dh_dx = diff.diff_x_T_to_U(h)
        dh_dy = diff.diff_y_T_to_V(h)
        d2h_dx2 = diff.diff_x_U_to_T(dh_dx)
        d2h_dy2 = diff.diff_y_V_to_T(dh_dy)
        div_grad = jnp.zeros_like(h)
        div_grad = div_grad.at[1:-1, 1:-1].set(
            d2h_dx2[1:-1, 1:-1] + d2h_dy2[1:-1, 1:-1]
        )

        # Deep interior only: [2:-2, 2:-2] avoids ghost boundary artefacts
        # on all four sides from the intermediate u and v ghost rings.
        np.testing.assert_allclose(lap[2:-2, 2:-2], div_grad[2:-2, 2:-2], atol=1e-10)
