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

from finitevolx._src.constants import GRAVITY
from finitevolx._src.grid import ArakawaCGrid2D
from finitevolx._src.operators.operators import (
    bernoulli_potential,
    kinetic_energy,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def grid():
    # 8 interior cells in each direction → total shape [10, 10]
    return ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)


# ---------------------------------------------------------------------------
# kinetic_energy
# ---------------------------------------------------------------------------


class TestKineticEnergy2D:
    def test_constant_unity_field(self, grid):
        """Constant u=v=1 everywhere → ke = 1.0 at interior."""
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        ke = kinetic_energy(u=u, v=v)

        assert ke.shape == (grid.Ny, grid.Nx)
        # u²_on_T = 0.5*(1² + 1²) = 1, v²_on_T = 0.5*(1² + 1²) = 1
        # ke = 0.5*(u²_on_T + v²_on_T) = 0.5*(1 + 1) = 1.0
        np.testing.assert_allclose(ke[1:-1, 1:-1], 1.0, rtol=1e-6)

    def test_ghost_ring_is_zero(self, grid):
        """Ghost cells must remain zero (interior-point idiom)."""
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        ke = kinetic_energy(u=u, v=v)

        np.testing.assert_array_equal(ke[0, :], 0.0)
        np.testing.assert_array_equal(ke[-1, :], 0.0)
        np.testing.assert_array_equal(ke[:, 0], 0.0)
        np.testing.assert_array_equal(ke[:, -1], 0.0)

    def test_output_shape(self, grid):
        """ke must have the same shape as u and v."""
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        ke = kinetic_energy(u=u, v=v)
        assert ke.shape == (grid.Ny, grid.Nx)

    def test_nonconst_u_linear_in_x(self, grid):
        """u varies linearly in x; v=0 → checks U→T averaging along x-axis.

        u[j, i] = i  (U-point at east face i+1/2)
        u²_on_T[j, i] = 0.5*(u[j,i]² + u[j,i-1]²) = 0.5*(i² + (i-1)²)
        ke[j, i] = 0.5 * u²_on_T[j, i]
        """
        ix = jnp.arange(grid.Nx, dtype=float)
        u = jnp.broadcast_to(ix, (grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))

        ke = kinetic_energy(u=u, v=v)

        # expected for interior columns i=1..Nx-2
        u2 = ix**2
        # u²_on_T[i] = 0.5*(u2[i] + u2[i-1])
        u2_on_T = 0.5 * (u2[1:-1] + u2[:-2])
        expected = 0.5 * u2_on_T  # ke = 0.5 * u²_on_T (v=0)

        np.testing.assert_allclose(ke[1, 1:-1], expected, rtol=1e-6)

    def test_nonconst_v_linear_in_y(self, grid):
        """v varies linearly in y; u=0 → checks V→T averaging along y-axis.

        v[j, i] = j  (V-point at north face j+1/2)
        v²_on_T[j, i] = 0.5*(v[j,i]² + v[j-1,i]²) = 0.5*(j² + (j-1)²)
        ke[j, i] = 0.5 * v²_on_T[j, i]
        """
        jy = jnp.arange(grid.Ny, dtype=float)
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.broadcast_to(jy[:, None], (grid.Ny, grid.Nx))

        ke = kinetic_energy(u=u, v=v)

        # expected for interior rows j=1..Ny-2
        v2 = jy**2
        # v²_on_T[j] = 0.5*(v2[j] + v2[j-1])
        v2_on_T = 0.5 * (v2[1:-1] + v2[:-2])
        expected = 0.5 * v2_on_T  # ke = 0.5 * v²_on_T (u=0)

        # v is constant in x → all interior columns are identical; compare one column
        np.testing.assert_allclose(ke[1:-1, 1], expected, rtol=1e-6)

    def test_nonconst_uv_combined(self, grid):
        """u and v both vary; verifies combined averaging is independent per axis."""
        ix = jnp.arange(grid.Nx, dtype=float)
        jy = jnp.arange(grid.Ny, dtype=float)
        u = jnp.broadcast_to(ix, (grid.Ny, grid.Nx))
        v = jnp.broadcast_to(jy[:, None], (grid.Ny, grid.Nx))

        ke = kinetic_energy(u=u, v=v)

        u2 = ix**2
        u2_on_T = 0.5 * (u2[1:-1] + u2[:-2])  # shape [Nx-2]
        v2 = jy**2
        v2_on_T = 0.5 * (v2[1:-1] + v2[:-2])  # shape [Ny-2]
        expected = 0.5 * (u2_on_T[None, :] + v2_on_T[:, None])

        np.testing.assert_allclose(ke[1:-1, 1:-1], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# bernoulli_potential
# ---------------------------------------------------------------------------


class TestBernoulliPotential2D:
    def test_constant_unity_fields(self, grid):
        """u=v=h=1 everywhere → p = ke + g*h = 1.0 + g at interior."""
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        h = jnp.ones((grid.Ny, grid.Nx))

        p = bernoulli_potential(h=h, u=u, v=v)

        assert p.shape == (grid.Ny, grid.Nx)
        np.testing.assert_allclose(p[1:-1, 1:-1], 1.0 + GRAVITY * 1.0, rtol=1e-6)

    def test_ghost_ring_is_zero(self, grid):
        """Ghost cells must remain zero (interior-point idiom)."""
        u = jnp.ones((grid.Ny, grid.Nx))
        v = jnp.ones((grid.Ny, grid.Nx))
        h = jnp.ones((grid.Ny, grid.Nx))
        p = bernoulli_potential(h=h, u=u, v=v)

        np.testing.assert_array_equal(p[0, :], 0.0)
        np.testing.assert_array_equal(p[-1, :], 0.0)
        np.testing.assert_array_equal(p[:, 0], 0.0)
        np.testing.assert_array_equal(p[:, -1], 0.0)

    def test_output_shape(self, grid):
        """p must have the same shape as h, u, v."""
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        h = jnp.zeros((grid.Ny, grid.Nx))
        p = bernoulli_potential(h=h, u=u, v=v)
        assert p.shape == (grid.Ny, grid.Nx)

    def test_equals_ke_plus_gh(self, grid):
        """p == ke(u,v) + g*h elementwise at interior, with spatially varying fields."""
        ix = jnp.arange(grid.Nx, dtype=float)
        jy = jnp.arange(grid.Ny, dtype=float)
        u = jnp.broadcast_to(ix, (grid.Ny, grid.Nx))
        v = jnp.broadcast_to(jy[:, None], (grid.Ny, grid.Nx))
        # h: distinct values to avoid accidental cancellation
        h = (ix[None, :] + jy[:, None]) * 0.1

        p = bernoulli_potential(h=h, u=u, v=v)
        ke = kinetic_energy(u=u, v=v)

        expected = ke[1:-1, 1:-1] + GRAVITY * h[1:-1, 1:-1]
        np.testing.assert_allclose(p[1:-1, 1:-1], expected, rtol=1e-6)

    def test_zero_velocity_gives_gravity_term_only(self, grid):
        """u=v=0 → ke=0 → p = g*h at interior."""
        u = jnp.zeros((grid.Ny, grid.Nx))
        v = jnp.zeros((grid.Ny, grid.Nx))
        ix = jnp.arange(grid.Nx, dtype=float)
        jy = jnp.arange(grid.Ny, dtype=float)
        h = ix[None, :] + jy[:, None]

        p = bernoulli_potential(h=h, u=u, v=v)
        np.testing.assert_allclose(p[1:-1, 1:-1], GRAVITY * h[1:-1, 1:-1], rtol=1e-6)
