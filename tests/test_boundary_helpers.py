"""Tests for functional boundary helper API.

Covers:
- Padding wrappers (zero, zero-gradient, no-flux, no-slip, free-slip)
- Corner fixing
- Composite wall_boundaries for all C-grid staggerings
- Physical correctness (wall velocity, symmetry)
- JAX transform compatibility
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    fix_boundary_corners,
    free_slip_boundaries,
    no_flux_boundaries,
    no_slip_boundaries,
    wall_boundaries,
    zero_boundaries,
    zero_gradient_boundaries,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interior_field(Ny, Nx):
    """Field with distinct interior values and junk ghost ring."""
    j = jnp.arange(Ny)[:, None]
    i = jnp.arange(Nx)[None, :]
    return jnp.sin(2.0 * jnp.pi * i / Nx) * jnp.cos(2.0 * jnp.pi * j / Ny)


def _ghost_ring(field):
    """Extract the 1-cell ghost ring as (south, north, west, east) 1D arrays."""
    return field[0, :], field[-1, :], field[:, 0], field[:, -1]


# ===========================================================================
# zero_boundaries
# ===========================================================================


class TestPadWidthValidation:
    def test_pad_width_zero_raises(self):
        f = jnp.ones((8, 8))
        with pytest.raises(ValueError, match="pad_width must be >= 1"):
            zero_boundaries(f, pad_width=0)

    def test_pad_width_too_large_raises(self):
        f = jnp.ones((6, 6))
        with pytest.raises(ValueError, match="too large"):
            zero_boundaries(f, pad_width=3)

    def test_all_functions_validate(self):
        """All pad_width functions should reject invalid pad_width."""
        f = jnp.ones((6, 6))
        for fn in (
            zero_boundaries,
            zero_gradient_boundaries,
            no_slip_boundaries,
            free_slip_boundaries,
        ):
            with pytest.raises(ValueError):
                fn(f, pad_width=0)


class TestZeroBoundaries:
    def test_ghost_is_zero(self):
        f = _interior_field(8, 8)
        out = zero_boundaries(f)
        south, north, west, east = _ghost_ring(out)
        np.testing.assert_array_equal(south, 0.0)
        np.testing.assert_array_equal(north, 0.0)
        np.testing.assert_array_equal(west, 0.0)
        np.testing.assert_array_equal(east, 0.0)

    def test_interior_preserved(self):
        f = _interior_field(8, 8)
        out = zero_boundaries(f)
        np.testing.assert_array_equal(out[1:-1, 1:-1], f[1:-1, 1:-1])

    def test_shape_preserved(self):
        f = jnp.ones((10, 12))
        assert zero_boundaries(f).shape == (10, 12)

    def test_pad_width_2(self):
        f = _interior_field(10, 10)
        out = zero_boundaries(f, pad_width=2)
        assert out.shape == (10, 10)
        # 2-cell ghost ring should be zero
        np.testing.assert_array_equal(out[:2, :], 0.0)
        np.testing.assert_array_equal(out[-2:, :], 0.0)
        np.testing.assert_array_equal(out[:, :2], 0.0)
        np.testing.assert_array_equal(out[:, -2:], 0.0)
        # Interior preserved
        np.testing.assert_array_equal(out[2:-2, 2:-2], f[2:-2, 2:-2])


# ===========================================================================
# zero_gradient_boundaries
# ===========================================================================


class TestZeroGradientBoundaries:
    def test_ghost_equals_nearest_interior(self):
        f = _interior_field(8, 8)
        out = zero_gradient_boundaries(f)
        # South ghost = first interior row
        np.testing.assert_array_equal(out[0, 1:-1], f[1, 1:-1])
        # North ghost = last interior row
        np.testing.assert_array_equal(out[-1, 1:-1], f[-2, 1:-1])
        # West ghost = first interior col
        np.testing.assert_array_equal(out[1:-1, 0], f[1:-1, 1])
        # East ghost = last interior col
        np.testing.assert_array_equal(out[1:-1, -1], f[1:-1, -2])

    def test_interior_preserved(self):
        f = _interior_field(8, 8)
        out = zero_gradient_boundaries(f)
        np.testing.assert_array_equal(out[1:-1, 1:-1], f[1:-1, 1:-1])

    def test_constant_field_unchanged(self):
        """A constant field should be unchanged by zero-gradient padding."""
        f = jnp.full((8, 8), 5.0)
        out = zero_gradient_boundaries(f)
        np.testing.assert_array_equal(out, f)

    def test_pad_width_2(self):
        f = _interior_field(10, 10)
        out = zero_gradient_boundaries(f, pad_width=2)
        assert out.shape == (10, 10)
        np.testing.assert_array_equal(out[2:-2, 2:-2], f[2:-2, 2:-2])
        # Edge of 2-cell ghost should equal nearest interior
        np.testing.assert_array_equal(out[0, 2:-2], f[2, 2:-2])
        np.testing.assert_array_equal(out[1, 2:-2], f[2, 2:-2])


# ===========================================================================
# no_flux_boundaries
# ===========================================================================


class TestNoFluxBoundaries:
    def test_alias_for_zero_boundaries(self):
        """no_flux_boundaries should produce identical output to zero_boundaries."""
        f = _interior_field(8, 8)
        np.testing.assert_array_equal(no_flux_boundaries(f), zero_boundaries(f))


# ===========================================================================
# no_slip_boundaries
# ===========================================================================


class TestNoSlipBoundaries:
    def test_ghost_is_negated_interior(self):
        f = _interior_field(8, 8)
        out = no_slip_boundaries(f)
        # South ghost = -first interior row
        np.testing.assert_allclose(out[0, 1:-1], -f[1, 1:-1], atol=1e-14)
        # North ghost = -last interior row
        np.testing.assert_allclose(out[-1, 1:-1], -f[-2, 1:-1], atol=1e-14)
        # West ghost = -first interior col
        np.testing.assert_allclose(out[1:-1, 0], -f[1:-1, 1], atol=1e-14)
        # East ghost = -last interior col
        np.testing.assert_allclose(out[1:-1, -1], -f[1:-1, -2], atol=1e-14)

    def test_wall_midpoint_is_zero(self):
        """Average of ghost + interior at wall midpoint should be zero."""
        f = _interior_field(8, 8)
        out = no_slip_boundaries(f)
        # South wall: midpoint = (ghost[0,:] + interior[1,:]) / 2
        south_mid = 0.5 * (out[0, 1:-1] + out[1, 1:-1])
        np.testing.assert_allclose(south_mid, 0.0, atol=1e-14)
        # North wall
        north_mid = 0.5 * (out[-1, 1:-1] + out[-2, 1:-1])
        np.testing.assert_allclose(north_mid, 0.0, atol=1e-14)

    def test_interior_preserved(self):
        f = _interior_field(8, 8)
        out = no_slip_boundaries(f)
        np.testing.assert_array_equal(out[1:-1, 1:-1], f[1:-1, 1:-1])

    def test_pad_width_2(self):
        f = _interior_field(10, 10)
        out = no_slip_boundaries(f, pad_width=2)
        assert out.shape == (10, 10)
        np.testing.assert_array_equal(out[2:-2, 2:-2], f[2:-2, 2:-2])
        # Outermost south ghost row (excluding corners): negate 2nd interior row
        np.testing.assert_allclose(out[0, 2:-2], -f[3, 2:-2], atol=1e-14)
        # Inner south ghost row (excluding corners): negate 1st interior row
        np.testing.assert_allclose(out[1, 2:-2], -f[2, 2:-2], atol=1e-14)


# ===========================================================================
# free_slip_boundaries
# ===========================================================================


class TestFreeSlipBoundaries:
    def test_ghost_equals_interior(self):
        f = _interior_field(8, 8)
        out = free_slip_boundaries(f)
        # South ghost = first interior row
        np.testing.assert_allclose(out[0, 1:-1], f[1, 1:-1], atol=1e-14)
        # North ghost = last interior row
        np.testing.assert_allclose(out[-1, 1:-1], f[-2, 1:-1], atol=1e-14)
        # West ghost = first interior col
        np.testing.assert_allclose(out[1:-1, 0], f[1:-1, 1], atol=1e-14)
        # East ghost = last interior col
        np.testing.assert_allclose(out[1:-1, -1], f[1:-1, -2], atol=1e-14)

    def test_zero_normal_gradient(self):
        """Normal gradient at wall midpoint should be zero."""
        f = _interior_field(8, 8)
        out = free_slip_boundaries(f)
        # South wall: gradient = (interior - ghost) / dx = 0
        grad_south = out[1, 1:-1] - out[0, 1:-1]
        np.testing.assert_allclose(grad_south, 0.0, atol=1e-14)

    def test_interior_preserved(self):
        f = _interior_field(8, 8)
        out = free_slip_boundaries(f)
        np.testing.assert_array_equal(out[1:-1, 1:-1], f[1:-1, 1:-1])

    def test_pad_width_2(self):
        f = _interior_field(10, 10)
        out = free_slip_boundaries(f, pad_width=2)
        assert out.shape == (10, 10)
        np.testing.assert_array_equal(out[2:-2, 2:-2], f[2:-2, 2:-2])
        # Outermost south ghost row (excluding corners): copy 2nd interior row
        np.testing.assert_allclose(out[0, 2:-2], f[3, 2:-2], atol=1e-14)
        # Inner south ghost row (excluding corners): copy 1st interior row
        np.testing.assert_allclose(out[1, 2:-2], f[2, 2:-2], atol=1e-14)


# ===========================================================================
# fix_boundary_corners
# ===========================================================================


class TestFixBoundaryCorners:
    def test_corner_is_average(self):
        f = jnp.zeros((6, 6))
        # Set some edge ghost values
        f = f.at[0, 1].set(2.0)  # south edge near SW corner
        f = f.at[1, 0].set(4.0)  # west edge near SW corner
        f = f.at[0, -2].set(6.0)  # south edge near SE corner
        f = f.at[1, -1].set(8.0)  # east edge near SE corner
        out = fix_boundary_corners(f)
        # SW corner = avg(south[0,1], west[1,0]) = (2+4)/2 = 3
        assert float(out[0, 0]) == pytest.approx(3.0)
        # SE corner = avg(south[0,-2], east[1,-1]) = (6+8)/2 = 7
        assert float(out[0, -1]) == pytest.approx(7.0)

    def test_symmetric_field(self):
        """Symmetric field should give symmetric corners."""
        C = 5.0
        f = jnp.full((6, 6), C)
        out = fix_boundary_corners(f)
        assert float(out[0, 0]) == pytest.approx(C)
        assert float(out[0, -1]) == pytest.approx(C)
        assert float(out[-1, 0]) == pytest.approx(C)
        assert float(out[-1, -1]) == pytest.approx(C)

    def test_only_corners_modified(self):
        """Non-corner cells should be unchanged."""
        f = _interior_field(8, 8)
        out = fix_boundary_corners(f)
        # Interior
        np.testing.assert_array_equal(out[1:-1, 1:-1], f[1:-1, 1:-1])
        # Edges (non-corner)
        np.testing.assert_array_equal(out[0, 1:-1], f[0, 1:-1])
        np.testing.assert_array_equal(out[-1, 1:-1], f[-1, 1:-1])
        np.testing.assert_array_equal(out[1:-1, 0], f[1:-1, 0])
        np.testing.assert_array_equal(out[1:-1, -1], f[1:-1, -1])


# ===========================================================================
# wall_boundaries
# ===========================================================================


class TestWallBoundaries:
    def test_h_grid_zero_gradient(self):
        """h-grid: tracer gets free-slip (zero gradient) + corners."""
        f = _interior_field(8, 8)
        out = wall_boundaries(f, staggering="h")
        # Ghost should equal interior (free-slip)
        np.testing.assert_allclose(out[0, 1:-1], f[1, 1:-1], atol=1e-14)
        np.testing.assert_allclose(out[-1, 1:-1], f[-2, 1:-1], atol=1e-14)

    def test_u_grid_no_flux_ew(self):
        """u-grid: zero at E/W walls (normal flux), free-slip at N/S."""
        f = _interior_field(8, 8)
        out = wall_boundaries(f, staggering="u")
        # E/W ghost cols should be zero
        np.testing.assert_array_equal(out[1:-1, 0], 0.0)
        np.testing.assert_array_equal(out[1:-1, -1], 0.0)
        # N/S ghost rows should match interior (free-slip)
        np.testing.assert_allclose(out[0, 1:-1], f[1, 1:-1], atol=1e-14)
        np.testing.assert_allclose(out[-1, 1:-1], f[-2, 1:-1], atol=1e-14)

    def test_v_grid_no_flux_ns(self):
        """v-grid: zero at N/S walls (normal flux), free-slip at E/W."""
        f = _interior_field(8, 8)
        out = wall_boundaries(f, staggering="v")
        # N/S ghost rows should be zero
        np.testing.assert_array_equal(out[0, 1:-1], 0.0)
        np.testing.assert_array_equal(out[-1, 1:-1], 0.0)
        # E/W ghost cols should match interior (free-slip)
        np.testing.assert_allclose(out[1:-1, 0], f[1:-1, 1], atol=1e-14)
        np.testing.assert_allclose(out[1:-1, -1], f[1:-1, -2], atol=1e-14)

    def test_q_grid_no_slip(self):
        """q-grid: no-slip (sign-flipped) at all walls."""
        f = _interior_field(8, 8)
        out = wall_boundaries(f, staggering="q")
        # Ghost = -interior at all walls
        np.testing.assert_allclose(out[0, 1:-1], -f[1, 1:-1], atol=1e-14)
        np.testing.assert_allclose(out[-1, 1:-1], -f[-2, 1:-1], atol=1e-14)
        np.testing.assert_allclose(out[1:-1, 0], -f[1:-1, 1], atol=1e-14)
        np.testing.assert_allclose(out[1:-1, -1], -f[1:-1, -2], atol=1e-14)

    def test_all_grids_have_averaged_corners(self):
        """All staggerings should have averaged corners."""
        f = _interior_field(8, 8)
        for grid in ("h", "u", "v", "q"):
            out = wall_boundaries(f, staggering=grid)
            # Corner should be average of adjacent edge ghost cells
            expected_sw = 0.5 * (out[0, 1] + out[1, 0])
            assert float(out[0, 0]) == pytest.approx(float(expected_sw), abs=1e-14), (
                f"grid={grid}"
            )

    def test_invalid_staggering(self):
        with pytest.raises(ValueError, match="staggering must be"):
            wall_boundaries(jnp.ones((6, 6)), staggering="z")


# ===========================================================================
# Physical correctness
# ===========================================================================


class TestBoundaryPhysics:
    """Scientific tests for physical correctness of boundary conditions."""

    def test_no_slip_wall_tangential_velocity_zero(self):
        """No-slip: tangential velocity at wall face should be zero.

        For a u-velocity field at the south wall, the wall-face value
        is the average of the ghost cell and first interior cell.
        With no-slip BC, this average must be zero.
        """
        Ny, Nx = 10, 10
        # Tangential velocity (e.g., u at south wall)
        u = jnp.ones((Ny, Nx)) * 3.0
        u_bc = no_slip_boundaries(u)

        # South wall tangential velocity at face
        u_wall_south = 0.5 * (u_bc[0, :] + u_bc[1, :])
        np.testing.assert_allclose(u_wall_south[1:-1], 0.0, atol=1e-14)

    def test_free_slip_wall_normal_gradient_zero(self):
        """Free-slip: ∂u/∂n = 0 at the wall."""
        Ny, Nx = 10, 10
        u = jnp.linspace(0, 1, Nx)[None, :] * jnp.ones((Ny, 1))
        u_bc = free_slip_boundaries(u)

        # Normal gradient at south wall: (u[1,:] - u[0,:]) / dx
        grad_south = u_bc[1, 1:-1] - u_bc[0, 1:-1]
        np.testing.assert_allclose(grad_south, 0.0, atol=1e-14)

    def test_rigid_wall_no_mass_flux(self):
        """u at E/W wall + v at N/S wall should be zero (no mass through walls)."""
        Ny, Nx = 10, 10
        u = jnp.ones((Ny, Nx)) * 2.0
        v = jnp.ones((Ny, Nx)) * 3.0

        u_bc = wall_boundaries(u, staggering="u")
        v_bc = wall_boundaries(v, staggering="v")

        # u normal to E/W walls (interior edges, excluding corners)
        np.testing.assert_array_equal(u_bc[1:-1, 0], 0.0)
        np.testing.assert_array_equal(u_bc[1:-1, -1], 0.0)
        # v normal to N/S walls (interior edges, excluding corners)
        np.testing.assert_array_equal(v_bc[0, 1:-1], 0.0)
        np.testing.assert_array_equal(v_bc[-1, 1:-1], 0.0)

    def test_no_slip_antisymmetry(self):
        """No-slip ghost cells should be antisymmetric about the wall."""
        Ny, Nx = 8, 8
        f = _interior_field(Ny, Nx)
        out = no_slip_boundaries(f)

        # For each wall (excluding corners), ghost + interior = 0
        np.testing.assert_allclose(out[0, 1:-1] + out[1, 1:-1], 0.0, atol=1e-14)

    def test_free_slip_symmetry(self):
        """Free-slip ghost cells should be symmetric about the wall."""
        Ny, Nx = 8, 8
        f = _interior_field(Ny, Nx)
        out = free_slip_boundaries(f)

        # For each wall, ghost == interior (symmetric pair)
        np.testing.assert_allclose(out[0, 1:-1], out[1, 1:-1], atol=1e-14)


# ===========================================================================
# JAX compatibility
# ===========================================================================


class TestJaxCompat:
    def test_jit_all_functions(self):
        f = _interior_field(8, 8)
        for fn in (
            zero_boundaries,
            zero_gradient_boundaries,
            no_flux_boundaries,
            no_slip_boundaries,
            free_slip_boundaries,
            fix_boundary_corners,
        ):
            result = jax.jit(fn)(f)
            assert result.shape == f.shape
            assert jnp.all(jnp.isfinite(result))

    @pytest.mark.parametrize("grid", ["h", "u", "v", "q"])
    def test_jit_wall_boundaries(self, grid):
        f = _interior_field(8, 8)

        @jax.jit
        def apply(field):
            return wall_boundaries(field, staggering=grid)

        result = apply(f)
        assert result.shape == f.shape
        assert jnp.all(jnp.isfinite(result))

    def test_grad_through_boundaries(self):
        """Gradients should propagate through boundary operations."""
        f = _interior_field(8, 8)

        def loss(field):
            out = no_slip_boundaries(field)
            return jnp.sum(out**2)

        grad = jax.grad(loss)(f)
        assert grad.shape == f.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_vmap(self):
        """vmap over batch of fields."""
        B = 4
        fields = jnp.stack([_interior_field(8, 8) * (i + 1) for i in range(B)])

        result = jax.vmap(free_slip_boundaries)(fields)
        assert result.shape == (B, 8, 8)
