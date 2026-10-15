"""Tests for boundary condition helpers."""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    BoundaryConditionSet,
    CartesianGrid2D,
    Difference2D,
    Dirichlet1D,
    Extrapolation1D,
    FieldBCSet,
    Neumann1D,
    Robin1D,
    Slip1D,
    Sponge1D,
)
from finitevolx._src.boundary.bc_1d import Outflow1D, Periodic1D, Reflective1D
from finitevolx._src.boundary.boundary import enforce_periodic, pad_interior


class TestPadInterior:
    def test_output_shape(self):
        f = jnp.ones((6, 6))
        result = pad_interior(f)
        assert result.shape == (6, 6)

    def test_edge_mode_preserves_interior(self):
        # interior = all twos; edge pad should fill ghosts with 2
        f = 2.0 * jnp.ones((6, 6))
        result = pad_interior(f, mode="edge")
        np.testing.assert_allclose(result, 2.0)

    def test_interior_values_preserved(self):
        # Set interior to a known pattern
        f = jnp.zeros((6, 6))
        f = f.at[1:-1, 1:-1].set(jnp.arange(16, dtype=float).reshape(4, 4))
        result = pad_interior(f)
        # interior block must match
        np.testing.assert_array_equal(result[1:-1, 1:-1], f[1:-1, 1:-1])

    def test_constant_mode(self):
        f = jnp.ones((8, 8))
        result = pad_interior(f, mode="constant")
        # constant pad fills ghost with 0
        np.testing.assert_array_equal(result[0, :], 0.0)
        np.testing.assert_array_equal(result[-1, :], 0.0)
        np.testing.assert_array_equal(result[:, 0], 0.0)
        np.testing.assert_array_equal(result[:, -1], 0.0)


class TestEnforcePeriodic:
    def test_output_shape(self):
        f = jnp.ones((6, 6))
        result = enforce_periodic(f)
        assert result.shape == (6, 6)

    def test_south_ghost_equals_north_interior(self):
        # row 0 should equal row Ny-2
        f = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[0, :], result[-2, :])

    def test_north_ghost_equals_south_interior(self):
        f = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[-1, :], result[1, :])

    def test_west_ghost_equals_east_interior(self):
        f = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[:, 0], result[:, -2])

    def test_east_ghost_equals_west_interior(self):
        f = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[:, -1], result[:, 1])

    def test_constant_field_stays_constant(self):
        f = 5.0 * jnp.ones((8, 8))
        result = enforce_periodic(f)
        np.testing.assert_allclose(result, 5.0)


class TestBoundaryConditionAtoms:
    def test_dirichlet_sets_south_ghost_from_boundary_value(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = Dirichlet1D("south", value=10.0)(field, dx=2.0, dy=3.0)
        expected = 20.0 - field[1, :]
        np.testing.assert_allclose(result[0, :], expected)

    def test_neumann_sets_east_ghost_from_outward_gradient(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = Neumann1D("east", value=1.5)(field, dx=2.0, dy=3.0)
        expected = field[:, -2] + 3.0
        np.testing.assert_allclose(result[:, -1], expected)

    def test_neumann_sets_south_ghost_from_outward_gradient(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = Neumann1D("south", value=1.5)(field, dx=2.0, dy=3.0)
        expected = field[1, :] + 4.5
        np.testing.assert_allclose(result[0, :], expected)

    def test_neumann_sets_west_ghost_from_outward_gradient(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = Neumann1D("west", value=1.5)(field, dx=2.0, dy=3.0)
        expected = field[:, 1] + 3.0
        np.testing.assert_allclose(result[:, 0], expected)

    def test_sponge_relaxes_ghost_face_toward_background(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = Sponge1D("north", background=-1.0, weight=0.25)(field, dx=1.0, dy=1.0)
        expected = 0.75 * field[-2, :] - 0.25
        np.testing.assert_allclose(result[-1, :], expected)

    def test_sponge_rejects_invalid_weight(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Sponge1D("north", background=0.0, weight=1.5)

    def test_reflective_matches_adjacent_interior(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = Reflective1D("west")(field, dx=1.0, dy=1.0)
        np.testing.assert_array_equal(result[:, 0], field[:, 1])

    def test_periodic_atom_matches_legacy_periodic_face(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = Periodic1D("south")(field, dx=1.0, dy=1.0)
        np.testing.assert_array_equal(result[0, :], field[-2, :])


class TestSlip1D:
    """Unit tests for the slip boundary condition."""

    @staticmethod
    def _field():
        return (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )

    def test_free_slip_ghost_equals_interior(self):
        # coefficient=1: ghost = +interior (same as Reflective1D)
        field = self._field()
        result = Slip1D("west", coefficient=1.0)(field, dx=1.0, dy=1.0)
        np.testing.assert_array_equal(result[:, 0], field[:, 1])

    def test_no_slip_ghost_negates_interior(self):
        # coefficient=0: ghost = -interior -> wall value = 0
        field = self._field()
        result = Slip1D("west", coefficient=0.0)(field, dx=1.0, dy=1.0)
        np.testing.assert_array_equal(result[:, 0], -field[:, 1])

    def test_no_slip_wall_value_is_zero(self):
        # wall value = 0.5·(ghost + interior) = 0
        field = self._field()
        result = Slip1D("south", coefficient=0.0)(field, dx=1.0, dy=1.0)
        wall = 0.5 * (result[0, :] + result[1, :])
        np.testing.assert_allclose(wall, 0.0)

    def test_free_slip_wall_value_equals_interior(self):
        # wall value = 0.5·(ghost + interior) = interior
        field = self._field()
        result = Slip1D("north", coefficient=1.0)(field, dx=1.0, dy=1.0)
        wall = 0.5 * (result[-1, :] + result[-2, :])
        np.testing.assert_allclose(wall, field[-2, :])

    def test_partial_slip_ghost_is_interpolated(self):
        # coefficient=0.5: ghost = 0 * interior, wall value = 0.5 * interior
        field = self._field()
        result = Slip1D("east", coefficient=0.5)(field, dx=1.0, dy=1.0)
        expected = 0.0 * field[:, -2]
        np.testing.assert_allclose(result[:, -1], expected)

    def test_partial_slip_wall_value_is_scaled(self):
        # wall value = coefficient * interior
        field = self._field()
        alpha = 0.3
        result = Slip1D("south", coefficient=alpha)(field, dx=1.0, dy=1.0)
        wall = 0.5 * (result[0, :] + result[1, :])
        np.testing.assert_allclose(wall, alpha * field[1, :])

    def test_all_faces_produce_correct_ghost(self):
        # Test each face with no-slip; ghost should be -adjacent interior
        field = self._field()
        for face, ghost_slice, interior_slice in [
            ("south", (0, slice(None)), (1, slice(None))),
            ("north", (-1, slice(None)), (-2, slice(None))),
            ("west", (slice(None), 0), (slice(None), 1)),
            ("east", (slice(None), -1), (slice(None), -2)),
        ]:
            result = Slip1D(face, coefficient=0.0)(field, dx=1.0, dy=1.0)
            np.testing.assert_array_equal(
                result[ghost_slice], -field[interior_slice], err_msg=f"face={face}"
            )

    def test_invalid_coefficient_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Slip1D("south", coefficient=1.5)

    def test_invalid_negative_coefficient_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Slip1D("north", coefficient=-0.1)


class TestBoundaryConditionSet:
    def test_periodic_named_constructor_matches_legacy_helper(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = BoundaryConditionSet.periodic()(field, dx=1.0, dy=1.0)
        np.testing.assert_array_equal(result, enforce_periodic(field))

    def test_open_named_constructor_matches_edge_padding(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        result = BoundaryConditionSet.open()(field, dx=1.0, dy=1.0)
        np.testing.assert_array_equal(result, pad_interior(field, mode="edge"))

    def test_east_west_corner_updates_win(self):
        field = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        bcs = BoundaryConditionSet(
            south=Dirichlet1D("south", value=100.0),
            west=Outflow1D("west"),
            east=Outflow1D("east"),
        )
        result = bcs(field, dx=1.0, dy=1.0)
        np.testing.assert_array_equal(result[0, 2:-2], 200.0 - field[1, 2:-2])
        np.testing.assert_array_equal(result[0, 0], result[0, 1])
        np.testing.assert_array_equal(result[0, -1], result[0, -2])


class TestFieldBCSet:
    def test_field_map_and_default_are_applied_per_variable(self):
        h = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )
        u = (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(100.0 + jnp.arange(16, dtype=float).reshape(4, 4))
        )
        state = {"h": h, "u": u}
        field_bcs = FieldBCSet(
            bc_map={"h": BoundaryConditionSet.periodic()},
            default=BoundaryConditionSet.open(),
        )

        result = field_bcs(state, dx=1.0, dy=1.0)

        np.testing.assert_array_equal(result["h"], enforce_periodic(h))
        np.testing.assert_array_equal(result["u"], pad_interior(u, mode="edge"))


class TestRobin1D:
    """Unit tests for the Robin boundary condition."""

    @staticmethod
    def _field():
        return (
            jnp.zeros((6, 6))
            .at[1:-1, 1:-1]
            .set(jnp.arange(16, dtype=float).reshape(4, 4))
        )

    def test_ghost_value_south(self):
        field = self._field()
        alpha, beta, gamma = 2.0, 1.0, 5.0
        result = Robin1D("south", alpha=alpha, beta=beta, gamma=gamma)(
            field, dx=2.0, dy=3.0
        )
        # s = beta / spacing = 1.0 / 3.0 (no face-dependent sign)
        s = beta / 3.0
        interior = field[1, :]
        expected = (gamma - interior * (alpha / 2.0 - s)) / (alpha / 2.0 + s)
        np.testing.assert_allclose(result[0, :], expected)

    def test_ghost_value_east(self):
        field = self._field()
        alpha, beta, gamma = 1.0, 2.0, 3.0
        result = Robin1D("east", alpha=alpha, beta=beta, gamma=gamma)(
            field, dx=2.0, dy=3.0
        )
        s = beta / 2.0  # east: spacing=dx=2
        interior = field[:, -2]
        expected = (gamma - interior * (alpha / 2.0 - s)) / (alpha / 2.0 + s)
        np.testing.assert_allclose(result[:, -1], expected)

    def test_reduces_to_dirichlet_when_beta_zero(self):
        """With β=0 the Robin condition α·u = γ gives u_wall = γ/α."""
        field = self._field()
        value = 10.0
        robin_result = Robin1D("south", alpha=1.0, beta=0.0, gamma=value)(
            field, dx=2.0, dy=3.0
        )
        dirichlet_result = Dirichlet1D("south", value=value)(field, dx=2.0, dy=3.0)
        np.testing.assert_allclose(robin_result[0, :], dirichlet_result[0, :])

    def test_reduces_to_neumann_when_alpha_zero(self):
        """With α=0 the Robin condition β·∂u/∂n = γ gives ∂u/∂n = γ/β."""
        field = self._field()
        grad_value = 1.5
        beta = 2.0
        robin_result = Robin1D("east", alpha=0.0, beta=beta, gamma=grad_value * beta)(
            field, dx=2.0, dy=3.0
        )
        neumann_result = Neumann1D("east", value=grad_value)(field, dx=2.0, dy=3.0)
        np.testing.assert_allclose(robin_result[:, -1], neumann_result[:, -1])

    def test_both_zero_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            Robin1D("south", alpha=0.0, beta=0.0, gamma=1.0)

    def test_all_faces(self):
        field = self._field()
        for face in ("south", "north", "west", "east"):
            result = Robin1D(face, alpha=1.0, beta=1.0, gamma=0.0)(
                field, dx=1.0, dy=1.0
            )
            assert result.shape == field.shape

    def test_jit_compatible(self):
        field = self._field()
        bc = Robin1D("north", alpha=1.0, beta=1.0, gamma=0.0)
        result_eager = bc(field, dx=1.0, dy=1.0)
        result_jit = jax.jit(bc)(field, dx=1.0, dy=1.0)
        np.testing.assert_allclose(result_jit, result_eager)

    def test_works_with_boundary_condition_set(self):
        field = self._field()
        bcs = BoundaryConditionSet(
            south=Robin1D("south", alpha=1.0, beta=0.0, gamma=5.0),
        )
        result = bcs(field, dx=1.0, dy=1.0)
        assert result.shape == field.shape


class TestOutwardGradientAnalytic:
    """Ghost cells reproduce the analytic field phi = x + 2 y on every face.

    The ghost cell lies outward of the interior, so ``value`` is the true
    outward-normal gradient: -1 (west), +1 (east), -2 (south), +2 (north).
    """

    NY, NX, DX, DY = 7, 6, 0.5, 0.25
    OUTWARD: ClassVar[dict[str, float]] = {
        "south": -2.0,
        "north": 2.0,
        "west": -1.0,
        "east": 1.0,
    }

    @classmethod
    def _phi(cls):
        x = (jnp.arange(cls.NX) - 1) * cls.DX
        y = (jnp.arange(cls.NY) - 1) * cls.DY
        return x[None, :] + 2.0 * y[:, None]

    @staticmethod
    def _ghost(arr, face):
        return {
            "south": arr[0, 1:-1],
            "north": arr[-1, 1:-1],
            "west": arr[1:-1, 0],
            "east": arr[1:-1, -1],
        }[face]

    @pytest.mark.slow
    @pytest.mark.parametrize("face", ["south", "north", "west", "east"])
    def test_neumann(self, face):
        phi = self._phi()
        field = phi.at[0, :].set(0).at[-1, :].set(0).at[:, 0].set(0).at[:, -1].set(0)
        out = Neumann1D(face, self.OUTWARD[face])(field, self.DX, self.DY)
        np.testing.assert_allclose(
            self._ghost(out, face), self._ghost(phi, face), atol=1e-14
        )

    @pytest.mark.parametrize("face", ["south", "north", "west", "east"])
    @pytest.mark.parametrize("alpha, beta", [(0.0, 1.0), (1.0, 0.5), (2.0, -0.3)])
    def test_robin(self, face, alpha, beta):
        phi = self._phi()
        field = phi.at[0, :].set(0).at[-1, :].set(0).at[:, 0].set(0).at[:, -1].set(0)
        spacing = self.DY if face in ("south", "north") else self.DX
        # Interior neighbour one cell inward of the ghost, at tangential index 2.
        ghost, interior = {
            "south": (phi[0, 2], phi[1, 2]),
            "north": (phi[-1, 2], phi[-2, 2]),
            "west": (phi[2, 0], phi[2, 1]),
            "east": (phi[2, -1], phi[2, -2]),
        }[face]
        # gamma = alpha * phi_face + beta * dphi/dn, from the analytic field.
        gamma = float(
            alpha * 0.5 * (ghost + interior) + beta * (ghost - interior) / spacing
        )
        out = Robin1D(face, alpha=alpha, beta=beta, gamma=gamma)(
            field, self.DX, self.DY
        )
        got = {
            "south": out[0, 2],
            "north": out[-1, 2],
            "west": out[2, 0],
            "east": out[2, -1],
        }[face]
        np.testing.assert_allclose(float(got), float(ghost), atol=1e-14)


class TestExtrapolation1D:
    """Unit tests for the high-order extrapolation boundary condition."""

    def test_order1_linear_field_exact(self):
        """Linear field f(j) = 2*j should be extrapolated exactly by order 1."""
        # 8x8 field, linear in y: f[j, :] = 2*j
        field = jnp.zeros((8, 8))
        for j in range(8):
            field = field.at[j, :].set(2.0 * j)
        # South ghost (row 0): extrapolate from rows 1, 2
        # Linear: f(0) = 2*1 - (2*2 - 2*1) = 2 - 2 = 0 ✓ (already 0)
        result = Extrapolation1D("south", order=1)(field, dx=1.0, dy=1.0)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-12)
        # North ghost (row -1 = row 7): extrapolate from rows 6, 5
        # Linear: f(7) = 2*6 + (2*6 - 2*5) = 12 + 2 = 14
        result = Extrapolation1D("north", order=1)(field, dx=1.0, dy=1.0)
        np.testing.assert_allclose(result[-1, :], 14.0, atol=1e-12)

    def test_order2_quadratic_field_exact(self):
        """Quadratic field f(j) = j^2 should be extrapolated exactly by order 2."""
        field = jnp.zeros((8, 8))
        for j in range(8):
            field = field.at[j, :].set(float(j**2))
        # South ghost (row 0): extrapolate from rows 1, 2, 3
        # f(0) should be 0 (already correct for j^2)
        result = Extrapolation1D("south", order=2)(field, dx=1.0, dy=1.0)
        np.testing.assert_allclose(result[0, :], 0.0, atol=1e-12)

    def test_order3_cubic_field_exact(self):
        """Cubic field f(j) = j^3 should be extrapolated exactly by order 3."""
        field = jnp.zeros((10, 10))
        for j in range(10):
            field = field.at[j, :].set(float(j**3))
        # North ghost: extrapolate from rows 8, 7, 6, 5
        # f(9) = 729
        result = Extrapolation1D("north", order=3)(field, dx=1.0, dy=1.0)
        np.testing.assert_allclose(result[-1, :], 729.0, atol=1e-10)

    def test_all_faces_order1(self):
        """Order 1 extrapolation works on all four faces."""
        # Linear in x: f[:, i] = 3*i
        field = jnp.zeros((8, 8))
        for i in range(8):
            field = field.at[:, i].set(3.0 * i)

        # West ghost (col 0): extrapolate from cols 1, 2
        result = Extrapolation1D("west", order=1)(field, dx=1.0, dy=1.0)
        np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-12)

        # East ghost (col -1 = col 7): extrapolate from cols 6, 5
        result = Extrapolation1D("east", order=1)(field, dx=1.0, dy=1.0)
        np.testing.assert_allclose(result[:, -1], 21.0, atol=1e-12)

    def test_invalid_order_zero_raises(self):
        with pytest.raises(ValueError, match=r"\[1, 5\]"):
            Extrapolation1D("south", order=0)

    def test_invalid_order_six_raises(self):
        with pytest.raises(ValueError, match=r"\[1, 5\]"):
            Extrapolation1D("south", order=6)

    def test_jit_compatible(self):
        field = jnp.zeros((8, 8))
        for j in range(8):
            field = field.at[j, :].set(2.0 * j)
        bc = Extrapolation1D("south", order=1)
        result_eager = bc(field, dx=1.0, dy=1.0)
        result_jit = jax.jit(bc)(field, dx=1.0, dy=1.0)
        np.testing.assert_allclose(result_jit, result_eager)

    def test_works_with_boundary_condition_set(self):
        field = jnp.ones((8, 8))
        bcs = BoundaryConditionSet(
            south=Extrapolation1D("south", order=2),
            north=Extrapolation1D("north", order=1),
        )
        result = bcs(field, dx=1.0, dy=1.0)
        assert result.shape == field.shape

    def test_order1_coefficients(self):
        """Verify the precomputed coefficients for order 1."""
        bc = Extrapolation1D("south", order=1)
        assert bc._coeffs == (2.0, -1.0)

    def test_order5_coefficients(self):
        """Verify the precomputed coefficients for order 5."""
        bc = Extrapolation1D("south", order=5)
        assert bc._coeffs == (6.0, -15.0, 20.0, -15.0, 6.0, -1.0)


class TestStaggeredPoints:
    """``point=``: atoms on fields whose boundary node lies on the wall (#286).

    Same-index staggering: ``U[:, 0]`` / ``U[:, Nx-2]`` are the west / east
    wall faces, ``V[0, :]`` / ``V[Ny-2, :]`` the south / north ones, and an
    X field has a node on every wall.
    """

    Ny, Nx = 6, 7
    dx, dy = 0.25, 0.5

    def _field(self):
        j = jnp.arange(self.Ny)[:, None]
        i = jnp.arange(self.Nx)[None, :]
        return 1.0 + jnp.sin(1.3 * i + 0.7 * j)  # [Ny, Nx], no symmetry

    def test_closed_basin_has_zero_net_mass_flux(self):
        """u = v = 1, closed(point=U/V): no mass crosses the walls."""
        g = CartesianGrid2D.from_interior(
            self.Nx - 2,
            self.Ny - 2,
            Lx=(self.Nx - 2) * self.dx,
            Ly=(self.Ny - 2) * self.dy,
        )
        u = jnp.ones((self.Ny, self.Nx))
        v = jnp.ones((self.Ny, self.Nx))
        u_bc = BoundaryConditionSet.closed(point="U")(u, self.dx, self.dy)
        v_bc = BoundaryConditionSet.closed(point="V")(v, self.dx, self.dy)
        np.testing.assert_array_equal(u_bc[1:-1, 0], 0.0)
        np.testing.assert_array_equal(u_bc[1:-1, -2], 0.0)
        np.testing.assert_array_equal(v_bc[0, 1:-1], 0.0)
        np.testing.assert_array_equal(v_bc[-2, 1:-1], 0.0)
        div = Difference2D(g).divergence(u_bc, v_bc)
        net = float(div[1:-1, 1:-1].sum() * self.dx * self.dy)
        assert abs(net) < 1e-14
        # the T-point default leaves the east wall face open (the old leak)
        u_t = BoundaryConditionSet.closed()(u, self.dx, self.dy)
        v_t = BoundaryConditionSet.closed()(v, self.dx, self.dy)
        leak = Difference2D(g).divergence(u_t, v_t)[1:-1, 1:-1].sum()
        assert abs(float(leak)) > 1.0

    @pytest.mark.parametrize(
        ("face", "point", "wall", "adjacent"),
        [
            ("west", "U", (slice(None), 0), (slice(None), 1)),
            ("east", "U", (slice(None), -2), (slice(None), -3)),
            ("south", "V", (0, slice(None)), (1, slice(None))),
            ("north", "V", (-2, slice(None)), (-3, slice(None))),
            ("east", "X", (slice(None), -2), (slice(None), -3)),
            ("north", "X", (-2, slice(None)), (-3, slice(None))),
        ],
    )
    def test_wall_node_values(self, face, point, wall, adjacent):
        """Each atom imposes its condition on the wall node itself."""
        f = self._field()
        spacing = self.dx if face in ("west", "east") else self.dy
        adj = f[adjacent]
        cases = [
            (Dirichlet1D(face, 0.3, point=point), jnp.full_like(adj, 0.3)),
            (Neumann1D(face, 0.4, point=point), adj + 0.4 * spacing),
            (Outflow1D(face, point=point), adj),
            (Slip1D(face, 0.25, point=point), 0.25 * adj),
            # alpha u_w + beta (u_w - u_adj) / spacing = gamma
            (
                Robin1D(face, alpha=2.0, beta=0.5, gamma=1.0, point=point),
                (1.0 + 0.5 / spacing * adj) / (2.0 + 0.5 / spacing),
            ),
        ]
        for bc, expected in cases:
            out = bc(f, dx=self.dx, dy=self.dy)
            np.testing.assert_allclose(
                out[wall], expected, rtol=1e-14, err_msg=repr(bc)
            )

    def test_extrapolation_on_wall_node(self):
        """Linear extrapolation to the east wall node of a U-field."""
        f = self._field()
        out = Extrapolation1D("east", order=1, point="U")(f, dx=self.dx, dy=self.dy)
        np.testing.assert_allclose(out[:, -2], 2 * f[:, -3] - f[:, -4], rtol=1e-14)

    def test_outside_slot_is_odd_mirror(self):
        """East / north wall nodes: phi[-1] = 2 phi_wall - phi[-3]."""
        f = self._field()
        out = Dirichlet1D("east", 0.0, point="U")(f, dx=self.dx, dy=self.dy)
        np.testing.assert_allclose(out[:, -1], -f[:, -3], rtol=1e-14)
        out = Dirichlet1D("north", 0.5, point="V")(f, dx=self.dx, dy=self.dy)
        np.testing.assert_allclose(out[-1, :], 1.0 - f[-3, :], rtol=1e-14)
        # the interior below the wall node is untouched
        np.testing.assert_array_equal(out[:-2, :], f[:-2, :])

    @pytest.mark.parametrize(
        ("face", "point"),
        [("west", "V"), ("east", "V"), ("south", "U"), ("north", "U"), ("east", "T")],
    )
    def test_cell_centred_pairs_match_default(self, face, point):
        """Tangential components behave exactly like T-points (ghost = 2v - interior)."""
        f = self._field()
        out = Dirichlet1D(face, 0.3, point=point)(f, dx=self.dx, dy=self.dy)
        ref = Dirichlet1D(face, 0.3)(f, dx=self.dx, dy=self.dy)
        np.testing.assert_array_equal(out, ref)

    def test_periodic_wraps_wall_faces(self):
        """Periodic copies U[:, Nx-2] (the same face as U[:, 0]) into U[:, 0]."""
        f = self._field()
        out = BoundaryConditionSet.periodic()(f, self.dx, self.dy)
        np.testing.assert_array_equal(out[1:-1, 0], f[1:-1, -2])
        np.testing.assert_array_equal(out[1:-1, -1], f[1:-1, 1])

    def test_bad_point_raises(self):
        with pytest.raises(ValueError, match="point must be"):
            Dirichlet1D("west", 0.0, point="Q")(self._field(), dx=1.0, dy=1.0)  # ty: ignore[invalid-argument-type]

    def test_jit(self):
        f = self._field()
        bc = BoundaryConditionSet.closed(point="U")
        out = jax.jit(lambda x: bc(x, self.dx, self.dy))(f)
        np.testing.assert_array_equal(out, bc(f, self.dx, self.dy))
