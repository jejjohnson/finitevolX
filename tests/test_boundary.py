"""Tests for boundary condition helpers."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    BoundaryConditionSet,
    Dirichlet1D,
    FieldBCSet,
    Neumann1D,
    Sponge1D,
)
from finitevolx._src.bc_1d import Outflow1D, Periodic1D, Reflective1D
from finitevolx._src.boundary import enforce_periodic, pad_interior


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
