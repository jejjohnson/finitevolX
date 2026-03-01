"""Tests for boundary condition helpers."""
import jax.numpy as jnp
import numpy as np
import pytest

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
        f = jnp.zeros((6, 6)).at[1:-1, 1:-1].set(
            jnp.arange(16, dtype=float).reshape(4, 4)
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[0, :], result[-2, :])

    def test_north_ghost_equals_south_interior(self):
        f = jnp.zeros((6, 6)).at[1:-1, 1:-1].set(
            jnp.arange(16, dtype=float).reshape(4, 4)
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[-1, :], result[1, :])

    def test_west_ghost_equals_east_interior(self):
        f = jnp.zeros((6, 6)).at[1:-1, 1:-1].set(
            jnp.arange(16, dtype=float).reshape(4, 4)
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[:, 0], result[:, -2])

    def test_east_ghost_equals_west_interior(self):
        f = jnp.zeros((6, 6)).at[1:-1, 1:-1].set(
            jnp.arange(16, dtype=float).reshape(4, 4)
        )
        result = enforce_periodic(f)
        np.testing.assert_array_equal(result[:, -1], result[:, 1])

    def test_constant_field_stays_constant(self):
        f = 5.0 * jnp.ones((8, 8))
        result = enforce_periodic(f)
        np.testing.assert_allclose(result, 5.0)
