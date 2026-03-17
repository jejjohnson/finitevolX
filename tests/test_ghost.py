"""Tests for ghost-ring zeroing utilities."""

import jax
import jax.numpy as jnp

from finitevolx._src.operators._ghost import interior, zero_z_ghosts

# ---------------------------------------------------------------------------
# interior
# ---------------------------------------------------------------------------


class TestInterior:
    """Tests for the ``interior`` helper."""

    def test_1d_ghost1(self):
        like = jnp.zeros(6)
        values = jnp.ones(4)
        out = interior(values, like, ghost=1)
        assert out.shape == (6,)
        assert float(out[0]) == 0.0
        assert float(out[-1]) == 0.0
        assert jnp.all(out[1:-1] == 1.0)

    def test_2d_ghost1(self):
        like = jnp.zeros((6, 8))
        values = jnp.ones((4, 6))
        out = interior(values, like, ghost=1)
        assert out.shape == (6, 8)
        # Ghost ring is zero
        assert jnp.all(out[0, :] == 0.0)
        assert jnp.all(out[-1, :] == 0.0)
        assert jnp.all(out[:, 0] == 0.0)
        assert jnp.all(out[:, -1] == 0.0)
        # Interior is filled
        assert jnp.all(out[1:-1, 1:-1] == 1.0)

    def test_3d_ghost1(self):
        like = jnp.zeros((4, 6, 8))
        values = jnp.ones((2, 4, 6))
        out = interior(values, like, ghost=1)
        assert out.shape == (4, 6, 8)
        assert jnp.all(out[0, :, :] == 0.0)
        assert jnp.all(out[-1, :, :] == 0.0)
        assert jnp.all(out[1:-1, 1:-1, 1:-1] == 1.0)

    def test_2d_ghost2(self):
        like = jnp.zeros((8, 10))
        values = jnp.ones((4, 6))
        out = interior(values, like, ghost=2)
        assert out.shape == (8, 10)
        # Two-cell ring is zero
        assert jnp.all(out[:2, :] == 0.0)
        assert jnp.all(out[-2:, :] == 0.0)
        assert jnp.all(out[:, :2] == 0.0)
        assert jnp.all(out[:, -2:] == 0.0)
        # Interior is filled
        assert jnp.all(out[2:-2, 2:-2] == 1.0)

    def test_dtype_inference(self):
        like = jnp.zeros((4, 4), dtype=jnp.float32)
        values = jnp.ones((2, 2), dtype=jnp.float32)
        out = interior(values, like)
        assert out.dtype == jnp.float32

    def test_jit_compatible(self):
        like = jnp.zeros((6, 8))
        values = jnp.ones((4, 6))
        out_jit = jax.jit(interior)(values, like)
        out_eager = interior(values, like)
        assert jnp.allclose(out_jit, out_eager)


# ---------------------------------------------------------------------------
# zero_z_ghosts
# ---------------------------------------------------------------------------


class TestZeroZGhosts:
    """Tests for the ``zero_z_ghosts`` helper."""

    def test_preserves_interior(self):
        arr = jnp.ones((4, 6, 8))
        out = zero_z_ghosts(arr)
        assert out.shape == arr.shape
        assert jnp.all(out[0] == 0.0)
        assert jnp.all(out[-1] == 0.0)
        assert jnp.all(out[1:-1] == 1.0)

    def test_jit_compatible(self):
        arr = jnp.ones((4, 6, 8))
        out_jit = jax.jit(zero_z_ghosts)(arr)
        out_eager = zero_z_ghosts(arr)
        assert jnp.allclose(out_jit, out_eager)

    def test_small_array(self):
        arr = jnp.ones((2, 3, 3))
        out = zero_z_ghosts(arr)
        # Both slices zeroed — entire array is zero
        assert jnp.all(out == 0.0)
