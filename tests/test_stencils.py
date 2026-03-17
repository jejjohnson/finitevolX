"""Tests for raw stencil primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from finitevolx._src.operators.stencils import (
    avg_x_bwd,
    avg_x_bwd_1d,
    avg_x_bwd_3d,
    avg_x_fwd,
    avg_x_fwd_1d,
    avg_x_fwd_3d,
    avg_xbwd_yfwd,
    avg_xfwd_ybwd,
    avg_xy_bwd,
    avg_xy_fwd,
    avg_y_bwd,
    avg_y_bwd_3d,
    avg_y_fwd,
    avg_y_fwd_3d,
    diff_x_bwd,
    diff_x_bwd_1d,
    diff_x_bwd_3d,
    diff_x_fwd,
    diff_x_fwd_1d,
    diff_x_fwd_3d,
    diff_y_bwd,
    diff_y_bwd_3d,
    diff_y_fwd,
    diff_y_fwd_3d,
)

# =====================================================================
# 1-D stencils
# =====================================================================


class TestDiffStencils1D:
    """Tests for 1-D raw difference stencils."""

    def test_diff_x_fwd_1d_shape(self):
        h = jnp.zeros(8)
        out = diff_x_fwd_1d(h)
        assert out.shape == (6,)

    def test_diff_x_bwd_1d_shape(self):
        h = jnp.zeros(8)
        out = diff_x_bwd_1d(h)
        assert out.shape == (6,)

    def test_diff_x_fwd_1d_linear(self):
        """Forward diff of a linear field should be constant."""
        h = jnp.arange(8, dtype=jnp.float32)  # 0,1,2,...,7
        out = diff_x_fwd_1d(h)
        assert jnp.allclose(out, 1.0)

    def test_diff_x_bwd_1d_linear(self):
        """Backward diff of a linear field should be constant."""
        h = jnp.arange(8, dtype=jnp.float32)
        out = diff_x_bwd_1d(h)
        assert jnp.allclose(out, 1.0)

    def test_fwd_bwd_second_derivative_1d(self):
        """diff_x_fwd - diff_x_bwd should give the second derivative."""
        # Quadratic: h[i] = i^2
        x = jnp.arange(8, dtype=jnp.float32)
        h = x**2
        d2h = diff_x_fwd_1d(h) - diff_x_bwd_1d(h)
        # Second derivative of x^2 is 2 everywhere
        assert jnp.allclose(d2h, 2.0)

    def test_diff_x_fwd_1d_constant(self):
        """Forward diff of a constant field should be zero."""
        h = jnp.ones(8)
        out = diff_x_fwd_1d(h)
        assert jnp.allclose(out, 0.0)

    def test_diff_x_fwd_1d_jit(self):
        h = jnp.arange(8, dtype=jnp.float32)
        out_eager = diff_x_fwd_1d(h)
        out_jit = jax.jit(diff_x_fwd_1d)(h)
        assert jnp.allclose(out_eager, out_jit)


# =====================================================================
# 2-D stencils
# =====================================================================


class TestDiffStencils2D:
    """Tests for 2-D raw difference stencils."""

    @pytest.fixture()
    def shape(self):
        return (10, 12)

    def test_diff_x_fwd_shape(self, shape):
        h = jnp.zeros(shape)
        out = diff_x_fwd(h)
        assert out.shape == (shape[0] - 2, shape[1] - 2)

    def test_diff_y_fwd_shape(self, shape):
        h = jnp.zeros(shape)
        out = diff_y_fwd(h)
        assert out.shape == (shape[0] - 2, shape[1] - 2)

    def test_diff_x_bwd_shape(self, shape):
        h = jnp.zeros(shape)
        out = diff_x_bwd(h)
        assert out.shape == (shape[0] - 2, shape[1] - 2)

    def test_diff_y_bwd_shape(self, shape):
        h = jnp.zeros(shape)
        out = diff_y_bwd(h)
        assert out.shape == (shape[0] - 2, shape[1] - 2)

    def test_diff_x_fwd_linear_in_x(self, shape):
        """Forward x-diff of a field linear in x should be constant."""
        ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        h = jnp.broadcast_to(x, (ny, nx))
        out = diff_x_fwd(h)
        assert jnp.allclose(out, 1.0)

    def test_diff_y_fwd_linear_in_y(self, shape):
        """Forward y-diff of a field linear in y should be constant."""
        ny, nx = shape
        y = jnp.arange(ny, dtype=jnp.float32)[:, None]
        h = jnp.broadcast_to(y, (ny, nx))
        out = diff_y_fwd(h)
        assert jnp.allclose(out, 1.0)

    def test_diff_x_fwd_constant_in_y(self, shape):
        """Forward x-diff should be zero for a field constant in x."""
        ny, nx = shape
        y = jnp.arange(ny, dtype=jnp.float32)[:, None]
        h = jnp.broadcast_to(y, (ny, nx))
        out = diff_x_fwd(h)
        assert jnp.allclose(out, 0.0)

    def test_diff_y_fwd_constant_in_x(self, shape):
        """Forward y-diff should be zero for a field constant in y."""
        ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        h = jnp.broadcast_to(x, (ny, nx))
        out = diff_y_fwd(h)
        assert jnp.allclose(out, 0.0)

    def test_fwd_bwd_second_derivative_x(self, shape):
        """diff_x_fwd - diff_x_bwd gives the second x-derivative."""
        ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        h = jnp.broadcast_to(x**2, (ny, nx))
        d2h = diff_x_fwd(h) - diff_x_bwd(h)
        assert jnp.allclose(d2h, 2.0)

    def test_fwd_bwd_second_derivative_y(self, shape):
        """diff_y_fwd - diff_y_bwd gives the second y-derivative."""
        ny, nx = shape
        y = jnp.arange(ny, dtype=jnp.float32)[:, None]
        h = jnp.broadcast_to(y**2, (ny, nx))
        d2h = diff_y_fwd(h) - diff_y_bwd(h)
        assert jnp.allclose(d2h, 2.0)

    def test_symmetry_fwd_bwd_x(self, shape):
        """diff_x_bwd(h) == -diff_x_fwd(h) reversed, for symmetric h."""
        h = jnp.ones(shape)
        assert jnp.allclose(diff_x_fwd(h), 0.0)
        assert jnp.allclose(diff_x_bwd(h), 0.0)

    def test_diff_x_fwd_jit(self, shape):
        h = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape)
        out_eager = diff_x_fwd(h)
        out_jit = jax.jit(diff_x_fwd)(h)
        assert jnp.allclose(out_eager, out_jit)

    def test_diff_y_fwd_jit(self, shape):
        h = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape)
        out_eager = diff_y_fwd(h)
        out_jit = jax.jit(diff_y_fwd)(h)
        assert jnp.allclose(out_eager, out_jit)


# =====================================================================
# 3-D stencils
# =====================================================================


class TestDiffStencils3D:
    """Tests for 3-D raw difference stencils."""

    @pytest.fixture()
    def shape(self):
        return (6, 10, 12)

    def test_diff_x_fwd_3d_shape(self, shape):
        h = jnp.zeros(shape)
        out = diff_x_fwd_3d(h)
        assert out.shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    def test_diff_y_fwd_3d_shape(self, shape):
        h = jnp.zeros(shape)
        out = diff_y_fwd_3d(h)
        assert out.shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    def test_diff_x_bwd_3d_shape(self, shape):
        h = jnp.zeros(shape)
        out = diff_x_bwd_3d(h)
        assert out.shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    def test_diff_y_bwd_3d_shape(self, shape):
        h = jnp.zeros(shape)
        out = diff_y_bwd_3d(h)
        assert out.shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    def test_diff_x_fwd_3d_linear(self, shape):
        """Forward x-diff of a 3D field linear in x should be constant."""
        nz, ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        h = jnp.broadcast_to(x, (nz, ny, nx))
        out = diff_x_fwd_3d(h)
        assert jnp.allclose(out, 1.0)

    def test_diff_y_fwd_3d_linear(self, shape):
        """Forward y-diff of a 3D field linear in y should be constant."""
        nz, ny, nx = shape
        y = jnp.arange(ny, dtype=jnp.float32)[None, :, None]
        h = jnp.broadcast_to(y, (nz, ny, nx))
        out = diff_y_fwd_3d(h)
        assert jnp.allclose(out, 1.0)

    def test_fwd_bwd_second_derivative_3d_x(self, shape):
        """diff_x_fwd_3d - diff_x_bwd_3d gives the second x-derivative."""
        nz, ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        h = jnp.broadcast_to(x**2, (nz, ny, nx))
        d2h = diff_x_fwd_3d(h) - diff_x_bwd_3d(h)
        assert jnp.allclose(d2h, 2.0)

    def test_fwd_bwd_second_derivative_3d_y(self, shape):
        """diff_y_fwd_3d - diff_y_bwd_3d gives the second y-derivative."""
        nz, ny, nx = shape
        y = jnp.arange(ny, dtype=jnp.float32)[None, :, None]
        h = jnp.broadcast_to(y**2, (nz, ny, nx))
        d2h = diff_y_fwd_3d(h) - diff_y_bwd_3d(h)
        assert jnp.allclose(d2h, 2.0)

    def test_diff_x_fwd_3d_jit(self, shape):
        h = jnp.arange(shape[0] * shape[1] * shape[2], dtype=jnp.float32).reshape(shape)
        out_eager = diff_x_fwd_3d(h)
        out_jit = jax.jit(diff_x_fwd_3d)(h)
        assert jnp.allclose(out_eager, out_jit)

    def test_diff_x_constant_in_z(self, shape):
        """x-diff should be independent of z for a field constant in z."""
        nz, ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        h = jnp.broadcast_to(x, (nz, ny, nx))
        out = diff_x_fwd_3d(h)
        # All z-levels should be identical
        for k in range(out.shape[0]):
            assert jnp.allclose(out[k], out[0])


# =====================================================================
# 1-D averaging stencils
# =====================================================================


class TestAvgStencils1D:
    """Tests for 1-D raw averaging stencils."""

    def test_avg_x_fwd_1d_shape(self):
        h = jnp.zeros(8)
        assert avg_x_fwd_1d(h).shape == (6,)

    def test_avg_x_bwd_1d_shape(self):
        h = jnp.zeros(8)
        assert avg_x_bwd_1d(h).shape == (6,)

    def test_avg_x_fwd_1d_constant(self):
        """Average of a constant field should be that constant."""
        h = 3.0 * jnp.ones(8)
        assert jnp.allclose(avg_x_fwd_1d(h), 3.0)

    def test_avg_x_bwd_1d_constant(self):
        h = 3.0 * jnp.ones(8)
        assert jnp.allclose(avg_x_bwd_1d(h), 3.0)

    def test_avg_x_fwd_1d_linear(self):
        """Average of a linear field at half-points should be the midpoint value."""
        h = jnp.arange(8, dtype=jnp.float32)
        out = avg_x_fwd_1d(h)
        # h̄[i+½] = ½(h[i]+h[i+1]) for i in [1..6] → midpoints 1.5, 2.5, ..., 6.5
        expected = jnp.arange(1, 7, dtype=jnp.float32) + 0.5
        assert jnp.allclose(out, expected)

    def test_avg_x_fwd_1d_jit(self):
        h = jnp.arange(8, dtype=jnp.float32)
        assert jnp.allclose(jax.jit(avg_x_fwd_1d)(h), avg_x_fwd_1d(h))


# =====================================================================
# 2-D averaging stencils (2-point)
# =====================================================================


class TestAvgStencils2D:
    """Tests for 2-D raw averaging stencils (2-point)."""

    @pytest.fixture()
    def shape(self):
        return (10, 12)

    def test_avg_x_fwd_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_x_fwd(h).shape == (shape[0] - 2, shape[1] - 2)

    def test_avg_y_fwd_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_y_fwd(h).shape == (shape[0] - 2, shape[1] - 2)

    def test_avg_x_bwd_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_x_bwd(h).shape == (shape[0] - 2, shape[1] - 2)

    def test_avg_y_bwd_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_y_bwd(h).shape == (shape[0] - 2, shape[1] - 2)

    def test_avg_x_fwd_constant(self, shape):
        h = 5.0 * jnp.ones(shape)
        assert jnp.allclose(avg_x_fwd(h), 5.0)

    def test_avg_y_fwd_constant(self, shape):
        h = 5.0 * jnp.ones(shape)
        assert jnp.allclose(avg_y_fwd(h), 5.0)

    def test_avg_x_fwd_linear_in_x(self, shape):
        """Average of a field linear in x should give midpoint values."""
        ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        h = jnp.broadcast_to(x, (ny, nx))
        out = avg_x_fwd(h)
        # Each value should be x[i] + 0.5 for interior i
        expected = jnp.broadcast_to(
            jnp.arange(1, nx - 1, dtype=jnp.float32) + 0.5,
            (ny - 2, nx - 2),
        )
        assert jnp.allclose(out, expected)

    def test_avg_y_fwd_linear_in_y(self, shape):
        """Average of a field linear in y should give midpoint values."""
        ny, nx = shape
        y = jnp.arange(ny, dtype=jnp.float32)[:, None]
        h = jnp.broadcast_to(y, (ny, nx))
        out = avg_y_fwd(h)
        expected = jnp.broadcast_to(
            (jnp.arange(1, ny - 1, dtype=jnp.float32) + 0.5)[:, None],
            (ny - 2, nx - 2),
        )
        assert jnp.allclose(out, expected)

    def test_avg_preserves_constant_across_directions(self, shape):
        """All 2-point averages of a constant field give the same constant."""
        h = 7.0 * jnp.ones(shape)
        for fn in [avg_x_fwd, avg_x_bwd, avg_y_fwd, avg_y_bwd]:
            assert jnp.allclose(fn(h), 7.0)

    def test_avg_x_fwd_jit(self, shape):
        h = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape)
        assert jnp.allclose(jax.jit(avg_x_fwd)(h), avg_x_fwd(h))


# =====================================================================
# 2-D averaging stencils (4-point bilinear)
# =====================================================================


class TestAvgStencils2DBilinear:
    """Tests for 2-D raw averaging stencils (4-point bilinear)."""

    @pytest.fixture()
    def shape(self):
        return (10, 12)

    def test_avg_xy_fwd_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_xy_fwd(h).shape == (shape[0] - 2, shape[1] - 2)

    def test_avg_xy_bwd_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_xy_bwd(h).shape == (shape[0] - 2, shape[1] - 2)

    def test_avg_xbwd_yfwd_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_xbwd_yfwd(h).shape == (shape[0] - 2, shape[1] - 2)

    def test_avg_xfwd_ybwd_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_xfwd_ybwd(h).shape == (shape[0] - 2, shape[1] - 2)

    def test_avg_xy_fwd_constant(self, shape):
        h = 4.0 * jnp.ones(shape)
        assert jnp.allclose(avg_xy_fwd(h), 4.0)

    def test_avg_xy_bwd_constant(self, shape):
        h = 4.0 * jnp.ones(shape)
        assert jnp.allclose(avg_xy_bwd(h), 4.0)

    def test_avg_xbwd_yfwd_constant(self, shape):
        h = 4.0 * jnp.ones(shape)
        assert jnp.allclose(avg_xbwd_yfwd(h), 4.0)

    def test_avg_xfwd_ybwd_constant(self, shape):
        h = 4.0 * jnp.ones(shape)
        assert jnp.allclose(avg_xfwd_ybwd(h), 4.0)

    def test_avg_xy_fwd_bilinear_field(self, shape):
        """Bilinear average of h = x*y at (j+½, i+½) should be (j+½)*(i+½)."""
        ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        y = jnp.arange(ny, dtype=jnp.float32)[:, None]
        h = x * y
        out = avg_xy_fwd(h)
        # Expected: (j+0.5)*(i+0.5) for interior j in [1,ny-2], i in [1,nx-2]
        xi = jnp.arange(1, nx - 1, dtype=jnp.float32) + 0.5
        yj = (jnp.arange(1, ny - 1, dtype=jnp.float32) + 0.5)[:, None]
        expected = yj * xi
        assert jnp.allclose(out, expected)

    def test_avg_xy_fwd_jit(self, shape):
        h = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape)
        assert jnp.allclose(jax.jit(avg_xy_fwd)(h), avg_xy_fwd(h))


# =====================================================================
# 3-D averaging stencils
# =====================================================================


class TestAvgStencils3D:
    """Tests for 3-D raw averaging stencils."""

    @pytest.fixture()
    def shape(self):
        return (6, 10, 12)

    def test_avg_x_fwd_3d_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_x_fwd_3d(h).shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    def test_avg_y_fwd_3d_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_y_fwd_3d(h).shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    def test_avg_x_bwd_3d_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_x_bwd_3d(h).shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    def test_avg_y_bwd_3d_shape(self, shape):
        h = jnp.zeros(shape)
        assert avg_y_bwd_3d(h).shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    def test_avg_x_fwd_3d_constant(self, shape):
        h = 2.0 * jnp.ones(shape)
        assert jnp.allclose(avg_x_fwd_3d(h), 2.0)

    def test_avg_y_fwd_3d_constant(self, shape):
        h = 2.0 * jnp.ones(shape)
        assert jnp.allclose(avg_y_fwd_3d(h), 2.0)

    def test_avg_x_fwd_3d_linear(self, shape):
        """Average of a 3D field linear in x should give midpoint values."""
        nz, ny, nx = shape
        x = jnp.arange(nx, dtype=jnp.float32)
        h = jnp.broadcast_to(x, (nz, ny, nx))
        out = avg_x_fwd_3d(h)
        expected_row = jnp.arange(1, nx - 1, dtype=jnp.float32) + 0.5
        expected = jnp.broadcast_to(expected_row, (nz - 2, ny - 2, nx - 2))
        assert jnp.allclose(out, expected)

    def test_avg_x_fwd_3d_jit(self, shape):
        h = jnp.arange(shape[0] * shape[1] * shape[2], dtype=jnp.float32).reshape(shape)
        assert jnp.allclose(jax.jit(avg_x_fwd_3d)(h), avg_x_fwd_3d(h))
