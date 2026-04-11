"""Tests for Coriolis2D and Coriolis3D."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators.coriolis import Coriolis2D, Coriolis3D


@pytest.fixture
def grid2d():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    return CartesianGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)


class TestCoriolis2D:
    def test_output_shapes(self, grid2d):
        cor = Coriolis2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        assert du_cor.shape == (grid2d.Ny, grid2d.Nx)
        assert dv_cor.shape == (grid2d.Ny, grid2d.Nx)

    def test_ghost_ring_zero(self, grid2d):
        cor = Coriolis2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor[0, :], 0.0)
        np.testing.assert_allclose(du_cor[-1, :], 0.0)
        np.testing.assert_allclose(du_cor[:, 0], 0.0)
        np.testing.assert_allclose(du_cor[:, -1], 0.0)
        np.testing.assert_allclose(dv_cor[0, :], 0.0)
        np.testing.assert_allclose(dv_cor[-1, :], 0.0)
        np.testing.assert_allclose(dv_cor[:, 0], 0.0)
        np.testing.assert_allclose(dv_cor[:, -1], 0.0)

    def test_zero_velocity_zero_tendency(self, grid2d):
        """Zero velocity gives zero Coriolis tendency."""
        cor = Coriolis2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor, 0.0, atol=1e-15)
        np.testing.assert_allclose(dv_cor, 0.0, atol=1e-15)

    def test_zero_f_zero_tendency(self, grid2d):
        """Zero Coriolis parameter gives zero tendency regardless of velocity."""
        cor = Coriolis2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.zeros((grid2d.Ny, grid2d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor, 0.0, atol=1e-15)
        np.testing.assert_allclose(dv_cor, 0.0, atol=1e-15)

    def test_uniform_f_uniform_v_du_cor(self, grid2d):
        """Uniform f=1 and uniform v=1 gives du_cor=1 in the interior.

        f_on_u = T_to_U(1) = 1, v_on_u = V_to_U(1) = 1 → du_cor = 1.
        """
        cor = Coriolis2D(grid=grid2d)
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor[1:-1, 1:-1], 1.0, rtol=1e-6)
        np.testing.assert_allclose(dv_cor[1:-1, 1:-1], 0.0, atol=1e-15)

    def test_uniform_f_uniform_u_dv_cor(self, grid2d):
        """Uniform f=1 and uniform u=1 gives dv_cor=-1 in the interior.

        f_on_v = T_to_V(1) = 1, u_on_v = U_to_V(1) = 1 → dv_cor = -1.
        """
        cor = Coriolis2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.zeros((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor[1:-1, 1:-1], 0.0, atol=1e-15)
        np.testing.assert_allclose(dv_cor[1:-1, 1:-1], -1.0, rtol=1e-6)

    def test_sign_convention(self, grid2d):
        """du_cor = +f*v, dv_cor = -f*u — correct sign convention."""
        cor = Coriolis2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = 2.0 * jnp.ones((grid2d.Ny, grid2d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        # du_cor = +f*v = +2 in strict interior
        assert jnp.all(du_cor[1:-1, 1:-1] > 0), "du_cor should be positive"
        # dv_cor = -f*u = -2 in strict interior
        assert jnp.all(dv_cor[1:-1, 1:-1] < 0), "dv_cor should be negative"

    def test_no_nan_output(self, grid2d):
        """Tendencies must not contain NaN for well-defined inputs."""
        cor = Coriolis2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        assert jnp.all(jnp.isfinite(du_cor)), "du_cor contains NaN or Inf"
        assert jnp.all(jnp.isfinite(dv_cor)), "dv_cor contains NaN or Inf"

    def test_spatially_varying_f(self, grid2d):
        """Spatially varying f correctly modulates the tendency magnitude."""
        cor = Coriolis2D(grid=grid2d)
        # f linearly varying in x: f[j, i] = i * dx
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        f = jnp.broadcast_to(x, (grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        u = jnp.zeros((grid2d.Ny, grid2d.Nx))
        du_cor, _ = cor(u, v, f)
        # f_on_u[j, i+1/2] = 1/2 * (f[j,i] + f[j,i+1]) = 1/2*(i + i+1)*dx
        # For j in interior, i in [1, Nx-2]:
        # f_on_u[j, i] = 0.5 * (x[i] + x[i+1])
        # note: interior [1:-1,1:-1] corresponds to i+1/2 for interior i
        i = jnp.arange(1, grid2d.Nx - 1, dtype=float)
        expected = 0.5 * (i * grid2d.dx + (i + 1) * grid2d.dx)
        # du_cor[j, 1:-1] = f_on_u * v_on_u = expected * 1 = expected
        np.testing.assert_allclose(
            np.asarray(du_cor[grid2d.Ny // 2, 1:-1]),
            np.asarray(expected),
            rtol=1e-5,
        )

    def test_mask_zeros_land_points(self, grid2d):
        """Mask zeros tendencies at dry face cells and leaves wet cells unchanged.

        Under the trailing-pad (positive half-step) convention,
        ``mask.u[j, i] = h[j, i] AND h[j, i+1]`` (the east face of T[j, i]),
        so a land column at index 5 zeros ``mask.u[:, 4]`` (east face of
        column 4 borders land) and ``mask.u[:, 5]`` (east face of column 5
        is land).  The ``mask.v`` (north face) is unaffected by a single
        land column except *at* column 5, where ``mask.v[:, 5] =
        h[:, 5] AND h[:+1, 5] = F``.
        """
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))

        # Build a mask with a land column at index 5.
        h_mask = np.ones((grid2d.Ny, grid2d.Nx), dtype=bool)
        h_mask[:, 5] = False
        mask = Mask2D.from_mask(h_mask)

        cor = Coriolis2D(grid=grid2d, mask=mask)
        cor_unmasked = Coriolis2D(grid=grid2d)
        du_masked, dv_masked = cor(u, v, f)
        du_unmasked, _ = cor_unmasked(u, v, f)

        # mask.u (east face) is dry at columns 4 and 5: column 4's east
        # face borders the land column 5, and column 5's east face is land.
        for col in (4, 5):
            assert not bool(mask.u[3, col].item()), f"mask.u[:, {col}] should be dry"
            np.testing.assert_allclose(np.asarray(du_masked[:, col]), 0.0, atol=1e-15)

        # mask.v (north face) is dry only at column 5 (north face of land
        # column 5 is itself land); column 6 is unaffected by the gap.
        assert not bool(mask.v[3, 5].item()), "mask.v[:, 5] should be dry"
        np.testing.assert_allclose(np.asarray(dv_masked[:, 5]), 0.0, atol=1e-15)

        # An interior column away from land should match the unmasked result.
        np.testing.assert_allclose(
            np.asarray(du_masked[1:-1, 2]),
            np.asarray(du_unmasked[1:-1, 2]),
            rtol=1e-6,
        )

    def test_jit_compatible(self, grid2d):
        """Operator is compatible with jax.jit."""
        cor = Coriolis2D(grid=grid2d)
        u = jnp.ones((grid2d.Ny, grid2d.Nx))
        v = jnp.ones((grid2d.Ny, grid2d.Nx))
        f = jnp.ones((grid2d.Ny, grid2d.Nx))
        jit_cor = jax.jit(cor)
        du_cor, dv_cor = jit_cor(u, v, f)
        assert du_cor.shape == (grid2d.Ny, grid2d.Nx)
        assert dv_cor.shape == (grid2d.Ny, grid2d.Nx)


class TestCoriolis3D:
    def test_output_shapes(self, grid3d):
        cor = Coriolis3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        f = jnp.ones((grid3d.Ny, grid3d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        assert du_cor.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert dv_cor.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_ghost_ring_zero(self, grid3d):
        cor = Coriolis3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        f = jnp.ones((grid3d.Ny, grid3d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor[0, :, :], 0.0)
        np.testing.assert_allclose(du_cor[-1, :, :], 0.0)
        np.testing.assert_allclose(du_cor[:, 0, :], 0.0)
        np.testing.assert_allclose(du_cor[:, -1, :], 0.0)
        np.testing.assert_allclose(du_cor[:, :, 0], 0.0)
        np.testing.assert_allclose(du_cor[:, :, -1], 0.0)
        np.testing.assert_allclose(dv_cor[0, :, :], 0.0)
        np.testing.assert_allclose(dv_cor[-1, :, :], 0.0)
        np.testing.assert_allclose(dv_cor[:, 0, :], 0.0)
        np.testing.assert_allclose(dv_cor[:, -1, :], 0.0)
        np.testing.assert_allclose(dv_cor[:, :, 0], 0.0)
        np.testing.assert_allclose(dv_cor[:, :, -1], 0.0)

    def test_zero_velocity_zero_tendency(self, grid3d):
        """Zero velocity gives zero Coriolis tendency."""
        cor = Coriolis3D(grid=grid3d)
        u = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        f = jnp.ones((grid3d.Ny, grid3d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor, 0.0, atol=1e-15)
        np.testing.assert_allclose(dv_cor, 0.0, atol=1e-15)

    def test_uniform_f_uniform_v_du_cor(self, grid3d):
        """Uniform f=1, v=1 gives du_cor=1 at all z-levels in the interior."""
        cor = Coriolis3D(grid=grid3d)
        u = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        f = jnp.ones((grid3d.Ny, grid3d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor[1:-1, 1:-1, 1:-1], 1.0, rtol=1e-6)
        np.testing.assert_allclose(dv_cor[1:-1, 1:-1, 1:-1], 0.0, atol=1e-15)

    def test_uniform_f_uniform_u_dv_cor(self, grid3d):
        """Uniform f=1, u=1 gives dv_cor=-1 at all z-levels in the interior."""
        cor = Coriolis3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.zeros((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        f = jnp.ones((grid3d.Ny, grid3d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        np.testing.assert_allclose(du_cor[1:-1, 1:-1, 1:-1], 0.0, atol=1e-15)
        np.testing.assert_allclose(dv_cor[1:-1, 1:-1, 1:-1], -1.0, rtol=1e-6)

    def test_consistent_with_2d_per_zlevel(self, grid3d):
        """3-D Coriolis result matches 2-D at each z-level independently."""
        grid2d = CartesianGrid2D.from_interior(
            grid3d.Ny - 2, grid3d.Nx - 2, grid3d.Lx, grid3d.Ly
        )
        cor2d = Coriolis2D(grid=grid2d)
        cor3d = Coriolis3D(grid=grid3d)

        x = jnp.arange(grid3d.Nx, dtype=float) * grid3d.dx
        y = jnp.arange(grid3d.Ny, dtype=float) * grid3d.dy
        u2d = jnp.broadcast_to(-y[:, None], (grid3d.Ny, grid3d.Nx))
        v2d = jnp.broadcast_to(x, (grid3d.Ny, grid3d.Nx))
        f2d = jnp.ones((grid3d.Ny, grid3d.Nx))

        u3d = jnp.broadcast_to(u2d[None], (grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v3d = jnp.broadcast_to(v2d[None], (grid3d.Nz, grid3d.Ny, grid3d.Nx))

        du2d, dv2d = cor2d(u2d, v2d, f2d)
        du3d, dv3d = cor3d(u3d, v3d, f2d)

        for k in range(1, grid3d.Nz - 1):
            np.testing.assert_allclose(
                np.asarray(du3d[k, 1:-1, 1:-1]),
                np.asarray(du2d[1:-1, 1:-1]),
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(dv3d[k, 1:-1, 1:-1]),
                np.asarray(dv2d[1:-1, 1:-1]),
                rtol=1e-5,
            )

    def test_no_nan_output(self, grid3d):
        """Tendencies must not contain NaN for well-defined inputs."""
        cor = Coriolis3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        f = jnp.ones((grid3d.Ny, grid3d.Nx))
        du_cor, dv_cor = cor(u, v, f)
        assert jnp.all(jnp.isfinite(du_cor)), "du_cor contains NaN or Inf"
        assert jnp.all(jnp.isfinite(dv_cor)), "dv_cor contains NaN or Inf"

    def test_jit_compatible(self, grid3d):
        """Operator is compatible with jax.jit."""
        cor = Coriolis3D(grid=grid3d)
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        f = jnp.ones((grid3d.Ny, grid3d.Nx))
        jit_cor = jax.jit(cor)
        du_cor, dv_cor = jit_cor(u, v, f)
        assert du_cor.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)
        assert dv_cor.shape == (grid3d.Nz, grid3d.Ny, grid3d.Nx)

    def test_mask_zeros_land_points_3d(self, grid3d):
        """3-D mask zeros dry face cells at every z-level.

        Under the trailing-pad (positive half-step) convention,
        ``mask.u[k, j, i] = h[k, j, i] AND h[k, j, i+1]`` (the east
        face of ``T[k, j, i]``), so a land column at ``i=4`` zeros
        ``mask.u[:, :, 3]`` (east face of column 3 borders land) and
        ``mask.u[:, :, 4]`` (east face of column 4 is itself land).
        After applying the mask, ``du_cor[:, :, 3]`` and
        ``du_cor[:, :, 4]`` must be zero at all z-levels, while an
        adjacent wet column is left unchanged.

        Uses ``Mask3D`` (not a broadcast 2-D mask) per #209 Q4 —
        Coriolis3D was promoted to take ``Mask3D`` for type-uniformity
        with the rest of the 3-D operator suite.
        """
        u = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        v = jnp.ones((grid3d.Nz, grid3d.Ny, grid3d.Nx))
        f = jnp.ones((grid3d.Ny, grid3d.Nx))

        # Build a 3-D mask by broadcasting the horizontal mask over z:
        # land column at i=4 at every z-level.
        h_mask_2d = np.ones((grid3d.Ny, grid3d.Nx), dtype=bool)
        h_mask_2d[:, 4] = False
        h_mask_3d = np.broadcast_to(h_mask_2d, (grid3d.Nz, grid3d.Ny, grid3d.Nx)).copy()
        mask = Mask3D.from_mask(h_mask_3d)

        cor = Coriolis3D(grid=grid3d, mask=mask)
        cor_unmasked = Coriolis3D(grid=grid3d)
        du_masked, dv_masked = cor(u, v, f)
        du_unmasked, _ = cor_unmasked(u, v, f)

        # mask.u (east face) is dry at columns 3 and 4 at every z-level.
        for col in (3, 4):
            assert not bool(mask.u[1, 3, col].item()), (
                f"mask.u[:, :, {col}] should be dry"
            )
            np.testing.assert_allclose(
                np.asarray(du_masked[:, :, col]), 0.0, atol=1e-15
            )

        # mask.v (north face) is dry only at column 4 (the land column).
        assert not bool(mask.v[1, 3, 4].item()), "mask.v[:, :, 4] should be dry"
        np.testing.assert_allclose(np.asarray(dv_masked[:, :, 4]), 0.0, atol=1e-15)

        # An interior column well away from land should match the unmasked result.
        np.testing.assert_allclose(
            np.asarray(du_masked[1:-1, 1:-1, 1]),
            np.asarray(du_unmasked[1:-1, 1:-1, 1]),
            rtol=1e-6,
        )
