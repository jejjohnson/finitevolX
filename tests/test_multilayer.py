"""Tests for the multilayer() vmap helper.

Covers:
- Output shape and per-layer correctness.
- Equivalence with Difference3D in the horizontal interior.
- The key difference: multilayer writes to ALL layers (no ghost z-layers),
  whereas Difference3D only writes to k=1..Nz-2 and leaves k=0, k=Nz-1 zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx as fvx
from finitevolx import multilayer

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def grid2d():
    return fvx.CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    # 4 interior z-levels → Nz = 6 total (including ghost layers at k=0, k=5)
    return fvx.CartesianGrid3D.from_interior(8, 8, 4, 1.0, 1.0, 1.0)


class TestMultilayerOutputShape:
    def test_single_arg_shape(self, grid2d):
        """multilayer wraps vmap: [nl, Ny, Nx] -> [nl, Ny, Nx]."""
        diff2d = fvx.Difference2D(grid=grid2d)
        nl = 3
        h = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        result = multilayer(diff2d.diff_x_T_to_U)(h)
        assert result.shape == (nl, grid2d.Ny, grid2d.Nx)

    def test_multi_arg_shape(self, grid2d):
        """multilayer handles multi-argument operators passed directly."""
        diff2d = fvx.Difference2D(grid=grid2d)
        nl = 5
        u = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        v = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        result = multilayer(diff2d.divergence)(u, v)
        assert result.shape == (nl, grid2d.Ny, grid2d.Nx)


class TestMultilayerPerLayerEquivalence:
    def test_each_layer_matches_2d_call(self, grid2d):
        """Each layer in multilayer output matches a direct 2D call."""
        diff2d = fvx.Difference2D(grid=grid2d)
        nl = 4
        # Each layer is a distinct multiple of the base ramp so that a bug that
        # mixes up layer indices would produce wrong values for at least one k.
        h = jnp.stack(
            [
                jnp.arange(grid2d.Ny * grid2d.Nx, dtype=float).reshape(
                    grid2d.Ny, grid2d.Nx
                )
                * (k + 1)
                for k in range(nl)
            ]
        )
        result_ml = multilayer(diff2d.diff_x_T_to_U)(h)
        for k in range(nl):
            result_2d = diff2d.diff_x_T_to_U(h[k])
            np.testing.assert_array_equal(result_ml[k], result_2d)

    def test_linear_field_all_layers(self, grid2d):
        """Interior of every layer recovers the correct finite difference."""
        diff2d = fvx.Difference2D(grid=grid2d)
        c = 2.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        layer = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        nl = 3
        h = jnp.stack([layer] * nl)
        result = multilayer(diff2d.diff_x_T_to_U)(h)
        # Interior of every layer should be ~c (constant gradient)
        for k in range(nl):
            np.testing.assert_allclose(result[k, 1:-1, 1:-1], c, rtol=1e-5)


class TestMultilayerVsDifference3D:
    """Compare multilayer(vmap) with Difference3D to show interior agreement
    and boundary difference.

    For a field of shape [Nz, Ny, Nx] with no special z-ghost treatment:

    - The horizontal interior [k, 1:-1, 1:-1] is IDENTICAL for both approaches
      when k is an interior z-level (k=1..Nz-2).
    - Difference3D leaves k=0 and k=Nz-1 as zero (ghost z-layers).
    - multilayer(fn) writes to ALL layers including k=0 and k=Nz-1.
    """

    def test_interior_agreement_diff_x(self, grid2d, grid3d):
        """Interior [k=1..Nz-2, 1:-1, 1:-1] matches between vmap and 3D."""
        diff2d = fvx.Difference2D(grid=grid2d)
        diff3d = fvx.Difference3D(grid=grid3d)
        Nz = grid3d.Nz  # total z-size including ghost layers (= 6 for interior 4)

        rng = jnp.arange(Nz * grid3d.Ny * grid3d.Nx, dtype=float).reshape(
            Nz, grid3d.Ny, grid3d.Nx
        )

        ml_result = multilayer(diff2d.diff_x_T_to_U)(rng)
        d3_result = diff3d.diff_x_T_to_U(rng)

        # Interior z-levels and horizontal interior match
        np.testing.assert_allclose(
            ml_result[1:-1, 1:-1, 1:-1],
            d3_result[1:-1, 1:-1, 1:-1],
            rtol=1e-6,
        )

    def test_interior_agreement_diff_y(self, grid2d, grid3d):
        """Interior [k=1..Nz-2, 1:-1, 1:-1] matches between vmap and 3D (y)."""
        diff2d = fvx.Difference2D(grid=grid2d)
        diff3d = fvx.Difference3D(grid=grid3d)
        Nz = grid3d.Nz

        rng = jnp.arange(Nz * grid3d.Ny * grid3d.Nx, dtype=float).reshape(
            Nz, grid3d.Ny, grid3d.Nx
        )

        ml_result = multilayer(diff2d.diff_y_T_to_V)(rng)
        d3_result = diff3d.diff_y_T_to_V(rng)

        np.testing.assert_allclose(
            ml_result[1:-1, 1:-1, 1:-1],
            d3_result[1:-1, 1:-1, 1:-1],
            rtol=1e-6,
        )

    def test_diff3d_ghost_z_layers_are_zero(self, grid2d, grid3d):
        """Difference3D leaves k=0 and k=Nz-1 as zero (ghost z-shells)."""
        diff3d = fvx.Difference3D(grid=grid3d)
        Nz = grid3d.Nz

        rng = jnp.arange(Nz * grid3d.Ny * grid3d.Nx, dtype=float).reshape(
            Nz, grid3d.Ny, grid3d.Nx
        )

        d3_result = diff3d.diff_x_T_to_U(rng)

        # Top and bottom z-shells are zero by the 3D ghost-layer convention
        np.testing.assert_array_equal(d3_result[0], 0.0)
        np.testing.assert_array_equal(d3_result[-1], 0.0)

    def test_multilayer_all_z_levels_written(self, grid2d, grid3d):
        """multilayer writes to ALL layers — there is no ghost z-layer concept."""
        diff2d = fvx.Difference2D(grid=grid2d)
        Nz = grid3d.Nz

        # Use a field that produces non-zero differences everywhere
        c = 1.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.dx
        layer = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        h = jnp.stack([layer] * Nz)  # [Nz, Ny, Nx]

        ml_result = multilayer(diff2d.diff_x_T_to_U)(h)

        # k=0 and k=Nz-1 are real layers: interior values are non-zero
        np.testing.assert_allclose(ml_result[0, 1:-1, 1:-1], c, rtol=1e-5)
        np.testing.assert_allclose(ml_result[-1, 1:-1, 1:-1], c, rtol=1e-5)
