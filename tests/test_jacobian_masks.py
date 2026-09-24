"""Golden-output regression tests for the ArakawaJacobian2D mask field.

ArakawaJacobian2D is the Layer-3 form of :func:`arakawa_jacobian` (#206):
full-shape T-point output with a zero ghost ring, dry T-cells zeroed via
``mask.h``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from finitevolx import ArakawaJacobian2D, Mask2D, arakawa_jacobian
from tests.fixtures._helpers import assert_matches_golden
from tests.fixtures.inputs import (
    make_grid_2d,
    make_h_field_2d,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_psi_field_2layer,
    make_q_field_2d,
)


def _call(mask=None):
    op = ArakawaJacobian2D(grid=make_grid_2d(), mask=mask)
    return op(make_h_field_2d(), make_q_field_2d())


class TestArakawaJacobian2DMasks:
    def test_unmasked_golden(self):
        assert_matches_golden(_call(), "ArakawaJacobian2D", "__call__", "unmasked")

    def test_masked_golden(self):
        assert_matches_golden(
            _call(make_mask_2d()), "ArakawaJacobian2D", "__call__", "masked"
        )

    def test_all_ocean_matches_unmasked(self):
        np.testing.assert_array_equal(_call(), _call(make_mask_2d_all_ocean()))

    def test_dry_cells_zero(self):
        mask = make_mask_2d()
        out = np.asarray(_call(mask))
        dry = ~np.asarray(mask.h)
        assert dry.any()
        assert np.all(np.isfinite(out))
        np.testing.assert_array_equal(out[dry], 0.0)


class TestArakawaJacobian2DShape:
    def test_matches_functional_interior_and_zero_ghosts(self):
        grid = make_grid_2d()
        f, g = make_h_field_2d(), make_q_field_2d()
        out = ArakawaJacobian2D(grid=grid)(f, g)
        assert out.shape == f.shape
        # out[j, i] = J[j-1, i-1] for 1 <= j <= Ny-2, 1 <= i <= Nx-2
        np.testing.assert_array_equal(
            out[1:-1, 1:-1], arakawa_jacobian(f, g, grid.dx, grid.dy)
        )
        ghost = np.ones(out.shape, dtype=bool)
        # ghost[j, i] = False for 1 <= j <= Ny-2, 1 <= i <= Nx-2
        ghost[1:-1, 1:-1] = False
        np.testing.assert_array_equal(np.asarray(out)[ghost], 0.0)

    def test_batched_layers(self):
        """Leading axes pass through; each layer masked by the same mask.h."""
        mask = make_mask_2d()
        op = ArakawaJacobian2D(grid=make_grid_2d(), mask=mask)
        psi = make_psi_field_2layer()
        q = jnp.stack([make_q_field_2d(), make_h_field_2d()])
        out = op(psi, q)
        assert out.shape == psi.shape
        for k in range(psi.shape[0]):
            np.testing.assert_array_equal(out[k], op(psi[k], q[k]))

    def test_self_jacobian_vanishes(self):
        """J(f, f) = 0 exactly at every point."""
        f = make_h_field_2d()
        out = ArakawaJacobian2D(grid=make_grid_2d(), mask=make_mask_2d())(f, f)
        np.testing.assert_allclose(out, 0.0, atol=1e-12)


class TestArakawaJacobian2DNaNOnLand:
    def test_nan_land_inputs_do_not_leak(self):
        """NaN land values in f / g give the same output as finite ones."""
        mask = make_mask_2d()
        # dry[j, i] = interior dry T-cell (the ghost ring is BC-owned)
        dry = ~np.asarray(mask.h)
        dry[0, :] = dry[-1, :] = dry[:, 0] = dry[:, -1] = False
        op = ArakawaJacobian2D(grid=make_grid_2d(), mask=mask)
        f, g = make_h_field_2d(), make_q_field_2d()
        out = np.asarray(op(f, g))
        # f_nan[j, i] = NaN on dry interior T-cells (same for g)
        out_nan = np.asarray(op(jnp.where(dry, jnp.nan, f), jnp.where(dry, jnp.nan, g)))
        assert np.all(np.isfinite(out_nan))
        np.testing.assert_array_equal(out_nan, out)

    def test_dtype_follows_inputs(self):
        f = make_h_field_2d().astype(jnp.float32)
        out = ArakawaJacobian2D(grid=make_grid_2d())(f, f)
        assert out.dtype == jnp.float32

    def test_closed_basin_keeps_bc_ghosts(self):
        """Under a closed-basin mask the ghost-ring psi still reaches row 1."""
        grid = make_grid_2d()
        # h_wet[j, i] = True for 1 <= j <= Ny-2, 1 <= i <= Nx-2
        h_wet = np.zeros((grid.Ny, grid.Nx), dtype=bool)
        h_wet[1:-1, 1:-1] = True
        op = ArakawaJacobian2D(grid=grid, mask=Mask2D.from_mask(h_wet))
        f, g = make_h_field_2d(), make_q_field_2d()
        # f2[0, i] = f[0, i] + 1 (south ghost row only)
        d = np.asarray(op(f.at[0, :].add(1.0), g) - op(f, g))
        assert np.any(d[1, 1:-1] != 0.0)
        np.testing.assert_array_equal(d[2:, :], 0.0)
