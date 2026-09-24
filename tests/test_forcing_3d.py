"""Tests for the 3-D forcing module operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    AbstractForcing,
    CartesianGrid3D,
    LinearDrag2D,
    LinearDrag3D,
    QuadraticDrag2D,
    QuadraticDrag3D,
    RayleighDamping2D,
    RayleighDamping3D,
    WindStress2D,
    WindStress3D,
)
from tests.fixtures.inputs import (
    make_grid_3d,
    make_h_field_3d,
    make_mask_3d,
    make_mask_3d_all_ocean,
    make_u_field_3d,
    make_v_field_3d,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def grid():
    return CartesianGrid3D.from_interior(6, 5, 4, 1.0, 1.0, 1.0)


def _field3d(grid, value=1.0):
    return value * jnp.ones((grid.Nz, grid.Ny, grid.Nx))


def _field2d(grid, value=1.0):
    return value * jnp.ones((grid.Ny, grid.Nx))


def _assert_only_level_nonzero(arr, k):
    others = np.delete(np.asarray(arr), k, axis=0)
    np.testing.assert_array_equal(others, 0.0)


class TestWindStress3D:
    def test_is_forcing(self, grid):
        assert isinstance(WindStress3D(grid), AbstractForcing)

    def test_top_level_injection(self, grid):
        """Levels stack bottom-up, so the top interior level is Nz - 2."""
        tau_x, tau_y = _field2d(grid, 0.1), _field2d(grid, -0.2)
        du, dv = WindStress3D(grid, rho0=1000.0)(tau_x, tau_y, dz_top=10.0)
        assert du.shape == dv.shape == (grid.Nz, grid.Ny, grid.Nx)
        k_top = grid.Nz - 2
        _assert_only_level_nonzero(du, k_top)
        _assert_only_level_nonzero(dv, k_top)
        du_2d, dv_2d = WindStress2D(grid.horizontal_grid(), rho0=1000.0)(
            tau_x, tau_y, dz_top=10.0
        )
        np.testing.assert_allclose(du[k_top], du_2d)
        np.testing.assert_allclose(dv[k_top], dv_2d)
        np.testing.assert_allclose(du[k_top, 1:-1, 1:-1], 0.1 / (1000.0 * 10.0))

    def test_spatially_varying_thickness(self, grid):
        dz = _field2d(grid, 10.0).at[:, 4:].set(30.0)
        du, _ = WindStress3D(grid, rho0=1.0)(_field2d(grid), _field2d(grid), dz)
        np.testing.assert_allclose(du[grid.Nz - 2, 1:-1, 3], 1.0 / 20.0)


class TestBottomDrag3D:
    def test_linear_bottom_level_injection(self, grid):
        u, v = _field3d(grid, 2.0), _field3d(grid, -1.0)
        du, dv = LinearDrag3D(grid)(u, v, r=0.1)
        # Levels stack bottom-up, so the bottom interior level is k = 1.
        k_bot = 1
        _assert_only_level_nonzero(du, k_bot)
        _assert_only_level_nonzero(dv, k_bot)
        np.testing.assert_allclose(du[k_bot, 1:-1, 1:-1], -0.2)
        np.testing.assert_allclose(dv[k_bot, 1:-1, 1:-1], 0.1)

    def test_linear_matches_2d_on_bottom_slice(self):
        grid = make_grid_3d()
        u, v = make_u_field_3d(), make_v_field_3d()
        r = jnp.linspace(1.0, 2.0, grid.Nx)[None, :] * jnp.ones((grid.Ny, 1))
        k_bot = 1
        du, dv = LinearDrag3D(grid)(u, v, r)
        du_2d, dv_2d = LinearDrag2D(grid.horizontal_grid())(u[k_bot], v[k_bot], r)
        np.testing.assert_allclose(du[k_bot], du_2d)
        np.testing.assert_allclose(dv[k_bot], dv_2d)

    def test_quadratic_bottom_level_injection(self, grid):
        u, v = _field3d(grid, 3.0), _field3d(grid, 4.0)
        du, dv = QuadraticDrag3D(grid)(u, v, cd=1.0, h_bot=5.0)
        k_bot = 1
        _assert_only_level_nonzero(du, k_bot)
        _assert_only_level_nonzero(dv, k_bot)
        np.testing.assert_allclose(du[k_bot, 1:-1, 1:-1], -5.0 * 3.0 / 5.0)
        np.testing.assert_allclose(dv[k_bot, 1:-1, 1:-1], -5.0 * 4.0 / 5.0)

    def test_quadratic_matches_2d_on_bottom_slice(self):
        grid = make_grid_3d()
        u, v = make_u_field_3d(), make_v_field_3d()
        k_bot = 1
        du, dv = QuadraticDrag3D(grid)(u, v, cd=2.5e-3, h_bot=50.0)
        du_2d, dv_2d = QuadraticDrag2D(grid.horizontal_grid())(
            u[k_bot], v[k_bot], cd=2.5e-3, h_bot=50.0
        )
        np.testing.assert_allclose(du[k_bot], du_2d)
        np.testing.assert_allclose(dv[k_bot], dv_2d)


class TestRayleighDamping3D:
    def test_all_interior_levels_damped(self, grid):
        q = _field3d(grid, 4.0)
        dq = RayleighDamping3D(grid)(q, r=0.25)
        np.testing.assert_allclose(dq[1:-1, 1:-1, 1:-1], -1.0)
        np.testing.assert_array_equal(dq[0], 0.0)
        np.testing.assert_array_equal(dq[-1], 0.0)
        np.testing.assert_array_equal(dq[:, 0, :], 0.0)
        np.testing.assert_array_equal(dq[:, :, -1], 0.0)

    def test_per_level_rate(self, grid):
        r = jnp.arange(grid.Nz, dtype=float)
        dq = RayleighDamping3D(grid)(_field3d(grid), r)
        for k in range(1, grid.Nz - 1):
            np.testing.assert_allclose(dq[k, 1:-1, 1:-1], -r[k])

    def test_horizontal_rate_broadcast_over_depth(self, grid):
        r = jnp.arange(grid.Ny * grid.Nx, dtype=float).reshape(grid.Ny, grid.Nx)
        dq = RayleighDamping3D(grid)(_field3d(grid), r)
        for k in range(1, grid.Nz - 1):
            np.testing.assert_allclose(dq[k, 1:-1, 1:-1], -r[1:-1, 1:-1])

    def test_full_3d_rate_and_reference(self):
        grid = make_grid_3d()
        q = make_h_field_3d()
        r = 0.5 * jnp.ones_like(q)
        np.testing.assert_allclose(RayleighDamping3D(grid)(q, r, q_ref=q), 0.0)

    def test_reference_variants_match_2d(self, grid):
        q = jnp.arange(grid.Nz * grid.Ny * grid.Nx, dtype=float).reshape(
            grid.Nz, grid.Ny, grid.Nx
        )
        damp2d = RayleighDamping2D(grid.horizontal_grid())
        refs = [
            1.0,
            jnp.arange(grid.Nz, dtype=float),
            _field2d(grid, 2.0),
            0.5 * q,
        ]
        for q_ref in refs:
            dq = RayleighDamping3D(grid)(q, 0.1, q_ref=q_ref)
            ref_arr = jnp.asarray(q_ref)
            for k in range(1, grid.Nz - 1):
                ref_k = ref_arr[k] if ref_arr.ndim in (1, 3) else ref_arr
                np.testing.assert_allclose(dq[k], damp2d(q[k], 0.1, ref_k))


class TestMasks3D:
    def test_dry_points_zeroed(self):
        grid, mask = make_grid_3d(), make_mask_3d()
        u, v, h = make_u_field_3d(), make_v_field_3d(), make_h_field_3d()
        tau = jnp.ones((grid.Ny, grid.Nx))
        mu, mv, mh = (np.asarray(m) for m in (mask.u, mask.v, mask.h))
        for du, dv in [
            WindStress3D(grid, mask=mask)(tau, tau, 10.0),
            LinearDrag3D(grid, mask=mask)(u, v, 0.1),
            QuadraticDrag3D(grid, mask=mask)(u, v, 1e-3, 1.0),
        ]:
            np.testing.assert_array_equal(np.asarray(du)[~mu], 0.0)
            np.testing.assert_array_equal(np.asarray(dv)[~mv], 0.0)
        dq = RayleighDamping3D(grid, mask=mask)(h, 0.1)
        np.testing.assert_array_equal(np.asarray(dq)[~mh], 0.0)

    def test_all_ocean_mask_matches_unmasked(self):
        grid, mask = make_grid_3d(), make_mask_3d_all_ocean()
        u, v, h = make_u_field_3d(), make_v_field_3d(), make_h_field_3d()
        tau = jnp.ones((grid.Ny, grid.Nx))
        for op_cls, args in [
            (WindStress3D, (tau, tau, 10.0)),
            (LinearDrag3D, (u, v, 0.1)),
            (QuadraticDrag3D, (u, v, 1e-3, 1.0)),
            (RayleighDamping3D, (h, 0.1)),
        ]:
            got = op_cls(grid, mask=mask)(*args)
            want = op_cls(grid)(*args)
            for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
                np.testing.assert_allclose(g, w)


class TestTransforms3D:
    def test_jit(self, grid):
        drag = LinearDrag3D(grid)
        du, _ = jax.jit(lambda u: drag(u, u, 0.5))(_field3d(grid))
        np.testing.assert_allclose(du[1, 1:-1, 1:-1], -0.5)

    def test_grad_wrt_per_level_rate(self, grid):
        q = _field3d(grid)
        damp = RayleighDamping3D(grid)

        def loss(r):
            return jnp.sum(damp(q, r))

        g = jax.grad(loss)(jnp.ones(grid.Nz))
        n_h = (grid.Ny - 2) * (grid.Nx - 2)
        np.testing.assert_allclose(g[1:-1], -n_h)
        np.testing.assert_allclose(g[jnp.array([0, -1])], 0.0)


class TestMaskNaNSafety3D:
    """Zero-thickness or NaN-filled dry cells must give exact zeros."""

    def test_zero_thickness_and_nan_land(self):
        grid, mask = make_grid_3d(), make_mask_3d()
        k_top, k_bot = grid.Nz - 2, 1
        mu, mv, mh = (np.asarray(m) for m in (mask.u, mask.v, mask.h))
        tau = jnp.ones((grid.Ny, grid.Nx))
        dz_top = jnp.where(mh[k_top], 50.0, 0.0)
        h_bot = jnp.where(mh[k_bot], 50.0, 0.0)
        u = jnp.where(mu, make_u_field_3d(), jnp.nan)
        v = jnp.where(mv, make_v_field_3d(), jnp.nan)
        q = jnp.where(mh, make_h_field_3d(), jnp.nan)
        for du, dv in [
            WindStress3D(grid, mask=mask)(tau, tau, dz_top=dz_top),
            LinearDrag3D(grid, mask=mask)(u, v, r=0.1),
            QuadraticDrag3D(grid, mask=mask)(u, v, cd=1e-3, h_bot=h_bot),
        ]:
            assert bool(jnp.isfinite(du).all()) and bool(jnp.isfinite(dv).all())
            np.testing.assert_array_equal(np.asarray(du)[~mu], 0.0)
            np.testing.assert_array_equal(np.asarray(dv)[~mv], 0.0)
        dq = RayleighDamping3D(grid, mask=mask)(q, r=0.1)
        assert bool(jnp.isfinite(dq).all())
        np.testing.assert_array_equal(np.asarray(dq)[~mh], 0.0)

    def test_grad_finite_with_zero_land_thickness(self):
        grid, mask = make_grid_3d(), make_mask_3d()
        k_bot = 1
        h_bot = jnp.where(np.asarray(mask.h)[k_bot], 50.0, 0.0)
        u, v = make_u_field_3d(), make_v_field_3d()
        drag = QuadraticDrag3D(grid, mask=mask)

        def loss(h):
            du, dv = drag(u, v, cd=1e-3, h_bot=h)
            return jnp.sum(du) + jnp.sum(dv)

        assert bool(jnp.isfinite(jax.grad(loss)(h_bot)).all())
