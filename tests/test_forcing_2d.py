"""Tests for the 2-D forcing module operators."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    AbstractForcing,
    CartesianGrid2D,
    ForcingSum,
    LinearDrag2D,
    QuadraticDrag2D,
    RayleighDamping2D,
    WindStress2D,
    multilayer,
)
from tests.fixtures.inputs import (
    make_grid_2d,
    make_h_field_2d,
    make_mask_2d,
    make_mask_2d_all_ocean,
    make_u_field_2d,
    make_v_field_2d,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def grid():
    return CartesianGrid2D.from_interior(8, 6, 1.0, 1.0)


def _ones(grid, value=1.0):
    return value * jnp.ones((grid.Ny, grid.Nx))


def _assert_ghost_ring_zero(arr):
    np.testing.assert_array_equal(arr[0, :], 0.0)
    np.testing.assert_array_equal(arr[-1, :], 0.0)
    np.testing.assert_array_equal(arr[:, 0], 0.0)
    np.testing.assert_array_equal(arr[:, -1], 0.0)


class TestWindStress2D:
    def test_is_forcing(self, grid):
        assert isinstance(WindStress2D(grid), AbstractForcing)

    def test_uniform_stress(self, grid):
        wind = WindStress2D(grid, rho0=1000.0)
        du, dv = wind(_ones(grid, 0.1), _ones(grid, -0.2), dz_top=50.0)
        assert du.shape == dv.shape == (grid.Ny, grid.Nx)
        np.testing.assert_allclose(du[1:-1, 1:-1], 0.1 / (1000.0 * 50.0))
        np.testing.assert_allclose(dv[1:-1, 1:-1], -0.2 / (1000.0 * 50.0))
        _assert_ghost_ring_zero(du)
        _assert_ghost_ring_zero(dv)

    def test_interpolates_to_faces(self, grid):
        """tau_x linear in x -> du is the face-midpoint value."""
        x = jnp.arange(grid.Nx, dtype=float)
        tau_x = jnp.broadcast_to(x, (grid.Ny, grid.Nx))
        tau_y = jnp.broadcast_to(x, (grid.Ny, grid.Nx))
        du, dv = WindStress2D(grid, rho0=1.0)(tau_x, tau_y, dz_top=1.0)
        # tau_x_on_u[j, i+1/2] = i + 1/2
        interior_shape = (grid.Ny - 2, grid.Nx - 2)
        np.testing.assert_allclose(
            du[1:-1, 1:-1], np.broadcast_to(x[1:-1] + 0.5, interior_shape)
        )
        # tau_y_on_v[j+1/2, i] = i (uniform in y)
        np.testing.assert_allclose(
            dv[1:-1, 1:-1], np.broadcast_to(x[1:-1], interior_shape)
        )

    def test_spatially_varying_thickness(self, grid):
        dz = _ones(grid, 10.0).at[:, 4:].set(30.0)
        du, dv = WindStress2D(grid, rho0=1.0)(_ones(grid), _ones(grid), dz_top=dz)
        # U-face between the two regions sees the averaged thickness 20
        np.testing.assert_allclose(du[1:-1, 3], 1.0 / 20.0)
        np.testing.assert_allclose(du[1:-1, 1], 1.0 / 10.0)
        np.testing.assert_allclose(du[1:-1, 5], 1.0 / 30.0)
        np.testing.assert_allclose(dv[1:-1, 5], 1.0 / 30.0)

    def test_default_rho0(self, grid):
        du, _ = WindStress2D(grid)(_ones(grid), _ones(grid), dz_top=1.0)
        np.testing.assert_allclose(du[1:-1, 1:-1], 1.0 / 1025.0)


class TestLinearDrag2D:
    def test_scalar_coefficient(self, grid):
        u, v = _ones(grid, 2.0), _ones(grid, -1.0)
        du, dv = LinearDrag2D(grid)(u, v, r=0.1)
        np.testing.assert_allclose(du[1:-1, 1:-1], -0.2)
        np.testing.assert_allclose(dv[1:-1, 1:-1], 0.1)
        _assert_ghost_ring_zero(du)
        _assert_ghost_ring_zero(dv)

    def test_spatially_varying_coefficient_interpolated(self, grid):
        r = _ones(grid, 1.0).at[:, 4:].set(3.0)
        du, dv = LinearDrag2D(grid)(_ones(grid), _ones(grid), r=r)
        np.testing.assert_allclose(du[1:-1, 3], -2.0)
        np.testing.assert_allclose(dv[1:-1, 3], -1.0)
        np.testing.assert_allclose(dv[1:-1, 4], -3.0)

    def test_opposes_flow(self):
        grid = make_grid_2d()
        u, v = make_u_field_2d(), make_v_field_2d()
        du, dv = LinearDrag2D(grid)(u, v, r=1e-3)
        assert bool((du[1:-1, 1:-1] * u[1:-1, 1:-1] <= 0).all())
        assert bool((dv[1:-1, 1:-1] * v[1:-1, 1:-1] <= 0).all())

    def test_multilayer(self, grid):
        u = jnp.stack([_ones(grid, k + 1.0) for k in range(3)])
        drag = LinearDrag2D(grid)
        du, _ = multilayer(lambda u_k, v_k: drag(u_k, v_k, 0.5))(u, u)
        assert du.shape == u.shape
        np.testing.assert_allclose(du[2, 1:-1, 1:-1], -1.5)


class TestQuadraticDrag2D:
    def test_uniform_zonal_flow(self, grid):
        du, dv = QuadraticDrag2D(grid)(
            _ones(grid, 2.0), _ones(grid, 0.0), cd=1e-3, h_bot=10.0
        )
        np.testing.assert_allclose(du[1:-1, 1:-1], -1e-3 * 2.0 * 2.0 / 10.0)
        np.testing.assert_allclose(dv, 0.0)
        _assert_ghost_ring_zero(du)

    def test_speed_uses_both_components(self, grid):
        """|u| = sqrt(3^2 + 4^2) = 5 for uniform flow."""
        du, dv = QuadraticDrag2D(grid)(
            _ones(grid, 3.0), _ones(grid, 4.0), cd=1.0, h_bot=1.0
        )
        np.testing.assert_allclose(du[1:-1, 1:-1], -5.0 * 3.0)
        np.testing.assert_allclose(dv[1:-1, 1:-1], -5.0 * 4.0)

    def test_quadratic_scaling(self):
        grid = make_grid_2d()
        u, v = make_u_field_2d(), make_v_field_2d()
        drag = QuadraticDrag2D(grid)
        du1, dv1 = drag(u, v, cd=1e-3, h_bot=50.0)
        du2, dv2 = drag(2 * u, 2 * v, cd=1e-3, h_bot=50.0)
        np.testing.assert_allclose(du2, 4.0 * du1, rtol=1e-12)
        np.testing.assert_allclose(dv2, 4.0 * dv1, rtol=1e-12)

    def test_field_coefficients(self, grid):
        cd = _ones(grid, 1.0).at[:, 4:].set(3.0)
        h_bot = _ones(grid, 2.0)
        du, _ = QuadraticDrag2D(grid)(_ones(grid), _ones(grid, 0.0), cd, h_bot)
        np.testing.assert_allclose(du[1:-1, 3], -2.0 / 2.0)

    def test_grad_finite_at_rest(self, grid):
        drag = QuadraticDrag2D(grid)

        def loss(u):
            du, dv = drag(u, jnp.zeros_like(u), cd=1e-3, h_bot=10.0)
            return jnp.sum(du) + jnp.sum(dv)

        g = jax.grad(loss)(jnp.zeros((grid.Ny, grid.Nx)))
        assert bool(jnp.isfinite(g).all())
        np.testing.assert_allclose(g, 0.0)


class TestRayleighDamping2D:
    def test_pure_damping(self, grid):
        q = _ones(grid, 4.0)
        dq = RayleighDamping2D(grid)(q, r=0.25)
        np.testing.assert_allclose(dq[1:-1, 1:-1], -1.0)
        _assert_ghost_ring_zero(dq)

    def test_reference_state(self):
        grid = make_grid_2d()
        h = make_h_field_2d()
        dq = RayleighDamping2D(grid)(h, r=0.5, q_ref=h)
        np.testing.assert_allclose(dq, 0.0)

    def test_field_rate_and_reference(self, grid):
        r = jnp.arange(grid.Ny * grid.Nx, dtype=float).reshape(grid.Ny, grid.Nx)
        dq = RayleighDamping2D(grid)(_ones(grid, 2.0), r, q_ref=1.0)
        np.testing.assert_allclose(dq[1:-1, 1:-1], -r[1:-1, 1:-1])


class TestMasks:
    def test_dry_faces_zeroed(self):
        grid, mask = make_grid_2d(), make_mask_2d()
        u, v, h = make_u_field_2d(), make_v_field_2d(), make_h_field_2d()
        tx = jnp.ones_like(h)
        outputs = [
            (WindStress2D(grid, mask=mask)(tx, tx, 10.0), (mask.u, mask.v)),
            (LinearDrag2D(grid, mask=mask)(u, v, 0.1), (mask.u, mask.v)),
            (QuadraticDrag2D(grid, mask=mask)(u, v, 1e-3, 1.0), (mask.u, mask.v)),
        ]
        for (du, dv), (mu, mv) in outputs:
            np.testing.assert_array_equal(np.asarray(du)[~np.asarray(mu)], 0.0)
            np.testing.assert_array_equal(np.asarray(dv)[~np.asarray(mv)], 0.0)
        dq = RayleighDamping2D(grid, mask=mask)(h, 0.1)
        np.testing.assert_array_equal(np.asarray(dq)[~np.asarray(mask.h)], 0.0)

    def test_all_ocean_mask_matches_unmasked(self):
        grid, mask = make_grid_2d(), make_mask_2d_all_ocean()
        u, v, h = make_u_field_2d(), make_v_field_2d(), make_h_field_2d()
        for op_cls, args in [
            (WindStress2D, (h, h, 10.0)),
            (LinearDrag2D, (u, v, 0.1)),
            (QuadraticDrag2D, (u, v, 1e-3, 1.0)),
            (RayleighDamping2D, (h, 0.1)),
        ]:
            got = op_cls(grid, mask=mask)(*args)
            want = op_cls(grid)(*args)
            for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
                np.testing.assert_allclose(g, w)


class TestComposition:
    def test_forcing_sum(self, grid):
        """Operators with a shared signature compose with ForcingSum."""
        u, v = _ones(grid, 2.0), _ones(grid, -1.0)
        a, b = LinearDrag2D(grid), LinearDrag2D(grid)
        du_s, dv_s = ForcingSum(a, b)(u, v, 0.1)
        du, dv = a(u, v, 0.1)
        np.testing.assert_allclose(du_s, 2.0 * du)
        np.testing.assert_allclose(dv_s, 2.0 * dv)

    def test_filter_jit_and_grad_wrt_coefficient(self, grid):
        u, v = _ones(grid), _ones(grid)
        drag = LinearDrag2D(grid)

        @eqx.filter_jit
        def loss(r):
            du, dv = drag(u, v, r)
            return jnp.sum(du) + jnp.sum(dv)

        n_interior = (grid.Ny - 2) * (grid.Nx - 2)
        np.testing.assert_allclose(jax.grad(loss)(0.3), -2.0 * n_interior)


class TestMaskNaNSafety:
    """Dry cells with zero thickness or NaN inputs must give exact zeros.

    ``jnp`` multiplication by a *boolean* mask already lowers to a select on
    the forward pass, so the forward tests pin the contract; the gradient
    test is the regression — a 0/0 at dry faces used to make ``jax.grad``
    return NaN even though the masked output was zero.
    """

    @staticmethod
    def _land_fields():
        grid, mask = make_grid_2d(), make_mask_2d()
        wet = np.asarray(mask.h)
        # Thickness is zero over land (typical layer-thickness field).
        dz = jnp.where(wet, 50.0, 0.0)
        return grid, mask, wet, dz

    def test_zero_thickness_over_land_is_finite(self):
        grid, mask, _, dz = self._land_fields()
        u, v = make_u_field_2d(), make_v_field_2d()
        tau = jnp.ones((grid.Ny, grid.Nx))
        for du, dv in [
            WindStress2D(grid, mask=mask)(tau, tau, dz_top=dz),
            QuadraticDrag2D(grid, mask=mask)(u, v, cd=1e-3, h_bot=dz),
        ]:
            assert bool(jnp.isfinite(du).all()) and bool(jnp.isfinite(dv).all())
            np.testing.assert_array_equal(np.asarray(du)[~np.asarray(mask.u)], 0.0)
            np.testing.assert_array_equal(np.asarray(dv)[~np.asarray(mask.v)], 0.0)

    def test_wet_faces_unchanged_by_sanitising(self):
        grid, mask, _, dz = self._land_fields()
        tau = jnp.ones((grid.Ny, grid.Nx))
        du, _ = WindStress2D(grid, mask=mask)(tau, tau, dz_top=dz)
        du_ref, _ = WindStress2D(grid)(tau, tau, dz_top=jnp.where(dz > 0, dz, 1.0))
        mu = np.asarray(mask.u)
        np.testing.assert_allclose(np.asarray(du)[mu], np.asarray(du_ref)[mu])

    def test_nan_over_land_is_zeroed(self):
        grid, mask, wet, _ = self._land_fields()
        nan_land = jnp.where(wet, 1.0, jnp.nan)
        u = jnp.where(np.asarray(mask.u), make_u_field_2d(), jnp.nan)
        v = jnp.where(np.asarray(mask.v), make_v_field_2d(), jnp.nan)
        dq = RayleighDamping2D(grid, mask=mask)(nan_land, r=0.1)
        np.testing.assert_array_equal(np.asarray(dq)[~wet], 0.0)
        for du, dv in [
            LinearDrag2D(grid, mask=mask)(u, v, r=0.1),
            QuadraticDrag2D(grid, mask=mask)(u, v, cd=1e-3, h_bot=10.0),
        ]:
            np.testing.assert_array_equal(np.asarray(du)[~np.asarray(mask.u)], 0.0)
            np.testing.assert_array_equal(np.asarray(dv)[~np.asarray(mask.v)], 0.0)

    def test_grad_finite_with_zero_land_thickness(self):
        grid, mask, _, dz = self._land_fields()
        wind = WindStress2D(grid, mask=mask)
        tau = jnp.ones((grid.Ny, grid.Nx))

        def loss(dz):
            du, dv = wind(tau, tau, dz_top=dz)
            return jnp.sum(du) + jnp.sum(dv)

        assert bool(jnp.isfinite(jax.grad(loss)(dz)).all())
