"""Tests for the §2 vorticity-flux additions.

Covers the public ``pv_flux_arakawa_lamb`` free function (parity with the
``Vorticity2D`` method) and the dissipative ``vorticity_flux_upwind`` operator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    CartesianGrid2D,
    Vorticity2D,
    pv_flux_arakawa_lamb,
    vorticity_flux_upwind,
)
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators.stencils import avg_xfwd_ybwd, avg_y_bwd

jax.config.update("jax_enable_x64", True)


@pytest.fixture()
def grid():
    return CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)


def _rand(seed, shape):
    return jax.random.normal(jax.random.PRNGKey(seed), shape)


class TestPVFluxArakawaLambFreeFn:
    def test_parity_with_method(self, grid):
        q = _rand(0, (grid.Ny, grid.Nx))
        u = _rand(1, (grid.Ny, grid.Nx))
        v = _rand(2, (grid.Ny, grid.Nx))
        vort = Vorticity2D(grid=grid)
        qu_m, qv_m = vort.pv_flux_arakawa_lamb(q, u, v)
        qu_f, qv_f = pv_flux_arakawa_lamb(q, u, v)
        np.testing.assert_allclose(qu_f, qu_m, atol=1e-12)
        np.testing.assert_allclose(qv_f, qv_m, atol=1e-12)

    def test_alpha_endpoints_match_component_schemes(self, grid):
        q, u, v = (_rand(s, (grid.Ny, grid.Nx)) for s in (3, 4, 5))
        vort = Vorticity2D(grid=grid)
        qu_e, qv_e = vort.pv_flux_energy_conserving(q, u, v)
        qu_s, qv_s = vort.pv_flux_enstrophy_conserving(q, u, v)
        qu1, qv1 = pv_flux_arakawa_lamb(q, u, v, alpha=1.0)
        np.testing.assert_allclose(qu1, qu_e, atol=1e-12)
        np.testing.assert_allclose(qv1, qv_e, atol=1e-12)
        qu0, qv0 = pv_flux_arakawa_lamb(q, u, v, alpha=0.0)
        np.testing.assert_allclose(qu0, qu_s, atol=1e-12)
        np.testing.assert_allclose(qv0, qv_s, atol=1e-12)

    def test_ghost_ring_zero(self, grid):
        q, u, v = (_rand(s, (grid.Ny, grid.Nx)) for s in (6, 7, 8))
        qu, _ = pv_flux_arakawa_lamb(q, u, v)
        assert bool((qu[0] == 0).all()) and bool((qu[-1] == 0).all())
        assert bool((qu[:, 0] == 0).all()) and bool((qu[:, -1] == 0).all())


class TestVorticityFluxUpwind:
    @pytest.mark.parametrize("order", [1, 3, 5])
    def test_shapes_and_ghost_ring(self, grid, order):
        omega = _rand(0, (grid.Ny, grid.Nx))
        U, V = _rand(1, (grid.Ny, grid.Nx)), _rand(2, (grid.Ny, grid.Nx))
        oV, oU = vorticity_flux_upwind(omega, U, V, order=order)
        assert oV.shape == omega.shape and oU.shape == omega.shape
        assert bool((oV[0] == 0).all()) and bool((oV[:, -1] == 0).all())
        assert bool((oU[-1] == 0).all()) and bool((oU[:, 0] == 0).all())

    @pytest.mark.parametrize("order", [1, 3, 5])
    def test_constant_vorticity_is_exact(self, grid, order):
        # For omega == c the reconstruction is exact: omega_V = c * (V->U flux).
        c = 2.5
        omega = jnp.full((grid.Ny, grid.Nx), c)
        U, V = _rand(3, (grid.Ny, grid.Nx)), _rand(4, (grid.Ny, grid.Nx))
        v_on_u = interior(avg_xfwd_ybwd(V), V)
        oV, _ = vorticity_flux_upwind(omega, U, V, order=order)
        expected = interior(c * v_on_u[1:-1, 1:-1], omega)
        np.testing.assert_allclose(oV, expected, atol=1e-12)

    def test_linear_field_third_order_exact_interior(self, grid):
        # A linear vorticity field is reconstructed exactly by the 3rd-order
        # upwind stencil in the deep interior (away from the order-1 fallback).
        j = jnp.arange(grid.Ny)[:, None].astype(float)
        omega = jnp.broadcast_to(0.3 * j + 0.7, (grid.Ny, grid.Nx))  # linear in y
        U = jnp.zeros((grid.Ny, grid.Nx))
        V = jnp.ones((grid.Ny, grid.Nx))  # uniform positive advection
        v_on_u = interior(avg_xfwd_ybwd(V), V)
        centred = interior(avg_y_bwd(omega), omega)  # exact midpoint for linear
        expected = interior(centred[1:-1, 1:-1] * v_on_u[1:-1, 1:-1], omega)
        oV, _ = vorticity_flux_upwind(omega, U, V, order=3)
        np.testing.assert_allclose(oV[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-10)

    def test_upwind_sign_sensitivity(self, grid):
        # Reversing the advecting flux changes the upwind bias (and the flux).
        omega = _rand(5, (grid.Ny, grid.Nx))
        U, V = _rand(6, (grid.Ny, grid.Nx)), _rand(7, (grid.Ny, grid.Nx))
        oV_p, _ = vorticity_flux_upwind(omega, U, V, order=3)
        oV_m, _ = vorticity_flux_upwind(omega, U, -V, order=3)
        # not merely sign-flipped: the stencil itself differs
        assert not np.allclose(oV_p, -oV_m, atol=1e-8)

    def test_invalid_order_raises(self, grid):
        omega = jnp.ones((grid.Ny, grid.Nx))
        with pytest.raises(ValueError, match="order must be"):
            vorticity_flux_upwind(omega, omega, omega, order=2)

    def test_differentiable(self, grid):
        omega = _rand(8, (grid.Ny, grid.Nx))
        U, V = _rand(9, (grid.Ny, grid.Nx)), _rand(10, (grid.Ny, grid.Nx))

        def loss(omega):
            oV, oU = vorticity_flux_upwind(omega, U, V, order=3)
            return (oV**2 + oU**2).sum()

        g = jax.grad(loss)(omega)
        assert g.shape == omega.shape and bool(jnp.isfinite(g).all())


# ---------------------------------------------------------------------------
# Which form each PV-flux method computes, and the ghost-corner row (#283, #284)
# ---------------------------------------------------------------------------


def _periodic(a):
    """Fill the ghost ring periodically (south/north, then west/east)."""
    a = a.at[0, :].set(a[-2, :]).at[-1, :].set(a[1, :])
    return a.at[:, 0].set(a[:, -2]).at[:, -1].set(a[:, 1])


class TestPVFluxForms:
    @pytest.fixture()
    def fields(self):
        n = 16
        g = CartesianGrid2D.from_interior(n - 2, n - 2, 1.0, 1.0)
        key = jax.random.key(0)
        q, u, v = (
            _periodic(jax.random.normal(jax.random.fold_in(key, k), (n, n)))
            for k in range(3)
        )
        return g, q, u, v

    def test_energy_is_corner_product_then_average(self, fields):
        """Periodic reference: fill the corner product's ghosts, then average.

        Before #284 the first interior row / column read a zero ghost corner
        and came out exactly halved.
        """
        from finitevolx import Interpolation2D

        g, q, u, v = fields
        interp = Interpolation2D(g)
        ref_qu = interp.X_to_U(_periodic(q * interp.U_to_X(u)))
        ref_qv = interp.X_to_V(_periodic(q * interp.V_to_X(v)))
        qu, qv = Vorticity2D(g).pv_flux_energy_conserving(q, u, v)
        np.testing.assert_allclose(qu, ref_qu, rtol=0, atol=1e-14)
        np.testing.assert_allclose(qv, ref_qv, rtol=0, atol=1e-14)

    def test_enstrophy_is_average_then_multiply(self, fields):
        from finitevolx import Interpolation2D

        g, q, u, v = fields
        interp = Interpolation2D(g)
        ref_qu = interior(interp.X_to_U(q)[1:-1, 1:-1] * u[1:-1, 1:-1], u)
        ref_qv = interior(interp.X_to_V(q)[1:-1, 1:-1] * v[1:-1, 1:-1], v)
        qu, qv = Vorticity2D(g).pv_flux_enstrophy_conserving(q, u, v)
        np.testing.assert_array_equal(qu, ref_qu)
        np.testing.assert_array_equal(qv, ref_qv)

    def test_closed_basin_unchanged(self, fields):
        """With a zero ghost ring on q (closed basin), the corner products on
        the south / west ghost corners are zero, which is what the old
        interior()-intermediate assumed: the result equals that composition.
        """
        from finitevolx import Interpolation2D

        g, q, u, v = fields
        q0 = interior(q[1:-1, 1:-1], q)
        interp = Interpolation2D(g)
        u_on_q, v_on_q = interp.U_to_X(u), interp.V_to_X(v)
        ref_qu = interp.X_to_U(interior(q0[1:-1, 1:-1] * u_on_q[1:-1, 1:-1], q0))
        ref_qv = interp.X_to_V(interior(q0[1:-1, 1:-1] * v_on_q[1:-1, 1:-1], q0))
        qu, qv = Vorticity2D(g).pv_flux_energy_conserving(q0, u, v)
        np.testing.assert_allclose(qu, ref_qu, rtol=0, atol=1e-14)
        np.testing.assert_allclose(qv, ref_qv, rtol=0, atol=1e-14)

    def test_free_function_matches_method(self, fields):
        g, q, u, v = fields
        qu, qv = pv_flux_arakawa_lamb(q, u, v, alpha=0.25)
        ref = Vorticity2D(g).pv_flux_arakawa_lamb(q, u, v, alpha=0.25)
        np.testing.assert_allclose(qu, ref[0], rtol=0, atol=1e-14)
        np.testing.assert_allclose(qv, ref[1], rtol=0, atol=1e-14)
