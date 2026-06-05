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
