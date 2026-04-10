"""Tests for the public face-flux API (uv_center_flux, uv_node_flux).

Covers:
- Consistency with Advection2D divergence
- Conservation properties
- Method dispatch (all supported methods)
- Masked-domain face fluxes
- JAX transform compatibility
- Scientific accuracy (smooth field convergence, sign sensitivity)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    Advection2D,
    CartesianGrid2D,
    Mask2D,
    uv_center_flux,
    uv_node_flux,
)
from finitevolx._src.operators._ghost import interior

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def periodic_grid():
    """12x12 periodic-style grid with 2-cell ghost ring."""
    Ny, Nx = 12, 12
    Lx, Ly = 1.0, 1.0
    grid = CartesianGrid2D.from_interior(Nx - 2, Ny - 2, Lx, Ly)
    return grid, Ny, Nx


@pytest.fixture()
def large_grid():
    """20x20 grid for convergence tests."""
    Ny, Nx = 20, 20
    Lx, Ly = 1.0, 1.0
    grid = CartesianGrid2D.from_interior(Nx - 2, Ny - 2, Lx, Ly)
    return grid, Ny, Nx


def _smooth_field(Ny, Nx):
    """Smooth 2D sine field."""
    j = jnp.arange(Ny)[:, None]
    i = jnp.arange(Nx)[None, :]
    return jnp.sin(2.0 * jnp.pi * i / Nx) * jnp.cos(2.0 * jnp.pi * j / Ny)


# ===========================================================================
# Basic correctness
# ===========================================================================


class TestUvCenterFluxBasic:
    """Basic correctness tests for uv_center_flux."""

    def test_output_shape(self, periodic_grid):
        grid, Ny, Nx = periodic_grid
        h = jnp.ones((Ny, Nx))
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        fe, fn = uv_center_flux(h, u, v, grid)
        assert fe.shape == (Ny, Nx)
        assert fn.shape == (Ny, Nx)

    def test_constant_field(self, periodic_grid):
        """Constant scalar with uniform velocity: face flux = C * u."""
        grid, Ny, Nx = periodic_grid
        C = 3.0
        h = jnp.full((Ny, Nx), C)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        fe, fn = uv_center_flux(h, u, v, grid, method="upwind1")
        # Interior face fluxes should be C * 1.0 = 3.0
        np.testing.assert_allclose(fe[1:-1, 1:-1], C, atol=1e-12)
        np.testing.assert_allclose(fn[1:-1, 1:-1], C, atol=1e-12)

    def test_zero_velocity(self, periodic_grid):
        """Zero velocity gives zero flux."""
        grid, Ny, Nx = periodic_grid
        h = _smooth_field(Ny, Nx)
        u = jnp.zeros((Ny, Nx))
        v = jnp.zeros((Ny, Nx))
        fe, fn = uv_center_flux(h, u, v, grid, method="weno5")
        np.testing.assert_allclose(fe, 0.0, atol=1e-15)
        np.testing.assert_allclose(fn, 0.0, atol=1e-15)

    def test_ghost_ring_zero(self, periodic_grid):
        """Ghost ring of face flux must be zero."""
        grid, Ny, Nx = periodic_grid
        h = _smooth_field(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        fe, _fn = uv_center_flux(h, u, v, grid, method="weno3")
        # Ghost cells
        assert jnp.all(fe[0, :] == 0.0)
        assert jnp.all(fe[-1, :] == 0.0)
        assert jnp.all(fe[:, 0] == 0.0)
        assert jnp.all(fe[:, -1] == 0.0)


class TestUvNodeFluxBasic:
    """Basic correctness tests for uv_node_flux."""

    def test_output_shape(self, periodic_grid):
        grid, Ny, Nx = periodic_grid
        q = jnp.ones((Ny, Nx))
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        uq, vq = uv_node_flux(q, u, v, grid)
        assert uq.shape == (Ny, Nx)
        assert vq.shape == (Ny, Nx)

    def test_constant_tracer(self, periodic_grid):
        """Constant tracer: node flux = C * velocity."""
        grid, Ny, Nx = periodic_grid
        C = 5.0
        q = jnp.full((Ny, Nx), C)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        uq, vq = uv_node_flux(q, u, v, grid, method="upwind1")
        np.testing.assert_allclose(uq[1:-1, 1:-1], C, atol=1e-12)
        np.testing.assert_allclose(vq[1:-1, 1:-1], C, atol=1e-12)

    def test_nonconstant_positive_velocity(self, periodic_grid):
        """Non-constant q with positive velocity: upwind1 picks left/below value."""
        grid, Ny, Nx = periodic_grid
        # q increasing in x: q[j, i] = i
        i_vals = jnp.arange(Nx, dtype=float)[None, :]
        q = jnp.broadcast_to(i_vals, (Ny, Nx))
        u = jnp.ones((Ny, Nx))  # positive x-velocity
        v = jnp.zeros((Ny, Nx))
        uq, _vq = uv_node_flux(q, u, v, grid, method="upwind1")
        # With positive u, upwind1 picks q[j, i] for east face → flux = q[j, i] * u
        # Interior east-face flux at [j, i] should equal i (the left cell value)
        for j in range(2, Ny - 2):
            for i in range(2, Nx - 2):
                assert float(uq[j, i]) == pytest.approx(float(q[j, i]), abs=1e-12)

    def test_nonconstant_negative_velocity(self, periodic_grid):
        """Non-constant q with negative velocity: upwind1 picks right/above value."""
        grid, Ny, Nx = periodic_grid
        # q increasing in y: q[j, i] = j
        j_vals = jnp.arange(Ny, dtype=float)[:, None]
        q = jnp.broadcast_to(j_vals, (Ny, Nx))
        u = jnp.zeros((Ny, Nx))
        v = -jnp.ones((Ny, Nx))  # negative y-velocity
        _uq, vq = uv_node_flux(q, u, v, grid, method="upwind1")
        # With negative v, upwind1 picks q[j+1, i] for north face → flux = q[j+1, i] * v
        for j in range(2, Ny - 2):
            for i in range(2, Nx - 2):
                assert float(vq[j, i]) == pytest.approx(
                    float(q[j + 1, i]) * (-1.0), abs=1e-12
                )


# ===========================================================================
# Consistency with Advection2D
# ===========================================================================


class TestConsistencyWithAdvection2D:
    """Face fluxes must be consistent with Advection2D divergence."""

    @pytest.mark.parametrize(
        "method",
        [
            "upwind1",
            "upwind2",
            "upwind3",
            "weno3",
            "weno5",
            "wenoz5",
            "weno7",
            "weno9",
            "minmod",
        ],
    )
    def test_divergence_matches_advection(self, periodic_grid, method):
        """fe/fn from uv_center_flux should reproduce Advection2D tendency."""
        grid, Ny, Nx = periodic_grid
        h = _smooth_field(Ny, Nx)
        u = 0.5 * jnp.ones((Ny, Nx))
        v = 0.3 * jnp.ones((Ny, Nx))

        # Advection2D tendency
        adv = Advection2D(grid)
        tendency = adv(h, u, v, method=method)

        # Reconstruct from face fluxes
        fe, fn = uv_center_flux(h, u, v, grid, method=method)
        dx, dy = grid.dx, grid.dy
        tendency_from_flux = interior(
            -(
                (fe[2:-2, 2:-2] - fe[2:-2, 1:-3]) / dx
                + (fn[2:-2, 2:-2] - fn[1:-3, 2:-2]) / dy
            ),
            h,
            ghost=2,
        )

        np.testing.assert_allclose(
            tendency_from_flux[2:-2, 2:-2],
            tendency[2:-2, 2:-2],
            atol=1e-12,
            err_msg=f"method={method}",
        )


# ===========================================================================
# All methods dispatch
# ===========================================================================


class TestAllMethods:
    """All reconstruction methods should work without error."""

    @pytest.mark.parametrize(
        "method",
        [
            "naive",
            "upwind1",
            "upwind2",
            "upwind3",
            "weno3",
            "weno5",
            "wenoz5",
            "weno7",
            "weno9",
            "minmod",
            "van_leer",
            "superbee",
            "mc",
        ],
    )
    def test_method_runs(self, periodic_grid, method):
        grid, Ny, Nx = periodic_grid
        h = _smooth_field(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        fe, fn = uv_center_flux(h, u, v, grid, method=method)
        assert jnp.all(jnp.isfinite(fe))
        assert jnp.all(jnp.isfinite(fn))

    def test_invalid_method(self, periodic_grid):
        grid, Ny, Nx = periodic_grid
        h = jnp.ones((Ny, Nx))
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        with pytest.raises(ValueError, match="Unknown method"):
            uv_center_flux(h, u, v, grid, method="bogus")


# ===========================================================================
# Masked domain
# ===========================================================================


class TestMaskedFlux:
    """Face fluxes with mask-based stencil fallback."""

    @pytest.fixture()
    def masked_setup(self):
        Ny, Nx = 16, 16
        grid = CartesianGrid2D.from_interior(Nx - 2, Ny - 2, 1.0, 1.0)
        h_mask = np.ones((Ny, Nx), dtype=bool)
        h_mask[6:10, 6:10] = False  # island
        mask = Mask2D.from_mask(h_mask)
        return grid, mask, Ny, Nx

    @pytest.mark.parametrize("method", ["weno3", "weno5", "wenoz5", "minmod"])
    def test_masked_runs(self, masked_setup, method):
        grid, mask, Ny, Nx = masked_setup
        h = _smooth_field(Ny, Nx) * jnp.asarray(mask.h, dtype=float)
        u = jnp.ones((Ny, Nx)) * jnp.asarray(mask.u, dtype=float)
        v = jnp.ones((Ny, Nx)) * jnp.asarray(mask.v, dtype=float)
        fe, fn = uv_center_flux(h, u, v, grid, method=method, mask=mask)
        assert jnp.all(jnp.isfinite(fe))
        assert jnp.all(jnp.isfinite(fn))

    def test_masked_matches_advection(self, masked_setup):
        """Masked face fluxes reproduce Advection2D masked tendency."""
        grid, mask, Ny, Nx = masked_setup
        h = _smooth_field(Ny, Nx) * jnp.asarray(mask.h, dtype=float)
        u = 0.5 * jnp.ones((Ny, Nx)) * jnp.asarray(mask.u, dtype=float)
        v = 0.3 * jnp.ones((Ny, Nx)) * jnp.asarray(mask.v, dtype=float)

        adv = Advection2D(grid)
        tendency = adv(h, u, v, method="weno5", mask=mask)

        fe, fn = uv_center_flux(h, u, v, grid, method="weno5", mask=mask)
        dx, dy = grid.dx, grid.dy
        tendency_from_flux = interior(
            -(
                (fe[2:-2, 2:-2] - fe[2:-2, 1:-3]) / dx
                + (fn[2:-2, 2:-2] - fn[1:-3, 2:-2]) / dy
            ),
            h,
            ghost=2,
        )

        np.testing.assert_allclose(
            tendency_from_flux[2:-2, 2:-2],
            tendency[2:-2, 2:-2],
            atol=1e-12,
        )


# ===========================================================================
# Conservation
# ===========================================================================


class TestConservation:
    """Face fluxes should satisfy discrete conservation properties."""

    def test_uniform_divergence_free_flow(self, periodic_grid):
        """Uniform flow with constant scalar: net flux through any closed
        contour of interior faces should be zero (divergence-free)."""
        grid, Ny, Nx = periodic_grid
        C = 2.0
        h = jnp.full((Ny, Nx), C)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))
        fe, fn = uv_center_flux(h, u, v, grid, method="weno5")

        # Divergence at each interior cell should be zero
        dx, dy = grid.dx, grid.dy
        div = (fe[2:-2, 2:-2] - fe[2:-2, 1:-3]) / dx + (
            fn[2:-2, 2:-2] - fn[1:-3, 2:-2]
        ) / dy
        np.testing.assert_allclose(div, 0.0, atol=1e-12)

    def test_flux_antisymmetry(self, periodic_grid):
        """Reversing velocity should negate the face flux (for constant scalar)."""
        grid, Ny, Nx = periodic_grid
        C = 1.0
        h = jnp.full((Ny, Nx), C)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))

        fe_pos, fn_pos = uv_center_flux(h, u, v, grid, method="upwind1")
        fe_neg, fn_neg = uv_center_flux(h, -u, -v, grid, method="upwind1")

        np.testing.assert_allclose(fe_pos, -fe_neg, atol=1e-12)
        np.testing.assert_allclose(fn_pos, -fn_neg, atol=1e-12)


# ===========================================================================
# Scientific accuracy
# ===========================================================================


class TestAccuracy:
    """Scientific accuracy of face fluxes."""

    def test_higher_order_more_accurate_smooth_field(self, large_grid):
        """On smooth data, higher-order methods should give more accurate fluxes."""
        grid, Ny, Nx = large_grid
        h = _smooth_field(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        v = jnp.zeros((Ny, Nx))

        # Use weno9 as reference (highest-order)
        fe_ref, _ = uv_center_flux(h, u, v, grid, method="weno9")

        fe_1, _ = uv_center_flux(h, u, v, grid, method="upwind1")
        fe_3, _ = uv_center_flux(h, u, v, grid, method="weno3")
        fe_5, _ = uv_center_flux(h, u, v, grid, method="weno5")

        err_1 = float(jnp.linalg.norm(fe_1[2:-2, 2:-2] - fe_ref[2:-2, 2:-2]))
        err_3 = float(jnp.linalg.norm(fe_3[2:-2, 2:-2] - fe_ref[2:-2, 2:-2]))
        err_5 = float(jnp.linalg.norm(fe_5[2:-2, 2:-2] - fe_ref[2:-2, 2:-2]))

        assert err_5 < err_3
        assert err_3 < err_1

    def test_sign_sensitivity(self, large_grid):
        """Positive/negative velocity should bias the face value correctly."""
        grid, Ny, Nx = large_grid
        # Monotonic field: increasing in x
        i = jnp.arange(Nx, dtype=float)[None, :]
        h = jnp.broadcast_to(i, (Ny, Nx))

        u_pos = jnp.ones((Ny, Nx))
        u_neg = -jnp.ones((Ny, Nx))

        fe_pos, _ = uv_center_flux(
            h, u_pos, jnp.zeros((Ny, Nx)), grid, method="upwind1"
        )
        fe_neg, _ = uv_center_flux(
            h, u_neg, jnp.zeros((Ny, Nx)), grid, method="upwind1"
        )

        # Positive flow biased left (lower values), negative biased right (higher)
        # fe_pos should be positive, fe_neg should be negative
        assert jnp.all(fe_pos[2:-2, 2:-2] >= 0)
        assert jnp.all(fe_neg[2:-2, 2:-2] <= 0)


# ===========================================================================
# JAX compatibility
# ===========================================================================


class TestJaxCompat:
    """JAX transform compatibility."""

    def test_jit(self, periodic_grid):
        grid, Ny, Nx = periodic_grid
        h = _smooth_field(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))

        @jax.jit
        def f(h, u, v):
            return uv_center_flux(h, u, v, grid, method="weno3")

        fe, fn = f(h, u, v)
        fe_ref, fn_ref = uv_center_flux(h, u, v, grid, method="weno3")
        np.testing.assert_allclose(fe, fe_ref, atol=1e-14)
        np.testing.assert_allclose(fn, fn_ref, atol=1e-14)

    def test_grad(self, periodic_grid):
        """Gradient through uv_center_flux should be computable."""
        grid, Ny, Nx = periodic_grid
        h = _smooth_field(Ny, Nx)
        u = jnp.ones((Ny, Nx))
        v = jnp.ones((Ny, Nx))

        def loss(h):
            fe, fn = uv_center_flux(h, u, v, grid, method="weno3")
            return jnp.sum(fe**2 + fn**2)

        grad_h = jax.grad(loss)(h)
        assert grad_h.shape == h.shape
        assert jnp.all(jnp.isfinite(grad_h))
