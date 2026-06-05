"""Tests for the implicit barotropic-mode filter (barotropic_filter).

The variable-coefficient Helmholtz solve is injected, so most tests pin the
*wiring* (transport sum, RHS scaling, gradient correction) with analytic
stand-in solvers.  One integration test runs the real multigrid solver.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx import barotropic_filter, build_multigrid_solver, divergence_2d
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators.stencils import diff_x_fwd

jax.config.update("jax_enable_x64", True)


class TestBarotropicFilterWiring:
    def test_quiescent_flow_zero_correction(self):
        u = jnp.zeros((2, 8, 8))
        h = jnp.ones((2, 8, 8))
        fu, fv = barotropic_filter(
            u,
            u,
            h,
            h,
            dx=1.0,
            dy=1.0,
            g=9.81,
            tau=1.0,
            dt=1.0,
            helm_solve=lambda rhs: rhs,
        )
        np.testing.assert_allclose(fu, 0.0, atol=1e-12)
        np.testing.assert_allclose(fv, 0.0, atol=1e-12)

    def test_constant_surface_mode_gives_no_force(self):
        # A spatially constant w has zero gradient -> zero correction.
        u = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8))
        h = jnp.ones((2, 8, 8))
        fu, fv = barotropic_filter(
            u,
            u,
            h,
            h,
            dx=1.0,
            dy=1.0,
            g=2.0,
            tau=1.0,
            dt=1.0,
            helm_solve=lambda rhs: jnp.full_like(rhs, 3.0),
        )
        np.testing.assert_allclose(fu, 0.0, atol=1e-12)
        np.testing.assert_allclose(fv, 0.0, atol=1e-12)

    def test_rhs_is_scaled_transport_divergence(self):
        u = jax.random.normal(jax.random.PRNGKey(1), (3, 8, 8))
        v = jax.random.normal(jax.random.PRNGKey(2), (3, 8, 8))
        hu = jnp.abs(jax.random.normal(jax.random.PRNGKey(3), (3, 8, 8))) + 0.5
        hv = jnp.abs(jax.random.normal(jax.random.PRNGKey(4), (3, 8, 8))) + 0.5
        g, tau, dt, dx, dy = 9.81, 0.5, 2.0, 1.5, 1.5
        captured = {}

        def recorder(rhs):
            captured["rhs"] = rhs
            return jnp.zeros_like(rhs)

        barotropic_filter(
            u, v, hu, hv, dx=dx, dy=dy, g=g, tau=tau, dt=dt, helm_solve=recorder
        )
        u_bt = jnp.sum(hu * u, axis=-3)
        v_bt = jnp.sum(hv * v, axis=-3)
        expected = divergence_2d(u_bt, v_bt, dx, dy) / (g * tau * dt)
        np.testing.assert_allclose(captured["rhs"], expected, atol=1e-12)

    def test_correction_is_minus_scaled_gradient(self):
        u = jnp.zeros((2, 8, 8))
        h = jnp.ones((2, 8, 8))
        w = jax.random.normal(jax.random.PRNGKey(5), (8, 8))
        g, tau, dx = 3.0, 0.7, 1.25
        fu, _ = barotropic_filter(
            u,
            u,
            h,
            h,
            dx=dx,
            dy=1.0,
            g=g,
            tau=tau,
            dt=1.0,
            helm_solve=lambda rhs: w,
        )
        expected = interior(-g * tau * diff_x_fwd(w) / dx, w)
        np.testing.assert_allclose(fu, expected, atol=1e-12)


class TestBarotropicFilterIntegration:
    def test_runs_with_real_multigrid_solver(self):
        Ny = Nx = 16
        solver = build_multigrid_solver(
            np.ones((Ny, Nx)), dx=1.0, dy=1.0, coeff=np.ones((Ny, Nx))
        )
        key = jax.random.PRNGKey(7)
        u = jax.random.normal(key, (2, Ny, Nx))
        v = jax.random.normal(jax.random.PRNGKey(8), (2, Ny, Nx))
        h = jnp.ones((2, Ny, Nx))
        fu, fv = barotropic_filter(
            u,
            v,
            h,
            h,
            dx=1.0,
            dy=1.0,
            g=9.81,
            tau=1.0,
            dt=1.0,
            helm_solve=solver,
        )
        assert fu.shape == (Ny, Nx) and fv.shape == (Ny, Nx)
        assert bool(jnp.isfinite(fu).all()) and bool(jnp.isfinite(fv).all())

    def test_differentiable_through_solver(self):
        Ny = Nx = 16
        solver = build_multigrid_solver(
            np.ones((Ny, Nx)), dx=1.0, dy=1.0, coeff=np.ones((Ny, Nx))
        )
        h = jnp.ones((2, Ny, Nx))

        def loss(u):
            fu, fv = barotropic_filter(
                u,
                u,
                h,
                h,
                dx=1.0,
                dy=1.0,
                g=9.81,
                tau=1.0,
                dt=1.0,
                helm_solve=solver,
            )
            return (fu**2 + fv**2).sum()

        u = jax.random.normal(jax.random.PRNGKey(9), (2, Ny, Nx))
        grad = jax.grad(loss)(u)
        assert grad.shape == u.shape and bool(jnp.isfinite(grad).all())
