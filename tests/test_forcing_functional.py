"""Tests for the Layer 0 functional forcing primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx import (
    interior,
    linear_drag_tendency,
    quadratic_drag_tendency,
    rayleigh_tendency,
    wind_stress_tendency,
)

jax.config.update("jax_enable_x64", True)


class TestWindStressTendency:
    def test_scalar_normalization(self):
        tau = 0.1 * jnp.ones((4, 5))
        du = wind_stress_tendency(tau, rho0=1025.0, dz_top=50.0)
        assert du.shape == tau.shape
        np.testing.assert_allclose(du, 0.1 / (1025.0 * 50.0), rtol=1e-12)

    def test_spatially_varying_thickness(self):
        tau = jnp.ones((2, 3))
        dz = jnp.arange(1.0, 7.0).reshape(2, 3)
        du = wind_stress_tendency(tau, rho0=2.0, dz_top=dz)
        np.testing.assert_allclose(du, 1.0 / (2.0 * dz), rtol=1e-12)

    def test_linear_in_stress(self):
        tau = jnp.linspace(-1.0, 1.0, 7)
        du = wind_stress_tendency(tau, rho0=1000.0, dz_top=10.0)
        du2 = wind_stress_tendency(2.0 * tau, rho0=1000.0, dz_top=10.0)
        np.testing.assert_allclose(du2, 2.0 * du, rtol=1e-12)

    def test_power_user_pattern(self):
        """Interior slice -> primitive -> interior() pad back."""
        u = jnp.zeros((6, 7))
        tau_on_u = jnp.ones((6, 7))
        du = interior(wind_stress_tendency(tau_on_u[1:-1, 1:-1], 1.0, 2.0), u)
        assert du.shape == u.shape
        np.testing.assert_allclose(du[1:-1, 1:-1], 0.5)
        np.testing.assert_allclose(du[0], 0.0)
        np.testing.assert_allclose(du[:, -1], 0.0)


class TestLinearDragTendency:
    def test_scalar_coefficient(self):
        vel = jnp.array([[1.0, -2.0], [3.0, 0.0]])
        np.testing.assert_allclose(linear_drag_tendency(vel, r=0.1), -0.1 * vel)

    def test_spatially_varying_coefficient(self):
        vel = jnp.ones((3, 3))
        r = jnp.arange(9.0).reshape(3, 3)
        np.testing.assert_allclose(linear_drag_tendency(vel, r), -r)

    def test_opposes_flow(self):
        vel = jnp.array([2.0, -3.0, 0.5])
        dvel = linear_drag_tendency(vel, r=1e-3)
        assert bool((jnp.sign(dvel) == -jnp.sign(vel)).all())


class TestQuadraticDragTendency:
    def test_formula(self):
        vel = jnp.array([1.0, -2.0])
        speed = jnp.array([3.0, 4.0])
        dvel = quadratic_drag_tendency(vel, speed, cd=1e-3, h_bot=10.0)
        np.testing.assert_allclose(dvel, -1e-3 * speed * vel / 10.0, rtol=1e-12)

    def test_quadratic_scaling(self):
        """For 1-D flow (speed = |vel|) doubling vel quadruples the drag."""
        vel = jnp.array([0.5, -1.5])
        d1 = quadratic_drag_tendency(vel, jnp.abs(vel), cd=2e-3, h_bot=5.0)
        d2 = quadratic_drag_tendency(2 * vel, jnp.abs(2 * vel), cd=2e-3, h_bot=5.0)
        np.testing.assert_allclose(d2, 4.0 * d1, rtol=1e-12)

    def test_opposes_flow(self):
        vel = jnp.array([2.0, -3.0])
        dvel = quadratic_drag_tendency(vel, jnp.abs(vel), cd=1e-3, h_bot=1.0)
        assert bool((jnp.sign(dvel) == -jnp.sign(vel)).all())


class TestRayleighTendency:
    def test_pure_damping(self):
        q = jnp.array([1.0, -2.0, 4.0])
        np.testing.assert_allclose(rayleigh_tendency(q, r=0.25), -0.25 * q)

    def test_relaxes_toward_reference(self):
        q = jnp.zeros((4, 4))
        q_ref = jnp.ones((4, 4))
        dq = rayleigh_tendency(q, r=1e-2, q_ref=q_ref)
        np.testing.assert_allclose(dq, 1e-2)

    def test_zero_at_reference(self):
        q = jnp.linspace(0.0, 1.0, 5)
        np.testing.assert_allclose(rayleigh_tendency(q, r=3.0, q_ref=q), 0.0)

    def test_scalar_reference(self):
        q = jnp.array([2.0])
        np.testing.assert_allclose(rayleigh_tendency(q, r=1.0, q_ref=0.5), -1.5)


class TestJaxTransforms:
    def test_jit(self):
        vel = jnp.ones((5, 5))
        out = jax.jit(linear_drag_tendency)(vel, 0.2)
        np.testing.assert_allclose(out, -0.2)

    def test_grad_wrt_coefficient(self):
        vel = jnp.array([1.0, 2.0, 3.0])

        def loss(r):
            return jnp.sum(linear_drag_tendency(vel, r))

        np.testing.assert_allclose(jax.grad(loss)(0.5), -6.0)

    def test_grad_wrt_drag_coefficient_quadratic(self):
        vel = jnp.array([2.0])

        def loss(cd):
            return jnp.sum(quadratic_drag_tendency(vel, jnp.abs(vel), cd, 4.0))

        np.testing.assert_allclose(jax.grad(loss)(1.0), -1.0)

    def test_vmap_over_layers(self):
        q = jnp.ones((3, 4, 4))
        r = jnp.array([1.0, 2.0, 3.0])
        dq = jax.vmap(rayleigh_tendency)(q, r)
        np.testing.assert_allclose(dq[:, 0, 0], -r)
