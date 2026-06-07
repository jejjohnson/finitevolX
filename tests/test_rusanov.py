"""Tests for the Rusanov / local Lax--Friedrichs flux (rusanov_flux)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx import rusanov_flux

jax.config.update("jax_enable_x64", True)


class TestRusanovFlux:
    def test_shape_drops_one_face(self):
        q = jnp.ones((6, 7))
        a = jnp.ones((6, 6))
        assert rusanov_flux(q, a, axis=-1).shape == (6, 6)
        a2 = jnp.ones((5, 7))
        assert rusanov_flux(q, a2, axis=-2).shape == (5, 7)

    def test_constant_scalar_pure_advection(self):
        # For constant q == c the dissipation term vanishes and F = c * a.
        c = 3.0
        q = jnp.full((5, 5), c)
        a = jnp.linspace(-2.0, 2.0, 4)[None, :] * jnp.ones((5, 1))
        np.testing.assert_allclose(rusanov_flux(q, a, axis=-1), c * a, atol=1e-12)

    def test_upwind_limit_matches_first_order(self):
        # F = 0.5 a (qL+qR) - 0.5 |a| (qR-qL) reduces to a*qL (a>0) / a*qR (a<0).
        q = jnp.array([[1.0, 2.0, 5.0, 11.0]])
        qL, qR = q[:, :-1], q[:, 1:]
        a_pos = jnp.full((1, 3), 2.0)
        np.testing.assert_allclose(
            rusanov_flux(q, a_pos, axis=-1, eps=0.0), a_pos * qL, atol=1e-12
        )
        a_neg = jnp.full((1, 3), -2.0)
        np.testing.assert_allclose(
            rusanov_flux(q, a_neg, axis=-1, eps=0.0), a_neg * qR, atol=1e-12
        )

    def test_dissipation_sign(self):
        # The flux sits below the pure centred flux when qR > qL and a > 0.
        q = jnp.array([[1.0, 4.0]])
        a = jnp.array([[1.0]])
        centred = 0.5 * a * (q[:, :-1] + q[:, 1:])
        assert float(rusanov_flux(q, a, axis=-1)[0, 0]) < float(centred[0, 0])

    def test_eps_smoothing_is_small(self):
        q = jnp.array([[1.0, 4.0]])
        a = jnp.array([[1.5]])
        hard = rusanov_flux(q, a, axis=-1, eps=0.0)
        soft = rusanov_flux(q, a, axis=-1, eps=1e-8)
        np.testing.assert_allclose(soft, hard, atol=1e-7)

    def test_differentiable_at_zero_velocity(self):
        # The point of the smooth abs: grad wrt the velocity at a = 0 is finite.
        q = jnp.array([[1.0, 4.0]])
        g = jax.grad(lambda s: rusanov_flux(q, jnp.array([[s]]), axis=-1).sum())(0.0)
        assert jnp.isfinite(g)

    def test_jit_and_grad_compatible(self):
        q = jax.random.normal(jax.random.PRNGKey(0), (8, 8))
        a = jax.random.normal(jax.random.PRNGKey(1), (8, 7))
        flux = jax.jit(lambda q, a: rusanov_flux(q, a, axis=-1))(q, a)
        assert flux.shape == (8, 7)
        g = jax.grad(lambda q: rusanov_flux(q, a, axis=-1).sum())(q)
        assert g.shape == q.shape and bool(jnp.isfinite(g).all())
