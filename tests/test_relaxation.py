"""Tests for the linear-drag and Rayleigh-relaxation tendency operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx import linear_drag, rayleigh_relaxation

jax.config.update("jax_enable_x64", True)


class TestLinearDrag:
    def test_single_layer_2d(self):
        u = jnp.ones((8, 8))
        v = 2.0 * jnp.ones((8, 8))
        du, dv = linear_drag(u, v, coef=0.1)
        np.testing.assert_allclose(du, -0.1, atol=1e-12)
        np.testing.assert_allclose(dv, -0.2, atol=1e-12)

    def test_only_selected_layer_drags(self):
        u = jnp.ones((3, 8, 8))
        du, _ = linear_drag(u, u, coef=0.5)
        assert bool((du[0] == 0).all()) and bool((du[1] == 0).all())
        np.testing.assert_allclose(du[-1], -0.5, atol=1e-12)

    def test_layer_index_selectable(self):
        u = jnp.ones((3, 4, 4))
        du, _ = linear_drag(u, u, coef=1.0, layer=0)
        np.testing.assert_allclose(du[0], -1.0, atol=1e-12)
        assert bool((du[1:] == 0).all())

    def test_opposes_velocity(self):
        u = jnp.array([[2.0, -3.0]])
        du, _ = linear_drag(u, u, coef=0.25)
        # drag always points opposite the flow
        assert bool((jnp.sign(du) == -jnp.sign(u)).all())


class TestRayleighRelaxation:
    def test_restores_toward_reference(self):
        x = jnp.zeros((6, 6))
        x_ref = jnp.ones((6, 6))
        dx = rayleigh_relaxation(x, x_ref, coef=1e-2, weight=1.0)
        np.testing.assert_allclose(dx, 1e-2, atol=1e-12)

    def test_zero_at_reference(self):
        x = jnp.full((6, 6), 3.0)
        dx = rayleigh_relaxation(x, x, coef=0.5, weight=1.0)
        np.testing.assert_allclose(dx, 0.0, atol=1e-12)

    def test_weight_localises(self):
        x = jnp.zeros((4, 4))
        x_ref = jnp.ones((4, 4))
        weight = jnp.zeros((4, 4)).at[0, :].set(1.0)
        dx = rayleigh_relaxation(x, x_ref, coef=1.0, weight=weight)
        assert bool((dx[0] != 0).all()) and bool((dx[1:] == 0).all())

    def test_drag_is_relaxation_to_zero(self):
        # linear_drag (single layer) is the x_ref = 0 special case.
        u = jnp.array([[1.0, -2.0, 3.0]])
        du, _ = linear_drag(u, u, coef=0.3)
        dx = rayleigh_relaxation(u, jnp.zeros_like(u), coef=0.3, weight=1.0)
        np.testing.assert_allclose(du, dx, atol=1e-12)
