"""Tests for the differentiable surrogates (smooth_abs / smooth_clamp / smooth_max).

The whole point of these primitives is a well-defined, nonzero gradient where
``jnp.abs`` / ``jnp.maximum`` / a hard clamp would have a kink.  The tests
pin both the forward values (approach the non-smooth function as the knob
tightens) and the gradient behaviour at the kink.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import smooth_abs, smooth_clamp, smooth_max

jax.config.update("jax_enable_x64", True)


class TestSmoothAbs:
    def test_matches_abs_away_from_origin(self):
        x = jnp.array([-3.0, -1.0, 0.5, 2.0, 5.0])
        np.testing.assert_allclose(smooth_abs(x, eps=1e-8), jnp.abs(x), atol=1e-6)

    def test_rounded_at_origin(self):
        # At x = 0 the value is exactly eps, not 0.
        assert float(smooth_abs(jnp.array(0.0), eps=1e-3)) == pytest.approx(1e-3)

    def test_gradient_defined_at_zero(self):
        # jnp.abs has an undefined/zero subgradient at 0; smooth_abs is 0 here
        # but, crucially, finite (no NaN) — and nonzero just off-centre.
        g0 = jax.grad(lambda x: smooth_abs(x, eps=1e-3))(0.0)
        assert jnp.isfinite(g0)
        assert float(g0) == pytest.approx(0.0, abs=1e-9)
        g = jax.grad(lambda x: smooth_abs(x, eps=1e-3))(1e-4)
        assert jnp.isfinite(g) and float(g) != 0.0

    def test_eps_zero_is_hard_abs(self):
        x = jnp.array([-2.0, 3.0])
        np.testing.assert_allclose(smooth_abs(x, eps=0.0), jnp.abs(x), atol=0.0)


class TestSmoothClamp:
    def test_above_clamp_is_near_identity(self):
        x = jnp.array([5.0, 10.0])
        np.testing.assert_allclose(smooth_clamp(x, 0.0, sharpness=50.0), x, atol=1e-3)

    def test_below_clamp_floors(self):
        x = jnp.array([-5.0, -10.0])
        out = smooth_clamp(x, 0.0, sharpness=50.0)
        np.testing.assert_allclose(out, 0.0, atol=1e-3)

    def test_never_below_min(self):
        # Never drops below x_min (keeps thickness positive); the softplus
        # term is >= 0, and strictly positive until it underflows far below
        # the clamp.
        x = jnp.linspace(-5.0, 5.0, 21)
        assert bool((smooth_clamp(x, 1.0) >= 1.0).all())
        assert bool((smooth_clamp(jnp.array([0.5, 0.9, 1.0]), 1.0) > 1.0).all())

    def test_gradient_positive_everywhere(self):
        # Unlike maximum(x, x_min) (zero gradient below the clamp), the
        # gradient is a strictly positive logistic everywhere.
        grad = jax.vmap(jax.grad(lambda x: smooth_clamp(x, 0.0)))
        g = grad(jnp.linspace(-5.0, 5.0, 21))
        assert bool((g > 0.0).all())


class TestSmoothMax:
    def test_matches_maximum(self):
        x = jnp.array([-2.0, 1.0, 4.0])
        y = jnp.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            smooth_max(x, y, sharpness=50.0), jnp.maximum(x, y), atol=1e-3
        )

    def test_both_gradients_nonzero(self):
        # Neither argument's sensitivity is lost at the crossover.
        gx = jax.grad(lambda x: smooth_max(x, 0.0))(0.0)
        assert jnp.isfinite(gx) and float(gx) == pytest.approx(0.5, abs=1e-6)
