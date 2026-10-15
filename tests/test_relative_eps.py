"""Relative WENO / TVD epsilons keep values and gradients finite (#243, #294).

The floors are relative to the stencil (WENO: ``eps_rel · Σβ`` on indicators
normalised by ``Σβ`` held out of the gradient, exact by Euler's theorem;
TVD: ``eps_rel · |num|`` plus ``sqrt(tiny)``), so rescaling the field
rescales the reconstruction exactly (``tests/test_scale_covariance.py``).
These tests pin what a careless relative floor breaks: gradients on flat
stencils (``0 · inf``), in float32 and at extreme amplitudes, the sign of a
tiny downwind slope, and ``van_leer(inf)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import finitevolx as fvx
from finitevolx._src.advection import limiters, reconstruction, weno

N = 8  # interior cells
WENO = ("weno3", "weno5", "wenoz5", "weno7", "weno9")
TVD = ("minmod", "van_leer", "superbee", "mc")


def _tendency_grad(h, method, mask=None):
    grid = fvx.CartesianGrid2D.from_interior(N, N, 1.0, 1.0)
    adv = fvx.Advection2D(grid, mask=mask)
    u = jnp.ones_like(h)
    v = 0.5 * jnp.ones_like(h)
    return jax.grad(lambda q: jnp.sum(adv(q, u, v, method=method) ** 2))(h)


def _smooth(dtype, amplitude=1.0):
    x = (np.arange(N + 2) - 0.5) / N
    xx, yy = np.meshgrid(x, x)
    return jnp.asarray(
        amplitude * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy), dtype
    )


@pytest.mark.slow  # gradient of a jitted Advection2D: compile-bound, 1-5 s each
class TestGradientsStayFinite:
    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    @pytest.mark.parametrize("method", WENO + TVD)
    def test_flat_field(self, method, dtype):
        """An ocean at rest: every stencil is flat (Σβ = 0, num = den = 0)."""
        g = _tendency_grad(jnp.ones((N + 2, N + 2), dtype), method)
        assert bool(jnp.isfinite(g).all())

    @pytest.mark.parametrize("method", WENO + TVD)
    def test_plateaus(self, method):
        """A step: flat stencils next to steep ones."""
        x = np.arange(N + 2)
        h = jnp.asarray(np.where(x[None, :] > N // 2, 2.0, 1.0) * np.ones((N + 2, 1)))
        assert bool(jnp.isfinite(_tendency_grad(h, method)).all())

    @pytest.mark.parametrize(
        "dtype, amplitude",
        [(jnp.float64, 1e-80), (jnp.float32, 1.0), (jnp.float32, 1e-9)],
    )
    @pytest.mark.parametrize("method", WENO + TVD)
    def test_float32_and_extreme_amplitudes(self, method, dtype, amplitude):
        g = _tendency_grad(_smooth(dtype, amplitude), method)
        assert bool(jnp.isfinite(g).all())

    @pytest.mark.parametrize("method", ("weno5", "minmod"))
    def test_masked_basin_at_rest(self, method):
        m = np.zeros((N + 2, N + 2), dtype=bool)
        m[1:-1, 1:-1] = True
        m[3:5, 3:5] = False
        g = _tendency_grad(jnp.ones((N + 2, N + 2)), method, fvx.Mask2D.from_mask(m))
        assert bool(jnp.isfinite(g).all())


class TestFlatStencilGivesLinearWeights:
    def test_weno5_gradient_is_the_linear_reconstruction(self):
        """On a flat stencil the weights are the linear ones, so the gradient
        is the fifth-order linear stencil (g1, g2, g3) = (0.1, 0.6, 0.3)."""
        grad = jax.grad(lambda q: weno.weno_5pts(*q))(jnp.ones(5))
        linear = (
            0.1 * np.array([1 / 3, -7 / 6, 11 / 6, 0, 0])
            + 0.6 * np.array([0, -1 / 6, 5 / 6, 1 / 3, 0])
            + 0.3 * np.array([0, 0, 1 / 3, 5 / 6, -1 / 6])
        )
        np.testing.assert_allclose(np.asarray(grad), linear, atol=1e-12)

    @pytest.mark.parametrize(
        "fn, npts",
        [
            (weno.weno_3pts, 3),
            (weno.weno_3pts_improved, 3),
            (weno.weno_5pts_improved, 5),
            (weno.weno_7pts, 7),
            (weno.weno_9pts, 9),
        ],
    )
    def test_flat_stencil_reconstructs_the_constant(self, fn, npts):
        q = jnp.full(npts, 2.5)
        assert float(fn(*q)) == pytest.approx(2.5, abs=1e-14)
        assert bool(jnp.isfinite(jax.grad(lambda z: fn(*z))(q)).all())


class TestTvdGuard:
    def test_tiny_downwind_slope_keeps_its_sign(self):
        """#294: num = 1, den = -0.5e-8 must give r < 0 (phi = 0, upwind)."""
        r = reconstruction._slope_ratio(jnp.asarray(1.0), jnp.asarray(-0.5e-8))
        assert float(r) < 0
        for name in TVD:
            assert float(reconstruction._LIMITERS[name](r)) == 0.0

    def test_flat_stencil_ratio_is_zero(self):
        r = reconstruction._slope_ratio(jnp.asarray(0.0), jnp.asarray(0.0))
        assert float(r) == 0.0

    def test_ratio_is_exact_to_eps_rel(self):
        num, den = jnp.asarray(0.3), jnp.asarray(0.7)
        r = reconstruction._slope_ratio(num, den)
        # relative floor 1e-6 · |num| on |den| = 0.7: within ~1e-6 relative
        assert float(r) == pytest.approx(0.3 / 0.7, rel=2e-6)


class TestVanLeer:
    def test_infinite_ratio(self):
        assert float(limiters.van_leer(jnp.asarray(jnp.inf))) == 2.0
        assert float(jax.grad(limiters.van_leer)(jnp.asarray(jnp.inf))) == 0.0

    @pytest.mark.parametrize("dtype, r", [(jnp.float64, 1e-200), (jnp.float32, 1e-25)])
    def test_tiny_ratio_gradient(self, dtype, r):
        assert float(
            jax.grad(limiters.van_leer)(jnp.asarray(r, dtype))
        ) == pytest.approx(2.0)

    def test_matches_the_textbook_form(self):
        r = jnp.linspace(-3.0, 30.0, 301)
        textbook = (r + jnp.abs(r)) / (1.0 + jnp.abs(r))
        np.testing.assert_allclose(
            limiters.van_leer(r), textbook, rtol=1e-14, atol=1e-15
        )
