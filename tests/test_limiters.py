"""Tests for flux limiter functions in finitevolx._src.advection.limiters."""

import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.limiters import mc, minmod, superbee, van_leer


class TestMinmod:
    def test_negative_r_gives_zero(self):
        r = jnp.array([-2.0, -1.0, -0.1])
        np.testing.assert_array_equal(minmod(r), 0.0)

    def test_zero_r_gives_zero(self):
        assert float(minmod(jnp.array(0.0))) == pytest.approx(0.0)

    def test_r_between_0_and_1(self):
        r = jnp.array([0.3, 0.7])
        expected = jnp.array([0.3, 0.7])
        np.testing.assert_allclose(minmod(r), expected, rtol=1e-6)

    def test_r_equal_1(self):
        assert float(minmod(jnp.array(1.0))) == pytest.approx(1.0)

    def test_r_greater_than_1_capped_at_1(self):
        r = jnp.array([1.5, 2.0, 5.0])
        np.testing.assert_array_equal(minmod(r), 1.0)

    def test_output_shape(self):
        r = jnp.ones((4, 4))
        assert minmod(r).shape == (4, 4)


class TestVanLeer:
    def test_negative_r_gives_zero(self):
        r = jnp.array([-2.0, -1.0, -0.1])
        np.testing.assert_array_equal(van_leer(r), 0.0)

    def test_zero_r_gives_zero(self):
        assert float(van_leer(jnp.array(0.0))) == pytest.approx(0.0)

    def test_r_equal_1_gives_1(self):
        assert float(van_leer(jnp.array(1.0))) == pytest.approx(1.0)

    def test_r_large_approaches_2(self):
        # φ(r) = (r + |r|) / (1 + |r|) → 2 as r → ∞
        r = jnp.array([100.0, 1000.0])
        np.testing.assert_allclose(van_leer(r), 2.0, atol=0.02)

    def test_symmetric_around_1(self):
        # Van Leer is symmetric: φ(1/r) = φ(r)/r
        r = jnp.array([2.0, 4.0])
        phi_r = van_leer(r)
        phi_inv_r = van_leer(1.0 / r)
        np.testing.assert_allclose(phi_inv_r, phi_r / r, rtol=1e-5)

    def test_output_shape(self):
        r = jnp.ones((3, 5))
        assert van_leer(r).shape == (3, 5)

    def test_infinite_r_is_finite(self):
        """#294: ``(r + |r|) / (1 + |r|)`` was ``inf / inf = NaN`` at r = inf."""
        np.testing.assert_array_equal(
            van_leer(jnp.array([jnp.inf, -jnp.inf])), jnp.array([2.0, 0.0])
        )

    def test_matches_the_textbook_form(self):
        r = jnp.array([-3.0, 0.0, 1e-300, 0.25, 1.0, 7.0, 1e300])
        r_abs = jnp.abs(r)
        np.testing.assert_allclose(
            van_leer(r), (r + r_abs) / (1.0 + r_abs), rtol=1e-15, atol=0.0
        )

    def test_gradient_finite(self):
        import jax

        r = jnp.array([-2.0, 0.0, 0.5, 3.0, jnp.inf])
        grad = jax.vmap(jax.grad(van_leer))(r)
        assert bool(jnp.all(jnp.isfinite(grad)))


class TestSuperbee:
    def test_negative_r_gives_zero(self):
        r = jnp.array([-2.0, -1.0, -0.1])
        np.testing.assert_array_equal(superbee(r), 0.0)

    def test_zero_r_gives_zero(self):
        assert float(superbee(jnp.array(0.0))) == pytest.approx(0.0)

    def test_r_equal_1_gives_1(self):
        assert float(superbee(jnp.array(1.0))) == pytest.approx(1.0)

    def test_r_half_gives_1(self):
        # φ(0.5) = max(0, max(min(1, 1), min(0.5, 2))) = 1
        assert float(superbee(jnp.array(0.5))) == pytest.approx(1.0)

    def test_r_large_gives_2(self):
        r = jnp.array([3.0, 10.0])
        np.testing.assert_array_equal(superbee(r), 2.0)

    def test_output_shape(self):
        r = jnp.ones((2, 6))
        assert superbee(r).shape == (2, 6)


class TestMC:
    def test_negative_r_gives_zero(self):
        r = jnp.array([-2.0, -1.0, -0.1])
        np.testing.assert_array_equal(mc(r), 0.0)

    def test_zero_r_gives_zero(self):
        assert float(mc(jnp.array(0.0))) == pytest.approx(0.0)

    def test_r_equal_1_gives_1(self):
        assert float(mc(jnp.array(1.0))) == pytest.approx(1.0)

    def test_r_large_gives_2(self):
        r = jnp.array([5.0, 10.0])
        np.testing.assert_array_equal(mc(r), 2.0)

    def test_r_small_positive(self):
        # φ(0.1) = max(0, min((1+0.1)/2, 0.2, 2)) = max(0, min(0.55, 0.2, 2)) = 0.2
        assert float(mc(jnp.array(0.1))) == pytest.approx(0.2, rel=1e-5)

    def test_output_shape(self):
        r = jnp.ones((5, 3))
        assert mc(r).shape == (5, 3)


class TestLimitersPublicAPI:
    """Verify the limiters are exported from the top-level package."""

    def test_import_from_package(self):
        from finitevolx import (
            mc as _mc,
            minmod as _minmod,
            superbee as _superbee,
            van_leer as _van_leer,
        )

        r = jnp.array(1.0)
        # All should return 1.0 for r=1
        assert float(_minmod(r)) == pytest.approx(1.0)
        assert float(_van_leer(r)) == pytest.approx(1.0)
        assert float(_superbee(r)) == pytest.approx(1.0)
        assert float(_mc(r)) == pytest.approx(1.0)


class TestTvdSlopeRatio:
    """#294: the TVD slope ratio is relative and sign-safe."""

    @pytest.mark.parametrize("limiter", ["minmod", "van_leer", "superbee", "mc"])
    def test_opposite_signs_give_exactly_the_upwind_value(self, limiter):
        """Δ_bwd = 1, Δ_fwd = -0.5e-8: r < 0, φ = 0, face value = h[i].

        The old ``Δ_bwd / (Δ_fwd + 1e-8)`` flipped the denominator's sign
        (r = +2e8) and returned h[i] - 5e-9 (minmod 1 - 2.5e-9).
        """
        from finitevolx import CartesianGrid1D, Reconstruction1D

        recon = Reconstruction1D(grid=CartesianGrid1D.from_interior(4, 1.0))
        # face i+1/2 at i = 2: h[1] = 0, h[2] = 1, h[3] = 1 - 0.5e-8
        h = jnp.array([0.0, 0.0, 1.0, 1.0 - 0.5e-8, 1.0 - 0.5e-8, 1.0 - 0.5e-8])
        fe = recon.tvd_x(h, jnp.ones_like(h), limiter=limiter)
        assert float(fe[2]) == 1.0
        # mirrored, negative flow: face 2+1/2 reads h[3] upwind, h[4] beyond
        hm = h[::-1]
        fe = recon.tvd_x(hm, -jnp.ones_like(hm), limiter=limiter)
        assert float(fe[2]) == -float(hm[3])

    @pytest.mark.parametrize("limiter", ["minmod", "van_leer", "superbee", "mc"])
    @pytest.mark.parametrize("amp", [1e-9, 1e-300])
    def test_small_amplitude_is_not_first_order(self, limiter, amp):
        """At amplitude 1e-9 the old absolute eps sent φ to 0 (upwind1); the
        face value now scales exactly with the field (here a linear ramp,
        r = 1, face value h[i] + Δ/2 for every limiter)."""
        from finitevolx import CartesianGrid1D, Reconstruction1D

        recon = Reconstruction1D(grid=CartesianGrid1D.from_interior(6, 1.0))
        ramp = jnp.arange(8.0)
        u = jnp.ones_like(ramp)
        small = recon.tvd_x(amp * ramp, u, limiter=limiter)
        unit = recon.tvd_x(ramp, u, limiter=limiter)
        np.testing.assert_allclose(small[2:-2], amp * unit[2:-2], rtol=1e-15)
        np.testing.assert_allclose(unit[2:-2], ramp[2:-2] + 0.5, rtol=1e-15)

    @pytest.mark.parametrize("limiter", ["minmod", "van_leer", "superbee", "mc"])
    def test_constant_and_zero_fields_are_finite(self, limiter):
        """Δ_fwd = Δ_bwd = 0 gives r = 0 (never 0/0); a zero field gives 0."""
        from finitevolx import CartesianGrid1D, Reconstruction1D

        recon = Reconstruction1D(grid=CartesianGrid1D.from_interior(6, 1.0))
        for value in (0.0, 3.0):
            h = jnp.full(8, value)
            fe = recon.tvd_x(h, jnp.ones_like(h), limiter=limiter)
            np.testing.assert_array_equal(fe[1:-1], value)
