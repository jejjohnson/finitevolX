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
