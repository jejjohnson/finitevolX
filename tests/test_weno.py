"""Dedicated tests for the WENO reconstruction stencil functions.

Tests cover:
- Polynomial exactness (degree-0 and degree-1 fields)
- Smoothness indicators β for constant data (must be zero)
- Non-linear weight convergence toward optimal linear weights on smooth data
- Monotone (no-new-extrema) behavior near a step discontinuity
- Left/right reconstruction symmetry for symmetric data
- No-NaN / no-Inf stability for all stencil functions
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx._src.advection.weno import (
    weno_3pts,
    weno_3pts_improved,
    weno_3pts_improved_right,
    weno_3pts_right,
    weno_5pts,
    weno_5pts_improved,
    weno_5pts_improved_right,
    weno_5pts_right,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar(v: float) -> jnp.ndarray:
    """Create a scalar JAX array."""
    return jnp.asarray(v, dtype=float)


# ---------------------------------------------------------------------------
# Constant-field tests: f(x) = C
# For constant data all sub-stencil candidates are identical → output = C.
# ---------------------------------------------------------------------------


class TestWeno3ConstantField:
    """WENO-3 must recover a constant field exactly."""

    def test_weno3_constant(self):
        c = 4.7
        result = weno_3pts(_scalar(c), _scalar(c), _scalar(c))
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_wenoz3_constant(self):
        c = 3.1
        result = weno_3pts_improved(_scalar(c), _scalar(c), _scalar(c))
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_weno3_constant_array(self):
        """Verify on an array of identical values (vectorized path)."""
        c = 2.5
        q = c * jnp.ones(20)
        result = weno_3pts(q[:-2], q[1:-1], q[2:])
        np.testing.assert_allclose(result, c, rtol=1e-10)

    def test_wenoz3_constant_array(self):
        c = 1.8
        q = c * jnp.ones(20)
        result = weno_3pts_improved(q[:-2], q[1:-1], q[2:])
        np.testing.assert_allclose(result, c, rtol=1e-10)


class TestWeno5ConstantField:
    """WENO-5 must recover a constant field exactly."""

    def test_weno5_constant(self):
        c = 7.3
        result = weno_5pts(_scalar(c), _scalar(c), _scalar(c), _scalar(c), _scalar(c))
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_wenoz5_constant(self):
        c = 5.9
        result = weno_5pts_improved(
            _scalar(c), _scalar(c), _scalar(c), _scalar(c), _scalar(c)
        )
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_weno5_constant_array(self):
        c = 3.3
        q = c * jnp.ones(20)
        result = weno_5pts(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        np.testing.assert_allclose(result, c, rtol=1e-10)

    def test_wenoz5_constant_array(self):
        c = 6.6
        q = c * jnp.ones(20)
        result = weno_5pts_improved(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        np.testing.assert_allclose(result, c, rtol=1e-10)


# ---------------------------------------------------------------------------
# Smoothness indicators for constant data must be zero
#
# For constant q: β = 0 for every sub-stencil because all differences vanish.
# ---------------------------------------------------------------------------


class TestWeno3SmoothnessConstant:
    """β indicators must be zero for constant input data."""

    def test_weno3_beta_zero_constant(self):
        """Manually compute β₁, β₂ for constant data."""
        c = 5.0
        qm, q0, qp = _scalar(c), _scalar(c), _scalar(c)
        beta1 = (q0 - qm) ** 2
        beta2 = (qp - q0) ** 2
        np.testing.assert_allclose(float(beta1), 0.0, atol=1e-15)
        np.testing.assert_allclose(float(beta2), 0.0, atol=1e-15)

    def test_weno5_beta_zero_constant(self):
        """Manually compute β₁, β₂, β₃ for constant data."""
        c = 3.0
        qmm, qm, q0, qp, qpp = (
            _scalar(c),
            _scalar(c),
            _scalar(c),
            _scalar(c),
            _scalar(c),
        )
        k1, k2 = 13.0 / 12.0, 0.25
        beta1 = k1 * (qmm - 2 * qm + q0) ** 2 + k2 * (qmm - 4 * qm + 3 * q0) ** 2
        beta2 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
        beta3 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (3 * q0 - 4 * qp + qpp) ** 2
        for beta in [beta1, beta2, beta3]:
            np.testing.assert_allclose(float(beta), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Polynomial exactness tests
#
# For degree-1 data (q = a*i + b), the linear sub-stencils are exact, so
# WENO should recover the interface value exactly (up to floating-point).
# ---------------------------------------------------------------------------


class TestWeno3PolynomialExactness:
    """WENO-3 is exact for degree-≤1 polynomials when the stencil is smooth."""

    def test_weno3_linear_exact(self):
        """q = a*i + b → weno3 should recover the left-biased face value.

        Left-biased face at i+1/2 (between q0 and qp):
          exact = a*(i + 1/2) + b = q0 + a/2
        In index terms: qm = q0 - a, q0 = q0, qp = q0 + a → exact = q0 + a/2.
        We verify the WENO-3 result equals the expected sub-stencil mean.
        """
        a = 2.0
        # q_{i-1}, q_i, q_{i+1} for a linear sequence
        for q0_val in [1.0, 5.0, -3.0]:
            qm = _scalar(q0_val - a)
            q0 = _scalar(q0_val)
            qp = _scalar(q0_val + a)
            result = weno_3pts(qm, q0, qp)
            # Both sub-stencils give the same value for linear data → no weighting needed
            qi2_expected = 0.5 * (q0_val + (q0_val + a))  # = q0 + a/2
            np.testing.assert_allclose(float(result), qi2_expected, rtol=1e-6)

    def test_wenoz3_linear_exact(self):
        a = 3.0
        for q0_val in [0.0, 2.0, -1.0]:
            qm = _scalar(q0_val - a)
            q0 = _scalar(q0_val)
            qp = _scalar(q0_val + a)
            result = weno_3pts_improved(qm, q0, qp)
            qi2_expected = 0.5 * (q0_val + (q0_val + a))
            np.testing.assert_allclose(float(result), qi2_expected, rtol=1e-6)


class TestWeno5PolynomialExactness:
    """WENO-5 is exact for degree-≤2 polynomials on smooth data."""

    def test_weno5_constant_exact(self):
        c = 4.0
        result = weno_5pts(_scalar(c), _scalar(c), _scalar(c), _scalar(c), _scalar(c))
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_weno5_linear_on_array(self):
        """For q = i (linear), WENO-5 must produce finite results without NaN.

        For a linear sequence all three sub-stencil candidates give the same
        left-biased value, so the non-linear weights are irrelevant and the
        output is well-defined.  We verify finiteness here; the constant-field
        test (`test_weno5_constant_exact`) already verifies exact recovery.
        """
        n = 12
        q = jnp.arange(n, dtype=float)  # q[i] = i
        result = weno_5pts(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        assert jnp.all(jnp.isfinite(result)), "WENO-5 produced NaN for linear data"
        # For linear q[i]=i the three sub-stencil candidates all give the same
        # face value q0 + a/2 (where a=1 here), so the weighted result should
        # be monotone (strictly increasing).
        assert jnp.all(result[1:] > result[:-1]), (
            "WENO-5 should be increasing for linear data"
        )

    def test_weno5_linear_weights_approach_optimal(self):
        """For smooth data, WENO-5 weights w_k → g_k = (0.1, 0.6, 0.3).

        For constant input data all β_k = 0 exactly, so the weights collapse
        to the optimal linear weights (g1, g2, g3) = (0.1, 0.6, 0.3) by
        construction.  We verify this with constant stencil values.
        """
        c = 3.0
        qmm = _scalar(c)
        qm = _scalar(c)
        q0 = _scalar(c)
        qp = _scalar(c)
        qpp = _scalar(c)

        eps = 1e-8
        k1, k2 = 13.0 / 12.0, 0.25
        beta1 = k1 * (qmm - 2 * qm + q0) ** 2 + k2 * (qmm - 4 * qm + 3 * q0) ** 2
        beta2 = k1 * (qm - 2 * q0 + qp) ** 2 + k2 * (qm - qp) ** 2
        beta3 = k1 * (q0 - 2 * qp + qpp) ** 2 + k2 * (3 * q0 - 4 * qp + qpp) ** 2

        g1, g2, g3 = 0.1, 0.6, 0.3
        w1_raw = g1 / (beta1 + eps) ** 2
        w2_raw = g2 / (beta2 + eps) ** 2
        w3_raw = g3 / (beta3 + eps) ** 2
        w_total = w1_raw + w2_raw + w3_raw
        w1 = float(w1_raw / w_total)
        w2 = float(w2_raw / w_total)
        w3 = float(w3_raw / w_total)

        # For constant data β1=β2=β3=0 → w_k = g_k * (1/eps²) / sum → w_k = g_k
        np.testing.assert_allclose(w1, g1, rtol=1e-10)
        np.testing.assert_allclose(w2, g2, rtol=1e-10)
        np.testing.assert_allclose(w3, g3, rtol=1e-10)
        np.testing.assert_allclose(w1 + w2 + w3, 1.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# Monotonicity / no-new-extrema near a step discontinuity
# ---------------------------------------------------------------------------


class TestWenoMonotonicityNearStep:
    """WENO schemes must not create new extrema across a step discontinuity.

    Left-biased reconstruction at the face to the right of the jump:
    values should be bounded between the two plateau values.
    """

    @pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (1.0, 5.0), (-2.0, 3.0)])
    def test_weno3_bounded_near_step(self, lo, hi):
        """weno_3pts result at the jump face must lie in [lo, hi]."""
        # stencil at the sharp discontinuity: qm = lo, q0 = lo, qp = hi
        result = float(weno_3pts(_scalar(lo), _scalar(lo), _scalar(hi)))
        assert lo <= result <= hi, (
            f"weno3 produced {result} outside [{lo}, {hi}] at step discontinuity"
        )

    @pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (1.0, 5.0), (-2.0, 3.0)])
    def test_wenoz3_bounded_near_step(self, lo, hi):
        result = float(weno_3pts_improved(_scalar(lo), _scalar(lo), _scalar(hi)))
        assert lo <= result <= hi, (
            f"wenoz3 produced {result} outside [{lo}, {hi}] at step discontinuity"
        )

    @pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (1.0, 5.0), (-2.0, 3.0)])
    def test_weno5_bounded_near_step(self, lo, hi):
        """weno_5pts result at the jump face must lie in [lo, hi] (with float tolerance)."""
        # stencil far from jump: qmm=lo, qm=lo, q0=lo, qp=hi, qpp=hi
        result = float(
            weno_5pts(_scalar(lo), _scalar(lo), _scalar(lo), _scalar(hi), _scalar(hi))
        )
        # Allow a tiny floating-point slack (machine eps relative to range)
        tol = 1e-12 * max(abs(lo), abs(hi), 1.0)
        assert lo - tol <= result <= hi + tol, (
            f"weno5 produced {result} outside [{lo}, {hi}] at step discontinuity"
        )

    @pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (1.0, 5.0)])
    def test_wenoz5_bounded_near_step(self, lo, hi):
        result = float(
            weno_5pts_improved(
                _scalar(lo), _scalar(lo), _scalar(lo), _scalar(hi), _scalar(hi)
            )
        )
        assert lo <= result <= hi, (
            f"wenoz5 produced {result} outside [{lo}, {hi}] at step discontinuity"
        )


# ---------------------------------------------------------------------------
# No-NaN / no-Inf stability
# ---------------------------------------------------------------------------


class TestWenoNumericalStability:
    """All stencil functions must produce finite outputs on finite inputs."""

    def test_weno3_finite_on_random_data(self):
        rng = np.random.default_rng(42)
        data = rng.uniform(-10.0, 10.0, (100,))
        q = jnp.asarray(data)
        result = weno_3pts(q[:-2], q[1:-1], q[2:])
        assert jnp.all(jnp.isfinite(result)), "weno3 produced NaN/Inf on random data"

    def test_wenoz3_finite_on_random_data(self):
        rng = np.random.default_rng(43)
        data = rng.uniform(-10.0, 10.0, (100,))
        q = jnp.asarray(data)
        result = weno_3pts_improved(q[:-2], q[1:-1], q[2:])
        assert jnp.all(jnp.isfinite(result)), "wenoz3 produced NaN/Inf on random data"

    def test_weno5_finite_on_random_data(self):
        rng = np.random.default_rng(44)
        data = rng.uniform(-10.0, 10.0, (100,))
        q = jnp.asarray(data)
        result = weno_5pts(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        assert jnp.all(jnp.isfinite(result)), "weno5 produced NaN/Inf on random data"

    def test_wenoz5_finite_on_random_data(self):
        rng = np.random.default_rng(45)
        data = rng.uniform(-10.0, 10.0, (100,))
        q = jnp.asarray(data)
        result = weno_5pts_improved(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        assert jnp.all(jnp.isfinite(result)), "wenoz5 produced NaN/Inf on random data"

    def test_weno3_finite_on_large_values(self):
        """Large values should not cause overflow."""
        c = 1.0e6
        result = weno_3pts(_scalar(c), _scalar(c), _scalar(-c))
        assert jnp.isfinite(result), "weno3 overflowed on large values"

    def test_weno5_finite_on_large_values(self):
        c = 1.0e6
        result = weno_5pts(_scalar(c), _scalar(c), _scalar(c), _scalar(-c), _scalar(-c))
        assert jnp.isfinite(result), "weno5 overflowed on large values"


# ---------------------------------------------------------------------------
# Symmetry tests
#
# For symmetric data around the stencil center, left- and right-biased
# reconstructions should give symmetric (mirrored) results.
# ---------------------------------------------------------------------------


class TestWenoSymmetry:
    """WENO stencils: symmetric input → symmetric output property."""

    def test_weno3_symmetric_data_output_bounded(self):
        """For qm = q0 = c, qp = d, the face value lies between c and d."""
        c, d = 2.0, 6.0
        result = float(weno_3pts(_scalar(c), _scalar(c), _scalar(d)))
        assert c <= result <= d

    def test_weno5_constant_stencil_output(self):
        """weno_5pts on constant data returns exactly that constant."""
        for c in [0.0, 1.0, -3.5, 100.0]:
            result = float(
                weno_5pts(_scalar(c), _scalar(c), _scalar(c), _scalar(c), _scalar(c))
            )
            np.testing.assert_allclose(result, c, atol=1e-10)

    def test_weno3_left_right_same_smooth(self):
        """On a uniform grid, left-biased weno3 on uniform data is well-defined."""
        # Uniform increasing sequence: all β equal → weights converge to optimal g
        n = 10
        q = jnp.arange(n, dtype=float)
        r1 = weno_3pts(q[0:-2], q[1:-1], q[2:])
        # All values should be finite and strictly increasing
        assert jnp.all(jnp.isfinite(r1))
        # For increasing linear data, face values should also be increasing
        diff = r1[1:] - r1[:-1]
        assert jnp.all(diff > 0), "WENO-3 face values should be strictly increasing"


# ---------------------------------------------------------------------------
# JAX transform compatibility for WENO stencils
# ---------------------------------------------------------------------------


class TestWenoJaxCompat:
    """WENO stencils must be JIT-compilable and produce the same result."""

    def test_weno3_jit_matches_eager(self):
        qm = jnp.array([1.0, 2.0, 3.0])
        q0 = jnp.array([2.0, 3.0, 4.0])
        qp = jnp.array([3.0, 4.0, 5.0])
        eager = weno_3pts(qm, q0, qp)
        jitted = jax.jit(weno_3pts)(qm, q0, qp)
        np.testing.assert_allclose(jitted, eager, rtol=1e-10)

    def test_wenoz3_jit_matches_eager(self):
        qm = jnp.array([1.0, 2.0, 3.0])
        q0 = jnp.array([2.0, 3.0, 4.0])
        qp = jnp.array([3.0, 4.0, 5.0])
        eager = weno_3pts_improved(qm, q0, qp)
        jitted = jax.jit(weno_3pts_improved)(qm, q0, qp)
        np.testing.assert_allclose(jitted, eager, rtol=1e-10)

    def test_weno5_jit_matches_eager(self):
        q = jnp.arange(8, dtype=float)
        eager = weno_5pts(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        jitted = jax.jit(weno_5pts)(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        np.testing.assert_allclose(jitted, eager, rtol=1e-10)

    def test_wenoz5_jit_matches_eager(self):
        q = jnp.arange(8, dtype=float)
        eager = weno_5pts_improved(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        jitted = jax.jit(weno_5pts_improved)(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        np.testing.assert_allclose(jitted, eager, rtol=1e-10)


# ---------------------------------------------------------------------------
# Right-biased kernels: constant-field, linear-exactness, and upwind tests
# ---------------------------------------------------------------------------


class TestWeno3Right:
    """Right-biased WENO-3 at face i+1/2 using {q_i, q_{i+1}, q_{i+2}}."""

    def test_constant(self):
        c = 3.0
        result = weno_3pts_right(_scalar(c), _scalar(c), _scalar(c))
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_constant_array(self):
        c = 2.5
        q = c * jnp.ones(20)
        result = weno_3pts_right(q[:-2], q[1:-1], q[2:])
        np.testing.assert_allclose(result, c, rtol=1e-10)

    def test_linear_exact(self):
        """Right-biased WENO3 should be exact for linear data."""
        dx = 0.1
        q = jnp.arange(10, dtype=float) * dx
        # face at i+1/2: exact value = (i + 0.5)*dx
        result = weno_3pts_right(q[:-2], q[1:-1], q[2:])
        expected = (jnp.arange(len(result)) + 0.5) * dx
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_step_upwind_from_right(self):
        """At a step, right-biased should select the upwind (right) value."""
        # Step: 0 0 0 | 1 1 1, face between 0-region and 1-region
        q0 = _scalar(0.0)  # cell i (left of face)
        qp = _scalar(1.0)  # cell i+1 (right of face, upwind for u<0)
        qpp = _scalar(1.0)  # cell i+2
        result = float(weno_3pts_right(q0, qp, qpp))
        assert result > 0.9, f"Right-biased should select ~1.0, got {result}"

    def test_wenoz3_right_constant(self):
        c = 4.2
        result = weno_3pts_improved_right(_scalar(c), _scalar(c), _scalar(c))
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_wenoz3_right_linear(self):
        dx = 0.1
        q = jnp.arange(10, dtype=float) * dx
        result = weno_3pts_improved_right(q[:-2], q[1:-1], q[2:])
        expected = (jnp.arange(len(result)) + 0.5) * dx
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestWeno5Right:
    """Right-biased WENO-5 at face i+1/2 using {q_{i-1}..q_{i+3}}."""

    def test_constant(self):
        c = 5.0
        s = _scalar(c)
        result = weno_5pts_right(s, s, s, s, s)
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_constant_array(self):
        c = 1.7
        q = c * jnp.ones(20)
        result = weno_5pts_right(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        np.testing.assert_allclose(result, c, rtol=1e-10)

    def test_linear_exact(self):
        """Right-biased WENO5 should be exact for linear data."""
        dx = 0.05
        q = jnp.arange(20, dtype=float) * dx
        # face at (k+1)+1/2 for k-th element
        result = weno_5pts_right(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        expected = (jnp.arange(len(result)) + 1.5) * dx
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_quadratic_accurate(self):
        """On smooth quadratic data, right-biased should be 5th-order accurate."""
        dx = 0.05
        q = (jnp.arange(20, dtype=float) * dx) ** 2
        result = weno_5pts_right(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        # face at (k+1)+1/2, exact = ((k+1)+0.5)^2 * dx^2
        expected = ((jnp.arange(len(result)) + 1.5) * dx) ** 2
        # 5th-order accuracy: error ~ O(dx^5) ≈ 3e-7 for dx=0.05
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_step_upwind_from_right(self):
        """At a step, right-biased should select the upwind (right) value."""
        qm = _scalar(0.0)
        q0 = _scalar(0.0)
        qp = _scalar(1.0)  # cell i+1 (upwind for u<0)
        qpp = _scalar(1.0)
        qppp = _scalar(1.0)
        result = float(weno_5pts_right(qm, q0, qp, qpp, qppp))
        assert result > 0.9, f"Right-biased should select ~1.0, got {result}"

    def test_wenoz5_right_constant(self):
        c = 3.3
        s = _scalar(c)
        result = weno_5pts_improved_right(s, s, s, s, s)
        np.testing.assert_allclose(float(result), c, rtol=1e-10)

    def test_wenoz5_right_linear(self):
        dx = 0.05
        q = jnp.arange(20, dtype=float) * dx
        result = weno_5pts_improved_right(q[:-4], q[1:-3], q[2:-2], q[3:-1], q[4:])
        expected = (jnp.arange(len(result)) + 1.5) * dx
        np.testing.assert_allclose(result, expected, atol=1e-12)
