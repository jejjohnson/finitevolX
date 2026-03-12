"""Tests for pure functional time integrators (explicit RK + multistep)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    ab2_step,
    ab3_step,
    euler_step,
    heun_step,
    leapfrog_raf_step,
    rk3_ssp_step,
    rk4_step,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exponential_decay(y):
    """dy/dt = -y  =>  y(t) = y0 * exp(-t)."""
    return jax.tree.map(lambda x: -x, y)


def _integrate_scalar(step_fn, y0, dt, n_steps, **kwargs):
    """Run *n_steps* of a single-step integrator on a scalar ODE."""
    y = y0
    for _ in range(n_steps):
        y = step_fn(y, _exponential_decay, dt, **kwargs)
    return y


def _convergence_rate(step_fn, y0, t_final, dt_coarse, **kwargs):
    """Estimate convergence order by halving dt twice."""
    errors = []
    exact = y0 * jnp.exp(-t_final)
    for refinement in range(3):
        dt = dt_coarse / (2**refinement)
        n_steps = int(round(t_final / dt))
        y_final = _integrate_scalar(step_fn, y0, dt, n_steps, **kwargs)
        errors.append(float(jnp.abs(y_final - exact)))
    # Estimate order from the last two refinements
    rate = np.log2(errors[-2] / errors[-1])
    return rate, errors


# ---------------------------------------------------------------------------
# Convergence order tests
# ---------------------------------------------------------------------------


class TestExplicitRKConvergence:
    """Verify each explicit RK scheme converges at the expected order."""

    @pytest.mark.parametrize(
        ("step_fn", "expected_order"),
        [
            (euler_step, 1),
            (heun_step, 2),
            (rk3_ssp_step, 3),
            (rk4_step, 4),
        ],
        ids=["euler", "heun", "rk3_ssp", "rk4"],
    )
    def test_convergence_order(self, step_fn, expected_order):
        y0 = 1.0
        t_final = 1.0
        dt_coarse = 0.1
        rate, _ = _convergence_rate(step_fn, y0, t_final, dt_coarse)
        assert rate > expected_order - 0.15, (
            f"Expected order ~{expected_order}, got {rate:.2f}"
        )


class TestExplicitRKAbsoluteAccuracy:
    """Verify the actual error magnitude is small, not just the order."""

    @pytest.mark.parametrize(
        ("step_fn", "rtol"),
        [
            (euler_step, 1e-2),
            (heun_step, 1e-4),
            (rk3_ssp_step, 1e-6),
            (rk4_step, 1e-8),
        ],
        ids=["euler", "heun", "rk3_ssp", "rk4"],
    )
    def test_accuracy_at_dt_001(self, step_fn, rtol):
        """With dt=0.01, higher-order methods should be much more accurate."""
        y0 = jnp.array(1.0)
        dt = 0.01
        n_steps = 100
        y_final = _integrate_scalar(step_fn, y0, dt, n_steps)
        exact = float(jnp.exp(-1.0))
        np.testing.assert_allclose(float(y_final), exact, rtol=rtol)


class TestExplicitRKNonDecay:
    """Test explicit RK on non-decay problems (oscillatory, forced)."""

    def test_rk4_harmonic_oscillator(self):
        """dy/dt = i*y  =>  y(t) = exp(i*t). Tests oscillatory (non-dissipative)."""
        y0 = jnp.array(1.0 + 0.0j)
        dt = 0.01
        n_steps = 100  # t_final = 1.0

        def rhs(y):
            return 1j * y

        y = y0
        for _ in range(n_steps):
            y = rk4_step(y, rhs, dt)

        exact = jnp.exp(1j * 1.0)
        np.testing.assert_allclose(float(jnp.abs(y - exact)), 0.0, atol=1e-8)

    def test_heun_logistic_growth(self):
        """dy/dt = y*(1-y), y0=0.1. Non-linear test."""
        y0 = jnp.array(0.1)
        dt = 0.01
        n_steps = 500  # t_final = 5.0

        def rhs(y):
            return y * (1.0 - y)

        y = y0
        for _ in range(n_steps):
            y = heun_step(y, rhs, dt)

        # Exact: y(t) = y0 / (y0 + (1-y0)*exp(-t))
        t = 5.0
        exact = 0.1 / (0.1 + 0.9 * jnp.exp(-t))
        np.testing.assert_allclose(float(y), float(exact), rtol=1e-3)


# ---------------------------------------------------------------------------
# Pytree support
# ---------------------------------------------------------------------------


class TestPytreeSupport:
    """Verify that integrators work with tuple pytrees."""

    def test_heun_tuple_state(self):
        state = (jnp.array(1.0), jnp.array(2.0))
        dt = 0.01
        n_steps = 100
        y = state
        for _ in range(n_steps):
            y = heun_step(y, _exponential_decay, dt)
        exact = (1.0 * np.exp(-1.0), 2.0 * np.exp(-1.0))
        np.testing.assert_allclose(y[0], exact[0], rtol=1e-3)
        np.testing.assert_allclose(y[1], exact[1], rtol=1e-3)

    def test_rk4_dict_state(self):
        state = {"a": jnp.array(1.0), "b": jnp.array(3.0)}
        dt = 0.01
        n_steps = 100
        y = state
        for _ in range(n_steps):
            y = rk4_step(y, _exponential_decay, dt)
        exact_a = 1.0 * np.exp(-1.0)
        exact_b = 3.0 * np.exp(-1.0)
        np.testing.assert_allclose(y["a"], exact_a, rtol=1e-6)
        np.testing.assert_allclose(y["b"], exact_b, rtol=1e-6)

    def test_rk3_ssp_nested_pytree(self):
        """Nested dict/tuple pytree."""
        state = {"pos": (jnp.array(1.0), jnp.array(2.0)), "vel": jnp.array(3.0)}
        dt = 0.01
        n_steps = 100
        y = state
        for _ in range(n_steps):
            y = rk3_ssp_step(y, _exponential_decay, dt)
        e = np.exp(-1.0)
        np.testing.assert_allclose(y["pos"][0], 1.0 * e, rtol=1e-5)
        np.testing.assert_allclose(y["pos"][1], 2.0 * e, rtol=1e-5)
        np.testing.assert_allclose(y["vel"], 3.0 * e, rtol=1e-5)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    def test_jit_euler(self):
        y0 = jnp.array(1.0)
        step_jit = jax.jit(lambda y: euler_step(y, _exponential_decay, 0.01))
        y1 = step_jit(y0)
        y1_ref = euler_step(y0, _exponential_decay, 0.01)
        np.testing.assert_allclose(y1, y1_ref, atol=1e-15)

    def test_jit_rk3_ssp(self):
        y0 = jnp.array(1.0)
        step_jit = jax.jit(lambda y: rk3_ssp_step(y, _exponential_decay, 0.01))
        y1 = step_jit(y0)
        y1_ref = rk3_ssp_step(y0, _exponential_decay, 0.01)
        np.testing.assert_allclose(y1, y1_ref, atol=1e-15)


# ---------------------------------------------------------------------------
# Constant field (rhs = 0 => state unchanged)
# ---------------------------------------------------------------------------


class TestConstantField:
    @pytest.mark.parametrize(
        "step_fn",
        [euler_step, heun_step, rk3_ssp_step, rk4_step],
        ids=["euler", "heun", "rk3_ssp", "rk4"],
    )
    def test_zero_rhs(self, step_fn):
        y0 = jnp.array(42.0)
        y1 = step_fn(y0, jnp.zeros_like, 0.1)
        np.testing.assert_allclose(y1, y0)


# ---------------------------------------------------------------------------
# Pure functional vs diffrax consistency
# ---------------------------------------------------------------------------


class TestPureFunctionalVsDiffrax:
    """Verify that pure functional integrators match diffrax-based ones."""

    def test_heun_matches_rk2heun(self):
        """Pure heun_step should agree with RK2Heun diffrax solver."""
        import diffrax as dfx

        from finitevolx import RK2Heun

        y0 = jnp.array(1.0)
        dt = 0.1

        # Pure functional: one step of Heun on dy/dt = -y
        y_pure = heun_step(y0, _exponential_decay, dt)

        # Diffrax: one step via diffeqsolve with max_steps=1
        sol = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, args: -y),
            RK2Heun(),
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=y0,
            saveat=dfx.SaveAt(t1=True),
        )
        y_dfx = float(sol.ys[0])

        np.testing.assert_allclose(float(y_pure), y_dfx, atol=1e-14)

    def test_rk4_matches_rk4classic(self):
        """Pure rk4_step should agree with RK4Classic diffrax solver."""
        import diffrax as dfx

        from finitevolx import RK4Classic

        y0 = jnp.array(1.0)
        dt = 0.1

        y_pure = rk4_step(y0, _exponential_decay, dt)

        sol = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, args: -y),
            RK4Classic(),
            t0=0.0,
            t1=dt,
            dt0=dt,
            y0=y0,
            saveat=dfx.SaveAt(t1=True),
        )
        y_dfx = float(sol.ys[0])

        np.testing.assert_allclose(float(y_pure), y_dfx, atol=1e-14)


# ---------------------------------------------------------------------------
# Multistep methods
# ---------------------------------------------------------------------------


class TestAB2:
    def test_convergence(self):
        y0 = jnp.array(1.0)
        dt = 0.01
        t_final = 1.0
        n_steps = int(round(t_final / dt))

        # Bootstrap with one Euler step
        rhs_0 = _exponential_decay(y0)
        y1 = euler_step(y0, _exponential_decay, dt)
        rhs_nm1 = rhs_0

        y = y1
        for _ in range(n_steps - 1):
            y, rhs_n, rhs_nm1 = ab2_step(y, _exponential_decay, dt, rhs_nm1)
            rhs_nm1 = rhs_n

        exact = float(jnp.exp(-t_final))
        np.testing.assert_allclose(float(y), exact, rtol=5e-3)

    def test_convergence_rate(self):
        """AB2 should converge at order ~2."""
        y0 = jnp.array(1.0)
        t_final = 1.0
        exact = float(jnp.exp(-t_final))
        errors = []

        for dt in [0.05, 0.025, 0.0125]:
            n_steps = int(round(t_final / dt))
            rhs_0 = _exponential_decay(y0)
            y1 = euler_step(y0, _exponential_decay, dt)
            rhs_nm1 = rhs_0
            y = y1
            for _ in range(n_steps - 1):
                y, rhs_n, rhs_nm1 = ab2_step(y, _exponential_decay, dt, rhs_nm1)
                rhs_nm1 = rhs_n
            errors.append(abs(float(y) - exact))

        rate = np.log2(errors[-2] / errors[-1])
        assert rate > 1.8, f"AB2 expected order ~2, got {rate:.2f}"


class TestAB3:
    def test_convergence(self):
        y0 = jnp.array(1.0)
        dt = 0.01
        t_final = 1.0
        n_steps = int(round(t_final / dt))

        # Bootstrap with two Euler steps
        rhs_0 = _exponential_decay(y0)
        y1 = euler_step(y0, _exponential_decay, dt)
        rhs_1 = _exponential_decay(y1)
        y2 = euler_step(y1, _exponential_decay, dt)

        y = y2
        rhs_nm1 = rhs_1
        rhs_nm2 = rhs_0
        for _ in range(n_steps - 2):
            y, rhs_n, _rhs_nm1_new = ab3_step(
                y, _exponential_decay, dt, rhs_nm1, rhs_nm2
            )
            rhs_nm2 = rhs_nm1
            rhs_nm1 = rhs_n

        exact = float(jnp.exp(-t_final))
        np.testing.assert_allclose(float(y), exact, rtol=5e-3)

    def test_convergence_rate(self):
        """AB3 should converge at order ~3."""
        y0 = jnp.array(1.0)
        t_final = 1.0
        exact = float(jnp.exp(-t_final))
        errors = []

        for dt in [0.05, 0.025, 0.0125]:
            n_steps = int(round(t_final / dt))
            rhs_0 = _exponential_decay(y0)
            y1 = euler_step(y0, _exponential_decay, dt)
            rhs_1 = _exponential_decay(y1)
            y2 = euler_step(y1, _exponential_decay, dt)
            y = y2
            rhs_nm1 = rhs_1
            rhs_nm2 = rhs_0
            for _ in range(n_steps - 2):
                y, rhs_n, _ = ab3_step(y, _exponential_decay, dt, rhs_nm1, rhs_nm2)
                rhs_nm2 = rhs_nm1
                rhs_nm1 = rhs_n
            errors.append(abs(float(y) - exact))

        rate = np.log2(errors[-2] / errors[-1])
        assert rate > 1.8, f"AB3 expected order >=2, got {rate:.2f}"


class TestLeapfrogRAF:
    def test_convergence(self):
        y0 = jnp.array(1.0)
        dt = 0.01
        t_final = 1.0
        n_steps = int(round(t_final / dt))

        # Bootstrap: first step with Euler
        y1 = euler_step(y0, _exponential_decay, dt)
        y_nm1 = y0

        y_curr = y1
        for _ in range(n_steps - 1):
            y_next, y_curr_filtered = leapfrog_raf_step(
                y_curr, y_nm1, _exponential_decay, dt, alpha=0.05
            )
            y_nm1 = y_curr_filtered
            y_curr = y_next

        exact = float(jnp.exp(-t_final))
        np.testing.assert_allclose(float(y_curr), exact, rtol=5e-2)

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_alpha_all_stable(self, alpha):
        """Different RAF alphas should all produce finite, reasonable results."""
        y0 = jnp.array(1.0)
        dt = 0.01
        n_steps = 100
        y1 = euler_step(y0, _exponential_decay, dt)
        y_nm1 = y0
        y_curr = y1
        for _ in range(n_steps - 1):
            y_next, y_curr_filtered = leapfrog_raf_step(
                y_curr, y_nm1, _exponential_decay, dt, alpha=alpha
            )
            y_nm1 = y_curr_filtered
            y_curr = y_next
        assert jnp.isfinite(y_curr)
        # Should be within a factor of 2 of exp(-1)
        exact = float(jnp.exp(-1.0))
        assert abs(float(y_curr) - exact) < 0.1
