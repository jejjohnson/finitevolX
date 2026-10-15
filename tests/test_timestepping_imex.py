"""Tests for IMEX and split-explicit time integrators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx import imex_ssp2_step, split_explicit_step

# ---------------------------------------------------------------------------
# IMEX-SSP2
# ---------------------------------------------------------------------------


class TestIMEX_SSP2:
    """Test IMEX-SSP2 on a simple advection-diffusion-like split."""

    def test_pure_explicit_decay(self):
        """When implicit part is zero, IMEX reduces to explicit averaging."""
        y0 = jnp.array(1.0)
        dt = 0.01
        n_steps = 100

        def rhs_explicit(y):
            return -y

        def rhs_implicit(_y):
            return jnp.array(0.0)

        def implicit_solve(rhs, _gamma_dt):
            return rhs  # No implicit contribution

        y = y0
        for _ in range(n_steps):
            y = imex_ssp2_step(y, rhs_explicit, rhs_implicit, implicit_solve, dt)

        exact = float(jnp.exp(-1.0))
        np.testing.assert_allclose(float(y), exact, rtol=1e-2)

    def test_pure_implicit_decay(self):
        """When explicit part is zero, IMEX handles implicit decay."""
        y0 = jnp.array(1.0)
        dt = 0.01
        n_steps = 100

        def rhs_explicit(_y):
            return jnp.array(0.0)

        def rhs_implicit(y):
            return -y

        def implicit_solve(rhs, gamma_dt):
            # Solve: Y - gamma_dt * (-Y) = rhs => Y * (1 + gamma_dt) = rhs
            return rhs / (1.0 + gamma_dt)

        y = y0
        for _ in range(n_steps):
            y = imex_ssp2_step(y, rhs_explicit, rhs_implicit, implicit_solve, dt)

        exact = float(jnp.exp(-1.0))
        np.testing.assert_allclose(float(y), exact, rtol=5e-2)

    def test_mixed_explicit_implicit(self):
        """Both explicit and implicit parts active simultaneously.

        Split dy/dt = -0.3*y (explicit) + -0.7*y (implicit).
        Total: dy/dt = -y => y(t) = exp(-t).
        """
        y0 = jnp.array(1.0)
        dt = 0.01
        n_steps = 100

        def rhs_explicit(y):
            return -0.3 * y

        def rhs_implicit(y):
            return -0.7 * y

        def implicit_solve(rhs, gamma_dt):
            # Solve: Y - gamma_dt * (-0.7*Y) = rhs => Y(1 + 0.7*gamma_dt) = rhs
            return rhs / (1.0 + 0.7 * gamma_dt)

        y = y0
        for _ in range(n_steps):
            y = imex_ssp2_step(y, rhs_explicit, rhs_implicit, implicit_solve, dt)

        exact = float(jnp.exp(-1.0))
        np.testing.assert_allclose(float(y), exact, rtol=5e-2)

    @staticmethod
    def _rates(lam_e, lam_i):
        """Observed orders of y' = (lam_e + lam_i) y on [0, 1], N = 20..160."""
        exact = np.exp(lam_e + lam_i)
        errors = []
        for n in [20, 40, 80, 160]:
            dt = 1.0 / n
            y = jnp.array(1.0)
            for _ in range(n):
                y = imex_ssp2_step(
                    y,
                    lambda y: lam_e * y,
                    lambda y: lam_i * y,
                    lambda rhs, gdt: rhs / (1.0 - lam_i * gdt),
                    dt,
                )
            errors.append(abs(float(y) - exact))
        return np.log2(np.array(errors[:-1]) / np.array(errors[1:]))

    def test_mixed_convergence(self):
        """Second order on a mixed explicit + implicit split (issue #311)."""
        rates = self._rates(-0.4, -0.6)
        assert np.all(rates > 1.8), f"mixed rates {rates}"

    def test_implicit_only_convergence(self):
        """Second order when only the implicit part is active (issue #311)."""
        rates = self._rates(0.0, -1.0)
        assert np.all(rates >= 1.9), f"implicit-only rates {rates}"

    def test_explicit_only_convergence(self):
        """Second order when only the explicit part is active."""
        rates = self._rates(-1.0, 0.0)
        assert np.all(rates >= 1.9), f"explicit-only rates {rates}"

    def test_pytree_state(self):
        """IMEX works with tuple pytree states."""
        y0 = (jnp.array(1.0), jnp.array(2.0))
        dt = 0.01

        def rhs_explicit(y):
            return jax.tree.map(lambda x: -x, y)

        def rhs_implicit(_y):
            return jax.tree.map(jnp.zeros_like, _y)

        def implicit_solve(rhs, _gamma_dt):
            return rhs

        y = imex_ssp2_step(y0, rhs_explicit, rhs_implicit, implicit_solve, dt)
        assert isinstance(y, tuple)
        assert len(y) == 2

    def test_jit_compatible(self):
        """IMEX step compiles under jax.jit."""
        y0 = jnp.array(1.0)

        def rhs_e(y):
            return -y

        def rhs_i(_y):
            return jnp.array(0.0)

        def solve(rhs, _gdt):
            return rhs

        step = jax.jit(lambda y: imex_ssp2_step(y, rhs_e, rhs_i, solve, 0.01))
        y1 = step(y0)
        y1_ref = imex_ssp2_step(y0, rhs_e, rhs_i, solve, 0.01)
        np.testing.assert_allclose(y1, y1_ref, atol=1e-15)


# ---------------------------------------------------------------------------
# Split-explicit
# ---------------------------------------------------------------------------


class TestSplitExplicit:
    """Test split-explicit stepping on a two-timescale system."""

    def test_constant_fast_mode(self):
        """When fast RHS is zero, slow mode evolves with Euler step."""
        y_3d = jnp.array(1.0)
        y_2d = jnp.array(0.0)
        dt_slow = 0.1
        n_substeps = 10

        def rhs_3d(state_3d, _state_2d_avg):
            return -state_3d

        def rhs_2d(_t_sub, _state_2d, _state_3d):
            return jnp.array(0.0)

        def couple(state_3d, _state_2d_avg):
            return state_3d

        y_3d_new, y_2d_new = split_explicit_step(
            y_3d, y_2d, rhs_3d, rhs_2d, couple, dt_slow, n_substeps
        )
        # Should be a single Euler step: 1.0 + 0.1 * (-1.0) = 0.9
        np.testing.assert_allclose(float(y_3d_new), 0.9, atol=1e-10)
        np.testing.assert_allclose(float(y_2d_new), 0.0, atol=1e-10)

    def test_fast_mode_subcycling(self):
        """Fast mode gets subcycled N times."""
        y_3d = jnp.array(0.0)
        y_2d = jnp.array(1.0)
        dt_slow = 1.0
        n_substeps = 100

        def rhs_3d(_state_3d, _state_2d_avg):
            return jnp.array(0.0)

        def rhs_2d(_t_sub, state_2d, _state_3d):
            return -state_2d

        def couple(state_3d, _state_2d_avg):
            return state_3d

        _, y_2d_new = split_explicit_step(
            y_3d, y_2d, rhs_3d, rhs_2d, couple, dt_slow, n_substeps
        )
        # RK4 with h = dt_fast = 0.01 for 100 steps on dy/dt = -y: the
        # amplification per step is the degree-4 Taylor polynomial of e^{-h}
        h = 0.01
        r = 1.0 - h + h**2 / 2 - h**3 / 6 + h**4 / 24
        np.testing.assert_allclose(float(y_2d_new), r**100, rtol=1e-12)

    def test_coupling_modifies_slow_state(self):
        """Verify the coupling function actually modifies the slow state."""
        y_3d = jnp.array(1.0)
        y_2d = jnp.array(2.0)
        dt_slow = 0.1
        n_substeps = 10

        def rhs_3d(state_3d, _state_2d_avg):
            return jnp.array(0.0)  # No tendency

        def rhs_2d(_t_sub, _state_2d, _state_3d):
            return jnp.array(0.0)

        def couple(state_3d, state_2d_avg):
            # Add the 2D average to the 3D state (typical coupling pattern)
            return state_3d + 0.5 * state_2d_avg

        y_3d_new, _ = split_explicit_step(
            y_3d, y_2d, rhs_3d, rhs_2d, couple, dt_slow, n_substeps
        )
        # With zero RHS and no Euler change, y_3d stays 1.0 before coupling.
        # After coupling: 1.0 + 0.5 * 2.0 = 2.0
        np.testing.assert_allclose(float(y_3d_new), 2.0, atol=1e-10)

    def test_more_substeps_improves_fast_accuracy(self):
        """Increasing substeps should improve fast-mode accuracy."""
        y_3d = jnp.array(0.0)
        y_2d = jnp.array(1.0)
        dt_slow = 1.0
        exact = float(jnp.exp(-1.0))

        def rhs_3d(_s3, _s2):
            return jnp.array(0.0)

        def rhs_2d(_t, s2, _s3):
            return -s2

        def couple(s3, _s2):
            return s3

        errors = []
        for n_sub in [10, 100, 1000]:
            _, y_2d_new = split_explicit_step(
                y_3d, y_2d, rhs_3d, rhs_2d, couple, dt_slow, n_sub
            )
            errors.append(abs(float(y_2d_new) - exact))

        # Each 10x refinement should improve accuracy
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

    def test_jit_compatible(self):
        """Split-explicit step works inside jax.jit."""
        y_3d = jnp.array(1.0)
        y_2d = jnp.array(1.0)

        def rhs_3d(s3, _s2):
            return -s3

        def rhs_2d(_t, s2, _s3):
            return -s2

        def couple(s3, _s2):
            return s3

        step_jit = jax.jit(
            lambda s3, s2: split_explicit_step(s3, s2, rhs_3d, rhs_2d, couple, 0.1, 10)
        )
        y3, y2 = step_jit(y_3d, y_2d)
        assert jnp.isfinite(y3)
        assert jnp.isfinite(y2)


class TestSplitExplicitFastMode:
    """The fast subcycle on oscillatory modes, the cycle mean and the slow forcing (#312)."""

    @staticmethod
    def _run(step, n_steps):
        y3 = jnp.array([1.0, 0.0])
        y2 = jnp.array([1.0, 0.0])
        for n in range(n_steps):
            y3, y2 = step(n, y3, y2)
        return y3, y2

    def test_oscillatory_fast_mode_stays_bounded(self):
        """Fast rotation w_f = 20 with a slow rotation w_s = 1 coupled through the mean.

        dt = 0.05, 50 substeps, t = 10 (10^4 fast steps, w_f dt_fast = 0.02).
        Forward Euler grew the fast amplitude to 7.39; RK4 changes it by
        (w dt)^6 / 144 per substep, ~4e-9 in total (bound 1e-6 from #312).
        """
        wf, ws, dt, nsub = 20.0, 1.0, 0.05, 50

        def rhs_2d(_t, y2, _y3):
            return jnp.array([wf * y2[1], -wf * y2[0]])

        def rhs_3d(y3, y2_avg):
            # slow rotation plus a coupling to the fast mean
            return jnp.array([ws * y3[1], -ws * y3[0]]) + 1e-3 * y2_avg

        def couple(y3, _y2_avg):
            return y3

        step = jax.jit(
            lambda n, a, b: split_explicit_step(a, b, rhs_3d, rhs_2d, couple, dt, nsub)
        )
        y3, y2 = self._run(step, 200)
        assert abs(float(jnp.linalg.norm(y2)) - 1.0) < 1e-6
        assert bool(jnp.all(jnp.isfinite(y3)))

    def test_cycle_mean_is_trapezoidal(self):
        """y' = 1 from y = 0 over one step: the cycle mean is exactly 1/2."""
        captured = {}

        def rhs_3d(_y3, y2_avg):
            captured["avg"] = y2_avg
            return jnp.array(0.0)

        split_explicit_step(
            jnp.array(0.0),
            jnp.array(0.0),
            rhs_3d,
            lambda _t, _y2, _y3: jnp.array(1.0),
            lambda y3, _avg: y3,
            1.0,
            4,
        )
        np.testing.assert_allclose(float(captured["avg"]), 0.5, rtol=1e-14)

    def test_slow_forcing_is_added_once_per_cycle(self):
        """slow_forcing(state_3d) acts on the fast mode as a constant tendency."""
        calls = []

        def slow_forcing(y3):
            calls.append(1)
            return 2.0 * y3

        _, y2 = split_explicit_step(
            jnp.array(1.5),
            jnp.array(0.0),
            lambda _y3, _avg: jnp.array(0.0),
            lambda _t, _y2, _y3: jnp.array(0.0),
            lambda y3, _avg: y3,
            0.5,
            10,
            slow_forcing=slow_forcing,
        )
        # dy2/dt = 2 * 1.5 = 3 over dt_slow = 0.5
        np.testing.assert_allclose(float(y2), 1.5, rtol=1e-13)
        assert len(calls) == 1

    def test_stage_times(self):
        """rhs_2d sees the RK4 stage times: y' = t integrates exactly to t^2 / 2."""
        _, y2 = split_explicit_step(
            jnp.array(0.0),
            jnp.array(0.0),
            lambda _y3, _avg: jnp.array(0.0),
            lambda t, _y2, _y3: t,
            lambda y3, _avg: y3,
            2.0,
            3,
        )
        np.testing.assert_allclose(float(y2), 2.0, rtol=1e-13)


def test_split_explicit_float32_with_time_dependent_rhs():
    """float32 states stay float32 (x64 on) even when rhs_2d uses the stage time."""
    from finitevolx import SplitExplicitRKSolver

    y3 = jnp.ones(3, jnp.float32)
    y2 = jnp.ones(3, jnp.float32)
    a3, a2 = split_explicit_step(
        y3, y2, lambda a, b: -a, lambda t, b, a: -b * jnp.cos(t), lambda a, b: a, 0.1, 5
    )
    assert a3.dtype == jnp.float32 and a2.dtype == jnp.float32
    b3, b2 = SplitExplicitRKSolver(n_substeps=5).step(
        lambda t, a, b: -a, lambda t, b, a: -b * jnp.cos(t), 0.0, y3, y2, 0.1
    )
    assert b3.dtype == jnp.float32 and b2.dtype == jnp.float32
    np.testing.assert_allclose(b2, a2, rtol=1e-6)
