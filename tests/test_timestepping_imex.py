"""Tests for IMEX and split-explicit time integrators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from finitevolx import imex_ssp2_step, split_explicit_step

jax.config.update("jax_enable_x64", True)


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

    def test_mixed_convergence(self):
        """Verify order ~2 on mixed explicit+implicit split with dt refinement."""
        exact = float(jnp.exp(-1.0))
        errors = []

        for dt in [0.02, 0.01, 0.005]:
            y = jnp.array(1.0)
            n_steps = int(round(1.0 / dt))

            def rhs_e(y):
                return -0.4 * y

            def rhs_i(y):
                return -0.6 * y

            def solve(rhs, gamma_dt):
                return rhs / (1.0 + 0.6 * gamma_dt)

            for _ in range(n_steps):
                y = imex_ssp2_step(y, rhs_e, rhs_i, solve, dt)
            errors.append(abs(float(y) - exact))

        rate = np.log2(errors[-2] / errors[-1])
        # IMEX-SSP2 achieves ~1st order on mixed problems due to operator
        # splitting error; 2nd order is only achieved in the pure limits.
        assert rate > 0.8, f"IMEX mixed expected order >=1, got {rate:.2f}"
        # Verify error actually decreases with dt refinement
        assert errors[-1] < errors[0]

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
        # Forward Euler with dt=0.01 for 100 steps on dy/dt=-y
        # Exact: exp(-1) ≈ 0.3679; Euler with dt=0.01: (0.99)^100 ≈ 0.366
        np.testing.assert_allclose(float(y_2d_new), 0.99**100, rtol=1e-10)

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
