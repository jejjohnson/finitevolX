"""Tests for diffrax-based time integrator classes."""

from __future__ import annotations

import itertools
import math

import diffrax as dfx
from integrix.march import (
    IMEX_SSP2,
    RK3SSP,
    SSP_RK2,
    SSP_RK104,
    AB3Solver,
    ForwardEulerDfx,
    RK2Heun,
    RK4Classic,
)
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _decay_rhs(t, y, args):
    """dy/dt = -y."""
    return -y


# ---------------------------------------------------------------------------
# Butcher-tableau RK solvers via diffeqsolve
# ---------------------------------------------------------------------------


class TestDiffraxExplicitRK:
    """Verify Butcher-tableau solvers produce correct results via diffeqsolve."""

    @pytest.mark.parametrize(
        ("solver_cls", "expected_order"),
        [
            (ForwardEulerDfx, 1),
            (RK2Heun, 2),
            (SSP_RK2, 2),
            (RK3SSP, 3),
            (RK4Classic, 4),
        ],
        ids=["euler", "heun", "ssp_rk2", "rk3_ssp", "rk4"],
    )
    def test_exponential_decay(self, solver_cls, expected_order):
        solver = solver_cls()
        sol = dfx.diffeqsolve(
            dfx.ODETerm(_decay_rhs),
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=1.0,
            saveat=dfx.SaveAt(t1=True),
        )
        exact = math.exp(-1.0)
        error = abs(float(sol.ys[0]) - exact)
        # All solvers with dt=0.01 should be within 1e-2 at worst (Euler)
        assert error < 0.1, f"Error {error:.2e} too large for {solver_cls.__name__}"


class TestDiffraxConvergence:
    """Check convergence rate by halving dt."""

    @pytest.mark.parametrize(
        ("solver_cls", "expected_order"),
        [
            (RK2Heun, 2),
            (RK3SSP, 3),
            (RK4Classic, 4),
        ],
        ids=["heun", "rk3_ssp", "rk4"],
    )
    @pytest.mark.slow
    def test_convergence_order(self, solver_cls, expected_order):
        solver = solver_cls()
        exact = math.exp(-1.0)
        errors = []
        for dt in [0.5, 0.25, 0.125]:
            sol = dfx.diffeqsolve(
                dfx.ODETerm(_decay_rhs),
                solver,
                t0=0.0,
                t1=1.0,
                dt0=dt,
                y0=1.0,
                saveat=dfx.SaveAt(t1=True),
            )
            errors.append(abs(float(sol.ys[0]) - exact))
        rate = np.log2(errors[-2] / errors[-1])
        assert rate > expected_order - 0.15, (
            f"Expected order ~{expected_order}, got {rate:.2f}"
        )


class TestDiffraxAbsoluteAccuracy:
    """Verify the actual error magnitude, not just convergence order."""

    @pytest.mark.parametrize(
        ("solver_cls", "rtol"),
        [
            (RK2Heun, 1e-4),
            (RK3SSP, 1e-6),
            (RK4Classic, 5e-8),
        ],
        ids=["heun", "rk3_ssp", "rk4"],
    )
    def test_accuracy_at_dt_001(self, solver_cls, rtol):
        solver = solver_cls()
        sol = dfx.diffeqsolve(
            dfx.ODETerm(_decay_rhs),
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=1.0,
            saveat=dfx.SaveAt(t1=True),
        )
        exact = math.exp(-1.0)
        np.testing.assert_allclose(float(sol.ys[0]), exact, rtol=rtol)


# ---------------------------------------------------------------------------
# SSP_RK104 (10-stage, 4th-order)
# ---------------------------------------------------------------------------


class TestSSP_RK104:
    """Verify the 10-stage SSP-RK4 solver works via diffeqsolve."""

    def test_exponential_decay(self):
        solver = SSP_RK104()
        sol = dfx.diffeqsolve(
            dfx.ODETerm(_decay_rhs),
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=1.0,
            saveat=dfx.SaveAt(t1=True),
        )
        exact = math.exp(-1.0)
        np.testing.assert_allclose(float(sol.ys[0]), exact, rtol=1e-7)

    @pytest.mark.slow
    def test_convergence_order(self):
        """SSP_RK104 should converge at 4th order."""
        solver = SSP_RK104()
        exact = math.exp(-1.0)
        errors = []
        for dt in [0.5, 0.25, 0.125]:
            sol = dfx.diffeqsolve(
                dfx.ODETerm(_decay_rhs),
                solver,
                t0=0.0,
                t1=1.0,
                dt0=dt,
                y0=1.0,
                saveat=dfx.SaveAt(t1=True),
            )
            errors.append(abs(float(sol.ys[0]) - exact))
        rate = np.log2(errors[-2] / errors[-1])
        assert rate > 3.8, f"SSP_RK104 expected order ~4, got {rate:.2f}"


# ---------------------------------------------------------------------------
# IMEX_SSP2 diffrax solver
# ---------------------------------------------------------------------------


class TestIMEX_SSP2_Diffrax:
    """Verify IMEX_SSP2 tableau definition is correct."""

    def test_instantiation(self):
        """IMEX_SSP2 can be instantiated with correct tableau structure."""
        solver = IMEX_SSP2()
        tab = solver.tableau
        assert isinstance(tab, dfx.MultiButcherTableau)

    def test_explicit_tableau_weights_sum_to_one(self):
        """Explicit b weights sum to 1."""
        tab = IMEX_SSP2.tableau.tableaus[0]
        np.testing.assert_allclose(float(jnp.sum(tab.b_sol)), 1.0, atol=1e-14)

    def test_implicit_tableau_weights_sum_to_one(self):
        """Implicit b weights sum to 1."""
        tab = IMEX_SSP2.tableau.tableaus[1]
        np.testing.assert_allclose(float(jnp.sum(tab.b_sol)), 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# SaveAt integration
# ---------------------------------------------------------------------------


class TestSaveAt:
    @pytest.mark.slow
    def test_save_multiple_times(self):
        solver = RK3SSP()
        ts = jnp.linspace(0.0, 1.0, 11)
        sol = dfx.diffeqsolve(
            dfx.ODETerm(_decay_rhs),
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=1.0,
            saveat=dfx.SaveAt(ts=ts),
        )
        assert sol.ys.shape == (11,)
        # Check monotonic decay
        assert jnp.all(jnp.diff(sol.ys) < 0)

    def test_saved_values_match_exact(self):
        """Saved values should be close to exp(-t) at each saved time."""
        solver = RK4Classic()
        ts = jnp.linspace(0.0, 1.0, 11)
        sol = dfx.diffeqsolve(
            dfx.ODETerm(_decay_rhs),
            solver,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            y0=1.0,
            saveat=dfx.SaveAt(ts=ts),
        )
        exact = jnp.exp(-ts)
        np.testing.assert_allclose(sol.ys, exact, rtol=1e-7)


# ---------------------------------------------------------------------------
# Butcher tableau consistency checks
# ---------------------------------------------------------------------------


class TestButcherTableauConsistency:
    """Verify Butcher tableaux satisfy basic consistency conditions."""

    @pytest.mark.parametrize(
        "solver_cls",
        [RK2Heun, RK3SSP, RK4Classic, SSP_RK2],
        ids=["heun", "rk3_ssp", "rk4", "ssp_rk2"],
    )
    def test_b_sol_sums_to_one(self, solver_cls):
        """Weights b must sum to 1 for consistency."""
        tab = solver_cls.tableau
        np.testing.assert_allclose(float(jnp.sum(tab.b_sol)), 1.0, atol=1e-14)

    @pytest.mark.parametrize(
        "solver_cls",
        [RK2Heun, RK3SSP, RK4Classic, SSP_RK2],
        ids=["heun", "rk3_ssp", "rk4", "ssp_rk2"],
    )
    def test_c_equals_row_sums_of_a(self, solver_cls):
        """For explicit RK: c_i = sum(a_{i,j}) for all i > 0."""
        tab = solver_cls.tableau
        # c has length num_stages - 1 (excludes first stage c=0)
        for i, a_row in enumerate(tab.a_lower):
            row_sum = float(jnp.sum(a_row))
            np.testing.assert_allclose(
                float(tab.c[i]),
                row_sum,
                atol=1e-14,
                err_msg=f"c[{i}]={float(tab.c[i])} != row_sum={row_sum}",
            )


# ---------------------------------------------------------------------------
# diffrax protocol: every solver inside diffeqsolve (#313)
# ---------------------------------------------------------------------------


def _imex_terms():
    # dy/dt = -y (explicit) - 10 y (implicit)
    return dfx.MultiTerm(
        dfx.ODETerm(lambda t, y, a: -y), dfx.ODETerm(lambda t, y, a: -10.0 * y)
    )


def _imex_fixed():
    return IMEX_SSP2(root_finder=dfx.VeryChord(rtol=1e-12, atol=1e-12))


class TestDiffeqsolveProtocol:
    @pytest.mark.parametrize(
        "solver_cls", [ForwardEulerDfx, RK2Heun, SSP_RK2, RK3SSP, RK4Classic, SSP_RK104]
    )
    def test_erk_stepto_with_dense_output(self, solver_cls):
        ts = jnp.linspace(0.0, 1.0, 11)
        sol = dfx.diffeqsolve(
            dfx.ODETerm(_decay_rhs),
            solver_cls(),
            0.0,
            1.0,
            None,
            jnp.array(1.0),
            stepsize_controller=dfx.StepTo(ts=ts),
            saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 1.0, 4)),
        )
        # dt = 0.1, saved times between steps: ~3x the measured relative error
        # (Euler 5.2e-2, Heun / SSP_RK2 1.8e-3, RK3SSP 7.2e-4 from the cubic
        # Hermite dense output, RK4 1e-6, SSP_RK104 2e-7)
        rtol = {
            ForwardEulerDfx: 0.15,
            RK2Heun: 5e-3,
            SSP_RK2: 5e-3,
            RK3SSP: 2e-3,
            RK4Classic: 3e-6,
            SSP_RK104: 1e-6,
        }[solver_cls]
        np.testing.assert_allclose(
            sol.ys, jnp.exp(-jnp.linspace(0.0, 1.0, 4)), rtol=rtol
        )

    @pytest.mark.parametrize("solver_cls", [RK2Heun, SSP_RK2, RK3SSP])
    def test_erk_order_matches_tableau(self, solver_cls):
        assert (
            solver_cls().order(None) == {RK2Heun: 2, SSP_RK2: 2, RK3SSP: 3}[solver_cls]
        )

    def test_rk3ssp_pid_stiff_decay(self):
        """y' = -50 y on [0, 10] with PID(rtol=1e-6): finite and within 1e-5 of exact.

        With the old zero b_error this took 5 steps and returned inf.
        """
        sol = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, a: -50.0 * y),
            RK3SSP(),
            0.0,
            10.0,
            0.01,
            jnp.array(1.0),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-12),
            max_steps=100_000,
        )
        assert bool(jnp.isfinite(sol.ys[0]))
        assert abs(float(sol.ys[0]) - math.exp(-500.0)) < 1e-5

    @pytest.mark.parametrize("solver_cls", [RK2Heun, SSP_RK2, RK3SSP])
    def test_adaptive_erk_accuracy(self, solver_cls):
        sol = dfx.diffeqsolve(
            dfx.ODETerm(_decay_rhs),
            solver_cls(),
            0.0,
            1.0,
            0.1,
            jnp.array(1.0),
            stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-9),
        )
        # local-error control at rtol 1e-7: global error well under 1e-5
        assert abs(float(sol.ys[0]) - math.exp(-1.0)) < 1e-5

    @pytest.mark.parametrize("solver", [ForwardEulerDfx(), RK4Classic(), SSP_RK104()])
    def test_fixed_step_solvers_refuse_adaptive(self, solver):
        with pytest.raises(Exception, match="error estimates"):
            dfx.diffeqsolve(
                dfx.ODETerm(_decay_rhs),
                solver,
                0.0,
                1.0,
                0.1,
                jnp.array(1.0),
                stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8),
            )

    @pytest.mark.slow
    def test_imex_stepto_matches_functional_step(self):
        """IMEX_SSP2 in diffeqsolve equals imex_ssp2_step (exact implicit solve)."""
        from integrix.march import imex_ssp2_step

        dt, n = 0.01, 100
        sol = dfx.diffeqsolve(
            _imex_terms(),
            _imex_fixed(),
            0.0,
            n * dt,
            dt,
            jnp.array(1.0),
            saveat=dfx.SaveAt(ts=jnp.linspace(0.0, n * dt, 5)),
        )
        y = jnp.array(1.0)
        for _ in range(n):
            y = imex_ssp2_step(
                y,
                lambda y: -y,
                lambda y: -10.0 * y,
                lambda r, gdt: r / (1 + 10 * gdt),
                dt,
            )
        # the linear implicit stage converges in one chord iteration: round-off only
        np.testing.assert_allclose(sol.ys[-1], y, rtol=1e-12)

    @pytest.mark.slow
    def test_imex_pid(self):
        assert IMEX_SSP2().order(None) == 2
        sol = dfx.diffeqsolve(
            _imex_terms(),
            IMEX_SSP2(),
            0.0,
            1.0,
            0.01,
            jnp.array(1.0),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8),
        )
        # embedded first-order estimate at rtol 1e-6: ~6e-5 relative global error
        np.testing.assert_allclose(sol.ys[0], math.exp(-11.0), rtol=1e-3)

    def test_solvers_pass_through_jax_jit(self):
        """Static int fields: the solver objects can be jax.jit arguments."""

        @jax.jit
        def run(solver, y0):
            sol = dfx.diffeqsolve(dfx.ODETerm(_decay_rhs), solver, 0.0, 0.1, 0.05, y0)
            return sol.ys[0]

        out = run(RK3SSP(), jnp.array(1.0))
        assert bool(jnp.isfinite(out))
        # (IMEX_SSP2 holds an optimistix root finder with a function field, so it
        # goes through eqx.filter_jit, as every diffrax implicit solver does.)


class TestAB3Solver:
    """AB3 with an RK3 start inside diffeqsolve (#314)."""

    @staticmethod
    def _error(dt):
        sol = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, a: -1.5 * y),
            AB3Solver(),
            0.0,
            1.0,
            None,
            jnp.array(1.0),
            stepsize_controller=dfx.StepTo(
                ts=jnp.linspace(0.0, 1.0, int(round(1 / dt)) + 1)
            ),
        )
        return abs(float(sol.ys[0]) - math.exp(-1.5))

    @pytest.mark.slow
    def test_convergence_rate_stepto(self):
        errs = [self._error(dt) for dt in (0.05, 0.025, 0.0125)]
        rates = [math.log2(a / b) for a, b in itertools.pairwise(errs)]
        assert all(r >= 2.9 for r in rates), rates

    def test_matches_rk3_bootstrap_then_ab3_step(self):
        from integrix.march import ab3_step, rk3_ssp_step

        f = lambda y: -1.5 * y
        dt = 0.1
        y1 = rk3_ssp_step(jnp.array(1.0), f, dt)
        y2 = rk3_ssp_step(y1, f, dt)
        y3, _, _ = ab3_step(y2, f, dt, f(y1), f(jnp.array(1.0)))
        sol = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, a: f(y)),
            AB3Solver(),
            0.0,
            0.3,
            dt,
            jnp.array(1.0),
            saveat=dfx.SaveAt(steps=True),
        )
        np.testing.assert_allclose(sol.ys[:3], jnp.stack([y1, y2, y3]), rtol=1e-14)

    def test_refuses_adaptive(self):
        with pytest.raises(Exception, match="error estimates"):
            dfx.diffeqsolve(
                dfx.ODETerm(_decay_rhs),
                AB3Solver(),
                0.0,
                1.0,
                0.1,
                jnp.array(1.0),
                stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8),
            )
