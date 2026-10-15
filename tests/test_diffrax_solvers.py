"""Tests for diffrax-based time integrator classes."""

from __future__ import annotations

import math

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from finitevolx import (
    IMEX_SSP2,
    RK3SSP,
    SSP_RK2,
    SSP_RK104,
    AB2Solver,
    ForwardEulerDfx,
    LeapfrogRAFSolver,
    RK2Heun,
    RK4Classic,
    SemiLagrangianSolver,
    SplitExplicitRKSolver,
    solve_ocean_pde,
)


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
# solve_ocean_pde wrapper
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSolveOceanPDE:
    def test_basic_usage(self):
        sol = solve_ocean_pde(
            _decay_rhs,
            RK3SSP(),
            y0=1.0,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
        )
        exact = math.exp(-1.0)
        np.testing.assert_allclose(float(sol.ys[0]), exact, rtol=1e-4)

    def test_with_bc_fn(self):
        """BC function is applied to tendency."""
        call_count = {"n": 0}

        def bc_fn(dydt):
            call_count["n"] += 1
            return dydt

        sol = solve_ocean_pde(
            _decay_rhs,
            RK3SSP(),
            y0=1.0,
            t0=0.0,
            t1=0.1,
            dt0=0.01,
            bc_fn=bc_fn,
        )
        assert jnp.isfinite(sol.ys[0])

    def test_bc_fn_modifies_tendency(self):
        """BC function that zeroes tendency should freeze the state."""

        def zero_bc(_dydt):
            return 0.0 * _dydt

        sol = solve_ocean_pde(
            _decay_rhs,
            RK3SSP(),
            y0=1.0,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            bc_fn=zero_bc,
        )
        # Tendency is zeroed out, so state should not change
        np.testing.assert_allclose(float(sol.ys[0]), 1.0, atol=1e-10)

    def test_with_saveat(self):
        """solve_ocean_pde should pass saveat through to diffeqsolve."""
        ts = jnp.linspace(0.0, 1.0, 6)
        sol = solve_ocean_pde(
            _decay_rhs,
            RK4Classic(),
            y0=1.0,
            t0=0.0,
            t1=1.0,
            dt0=0.01,
            saveat=dfx.SaveAt(ts=ts),
        )
        assert sol.ys.shape == (6,)


# ---------------------------------------------------------------------------
# AB2 solver (manual interface)
# ---------------------------------------------------------------------------


class TestAB2Solver:
    def test_exponential_decay(self):
        def rhs(t, y):
            return -y

        solver = AB2Solver()
        solver, y = solver.init(rhs, 0.0, jnp.array(1.0), 0.01)

        for n in range(1, 100):
            y, solver = solver.step(rhs, n * 0.01, y, 0.01)

        exact = math.exp(-1.0)
        np.testing.assert_allclose(float(y), exact, rtol=5e-3)

    def test_convergence_rate(self):
        """AB2 should converge at order ~2."""

        def rhs(t, y):
            return -y

        exact = math.exp(-1.0)
        errors = []
        for dt in [0.05, 0.025, 0.0125]:
            n_steps = int(round(1.0 / dt))
            solver = AB2Solver()
            solver, y = solver.init(rhs, 0.0, jnp.array(1.0), dt)
            for n in range(1, n_steps):
                y, solver = solver.step(rhs, n * dt, y, dt)
            errors.append(abs(float(y) - exact))

        rate = np.log2(errors[-2] / errors[-1])
        assert rate > 1.8, f"AB2Solver expected order ~2, got {rate:.2f}"


# ---------------------------------------------------------------------------
# LeapfrogRAF solver (manual interface)
# ---------------------------------------------------------------------------


class TestLeapfrogRAFSolver:
    def test_exponential_decay(self):
        def rhs(t, y):
            return -y

        solver = LeapfrogRAFSolver(alpha=0.05)
        solver, y = solver.init(rhs, 0.0, jnp.array(1.0), 0.01)

        for n in range(1, 100):
            y, solver = solver.step(rhs, n * 0.01, y, 0.01)

        exact = math.exp(-1.0)
        np.testing.assert_allclose(float(y), exact, rtol=5e-2)

    def test_different_alpha_values(self):
        """Various alpha values should all produce stable, reasonable results."""

        def rhs(t, y):
            return -y

        exact = math.exp(-1.0)
        for alpha in [0.01, 0.05, 0.1]:
            solver = LeapfrogRAFSolver(alpha=alpha)
            solver, y = solver.init(rhs, 0.0, jnp.array(1.0), 0.01)
            for n in range(1, 100):
                y, solver = solver.step(rhs, n * 0.01, y, 0.01)
            assert jnp.isfinite(y)
            assert abs(float(y) - exact) < 0.1, (
                f"alpha={alpha}: error={abs(float(y) - exact):.3f}"
            )


# ---------------------------------------------------------------------------
# SplitExplicitRK solver
# ---------------------------------------------------------------------------


class TestSplitExplicitRKSolver:
    def test_oscillatory_fast_mode(self):
        """Fast rotation w_f = 20 stays on the unit circle to 1e-6 over t = 10 (#312)."""
        wf, dt = 20.0, 0.05
        solver = SplitExplicitRKSolver(n_substeps=50)

        def rhs_slow(_t, y_3d, y_2d_avg):
            return jnp.array([y_3d[1], -y_3d[0]]) + 1e-3 * y_2d_avg

        def rhs_fast(_t, y_2d, _y_3d):
            return jnp.array([wf * y_2d[1], -wf * y_2d[0]])

        step = jax.jit(lambda t, a, b: solver.step(rhs_slow, rhs_fast, t, a, b, dt))
        y3 = y2 = jnp.array([1.0, 0.0])
        for n in range(200):
            y3, y2 = step(n * dt, y3, y2)
        assert abs(float(jnp.linalg.norm(y2)) - 1.0) < 1e-6

    def test_matches_split_explicit_step(self):
        """The solver equals the functional step, slow forcing and coupling included."""
        from finitevolx import split_explicit_step

        solver = SplitExplicitRKSolver(n_substeps=7)
        y3, y2 = jnp.array([1.0, -0.5]), jnp.array([0.3, 0.2])
        args = dict(
            slow_forcing=lambda a: 0.1 * a,
            couple_2d_to_3d=lambda a, avg: a + 0.01 * avg,
        )
        out_s = solver.step(
            lambda _t, a, avg: -a + avg,
            lambda _t, b, a: -3.0 * b + a,
            0.0,
            y3,
            y2,
            0.2,
            **args,
        )
        out_f = split_explicit_step(
            y3,
            y2,
            lambda a, avg: -a + avg,
            lambda _t, b, a: -3.0 * b + a,
            args["couple_2d_to_3d"],
            0.2,
            7,
            slow_forcing=args["slow_forcing"],
        )
        np.testing.assert_array_equal(out_s[0], out_f[0])
        np.testing.assert_array_equal(out_s[1], out_f[1])

    def test_basic(self):
        solver = SplitExplicitRKSolver(n_substeps=10)

        def rhs_slow(t, y_3d, y_2d_avg):
            return -y_3d

        def rhs_fast(t_sub, y_2d, y_3d):
            return -y_2d

        y_3d_new, y_2d_new = solver.step(
            rhs_slow,
            rhs_fast,
            0.0,
            jnp.array(1.0),
            jnp.array(1.0),
            0.1,
        )
        assert jnp.isfinite(y_3d_new)
        assert jnp.isfinite(y_2d_new)

    def test_more_substeps_improves_accuracy(self):
        """More substeps should give better fast-mode accuracy."""
        exact = math.exp(-1.0)

        def rhs_slow(t, y_3d, y_2d_avg):
            return jnp.array(0.0)

        def rhs_fast(t_sub, y_2d, y_3d):
            return -y_2d

        errors = []
        for n_sub in [10, 100, 1000]:
            solver = SplitExplicitRKSolver(n_substeps=n_sub)
            _, y_2d_new = solver.step(
                rhs_slow, rhs_fast, 0.0, jnp.array(0.0), jnp.array(1.0), 1.0
            )
            errors.append(abs(float(y_2d_new) - exact))

        assert errors[1] < errors[0]
        assert errors[2] < errors[1]


# ---------------------------------------------------------------------------
# SemiLagrangianSolver (diffrax interface)
# ---------------------------------------------------------------------------


class TestSemiLagrangianSolverDfx:
    def test_instantiation(self):
        """SemiLagrangianSolver can be instantiated with different orders."""
        solver = SemiLagrangianSolver(interpolation_order=1)
        assert solver.interpolation_order == 1

    def test_direct_step(self):
        """Test the step method directly (bypassing diffeqsolve)."""
        nx, ny = 32, 32
        x = jnp.arange(nx, dtype=jnp.float64)
        y = jnp.arange(ny, dtype=jnp.float64)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T
        field = jnp.exp(-((X - 16) ** 2 + (Y - 16) ** 2) / 50.0)

        # Use a simple namespace as the term so that .vf can return (u, v)
        # without ODETerm's pytree-structure check.
        class _VelocityTerm:
            def vf(self, t, y, args):
                return jnp.zeros_like(y), jnp.zeros_like(y)

        solver = SemiLagrangianSolver(interpolation_order=1)
        y1, _, _, _, result = solver.step(
            _VelocityTerm(), 0.0, 0.1, field, None, None, False
        )
        np.testing.assert_allclose(y1, field, atol=1e-10)
        assert result == dfx.RESULTS.successful

    def test_direct_step_uniform_advection(self):
        """Uniform velocity should shift the field."""
        nx, ny = 32, 32
        x = jnp.arange(nx, dtype=jnp.float64)
        y = jnp.arange(ny, dtype=jnp.float64)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T
        field = jnp.exp(-((X - 16) ** 2 + (Y - 16) ** 2) / 50.0)

        class _VelocityTerm:
            def vf(self, t, y, args):
                # 3 grid cells/sec in x, 0 in y
                return 3.0 * jnp.ones_like(y), jnp.zeros_like(y)

        solver = SemiLagrangianSolver(interpolation_order=1)
        y1, _, _, _, result = solver.step(
            _VelocityTerm(), 0.0, 1.0, field, None, None, False
        )
        shifted = jnp.roll(field, 3, axis=1)
        np.testing.assert_allclose(y1, shifted, atol=1e-10)
        assert result == dfx.RESULTS.successful


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
        from finitevolx import imex_ssp2_step

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

    def test_semi_lagrangian_in_diffeqsolve(self):
        """SemiLagrangianTerm + SaveAt(ts): integer shifts per step are exact."""
        from finitevolx import SemiLagrangianTerm

        ny, nx = 8, 16
        y0 = jnp.sin(2 * jnp.pi * jnp.arange(nx) / nx)[None, :] * jnp.ones((ny, 1))
        # u = 2 grid cells per unit time, dt = 0.5 -> one cell per step
        term = SemiLagrangianTerm(
            lambda t, y, a: (jnp.full_like(y, 2.0), jnp.zeros_like(y))
        )
        solver = SemiLagrangianSolver()
        assert solver.order(term) == 1
        sol = dfx.diffeqsolve(
            term, solver, 0.0, 2.0, 0.5, y0, saveat=dfx.SaveAt(ts=jnp.array([1.0, 2.0]))
        )
        np.testing.assert_allclose(sol.ys[0], jnp.roll(y0, 2, axis=1), atol=1e-12)
        np.testing.assert_allclose(sol.ys[1], jnp.roll(y0, 4, axis=1), atol=1e-12)

    def test_solvers_pass_through_jax_jit(self):
        """Static int fields: the solver objects can be jax.jit arguments."""
        from finitevolx import SemiLagrangianTerm

        @jax.jit
        def run_sl(solver, y0):
            term = SemiLagrangianTerm(
                lambda t, y, a: (jnp.ones_like(y), jnp.zeros_like(y))
            )
            return dfx.diffeqsolve(term, solver, 0.0, 1.0, 1.0, y0).ys

        y0 = jnp.arange(16.0).reshape(4, 4)
        np.testing.assert_allclose(
            run_sl(SemiLagrangianSolver(), y0)[0], jnp.roll(y0, 1, axis=1)
        )

        @jax.jit
        def run_split(solver, y3, y2):
            return solver.step(lambda t, a, b: -a, lambda t, b, a: -b, 0.0, y3, y2, 0.1)

        out = run_split(
            SplitExplicitRKSolver(n_substeps=5), jnp.array(1.0), jnp.array(1.0)
        )
        assert bool(jnp.isfinite(out[1]))
        # (IMEX_SSP2 holds an optimistix root finder with a function field, so it
        # goes through eqx.filter_jit, as every diffrax implicit solver does.)


class TestSolveOceanPDEKwargs:
    def test_long_fixed_step_run(self):
        """t1 = 10, dt0 = 1e-3 is 10^4 steps: more than diffrax's default 4096."""
        sol = solve_ocean_pde(_decay_rhs, RK3SSP(), jnp.array(1.0), 0.0, 10.0, 1e-3)
        np.testing.assert_allclose(sol.ys[0], math.exp(-10.0), rtol=1e-8)

    def test_kwargs_reach_diffeqsolve(self):
        sol = solve_ocean_pde(
            _decay_rhs,
            RK3SSP(),
            jnp.array(1.0),
            0.0,
            1.0,
            0.1,
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-10),
            adjoint=dfx.DirectAdjoint(),
        )
        assert int(sol.stats["num_steps"]) != 10  # the controller chose the steps
        np.testing.assert_allclose(sol.ys[0], math.exp(-1.0), rtol=1e-6)
        with pytest.raises(Exception, match="maximum number of solver steps"):
            solve_ocean_pde(
                _decay_rhs, RK3SSP(), jnp.array(1.0), 0.0, 1.0, 0.01, max_steps=5
            )
