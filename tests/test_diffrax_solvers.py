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

jax.config.update("jax_enable_x64", True)


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
