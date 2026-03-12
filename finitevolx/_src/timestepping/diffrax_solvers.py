"""Diffrax-based time integrators for ocean PDEs.

Provides Butcher-tableau solvers that plug directly into
:func:`diffrax.diffeqsolve`, plus custom :class:`~equinox.Module` wrappers
for multistep and split-explicit schemes that do not fit the standard
Runge-Kutta framework.

Usage::

    import diffrax as dfx
    from finitevolx import RK3SSP

    solver = RK3SSP()
    sol = dfx.diffeqsolve(
        dfx.ODETerm(rhs_fn),
        solver,
        t0=0.0,
        t1=10.0,
        dt0=0.01,
        y0=state,
    )

References
----------
- Shu & Osher (1988) — SSP-RK3.
- Ketcheson (2008) — SSP-RK(10,4).
- Pareschi & Russo (2005) — IMEX-SSP2.
- Shchepetkin & McWilliams (2005) — Split-explicit.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any, ClassVar

import diffrax as dfx
from diffrax import RESULTS
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import numpy as np

# ---------------------------------------------------------------------------
# Explicit Runge-Kutta solvers (Butcher tableau)
# ---------------------------------------------------------------------------


class ForwardEulerDfx(dfx.AbstractSolver):
    """Forward Euler via the diffrax ``AbstractSolver`` interface.

    Order 1, 1 stage.  Included for completeness; prefer :class:`diffrax.Euler`
    for production use.
    """

    term_structure: ClassVar[Any] = dfx.AbstractTerm
    interpolation_cls: ClassVar[Any] = dfx.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        dt = t1 - t0
        f0 = terms.vf(t0, y0, args)
        y1 = jax.tree.map(lambda y, f: y + dt * f, y0, f0)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)


class RK2Heun(dfx.AbstractERK):
    """Heun's method (RK2) via Butcher tableau.

    Order 2, 2 stages, SSP with C = 1.

    Butcher tableau::

        0   |
        1   | 1
        ----+--------
            | 1/2  1/2
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        c=np.array([1.0]),
        b_sol=np.array([0.5, 0.5]),
        b_error=np.zeros(2),
        a_lower=(np.array([1.0]),),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )


class RK3SSP(dfx.AbstractERK):
    """3rd-order Strong-Stability-Preserving Runge-Kutta.

    Order 3, 3 stages, SSP coefficient C = 1 (optimal).  Preserves
    monotonicity, positivity, and TVD properties.

    Butcher tableau::

        0   |
        1   | 1
        1/2 | 1/4  1/4
        ----+---------------
            | 1/6  1/6  2/3
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        c=np.array([1.0, 0.5]),
        b_sol=np.array([1.0 / 6, 1.0 / 6, 2.0 / 3]),
        b_error=np.zeros(3),
        a_lower=(
            np.array([1.0]),
            np.array([0.25, 0.25]),
        ),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )


class RK4Classic(dfx.AbstractERK):
    """Classic 4th-order Runge-Kutta.

    Order 4, 4 stages, not SSP.

    Butcher tableau::

        0   |
        1/2 | 1/2
        1/2 | 0    1/2
        1   | 0    0    1
        ----+------------------
            | 1/6  1/3  1/3  1/6
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        c=np.array([0.5, 0.5, 1.0]),
        b_sol=np.array([1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6]),
        b_error=np.zeros(4),
        a_lower=(
            np.array([0.5]),
            np.array([0.0, 0.5]),
            np.array([0.0, 0.0, 1.0]),
        ),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )


class SSP_RK2(dfx.AbstractERK):
    """2nd-order SSP Runge-Kutta (same as Heun).

    Order 2, 2 stages, SSP coefficient C = 1.
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        c=np.array([1.0]),
        b_sol=np.array([0.5, 0.5]),
        b_error=np.zeros(2),
        a_lower=(np.array([1.0]),),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )


class SSP_RK104(dfx.AbstractERK):
    """4th-order SSP Runge-Kutta with 10 stages (Ketcheson 2008).

    Order 4, 10 stages, SSP coefficient C = 6.  Highest SSP coefficient
    achievable at 4th-order accuracy.

    Reference: Ketcheson (2008). Highly efficient strong stability-preserving
    Runge-Kutta methods with low-storage implementations.
    """

    tableau: ClassVar[dfx.ButcherTableau] = dfx.ButcherTableau(
        # c[i] = sum(a_lower[i]) for i = 0..8 (excludes first stage c=0)
        c=np.array(
            [
                1.0 / 6,
                1.0 / 3,
                1.0 / 2,
                2.0 / 3,
                1.0 / 3,
                1.0 / 2,
                2.0 / 3,
                5.0 / 6,
                1.0,
            ]
        ),
        b_sol=np.array([1.0 / 10] * 10),
        b_error=np.zeros(10),
        a_lower=(
            # Rows 1-4: all entries 1/6
            np.array([1.0 / 6]),
            np.array([1.0 / 6, 1.0 / 6]),
            np.array([1.0 / 6, 1.0 / 6, 1.0 / 6]),
            np.array([1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6]),
            # Rows 5-9: 1/15 for cols 0-4, then 1/6 for later cols
            # (convex combination at stage 6: 3/5*y0 + 2/5*y5 maps 1/6 -> 1/15)
            np.array([1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15]),
            np.array([1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6]),
            np.array(
                [1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 15, 1.0 / 6, 1.0 / 6]
            ),
            np.array(
                [
                    1.0 / 15,
                    1.0 / 15,
                    1.0 / 15,
                    1.0 / 15,
                    1.0 / 15,
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                ]
            ),
            np.array(
                [
                    1.0 / 15,
                    1.0 / 15,
                    1.0 / 15,
                    1.0 / 15,
                    1.0 / 15,
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                    1.0 / 6,
                ]
            ),
        ),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )


# ---------------------------------------------------------------------------
# IMEX solver (explicit + implicit Butcher tableaux)
# ---------------------------------------------------------------------------

_GAMMA = 1.0 - 1.0 / math.sqrt(2.0)


class IMEX_SSP2(dfx.AbstractRungeKutta, dfx.AbstractImplicitSolver):
    """IMEX-SSP2(2,2,2) solver using diffrax MultiTerm.

    Splits the ODE as ``dy/dt = f_E(t,y) + f_I(t,y)``.  The explicit part
    is SSP with C = 1; the implicit part is A-stable (SDIRK with
    gamma = 1 - 1/sqrt(2)).

    Usage::

        explicit_term = dfx.ODETerm(advection_rhs)
        implicit_term = dfx.ODETerm(diffusion_rhs)
        terms = dfx.MultiTerm(explicit_term, implicit_term)
        solver = IMEX_SSP2()
        sol = dfx.diffeqsolve(terms, solver, ...)

    Explicit tableau::

        0     |
        1     | 1
        ------+---------
              | 1/2  1/2

    Implicit tableau (SDIRK, gamma = 1 - 1/sqrt(2))::

        gamma     | gamma
        1         | 1-2*gamma  gamma
        ----------+-------------------
                  | 1/2        1/2
    """

    tableau: ClassVar[dfx.MultiButcherTableau] = dfx.MultiButcherTableau(
        # Explicit part
        dfx.ButcherTableau(
            c=np.array([1.0]),
            b_sol=np.array([0.5, 0.5]),
            b_error=np.zeros(2),
            a_lower=(np.array([1.0]),),
        ),
        # Implicit part (SDIRK)
        # c1 = gamma (first stage), c[0] = a_lower[0][0] + a_diagonal[1]
        #    = (1 - 2*gamma) + gamma = 1 - gamma
        dfx.ButcherTableau(
            c=np.array([1.0 - _GAMMA]),
            b_sol=np.array([0.5, 0.5]),
            b_error=np.zeros(2),
            a_lower=(np.array([1.0 - 2.0 * _GAMMA]),),
            a_diagonal=np.array([_GAMMA, _GAMMA]),
            a_predictor=(np.array([1.0]),),
            c1=_GAMMA,
        ),
    )
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )

    term_structure: ClassVar[Any] = dfx.MultiTerm[
        tuple[dfx.AbstractTerm, dfx.AbstractTerm]
    ]
    calculate_jacobian: ClassVar[Any] = dfx.CalculateJacobian.first_stage

    root_finder: Any = dfx.with_stepsize_controller_tols(dfx.VeryChord)()
    root_find_max_steps: int = 10


# ---------------------------------------------------------------------------
# Multistep solvers (eqx.Module — not directly usable with diffeqsolve)
# ---------------------------------------------------------------------------


class AB2Solver(eqx.Module):
    """Adams-Bashforth 2nd-order solver (equinox Module).

    Maintains ``f_prev`` as part of the solver state.  Not compatible with
    ``diffrax.diffeqsolve``; use the manual ``init`` / ``step`` interface.

    Usage::

        solver = AB2Solver()
        solver, y = solver.init(rhs_fn, t0, y0, dt)
        for n in range(n_steps):
            y, solver = solver.step(rhs_fn, t0 + n * dt, y, dt)
    """

    f_prev: PyTree | None = None

    def init(self, rhs_fn: Callable, t0: float, y0, dt: float):
        """Bootstrap with an RK2 step, returning ``(updated_solver, y1)``.

        Stores ``f_prev = f(t0, y0)`` (i.e. the RHS at the start of the
        bootstrap step) so that the first AB2 step uses the correct history.
        """
        k1 = rhs_fn(t0, y0)
        k2 = rhs_fn(t0 + dt, jax.tree.map(lambda y, f: y + dt * f, y0, k1))
        y1 = jax.tree.map(lambda y, f1, f2: y + 0.5 * dt * (f1 + f2), y0, k1, k2)
        return eqx.tree_at(lambda s: s.f_prev, self, k1), y1

    def step(self, rhs_fn: Callable, t: float, y, dt: float):
        """AB2 step: ``y_{n+1} = y_n + (dt/2)(3 f_n - f_{n-1})``."""
        f_curr = rhs_fn(t, y)
        y_next = jax.tree.map(
            lambda yi, fi, fi_1: yi + (dt / 2.0) * (3.0 * fi - fi_1),
            y,
            f_curr,
            self.f_prev,
        )
        new_solver = eqx.tree_at(lambda s: s.f_prev, self, f_curr)
        return y_next, new_solver


class LeapfrogRAFSolver(eqx.Module):
    """Leapfrog with Robert-Asselin filter (equinox Module).

    Three-level scheme: ``y_{n+1} = y_{n-1} + 2 dt f(y_n)``, with the RAF
    applied to the middle level to damp the computational mode.

    Usage::

        solver = LeapfrogRAFSolver(alpha=0.05)
        solver, y1 = solver.init(rhs_fn, t0, y0, dt)
        y_curr = y1
        for n in range(1, n_steps):
            y_curr, solver = solver.step(rhs_fn, t0 + n * dt, y_curr, dt)
    """

    alpha: float = 0.05
    y_prev: PyTree | None = None

    def init(self, rhs_fn: Callable, t0: float, y0, dt: float):
        """Bootstrap with an RK2 step, returning ``(updated_solver, y1)``."""
        k1 = rhs_fn(t0, y0)
        k2 = rhs_fn(t0 + dt, jax.tree.map(lambda y, f: y + dt * f, y0, k1))
        y1 = jax.tree.map(lambda y, f1, f2: y + 0.5 * dt * (f1 + f2), y0, k1, k2)
        return eqx.tree_at(lambda s: s.y_prev, self, y0), y1

    def step(self, rhs_fn: Callable, t: float, y_curr, dt: float):
        """Leapfrog + RAF step."""
        f_curr = rhs_fn(t, y_curr)

        # Leapfrog
        y_next = jax.tree.map(
            lambda yp, fc: yp + 2.0 * dt * fc,
            self.y_prev,
            f_curr,
        )

        # Robert-Asselin filter on the middle level
        y_curr_filtered = jax.tree.map(
            lambda yp, yc, yn: yc + self.alpha * (yp - 2.0 * yc + yn),
            self.y_prev,
            y_curr,
            y_next,
        )

        new_solver = eqx.tree_at(lambda s: s.y_prev, self, y_curr_filtered)
        return y_next, new_solver


# ---------------------------------------------------------------------------
# Split-explicit solver (eqx.Module)
# ---------------------------------------------------------------------------


class SplitExplicitRKSolver(eqx.Module):
    """Split-explicit barotropic/baroclinic solver.

    Uses Forward-Euler substeps for the fast (2D barotropic) mode and
    Forward-Euler for the slow (3D baroclinic) mode, with time-averaging
    of the barotropic solution.

    Parameters
    ----------
    n_substeps : int
        Number of barotropic substeps per baroclinic step.
    """

    n_substeps: int = 50

    def step(
        self,
        rhs_slow: Callable,
        rhs_fast: Callable,
        t: float,
        y_3d,
        y_2d,
        dt_slow: float,
    ):
        """Take one split-explicit step.

        Parameters
        ----------
        rhs_slow : Callable[[float, PyTree, PyTree], PyTree]
            Slow RHS: ``rhs_slow(t, y_3d, y_2d_avg) -> tendency_3d``.
        rhs_fast : Callable[[float, PyTree, PyTree], PyTree]
            Fast RHS: ``rhs_fast(t_sub, y_2d, y_3d) -> tendency_2d``.
        t : float
            Current time.
        y_3d, y_2d : PyTree
            3D (slow) and 2D (fast) states.
        dt_slow : float
            Slow timestep.

        Returns
        -------
        tuple[PyTree, PyTree]
            ``(new_y_3d, new_y_2d)``.
        """
        dt_fast = dt_slow / self.n_substeps

        def _fast_body(carry, _):
            y_2d_curr, y_2d_acc, substep = carry
            t_sub = t + substep * dt_fast
            f_fast = rhs_fast(t_sub, y_2d_curr, y_3d)
            y_2d_new = jax.tree.map(lambda y, f: y + dt_fast * f, y_2d_curr, f_fast)
            y_2d_acc_new = jax.tree.map(lambda s, y: s + y, y_2d_acc, y_2d_new)
            return (y_2d_new, y_2d_acc_new, substep + 1), None

        y_2d_sum = jax.tree.map(jnp.zeros_like, y_2d)
        (y_2d_curr, y_2d_sum, _), _ = jax.lax.scan(
            _fast_body,
            (y_2d, y_2d_sum, 0),
            None,
            length=self.n_substeps,
        )

        # Time-average
        y_2d_avg = jax.tree.map(lambda s: s / self.n_substeps, y_2d_sum)

        # Slow step
        f_slow = rhs_slow(t, y_3d, y_2d_avg)
        y_3d_new = jax.tree.map(lambda y, f: y + dt_slow * f, y_3d, f_slow)

        return y_3d_new, y_2d_curr


# ---------------------------------------------------------------------------
# Semi-Lagrangian solver (diffrax AbstractSolver)
# ---------------------------------------------------------------------------


class SemiLagrangianSolver(dfx.AbstractSolver):
    """Semi-Lagrangian advection solver for diffrax.

    Traces characteristic curves backward in time and interpolates the old
    field at departure points.  Unconditionally stable (CFL > 1 allowed).

    The ``terms.vf(t, y, args)`` must return ``(u, v)`` velocity components
    in **grid index units per second** (i.e. physical velocity divided by
    grid spacing).

    Parameters
    ----------
    interpolation_order : int
        0 = nearest-neighbour, 1 = linear (diffusive, monotone).
        JAX currently only supports ``order <= 1``.
    """

    term_structure: ClassVar[Any] = dfx.AbstractTerm
    interpolation_cls: ClassVar[Any] = (
        dfx.ThirdOrderHermitePolynomialInterpolation.from_k
    )
    interpolation_order: int = 1

    def order(self, terms):
        return self.interpolation_order

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        dt = t1 - t0

        # Velocity in grid-index units/second
        u, v = terms.vf(t0, y0, args)

        ny, nx = y0.shape
        y_coords, x_coords = jnp.meshgrid(
            jnp.arange(ny, dtype=y0.dtype),
            jnp.arange(nx, dtype=y0.dtype),
            indexing="ij",
        )

        x_dep = x_coords - u * dt
        y_dep = y_coords - v * dt

        y1 = jax.scipy.ndimage.map_coordinates(
            y0, [y_dep, x_dep], order=self.interpolation_order, mode="wrap"
        )

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
