"""Deprecated time steppers, kept for one release after the move to integrix.

The generic time steppers live in :mod:`integrix.march`. The eight
diffrax solvers and the nine pure step functions keep their names and
interfaces there, so ``finitevolx.<name>`` returns the integrix object itself.
``AB2Solver``, ``LeapfrogRAFSolver`` and ``SplitExplicitRKSolver`` have a new
interface in integrix (``integrix.march.AbstractStepper``s run by
``integrix.march.run``), so the classes below keep finitevolX's old call
shapes and delegate their arithmetic to the integrix steppers.
"""

from __future__ import annotations

from collections.abc import Callable
import warnings

import equinox as eqx
import integrix.march as ixm
from jaxtyping import PyTree

#: Names finitevolX re-exports unchanged from :mod:`integrix.march`.
MOVED: tuple[str, ...] = (
    "ForwardEulerDfx",
    "RK2Heun",
    "SSP_RK2",
    "RK3SSP",
    "RK4Classic",
    "SSP_RK104",
    "IMEX_SSP2",
    "AB3Solver",
    "euler_step",
    "heun_step",
    "rk3_ssp_step",
    "rk4_step",
    "imex_ssp2_step",
    "ab2_step",
    "ab3_step",
    "leapfrog_raf_step",
    "split_explicit_step",
)

_RUN = (
    "an integrix.march.AbstractStepper run by integrix.march.run(stepper, rhs,"
    " t0, y0, dt, n_steps) with rhs(t, y, args)"
)

#: Names whose integrix counterpart has a new interface, and that interface.
ADAPTED: dict[str, str] = {
    "AB2Solver": f"integrix.march.AB2Solver, {_RUN} and f_prev in the loop carry",
    "LeapfrogRAFSolver": (
        f"integrix.march.LeapfrogRAFSolver(alpha=...), {_RUN} and y_prev in"
        " the loop carry"
    ),
    "SplitExplicitRKSolver": (
        f"integrix.march.SplitExplicitRKSolver(n_substeps=...), {_RUN} where"
        " rhs is integrix.march.SplitExplicitTerms(slow, fast, slow_forcing,"
        " couple) and y = (y_3d, y_2d)"
    ),
}


def deprecated_stepper(name: str) -> object:
    """Return the object behind a deprecated ``finitevolx.<name>``, warning once per call site.

    Parameters
    ----------
    name : str
        One of :data:`MOVED` or :data:`ADAPTED`.

    Returns
    -------
    object
        ``integrix.march.<name>`` for a name in :data:`MOVED`; the adapter
        class of this module for a name in :data:`ADAPTED`.
    """
    if name in MOVED:
        message = (
            f"finitevolx.{name} is deprecated and will be removed in the next"
            f" release: it is integrix.march.{name}, so import it from there"
            f" (`from integrix.march import {name}`)."
        )
        obj = getattr(ixm, name)
    else:
        message = (
            f"finitevolx.{name} is deprecated and will be removed in the next"
            f" release: use {ADAPTED[name]}."
        )
        obj = globals()[name]
    # stacklevel=3: this function, then finitevolx.__getattr__, then the caller
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    return obj


def _with_args(rhs_fn: Callable) -> Callable:
    """``rhs_fn(t, y)`` as the diffrax vector field ``rhs(t, y, args)``."""
    return lambda t, y, args: rhs_fn(t, y)


class AB2Solver(eqx.Module):
    """Deprecated: use ``integrix.march.AB2Solver`` with ``integrix.march.run``.

    Adams-Bashforth 2nd order, ``y_{n+1} = y_n + (dt/2)(3 f_n - f_{n-1})``,
    with ``f_prev`` held in the module. The arithmetic is
    ``integrix.march.AB2Solver``'s.

    Usage::

        solver = AB2Solver()
        solver, y = solver.init(rhs_fn, t0, y0, dt)
        for n in range(1, n_steps):
            y, solver = solver.step(rhs_fn, t0 + n * dt, y, dt)
    """

    f_prev: PyTree | None = None

    def init(self, rhs_fn: Callable, t0: float, y0, dt: float):
        """Bootstrap with a Heun step, returning ``(updated_solver, y1)``.

        Stores ``f_prev = f(t0, y0)`` so that the first AB2 step uses the
        correct history.
        """
        f0, y1 = ixm.AB2Solver().init(_with_args(rhs_fn), t0, y0, dt, None)
        return eqx.tree_at(lambda s: s.f_prev, self, f0), y1

    def step(self, rhs_fn: Callable, t: float, y, dt: float):
        """AB2 step: ``y_{n+1} = y_n + (dt/2)(3 f_n - f_{n-1})``."""
        y_next, f_curr = ixm.AB2Solver().step(
            _with_args(rhs_fn), t, y, dt, self.f_prev, None
        )
        return y_next, eqx.tree_at(lambda s: s.f_prev, self, f_curr)


class LeapfrogRAFSolver(eqx.Module):
    """Deprecated: use ``integrix.march.LeapfrogRAFSolver`` with ``integrix.march.run``.

    Leapfrog with a Robert-Asselin filter on the middle level, with the
    filtered ``y_prev`` held in the module. The arithmetic is
    ``integrix.march.LeapfrogRAFSolver``'s: first order for ``alpha > 0``,
    second order for ``alpha = 0``.

    Usage::

        solver = LeapfrogRAFSolver(alpha=0.05)
        solver, y = solver.init(rhs_fn, t0, y0, dt)
        for n in range(1, n_steps):
            y, solver = solver.step(rhs_fn, t0 + n * dt, y, dt)
    """

    alpha: float = 0.05
    y_prev: PyTree | None = None

    def init(self, rhs_fn: Callable, t0: float, y0, dt: float):
        """Bootstrap with a Heun step, returning ``(updated_solver, y1)``."""
        y_prev, y1 = ixm.LeapfrogRAFSolver(alpha=self.alpha).init(
            _with_args(rhs_fn), t0, y0, dt, None
        )
        return eqx.tree_at(lambda s: s.y_prev, self, y_prev), y1

    def step(self, rhs_fn: Callable, t: float, y_curr, dt: float):
        """Leapfrog + RAF step, returning ``(y_next, updated_solver)``."""
        y_next, y_filtered = ixm.LeapfrogRAFSolver(alpha=self.alpha).step(
            _with_args(rhs_fn), t, y_curr, dt, self.y_prev, None
        )
        return y_next, eqx.tree_at(lambda s: s.y_prev, self, y_filtered)


class SplitExplicitRKSolver(eqx.Module):
    """Deprecated: use ``integrix.march.SplitExplicitRKSolver`` with ``integrix.march.run``.

    Split-explicit barotropic/baroclinic step with finitevolX's old call
    shape. The arithmetic is ``integrix.march.SplitExplicitRKSolver``'s: RK4
    substeps of the fast (2D) mode, its trapezoidal time mean over the cycle,
    and one Forward-Euler step of the slow (3D) mode with that mean.

    Parameters
    ----------
    n_substeps : int
        Number of barotropic substeps per baroclinic step.
    """

    n_substeps: int = eqx.field(static=True, default=50)

    def step(
        self,
        rhs_slow: Callable,
        rhs_fast: Callable,
        t: float,
        y_3d,
        y_2d,
        dt_slow: float,
        slow_forcing: Callable | None = None,
        couple_2d_to_3d: Callable | None = None,
    ):
        """Take one split-explicit step.

        Parameters
        ----------
        rhs_slow : Callable[[float, PyTree, PyTree], PyTree]
            Slow RHS: ``rhs_slow(t, y_3d, y_2d_avg) -> tendency_3d``.
        rhs_fast : Callable[[float, PyTree, PyTree], PyTree]
            Fast RHS: ``rhs_fast(t_sub, y_2d, y_3d) -> tendency_2d``, with
            ``t_sub`` the absolute time of the RK4 stage.
        t : float
            Current time.
        y_3d, y_2d : PyTree
            3D (slow) and 2D (fast) states.
        dt_slow : float
            Slow timestep.
        slow_forcing : Callable[[PyTree], PyTree] or None, optional
            ``slow_forcing(y_3d) -> forcing_2d``, evaluated once and added to
            every fast tendency.
        couple_2d_to_3d : Callable[[PyTree, PyTree], PyTree] or None, optional
            ``couple_2d_to_3d(new_y_3d, y_2d_avg) -> corrected_y_3d``,
            applied after the slow step.

        Returns
        -------
        tuple[PyTree, PyTree]
            ``(new_y_3d, new_y_2d)``; ``new_y_2d`` is the end-of-cycle fast
            state at ``t + dt_slow``.
        """
        terms = ixm.SplitExplicitTerms(
            slow=lambda t_, y3, mean_2d, args: rhs_slow(t_, y3, mean_2d),
            fast=lambda t_, y2, y3, args: rhs_fast(t_, y2, y3),
            slow_forcing=(
                None
                if slow_forcing is None
                else (lambda t_, y3, args: slow_forcing(y3))
            ),
            couple=(
                None
                if couple_2d_to_3d is None
                else (lambda y3, mean_2d, args: couple_2d_to_3d(y3, mean_2d))
            ),
        )
        stepper = ixm.SplitExplicitRKSolver(n_substeps=self.n_substeps)
        (new_3d, new_2d), _ = stepper.step(terms, t, (y_3d, y_2d), dt_slow, None, None)
        return new_3d, new_2d
