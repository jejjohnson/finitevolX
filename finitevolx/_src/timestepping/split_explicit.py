"""Pure functional split-explicit time integrator.

Separates fast barotropic dynamics (2D, free-surface gravity waves) from slow
baroclinic dynamics (3D, internal waves) and integrates them at different
timesteps.  The fast mode is subcycled with ``n_substeps`` classical RK4 steps
per slow step (neutrally stable to round-off for the oscillatory gravity-wave
mode at ``omega * dt_fast`` well below ``2 sqrt 2``), the fast solution is
time-averaged over the cycle with the trapezoidal rule, and the slow mode
takes one Forward-Euler step with that average.

References
----------
- Shchepetkin & McWilliams (2005) — ROMS split-explicit scheme.
- Ducousso et al. (2021) — RK-based split-explicit time integration.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from finitevolx._src.timestepping.explicit_rk import rk4_step


def _barotropic_cycle(
    state_2d: PyTree,
    rhs_2d: Callable[[float, PyTree], PyTree],
    t0: float,
    dt_fast: float,
    n_substeps: int,
    forcing: PyTree | None = None,
) -> tuple[PyTree, PyTree]:
    """Subcycle the fast mode with RK4 and return its end state and mean.

    ``y_{k+1} = RK4(y_k)`` for ``k = 0 .. n-1`` on ``dy/dt = rhs_2d(t, y) + F``
    (``F = forcing`` held fixed over the cycle), and the trapezoidal mean

        <y> = (y_0 / 2 + y_1 + ... + y_{n-1} + y_n / 2) / n,

    which is exact for a state linear in time (the right-endpoint sum used
    before #312 was biased by ``dt_fast / 2``).

    Parameters
    ----------
    state_2d : PyTree
        Fast state ``y_0`` at ``t0``.
    rhs_2d : Callable[[float, PyTree], PyTree]
        Fast tendency ``rhs_2d(t, y)``.
    t0 : float
        Time at the start of the cycle (passed to ``rhs_2d`` at the RK4
        stage times ``t0 + k dt_fast + {0, 1/2, 1/2, 1} dt_fast``).
    dt_fast : float
        Fast timestep.
    n_substeps : int
        Number of fast steps (concrete, ``>= 1``).
    forcing : PyTree or None, optional
        Slow forcing added to every fast tendency, same structure as
        ``state_2d``.

    Returns
    -------
    tuple[PyTree, PyTree]
        ``(y_n, <y>)``: the fast state at ``t0 + n dt_fast`` and its
        trapezoidal time mean over the cycle.
    """
    if n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")

    def _rhs(s):
        # s = (y, t): time rides along as a state with dt/dt = 1, so RK4
        # evaluates rhs_2d at its exact stage times.
        y, t = s
        f = rhs_2d(t, y)
        if forcing is not None:
            f = jax.tree.map(jnp.add, f, forcing)
        return f, jnp.ones_like(t)

    def _body(carry, _):
        y, t, acc = carry
        y_new, t_new = rk4_step((y, t), _rhs, dt_fast)
        acc = jax.tree.map(jnp.add, acc, y_new)
        return (y_new, t_new, acc), None

    # time in the state's float dtype, so a float32 state stays float32 in the
    # scan carry even with x64 on (and a time-dependent rhs_2d cannot promote it)
    t_dtype = jnp.result_type(*jax.tree.leaves(state_2d), jnp.float32)
    t_start = jnp.asarray(t0, dtype=t_dtype)
    acc0 = jax.tree.map(lambda y: 0.5 * y, state_2d)  # y_0 / 2
    (y_end, _, acc), _ = jax.lax.scan(
        _body, (state_2d, t_start, acc0), None, length=n_substeps
    )
    # acc = y_0/2 + y_1 + ... + y_n  ->  subtract y_n/2, divide by n
    y_mean = jax.tree.map(lambda s, yn: (s - 0.5 * yn) / n_substeps, acc, y_end)
    return y_end, y_mean


def split_explicit_step(
    state_3d: PyTree,
    state_2d: PyTree,
    rhs_3d: Callable[[PyTree, PyTree], PyTree],
    rhs_2d: Callable[[float, PyTree, PyTree], PyTree],
    couple_2d_to_3d: Callable[[PyTree, PyTree], PyTree],
    dt_slow: float,
    n_substeps: int,
    slow_forcing: Callable[[PyTree], PyTree] | None = None,
    t0: float = 0.0,
) -> tuple[PyTree, PyTree]:
    """Split-explicit barotropic/baroclinic time step.

    Algorithm::

        1. F = slow_forcing(state_3d)  (if given; held fixed over the cycle)
        2. Subcycle the 2D barotropic state with n_substeps classical RK4
           steps (dt_fast = dt_slow / n_substeps) on
           d(y_2d)/dt = rhs_2d(t_sub, y_2d, state_3d) + F, and form the
           trapezoidal time mean <y_2d> over the cycle (y_0 included).
        3. Advance the 3D baroclinic state with one Forward-Euler step:
           y_3d + dt_slow * rhs_3d(state_3d, <y_2d>).
        4. couple_2d_to_3d(new y_3d, <y_2d>).

    The fast subcycle is neutrally stable for the oscillatory gravity-wave
    mode (RK4 amplification ``1 - (w dt)^6 / 144`` per substep); the
    Forward-Euler subcycle used before #312 grew by
    ``sqrt(1 + (w dt_fast)^2)`` per substep.  The slow step is first order
    in ``dt_slow``.

    **What the caller owns.**  The returned 2D state is the instantaneous
    end-of-cycle state ``y_n`` (at ``t + dt_slow``), not the mean, so it
    stays on time; the slow step used the mean.  Making the 3D state
    consistent with the averaged barotropic state (e.g. replacing the depth
    mean of the 3D velocity by the averaged barotropic transport) is the
    job of ``couple_2d_to_3d``, which receives that mean.

    Parameters
    ----------
    state_3d : PyTree
        3D baroclinic state (slow mode).
    state_2d : PyTree
        2D barotropic state (fast mode).
    rhs_3d : Callable[[PyTree, PyTree], PyTree]
        Slow RHS: ``rhs_3d(state_3d, state_2d_avg) -> tendency_3d``.
    rhs_2d : Callable[[float, PyTree, PyTree], PyTree]
        Fast RHS: ``rhs_2d(t_sub, state_2d, state_3d) -> tendency_2d``,
        with ``t_sub = t0 +`` the RK4 stage-time offset into the slow step.
    couple_2d_to_3d : Callable[[PyTree, PyTree], PyTree]
        Coupling function: ``couple_2d_to_3d(state_3d, state_2d_avg) ->
        state_3d_corrected``, applied after the slow step.
    dt_slow : float
        Slow (baroclinic) timestep.
    n_substeps : int
        Number of fast (barotropic) substeps per slow step.
    slow_forcing : Callable[[PyTree], PyTree] or None, optional
        ``slow_forcing(state_3d) -> forcing_2d``: the baroclinic forcing of
        the barotropic mode (same structure as ``state_2d``), evaluated once
        per slow step and added to every fast tendency.
    t0 : float, optional
        Time at the start of the slow step, passed on to ``rhs_2d``
        (default ``0.0``: ``t_sub`` is then the offset into the step).

    Returns
    -------
    tuple[PyTree, PyTree]
        ``(new_state_3d, new_state_2d)`` after the split-explicit step.
    """
    if n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps}")

    dt_fast = dt_slow / n_substeps
    forcing = None if slow_forcing is None else slow_forcing(state_3d)

    # --- Fast (barotropic) subcycling ---
    y_2d_end, y_2d_avg = _barotropic_cycle(
        state_2d,
        lambda t_sub, y_2d: rhs_2d(t_sub, y_2d, state_3d),
        t0,
        dt_fast,
        n_substeps,
        forcing,
    )

    # --- Slow (baroclinic) step ---
    f_slow = rhs_3d(state_3d, y_2d_avg)
    y_3d_new = jax.tree.map(lambda y, f: y + dt_slow * f, state_3d, f_slow)

    # --- Coupling ---
    y_3d_new = couple_2d_to_3d(y_3d_new, y_2d_avg)

    return y_3d_new, y_2d_end
