"""Pure functional split-explicit time integrator.

Separates fast barotropic dynamics (2D, free-surface gravity waves) from slow
baroclinic dynamics (3D, internal waves) and integrates them at different
timesteps.  The fast mode is subcycled with ``n_substeps`` Forward-Euler steps
per slow step, and the fast solution is time-averaged before coupling back to
the slow mode.

References
----------
- Shchepetkin & McWilliams (2005) — ROMS split-explicit scheme.
- Ducousso et al. (2021) — RK-based split-explicit time integration.
"""

from __future__ import annotations

from collections.abc import Callable

import jax


def split_explicit_step(
    state_3d,
    state_2d,
    rhs_3d: Callable,
    rhs_2d: Callable,
    couple_2d_to_3d: Callable,
    dt_slow: float,
    n_substeps: int,
):
    """Split-explicit barotropic/baroclinic time step.

    Algorithm::

        1. Subcycle 2D barotropic with n_substeps Forward-Euler steps
           (dt_fast = dt_slow / n_substeps), accumulating a time-average.
        2. Couple the time-averaged 2D state into the slow RHS.
        3. Advance the 3D baroclinic state with one Forward-Euler step
           using dt_slow.

    Parameters
    ----------
    state_3d : PyTree
        3D baroclinic state (slow mode).
    state_2d : PyTree
        2D barotropic state (fast mode).
    rhs_3d : Callable[[PyTree, PyTree], PyTree]
        Slow RHS: ``rhs_3d(state_3d, state_2d_avg) -> tendency_3d``.
    rhs_2d : Callable[[float, PyTree, PyTree], PyTree]
        Fast RHS: ``rhs_2d(t_sub, state_2d, state_3d) -> tendency_2d``.
        The first argument is the sub-step time offset from the beginning
        of the slow step.
    couple_2d_to_3d : Callable[[PyTree, PyTree], PyTree]
        Coupling function: ``couple_2d_to_3d(state_3d, state_2d_avg) ->
        state_3d_corrected``.  Applied after the slow step to ensure
        consistency between the 2D and 3D solutions.
    dt_slow : float
        Slow (baroclinic) timestep.
    n_substeps : int
        Number of fast (barotropic) substeps per slow step.

    Returns
    -------
    tuple[PyTree, PyTree]
        ``(new_state_3d, new_state_2d)`` after the split-explicit step.
    """
    dt_fast = dt_slow / n_substeps

    # --- Fast (barotropic) subcycling ---
    y_2d_sum = jax.tree.map(jax.numpy.zeros_like, state_2d)
    y_2d_curr = state_2d

    def _fast_body(carry, _):
        y_2d, y_2d_acc, substep = carry
        t_sub = substep * dt_fast
        f_fast = rhs_2d(t_sub, y_2d, state_3d)
        y_2d_new = jax.tree.map(lambda y, f: y + dt_fast * f, y_2d, f_fast)
        y_2d_acc_new = jax.tree.map(lambda s, y: s + y, y_2d_acc, y_2d_new)
        return (y_2d_new, y_2d_acc_new, substep + 1), None

    (y_2d_curr, y_2d_sum, _), _ = jax.lax.scan(
        _fast_body,
        (y_2d_curr, y_2d_sum, 0),
        None,
        length=n_substeps,
    )

    # Time-average the fast solution
    y_2d_avg = jax.tree.map(lambda s: s / n_substeps, y_2d_sum)

    # --- Slow (baroclinic) step ---
    f_slow = rhs_3d(state_3d, y_2d_avg)
    y_3d_new = jax.tree.map(lambda y, f: y + dt_slow * f, state_3d, f_slow)

    # --- Coupling ---
    y_3d_new = couple_2d_to_3d(y_3d_new, y_2d_avg)

    return y_3d_new, y_2d_curr
