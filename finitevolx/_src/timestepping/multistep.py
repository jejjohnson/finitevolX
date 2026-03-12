"""Pure functional multistep and leapfrog time integrators.

These methods require history from previous timesteps, so they return updated
history alongside the new state.  All implementations are pure (no mutation),
pytree-aware via :func:`jax.tree.map`, and fully :func:`jax.jit`-compatible.

References
----------
- Durran (2010) — Numerical Methods for Fluid Dynamics.
- Robert (1966) — The Robert-Asselin time filter.
"""

from __future__ import annotations

from collections.abc import Callable

import jax


def ab2_step(state, rhs_fn: Callable, dt: float, rhs_nm1):
    """2nd-order Adams-Bashforth.

    .. math::

        y_{n+1} = y_n + (dt/2)(3 f_n - f_{n-1})

    Only one RHS evaluation per step (efficiency advantage over RK2).
    Requires one previous RHS evaluation ``rhs_nm1 = f(y_{n-1})``.

    Parameters
    ----------
    state : PyTree
        Current state y_n.
    rhs_fn : Callable[[PyTree], PyTree]
        Right-hand-side function ``f(state) -> tendency``.
    dt : float
        Timestep.
    rhs_nm1 : PyTree
        RHS evaluated at the previous step, ``f(y_{n-1})``.

    Returns
    -------
    tuple[PyTree, PyTree, PyTree]
        ``(new_state, rhs_n, rhs_nm1)`` — the caller must thread ``rhs_n``
        and ``rhs_nm1`` into the next call (shifted by one level).
    """
    rhs_n = rhs_fn(state)
    new_state = jax.tree.map(
        lambda y, fn, fnm1: y + (dt / 2.0) * (3.0 * fn - fnm1),
        state,
        rhs_n,
        rhs_nm1,
    )
    return new_state, rhs_n, rhs_nm1


def ab3_step(state, rhs_fn: Callable, dt: float, rhs_nm1, rhs_nm2):
    """3rd-order Adams-Bashforth.

    .. math::

        y_{n+1} = y_n + (dt/12)(23 f_n - 16 f_{n-1} + 5 f_{n-2})

    One RHS evaluation per step.  Requires two previous RHS evaluations.

    Parameters
    ----------
    state : PyTree
        Current state y_n.
    rhs_fn : Callable[[PyTree], PyTree]
        Right-hand-side function.
    dt : float
        Timestep.
    rhs_nm1 : PyTree
        RHS at step n-1.
    rhs_nm2 : PyTree
        RHS at step n-2.

    Returns
    -------
    tuple[PyTree, PyTree, PyTree]
        ``(new_state, rhs_n, rhs_nm1)`` — thread ``rhs_n`` as the new
        ``rhs_nm1`` and the old ``rhs_nm1`` as the new ``rhs_nm2`` in
        subsequent calls.
    """
    rhs_n = rhs_fn(state)
    new_state = jax.tree.map(
        lambda y, fn, fnm1, fnm2: (
            y + (dt / 12.0) * (23.0 * fn - 16.0 * fnm1 + 5.0 * fnm2)
        ),
        state,
        rhs_n,
        rhs_nm1,
        rhs_nm2,
    )
    return new_state, rhs_n, rhs_nm1


def leapfrog_raf_step(
    state,
    state_nm1,
    rhs_fn: Callable,
    dt: float,
    alpha: float = 0.05,
):
    """Leapfrog with Robert-Asselin filter.

    .. math::

        \\tilde{y}_{n+1} &= y_{n-1} + 2 \\Delta t \\, f(y_n)
        \\bar{y}_n &= y_n + \\alpha (y_{n-1} - 2 y_n + \\tilde{y}_{n+1})

    The filtered middle value ``bar{y}_n`` damps the spurious computational
    mode inherent to the three-level leapfrog scheme.

    Parameters
    ----------
    state : PyTree
        Current state y_n.
    state_nm1 : PyTree
        Previous state y_{n-1}.
    rhs_fn : Callable[[PyTree], PyTree]
        Right-hand-side function.
    dt : float
        Timestep.
    alpha : float, optional
        Robert-Asselin filter coefficient (default 0.05).  Typical range
        0.01-0.1; larger values damp the computational mode more aggressively
        but introduce additional dissipation.

    Returns
    -------
    tuple[PyTree, PyTree]
        ``(y_{n+1}, filtered_y_n)`` — use ``y_{n+1}`` as the new current
        state and ``filtered_y_n`` as the new ``state_nm1`` at the next step.
    """
    rhs_n = rhs_fn(state)

    # Leapfrog step
    y_next = jax.tree.map(
        lambda ynm1, fn: ynm1 + 2.0 * dt * fn,
        state_nm1,
        rhs_n,
    )

    # Robert-Asselin filter on the middle level
    state_filtered = jax.tree.map(
        lambda ynm1, yn, ynp1: yn + alpha * (ynm1 - 2.0 * yn + ynp1),
        state_nm1,
        state,
        y_next,
    )

    return y_next, state_filtered
