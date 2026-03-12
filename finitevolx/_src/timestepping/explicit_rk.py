"""Pure functional explicit Runge-Kutta time integrators.

Each function takes a state (any JAX pytree), a right-hand-side callable, and a
timestep, and returns the updated state.  All implementations are pure (no
mutation), pytree-aware via :func:`jax.tree.map`, and fully
:func:`jax.jit`-compatible.

References
----------
- Shu & Osher (1988) — SSP-RK3 foundations.
- Gottlieb, Shu & Tadmor (2001) — Strong stability-preserving methods.
"""

from __future__ import annotations

from collections.abc import Callable

import jax


def euler_step(state, rhs_fn: Callable, dt: float):
    """Forward Euler: y_{n+1} = y_n + dt * f(y_n).

    Parameters
    ----------
    state : PyTree
        Current state (arbitrary JAX pytree).
    rhs_fn : Callable[[PyTree], PyTree]
        Right-hand-side function ``f(state) -> tendency``.
    dt : float
        Timestep.

    Returns
    -------
    PyTree
        Updated state after one Euler step.
    """
    k1 = rhs_fn(state)
    return jax.tree.map(lambda y, f: y + dt * f, state, k1)


def heun_step(state, rhs_fn: Callable, dt: float):
    """Heun (RK2) predictor-corrector.

    .. math::

        k_1 = f(y_n)
        k_2 = f(y_n + dt \\cdot k_1)
        y_{n+1} = y_n + (dt/2)(k_1 + k_2)

    Order 2, SSP with C = 1.

    Parameters
    ----------
    state : PyTree
        Current state.
    rhs_fn : Callable[[PyTree], PyTree]
        Right-hand-side function.
    dt : float
        Timestep.

    Returns
    -------
    PyTree
        Updated state after one Heun step.
    """
    k1 = rhs_fn(state)
    state_star = jax.tree.map(lambda y, f: y + dt * f, state, k1)
    k2 = rhs_fn(state_star)
    return jax.tree.map(lambda y, f1, f2: y + 0.5 * dt * (f1 + f2), state, k1, k2)


def rk3_ssp_step(state, rhs_fn: Callable, dt: float):
    """3rd-order Strong-Stability-Preserving Runge-Kutta (Shu-Osher form).

    .. math::

        y^{(1)} &= y_n + dt \\cdot f(y_n)
        y^{(2)} &= \\tfrac{3}{4} y_n + \\tfrac{1}{4} y^{(1)}
                    + \\tfrac{1}{4} dt \\cdot f(y^{(1)})
        y_{n+1} &= \\tfrac{1}{3} y_n + \\tfrac{2}{3} y^{(2)}
                    + \\tfrac{2}{3} dt \\cdot f(y^{(2)})

    Order 3, SSP coefficient C = 1 (optimal).  Preserves monotonicity,
    positivity, and TVD properties.

    Parameters
    ----------
    state : PyTree
        Current state.
    rhs_fn : Callable[[PyTree], PyTree]
        Right-hand-side function.
    dt : float
        Timestep.

    Returns
    -------
    PyTree
        Updated state after one SSP-RK3 step.
    """
    k1 = rhs_fn(state)
    y1 = jax.tree.map(lambda y, f: y + dt * f, state, k1)

    k2 = rhs_fn(y1)
    y2 = jax.tree.map(
        lambda y, y1_, f: 0.75 * y + 0.25 * y1_ + 0.25 * dt * f,
        state,
        y1,
        k2,
    )

    k3 = rhs_fn(y2)
    return jax.tree.map(
        lambda y, y2_, f: (1.0 / 3.0) * y + (2.0 / 3.0) * y2_ + (2.0 / 3.0) * dt * f,
        state,
        y2,
        k3,
    )


def rk4_step(state, rhs_fn: Callable, dt: float):
    """Classic 4th-order Runge-Kutta.

    .. math::

        k_1 &= f(y_n)
        k_2 &= f(y_n + (dt/2) k_1)
        k_3 &= f(y_n + (dt/2) k_2)
        k_4 &= f(y_n + dt \\cdot k_3)
        y_{n+1} &= y_n + (dt/6)(k_1 + 2 k_2 + 2 k_3 + k_4)

    Order 4, 4 stages, not SSP.

    Parameters
    ----------
    state : PyTree
        Current state.
    rhs_fn : Callable[[PyTree], PyTree]
        Right-hand-side function.
    dt : float
        Timestep.

    Returns
    -------
    PyTree
        Updated state after one RK4 step.
    """
    k1 = rhs_fn(state)
    y1 = jax.tree.map(lambda y, f: y + 0.5 * dt * f, state, k1)

    k2 = rhs_fn(y1)
    y2 = jax.tree.map(lambda y, f: y + 0.5 * dt * f, state, k2)

    k3 = rhs_fn(y2)
    y3 = jax.tree.map(lambda y, f: y + dt * f, state, k3)

    k4 = rhs_fn(y3)
    return jax.tree.map(
        lambda y, f1, f2, f3, f4: y + (dt / 6.0) * (f1 + 2.0 * f2 + 2.0 * f3 + f4),
        state,
        k1,
        k2,
        k3,
        k4,
    )
