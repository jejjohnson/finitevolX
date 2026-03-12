"""Pure functional IMEX (Implicit-Explicit) time integrator.

Splits the ODE as ``dy/dt = f_E(y) + f_I(y)`` where ``f_E`` is treated
explicitly (e.g. advection) and ``f_I`` is treated implicitly (e.g. vertical
diffusion).  The caller provides an ``implicit_solve`` callback that solves
the implicit system at each stage.

References
----------
- Pareschi & Russo (2005) — IMEX Runge-Kutta methods.
- Ascher, Ruuth & Spiteri (1997) — IMEX Runge-Kutta methods.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import jax

#: IMEX-SSP2 parameter: gamma = 1 - 1/sqrt(2)
_GAMMA = 1.0 - 1.0 / math.sqrt(2.0)


def imex_ssp2_step(
    state,
    rhs_explicit: Callable,
    rhs_implicit: Callable,
    implicit_solve: Callable,
    dt: float,
):
    """IMEX-SSP2(2,2,2) time step.

    The explicit part is SSP with C = 1; the implicit part is A-stable and
    L-stable (SDIRK with gamma = 1 - 1/sqrt(2)).

    Algorithm (two stages)::

        Stage 1:
            Y_1 = y_n + gamma * dt * f_I(Y_1)
            -> solved via implicit_solve(y_n, gamma * dt)

        Stage 2:
            Y_2_star = y_n + dt * f_E(y_n)
            Y_2 = Y_2_star + gamma * dt * f_I(Y_2)
            -> solved via implicit_solve(Y_2_star, gamma * dt)

        Update:
            y_{n+1} = y_n + (dt/2) * [f_E(y_n) + f_E(Y_2)]
                          + (dt/2) * [f_I(Y_1) + f_I(Y_2)]

    Parameters
    ----------
    state : PyTree
        Current state y_n.
    rhs_explicit : Callable[[PyTree], PyTree]
        Explicit (non-stiff) right-hand side, e.g. advection.
    rhs_implicit : Callable[[PyTree], PyTree]
        Implicit (stiff) right-hand side, e.g. vertical diffusion.
    implicit_solve : Callable[[PyTree, float], PyTree]
        Solves ``Y - gamma * dt * f_I(Y) = rhs`` for ``Y`` given ``(rhs,
        gamma * dt)``.  For vertical diffusion this is typically a tridiagonal
        (TDMA) solve along columns.
    dt : float
        Timestep.

    Returns
    -------
    PyTree
        Updated state after one IMEX-SSP2 step.
    """
    gamma_dt = _GAMMA * dt

    # Stage 1: implicit solve from y_n
    y1 = implicit_solve(state, gamma_dt)

    # Stage 2: explicit predictor then implicit correction
    fe_0 = rhs_explicit(state)
    y2_star = jax.tree.map(lambda y, f: y + dt * f, state, fe_0)
    y2 = implicit_solve(y2_star, gamma_dt)

    # Final update: average explicit and implicit contributions
    fe_1 = rhs_explicit(y2)
    fi_0 = rhs_implicit(y1)
    fi_1 = rhs_implicit(y2)

    return jax.tree.map(
        lambda y, fe0, fe1, fi0, fi1: (
            y + 0.5 * dt * (fe0 + fe1) + 0.5 * dt * (fi0 + fi1)
        ),
        state,
        fe_0,
        fe_1,
        fi_0,
        fi_1,
    )
