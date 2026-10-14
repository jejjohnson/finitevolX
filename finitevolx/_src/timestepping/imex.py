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
from jaxtyping import PyTree

#: IMEX-SSP2 parameter: gamma = 1 - 1/sqrt(2)
_GAMMA = 1.0 - 1.0 / math.sqrt(2.0)


def imex_ssp2_step(
    state: PyTree,
    rhs_explicit: Callable[[PyTree], PyTree],
    rhs_implicit: Callable[[PyTree], PyTree],
    implicit_solve: Callable[[PyTree, float], PyTree],
    dt: float,
) -> PyTree:
    """IMEX-SSP2(2,2,2) time step.

    The explicit part is SSP with C = 1; the implicit part is A-stable and
    L-stable (SDIRK with gamma = 1 - 1/sqrt(2)).

    Pareschi & Russo (2005) SSP2(2,2,2), ``gamma = 1 - 1/sqrt(2)``::

        explicit  A_E = [[0, 0], [1, 0]]          b_E = [1/2, 1/2]
        implicit  A_I = [[gamma, 0], [1-2*gamma, gamma]]   b_I = [1/2, 1/2]

    Algorithm (two stages)::

        Stage 1:
            Y_1 = y_n + gamma * dt * f_I(Y_1)
            -> implicit_solve(y_n, gamma * dt)

        Stage 2:
            Y_2_star = y_n + dt * f_E(Y_1) + (1 - 2*gamma) * dt * f_I(Y_1)
            Y_2 = Y_2_star + gamma * dt * f_I(Y_2)
            -> implicit_solve(Y_2_star, gamma * dt)

        Update:
            y_{n+1} = y_n + (dt/2) * [f_E(Y_1) + f_E(Y_2)]
                          + (dt/2) * [f_I(Y_1) + f_I(Y_2)]

    The method is second order in the explicit part, the implicit part and
    the coupled split (the implicit row ``(1 - 2*gamma, gamma)`` is what gives
    ``sum_i b_i c_i = 1/2``).

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

    # Stage 1: Y1 = y_n + gamma dt f_I(Y1)
    y1 = implicit_solve(state, gamma_dt)
    fe_1 = rhs_explicit(y1)
    fi_1 = rhs_implicit(y1)

    # Stage 2: Y2* = y_n + dt f_E(Y1) + (1 - 2 gamma) dt f_I(Y1)
    #          Y2  = Y2* + gamma dt f_I(Y2)
    y2_star = jax.tree.map(
        lambda y, fe, fi: y + dt * fe + (1.0 - 2.0 * _GAMMA) * dt * fi,
        state,
        fe_1,
        fi_1,
    )
    y2 = implicit_solve(y2_star, gamma_dt)
    fe_2 = rhs_explicit(y2)
    fi_2 = rhs_implicit(y2)

    # Update: y_{n+1} = y_n + dt/2 (f_E(Y1) + f_E(Y2)) + dt/2 (f_I(Y1) + f_I(Y2))
    return jax.tree.map(
        lambda y, fe1, fe2, fi1, fi2: (
            y + 0.5 * dt * (fe1 + fe2) + 0.5 * dt * (fi1 + fi2)
        ),
        state,
        fe_1,
        fe_2,
        fi_1,
        fi_2,
    )
