"""Convenience wrapper around :func:`diffrax.diffeqsolve` for ocean PDEs.

Provides :func:`solve_ocean_pde`, which optionally applies boundary conditions
inside the RHS evaluation so that the caller does not need to manually compose
the BC function into the tendency.
"""

from __future__ import annotations

from collections.abc import Callable

import diffrax as dfx
from jaxtyping import PyTree


def solve_ocean_pde(
    rhs_fn: Callable,
    solver: dfx.AbstractSolver,
    y0: PyTree,
    t0: float,
    t1: float,
    dt0: float,
    saveat: dfx.SaveAt | None = None,
    bc_fn: Callable | None = None,
    args: PyTree = None,
) -> dfx.Solution:
    """Integrate an ocean PDE using diffrax.

    Parameters
    ----------
    rhs_fn : Callable
        Right-hand side ``rhs_fn(t, y, args) -> dy/dt``.
    solver : diffrax.AbstractSolver
        Time integration scheme (e.g. ``RK3SSP()``, ``RK4Classic()``).
    y0 : PyTree
        Initial condition.
    t0, t1 : float
        Start and end times.
    dt0 : float
        Initial (or fixed) timestep.
    saveat : diffrax.SaveAt, optional
        Output saving specification.  Defaults to saving the final state.
    bc_fn : Callable, optional
        Boundary condition function ``bc_fn(dydt) -> dydt_corrected`` applied
        to the tendency after each RHS evaluation.
    args : PyTree, optional
        Static arguments forwarded to ``rhs_fn``.

    Returns
    -------
    diffrax.Solution
        Solution object containing the saved states.
    """
    if bc_fn is not None:

        def rhs_with_bc(t, y, args_):
            dydt = rhs_fn(t, y, args_)
            return bc_fn(dydt)

        term = dfx.ODETerm(rhs_with_bc)
    else:
        term = dfx.ODETerm(rhs_fn)

    return dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat if saveat is not None else dfx.SaveAt(t1=True),
    )
