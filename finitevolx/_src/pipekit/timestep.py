"""Explicit time step over a pipekit right-hand side.

`ExplicitStep` turns a tendency operator into a state → state operator, so
a pipekit `Graph` RHS composes with ``pipekit_jax.Rollout`` /
``ScanCycle``::

    rhs : {h, u, v} ↦ {∂ₜh, ∂ₜu, ∂ₜv}        (a pipekit Graph)
    step = ExplicitStep(rhs=rhs, dt=Δt)        s(t) ↦ s(t + Δt)
    traj = Rollout(step=step, n_steps=N)(s0)   {h, u, v: [N + 1, Ny, Nx]}

The schemes are finitevolX's pure step functions (`euler_step`, `heun_step`,
`rk3_ssp_step`, `rk4_step`); this module adds no arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pipekit import Operator
from pipekit._base.operator import nested_config

from finitevolx._src.pipekit.base import FVXOperator
from finitevolx._src.timestepping.explicit_rk import (
    euler_step,
    heun_step,
    rk3_ssp_step,
    rk4_step,
)

#: ``scheme -> step(state, rhs_fn, dt)``. ``dt`` may be a 0-d array (a
#: leaf): the steps only multiply by it.
SCHEMES: dict[str, Callable[..., Any]] = {
    "euler": euler_step,
    "heun": heun_step,
    "rk3_ssp": rk3_ssp_step,
    "rk4": rk4_step,
}


class ExplicitStep(FVXOperator):
    """One explicit step s ↦ s + Δt Φ(s) of ∂ₜs = F(s), for a dict state.

    Attributes
    ----------
    rhs : Operator
        Tendency operator: ``rhs(**s)`` returns a dict with the keys of
        ``s`` (typically a pipekit ``Graph``). Its leaves are leaves of the
        step, so gradients reach them.
    dt : Array
        Time step (a leaf).
    scheme : {"euler", "heun", "rk3_ssp", "rk4"}
        Which `SCHEMES` entry.
    """

    rhs: Operator
    dt: jax.Array = eqx.field(converter=jnp.asarray, kw_only=True)
    scheme: str = eqx.field(static=True, default="rk4", kw_only=True)

    def __check_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(
                f"scheme {self.scheme!r} is not available; valid: {sorted(SCHEMES)}"
            )

    def _apply(self, state: dict[str, Any]) -> dict[str, Any]:
        return SCHEMES[self.scheme](state, lambda s: self.rhs(**s), self.dt)

    def get_config(self) -> dict[str, Any]:
        """Config with ``rhs`` as a nested-operator payload."""
        return {
            "rhs": nested_config(self.rhs),
            "dt": float(self.dt),
            "scheme": self.scheme,
        }
