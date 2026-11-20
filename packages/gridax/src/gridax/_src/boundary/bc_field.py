"""Boundary-condition dispatchers for state dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.boundary.bc_set import BoundaryConditionSet
from gridax._src.pytree import pytree_dataclass

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


@pytree_dataclass
class FieldBCSet:
    """Dispatch per-face boundary conditions across multiple fields.

    Parameters
    ----------
    bc_map : dict[str, BoundaryConditionSet]
        Mapping from state-variable name to per-face boundary conditions.
    default : BoundaryConditionSet | None, optional
        Boundary-condition set used when a variable is not present in
        ``bc_map``. If omitted, unmatched state entries are returned unchanged.
    """

    bc_map: dict[str, BoundaryConditionSet]
    default: BoundaryConditionSet | None = None

    def __call__(
        self, state: dict[str, Float[Array, "*batch Ny Nx"]], dx: float, dy: float
    ) -> dict[str, Float[Array, "*batch Ny Nx"]]:
        """Return a new state dictionary with boundary conditions applied."""
        out: dict[str, Float[Array, "*batch Ny Nx"]] = {}
        for name, field in state.items():
            bc = self.bc_map.get(name, self.default)
            out[name] = field if bc is None else bc(field, dx=dx, dy=dy)
        return out
