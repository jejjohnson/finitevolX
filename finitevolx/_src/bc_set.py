from __future__ import annotations

"""Per-face boundary condition containers for 2-D ghost-ring arrays."""

import equinox as eqx
from jaxtyping import Array, Float

from finitevolx._src.bc_1d import BoundaryCondition1D, Outflow1D, Periodic1D


class BoundaryConditionSet(eqx.Module):
    """Apply a different boundary condition on each domain face.

    The application order is south, north, west, then east, so west/east
    updates overwrite the corner values written by south/north updates.

    Parameters
    ----------
    south : BoundaryCondition1D | None, optional
        Boundary condition for the south ghost row.
    north : BoundaryCondition1D | None, optional
        Boundary condition for the north ghost row.
    west : BoundaryCondition1D | None, optional
        Boundary condition for the west ghost column.
    east : BoundaryCondition1D | None, optional
        Boundary condition for the east ghost column.
    """

    south: BoundaryCondition1D | None = None
    north: BoundaryCondition1D | None = None
    west: BoundaryCondition1D | None = None
    east: BoundaryCondition1D | None = None

    @classmethod
    def periodic(cls) -> BoundaryConditionSet:
        """Return a fully periodic boundary-condition set."""
        return cls(
            south=Periodic1D("south"),
            north=Periodic1D("north"),
            west=Periodic1D("west"),
            east=Periodic1D("east"),
        )

    @classmethod
    def open(cls) -> BoundaryConditionSet:
        """Return a zero-gradient boundary-condition set on all faces."""
        return cls(
            south=Outflow1D("south"),
            north=Outflow1D("north"),
            west=Outflow1D("west"),
            east=Outflow1D("east"),
        )

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with all configured ghost faces updated."""
        out = field
        for bc in (self.south, self.north, self.west, self.east):
            if bc is not None:
                out = bc(out, dx=dx, dy=dy)
        return out
