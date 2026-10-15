"""Per-face boundary condition containers for 2-D ghost-ring arrays."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from finitevolx._src.boundary.bc_1d import (
    BoundaryCondition1D,
    Dirichlet1D,
    Outflow1D,
    Periodic1D,
    Point,
)
from finitevolx._src.mask import Mask2D


class BoundaryConditionSet(eqx.Module):
    """Apply a different boundary condition on each domain face.

    The application order is south, north, west, then east, so west/east
    updates overwrite the corner values written by south/north updates.

    Each atom carries its own ``point`` (the C-grid point of the field); the
    ``closed`` / ``open`` constructors pass one ``point`` to all four faces.
    A closed basin for the velocity is ``BoundaryConditionSet.closed(point="U")``
    for ``u`` and ``closed(point="V")`` for ``v``: zero normal velocity on the
    wall faces and no-slip (``ghost = -interior``) on the tangential walls.

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
    mask : Mask2D or Float[Array, "Ny Nx"] or None, optional
        Domain mask.  When provided, solvers can extract it from the
        BCSet instead of requiring a separate ``mask=`` argument.
    """

    south: BoundaryCondition1D | None = None
    north: BoundaryCondition1D | None = None
    west: BoundaryCondition1D | None = None
    east: BoundaryCondition1D | None = None
    mask: Mask2D | Float[Array, "Ny Nx"] | None = None

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
    def closed(
        cls,
        mask: Mask2D | Float[Array, "Ny Nx"] | None = None,
        point: Point = "T",
    ) -> BoundaryConditionSet:
        """Return a zero-Dirichlet boundary-condition set on all faces.

        ``point`` is the C-grid point of the field (``"T"``, ``"U"``, ``"V"``
        or ``"X"``); see :class:`Dirichlet1D`.
        """
        return cls(
            south=Dirichlet1D("south", value=0.0, point=point),
            north=Dirichlet1D("north", value=0.0, point=point),
            west=Dirichlet1D("west", value=0.0, point=point),
            east=Dirichlet1D("east", value=0.0, point=point),
            mask=mask,
        )

    @classmethod
    def open(cls, point: Point = "T") -> BoundaryConditionSet:
        """Return a zero-gradient boundary-condition set on all faces.

        ``point`` is the C-grid point of the field; see :class:`Outflow1D`.
        """
        return cls(
            south=Outflow1D("south", point=point),
            north=Outflow1D("north", point=point),
            west=Outflow1D("west", point=point),
            east=Outflow1D("east", point=point),
        )

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with all configured ghost faces updated.

        Boundary conditions are applied in south, north, west, east order, so
        west/east updates overwrite the corner values written by south/north.
        """
        out = field
        for bc in (self.south, self.north, self.west, self.east):
            if bc is not None:
                out = bc(out, dx=dx, dy=dy)
        return out
