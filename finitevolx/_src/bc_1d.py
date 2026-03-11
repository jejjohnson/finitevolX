from __future__ import annotations

"""Single-face boundary condition atoms for 2-D ghost-ring arrays."""

from typing import Literal

import equinox as eqx
from jaxtyping import Array, Float

type Face = Literal["south", "north", "west", "east"]
type BoundarySlice = Float[Array, "N"]


def _adjacent_interior(field: Float[Array, "Ny Nx"], face: Face) -> BoundarySlice:
    match face:
        case "south":
            return field[1, :]
        case "north":
            return field[-2, :]
        case "west":
            return field[:, 1]
        case "east":
            return field[:, -2]


def _opposite_interior(field: Float[Array, "Ny Nx"], face: Face) -> BoundarySlice:
    match face:
        case "south":
            return field[-2, :]
        case "north":
            return field[1, :]
        case "west":
            return field[:, -2]
        case "east":
            return field[:, 1]


def _normal_spacing(face: Face, dx: float, dy: float) -> float:
    match face:
        case "south" | "north":
            return dy
        case "west" | "east":
            return dx


def _outward_sign(face: Face) -> float:
    match face:
        case "south" | "west":
            return -1.0
        case "north" | "east":
            return 1.0


def _set_face(
    field: Float[Array, "Ny Nx"], face: Face, values: BoundarySlice
) -> Float[Array, "Ny Nx"]:
    match face:
        case "south":
            return field.at[0, :].set(values)
        case "north":
            return field.at[-1, :].set(values)
        case "west":
            return field.at[:, 0].set(values)
        case "east":
            return field.at[:, -1].set(values)


class Dirichlet1D(eqx.Module):
    """Apply a Dirichlet value on one domain face.

    The boundary value is imposed at the physical wall between the first
    interior cell and the ghost cell:

    ``phi_boundary = 0.5 * (phi_interior + phi_ghost)``

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    value : float
        Target boundary value.
    """

    face: Face = eqx.field(static=True)
    value: float

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one Dirichlet ghost face updated."""
        interior = _adjacent_interior(field, self.face)
        ghost = (2.0 * self.value) - interior
        return _set_face(field, self.face, ghost)


class Neumann1D(eqx.Module):
    """Apply an outward-normal gradient on one domain face.

    The gradient is interpreted as ``dphi/dn`` along the outward normal,
    evaluated over the half-cell distance between the first interior cell and
    the ghost cell.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    value : float, optional
        Outward-normal gradient. Defaults to ``0.0`` for zero-gradient
        boundaries.
    """

    face: Face = eqx.field(static=True)
    value: float = 0.0

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one Neumann ghost face updated."""
        spacing = _normal_spacing(self.face, dx=dx, dy=dy)
        interior = _adjacent_interior(field, self.face)
        ghost = interior + (_outward_sign(self.face) * self.value * spacing)
        return _set_face(field, self.face, ghost)


class Periodic1D(eqx.Module):
    """Apply periodic wrapping on one domain face.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    """

    face: Face = eqx.field(static=True)

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one periodic ghost face updated."""
        del dx, dy
        ghost = _opposite_interior(field, self.face)
        return _set_face(field, self.face, ghost)


class Outflow1D(eqx.Module):
    """Apply a one-face outflow boundary condition.

    Outflow is modelled here as a zero-gradient copy from the nearest
    interior cell into the ghost ring.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    """

    face: Face = eqx.field(static=True)

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one outflow ghost face updated."""
        del dx, dy
        ghost = _adjacent_interior(field, self.face)
        return _set_face(field, self.face, ghost)


class Reflective1D(eqx.Module):
    """Apply an even-symmetry reflective boundary on one face.

    This mirrors the nearest interior values into the ghost cells, which is
    appropriate for scalar tracers or tangential components with reflective
    symmetry.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    """

    face: Face = eqx.field(static=True)

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one reflective ghost face updated."""
        del dx, dy
        ghost = _adjacent_interior(field, self.face)
        return _set_face(field, self.face, ghost)


class Sponge1D(eqx.Module):
    """Relax one ghost face toward a background value.

    The ghost cells are updated as:

    ``phi_ghost = (1 - weight) * phi_interior + weight * background``

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    background : float
        Background state toward which the ghost cells relax.
    weight : float
        Relaxation weight in ``[0, 1]``.
    """

    face: Face = eqx.field(static=True)
    background: float
    weight: float

    def __check_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Sponge1D weight must lie in [0, 1].")

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one sponge ghost face updated."""
        del dx, dy
        interior = _adjacent_interior(field, self.face)
        ghost = ((1.0 - self.weight) * interior) + (self.weight * self.background)
        return _set_face(field, self.face, ghost)


class Slip1D(eqx.Module):
    """Slip boundary condition for tangential velocity at a solid wall.

    Controls the tangential velocity at the wall via a slip coefficient
    ``a in [0, 1]``:

    - ``a = 1.0`` — **free-slip**: ghost = +interior (zero gradient, frictionless wall).
    - ``a = 0.0`` — **no-slip**: ghost = -interior (zero tangential velocity at wall).
    - ``0 < a < 1`` — **partial-slip**: linear interpolation between the two.

    The ghost cell is set using:

    ``phi_ghost = (2*a - 1) * phi_interior``

    so that the extrapolated wall value
    ``phi_wall = 0.5 * (phi_ghost + phi_interior) = a * phi_interior``
    smoothly varies from zero (no-slip) to the interior value (free-slip).

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    coefficient : float
        Slip parameter ``a`` in ``[0, 1]``. Default is ``1.0`` (free-slip).

    Examples
    --------
    Free-slip on the west wall (tangential velocity preserved):

    >>> bc = Slip1D("west", coefficient=1.0)
    >>> field_out = bc(field, dx=1.0, dy=1.0)

    No-slip on the north wall (tangential velocity -> 0):

    >>> bc = Slip1D("north", coefficient=0.0)
    >>> field_out = bc(field, dx=1.0, dy=1.0)

    Partial-slip on the south wall:

    >>> bc = Slip1D("south", coefficient=0.5)
    >>> field_out = bc(field, dx=1.0, dy=1.0)
    """

    face: Face = eqx.field(static=True)
    coefficient: float = 1.0

    def __check_init__(self) -> None:
        if not 0.0 <= self.coefficient <= 1.0:
            raise ValueError("Slip1D coefficient must lie in [0, 1].")

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one slip ghost face updated.

        Parameters
        ----------
        field : Float[Array, "Ny Nx"]
            Input array with ghost-cell ring.
        dx : float
            Grid spacing in x (unused, kept for interface consistency).
        dy : float
            Grid spacing in y (unused, kept for interface consistency).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Field with the ghost face set according to the slip coefficient.
        """
        del dx, dy
        interior = _adjacent_interior(field, self.face)
        # ghost = (2*a - 1) * interior
        # wall value = 0.5 * (ghost + interior) = a * interior
        ghost = (2.0 * self.coefficient - 1.0) * interior
        return _set_face(field, self.face, ghost)


type BoundaryCondition1D = (
    Dirichlet1D | Neumann1D | Periodic1D | Outflow1D | Reflective1D | Sponge1D | Slip1D
)
