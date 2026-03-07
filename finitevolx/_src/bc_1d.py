from __future__ import annotations

"""Single-face boundary condition atoms for 2-D ghost-ring arrays."""

from typing import Literal

import equinox as eqx
from jaxtyping import Array, Float

Face = Literal["south", "north", "west", "east"]
FaceArray = Float[Array, "Nface"]


def _adjacent_interior(field: Float[Array, "Ny Nx"], face: Face) -> FaceArray:
    match face:
        case "south":
            return field[1, :]
        case "north":
            return field[-2, :]
        case "west":
            return field[:, 1]
        case "east":
            return field[:, -2]


def _opposite_interior(field: Float[Array, "Ny Nx"], face: Face) -> FaceArray:
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


def _set_face(
    field: Float[Array, "Ny Nx"], face: Face, values: FaceArray
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
        ghost = interior + (self.value * spacing)
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

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one sponge ghost face updated."""
        del dx, dy
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Sponge1D weight must lie in [0, 1].")
        interior = _adjacent_interior(field, self.face)
        ghost = ((1.0 - self.weight) * interior) + (self.weight * self.background)
        return _set_face(field, self.face, ghost)


BoundaryCondition1D = (
    Dirichlet1D | Neumann1D | Periodic1D | Outflow1D | Reflective1D | Sponge1D
)
