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


class Robin1D(eqx.Module):
    """Robin (mixed) boundary condition: α·u + β·∂u/∂n = γ.

    The Robin condition is a linear combination of Dirichlet and Neumann:

    - β = 0 recovers Dirichlet (α·u = γ → u = γ/α)
    - α = 0 recovers Neumann  (β·∂u/∂n = γ → ∂u/∂n = γ/β)

    The ghost cell is derived from the wall-value and wall-gradient
    discretisations used by ``Dirichlet1D`` and ``Neumann1D``::

        u_wall  = 0.5 * (u_ghost + u_interior)
        ∂u/∂n   = sign * (u_ghost - u_interior) / spacing

    Substituting into α·u_wall + β·∂u/∂n = γ and solving for u_ghost:

        s = β · sign / spacing
        u_ghost = (γ - u_int · (α/2 - s)) / (α/2 + s)

    The denominator ``α/2 + s`` is zero when ``α·spacing = -2·β·sign``.
    This is a singular configuration that produces inf/NaN ghost values;
    callers must choose (α, β) such that the denominator is non-zero for
    the given face and grid spacing.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    alpha : float
        Coefficient on the field value.
    beta : float
        Coefficient on the outward-normal gradient.
    gamma : float
        Right-hand-side value.
    """

    face: Face = eqx.field(static=True)
    alpha: float
    beta: float
    gamma: float

    def __check_init__(self) -> None:
        if self.alpha == 0.0 and self.beta == 0.0:
            raise ValueError(
                "Robin1D requires at least one of alpha or beta to be non-zero."
            )

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one Robin ghost face updated."""
        spacing = _normal_spacing(self.face, dx=dx, dy=dy)
        sign = _outward_sign(self.face)
        interior = _adjacent_interior(field, self.face)
        s = self.beta * sign / spacing
        ghost = (self.gamma - interior * (self.alpha / 2.0 - s)) / (
            self.alpha / 2.0 + s
        )
        return _set_face(field, self.face, ghost)


class Extrapolation1D(eqx.Module):
    """High-order polynomial extrapolation for ghost cells.

    Uses Lagrange polynomial extrapolation from ``order + 1`` interior points
    to fill the ghost cell.  This generalises ``Outflow1D`` (which is
    equivalent to order 0, i.e. constant extrapolation).

    Coefficients (nearest interior → farthest)::

        order 1: [ 2,  -1]
        order 2: [ 3,  -3,   1]
        order 3: [ 4,  -6,   4,  -1]
        order 4: [ 5, -10,  10,  -5,   1]
        order 5: [ 6, -15,  20, -15,   6, -1]

    The field must have at least ``order + 2`` points along the normal
    axis (``order + 1`` interior points plus the ghost cell).  On smaller
    grids the stencil will read into the opposite ghost ring and produce
    incorrect results.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    order : int
        Extrapolation order (1–5).
    """

    face: Face = eqx.field(static=True)
    order: int = eqx.field(static=True)
    _coeffs: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, face: Face, order: int = 1) -> None:
        if not 1 <= order <= 5:
            raise ValueError("Extrapolation1D order must be in [1, 5].")
        self.face = face
        self.order = order
        # Lagrange extrapolation coefficients: (-1)^k * C(n, k+1)
        n = order + 1
        coeffs: list[float] = []
        c = 1.0
        for k in range(n):
            c = c * (n - k) / (k + 1)
            coeffs.append((-1.0) ** k * c)
        # coeffs[k] multiplies the (k+1)-th interior point from the boundary
        self._coeffs = tuple(coeffs)

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one extrapolated ghost face updated."""
        del dx, dy
        ghost: BoundarySlice = _adjacent_interior(field, self.face) * 0.0
        for k, c in enumerate(self._coeffs):
            # k=0 → adjacent interior, k=1 → one step further in, etc.
            match self.face:
                case "south":
                    ghost = ghost + c * field[1 + k, :]
                case "north":
                    ghost = ghost + c * field[-(2 + k), :]
                case "west":
                    ghost = ghost + c * field[:, 1 + k]
                case "east":
                    ghost = ghost + c * field[:, -(2 + k)]
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
    Dirichlet1D
    | Neumann1D
    | Robin1D
    | Periodic1D
    | Outflow1D
    | Extrapolation1D
    | Reflective1D
    | Sponge1D
    | Slip1D
)
