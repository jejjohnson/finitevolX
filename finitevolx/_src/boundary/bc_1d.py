"""Single-face boundary condition atoms for 2-D ghost-ring arrays.

Every atom takes a static ``point`` (``"T"``, ``"U"``, ``"V"`` or ``"X"``):
the C-grid point the field lives on.  It decides where the wall is relative
to the field's nodes on that face (same-index staggering, AGENTS.md
contract 1):

- **Cell-centred along the face normal** (``T`` everywhere, ``V`` on the
  west / east faces, ``U`` on the south / north faces): the wall is half a
  cell outside the first interior node, between it and the ghost node
  (index ``0`` or ``-1``).  The atom fills that ghost node.
- **Node on the wall** (``U`` and ``X`` on the west / east faces, ``V`` and
  ``X`` on the south / north faces): the field's own node sits on the wall,
  at index ``0`` (west / south) or ``-2`` (east / north); index ``-1`` on
  the east / north side is outside the domain.  The atom writes the wall
  node directly and fills the outside slot with the odd mirror
  ``2 * phi_wall - phi_adjacent`` (so a centred difference across the wall
  equals the one-sided one, and a zero wall value stays zero under a
  centred Laplacian).
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

type Face = Literal["south", "north", "west", "east"]
type Point = Literal["T", "U", "V", "X"]
type BoundarySlice = Float[Array, "N"]


def _on_wall(face: Face, point: Point) -> bool:
    """True when the field's boundary node on *face* lies on the wall itself.

    Along x, ``U[j, i]`` and ``X[j, i]`` sit at ``i + 1/2``: node ``0`` is the
    west wall and node ``Nx-2`` the east wall.  Along y, ``V`` and ``X`` sit at
    ``j + 1/2`` likewise.  Every other (face, point) pair is cell-centred
    along the face normal.
    """
    if point not in ("T", "U", "V", "X"):
        raise ValueError(f"point must be 'T', 'U', 'V' or 'X', got {point!r}")
    match face:
        case "west" | "east":
            return point in ("U", "X")
        case "south" | "north":
            return point in ("V", "X")


def _interior_line(
    field: Float[Array, "Ny Nx"], face: Face, k: int = 0, point: Point = "T"
) -> BoundarySlice:
    """Return the ``k``-th interior line inward from *face* (``k = 0``: adjacent).

    The adjacent line is index ``1`` on the south / west side; on the north /
    east side it is ``-2`` for cell-centred points and ``-3`` when the
    boundary node ``-2`` is the wall itself.
    """
    far = 3 if _on_wall(face, point) else 2  # index of the adjacent line from the end
    match face:
        case "south":
            return field[1 + k, :]
        case "north":
            return field[-(far + k), :]
        case "west":
            return field[:, 1 + k]
        case "east":
            return field[:, -(far + k)]


def _adjacent_interior(
    field: Float[Array, "Ny Nx"], face: Face, point: Point = "T"
) -> BoundarySlice:
    return _interior_line(field, face, 0, point)


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


def _set_face(
    field: Float[Array, "Ny Nx"],
    face: Face,
    values: BoundarySlice,
    point: Point = "T",
) -> Float[Array, "Ny Nx"]:
    """Write the boundary line of *face*: the ghost line, or the wall node.

    For a node on the wall the north / east wall node is index ``-2`` and the
    outside slot ``-1`` gets the odd mirror ``2 * values - adjacent``.
    """
    if not _on_wall(face, point):
        match face:
            case "south":
                return field.at[0, :].set(values)
            case "north":
                return field.at[-1, :].set(values)
            case "west":
                return field.at[:, 0].set(values)
            case "east":
                return field.at[:, -1].set(values)
    # wall node: phi[0] (south / west) or phi[-2] (north / east)
    match face:
        case "south":
            return field.at[0, :].set(values)
        case "west":
            return field.at[:, 0].set(values)
        case "north":
            mirror = 2.0 * values - field[-3, :]  # phi[-1] = 2 phi_wall - phi[-3]
            return field.at[-2, :].set(values).at[-1, :].set(mirror)
        case "east":
            mirror = 2.0 * values - field[:, -3]  # phi[:, -1] = 2 phi_wall - phi[:, -3]
            return field.at[:, -2].set(values).at[:, -1].set(mirror)


class Dirichlet1D(eqx.Module):
    """Apply a Dirichlet value on one domain face.

    For a cell-centred field (``point="T"``, the default) the value is
    imposed at the physical wall between the first interior cell and the
    ghost cell:

    ``phi_boundary = 0.5 * (phi_interior + phi_ghost)``,
    i.e. ``phi_ghost = 2 * value - phi_interior``.

    For a node on the wall (the normal velocity ``U`` on west / east or
    ``V`` on south / north, any ``X`` field) the wall node is set to
    ``value`` directly.  A closed wall for the normal velocity is therefore
    ``Dirichlet1D("east", 0.0, point="U")``; without ``point="U"`` the east
    wall face ``U[:, Nx-2]`` is left untouched and mass leaks through it.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    value : float
        Target boundary value.
    point : Literal["T", "U", "V", "X"], optional
        C-grid point the field lives on (default ``"T"``).  For a node on
        the wall (``U`` / ``X`` on west / east, ``V`` / ``X`` on south /
        north) the condition is imposed on the wall node itself (see the
        module docstring).

    Examples
    --------
    No normal flow through the west and east walls of a U-field:

    >>> u = jnp.ones((6, 6))
    >>> u = Dirichlet1D("west", 0.0, point="U")(u, dx=1.0, dy=1.0)
    >>> u = Dirichlet1D("east", 0.0, point="U")(u, dx=1.0, dy=1.0)
    >>> float(u[2, 0]), float(u[2, -2])  # the two wall faces
    (0.0, 0.0)
    """

    face: Face = eqx.field(static=True)
    value: float
    point: Point = eqx.field(static=True, default="T")

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one Dirichlet ghost face (or wall node) updated."""
        interior = _adjacent_interior(field, self.face, self.point)  # [N]
        if _on_wall(self.face, self.point):
            # the node is on the wall: phi_wall = value
            wall = jnp.zeros_like(interior) + self.value  # [N]
            return _set_face(field, self.face, wall, self.point)
        # phi_wall = (phi_ghost + phi_interior) / 2 = value
        ghost = (2.0 * self.value) - interior  # [N]
        return _set_face(field, self.face, ghost)


class Neumann1D(eqx.Module):
    """Apply an outward-normal gradient on one domain face.

    The gradient is interpreted as ``dphi/dn`` along the outward normal,
    evaluated over the one-cell distance between the first interior node
    and the boundary node: the ghost node for a cell-centred field (the
    difference is centred on the wall face), the wall node itself for a
    node on the wall (a one-sided difference; the outside slot then makes
    the centred difference across the wall node equal ``value`` too).

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    value : float, optional
        Outward-normal gradient. Defaults to ``0.0`` for zero-gradient
        boundaries.
    point : Literal["T", "U", "V", "X"], optional
        C-grid point the field lives on (default ``"T"``).  For a node on
        the wall (``U`` / ``X`` on west / east, ``V`` / ``X`` on south /
        north) the condition is imposed on the wall node itself (see the
        module docstring).
    """

    face: Face = eqx.field(static=True)
    value: float = 0.0
    point: Point = eqx.field(static=True, default="T")

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one Neumann ghost face updated."""
        spacing = _normal_spacing(self.face, dx=dx, dy=dy)
        interior = _adjacent_interior(field, self.face, self.point)
        # The boundary node lies one cell outward of the interior on every
        # face, so dphi/dn = (boundary - interior) / spacing
        #   =>  boundary = interior + g * spacing.
        boundary = interior + (self.value * spacing)  # [N]
        return _set_face(field, self.face, boundary, self.point)


class Periodic1D(eqx.Module):
    """Apply periodic wrapping on one domain face.

    Periodic wrapping copies the interior line one period away, which is the
    right ghost value for every C-grid point (``U[:, 0]`` is the same face as
    ``U[:, Nx-2]``), so this atom takes no ``point``.

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
    point : Literal["T", "U", "V", "X"], optional
        C-grid point the field lives on (default ``"T"``).  For a node on
        the wall (``U`` / ``X`` on west / east, ``V`` / ``X`` on south /
        north) the condition is imposed on the wall node itself (see the
        module docstring).
    """

    face: Face = eqx.field(static=True)
    point: Point = eqx.field(static=True, default="T")

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one outflow ghost face updated."""
        del dx, dy
        ghost = _adjacent_interior(field, self.face, self.point)
        return _set_face(field, self.face, ghost, self.point)


class Reflective1D(eqx.Module):
    """Apply an even-symmetry reflective boundary on one face.

    This mirrors the nearest interior values into the ghost cells, which is
    appropriate for scalar tracers or tangential components with reflective
    symmetry.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    point : Literal["T", "U", "V", "X"], optional
        C-grid point the field lives on (default ``"T"``).  For a node on
        the wall (``U`` / ``X`` on west / east, ``V`` / ``X`` on south /
        north) the condition is imposed on the wall node itself (see the
        module docstring).
    """

    face: Face = eqx.field(static=True)
    point: Point = eqx.field(static=True, default="T")

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one reflective ghost face updated."""
        del dx, dy
        ghost = _adjacent_interior(field, self.face, self.point)
        return _set_face(field, self.face, ghost, self.point)


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
    point : Literal["T", "U", "V", "X"], optional
        C-grid point the field lives on (default ``"T"``).  For a node on
        the wall (``U`` / ``X`` on west / east, ``V`` / ``X`` on south /
        north) the condition is imposed on the wall node itself (see the
        module docstring).
    """

    face: Face = eqx.field(static=True)
    background: float
    weight: float
    point: Point = eqx.field(static=True, default="T")

    def __check_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Sponge1D weight must lie in [0, 1].")

    def __call__(
        self, field: Float[Array, "Ny Nx"], dx: float, dy: float
    ) -> Float[Array, "Ny Nx"]:
        """Return ``field`` with one sponge ghost face updated."""
        del dx, dy
        interior = _adjacent_interior(field, self.face, self.point)
        ghost = ((1.0 - self.weight) * interior) + (self.weight * self.background)
        return _set_face(field, self.face, ghost, self.point)


class Robin1D(eqx.Module):
    """Robin (mixed) boundary condition: α·u + β·∂u/∂n = γ.

    The Robin condition is a linear combination of Dirichlet and Neumann:

    - β = 0 recovers Dirichlet (α·u = γ → u = γ/α)
    - α = 0 recovers Neumann  (β·∂u/∂n = γ → ∂u/∂n = γ/β)

    The ghost cell is derived from the wall-value and wall-gradient
    discretisations used by ``Dirichlet1D`` and ``Neumann1D``::

        u_wall  = 0.5 * (u_ghost + u_interior)
        ∂u/∂n   = (u_ghost - u_interior) / spacing

    The ghost cell lies outward of the interior cell on every face, so the
    outward-normal gradient has the same form on all four faces.

    Substituting into α·u_wall + β·∂u/∂n = γ and solving for u_ghost:

        s = β / spacing
        u_ghost = (γ - u_int · (α/2 - s)) / (α/2 + s)

    The denominator ``α/2 + s`` is zero when ``α·spacing = -2·β``.
    This is a singular configuration that produces inf/NaN ghost values;
    callers must choose (α, β) such that the denominator is non-zero for
    the given face and grid spacing.

    For a node on the wall the condition holds at the wall node, with the
    one-sided gradient ``(u_wall - u_interior) / spacing``::

        u_wall = (γ + s · u_int) / (α + s)

    singular when ``α·spacing = -β``.

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
    point : Literal["T", "U", "V", "X"], optional
        C-grid point the field lives on (default ``"T"``).  For a node on
        the wall (``U`` / ``X`` on west / east, ``V`` / ``X`` on south /
        north) the condition is imposed on the wall node itself (see the
        module docstring).
    """

    face: Face = eqx.field(static=True)
    alpha: float
    beta: float
    gamma: float
    point: Point = eqx.field(static=True, default="T")

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
        interior = _adjacent_interior(field, self.face, self.point)
        s = self.beta / spacing
        if _on_wall(self.face, self.point):
            # alpha u_w + beta (u_w - u_int) / spacing = gamma
            wall = (self.gamma + s * interior) / (self.alpha + s)  # [N]
            return _set_face(field, self.face, wall, self.point)
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
    point : Literal["T", "U", "V", "X"], optional
        C-grid point the field lives on (default ``"T"``).  For a node on
        the wall (``U`` / ``X`` on west / east, ``V`` / ``X`` on south /
        north) the condition is imposed on the wall node itself (see the
        module docstring).
    """

    face: Face = eqx.field(static=True)
    order: int = eqx.field(static=True)
    _coeffs: tuple[float, ...] = eqx.field(static=True)
    point: Point = eqx.field(static=True)

    def __init__(self, face: Face, order: int = 1, point: Point = "T") -> None:
        if not 1 <= order <= 5:
            raise ValueError("Extrapolation1D order must be in [1, 5].")
        self.face = face
        self.order = order
        self.point = point
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
        ghost: BoundarySlice = _adjacent_interior(field, self.face, self.point) * 0.0
        for k, c in enumerate(self._coeffs):
            # k=0 → adjacent interior, k=1 → one step further in, etc.
            ghost = ghost + c * _interior_line(field, self.face, k, self.point)
        return _set_face(field, self.face, ghost, self.point)


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
    For a node on the wall (e.g. an ``X``-point field) the wall node is set
    to ``a * phi_interior`` directly.

    Parameters
    ----------
    face : Literal["south", "north", "west", "east"]
        Domain face to update.
    coefficient : float
        Slip parameter ``a`` in ``[0, 1]``. Default is ``1.0`` (free-slip).
    point : Literal["T", "U", "V", "X"], optional
        C-grid point the field lives on (default ``"T"``).  For a node on
        the wall (``U`` / ``X`` on west / east, ``V`` / ``X`` on south /
        north) the condition is imposed on the wall node itself (see the
        module docstring).

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
    point: Point = eqx.field(static=True, default="T")

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
        interior = _adjacent_interior(field, self.face, self.point)
        if _on_wall(self.face, self.point):
            # the node is on the wall: phi_wall = a * interior
            wall = self.coefficient * interior  # [N]
            return _set_face(field, self.face, wall, self.point)
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
