from __future__ import annotations

"""
Boundary condition helpers for finitevolX.

All arrays have total shape [Ny, Nx].
Ghost cells occupy the outermost ring: rows 0 and Ny-1, columns 0 and Nx-1.
The physical interior lives at [1:-1, 1:-1].
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.boundary.bc_set import BoundaryConditionSet


def pad_interior(
    field: Float[Array, "Ny Nx"], mode: str = "edge"
) -> Float[Array, "Ny Nx"]:
    """Extract physical interior and re-pad to original size.

    Strips the ghost-cell ring, then re-pads using the requested mode.
    Useful for enforcing boundary conditions by re-filling the ghost ring.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array of shape [Ny, Nx].
    mode : str, optional
        Padding mode passed to ``jnp.pad``.  Defaults to ``'edge'``
        (copy nearest interior value into ghost cells).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array of shape [Ny, Nx] with ghost cells re-filled.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> f = jnp.ones((6, 6))
    >>> pad_interior(f).shape
    (6, 6)
    """
    if mode == "edge":
        # BoundaryConditionSet.open() is zero-gradient and does not use dx/dy.
        return BoundaryConditionSet.open()(field, dx=1.0, dy=1.0)
    interior = field[1:-1, 1:-1]  # shape [Ny-2, Nx-2]
    return jnp.pad(interior, pad_width=1, mode=mode)  # shape [Ny, Nx]


def enforce_periodic(
    field: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """Fill the ghost-cell ring with periodic boundary conditions.

    Copies the last interior row/column into the opposite ghost row/column::

        ghost south  <- last interior row   (row  Ny-2)
        ghost north  <- first interior row  (row  1   )
        ghost west   <- last interior col   (col  Nx-2)
        ghost east   <- first interior col  (col  1   )

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array of shape [Ny, Nx].

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array with periodic ghost cells.
    """
    # Periodic1D wraps opposite interior values and does not use dx/dy.
    return BoundaryConditionSet.periodic()(field, dx=1.0, dy=1.0)


# ── Functional padding wrappers ──────────────────────────────────────────────


def zero_boundaries(
    field: Float[Array, "Ny Nx"],
    pad_width: int = 1,
) -> Float[Array, "Ny Nx"]:
    """Fill ghost ring with zeros (homogeneous Dirichlet).

    Strips the interior and re-pads with constant zero.  This is the
    appropriate BC for streamfunction or pressure in a rigid-wall domain.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array with ghost ring.
    pad_width : int
        Width of the ghost ring to fill (default 1).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array with zero-filled ghost ring.
    """
    interior = field[pad_width:-pad_width, pad_width:-pad_width]
    return jnp.pad(interior, pad_width=pad_width, mode="constant", constant_values=0.0)


def zero_gradient_boundaries(
    field: Float[Array, "Ny Nx"],
    pad_width: int = 1,
) -> Float[Array, "Ny Nx"]:
    """Fill ghost ring by copying nearest interior value (zero gradient / Neumann).

    Ghost cells are set to the adjacent interior value, enforcing
    ∂φ/∂n = 0 at the boundary (outflow / open boundary condition).

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array with ghost ring.
    pad_width : int
        Width of the ghost ring to fill (default 1).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array with edge-extrapolated ghost ring.
    """
    interior = field[pad_width:-pad_width, pad_width:-pad_width]
    return jnp.pad(interior, pad_width=pad_width, mode="edge")


def no_flux_boundaries(
    field: Float[Array, "Ny Nx"],
    pad_width: int = 1,
) -> Float[Array, "Ny Nx"]:
    """Fill ghost ring with zeros for normal-flux variables.

    Alias for :func:`zero_boundaries`.  Use this for velocity components
    normal to the boundary (e.g. u at east/west walls, v at north/south
    walls) to enforce zero mass flux through rigid walls.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array with ghost ring.
    pad_width : int
        Width of the ghost ring to fill (default 1).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array with zero-filled ghost ring.
    """
    return zero_boundaries(field, pad_width=pad_width)


def no_slip_boundaries(
    field: Float[Array, "Ny Nx"],
    pad_width: int = 1,
) -> Float[Array, "Ny Nx"]:
    """Fill ghost ring with sign-flipped interior values (no-slip wall).

    Sets ghost cells to the negative of the adjacent interior value::

        ghost = -interior

    This enforces zero tangential velocity at the wall midpoint:
    ``(ghost + interior) / 2 = 0``.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array with ghost ring.  Typically a tangential velocity
        component.
    pad_width : int
        Width of the ghost ring to fill (default 1).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array with no-slip ghost ring.
    """
    out = field
    for k in range(pad_width):
        # South ghost row: negate first interior row
        out = out.at[k, :].set(-field[2 * pad_width - 1 - k, :])
        # North ghost row: negate last interior row
        out = out.at[-(k + 1), :].set(-field[-(2 * pad_width - k), :])
        # West ghost col: negate first interior col
        out = out.at[:, k].set(-field[:, 2 * pad_width - 1 - k])
        # East ghost col: negate last interior col
        out = out.at[:, -(k + 1)].set(-field[:, -(2 * pad_width - k)])
    return out


def free_slip_boundaries(
    field: Float[Array, "Ny Nx"],
    pad_width: int = 1,
) -> Float[Array, "Ny Nx"]:
    """Fill ghost ring with symmetric reflection (free-slip wall).

    Sets ghost cells equal to the adjacent interior value::

        ghost = +interior

    This enforces zero normal derivative at the wall midpoint
    (∂φ/∂n = 0), allowing free tangential slip.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array with ghost ring.
    pad_width : int
        Width of the ghost ring to fill (default 1).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array with free-slip ghost ring.
    """
    out = field
    for k in range(pad_width):
        # South ghost row: copy first interior row
        out = out.at[k, :].set(field[2 * pad_width - 1 - k, :])
        # North ghost row: copy last interior row
        out = out.at[-(k + 1), :].set(field[-(2 * pad_width - k), :])
        # West ghost col: copy first interior col
        out = out.at[:, k].set(field[:, 2 * pad_width - 1 - k])
        # East ghost col: copy last interior col
        out = out.at[:, -(k + 1)].set(field[:, -(2 * pad_width - k)])
    return out


# ── Wall boundary & corner helpers ───────────────────────────────────────────


def fix_boundary_corners(
    field: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """Average corner ghost cells from their two face-adjacent ghost neighbours.

    Corner ghost cells ``[0,0]``, ``[0,-1]``, ``[-1,0]``, ``[-1,-1]`` are
    set to the average of the two adjacent edge ghost cells.  This produces
    a smooth corner value consistent with the edge boundary conditions.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array whose edge ghost cells are already filled.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array with averaged corner ghost cells.
    """
    out = field
    # SW corner: average of south-edge and west-edge ghost cells
    out = out.at[0, 0].set(0.5 * (field[0, 1] + field[1, 0]))
    # SE corner
    out = out.at[0, -1].set(0.5 * (field[0, -2] + field[1, -1]))
    # NW corner
    out = out.at[-1, 0].set(0.5 * (field[-1, 1] + field[-2, 0]))
    # NE corner
    out = out.at[-1, -1].set(0.5 * (field[-1, -2] + field[-2, -1]))
    return out


def wall_boundaries(
    field: Float[Array, "Ny Nx"],
    grid: str = "h",
) -> Float[Array, "Ny Nx"]:
    """Apply wall boundary conditions appropriate for a C-grid staggering.

    A composite helper that selects the correct ghost-cell treatment based
    on which C-grid variable the field represents:

    * ``'h'`` (tracer): zero-gradient (free-slip) + corner averaging.
    * ``'u'`` (x-velocity): zero at east/west walls (no-flux normal),
      free-slip at north/south walls, + corner averaging.
    * ``'v'`` (y-velocity): zero at north/south walls (no-flux normal),
      free-slip at east/west walls, + corner averaging.
    * ``'q'`` (vorticity / psi): no-slip at all walls + corner averaging.

    Parameters
    ----------
    field : Float[Array, "Ny Nx"]
        Input array with ghost ring.
    grid : {'h', 'u', 'v', 'q'}
        Which C-grid staggering the field represents.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Array with wall boundary conditions applied.
    """
    if grid == "h":
        # Tracer: zero-gradient (outflow/free-slip)
        out = free_slip_boundaries(field)
    elif grid == "u":
        # u-velocity: zero at E/W walls (normal), free-slip at N/S
        out = field
        # North/south: free-slip (copy interior)
        out = out.at[0, :].set(field[1, :])
        out = out.at[-1, :].set(field[-2, :])
        # East/west: no-flux (zero normal velocity)
        out = out.at[:, 0].set(0.0)
        out = out.at[:, -1].set(0.0)
    elif grid == "v":
        # v-velocity: zero at N/S walls (normal), free-slip at E/W
        out = field
        # East/west: free-slip (copy interior)
        out = out.at[:, 0].set(field[:, 1])
        out = out.at[:, -1].set(field[:, -2])
        # North/south: no-flux (zero normal velocity)
        out = out.at[0, :].set(0.0)
        out = out.at[-1, :].set(0.0)
    elif grid == "q":
        # Vorticity: no-slip at all walls
        out = no_slip_boundaries(field)
    else:
        raise ValueError(f"grid must be 'h', 'u', 'v', or 'q', got {grid!r}")

    return fix_boundary_corners(out)
