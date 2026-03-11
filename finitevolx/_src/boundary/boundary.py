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
