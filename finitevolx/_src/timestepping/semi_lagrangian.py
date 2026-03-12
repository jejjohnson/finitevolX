"""Pure functional semi-Lagrangian advection step.

Traces characteristic curves backward in time and interpolates the old field
at the departure points.  Unlike Eulerian methods, the semi-Lagrangian scheme
is **unconditionally stable** — the CFL number can exceed 1.

Uses :func:`jax.scipy.ndimage.map_coordinates` for interpolation.

References
----------
- Staniforth & Côté (1991) — Semi-Lagrangian integration schemes.
- ECMWF IFS uses semi-Lagrangian as its primary advection method.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def semi_lagrangian_step(
    field: jax.Array,
    u: jax.Array,
    v: jax.Array,
    dx: float,
    dy: float,
    dt: float,
    interp_order: int = 1,
    bc: str = "periodic",
) -> jax.Array:
    """Advect a 2D scalar field using semi-Lagrangian backtracking.

    Algorithm::

        1. Compute departure points: x_dep = x_i - u * dt,
                                      y_dep = y_j - v * dt
        2. Interpolate ``field`` at the departure points.
        3. Return interpolated values as the new field.

    Parameters
    ----------
    field : Array[Ny, Nx]
        Scalar field to advect.
    u, v : Array[Ny, Nx]
        Velocity components at the same grid points as ``field``, in
        **physical units** (m/s).
    dx, dy : float
        Grid spacing in x and y (m).
    dt : float
        Timestep (s).
    interp_order : int, optional
        Interpolation order passed to :func:`jax.scipy.ndimage.map_coordinates`.
        Currently JAX only supports ``order <= 1``.  Default 1.
    bc : str, optional
        Boundary handling: ``"periodic"`` (wrap) or ``"edge"``
        (Neumann-like clamp).  Default ``"periodic"``.

    Returns
    -------
    Array[Ny, Nx]
        Advected field.
    """
    if bc not in {"periodic", "edge"}:
        raise ValueError(f"bc must be 'periodic' or 'edge', got {bc!r}")
    if interp_order not in {0, 1}:
        raise ValueError(
            f"interp_order must be 0 or 1 (JAX limitation), got {interp_order}"
        )

    ny, nx = field.shape

    # Target grid coordinates (in index space)
    y_coords, x_coords = jnp.meshgrid(
        jnp.arange(ny, dtype=field.dtype),
        jnp.arange(nx, dtype=field.dtype),
        indexing="ij",
    )

    # Departure points in index space
    x_dep = x_coords - u * dt / dx
    y_dep = y_coords - v * dt / dy

    mode = "wrap" if bc == "periodic" else "nearest"

    return jax.scipy.ndimage.map_coordinates(
        field,
        [y_dep, x_dep],
        order=interp_order,
        mode=mode,
    )
