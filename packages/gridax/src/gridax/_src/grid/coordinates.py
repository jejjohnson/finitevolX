"""The coordinates of a Cartesian grid's T points, as an (N, D) point array.

A point method (semi-Lagrangian advection in ``integrix.march``, a functax
interpolation plan) takes the nodes of a field as N points in D dimensions.
This is the grid-built half of that call: the T points of the whole array,
ghost ring included, in the order ``field.ravel()`` visits them.
"""

from __future__ import annotations

from jaxtyping import Float
import numpy as np

from gridax._src.grid.cartesian import CartesianGrid1D, CartesianGrid2D


def t_point_coordinates(
    grid: CartesianGrid1D | CartesianGrid2D,
) -> Float[np.ndarray, "N D"]:
    """Coordinates of every T point of a Cartesian grid, one row per point.

    T[j, i] sits at (y, x) = (j dy, i dx), the same-index convention of
    ``docs/cgrid_discretization.md``, with j and i running over the whole
    array (the ghost ring included, so cell 1 is the first interior cell).
    The columns follow the array's axes, (y, x) for ``[Ny, Nx]``, and the
    rows follow ``field.ravel()``, so a raveled T field and these points
    line up.

    Parameters
    ----------
    grid : CartesianGrid1D or CartesianGrid2D
        The grid.

    Returns
    -------
    np.ndarray
        ``[Nx, 1]`` points (x) for a 1-D grid, ``[Ny * Nx, 2]`` points
        (y, x) for a 2-D grid, in float64.

    Raises
    ------
    TypeError
        If ``grid`` is not a 1-D or 2-D Cartesian grid. A spherical grid's
        T points are its ``lat_T`` and ``lon_T`` arrays, in radians.

    Examples
    --------
    >>> import gridax
    >>> grid = gridax.CartesianGrid2D.from_interior(2, 1, 2.0, 0.5)
    >>> points = gridax.t_point_coordinates(grid)
    >>> points.shape
    (12, 2)
    >>> points[:5].tolist()
    [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.5, 0.0]]
    """
    if isinstance(grid, CartesianGrid2D):
        # y_j = j dy, x_i = i dx
        y = np.arange(grid.Ny, dtype=np.float64) * grid.dy
        x = np.arange(grid.Nx, dtype=np.float64) * grid.dx
        rows, cols = np.meshgrid(y, x, indexing="ij")
        # [Ny * Nx, 2], row-major like field.ravel()
        return np.column_stack([rows.ravel(), cols.ravel()])
    if isinstance(grid, CartesianGrid1D):
        # x_i = i dx, [Nx, 1]
        x = np.arange(grid.Nx, dtype=np.float64) * grid.dx
        return np.column_stack([x])
    raise TypeError(
        "t_point_coordinates takes a CartesianGrid1D or CartesianGrid2D, got"
        f" {type(grid).__name__}; a spherical grid's T points are its lat_T"
        " and lon_T arrays."
    )
