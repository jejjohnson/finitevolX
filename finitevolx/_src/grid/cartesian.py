"""Cartesian Arakawa C-grid containers (1-D, 2-D, 3-D).

These are the standard uniform-spacing Cartesian grids used by the
core finite-volume operators in finitevolX.  They are concrete
subclasses of :class:`CurvilinearGrid1D` / ``2D`` / ``3D`` and add no
new fields — only the constructor :meth:`from_interior` and (in 3-D)
the :meth:`horizontal_grid` helper.
"""

from finitevolx._src.grid.base import (
    CurvilinearGrid1D,
    CurvilinearGrid2D,
    CurvilinearGrid3D,
)


class CartesianGrid1D(CurvilinearGrid1D):
    """1-D Cartesian Arakawa C-grid (uniform spacing).

    All field arrays have total shape ``[Nx]``.  The physical interior
    is ``Nx-2`` cells (slice ``[1:-1]``); one ghost cell on each end is
    reserved for boundary conditions.

    Notes
    -----
    Colocation convention (same-index rule)::

        T[i]  cell centre  at   i        * dx
        U[i]  east face    at  (i + 1/2) * dx

    The 1-D grid has only T- and U-points (no V/X corners).  Array
    index ``[i]`` encodes the **west** edge of the stencil
    neighbourhood, so ``U[i]`` is the face immediately east of ``T[i]``.
    """

    @classmethod
    def from_interior(cls, nx_interior: int, Lx: float) -> "CartesianGrid1D":
        """Construct grid from interior cell count.

        Parameters
        ----------
        nx_interior : int
            Number of interior (physical) cells.
        Lx : float
            Physical domain length.

        Returns
        -------
        CartesianGrid1D
        """
        Nx = nx_interior + 2
        dx = Lx / nx_interior
        return cls(Nx=Nx, Lx=Lx, dx=dx)


class CartesianGrid2D(CurvilinearGrid2D):
    """2-D Cartesian Arakawa C-grid (uniform spacing).

    All field arrays have total shape ``[Ny, Nx]``.  The physical
    interior is ``(Ny-2) x (Nx-2)`` (slice ``[1:-1, 1:-1]``); one
    ghost-cell ring on each side is reserved for boundary conditions.

    Notes
    -----
    Colocation convention (same-index rule)::

        T[j, i]  cell centre  at  ( i        * dx,  j        * dy )
        U[j, i]  east face    at  ((i + 1/2) * dx,  j        * dy )
        V[j, i]  north face   at  ( i        * dx, (j + 1/2) * dy )
        X[j, i]  NE corner    at  ((i + 1/2) * dx, (j + 1/2) * dy )

    The "same-index" rule means array index ``[j, i]`` encodes the
    **south-west** corner of the stencil neighbourhood::

           X[j,i] ---V[j,i]--- X[j,i+1]
             |           |           |
           U[j,i]     T[j,i]    U[j,i+1]
             |           |           |
          X[j-1,i]--V[j-1,i]--X[j-1,i+1]
    """

    @classmethod
    def from_interior(
        cls, nx_interior: int, ny_interior: int, Lx: float, Ly: float
    ) -> "CartesianGrid2D":
        """Construct grid from interior cell counts.

        Parameters
        ----------
        nx_interior : int
            Number of interior (physical) cells in x.
        ny_interior : int
            Number of interior (physical) cells in y.
        Lx : float
            Physical domain length in x.
        Ly : float
            Physical domain length in y.

        Returns
        -------
        CartesianGrid2D
        """
        Nx = nx_interior + 2
        Ny = ny_interior + 2
        dx = Lx / nx_interior
        dy = Ly / ny_interior
        return cls(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=dx, dy=dy)


class CartesianGrid3D(CurvilinearGrid3D):
    """3-D Cartesian Arakawa C-grid (uniform spacing).

    All field arrays have total shape ``[Nz, Ny, Nx]``.  The physical
    interior is ``(Nz-2) x (Ny-2) x (Nx-2)`` (slice
    ``[1:-1, 1:-1, 1:-1]``); one ghost-cell ring on each side of every
    axis is reserved for boundary conditions.

    The horizontal staggering (T, U, V, X) follows the 2-D convention
    applied independently at each z-level — i.e. ``Difference3D`` and
    related operators act on the ``(y, x)`` plane for fixed ``k``.  The
    vertical axis is **not** staggered within this same-shape array
    convention; W-point fields (vertical velocity at z-interfaces) live
    in a separate ``[Nz+1, Ny, Nx]`` array — see
    ``finitevolx._src.operators.diagnostics.vertical_velocity``.

    Notes
    -----
    Colocation convention (same-index rule, horizontal staggering only)::

        T[k, j, i]  cell centre  at  ( i        * dx,  j        * dy,  k * dz )
        U[k, j, i]  east face    at  ((i + 1/2) * dx,  j        * dy,  k * dz )
        V[k, j, i]  north face   at  ( i        * dx, (j + 1/2) * dy,  k * dz )
        X[k, j, i]  NE corner    at  ((i + 1/2) * dx, (j + 1/2) * dy,  k * dz )

    All four point types share the same z-coordinate ``k * dz``;
    vertical staggering (W-points at ``(k + 1/2) * dz``) is handled by
    separate ``[Nz+1, Ny, Nx]`` fields outside this grid container.
    """

    @classmethod
    def from_interior(
        cls,
        nx_interior: int,
        ny_interior: int,
        nz_interior: int,
        Lx: float,
        Ly: float,
        Lz: float,
    ) -> "CartesianGrid3D":
        """Construct grid from interior cell counts.

        Parameters
        ----------
        nx_interior : int
            Number of interior cells in x.
        ny_interior : int
            Number of interior cells in y.
        nz_interior : int
            Number of interior cells in z.
        Lx : float
            Physical domain length in x.
        Ly : float
            Physical domain length in y.
        Lz : float
            Physical domain length in z.

        Returns
        -------
        CartesianGrid3D
        """
        Nx = nx_interior + 2
        Ny = ny_interior + 2
        Nz = nz_interior + 2
        dx = Lx / nx_interior
        dy = Ly / ny_interior
        dz = Lz / nz_interior
        return cls(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dx=dx, dy=dy, dz=dz)

    def horizontal_grid(self) -> "CartesianGrid2D":
        """Extract the horizontal 2-D grid from this 3-D grid.

        Returns
        -------
        CartesianGrid2D
            A 2-D grid with the same Nx, Ny, Lx, Ly, dx, dy.
        """
        return CartesianGrid2D(
            Nx=self.Nx, Ny=self.Ny, Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy
        )
