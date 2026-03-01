"""
Arakawa C-grid definitions for finitevolX.

Grid conventions
----------------
All arrays have total shape [Ny, Nx].  The physical interior is
(Ny-2) x (Nx-2); one ghost-cell ring on each side is reserved for
boundary conditions.

Same-index colocation
---------------------
  T[j, i]  -> cell centre         (j,     i    )
  U[j, i]  -> east face           (j,     i+1/2)
  V[j, i]  -> north face          (j+1/2, i    )
  X[j, i]  -> north-east corner   (j+1/2, i+1/2)
"""

import equinox as eqx


class ArakawaCGrid1D(eqx.Module):
    """1-D Arakawa C-grid.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).
    Lx : float
        Physical domain length in x.
    dx : float
        Grid spacing in x.
    """

    Nx: int
    Lx: float
    dx: float

    @classmethod
    def from_interior(cls, nx_interior: int, Lx: float) -> "ArakawaCGrid1D":
        """Construct grid from interior cell count.

        Parameters
        ----------
        nx_interior : int
            Number of interior (physical) cells.
        Lx : float
            Physical domain length.

        Returns
        -------
        ArakawaCGrid1D
        """
        Nx = nx_interior + 2
        dx = Lx / nx_interior
        return cls(Nx=Nx, Lx=Lx, dx=dx)


class ArakawaCGrid2D(eqx.Module):
    """2-D Arakawa C-grid.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).
    Ny : int
        Total number of cells in y (including 2 ghost cells).
    Lx : float
        Physical domain length in x.
    Ly : float
        Physical domain length in y.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Notes
    -----
    Array layout (total shape [Ny, Nx])::

        j=Ny-1  +--+--+--+  ghost row (north)
                |  |  |  |
        j=2     +--+--+--+
        j=1     +--+--+--+  physical interior  [1:-1, 1:-1]
        j=0     +--+--+--+  ghost row (south)
                i=0  ...  i=Nx-1
                ghost    ghost
                (west)   (east)

    Colocation convention::

        T[j, i]  cell centre    at  (j dx,     i dy    )
        U[j, i]  east face      at  (j dx,    (i+1/2)dy)
        V[j, i]  north face     at ((j+1/2)dx, i dy    )
        X[j, i]  NE corner      at ((j+1/2)dx,(i+1/2)dy)
    """

    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dx: float
    dy: float

    @classmethod
    def from_interior(
        cls, nx_interior: int, ny_interior: int, Lx: float, Ly: float
    ) -> "ArakawaCGrid2D":
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
        ArakawaCGrid2D
        """
        Nx = nx_interior + 2
        Ny = ny_interior + 2
        dx = Lx / nx_interior
        dy = Ly / ny_interior
        return cls(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dx=dx, dy=dy)


class ArakawaCGrid3D(eqx.Module):
    """3-D Arakawa C-grid.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).
    Ny : int
        Total number of cells in y (including 2 ghost cells).
    Nz : int
        Total number of cells in z (including 2 ghost cells).
    Lx : float
        Physical domain length in x.
    Ly : float
        Physical domain length in y.
    Lz : float
        Physical domain length in z.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dz : float
        Grid spacing in z.
    """

    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float
    dx: float
    dy: float
    dz: float

    @classmethod
    def from_interior(
        cls,
        nx_interior: int,
        ny_interior: int,
        nz_interior: int,
        Lx: float,
        Ly: float,
        Lz: float,
    ) -> "ArakawaCGrid3D":
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
        ArakawaCGrid3D
        """
        Nx = nx_interior + 2
        Ny = ny_interior + 2
        Nz = nz_interior + 2
        dx = Lx / nx_interior
        dy = Ly / ny_interior
        dz = Lz / nz_interior
        return cls(
            Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dx=dx, dy=dy, dz=dz
        )
