"""Abstract base classes for finitevolX grid containers.

This module defines the layered abstraction shared by every concrete
grid type (Cartesian, spherical, cylindrical, ...):

    ArakawaCGrid1D / 2D / 3D       — pure C-grid topology (Nx, Ny, Nz +
                                     ghost-cell convention).  No metric.
        ↓
    CurvilinearGrid1D / 2D / 3D    — adds a uniform metric (Lx, Ly, Lz,
                                     dx, dy, dz).  Coordinate-system
                                     agnostic; serves as the parent for
                                     all concrete grids.
        ↓
    CartesianGrid* (cartesian.py)
    SphericalGrid* (spherical.py)
    CylindricalGrid* (cylindrical.py)

The split lets operators that only need topology (e.g. shape and
ghost-cell handling) accept ``ArakawaCGrid*``, operators that need a
generic uniform spacing accept ``CurvilinearGrid*``, and operators
that require coordinate-system-specific metric (e.g. ``cos(lat)``)
take the concrete subclass directly.
"""

import equinox as eqx

# ----------------------------------------------------------------------
# Layer 1 — pure C-grid topology (no metric)
# ----------------------------------------------------------------------


class ArakawaCGrid1D(eqx.Module):
    """1-D Arakawa C-grid topology (no metric).

    Records only the integer dimension and the ghost-cell convention;
    no physical spacing.  Concrete subclasses (``CartesianGrid1D``,
    etc.) attach a metric.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).

    Notes
    -----
    Array layout (total length ``Nx``)::

        i= 0    1    2   ...   Nx-2  Nx-1
          [g]  [ ]  [ ]  ...    [ ]  [g]
           ^    \\______ interior ______/    ^
        ghost                              ghost
        (west)                             (east)

    The 1-D C-grid uses two staggering locations (T-cell centres and
    U-east-faces); see ``CartesianGrid1D`` for the colocation rule.
    """

    Nx: int


class ArakawaCGrid2D(eqx.Module):
    """2-D Arakawa C-grid topology (no metric).

    Records only the integer dimensions and the ghost-cell convention;
    no physical spacing.  Concrete subclasses (``CartesianGrid2D``,
    ``SphericalGrid2D``, ...) attach a metric.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).
    Ny : int
        Total number of cells in y (including 2 ghost cells).

    Notes
    -----
    Array layout (total shape ``[Ny, Nx]``)::

        j=Ny-1  +--+--+--+--+  ghost row (north)
                |  |  |  |  |
        j=Ny-2  +--+--+--+--+
                |  |  |  |  |  physical interior  [1:-1, 1:-1]
        j=1     +--+--+--+--+
                |  |  |  |  |
        j=0     +--+--+--+--+  ghost row (south)
                i=0  ...    i=Nx-1
                ghost      ghost
                (west)     (east)

    The 2-D C-grid uses four staggering locations (T, U, V, X); array
    index ``[j, i]`` encodes the **south-west** corner of the stencil
    neighbourhood.  See ``CartesianGrid2D`` for the colocation rule.
    """

    Nx: int
    Ny: int


class ArakawaCGrid3D(eqx.Module):
    """3-D Arakawa C-grid topology (no metric).

    Records only the integer dimensions and the ghost-cell convention;
    no physical spacing.  Concrete subclasses (``CartesianGrid3D``,
    ``SphericalGrid3D``, ...) attach a metric.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).
    Ny : int
        Total number of cells in y (including 2 ghost cells).
    Nz : int
        Total number of cells in z (including 2 ghost cells).

    Notes
    -----
    Array layout (total shape ``[Nz, Ny, Nx]``).  Each z-level is a
    2-D slab identical to ``ArakawaCGrid2D``::

        k=Nz-1  +================+  ghost slab (top)
                |   2-D slab     |
        k=Nz-2  +================+
                |   2-D slab     |  physical interior  [1:-1, 1:-1, 1:-1]
        k=1     +================+
                |   2-D slab     |
        k=0     +================+  ghost slab (bottom)

    Only the horizontal axes are staggered (T, U, V, X share the same
    z-level); vertical staggering for W-fields uses a separate
    ``[Nz+1, Ny, Nx]`` array — see
    ``finitevolx._src.operators.diagnostics.vertical_velocity``.
    """

    Nx: int
    Ny: int
    Nz: int


# ----------------------------------------------------------------------
# Layer 2 — uniform curvilinear metric (Lx, dx, ...)
# ----------------------------------------------------------------------


class CurvilinearGrid1D(ArakawaCGrid1D):
    """1-D curvilinear C-grid with a uniform metric.

    Adds physical extent (``Lx``) and uniform grid spacing (``dx``) to
    :class:`ArakawaCGrid1D`.  Concrete subclasses (``CartesianGrid1D``,
    cylindrical 1-D, ...) inherit from this layer and provide their
    coordinate-system-specific construction logic.

    Parameters
    ----------
    Nx : int
        Total number of cells in x (including 2 ghost cells).
    Lx : float
        Physical domain length in x.
    dx : float
        Grid spacing in x.
    """

    Lx: float
    dx: float


class CurvilinearGrid2D(ArakawaCGrid2D):
    """2-D curvilinear C-grid with a uniform metric.

    Adds physical extents (``Lx``, ``Ly``) and uniform grid spacings
    (``dx``, ``dy``) to :class:`ArakawaCGrid2D`.  Concrete subclasses
    (``CartesianGrid2D``, ``SphericalGrid2D``, ``CylindricalGrid2D``)
    inherit from this layer and provide their coordinate-system-specific
    construction logic.

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
    """

    Lx: float
    Ly: float
    dx: float
    dy: float


class CurvilinearGrid3D(ArakawaCGrid3D):
    """3-D curvilinear C-grid with a uniform metric.

    Adds physical extents (``Lx``, ``Ly``, ``Lz``) and uniform grid
    spacings (``dx``, ``dy``, ``dz``) to :class:`ArakawaCGrid3D`.
    Concrete subclasses (``CartesianGrid3D``, ``SphericalGrid3D``,
    ``CylindricalGrid3D``) inherit from this layer and provide their
    coordinate-system-specific construction logic.

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

    Lx: float
    Ly: float
    Lz: float
    dx: float
    dy: float
    dz: float
