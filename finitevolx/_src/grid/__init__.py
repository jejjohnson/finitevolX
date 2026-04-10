"""finitevolX grid containers — abstract base, Cartesian, spherical, cylindrical."""

from finitevolx._src.grid.base import (
    ArakawaCGrid1D,
    ArakawaCGrid2D,
    ArakawaCGrid3D,
    CurvilinearGrid1D,
    CurvilinearGrid2D,
    CurvilinearGrid3D,
)
from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D

__all__ = [
    # Abstract topology
    "ArakawaCGrid1D",
    "ArakawaCGrid2D",
    "ArakawaCGrid3D",
    # Curvilinear (uniform metric)
    "CurvilinearGrid1D",
    "CurvilinearGrid2D",
    "CurvilinearGrid3D",
    # Cartesian concrete
    "CartesianGrid1D",
    "CartesianGrid2D",
    "CartesianGrid3D",
    # Spherical concrete
    "SphericalGrid2D",
    "SphericalGrid3D",
]
