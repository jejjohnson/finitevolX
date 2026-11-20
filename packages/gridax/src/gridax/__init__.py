"""gridax: structured grids and land / ocean masks as data, on NumPy.

gridax holds the arrays that describe a grid: Arakawa C-grid containers
(Cartesian and spherical, 1-D to 3-D, with one ghost ring) and the staggered
wet / dry masks built from a T-point mask. Everything is built with NumPy
(and SciPy for the mask morphology), so a NumPy-only install can describe the
grid a model ran on. When JAX is importable every class is also a JAX pytree,
so grids and masks pass through ``jax.jit`` and ``jax.vmap``.

Operations applied to a field (differences, advection, solvers) live in
gridax-ops, which builds on this package.
"""

from gridax._src.constants import R_EARTH
from gridax._src.grid import (
    ArakawaCGrid1D,
    ArakawaCGrid2D,
    ArakawaCGrid3D,
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
    CurvilinearGrid1D,
    CurvilinearGrid2D,
    CurvilinearGrid3D,
    SphericalGrid2D,
    SphericalGrid3D,
)
from gridax._src.mask import (
    Mask1D,
    Mask2D,
    Mask3D,
    StencilCapability1D,
    StencilCapability2D,
    StencilCapability3D,
    count_contiguous,
    dilate_mask,
    h_from_pooled,
    make_sponge,
    pool_bool,
)

__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    # Grids: topology, uniform metric, concrete coordinate systems
    "ArakawaCGrid1D",
    "ArakawaCGrid2D",
    "ArakawaCGrid3D",
    "CurvilinearGrid1D",
    "CurvilinearGrid2D",
    "CurvilinearGrid3D",
    "CartesianGrid1D",
    "CartesianGrid2D",
    "CartesianGrid3D",
    "SphericalGrid2D",
    "SphericalGrid3D",
    # Masks
    "Mask1D",
    "Mask2D",
    "Mask3D",
    "StencilCapability1D",
    "StencilCapability2D",
    "StencilCapability3D",
    # Mask construction primitives
    "count_contiguous",
    "dilate_mask",
    "h_from_pooled",
    "make_sponge",
    "pool_bool",
    # Constants
    "R_EARTH",
]
