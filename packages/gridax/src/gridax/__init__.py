"""gridax: structured grids and land / ocean masks as data, on NumPy.

gridax holds the arrays that describe a grid: Arakawa C-grid containers
(Cartesian and spherical, 1-D to 3-D, with one ghost ring) and the staggered
wet / dry masks built from a T-point mask. Everything is built with NumPy
(and SciPy for the mask morphology), so a NumPy-only install can describe the
grid a model ran on. When JAX is importable every class is also a JAX pytree,
so grids and masks pass through ``jax.jit`` and ``jax.vmap``.

:mod:`gridax.analysis` adds the backend-agnostic analysis kernels on
collocated grids: finite differences and axis remapping, written against the
array API so they run on NumPy, JAX or any other backend. The staggered
C-grid operations (differences on T / U / V / X points, advection, boundary
conditions, solvers) live in gridax-jax, which builds on this package.
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
from gridax._src.operators._ghost import interior
from gridax._src.operators.differentiable import smooth_abs, smooth_clamp, smooth_max
from gridax._src.operators.stencils import (
    avg_x_bwd,
    avg_x_bwd_1d,
    avg_x_bwd_3d,
    avg_x_fwd,
    avg_x_fwd_1d,
    avg_x_fwd_3d,
    avg_xbwd_yfwd,
    avg_xfwd_ybwd,
    avg_xy_bwd,
    avg_xy_fwd,
    avg_y_bwd,
    avg_y_bwd_3d,
    avg_y_fwd,
    avg_y_fwd_3d,
    diff_x_bwd,
    diff_x_bwd_1d,
    diff_x_bwd_3d,
    diff_x_fwd,
    diff_x_fwd_1d,
    diff_x_fwd_3d,
    diff_y_bwd,
    diff_y_bwd_3d,
    diff_y_fwd,
    diff_y_fwd_3d,
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
    # Raw C-grid stencils (index arithmetic, no metric)
    "avg_x_bwd",
    "avg_x_bwd_1d",
    "avg_x_bwd_3d",
    "avg_x_fwd",
    "avg_x_fwd_1d",
    "avg_x_fwd_3d",
    "avg_xbwd_yfwd",
    "avg_xfwd_ybwd",
    "avg_xy_bwd",
    "avg_xy_fwd",
    "avg_y_bwd",
    "avg_y_bwd_3d",
    "avg_y_fwd",
    "avg_y_fwd_3d",
    "diff_x_bwd",
    "diff_x_bwd_1d",
    "diff_x_bwd_3d",
    "diff_x_fwd",
    "diff_x_fwd_1d",
    "diff_x_fwd_3d",
    "diff_y_bwd",
    "diff_y_bwd_3d",
    "diff_y_fwd",
    "diff_y_fwd_3d",
    "interior",
    # Differentiable surrogates
    "smooth_abs",
    "smooth_clamp",
    "smooth_max",
    # Constants
    "R_EARTH",
]
