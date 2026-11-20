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

from gridax._src.advection.advection import (
    Advection1D,
    Advection2D,
    Advection3D,
)
from gridax._src.advection.face_flux import (
    rusanov_flux,
    uv_center_flux,
    uv_node_flux,
)
from gridax._src.advection.flux import (
    upwind_flux,
)
from gridax._src.advection.limiters import (
    mc,
    minmod,
    superbee,
    van_leer,
)
from gridax._src.advection.linear import (
    linear_2pts,
    linear_3pts_left,
    linear_3pts_right,
    linear_4pts,
    linear_5pts_left,
    linear_5pts_right,
    linear_6pts,
)
from gridax._src.advection.reconstruct_fn import (
    plusminus,
    reconstruct,
    upwind_1pt,
    upwind_3pt,
    upwind_5pt,
)
from gridax._src.advection.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)
from gridax._src.advection.spherical_advection import (
    SphericalAdvection2D,
    SphericalAdvection3D,
)
from gridax._src.advection.weno import (
    weno_3pts,
    weno_3pts_improved,
    weno_3pts_improved_right,
    weno_3pts_right,
    weno_5pts,
    weno_5pts_improved,
    weno_5pts_improved_right,
    weno_5pts_right,
    weno_7pts,
    weno_7pts_right,
    weno_9pts,
    weno_9pts_right,
)
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
from gridax._src.operators.coriolis import (
    Coriolis2D,
    Coriolis3D,
)
from gridax._src.operators.diagnostic_operators import (
    Energetics2D,
    QGPotentialVorticity2D,
    Strain2D,
)
from gridax._src.operators.diagnostics import (
    available_potential_energy,
    bernoulli_potential,
    beta_param,
    coriolis_fn,
    coriolis_param,
    enstrophy,
    kinetic_energy,
    okubo_weiss,
    potential_enstrophy,
    potential_vorticity,
    potential_vorticity_multilayer,
    qg_potential_vorticity,
    relative_vorticity_cgrid,
    shear_strain,
    ssh_to_streamfn,
    strain_magnitude_squared,
    streamfn_to_ssh,
    stretching_term,
    sw_potential_vorticity,
    sw_potential_vorticity_multilayer,
    tensor_strain,
    total_energy,
    total_enstrophy,
    vertical_velocity,
)
from gridax._src.operators.difference import (
    Difference1D,
    Difference2D,
    Difference3D,
)
from gridax._src.operators.differentiable import smooth_abs, smooth_clamp, smooth_max
from gridax._src.operators.divergence import (
    Divergence2D,
    divergence_2d,
)
from gridax._src.operators.interpolation import (
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
)
from gridax._src.operators.jacobian import (
    ArakawaJacobian2D,
    arakawa_jacobian,
)
from gridax._src.operators.reductions import (
    area_mean,
    area_sum,
    area_weights,
    cartesian_area_mean,
    cartesian_area_sum,
    cartesian_area_weights,
    cartesian_volume_mean,
    cartesian_volume_sum,
    cartesian_volume_weights,
    masked_mean,
    masked_moments,
    masked_std,
    spherical_area_mean,
    spherical_area_sum,
    spherical_area_weights,
    spherical_volume_mean,
    spherical_volume_sum,
    spherical_volume_weights,
    volume_mean,
    volume_sum,
    volume_weights,
)
from gridax._src.operators.spherical_compound import (
    SphericalDivergence2D,
    SphericalDivergence3D,
    SphericalLaplacian2D,
    SphericalLaplacian3D,
    SphericalVorticity2D,
    SphericalVorticity3D,
)
from gridax._src.operators.spherical_difference import (
    SphericalDifference2D,
    SphericalDifference3D,
)
from gridax._src.operators.stencils import (
    avg_x_bwd_1d,
    avg_x_bwd_2d,
    avg_x_bwd_3d,
    avg_x_fwd_1d,
    avg_x_fwd_2d,
    avg_x_fwd_3d,
    avg_xbwd_ybwd_2d,
    avg_xbwd_ybwd_3d,
    avg_xbwd_yfwd_2d,
    avg_xbwd_yfwd_3d,
    avg_xfwd_ybwd_2d,
    avg_xfwd_ybwd_3d,
    avg_xfwd_yfwd_2d,
    avg_xfwd_yfwd_3d,
    avg_y_bwd_2d,
    avg_y_bwd_3d,
    avg_y_fwd_2d,
    avg_y_fwd_3d,
    avg_z_bwd_3d,
    avg_z_fwd_3d,
    diff_x_bwd_1d,
    diff_x_bwd_2d,
    diff_x_bwd_3d,
    diff_x_fwd_1d,
    diff_x_fwd_2d,
    diff_x_fwd_3d,
    diff_y_bwd_2d,
    diff_y_bwd_3d,
    diff_y_fwd_2d,
    diff_y_fwd_3d,
    diff_z_bwd_3d,
    diff_z_fwd_3d,
)
from gridax._src.operators.vorticity import (
    Vorticity2D,
    Vorticity3D,
    pv_flux_arakawa_lamb,
    pv_flux_blend,
    vorticity_flux_upwind,
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
    "diff_x_fwd_1d",
    "diff_x_bwd_1d",
    "avg_x_fwd_1d",
    "avg_x_bwd_1d",
    "diff_x_fwd_2d",
    "diff_x_bwd_2d",
    "diff_y_fwd_2d",
    "diff_y_bwd_2d",
    "avg_x_fwd_2d",
    "avg_x_bwd_2d",
    "avg_y_fwd_2d",
    "avg_y_bwd_2d",
    "diff_x_fwd_3d",
    "diff_x_bwd_3d",
    "diff_y_fwd_3d",
    "diff_y_bwd_3d",
    "diff_z_fwd_3d",
    "diff_z_bwd_3d",
    "avg_x_fwd_3d",
    "avg_x_bwd_3d",
    "avg_y_fwd_3d",
    "avg_y_bwd_3d",
    "avg_z_fwd_3d",
    "avg_z_bwd_3d",
    "avg_xfwd_yfwd_2d",
    "avg_xbwd_ybwd_2d",
    "avg_xbwd_yfwd_2d",
    "avg_xfwd_ybwd_2d",
    "avg_xfwd_yfwd_3d",
    "avg_xbwd_ybwd_3d",
    "avg_xbwd_yfwd_3d",
    "avg_xfwd_ybwd_3d",
    "interior",
    # Differentiable surrogates
    "smooth_abs",
    "smooth_clamp",
    "smooth_max",
    # Flux limiters
    "mc",
    "minmod",
    "superbee",
    "van_leer",
    # Linear reconstruction stencils
    "linear_2pts",
    "linear_3pts_left",
    "linear_3pts_right",
    "linear_4pts",
    "linear_5pts_left",
    "linear_5pts_right",
    "linear_6pts",
    # WENO reconstruction stencils
    "weno_3pts",
    "weno_3pts_improved",
    "weno_3pts_improved_right",
    "weno_3pts_right",
    "weno_5pts",
    "weno_5pts_improved",
    "weno_5pts_improved_right",
    "weno_5pts_right",
    "weno_7pts",
    "weno_7pts_right",
    "weno_9pts",
    "weno_9pts_right",
    # Functional reconstruction
    "plusminus",
    "reconstruct",
    "upwind_1pt",
    "upwind_3pt",
    "upwind_5pt",
    # Upwind face flux
    "upwind_flux",
    # Area and volume reductions, masked statistics
    "area_mean",
    "area_sum",
    "area_weights",
    "cartesian_area_mean",
    "cartesian_area_sum",
    "cartesian_area_weights",
    "cartesian_volume_mean",
    "cartesian_volume_sum",
    "cartesian_volume_weights",
    "masked_mean",
    "masked_moments",
    "masked_std",
    "spherical_area_mean",
    "spherical_area_sum",
    "spherical_area_weights",
    "spherical_volume_mean",
    "spherical_volume_sum",
    "spherical_volume_weights",
    "volume_mean",
    "volume_sum",
    "volume_weights",
    # C-grid difference operators
    "Difference1D",
    "Difference2D",
    "Difference3D",
    # C-grid interpolation operators
    "Interpolation1D",
    "Interpolation2D",
    "Interpolation3D",
    # Divergence
    "Divergence2D",
    "divergence_2d",
    # Vorticity and PV fluxes
    "Vorticity2D",
    "Vorticity3D",
    "pv_flux_arakawa_lamb",
    "pv_flux_blend",
    "vorticity_flux_upwind",
    # Arakawa Jacobian
    "ArakawaJacobian2D",
    "arakawa_jacobian",
    # Coriolis
    "Coriolis2D",
    "Coriolis3D",
    # Diagnostic operators
    "Energetics2D",
    "QGPotentialVorticity2D",
    "Strain2D",
    # Functional diagnostics
    "available_potential_energy",
    "bernoulli_potential",
    "beta_param",
    "coriolis_fn",
    "coriolis_param",
    "enstrophy",
    "kinetic_energy",
    "okubo_weiss",
    "potential_enstrophy",
    "potential_vorticity",
    "potential_vorticity_multilayer",
    "qg_potential_vorticity",
    "relative_vorticity_cgrid",
    "shear_strain",
    "ssh_to_streamfn",
    "strain_magnitude_squared",
    "streamfn_to_ssh",
    "stretching_term",
    "sw_potential_vorticity",
    "sw_potential_vorticity_multilayer",
    "tensor_strain",
    "total_energy",
    "total_enstrophy",
    "vertical_velocity",
    # Spherical differences
    "SphericalDifference2D",
    "SphericalDifference3D",
    # Spherical divergence, vorticity and Laplacian
    "SphericalDivergence2D",
    "SphericalDivergence3D",
    "SphericalLaplacian2D",
    "SphericalLaplacian3D",
    "SphericalVorticity2D",
    "SphericalVorticity3D",
    # Advection
    "Advection1D",
    "Advection2D",
    "Advection3D",
    # Reconstruction operators
    "Reconstruction1D",
    "Reconstruction2D",
    "Reconstruction3D",
    # Spherical advection
    "SphericalAdvection2D",
    "SphericalAdvection3D",
    # Face fluxes
    "rusanov_flux",
    "uv_center_flux",
    "uv_node_flux",
    # Constants
    "R_EARTH",
]
