from finitevolx._src.advection.advection import Advection1D, Advection2D, Advection3D
from finitevolx._src.advection.flux import upwind_flux
from finitevolx._src.advection.limiters import mc, minmod, superbee, van_leer
from finitevolx._src.advection.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)
from finitevolx._src.boundary.bc_1d import (
    Dirichlet1D,
    Extrapolation1D,
    Neumann1D,
    Outflow1D,
    Periodic1D,
    Reflective1D,
    Robin1D,
    Slip1D,
    Sponge1D,
)
from finitevolx._src.boundary.bc_field import FieldBCSet
from finitevolx._src.boundary.bc_set import BoundaryConditionSet
from finitevolx._src.boundary.boundary import enforce_periodic, pad_interior
from finitevolx._src.diffusion.diffusion import (
    BiharmonicDiffusion2D,
    BiharmonicDiffusion3D,
    Diffusion2D,
    Diffusion3D,
    diffusion_2d,
)
from finitevolx._src.diffusion.momentum import MomentumAdvection2D, MomentumAdvection3D
from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask, StencilCapability
from finitevolx._src.grid.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.operators.coriolis import Coriolis2D, Coriolis3D
from finitevolx._src.operators.diagnostics import (
    available_potential_energy,
    bernoulli_potential,
    enstrophy,
    kinetic_energy,
    okubo_weiss,
    potential_enstrophy,
    potential_vorticity,
    qg_potential_vorticity,
    relative_vorticity_cgrid,
    shear_strain,
    strain_magnitude_squared,
    stretching_term,
    tensor_strain,
    total_energy,
    total_enstrophy,
    vertical_velocity,
)
from finitevolx._src.operators.difference import (
    Difference1D,
    Difference2D,
    Difference3D,
)
from finitevolx._src.operators.divergence import Divergence2D, divergence_2d
from finitevolx._src.operators.geographic import (
    curl_sphere,
    diff2_lat_T,
    diff2_lon_T,
    diff_lat_T_to_V,
    diff_lat_U_to_X,
    diff_lat_V_to_T,
    diff_lon_T_to_U,
    diff_lon_U_to_T,
    diff_lon_V_to_X,
    divergence_sphere,
    geostrophic_velocity_sphere,
    laplacian_sphere,
    potential_vorticity_sphere,
)
from finitevolx._src.operators.interpolation import (
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
)
from finitevolx._src.operators.jacobian import arakawa_jacobian
from finitevolx._src.operators.vorticity import Vorticity2D, Vorticity3D
from finitevolx._src.solvers.elliptic import (
    CapacitanceSolver,
    CGInfo,
    build_capacitance_solver,
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
    make_spectral_preconditioner,
    masked_laplacian,
    solve_cg,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_helmholtz_fft,
    solve_poisson_dct,
    solve_poisson_dst,
    solve_poisson_fft,
)
from finitevolx._src.solvers.spectral_transforms import (
    dct,
    dctn,
    dst,
    dstn,
    idct,
    idctn,
    idst,
    idstn,
)
from finitevolx._src.vertical.multilayer import multilayer
from finitevolx._src.vertical.vertical_modes import (
    build_coupling_matrix,
    decompose_vertical_modes,
    layer_to_mode,
    mode_to_layer,
)

__all__ = [
    # Upwind flux dispatch
    "upwind_flux",
    # Advection
    "Advection1D",
    "Advection2D",
    "Advection3D",
    # Coriolis
    "Coriolis2D",
    "Coriolis3D",
    # Momentum advection
    "MomentumAdvection2D",
    "MomentumAdvection3D",
    # Jacobian
    "arakawa_jacobian",
    # Multilayer vmap helper
    "multilayer",
    # Grid
    "ArakawaCGrid1D",
    "ArakawaCGrid2D",
    "ArakawaCGrid3D",
    "ArakawaCGridMask",
    # Boundary condition sets and 1D BCs
    "BoundaryConditionSet",
    "Dirichlet1D",
    "Extrapolation1D",
    "FieldBCSet",
    "Neumann1D",
    "Outflow1D",
    "Periodic1D",
    "Reflective1D",
    "Robin1D",
    "Slip1D",
    "Sponge1D",
    # Finite difference
    "Difference1D",
    "Difference2D",
    "Difference3D",
    # Diffusion
    "BiharmonicDiffusion2D",
    "BiharmonicDiffusion3D",
    "Diffusion2D",
    "Diffusion3D",
    "diffusion_2d",
    # Divergence
    "Divergence2D",
    "divergence_2d",
    # Boundary helpers
    "enforce_periodic",
    "pad_interior",
    # Interpolation
    "Interpolation1D",
    "Interpolation2D",
    "Interpolation3D",
    # Reconstruction
    "Reconstruction1D",
    "Reconstruction2D",
    "Reconstruction3D",
    # Flux limiters
    "mc",
    "minmod",
    "superbee",
    "van_leer",
    # Masks
    "StencilCapability",
    # Diagnostics
    "available_potential_energy",
    "bernoulli_potential",
    "enstrophy",
    "kinetic_energy",
    "okubo_weiss",
    "potential_enstrophy",
    "potential_vorticity",
    "qg_potential_vorticity",
    "relative_vorticity_cgrid",
    "shear_strain",
    "strain_magnitude_squared",
    "stretching_term",
    "tensor_strain",
    "total_energy",
    "total_enstrophy",
    "vertical_velocity",
    # Geographic (spherical) operators
    "curl_sphere",
    "diff2_lat_T",
    "diff2_lon_T",
    "diff_lat_T_to_V",
    "diff_lat_U_to_X",
    "diff_lat_V_to_T",
    "diff_lon_T_to_U",
    "diff_lon_U_to_T",
    "diff_lon_V_to_X",
    "divergence_sphere",
    "geostrophic_velocity_sphere",
    "laplacian_sphere",
    "potential_vorticity_sphere",
    # Vorticity
    "Vorticity2D",
    "Vorticity3D",
    # Vertical coupling and mode transforms
    "build_coupling_matrix",
    "decompose_vertical_modes",
    "layer_to_mode",
    "mode_to_layer",
    # Spectral transforms
    "dct",
    "dctn",
    "dst",
    "dstn",
    "idct",
    "idctn",
    "idst",
    "idstn",
    # Elliptic eigenvalue helpers
    "dct2_eigenvalues",
    "dst1_eigenvalues",
    "fft_eigenvalues",
    # Spectral Poisson/Helmholtz solvers
    "solve_poisson_dst",
    "solve_helmholtz_dst",
    "solve_poisson_dct",
    "solve_helmholtz_dct",
    "solve_poisson_fft",
    "solve_helmholtz_fft",
    # Capacitance matrix method
    "CapacitanceSolver",
    "build_capacitance_solver",
    # Preconditioned Conjugate Gradient
    "CGInfo",
    "make_spectral_preconditioner",
    "masked_laplacian",
    "solve_cg",
]
