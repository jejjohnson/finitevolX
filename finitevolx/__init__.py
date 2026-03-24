from finitevolx._src.advection.advection import Advection1D, Advection2D, Advection3D
from finitevolx._src.advection.face_flux import uv_center_flux, uv_node_flux
from finitevolx._src.advection.flux import upwind_flux
from finitevolx._src.advection.limiters import mc, minmod, superbee, van_leer
from finitevolx._src.advection.linear import (
    linear_2pts,
    linear_3pts_left,
    linear_3pts_right,
    linear_4pts,
    linear_5pts_left,
    linear_5pts_right,
    linear_6pts,
)
from finitevolx._src.advection.reconstruct_fn import (
    plusminus,
    reconstruct,
    upwind_1pt,
    upwind_3pt,
    upwind_5pt,
)
from finitevolx._src.advection.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)
from finitevolx._src.advection.weno import (
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
from finitevolx._src.boundary.boundary import (
    enforce_periodic,
    fix_boundary_corners,
    free_slip_boundaries,
    no_flux_boundaries,
    no_slip_boundaries,
    pad_interior,
    wall_boundaries,
    zero_boundaries,
    zero_gradient_boundaries,
)
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
from finitevolx._src.grid.spherical_grid import (
    SphericalArakawaCGrid2D,
    SphericalArakawaCGrid3D,
)
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators.coriolis import Coriolis2D, Coriolis3D
from finitevolx._src.operators.diagnostics import (
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
from finitevolx._src.operators.difference import (
    Difference1D,
    Difference2D,
    Difference3D,
)
from finitevolx._src.operators.divergence import Divergence2D, divergence_2d
from finitevolx._src.operators.interpolation import (
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
)
from finitevolx._src.operators.jacobian import arakawa_jacobian
from finitevolx._src.operators.spherical_compound import (
    SphericalDivergence2D,
    SphericalDivergence3D,
    SphericalLaplacian2D,
    SphericalLaplacian3D,
    SphericalVorticity2D,
    SphericalVorticity3D,
)
from finitevolx._src.operators.spherical_difference import (
    SphericalDifference2D,
    SphericalDifference3D,
)
from finitevolx._src.operators.stencils import (
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
from finitevolx._src.operators.vorticity import Vorticity2D, Vorticity3D
from finitevolx._src.solvers.elliptic import (
    # Types
    BoundaryCondition,
    # Capacitance
    CapacitanceSolver,
    CGInfo,
    # Solver classes
    DirichletHelmholtzSolver2D,
    MixedBCHelmholtzSolver2D,
    MixedBCHelmholtzSolver3D,
    NeumannHelmholtzSolver2D,
    RegularNeumannHelmholtzSolver2D,
    SpectralHelmholtzSolver1D,
    SpectralHelmholtzSolver2D,
    SpectralHelmholtzSolver3D,
    StaggeredDirichletHelmholtzSolver2D,
    build_capacitance_solver,
    # Eigenvalue functions (FD2)
    dct1_eigenvalues,
    # Eigenvalue functions (pseudo-spectral)
    dct1_eigenvalues_ps,
    dct2_eigenvalues,
    dct2_eigenvalues_ps,
    dct3_eigenvalues,
    dct3_eigenvalues_ps,
    dct4_eigenvalues,
    dct4_eigenvalues_ps,
    dst1_eigenvalues,
    dst1_eigenvalues_ps,
    dst2_eigenvalues,
    dst2_eigenvalues_ps,
    dst3_eigenvalues,
    dst3_eigenvalues_ps,
    dst4_eigenvalues,
    dst4_eigenvalues_ps,
    fft_eigenvalues,
    fft_eigenvalues_ps,
    # Preconditioners
    make_multigrid_preconditioner,
    make_nystrom_preconditioner,
    make_preconditioner,
    make_spectral_preconditioner,
    # CG solver
    masked_laplacian,
    # RHS modification
    modify_rhs_1d,
    modify_rhs_2d,
    modify_rhs_3d,
    # Convenience wrappers
    pressure_from_divergence,
    pv_inversion,
    solve_cg,
    # Generic per-axis BC solvers
    solve_helmholtz_2d,
    solve_helmholtz_3d,
    # Helmholtz solvers — 2D
    solve_helmholtz_dct,
    solve_helmholtz_dct1,
    # Helmholtz solvers — 1D
    solve_helmholtz_dct1_1d,
    # Helmholtz solvers — 3D
    solve_helmholtz_dct1_3d,
    solve_helmholtz_dct2,
    solve_helmholtz_dct2_1d,
    solve_helmholtz_dct2_3d,
    solve_helmholtz_dst,
    solve_helmholtz_dst1,
    solve_helmholtz_dst1_1d,
    solve_helmholtz_dst1_3d,
    solve_helmholtz_dst2,
    solve_helmholtz_dst2_1d,
    solve_helmholtz_dst2_3d,
    solve_helmholtz_fft,
    solve_helmholtz_fft_1d,
    solve_helmholtz_fft_3d,
    solve_poisson_2d,
    solve_poisson_3d,
    # Poisson solvers — 2D
    solve_poisson_dct,
    solve_poisson_dct1,
    # Poisson solvers — 1D
    solve_poisson_dct1_1d,
    # Poisson solvers — 3D
    solve_poisson_dct1_3d,
    solve_poisson_dct2,
    solve_poisson_dct2_1d,
    solve_poisson_dct2_3d,
    solve_poisson_dst,
    solve_poisson_dst1,
    solve_poisson_dst1_1d,
    solve_poisson_dst1_3d,
    solve_poisson_dst2,
    solve_poisson_dst2_1d,
    solve_poisson_dst2_3d,
    solve_poisson_fft,
    solve_poisson_fft_1d,
    solve_poisson_fft_3d,
    streamfunction_from_vorticity,
)
from finitevolx._src.solvers.multigrid import (
    MultigridSolver,
    build_multigrid_solver,
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
from finitevolx._src.solvers.tridiagonal import (
    solve_tridiagonal,
    solve_tridiagonal_batched,
)
from finitevolx._src.timestepping._solve import solve_ocean_pde
from finitevolx._src.timestepping.diffrax_solvers import (
    IMEX_SSP2,
    RK3SSP,
    SSP_RK2,
    SSP_RK104,
    AB2Solver,
    ForwardEulerDfx,
    LeapfrogRAFSolver,
    RK2Heun,
    RK4Classic,
    SemiLagrangianSolver,
    SplitExplicitRKSolver,
)
from finitevolx._src.timestepping.explicit_rk import (
    euler_step,
    heun_step,
    rk3_ssp_step,
    rk4_step,
)
from finitevolx._src.timestepping.imex import imex_ssp2_step
from finitevolx._src.timestepping.multistep import (
    ab2_step,
    ab3_step,
    leapfrog_raf_step,
)
from finitevolx._src.timestepping.semi_lagrangian import semi_lagrangian_step
from finitevolx._src.timestepping.split_explicit import split_explicit_step
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
    # Raw face fluxes
    "uv_center_flux",
    "uv_node_flux",
    # Advection
    "Advection1D",
    "Advection2D",
    "Advection3D",
    # WENO stencil functions
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
    # Linear stencil functions
    "linear_2pts",
    "linear_3pts_left",
    "linear_3pts_right",
    "linear_4pts",
    "linear_5pts_left",
    "linear_5pts_right",
    "linear_6pts",
    # Functional reconstruction dispatcher
    "reconstruct",
    "plusminus",
    "upwind_1pt",
    "upwind_3pt",
    "upwind_5pt",
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
    "SphericalArakawaCGrid2D",
    "SphericalArakawaCGrid3D",
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
    # Spherical finite difference
    "SphericalDifference2D",
    "SphericalDifference3D",
    # Raw stencils (Layer 1 — no metric scaling)
    # Difference stencils
    "diff_x_fwd",
    "diff_x_bwd",
    "diff_y_fwd",
    "diff_y_bwd",
    "diff_x_fwd_1d",
    "diff_x_bwd_1d",
    "diff_x_fwd_3d",
    "diff_x_bwd_3d",
    "diff_y_fwd_3d",
    "diff_y_bwd_3d",
    # Averaging stencils
    "avg_x_fwd",
    "avg_x_bwd",
    "avg_y_fwd",
    "avg_y_bwd",
    "avg_xy_fwd",
    "avg_xy_bwd",
    "avg_xbwd_yfwd",
    "avg_xfwd_ybwd",
    "avg_x_fwd_1d",
    "avg_x_bwd_1d",
    "avg_x_fwd_3d",
    "avg_x_bwd_3d",
    "avg_y_fwd_3d",
    "avg_y_bwd_3d",
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
    "fix_boundary_corners",
    "free_slip_boundaries",
    "interior",
    "no_flux_boundaries",
    "no_slip_boundaries",
    "pad_interior",
    "wall_boundaries",
    "zero_boundaries",
    "zero_gradient_boundaries",
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
    "potential_vorticity_multilayer",
    "qg_potential_vorticity",
    "relative_vorticity_cgrid",
    "shear_strain",
    "strain_magnitude_squared",
    "stretching_term",
    "tensor_strain",
    "total_energy",
    "total_enstrophy",
    "vertical_velocity",
    # Shallow-water PV
    "sw_potential_vorticity",
    "sw_potential_vorticity_multilayer",
    # Coriolis / beta-plane constructors
    "coriolis_fn",
    "coriolis_param",
    "beta_param",
    # Streamfunction / SSH conversion
    "streamfn_to_ssh",
    "ssh_to_streamfn",
    # Spherical compound operators
    "SphericalDivergence2D",
    "SphericalDivergence3D",
    "SphericalLaplacian2D",
    "SphericalLaplacian3D",
    "SphericalVorticity2D",
    "SphericalVorticity3D",
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
    # Spectral BC type
    "BoundaryCondition",
    # Spectral solver classes
    "DirichletHelmholtzSolver2D",
    "MixedBCHelmholtzSolver2D",
    "MixedBCHelmholtzSolver3D",
    "NeumannHelmholtzSolver2D",
    "RegularNeumannHelmholtzSolver2D",
    "SpectralHelmholtzSolver1D",
    "SpectralHelmholtzSolver2D",
    "SpectralHelmholtzSolver3D",
    "StaggeredDirichletHelmholtzSolver2D",
    # Elliptic eigenvalue helpers (FD2)
    "dct1_eigenvalues",
    "dct2_eigenvalues",
    "dct3_eigenvalues",
    "dct4_eigenvalues",
    "dst1_eigenvalues",
    "dst2_eigenvalues",
    "dst3_eigenvalues",
    "dst4_eigenvalues",
    "fft_eigenvalues",
    # Elliptic eigenvalue helpers (pseudo-spectral)
    "dct1_eigenvalues_ps",
    "dct2_eigenvalues_ps",
    "dct3_eigenvalues_ps",
    "dct4_eigenvalues_ps",
    "dst1_eigenvalues_ps",
    "dst2_eigenvalues_ps",
    "dst3_eigenvalues_ps",
    "dst4_eigenvalues_ps",
    "fft_eigenvalues_ps",
    # RHS modification for inhomogeneous BCs
    "modify_rhs_1d",
    "modify_rhs_2d",
    "modify_rhs_3d",
    # Generic per-axis BC solvers
    "solve_helmholtz_2d",
    "solve_helmholtz_3d",
    "solve_poisson_2d",
    "solve_poisson_3d",
    # Spectral Helmholtz solvers — 2D
    "solve_helmholtz_dct",
    "solve_helmholtz_dct1",
    "solve_helmholtz_dct2",
    "solve_helmholtz_dst",
    "solve_helmholtz_dst1",
    "solve_helmholtz_dst2",
    "solve_helmholtz_fft",
    # Spectral Helmholtz solvers — 1D
    "solve_helmholtz_dct1_1d",
    "solve_helmholtz_dct2_1d",
    "solve_helmholtz_dst1_1d",
    "solve_helmholtz_dst2_1d",
    "solve_helmholtz_fft_1d",
    # Spectral Helmholtz solvers — 3D
    "solve_helmholtz_dct1_3d",
    "solve_helmholtz_dct2_3d",
    "solve_helmholtz_dst1_3d",
    "solve_helmholtz_dst2_3d",
    "solve_helmholtz_fft_3d",
    # Spectral Poisson solvers — 2D
    "solve_poisson_dct",
    "solve_poisson_dct1",
    "solve_poisson_dct2",
    "solve_poisson_dst",
    "solve_poisson_dst1",
    "solve_poisson_dst2",
    "solve_poisson_fft",
    # Spectral Poisson solvers — 1D
    "solve_poisson_dct1_1d",
    "solve_poisson_dct2_1d",
    "solve_poisson_dst1_1d",
    "solve_poisson_dst2_1d",
    "solve_poisson_fft_1d",
    # Spectral Poisson solvers — 3D
    "solve_poisson_dct1_3d",
    "solve_poisson_dct2_3d",
    "solve_poisson_dst1_3d",
    "solve_poisson_dst2_3d",
    "solve_poisson_fft_3d",
    # Capacitance matrix method
    "CapacitanceSolver",
    "build_capacitance_solver",
    # Preconditioners
    "make_multigrid_preconditioner",
    "make_nystrom_preconditioner",
    "make_preconditioner",
    "make_spectral_preconditioner",
    # Preconditioned Conjugate Gradient
    "CGInfo",
    "masked_laplacian",
    "solve_cg",
    # Convenience solver wrappers
    "streamfunction_from_vorticity",
    "pressure_from_divergence",
    "pv_inversion",
    # Multigrid Helmholtz solver
    "MultigridSolver",
    "build_multigrid_solver",
    # Tridiagonal (TDMA) solver
    "solve_tridiagonal",
    "solve_tridiagonal_batched",
    # Time integration — pure functional
    "euler_step",
    "heun_step",
    "rk3_ssp_step",
    "rk4_step",
    "ab2_step",
    "ab3_step",
    "leapfrog_raf_step",
    "imex_ssp2_step",
    "split_explicit_step",
    "semi_lagrangian_step",
    # Time integration — diffrax solvers
    "ForwardEulerDfx",
    "RK2Heun",
    "RK3SSP",
    "RK4Classic",
    "SSP_RK2",
    "SSP_RK104",
    "IMEX_SSP2",
    "AB2Solver",
    "LeapfrogRAFSolver",
    "SplitExplicitRKSolver",
    "SemiLagrangianSolver",
    "solve_ocean_pde",
]
