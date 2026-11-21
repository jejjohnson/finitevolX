import gridax as _gridax
from gridax import (
    Advection1D,
    Advection2D,
    Advection3D,
    ArakawaCGrid1D,
    ArakawaCGrid2D,
    ArakawaCGrid3D,
    ArakawaJacobian2D,
    BiharmonicDiffusion2D,
    BiharmonicDiffusion3D,
    BoundaryConditionSet,
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
    Coriolis2D,
    Coriolis3D,
    CurvilinearGrid1D,
    CurvilinearGrid2D,
    CurvilinearGrid3D,
    Difference1D,
    Difference2D,
    Difference3D,
    Diffusion2D,
    Diffusion3D,
    Dirichlet1D,
    Divergence2D,
    Energetics2D,
    Extrapolation1D,
    FieldBCSet,
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
    Mask1D,
    Mask2D,
    Mask3D,
    MomentumAdvection2D,
    MomentumAdvection3D,
    Neumann1D,
    Outflow1D,
    Periodic1D,
    QGPotentialVorticity2D,
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
    Reflective1D,
    Robin1D,
    Slip1D,
    SphericalAdvection2D,
    SphericalAdvection3D,
    SphericalBiharmonicDiffusion2D,
    SphericalBiharmonicDiffusion3D,
    SphericalDifference2D,
    SphericalDifference3D,
    SphericalDiffusion2D,
    SphericalDiffusion3D,
    SphericalDivergence2D,
    SphericalDivergence3D,
    SphericalGrid2D,
    SphericalGrid3D,
    SphericalLaplacian2D,
    SphericalLaplacian3D,
    SphericalMomentumAdvection2D,
    SphericalMomentumAdvection3D,
    SphericalVorticity2D,
    SphericalVorticity3D,
    Sponge1D,
    StencilCapability1D,
    StencilCapability2D,
    StencilCapability3D,
    Strain2D,
    Vorticity2D,
    Vorticity3D,
    arakawa_jacobian,
    area_mean,
    area_sum,
    area_weights,
    available_potential_energy,
    avg_x_bwd_1d,
    avg_x_bwd_2d,
    avg_x_bwd_3d,
    avg_x_fwd_1d,
    avg_x_fwd_2d,
    avg_x_fwd_3d,
    avg_xbwd_ybwd_2d,
    avg_xbwd_yfwd_2d,
    avg_xfwd_ybwd_2d,
    avg_xfwd_yfwd_2d,
    avg_y_bwd_2d,
    avg_y_bwd_3d,
    avg_y_fwd_2d,
    avg_y_fwd_3d,
    bernoulli_potential,
    cartesian_area_mean,
    cartesian_area_sum,
    cartesian_area_weights,
    cartesian_volume_mean,
    cartesian_volume_sum,
    cartesian_volume_weights,
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
    diffusion_2d,
    enforce_periodic,
    enstrophy,
    fix_boundary_corners,
    free_slip_boundaries,
    interior,
    kinetic_energy,
    linear_2pts,
    linear_3pts_left,
    linear_3pts_right,
    linear_4pts,
    linear_5pts_left,
    linear_5pts_right,
    linear_6pts,
    masked_mean,
    masked_moments,
    masked_std,
    mc,
    minmod,
    no_flux_boundaries,
    no_slip_boundaries,
    okubo_weiss,
    pad_interior,
    plusminus,
    potential_enstrophy,
    potential_vorticity,
    potential_vorticity_multilayer,
    pv_flux_arakawa_lamb,
    pv_flux_blend,
    qg_potential_vorticity,
    reconstruct,
    rusanov_flux,
    shear_strain,
    smooth_abs,
    smooth_clamp,
    smooth_max,
    spherical_area_mean,
    spherical_area_sum,
    spherical_area_weights,
    spherical_volume_mean,
    spherical_volume_sum,
    spherical_volume_weights,
    ssh_to_streamfn,
    strain_magnitude_squared,
    streamfn_to_ssh,
    stretching_term,
    superbee,
    sw_potential_vorticity,
    sw_potential_vorticity_multilayer,
    t_point_coordinates,
    tensor_strain,
    total_energy,
    total_enstrophy,
    upwind_1pt,
    upwind_3pt,
    upwind_5pt,
    upwind_flux,
    uv_center_flux,
    uv_node_flux,
    van_leer,
    vertical_velocity,
    volume_mean,
    volume_sum,
    volume_weights,
    vorticity_flux_upwind,
    wall_boundaries,
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
    zero_boundaries,
    zero_gradient_boundaries,
)
from gridax._src.operators.stencils import RENAMED as _RENAMED_STENCILS

from gridax_jax._src.solvers.elliptic import (
    # Types
    BoundaryCondition,
    # Capacitance
    CapacitanceSolver,
    CGInfo,
    # Solver classes
    DirichletHelmholtzSolver2D,
    MixedBCHelmholtzSolver2D,
    MixedBCHelmholtzSolver3D,
    # Preconditioners
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
    make_gaussx_multigrid_preconditioner,
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
from gridax_jax._src.solvers.inhomogeneous import (
    KnownValueLifting,
    SolveDomain,
    boundary_ring,
    coast_ring,
)
from gridax_jax._src.solvers.linear_operators import (
    helmholtz_operator,
    masked_helmholtz_operator,
)
from gridax_jax._src.solvers.multigrid import (
    MultigridSolver,
    build_multigrid_solver,
)
from gridax_jax._src.solvers.spectral_transforms import (
    dct,
    dctn,
    dst,
    dstn,
    idct,
    idctn,
    idst,
    idstn,
)
from gridax_jax._src.solvers.tridiagonal import (
    solve_tridiagonal,
    solve_tridiagonal_batched,
)
from gridax_jax._src.timestepping._deprecated import (
    ADAPTED as _ADAPTED_STEPPERS,
    MOVED as _MOVED_STEPPERS,
    SEMI_LAGRANGIAN as _SEMI_LAGRANGIAN,
    deprecated_stepper as _deprecated_stepper,
)
from gridax_jax._src.timestepping._solve import solve_ocean_pde
from gridax_jax._src.utils.dtype import check_dtype

__all__ = [
    # Upwind flux dispatch
    "upwind_flux",
    # Raw face fluxes
    "uv_center_flux",
    "uv_node_flux",
    "rusanov_flux",
    # Advection
    "Advection1D",
    "Advection2D",
    "Advection3D",
    "SphericalAdvection2D",
    "SphericalAdvection3D",
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
    "SphericalMomentumAdvection2D",
    "SphericalMomentumAdvection3D",
    # Jacobian
    "ArakawaJacobian2D",
    "arakawa_jacobian",
    # Area- and volume-weighted reductions (grid-polymorphic + explicit)
    "area_weights",
    "area_sum",
    "area_mean",
    "masked_mean",
    "masked_std",
    "masked_moments",
    "volume_weights",
    "volume_sum",
    "volume_mean",
    "cartesian_area_weights",
    "cartesian_area_sum",
    "cartesian_area_mean",
    "cartesian_volume_weights",
    "cartesian_volume_sum",
    "cartesian_volume_mean",
    "spherical_area_weights",
    "spherical_area_sum",
    "spherical_area_mean",
    "spherical_volume_weights",
    "spherical_volume_sum",
    "spherical_volume_mean",
    # Multilayer vmap helper
    # Grid — abstract topology
    "ArakawaCGrid1D",
    "ArakawaCGrid2D",
    "ArakawaCGrid3D",
    # Grid — curvilinear (uniform metric)
    "CurvilinearGrid1D",
    "CurvilinearGrid2D",
    "CurvilinearGrid3D",
    # Grid — Cartesian concrete
    "CartesianGrid1D",
    "CartesianGrid2D",
    "CartesianGrid3D",
    # Grid — Spherical concrete
    "SphericalGrid2D",
    "SphericalGrid3D",
    # Grid — point coordinates
    "t_point_coordinates",
    # Mask
    "Mask1D",
    "Mask2D",
    "Mask3D",
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
    "diff_x_fwd_2d",
    "diff_x_bwd_2d",
    "diff_y_fwd_2d",
    "diff_y_bwd_2d",
    "diff_x_fwd_1d",
    "diff_x_bwd_1d",
    "diff_x_fwd_3d",
    "diff_x_bwd_3d",
    "diff_y_fwd_3d",
    "diff_y_bwd_3d",
    # Averaging stencils
    "avg_x_fwd_2d",
    "avg_x_bwd_2d",
    "avg_y_fwd_2d",
    "avg_y_bwd_2d",
    "avg_xfwd_yfwd_2d",
    "avg_xbwd_ybwd_2d",
    "avg_xbwd_yfwd_2d",
    "avg_xfwd_ybwd_2d",
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
    "SphericalBiharmonicDiffusion2D",
    "SphericalBiharmonicDiffusion3D",
    "SphericalDiffusion2D",
    "SphericalDiffusion3D",
    "diffusion_2d",
    # Divergence
    "Divergence2D",
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
    "StencilCapability1D",
    "StencilCapability2D",
    "StencilCapability3D",
    # Equation of state
    # Diagnostics
    "Energetics2D",
    "QGPotentialVorticity2D",
    "Strain2D",
    "available_potential_energy",
    "bernoulli_potential",
    "enstrophy",
    "kinetic_energy",
    "okubo_weiss",
    "potential_enstrophy",
    "potential_vorticity",
    "potential_vorticity_multilayer",
    "qg_potential_vorticity",
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
    "pv_flux_arakawa_lamb",
    "pv_flux_blend",
    "vorticity_flux_upwind",
    # Barotropic-mode filter
    # Linear drag / Rayleigh relaxation
    # Forcing — Layer 0 functional primitives
    # Forcing — base class and composition
    # Forcing — 2D operators
    # Forcing — 3D operators
    # Differentiable surrogates
    "smooth_abs",
    "smooth_clamp",
    "smooth_max",
    # The dtype rule: the caller decides, a silent downcast is an error
    "check_dtype",
    # Vertical coupling and mode transforms
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
    "make_gaussx_multigrid_preconditioner",
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
    # Known-value lifting (inhomogeneous BCs)
    "coast_ring",
    "boundary_ring",
    "SolveDomain",
    "KnownValueLifting",
    # The Helmholtz operator as a gaussx operator
    "helmholtz_operator",
    "masked_helmholtz_operator",
    # Multigrid Helmholtz solver
    "MultigridSolver",
    "build_multigrid_solver",
    # Tridiagonal (TDMA) solver
    "solve_tridiagonal",
    "solve_tridiagonal_batched",
    # Time integration — pure functional (deprecated aliases of
    # integrix.march, served by __getattr__ below)
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
    # Time integration — diffrax solvers (deprecated aliases of integrix.march,
    # served by __getattr__ below)
    "ForwardEulerDfx",
    "RK2Heun",
    "RK3SSP",
    "RK4Classic",
    "SSP_RK2",
    "SSP_RK104",
    "IMEX_SSP2",
    "AB2Solver",
    "AB3Solver",
    "LeapfrogRAFSolver",
    "SplitExplicitRKSolver",
    "SemiLagrangianSolver",
    "SemiLagrangianTerm",
    "solve_ocean_pde",
]


# Moved to sesmax-core (#422, jejjohnson/sesmax#121): forcing, vertical modes,
# relaxation and the barotropic filter are physics next to the model state,
# not grid operations. The old names stay for one release and warn on access;
# gridax-jax imports nothing from sesmax (edges point down).
_MOVED_TO_SESMAX = {
    "AbstractForcing": ("gridax_jax._src.forcing._base", "sesmax_core.forcing"),
    "ForcingProduct": ("gridax_jax._src.forcing._base", "sesmax_core.forcing"),
    "ForcingSum": ("gridax_jax._src.forcing._base", "sesmax_core.forcing"),
    "LinearDrag2D": ("gridax_jax._src.forcing.drag", "sesmax_core.forcing"),
    "LinearDrag3D": ("gridax_jax._src.forcing.drag", "sesmax_core.forcing"),
    "QuadraticDrag2D": ("gridax_jax._src.forcing.drag", "sesmax_core.forcing"),
    "QuadraticDrag3D": ("gridax_jax._src.forcing.drag", "sesmax_core.forcing"),
    "RayleighDamping2D": ("gridax_jax._src.forcing.rayleigh", "sesmax_core.forcing"),
    "RayleighDamping3D": ("gridax_jax._src.forcing.rayleigh", "sesmax_core.forcing"),
    "TimeVaryingForcing": ("gridax_jax._src.forcing._base", "sesmax_core.forcing"),
    "WindCurl2D": ("gridax_jax._src.forcing.wind", "sesmax_core.forcing"),
    "WindStress2D": ("gridax_jax._src.forcing.wind", "sesmax_core.forcing"),
    "WindStress3D": ("gridax_jax._src.forcing.wind", "sesmax_core.forcing"),
    "barotropic_filter": (
        "gridax_jax._src.operators.barotropic",
        "sesmax_core.vertical",
    ),
    "build_coupling_matrix": (
        "gridax_jax._src.vertical.vertical_modes",
        "sesmax_core.vertical",
    ),
    "decompose_vertical_modes": (
        "gridax_jax._src.vertical.vertical_modes",
        "sesmax_core.vertical",
    ),
    "layer_to_mode": (
        "gridax_jax._src.vertical.vertical_modes",
        "sesmax_core.vertical",
    ),
    "linear_drag": ("gridax_jax._src.operators.relaxation", "sesmax_core.forcing"),
    "linear_drag_tendency": (
        "gridax_jax._src.forcing.functional",
        "sesmax_core.forcing",
    ),
    "mode_to_layer": (
        "gridax_jax._src.vertical.vertical_modes",
        "sesmax_core.vertical",
    ),
    "multilayer": ("gridax_jax._src.vertical.multilayer", "sesmax_core.vertical"),
    "quadratic_drag_tendency": (
        "gridax_jax._src.forcing.functional",
        "sesmax_core.forcing",
    ),
    "rayleigh_relaxation": (
        "gridax_jax._src.operators.relaxation",
        "sesmax_core.forcing",
    ),
    "rayleigh_tendency": ("gridax_jax._src.forcing.functional", "sesmax_core.forcing"),
    "wind_stress_tendency": (
        "gridax_jax._src.forcing.functional",
        "sesmax_core.forcing",
    ),
}


# Moved to geophysix (#446): the linear equation of state (geophysix.seawater,
# whose parameters are one LinearEOS record) and gridax's mean-latitude
# Coriolis helpers (geophysix.rotation is pointwise). The old names stay for
# one release and warn on access.
_MOVED_TO_GEOPHYSIX = {
    "linear_density": "geophysix.seawater.linear_density",
    "linear_density_anomaly": "geophysix.seawater.linear_density_anomaly",
    "linear_drho_dT": "geophysix.seawater.linear_drho_dT",
    "linear_drho_dS": "geophysix.seawater.linear_drho_dS",
    "buoyancy": "geophysix.seawater.buoyancy",
    "reduced_gravity": "geophysix.seawater.reduced_gravity",
    "coriolis_param": "geophysix.rotation.coriolis (pointwise; pass the mean latitude)",
    "beta_param": "geophysix.rotation.beta (pointwise; pass the mean latitude)",
    "coriolis_fn": "geophysix.rotation.beta_plane (y0 is required there)",
}


def __getattr__(name: str) -> object:
    if name in _MOVED_TO_SESMAX:
        import importlib
        import warnings

        module, home = _MOVED_TO_SESMAX[name]
        warnings.warn(
            f"gridax_jax.{name} is deprecated and will be removed in the next"
            f" release: it moved to {home}.{name} (pip install sesmax-core).",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(module), name)
    if name in _MOVED_TO_GEOPHYSIX:
        import warnings

        from gridax._src.operators import diagnostics

        from gridax_jax._src.operators import eos

        warnings.warn(
            f"gridax_jax.{name} is deprecated and will be removed in the next"
            f" release: use {_MOVED_TO_GEOPHYSIX[name]}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(eos if hasattr(eos, name) else diagnostics, name)
    # The C-grid calculus free functions repeat a Difference2D method (#456).
    if name in _gridax._DEPRECATED_CALCULUS:
        import warnings

        warnings.warn(
            f"gridax_jax.{name} is deprecated and will be removed in the next"
            f" release: use {_gridax._DEPRECATED_CALCULUS[name]}.",
            DeprecationWarning,
            stacklevel=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return getattr(_gridax, name)
    # The time steppers moved to integrix.march, the semi-Lagrangian scheme
    # included (#476); their old names stay for one release and warn on access.
    if name in _MOVED_STEPPERS or name in _ADAPTED_STEPPERS or name in _SEMI_LAGRANGIAN:
        return _deprecated_stepper(name)
    # The raw stencils were renamed in gridax (#435): every name carries its
    # rank and the corner averages use one scheme. The old names stay for
    # one release and warn on access.
    if name in _RENAMED_STENCILS:
        import warnings

        new_name = _RENAMED_STENCILS[name]
        warnings.warn(
            f"gridax_jax.{name} is deprecated and will be removed in the next"
            f" release: it is gridax.{new_name}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_gridax, new_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
