from finitevolx._src.advection import Advection1D, Advection2D, Advection3D
from finitevolx._src.bc_1d import (
    Dirichlet1D,
    Neumann1D,
    Outflow1D,
    Periodic1D,
    Reflective1D,
    SlipBC1D,
    Sponge1D,
)
from finitevolx._src.bc_field import FieldBCSet
from finitevolx._src.bc_set import BoundaryConditionSet
from finitevolx._src.boundary import enforce_periodic, pad_interior
from finitevolx._src.difference import Difference1D, Difference2D, Difference3D
from finitevolx._src.elliptic import (
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
from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.interpolation import (
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
)
from finitevolx._src.masks.cgrid_mask import ArakawaCGridMask, StencilCapability
from finitevolx._src.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)
from finitevolx._src.reconstructions.limiters import mc, minmod, superbee, van_leer
from finitevolx._src.spectral_transforms import (
    dct,
    dctn,
    dst,
    dstn,
    idct,
    idctn,
    idst,
    idstn,
)
from finitevolx._src.vorticity import Vorticity2D, Vorticity3D

__all__ = [
    # Advection
    "Advection1D",
    "Advection2D",
    "Advection3D",
    # Grid
    "ArakawaCGrid1D",
    "ArakawaCGrid2D",
    "ArakawaCGrid3D",
    "ArakawaCGridMask",
    # Boundary condition sets and 1D BCs
    "BoundaryConditionSet",
    "Dirichlet1D",
    "FieldBCSet",
    "Neumann1D",
    "Outflow1D",
    "Periodic1D",
    "Reflective1D",
    "SlipBC1D",
    "Sponge1D",
    # Finite difference
    "Difference1D",
    "Difference2D",
    "Difference3D",
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
    # Vorticity
    "Vorticity2D",
    "Vorticity3D",
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
