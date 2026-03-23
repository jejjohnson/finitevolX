"""Spectral elliptic solvers — re-exported from spectraldiffx.

Solves the equation (∇² − λ) ψ = f on rectangular domains using
DST (Dirichlet), DCT (Neumann), FFT (periodic), or mixed-BC spectral methods.

See spectraldiffx documentation for full solver details.
"""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import Array, Float
from spectraldiffx import (
    # Types
    BoundaryCondition,
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
    # RHS modification for inhomogeneous BCs
    modify_rhs_1d,
    modify_rhs_2d,
    modify_rhs_3d,
    # Generic per-axis BC solvers
    solve_helmholtz_2d,
    solve_helmholtz_3d,
    # Helmholtz solvers — 2D (legacy short names)
    solve_helmholtz_dct,
    # Helmholtz solvers — 2D (explicit transform names)
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
    # Poisson solvers — 2D (legacy short names)
    solve_poisson_dct,
    # Poisson solvers — 2D (explicit transform names)
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
)

# ---------------------------------------------------------------------------
# Internal dispatch helper
# ---------------------------------------------------------------------------

# Lookup for Helmholtz solvers by BC type — safe to call with JAX tracers.
_HELMHOLTZ_DISPATCH: dict[str, Callable] = {
    "fft": solve_helmholtz_fft,
    "dst": solve_helmholtz_dst,
    "dct": solve_helmholtz_dct,
}


def _spectral_solve(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
) -> Float[Array, "Ny Nx"]:
    """Dispatch to the appropriate rectangular spectral solver.

    Uses the Helmholtz solvers unconditionally (they handle lambda=0
    internally via null-mode guards) so that this function is safe to
    call when *lambda_* is a JAX tracer (e.g. inside ``vmap``).
    """
    solver = _HELMHOLTZ_DISPATCH.get(bc)
    if solver is None:
        raise ValueError(f"base_bc must be 'fft', 'dst', or 'dct'; got {bc!r}")
    return solver(rhs, dx, dy, lambda_)


__all__ = [
    # Types
    "BoundaryCondition",
    # Solver classes
    "DirichletHelmholtzSolver2D",
    "MixedBCHelmholtzSolver2D",
    "MixedBCHelmholtzSolver3D",
    "NeumannHelmholtzSolver2D",
    "RegularNeumannHelmholtzSolver2D",
    "SpectralHelmholtzSolver1D",
    "SpectralHelmholtzSolver2D",
    "SpectralHelmholtzSolver3D",
    "StaggeredDirichletHelmholtzSolver2D",
    # Eigenvalue functions (FD2)
    "dct1_eigenvalues",
    "dct2_eigenvalues",
    "dct3_eigenvalues",
    "dct4_eigenvalues",
    "dst1_eigenvalues",
    "dst2_eigenvalues",
    "dst3_eigenvalues",
    "dst4_eigenvalues",
    "fft_eigenvalues",
    # Eigenvalue functions (pseudo-spectral)
    "dct1_eigenvalues_ps",
    "dct2_eigenvalues_ps",
    "dct3_eigenvalues_ps",
    "dct4_eigenvalues_ps",
    "dst1_eigenvalues_ps",
    "dst2_eigenvalues_ps",
    "dst3_eigenvalues_ps",
    "dst4_eigenvalues_ps",
    "fft_eigenvalues_ps",
    # RHS modification
    "modify_rhs_1d",
    "modify_rhs_2d",
    "modify_rhs_3d",
    # Generic per-axis BC solvers
    "solve_helmholtz_2d",
    "solve_helmholtz_3d",
    "solve_poisson_2d",
    "solve_poisson_3d",
    # Helmholtz solvers — 2D
    "solve_helmholtz_dct",
    "solve_helmholtz_dct1",
    "solve_helmholtz_dct2",
    "solve_helmholtz_dst",
    "solve_helmholtz_dst1",
    "solve_helmholtz_dst2",
    "solve_helmholtz_fft",
    # Helmholtz solvers — 1D
    "solve_helmholtz_dct1_1d",
    "solve_helmholtz_dct2_1d",
    "solve_helmholtz_dst1_1d",
    "solve_helmholtz_dst2_1d",
    "solve_helmholtz_fft_1d",
    # Helmholtz solvers — 3D
    "solve_helmholtz_dct1_3d",
    "solve_helmholtz_dct2_3d",
    "solve_helmholtz_dst1_3d",
    "solve_helmholtz_dst2_3d",
    "solve_helmholtz_fft_3d",
    # Poisson solvers — 2D
    "solve_poisson_dct",
    "solve_poisson_dct1",
    "solve_poisson_dct2",
    "solve_poisson_dst",
    "solve_poisson_dst1",
    "solve_poisson_dst2",
    "solve_poisson_fft",
    # Poisson solvers — 1D
    "solve_poisson_dct1_1d",
    "solve_poisson_dct2_1d",
    "solve_poisson_dst1_1d",
    "solve_poisson_dst2_1d",
    "solve_poisson_fft_1d",
    # Poisson solvers — 3D
    "solve_poisson_dct1_3d",
    "solve_poisson_dct2_3d",
    "solve_poisson_dst1_3d",
    "solve_poisson_dst2_3d",
    "solve_poisson_fft_3d",
]
