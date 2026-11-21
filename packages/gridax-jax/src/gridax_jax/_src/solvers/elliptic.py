"""Capacitance matrix solver and convenience wrappers for 2-D elliptic PDEs.

This module re-exports the spectral, iterative, and preconditioner APIs from
their dedicated sub-modules and adds:

Capacitance matrix method (irregular/masked domains)
-----------------------------------------------------
Extends the fast spectral solver to domains that are subsets of a
rectangle (e.g. ocean basins with land masks) using the classic
Sherman-Morrison correction via boundary Green's functions.

``build_capacitance_solver`` performs a one-time offline precomputation
(N_c rectangular solves, where N_c = number of coast cells, the dry cells
next to the wet mask).  The returned ``CapacitanceSolver`` callable is then
cheap to evaluate for any right-hand side and solves the same problem as
``method="cg"``: every wet cell an unknown, ψ = 0 on the coast.

The streamfunction, pressure and QG PV inversions built on these pieces
are ``sesmax_core.gfd``'s (#475, jejjohnson/geostack#291 rule 3).
"""

from __future__ import annotations

from gridax import Mask2D
import jax.numpy as jnp
import numpy as np
from spectraldiffx import CapacitanceSolver

# Re-export from iterative module
from gridax_jax._src.solvers.iterative import masked_laplacian  # noqa: F401
from gridax_jax._src.solvers.linear_operators import masked_helmholtz_operator

# Re-export from preconditioners module
from gridax_jax._src.solvers.preconditioners import (  # noqa: F401
    make_spectral_preconditioner,
)

# Re-export from spectral module
from gridax_jax._src.solvers.spectral import (  # noqa: F401
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
    # RHS modification
    modify_rhs_1d,
    modify_rhs_2d,
    modify_rhs_3d,
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
)

# ---------------------------------------------------------------------------
# Capacitance matrix solver — thin wrapper for Mask2D support
# ---------------------------------------------------------------------------


def build_capacitance_solver(
    mask: np.ndarray | Mask2D,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    base_bc: str = "dst",
) -> CapacitanceSolver:
    """Pre-compute the capacitance matrix and return a ready-to-use solver.

    The solver's unknowns are **every wet cell** of *mask*, with ψ = 0 on the
    coast ring (the dry cells 4-adjacent to a wet cell) and outside: the same
    problem ``method="cg"`` solves through :func:`masked_laplacian` on that
    mask (for ψ with a :class:`Mask2D`, the wall on the C-grid wall nodes).

    The masked operator is the wet sub-block ``B[W][:, W]`` of the
    rectangular five-point operator ``B = (∇² − λ)``
    (:func:`~finitevolx.masked_helmholtz_operator` with ``bc=base_bc``), a
    ``gaussx.MaskedOperator`` whose coupling set is the coast ring.  This is
    an **offline** function: it factorises the ``N_c × N_c`` capacitance
    matrix (``N_c`` = number of coast cells) with ``N_c`` rectangular solves.
    The returned :class:`CapacitanceSolver` is then two rectangular solves
    plus an ``O(N_c²)`` back-substitution per right-hand side and is
    JIT-compilable.

    Parameters
    ----------
    mask : np.ndarray of bool shape (Ny, Nx), or Mask2D
        Physical domain mask.  ``True`` = wet (ocean/fluid), ``False`` = dry
        (land/walls).  A :class:`Mask2D` gives its ψ (X-point) mask
        ``xy_corner_strict``.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ ≥ 0.  Use ``0.0`` for pure Poisson.
    base_bc : {"dst", "fft", "dct"}
        Rectangular base operator: Dirichlet (DST-I, the default:
        non-singular for every λ ≥ 0), periodic (FFT) or Neumann (DCT-II).
        The base only enters through cells beyond the array edge, so every
        base gives the ``masked_laplacian`` solution as long as no wet cell
        touches the array edge (true for any mask with a dry ghost ring);
        ``"fft"`` and ``"dct"`` raise when one does.  A singular base
        (``"fft"`` / ``"dct"`` at λ = 0) is handled exactly: gaussx borders
        the capacitance system with the base's null vector.

    Returns
    -------
    CapacitanceSolver
        A callable equinox Module with all precomputed arrays baked in; it
        returns ψ on the full grid, zero on every dry cell.

    Raises
    ------
    ValueError
        For an unknown ``base_bc``, a mask with no wet cells, a mask with
        no dry cells (nothing to correct: use the rectangular solver), or a
        ``"fft"`` / ``"dct"`` base with a wet cell on the array edge.

    Examples
    --------
    >>> mask = np.zeros((34, 34), dtype=bool)
    >>> mask[1:-1, 1:-1] = True
    >>> solver = build_capacitance_solver(Mask2D.from_mask(mask), dx, dy, lambda_=4.0)
    >>> psi = streamfunction_from_vorticity(
    ...     zeta, dx, dy, lambda_=4.0, method="capacitance", capacitance_solver=solver
    ... )
    """
    if base_bc not in ("fft", "dst", "dct"):
        raise ValueError(f"base_bc must be 'fft', 'dst', or 'dct'; got {base_bc!r}")
    if isinstance(mask, Mask2D):
        mask = np.asarray(mask.xy_corner_strict)
    wet = np.asarray(mask) > 0.5  # [Ny, Nx], concrete
    if not wet.any():
        raise ValueError("The mask has no wet (True) cells.")
    if wet.all():
        raise ValueError(
            "The mask has no dry (False) cells, so there is no coast to "
            f"correct for; use solve_helmholtz_{base_bc} directly."
        )
    edge = wet[0].any() | wet[-1].any() | wet[:, 0].any() | wet[:, -1].any()
    if base_bc != "dst" and edge:
        raise ValueError(
            f"base_bc={base_bc!r} reaches beyond the array edge, where "
            "masked_laplacian (method='cg') has psi = 0, but the mask has wet "
            "cells on the edge: use base_bc='dst' or a mask with a dry ring."
        )
    # The wet block of B = (∇² - λ) on the rectangle (the same eigenvalues as
    # the spectral solvers), coupled to the dry cells through the coast ring.
    operator = masked_helmholtz_operator(
        jnp.asarray(wet), dx, dy, lambda_=lambda_, bc=base_bc
    )
    return CapacitanceSolver(
        operator=operator,
        interior_indices=jnp.asarray(np.flatnonzero(wet.ravel())),
        shape=wet.shape,
        dx=float(dx),
        dy=float(dy),
        lambda_=float(lambda_),
        base_bc=base_bc,
    )
