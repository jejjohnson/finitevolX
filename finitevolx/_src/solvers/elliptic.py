"""Capacitance matrix solver and convenience wrappers for 2-D elliptic PDEs.

This module re-exports the spectral, iterative, and preconditioner APIs from
their dedicated sub-modules and adds:

Capacitance matrix method (irregular/masked domains)
-----------------------------------------------------
Extends the fast spectral solver to domains that are subsets of a
rectangle (e.g. ocean basins with land masks) using the classic
Sherman-Morrison correction via boundary Green's functions.

``build_capacitance_solver`` performs a one-time offline precomputation
(N_b rectangular solves, where N_b = number of irregular-boundary points).
The returned ``CapacitanceSolver`` callable is then cheap to evaluate for
any right-hand side.

Convenience wrappers
--------------------
* :func:`streamfunction_from_vorticity` — ∇²ψ − λψ = ζ
* :func:`pressure_from_divergence` — ∇²p = ∇·u
* :func:`pv_inversion` — (∇² − λ)ψ = q  (multi-layer / batched)

All three wrappers accept ``known_values`` (and optionally ``known_mask``)
for inhomogeneous Dirichlet data, solved with the lifting trick from
:mod:`finitevolx._src.solvers.inhomogeneous`.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
from spectraldiffx import (
    CapacitanceSolver,
    build_capacitance_solver as _build_capacitance_solver_base,
)

from finitevolx._src.mask import Mask2D
from finitevolx._src.solvers.inhomogeneous import (
    SolveDomain,
    lift_rhs,
    reconstruct_from_lift,
)

# Re-export from iterative module
from finitevolx._src.solvers.iterative import (  # noqa: F401
    CGInfo,
    masked_laplacian,
    solve_cg,
)

# Re-export from preconditioners module
from finitevolx._src.solvers.preconditioners import (  # noqa: F401
    make_multigrid_preconditioner,
    make_nystrom_preconditioner,
    make_preconditioner,
    make_spectral_preconditioner,
)

# Re-export from spectral module
from finitevolx._src.solvers.spectral import (  # noqa: F401
    _HELMHOLTZ_DISPATCH,
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
    _spectral_solve,
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
    base_bc: str = "fft",
) -> CapacitanceSolver:
    """Pre-compute the capacitance matrix and return a ready-to-use solver.

    This is an **offline** function that performs *N_b* rectangular spectral
    solves (``N_b`` = number of inner-boundary points).  The result is a
    :class:`CapacitanceSolver` whose ``__call__`` method is JIT-compilable.

    Parameters
    ----------
    mask : np.ndarray of bool shape (Ny, Nx), or Mask2D
        Physical domain mask.  ``True`` = interior (ocean/fluid),
        ``False`` = exterior (land/walls).

        When an :class:`Mask2D` is passed, the ``psi``
        staggering mask is extracted automatically.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Use ``0.0`` for pure Poisson.
    base_bc : {"fft", "dst", "dct"}
        Rectangular spectral solver used as the base.

    Returns
    -------
    CapacitanceSolver
        A callable equinox Module with all precomputed arrays baked in.
    """
    if isinstance(mask, Mask2D):
        mask = np.asarray(mask.xy_corner_strict, dtype=bool)
    return _build_capacitance_solver_base(mask, dx, dy, lambda_, base_bc)


# ---------------------------------------------------------------------------
# Convenience wrappers: streamfunction, pressure, PV inversion
# ---------------------------------------------------------------------------

# Type alias for the mask parameter accepted by the convenience wrappers.
_MaskLike = Float[Array, "Ny Nx"] | Mask2D
_PrecondLike = Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]]


def _resolve_mask_arr(
    mask: _MaskLike | None,
) -> Float[Array, "Ny Nx"] | None:
    """Extract a float mask array from *mask*, or return None."""
    if mask is None:
        return None
    if isinstance(mask, Mask2D):
        return jnp.asarray(mask.xy_corner_strict, dtype=jnp.float32)
    return mask


def _solve_spectral(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
) -> Float[Array, "Ny Nx"]:
    """Solve using a rectangular spectral solver (DST/DCT/FFT)."""
    return _spectral_solve(rhs, dx, dy, lambda_, bc)


def _solve_cg_method(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    mask: _MaskLike | None,
    preconditioner: _PrecondLike | None,
) -> Float[Array, "Ny Nx"]:
    """Solve using preconditioned Conjugate Gradient on a masked domain."""
    mask_arr = _resolve_mask_arr(mask)
    if mask_arr is None:
        raise ValueError("method='cg' requires a mask (array or Mask2D)")

    def _matvec(x: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return masked_laplacian(x, mask_arr, dx, dy, lambda_=lambda_)

    if preconditioner is None:
        preconditioner = make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc="fft")

    x, _info = solve_cg(_matvec, rhs * mask_arr, preconditioner=preconditioner)
    return x * mask_arr


def _solve_capacitance_method(
    rhs: Float[Array, "Ny Nx"],
    capacitance_solver: CapacitanceSolver | None,
) -> Float[Array, "Ny Nx"]:
    """Solve using a pre-built capacitance matrix solver."""
    if capacitance_solver is None:
        raise ValueError(
            "method='capacitance' requires a pre-built CapacitanceSolver "
            "(see build_capacitance_solver)"
        )
    return capacitance_solver(rhs)


def _solve_dispatch(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
    method: str,
    mask: _MaskLike | None,
    capacitance_solver: CapacitanceSolver | None,
    preconditioner: _PrecondLike | None,
) -> Float[Array, "Ny Nx"]:
    """Dispatch an elliptic solve to the selected solver method."""
    if method == "spectral":
        return _solve_spectral(rhs, dx, dy, lambda_, bc)
    if method == "cg":
        return _solve_cg_method(rhs, dx, dy, lambda_, mask, preconditioner)
    if method == "capacitance":
        return _solve_capacitance_method(rhs, capacitance_solver)
    raise ValueError(
        f"method must be 'spectral', 'cg', or 'capacitance'; got {method!r}"
    )


_METHODS = ("spectral", "cg", "capacitance")


def _check_method(method: str) -> None:
    """Raise the standard error for an unknown solver method."""
    if method not in _METHODS:
        raise ValueError(
            f"method must be 'spectral', 'cg', or 'capacitance'; got {method!r}"
        )


def _known_value_domain(
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    method: str,
    bc: str,
    mask: _MaskLike | None,
    known_mask: Array | None,
) -> SolveDomain:
    """Build the :class:`SolveDomain` for a solve with known values.

    Mask-based methods use the caller's *mask* (required).  The spectral
    path has no mask: it solves the standard rectangular basin -- dry ghost
    ring, wet interior -- whose inner ring (rows/columns 1 and -2) carries
    the known values.  This is the same problem ``method="cg"`` solves with
    that basin mask, so both methods agree to solver tolerance.
    """
    _check_method(method)
    if method == "spectral":
        if bc != "dst":
            raise ValueError(
                "known_values prescribe Dirichlet data, so method='spectral' "
                f"requires bc='dst'; got bc={bc!r}.  Use method='cg' or "
                "'capacitance' with a mask for other boundary types."
            )
        if mask is not None:
            raise ValueError(
                "method='spectral' with known_values solves the rectangular "
                "basin only and does not accept a mask; use method='cg' or "
                "'capacitance' for a custom mask."
            )
        if known_mask is not None:
            raise ValueError(
                "method='spectral' does not support interior known values "
                "(known_mask); use method='cg'."
            )
        ny, nx = shape[-2], shape[-1]
        # basin[j, i] = 1 for 1 <= j <= Ny-2, 1 <= i <= Nx-2  (dry ghost ring)
        basin = jnp.zeros((ny, nx), dtype=dtype).at[1:-1, 1:-1].set(1.0)
        return SolveDomain(basin)

    mask_arr = _resolve_mask_arr(mask)
    if mask_arr is None:
        raise ValueError(f"known_values with method={method!r} requires a mask")
    if method == "capacitance" and known_mask is not None:
        raise ValueError(
            "method='capacitance' does not support known_mask: the capacitance "
            "solver can only hold its own inner ring at zero.  Use method='cg'."
        )
    return SolveDomain(mask_arr, known_mask)


def _solve_spectral_basin(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
) -> Float[Array, "Ny Nx"]:
    """DST solve on the basin's solve domain ``[2:-2, 2:-2]``, zero elsewhere.

    The rectangular-basin solve domain (wet interior minus its inner ring)
    is itself a rectangle, so homogeneous Dirichlet there is a plain DST.
    """
    helmholtz = _HELMHOLTZ_DISPATCH["dst"]
    # Solve cells: 2 <= j <= Ny-3, 2 <= i <= Nx-3 (inside the known ring 1 / N-2).
    # (A - lambda) psi[j, i] = rhs[j, i] there, psi = 0 on the ring (DST-I).
    psi = helmholtz(rhs[2:-2, 2:-2], dx, dy, lambda_)
    # psi_hom[j, i] = psi[j-2, i-2] on the solve cells, 0 elsewhere
    return jnp.zeros_like(rhs).at[2:-2, 2:-2].set(psi)


def _solve_with_known_values(
    rhs: Float[Array, "Ny Nx"],
    known_values: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
    method: str,
    mask: _MaskLike | None,
    known_mask: Array | None,
    capacitance_solver: CapacitanceSolver | None,
    preconditioner: _PrecondLike | None,
) -> Float[Array, "Ny Nx"]:
    """Inhomogeneous solve: lift, solve homogeneously, reconstruct.

    psi = value_lift + psi_hom, where psi_hom solves
    (A - lambda) psi_hom = f - (A - lambda) value_lift on the solve domain.
    """
    domain = _known_value_domain(rhs.shape, rhs.dtype, method, bc, mask, known_mask)
    rhs_corrected, value_lift = lift_rhs(domain, rhs, known_values, dx, dy, lambda_)
    if method == "spectral":
        psi_hom = _solve_spectral_basin(rhs_corrected, dx, dy, lambda_)
    else:
        # CG solves on the effective domain directly.  A capacitance solver
        # built on the wet mask already holds that mask's inner ring at zero,
        # so its unknowns are exactly the effective domain.
        eff_mask = domain.effective_mask.astype(rhs.dtype)
        psi_hom = _solve_dispatch(
            rhs_corrected,
            dx,
            dy,
            lambda_,
            bc,
            method,
            eff_mask,
            capacitance_solver,
            preconditioner,
        )
    return reconstruct_from_lift(domain, psi_hom, value_lift)


def streamfunction_from_vorticity(
    zeta: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    bc: str = "dst",
    lambda_: float = 0.0,
    method: str = "spectral",
    mask: _MaskLike | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
    preconditioner: _PrecondLike | None = None,
    known_values: Float[Array, "Ny Nx"] | None = None,
    known_mask: Array | None = None,
) -> Float[Array, "Ny Nx"]:
    r"""Invert the vorticity–streamfunction relation ∇²ψ − λψ = ζ.

    Solves the Poisson (λ = 0) or Helmholtz (λ ≠ 0) equation to recover the
    streamfunction from relative vorticity.

    Three solver methods are available:

    * ``"spectral"`` — Direct spectral solver (DST/DCT/FFT) for rectangular
      domains.  Selected by *bc*.  Default.
    * ``"cg"`` — Preconditioned Conjugate Gradient for masked / irregular
      domains.  Requires *mask*.  Uses a spectral preconditioner by default,
      or a custom one via *preconditioner*.
    * ``"capacitance"`` — Capacitance matrix method for masked domains.
      Requires a pre-built :class:`CapacitanceSolver` via
      *capacitance_solver*.

    Parameters
    ----------
    zeta : Float[Array, "Ny Nx"]
        Relative vorticity (right-hand side).
    dx, dy : float
        Grid spacings.
    bc : {"dst", "dct", "fft"}
        Boundary-condition type for the spectral solver (used by
        ``method="spectral"``).
        ``"dst"`` (Dirichlet, ψ = 0 on boundary) is the most common choice
        for streamfunction inversion.
    lambda_ : float
        Helmholtz parameter.  Use 0.0 for the pure Poisson problem
        (streamfunction from vorticity).  Non-zero values arise in QG PV
        inversion: (∇² − λ)ψ = q.
    method : {"spectral", "cg", "capacitance"}
        Solver method.  Default: ``"spectral"``.
    mask : Float[Array, "Ny Nx"] or Mask2D or None
        Domain mask.  Required for ``method="cg"``.  When an
        :class:`Mask2D` is passed the ``psi`` staggering mask is
        extracted automatically.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.  Signature:
        ``preconditioner(r: Array) -> Array``.  When ``None``, a spectral
        preconditioner (FFT-based) is used automatically.
    known_values : Float[Array, "Ny Nx"] or None
        Prescribed (inhomogeneous Dirichlet) values of the solution, used
        at the inner boundary ring -- the wet cells adjacent to a dry cell --
        and at any ``known_mask`` cells; values elsewhere are ignored.  The
        solution equals ``known_values`` exactly at those cells.  ``None``
        (default) keeps the homogeneous solve unchanged.  Mask-based methods
        require *mask*.  For ``method="capacitance"`` build the solver on the
        same wet *mask*: it holds its own inner ring at zero, which is exactly
        the lifted ring (use ``base_bc="dst"`` when ``lambda_ == 0``).
        ``method="spectral"`` requires ``bc="dst"`` and no *mask*: it solves
        the rectangular basin whose dry ghost ring surrounds the wet
        interior, matching ``method="cg"`` with that basin mask.
    known_mask : Bool[Array, "Ny Nx"] or None
        Extra interior wet cells with known values (e.g. sparse
        observations), pinned alongside the boundary ring.
        ``method="cg"`` only.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Streamfunction ψ.
    """
    if known_values is not None:
        return _solve_with_known_values(
            zeta,
            known_values,
            dx,
            dy,
            lambda_,
            bc,
            method,
            mask,
            known_mask,
            capacitance_solver,
            preconditioner,
        )
    return _solve_dispatch(
        zeta, dx, dy, lambda_, bc, method, mask, capacitance_solver, preconditioner
    )


def pressure_from_divergence(
    div_u: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    bc: str = "dct",
    method: str = "spectral",
    mask: _MaskLike | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
    preconditioner: _PrecondLike | None = None,
    known_values: Float[Array, "Ny Nx"] | None = None,
    known_mask: Array | None = None,
) -> Float[Array, "Ny Nx"]:
    r"""Solve ∇²p = ∇·u for the pressure correction.

    Used in pressure-projection methods (Chorin splitting) where the
    divergence of the provisional velocity field must be removed.

    Solver selection follows the same three-method dispatch as
    :func:`streamfunction_from_vorticity`.

    Parameters
    ----------
    div_u : Float[Array, "Ny Nx"]
        Divergence of the velocity field (right-hand side).
    dx, dy : float
        Grid spacings.
    bc : {"dct", "dst", "fft"}
        Boundary-condition type for the spectral solver.
        ``"dct"`` (Neumann, ∂p/∂n = 0) is the standard choice for
        pressure with solid walls.
    method : {"spectral", "cg", "capacitance"}
        Solver method.  Default: ``"spectral"``.
    mask : Float[Array, "Ny Nx"] or Mask2D or None
        Domain mask.  Required for ``method="cg"``.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.
    known_values : Float[Array, "Ny Nx"] or None
        Prescribed (inhomogeneous Dirichlet) values of the solution, used
        at the inner boundary ring -- the wet cells adjacent to a dry cell --
        and at any ``known_mask`` cells; values elsewhere are ignored.  The
        solution equals ``known_values`` exactly at those cells.  ``None``
        (default) keeps the homogeneous solve unchanged.  Mask-based methods
        require *mask*.  For ``method="capacitance"`` build the solver on the
        same wet *mask*: it holds its own inner ring at zero, which is exactly
        the lifted ring (use ``base_bc="dst"`` when ``lambda_ == 0``).
        ``method="spectral"`` requires ``bc="dst"`` and no *mask*: it solves
        the rectangular basin whose dry ghost ring surrounds the wet
        interior, matching ``method="cg"`` with that basin mask.
    known_mask : Bool[Array, "Ny Nx"] or None
        Extra interior wet cells with known values (e.g. sparse
        observations), pinned alongside the boundary ring.
        ``method="cg"`` only.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Pressure field p.
    """
    if known_values is not None:
        return _solve_with_known_values(
            div_u,
            known_values,
            dx,
            dy,
            0.0,
            bc,
            method,
            mask,
            known_mask,
            capacitance_solver,
            preconditioner,
        )
    return _solve_dispatch(
        div_u, dx, dy, 0.0, bc, method, mask, capacitance_solver, preconditioner
    )


def pv_inversion(
    pv: Float[Array, "... Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Float[Array, " nl"],
    bc: str = "dst",
    method: str = "spectral",
    mask: _MaskLike | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
    preconditioner: _PrecondLike | None = None,
    known_values: Float[Array, "... Ny Nx"] | None = None,
    known_mask: Array | None = None,
) -> Float[Array, "... Ny Nx"]:
    r"""QG potential-vorticity inversion: solve (∇² − λ)ψ = q.

    Supports batched / multi-layer PV fields.  When *lambda_* is a 1-D
    array of shape ``(nl,)``, each layer is solved with its own Helmholtz
    parameter (e.g. 1/Rd² per vertical mode from
    :func:`~finitevolx.decompose_vertical_modes`).

    Solver selection follows the same three-method dispatch as
    :func:`streamfunction_from_vorticity`.

    Parameters
    ----------
    pv : Float[Array, "... Ny Nx"]
        Potential-vorticity field.  Leading dimensions are batched.
    dx, dy : float
        Grid spacings.
    lambda_ : float or Float[Array, " nl"]
        Helmholtz parameter(s).  Scalar for a single layer; array of
        shape ``(nl,)`` for multi-layer inversion.
    bc : {"dst", "dct", "fft"}
        Boundary-condition type (for ``method="spectral"``).
    method : {"spectral", "cg", "capacitance"}
        Solver method.  Default: ``"spectral"``.
    mask : Float[Array, "Ny Nx"] or Mask2D or None
        Domain mask.  Required for ``method="cg"``.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.
    known_values : Float[Array, "... Ny Nx"] or None
        Prescribed Dirichlet values (see
        :func:`streamfunction_from_vorticity`), broadcast against *pv*: a
        ``(Ny, Nx)`` field applies to every layer, ``(nl, Ny, Nx)`` gives
        per-layer values.  Each layer's lift uses its own *lambda_*.
    known_mask : Bool[Array, "Ny Nx"] or None
        Extra interior wet cells with known values, shared by all layers.
        ``method="cg"`` only.

    Returns
    -------
    Float[Array, "... Ny Nx"]
        Streamfunction ψ, same shape as *pv*.
    """
    lam = jnp.asarray(lambda_)
    kv = None if known_values is None else jnp.broadcast_to(known_values, pv.shape)

    if lam.ndim == 0 and kv is not None:

        def _solve_one_known(
            rhs: Float[Array, "Ny Nx"], kv_i: Float[Array, "Ny Nx"]
        ) -> Float[Array, "Ny Nx"]:
            return _solve_with_known_values(
                rhs,
                kv_i,
                dx,
                dy,
                float(lam),
                bc,
                method,
                mask,
                known_mask,
                capacitance_solver,
                preconditioner,
            )

        if pv.ndim == 2:
            return _solve_one_known(pv, kv)
        shape = pv.shape
        flat = pv.reshape(-1, shape[-2], shape[-1])
        kv_flat = kv.reshape(-1, shape[-2], shape[-1])
        return eqx.filter_vmap(_solve_one_known)(flat, kv_flat).reshape(shape)

    if lam.ndim == 0:
        # Scalar lambda: vmap over all leading dims if present
        if pv.ndim == 2:
            return _solve_dispatch(
                pv,
                dx,
                dy,
                float(lam),
                bc,
                method,
                mask,
                capacitance_solver,
                preconditioner,
            )
        # Flatten leading dims, solve each, reshape
        shape = pv.shape
        flat = pv.reshape(-1, shape[-2], shape[-1])

        def _solve_one(rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
            return _solve_dispatch(
                rhs,
                dx,
                dy,
                float(lam),
                bc,
                method,
                mask,
                capacitance_solver,
                preconditioner,
            )

        out = eqx.filter_vmap(_solve_one)(flat)
        return out.reshape(shape)

    # Array lambda: leading dim must match lam.shape[0]
    if pv.ndim < 3:
        raise ValueError(
            f"pv must have at least 3 dims when lambda_ is an array, "
            f"got shape {pv.shape}"
        )
    nl = lam.shape[0]
    if pv.shape[-3] != nl:
        raise ValueError(
            f"pv.shape[-3]={pv.shape[-3]} does not match lambda_ length {nl}"
        )

    # Solve each layer with its own lambda.
    # We call the Helmholtz solver directly (not _solve_dispatch) because
    # lam_i is a JAX tracer inside vmap and Python-level ``if lam == 0``
    # branches in _spectral_solve would fail.
    if method == "capacitance":
        raise ValueError(
            "method='capacitance' does not support array-valued lambda_; "
            "solve each layer separately or use method='spectral' or 'cg' "
            "for multi-layer problems."
        )

    # Known values: shared solve domain; each layer lifts with its own lambda.
    domain = (
        None
        if kv is None
        else _known_value_domain(pv.shape, pv.dtype, method, bc, mask, known_mask)
    )

    if method == "cg":
        mask_arr = _resolve_mask_arr(mask)
        if mask_arr is None:
            raise ValueError("method='cg' requires a mask")
        # With known values the solve runs on the effective domain.
        solve_mask = (
            mask_arr if domain is None else domain.effective_mask.astype(pv.dtype)
        )

        _precond = preconditioner

        def _solve_layer(
            rhs: Float[Array, "Ny Nx"], lam_i: float
        ) -> Float[Array, "Ny Nx"]:
            def _matvec(x: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
                return masked_laplacian(x, solve_mask, dx, dy, lambda_=lam_i)

            pc = (
                _precond
                if _precond is not None
                else make_spectral_preconditioner(dx, dy, lambda_=lam_i, bc="fft")
            )
            x, _info = solve_cg(_matvec, rhs * solve_mask, preconditioner=pc)
            return x * solve_mask

    elif method == "spectral" and domain is not None:

        def _solve_layer(
            rhs: Float[Array, "Ny Nx"], lam_i: float
        ) -> Float[Array, "Ny Nx"]:
            return _solve_spectral_basin(rhs, dx, dy, lam_i)

    elif method == "spectral":
        _helmholtz = _HELMHOLTZ_DISPATCH.get(bc)
        if _helmholtz is None:
            raise ValueError(f"bc must be 'fft', 'dst', or 'dct'; got {bc!r}")

        def _solve_layer(
            rhs: Float[Array, "Ny Nx"], lam_i: float
        ) -> Float[Array, "Ny Nx"]:
            return _helmholtz(rhs, dx, dy, lam_i)

    else:
        raise ValueError(
            f"method must be 'spectral', 'cg', or 'capacitance'; got {method!r}"
        )

    # Flatten any leading batch dims: (..., nl, Ny, Nx) -> (batch, nl, Ny, Nx)
    shape = pv.shape
    ny, nx = shape[-2], shape[-1]
    pv_4d = pv.reshape(-1, nl, ny, nx)

    if domain is None or kv is None:
        # vmap over layer axis (pairing each layer with its lambda)
        _solve_layers = eqx.filter_vmap(_solve_layer, in_axes=(0, 0))

        # vmap over the (flattened) batch axis
        out_4d = eqx.filter_vmap(lambda batch: _solve_layers(batch, lam))(pv_4d)
        return out_4d.reshape(shape)

    known_domain = domain

    def _solve_layer_known(
        rhs: Float[Array, "Ny Nx"], kv_i: Float[Array, "Ny Nx"], lam_i: float
    ) -> Float[Array, "Ny Nx"]:
        # psi = lift + psi_hom, psi_hom solving the lambda_i-corrected RHS
        rhs_c, value_lift = lift_rhs(known_domain, rhs, kv_i, dx, dy, lam_i)
        return reconstruct_from_lift(
            known_domain, _solve_layer(rhs_c, lam_i), value_lift
        )

    kv_4d = jnp.reshape(kv, (-1, nl, ny, nx))
    _solve_layers_known = eqx.filter_vmap(_solve_layer_known, in_axes=(0, 0, 0))
    out_4d = eqx.filter_vmap(lambda batch, k: _solve_layers_known(batch, k, lam))(
        pv_4d, kv_4d
    )
    return out_4d.reshape(shape)
