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

from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask

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
    _spectral_solve,
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
    solve_helmholtz_dct,
    solve_helmholtz_dst,
    solve_helmholtz_fft,
    solve_poisson_dct,
    solve_poisson_dst,
    solve_poisson_fft,
)

# ---------------------------------------------------------------------------
# Capacitance matrix solver — thin wrapper for ArakawaCGridMask support
# ---------------------------------------------------------------------------


def build_capacitance_solver(
    mask: np.ndarray | ArakawaCGridMask,
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
    mask : np.ndarray of bool shape (Ny, Nx), or ArakawaCGridMask
        Physical domain mask.  ``True`` = interior (ocean/fluid),
        ``False`` = exterior (land/walls).

        When an :class:`ArakawaCGridMask` is passed, the ``psi``
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
    if isinstance(mask, ArakawaCGridMask):
        mask = np.asarray(mask.psi, dtype=bool)
    return _build_capacitance_solver_base(mask, dx, dy, lambda_, base_bc)


# ---------------------------------------------------------------------------
# Convenience wrappers: streamfunction, pressure, PV inversion
# ---------------------------------------------------------------------------

# Type alias for the mask parameter accepted by the convenience wrappers.
_MaskLike = Float[Array, "Ny Nx"] | ArakawaCGridMask
_PrecondLike = Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]]


def _resolve_mask_arr(
    mask: _MaskLike | None,
) -> Float[Array, "Ny Nx"] | None:
    """Extract a float mask array from *mask*, or return None."""
    if mask is None:
        return None
    if isinstance(mask, ArakawaCGridMask):
        return jnp.asarray(mask.psi, dtype=jnp.float32)
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
        raise ValueError("method='cg' requires a mask (array or ArakawaCGridMask)")

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
    mask : Float[Array, "Ny Nx"] or ArakawaCGridMask or None
        Domain mask.  Required for ``method="cg"``.  When an
        :class:`ArakawaCGridMask` is passed the ``psi`` staggering mask is
        extracted automatically.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.  Signature:
        ``preconditioner(r: Array) -> Array``.  When ``None``, a spectral
        preconditioner (FFT-based) is used automatically.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Streamfunction ψ.
    """
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
    mask : Float[Array, "Ny Nx"] or ArakawaCGridMask or None
        Domain mask.  Required for ``method="cg"``.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Pressure field p.
    """
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
    mask : Float[Array, "Ny Nx"] or ArakawaCGridMask or None
        Domain mask.  Required for ``method="cg"``.
    capacitance_solver : CapacitanceSolver or None
        Pre-built capacitance solver.  Required for
        ``method="capacitance"``.
    preconditioner : callable or None
        Custom preconditioner for ``method="cg"``.

    Returns
    -------
    Float[Array, "... Ny Nx"]
        Streamfunction ψ, same shape as *pv*.
    """
    lam = jnp.asarray(lambda_)

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

    elif method == "cg":
        mask_arr = _resolve_mask_arr(mask)
        if mask_arr is None:
            raise ValueError("method='cg' requires a mask")

        _precond = preconditioner

        def _solve_layer(
            rhs: Float[Array, "Ny Nx"], lam_i: float
        ) -> Float[Array, "Ny Nx"]:
            def _matvec(x: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
                return masked_laplacian(x, mask_arr, dx, dy, lambda_=lam_i)

            pc = (
                _precond
                if _precond is not None
                else make_spectral_preconditioner(dx, dy, lambda_=lam_i, bc="fft")
            )
            x, _info = solve_cg(_matvec, rhs * mask_arr, preconditioner=pc)
            return x * mask_arr

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

    # vmap over layer axis (pairing each layer with its lambda)
    _solve_layers = eqx.filter_vmap(_solve_layer, in_axes=(0, 0))

    # vmap over the (flattened) batch axis
    out_4d = eqx.filter_vmap(lambda batch: _solve_layers(batch, lam))(pv_4d)
    return out_4d.reshape(shape)
