"""Iterative solvers for 2-D elliptic PDEs.

Preconditioned Conjugate Gradient (CG)
---------------------------------------
:func:`solve_cg` solves ``A·ψ = f`` for any symmetric operator ``A`` via the
Preconditioned Conjugate Gradient algorithm.  It runs inside JAX's
``lax.while_loop`` so the entire iteration is JIT-compilable.

Masked Laplacian
----------------
:func:`masked_laplacian` applies the discrete Helmholtz operator
``(∇² − λ)`` on a masked (irregular) domain, enforcing homogeneous
Dirichlet conditions at the mask boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from finitevolx._src.mask.cgrid_mask import ArakawaCGridMask

# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------


class CGInfo(NamedTuple):
    """Convergence diagnostics returned by :func:`solve_cg`."""

    iterations: int
    """Number of PCG iterations performed."""
    residual_norm: float
    """L2 norm of the final residual ``A(x) - rhs``."""
    converged: bool
    """True if the solver converged within the requested tolerances."""


# ---------------------------------------------------------------------------
# Preconditioned Conjugate Gradient solver
# ---------------------------------------------------------------------------


def solve_cg(
    matvec: Callable[[Float[Array, ...]], Float[Array, ...]],
    rhs: Float[Array, ...],
    x0: Float[Array, ...] | None = None,
    preconditioner: Callable[[Float[Array, ...]], Float[Array, ...]] | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int | None = 500,
) -> tuple[Float[Array, ...], CGInfo]:
    """Preconditioned Conjugate Gradient solver for symmetric linear operators.

    Solves ``A(x) = rhs`` where ``matvec`` implements the symmetric linear
    operator ``A``.  The operator need not be explicitly formed; only
    matrix-vector products are required.

    The solve is delegated to :class:`lineax.CG`, which provides a
    numerically stabilised PCG implementation with periodic residual
    recomputation to guard against floating-point drift.

    The operator ``A`` is assumed to be *negative definite*, which is
    the case for the Helmholtz operator ``(nabla^2 - lambda)`` when
    ``lambda > 0``.  Note that ``lambda < 0`` yields an indefinite operator
    and will cause CG to fail; use ``lambda = 0`` only if the operator is
    strictly negative semidefinite via the mask (e.g. the masked Laplacian
    with homogeneous Dirichlet BCs on a non-trivial domain).
    The preconditioner, if supplied, should approximate ``A^{-1}``
    (the negative-definite inverse).  The required sign-flip for
    ``lineax.CG`` is handled internally.

    Parameters
    ----------
    matvec : callable
        Function implementing the symmetric linear operator ``A``.
        Signature: ``matvec(x: Array) -> Array``.
    rhs : Float[Array, "..."]
        Right-hand side of the linear system.
    x0 : Float[Array, "..."] or None
        Initial guess.  Defaults to zeros.
    preconditioner : callable or None
        Optional approximate inverse of ``A``.
        Signature: ``preconditioner(r: Array) -> Array``.
        When ``None``, no preconditioning is applied (identity).
    rtol : float
        Relative convergence tolerance.  Default: 1e-6.
    atol : float
        Absolute convergence tolerance.  Default: 1e-6.
    max_steps : int or None
        Maximum CG iterations.  ``None`` means unlimited.  Default: 500.

    Returns
    -------
    x : Float[Array, "..."]
        Approximate solution, same shape as *rhs*.
    info : CGInfo
        Named tuple with fields ``iterations``, ``residual_norm``, and
        ``converged``.
    """
    zero = jnp.zeros_like(rhs)
    # Tag as NSD; caller is responsible for ensuring A is actually negative (semi)definite.
    operator = lx.FunctionLinearOperator(
        matvec, zero, tags=[lx.negative_semidefinite_tag]
    )
    solver = lx.CG(rtol=rtol, atol=atol, max_steps=max_steps)

    options: dict = {}
    if x0 is not None:
        options["y0"] = x0
    if preconditioner is not None:
        # lineax CG internally negates the NSD operator to make it positive, so it
        # expects a preconditioner that approximates (-A)^{-1} = -A^{-1} (positive).
        # The caller's `preconditioner` approximates A^{-1} (negative), so we negate
        # it here; this is transparent to the caller.
        def _lx_precond(r: Array) -> Array:
            return -preconditioner(r)  # negate: maps caller's A^{-1} -> (-A)^{-1}

        options["preconditioner"] = lx.FunctionLinearOperator(
            _lx_precond, zero, tags=[lx.positive_semidefinite_tag]
        )

    sol = lx.linear_solve(operator, rhs, solver=solver, options=options)

    x_out = sol.value
    res_norm = jnp.linalg.norm(matvec(x_out) - rhs)
    info = CGInfo(
        iterations=sol.stats["num_steps"],
        residual_norm=res_norm,
        converged=sol.result == lx.RESULTS.successful,
    )
    return x_out, info


# ---------------------------------------------------------------------------
# Masked Laplacian operator
# ---------------------------------------------------------------------------


def masked_laplacian(
    psi: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"] | ArakawaCGridMask,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Apply the masked discrete Helmholtz operator (∇² − λ)·ψ.

    Enforces homogeneous Dirichlet conditions at the mask boundary by zeroing
    *psi* outside the mask before applying the 5-point stencil.  The output
    is also zeroed outside the mask.

    Neighbors at the rectangle edges wrap around (periodic roll), which is
    consistent with using the FFT as a preconditioner.

    Parameters
    ----------
    psi : Float[Array, "Ny Nx"]
        Field to which the operator is applied.
    mask : Float[Array, "Ny Nx"] or ArakawaCGridMask
        Binary mask: 1 inside the physical domain, 0 outside (land/exterior).
        When an :class:`ArakawaCGridMask` is passed, the ``psi`` staggering
        mask is used (all four surrounding h-cells must be wet).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter.  Default: 0.0 (pure Laplacian).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Result of (∇² − λ)·(ψ·mask), zeroed outside the mask.
    """
    if isinstance(mask, ArakawaCGridMask):
        mask_arr = mask.psi.astype(psi.dtype)
    else:
        mask_arr = mask
    psi_m = psi * mask_arr  # enforce zero outside domain
    # 5-point finite-difference stencil with periodic roll at edges
    lap = (
        jnp.roll(psi_m, 1, axis=1) + jnp.roll(psi_m, -1, axis=1) - 2.0 * psi_m
    ) / dx**2 + (
        jnp.roll(psi_m, 1, axis=0) + jnp.roll(psi_m, -1, axis=0) - 2.0 * psi_m
    ) / dy**2
    return (lap - lambda_ * psi_m) * mask_arr
