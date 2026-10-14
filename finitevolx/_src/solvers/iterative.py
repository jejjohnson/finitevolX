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
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import lineax as lx

from finitevolx._src.mask import Mask2D

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
    atol: float | None = None,
    max_steps: int | None = 500,
    *,
    throw: bool = False,
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
    atol : float or None
        Absolute convergence tolerance.  ``None`` (the default) resolves it
        from the dtype: ``1e-6`` in float64; below float64, the solve runs on
        the rescaled system ``A y = rhs / s`` with ``s = max|rhs_i|`` and
        ``atol = √ε``, and returns ``s y``.  See Notes.
    max_steps : int or None
        Maximum CG iterations.  ``None`` means unlimited.  Default: 500.
    throw : bool
        Raise when CG does not converge.  Default ``False``: the iterate is
        returned and ``info.converged`` reports the failure, so check it.

    Notes
    -----
    lineax's CG stops when every entry satisfies
    ``|r_i| <= atol + rtol |rhs_i|``.  Where ``rhs_i = 0`` (land, the ghost
    ring, quiet regions) that is ``|r_i| <= atol`` with an *absolute*
    ``atol``.  A fixed ``1e-6`` sits at float32's rounding floor once
    ``|rhs| ~ 1`` and below it for an elliptic right-hand side of order
    ``1/dx²``, so float32 CG never meets it: it runs to ``max_steps`` and can
    diverge with a spectral preconditioner.  Rescaling by ``s = max|rhs_i|``
    makes the effective ``atol = √ε · max|rhs_i|`` relative to the
    right-hand side (and unit-agnostic).  ``s`` carries no gradient: it only
    changes the iteration count.  This is the policy gaussx's CG strategies
    use (gaussx gh-639).

    In float64 the default stays the historical *absolute* ``atol = 1e-6``,
    which is not unit-agnostic: a right-hand side far below ``1e-6`` in
    magnitude (e.g. SI vorticity ``~1e-9``) is "converged" at the zero
    iterate.  Pass an ``atol`` scaled to the right-hand side in that case.

    Returns
    -------
    x : Float[Array, "..."]
        Approximate solution, same shape as *rhs*.
    info : CGInfo
        Named tuple with fields ``iterations``, ``residual_norm``, and
        ``converged``.
    """
    atol, scale = _cg_atol_and_scale(atol, rhs)
    rhs_solve = rhs if scale is None else rhs / scale

    zero = jnp.zeros_like(rhs)
    # Tag as NSD; caller is responsible for ensuring A is actually negative (semi)definite.
    operator = lx.FunctionLinearOperator(
        matvec, zero, tags=[lx.negative_semidefinite_tag]
    )
    solver = lx.CG(rtol=rtol, atol=atol, max_steps=max_steps)

    options: dict = {}
    if x0 is not None:
        options["y0"] = x0 if scale is None else x0 / scale
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

    sol = lx.linear_solve(
        operator, rhs_solve, solver=solver, options=options, throw=throw
    )

    x_out = sol.value if scale is None else sol.value * scale
    res_norm = jnp.linalg.norm(matvec(x_out) - rhs)
    info = CGInfo(
        iterations=sol.stats["num_steps"],
        residual_norm=res_norm,
        converged=sol.result == lx.RESULTS.successful,
    )
    return x_out, info


def _cg_atol_and_scale(
    atol: float | None, rhs: Float[Array, ...]
) -> tuple[float, Float[Array, ""] | None]:
    """Resolve ``atol`` and the right-hand-side scale for :func:`solve_cg`.

    Returns ``(atol, None)`` when ``atol`` is set or ``rhs`` is float64 or
    wider (``None`` → ``1e-6``, the historical default), else
    ``(√ε, s)`` with ``s = max|rhs_i|`` (gradient stopped; ``1`` for a zero or
    non-finite ``rhs``): solve ``A y = rhs / s``, return ``s y``.
    """
    if atol is not None:
        return atol, None
    finfo = jnp.finfo(rhs.dtype)
    if finfo.bits >= 64 or rhs.size == 0:
        return 1e-6, None
    # s = max_i |rhs_i|, a constant of the solve (it only sets the iteration count)
    scale = jax.lax.stop_gradient(jnp.max(jnp.abs(rhs)))
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0), scale, jnp.ones_like(scale))
    return math.sqrt(float(finfo.eps)), scale


# ---------------------------------------------------------------------------
# Masked Laplacian operator
# ---------------------------------------------------------------------------


def masked_laplacian(
    psi: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"] | Mask2D,
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
    mask : Float[Array, "Ny Nx"] or Mask2D
        Binary mask: 1 inside the physical domain, 0 outside (land/exterior).
        When an :class:`Mask2D` is passed, the ``psi`` staggering
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
    if isinstance(mask, Mask2D):
        mask_arr = mask.xy_corner_strict.astype(psi.dtype)
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
