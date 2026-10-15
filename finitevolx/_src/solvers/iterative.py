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
Dirichlet conditions at the mask boundary and at the array edge.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import lineax as lx
import numpy as np
from scipy import ndimage

from finitevolx._src.mask import Mask2D

# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------


class CGInfo(NamedTuple):
    """Convergence diagnostics returned by :func:`solve_cg`."""

    iterations: int
    """Number of PCG iterations performed."""
    residual_norm: float
    """L2 norm of the explicit final residual ``rhs - A(x)``."""
    converged: bool
    """True if lineax reported success *and* the explicit residual satisfies
    ``max|rhs - A(x)| <= atol + rtol max|rhs|``."""


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
    and can make CG stall (``info.converged`` is then ``False``); use
    ``lambda = 0`` only if the operator is strictly negative semidefinite
    via the mask (e.g. the masked Laplacian with homogeneous Dirichlet BCs
    on a non-trivial domain).
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
        Raise when CG does not converge (as ``info.converged`` defines it),
        through :func:`equinox.error_if`, so it also works under ``jit``.
        Default ``False``: the iterate is returned and ``info.converged``
        reports the failure, so check it.

    Notes
    -----
    **Convergence.**  ``max_steps`` exhaustion never raises unless
    ``throw=True``.  ``info.converged`` is checked against the explicit
    residual of the *un-preconditioned* system, ``r = rhs - A(x)``:
    ``max|r_i| <= atol + rtol max|rhs_i|`` with the ``atol`` / ``rtol`` the
    solve used (``s · √ε`` for the rescaled float32 solve).  lineax alone
    reports success when CG stalls, e.g. after 0–1 iterations on an
    indefinite operator (``lambda_ < 0``) or with a preconditioner of the
    wrong sign; this check reports ``converged=False`` there.

    **Masks.**  The returned ``x`` can be non-zero outside the mask: a
    preconditioner that ignores the mask (the spectral ones) leaks into the
    dry points, which the masked operator then never sees.  Multiply by the
    mask before using ``x``; the wrappers (:func:`streamfunction_from_vorticity`,
    :func:`pressure_from_divergence`, :func:`pv_inversion`) do.

    **Gradients.**  The solve is differentiable with respect to anything
    ``matvec`` and ``rhs`` close over (``lambda_``, ``dx``, …), through
    lineax's implicit derivative.  ``x0`` and the preconditioner do not
    change the converged solution, so their gradient is stopped; that is
    what lets a preconditioner built from the same ``lambda_`` / ``dx``
    stay in a differentiated solve.

    **Tolerances.**  lineax's CG stops when every entry satisfies
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
    # lineax refuses tangents through ``options``.  Neither the initial guess
    # nor the preconditioner changes the converged solution x = A⁻¹ b, so its
    # derivative (implicit function theorem, through ``operator``) needs no
    # tangent from them: stop the gradient of their closed-over arrays
    # (e.g. ``lambda_`` / ``dx`` captured by a spectral preconditioner).
    options = jax.tree_util.tree_map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
        options,
    )

    # throw=False: max_steps / breakdown are reported through `converged`.
    sol = lx.linear_solve(
        operator, rhs_solve, solver=solver, options=options, throw=False
    )

    x_out = sol.value if scale is None else sol.value * scale
    # Explicit residual of the un-preconditioned system, r = b - A x  [Ny, Nx].
    # lineax's own success flag is not enough: it also exits "successfully"
    # on a stalled residual (indefinite A, wrong-sign preconditioner).
    residual = rhs - matvec(x_out)
    # ‖r‖∞ ≤ atol' + rtol ‖b‖∞, implied by lineax's entrywise stopping rule
    # |r_i| ≤ atol + rtol |b_i| (atol' = s · atol on the rescaled system).
    atol_eff = atol if scale is None else atol * scale
    tol = atol_eff + rtol * jnp.max(jnp.abs(rhs), initial=0.0)
    converged = (sol.result == lx.RESULTS.successful) & (
        jnp.max(jnp.abs(residual), initial=0.0) <= tol
    )
    if throw:
        x_out = eqx.error_if(
            x_out,
            jnp.logical_not(converged),
            "solve_cg did not converge: max_steps reached, or the residual "
            "|b - A x| exceeds atol + rtol |b| (an indefinite operator, e.g. "
            "lambda_ < 0, or a preconditioner of the wrong sign).",
        )
    info = CGInfo(
        iterations=sol.stats["num_steps"],
        residual_norm=jnp.linalg.norm(residual),
        converged=converged,
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
    lambda_: float | Array = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Apply the masked discrete Helmholtz operator (∇² − λ)·ψ.

    Enforces homogeneous Dirichlet conditions at the mask boundary by zeroing
    *psi* outside the mask before applying the 5-point stencil.  The output
    is also zeroed outside the mask.

    Neighbours beyond the array edge are zero (dry), so the array edge is a
    homogeneous Dirichlet boundary one node outside the array, wherever the
    mask is wet there; nothing wraps around.  On an all-ones mask this is
    the dense Dirichlet 5-point operator that the DST-I solvers invert, so
    ``make_spectral_preconditioner(..., bc="dst")`` matches it exactly on a
    rectangle; the FFT preconditioner (periodic) is an approximation.

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
    psi_m = psi * mask_arr  # enforce zero outside domain  [Ny, Nx]
    # Zero ring outside the array (Dirichlet, not periodic); pad once and
    # bind each neighbour view once, so the stencil lowers to one fused pass.
    p = jnp.pad(psi_m, 1)  # [Ny+2, Nx+2]
    east = p[1:-1, 2:]  # ψ[j, i+1]  [Ny, Nx]
    west = p[1:-1, :-2]  # ψ[j, i-1]  [Ny, Nx]
    north = p[2:, 1:-1]  # ψ[j+1, i]  [Ny, Nx]
    south = p[:-2, 1:-1]  # ψ[j-1, i]  [Ny, Nx]
    # ∇²ψ[j, i] = (ψ[j, i+1] + ψ[j, i-1] - 2ψ[j, i]) / dx²
    #           + (ψ[j+1, i] + ψ[j-1, i] - 2ψ[j, i]) / dy²
    lap = (east + west - 2.0 * psi_m) / dx**2 + (north + south - 2.0 * psi_m) / dy**2
    return (lap - lambda_ * psi_m) * mask_arr  # [Ny, Nx]


def _masked_neumann_laplacian(
    p: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Masked Helmholtz operator (∇² − λ)·p with zero flux through every coast face.

    The Neumann counterpart of :func:`masked_laplacian`, for T-point fields
    such as pressure: a face carries flux only when both of its cells are
    wet, so ∂p/∂n = 0 on every wet–dry face and on the array edge.  On the
    T mask ``[1:-1, 1:-1]`` of a ``from_interior`` rectangle it is exactly the
    DCT-II (Neumann-on-the-faces) operator.  Symmetric and negative
    semidefinite; at λ = 0 its null space is the constants on each connected
    wet component.  The output is zeroed outside the mask.
    """
    return _masked_t_laplacian(p, mask, dx, dy, lambda_, face_dirichlet=False)


def _masked_face_dirichlet_laplacian(
    p: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array = 0.0,
) -> Float[Array, "Ny Nx"]:
    """Masked Helmholtz operator (∇² − λ)·p with p = 0 on every coast *face*.

    For T-point fields: the wall sits on the wet–dry face, half a cell from
    the wet centre, so the face flux is ``(0 − p) / (h/2)``.  This is where
    the T-point spectral Dirichlet solve (DST-II) puts the wall, and it is
    exactly that operator on the T mask of a ``from_interior`` rectangle.
    (:func:`masked_laplacian` instead holds p = 0 at the dry cell *centre*,
    a full cell away: right for ψ at X points, first order for T points.)
    Symmetric and negative definite.  The output is zeroed outside the mask.
    """
    return _masked_t_laplacian(p, mask, dx, dy, lambda_, face_dirichlet=True)


def _masked_t_laplacian(
    p: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array,
    face_dirichlet: bool,
) -> Float[Array, "Ny Nx"]:
    """Five-point (∇² − λ)·p on a masked T field with the wall on the faces.

    Open faces (both cells wet) carry ``p_nb − p``; a coast face (wet–dry,
    or the array edge) carries 0 (zero flux) or, with *face_dirichlet*,
    ``−2 p`` (p = 0 on the face, half a cell away).
    """
    c = 2.0 if face_dirichlet else 0.0
    p_m = p * mask  # [Ny, Nx]
    # Pad once with a dry ring and bind each neighbour view once (one fused pass).
    pp = jnp.pad(p_m, 1)  # [Ny+2, Nx+2]
    mp = jnp.pad(mask, 1)  # [Ny+2, Nx+2]
    # Face weights: mu_e[j, i] = m[j, i] m[j, i+1] (the face at i+1/2 is open), etc.
    mu_e = mask * mp[1:-1, 2:]
    mu_w = mask * mp[1:-1, :-2]
    mu_n = mask * mp[2:, 1:-1]
    mu_s = mask * mp[:-2, 1:-1]
    # ∇²p[j, i] = (mu_e (p[j, i+1] - p) - mu_w (p - p[j, i-1])
    #              - c (2 - mu_e - mu_w) p) / dx²            (x faces)
    #           + (mu_n (p[j+1, i] - p) - mu_s (p - p[j-1, i])
    #              - c (2 - mu_n - mu_s) p) / dy²            (y faces)
    # where (1 - mu) = 1 on a coast face of a wet cell.
    lap_x = mu_e * (pp[1:-1, 2:] - p_m) - mu_w * (p_m - pp[1:-1, :-2])
    lap_y = mu_n * (pp[2:, 1:-1] - p_m) - mu_s * (p_m - pp[:-2, 1:-1])
    if face_dirichlet:
        lap_x = lap_x - c * (2.0 - mu_e - mu_w) * p_m
        lap_y = lap_y - c * (2.0 - mu_n - mu_s) * p_m
    lap = lap_x / dx**2 + lap_y / dy**2
    return (lap - lambda_ * p_m) * mask  # [Ny, Nx]


def _wet_components(mask: np.ndarray) -> tuple[Array, Array]:
    """Label the 4-connected wet components of a concrete mask (construction time).

    Returns ``(labels, counts)`` for :func:`_project_zero_mean`: ``labels``
    is 0 on dry cells and 1..n on the wet components, ``counts[k]`` the wet
    cells of component ``k`` (``counts[0]`` set to 1, never used).
    """
    labels, n_comp = ndimage.label(np.asarray(mask) > 0.5)  # 4-connectivity
    counts = np.bincount(labels.ravel(), minlength=n_comp + 1).astype(np.float64)
    counts[0] = 1.0  # dry label: never used as a divisor of wet values
    return jnp.asarray(labels), jnp.asarray(counts)


def _project_zero_mean(
    u: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
    labels: Int[Array, "Ny Nx"],
    counts: Float[Array, " n"],
) -> Float[Array, "Ny Nx"]:
    """Remove the wet mean of *u* on each connected wet component.

    The Neumann operator at ``lambda = 0`` annihilates a constant on every
    connected wet component, so ``A u = rhs`` is solvable only for a
    zero-mean ``rhs`` and fixes ``u`` up to those constants.  This
    orthogonal projection (symmetric, idempotent) imposes the zero-mean
    gauge::

        u[j, i] <- u[j, i] - mean(u over the component of (j, i))

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        Field to project.
    mask : Float[Array, "Ny Nx"]
        Wet mask.
    labels : Int[Array, "Ny Nx"]
        Component label of each cell (0 = dry, 1..n wet components), from
        :func:`_wet_components`.
    counts : Float[Array, " n"]
        Wet-cell count per label (``counts[0]`` is unused, set to 1).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Projected field, zero on dry cells.
    """
    mask = mask.astype(u.dtype)
    u_m = u * mask
    # per-component sum, [n_components + 1]
    sums = jax.ops.segment_sum(u_m.ravel(), labels.ravel(), counts.shape[0])
    mean = (sums / counts.astype(u.dtype))[labels]  # [Ny, Nx]
    return (u_m - mean) * mask
