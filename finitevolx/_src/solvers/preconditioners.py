"""Preconditioners for iterative elliptic solvers.

All preconditioners return a callable ``M_inv(r) -> Array`` that
approximates the action of ``A^{-1}`` on a residual vector.  They are
designed for use with :func:`~finitevolx._src.solvers.iterative.solve_cg`.

Available preconditioners
-------------------------
``make_spectral_preconditioner``
    Rectangular spectral solve (DST/DCT/FFT) as an approximate inverse.
    Nearly free (one FFT pair).  Best when the domain is close to rectangular.

``make_nystrom_preconditioner``
    Low-rank approximate inverse from randomised operator probing.
    Good when the operator is expensive or operator-only access is available.

``make_multigrid_preconditioner``
    Single multigrid V-cycle as an approximate inverse.  Captures both
    high- and low-frequency components.  Best for variable-coefficient
    problems or when spectral preconditioning is insufficient.

``make_preconditioner``
    Convenience factory that dispatches to the above based on a string key.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import gaussx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from finitevolx._src.solvers.multigrid import MultigridSolver

# ---------------------------------------------------------------------------
# Spectral preconditioner
# ---------------------------------------------------------------------------


def make_spectral_preconditioner(
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    bc: str = "fft",
) -> Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]]:
    """Return a spectral-solve preconditioner for PCG.

    The preconditioner applies the rectangular spectral solver as an
    approximate inverse of the Helmholtz operator ``(∇² − λ)``.  It is
    particularly effective when the physical domain is a subset of the
    rectangle (masked-domain problems).

    Parameters
    ----------
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter.  Must match the parameter used in the main
        problem.  Default: 0.0 (Poisson).
    bc : {"fft", "dst", "dct"}
        Spectral solver type to use as the preconditioner.
        Default: ``"fft"`` (periodic).

    Returns
    -------
    Callable
        A function ``M_inv(r: Array) -> Array`` that applies the
        approximate inverse.
    """
    if bc not in {"fft", "dst", "dct"}:
        raise ValueError(f"bc must be 'fft', 'dst', or 'dct'; got {bc!r}")

    from finitevolx._src.solvers.spectral import _spectral_solve

    def _preconditioner(r: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return _spectral_solve(r, dx, dy, lambda_, bc)

    return _preconditioner


# ---------------------------------------------------------------------------
# Randomized Nyström preconditioner
# ---------------------------------------------------------------------------


def make_nystrom_preconditioner(
    matvec: Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]],
    shape: tuple[int, int],
    rank: int = 50,
    key: jax.Array | None = None,
) -> Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]]:
    r"""Build a randomized Nyström preconditioner for a symmetric operator.

    Delegates the low-rank Nyström construction to
    :func:`gaussx.randomized_nystrom` and :class:`gaussx.NystromPreconditioner`
    (Frangella, Tropp & Udell, 2023), then adapts it to the finitevolX
    convention: the returned callable operates on 2-D fields and approximates
    ``A^{-1}`` for the symmetric **negative-definite** operator ``A``.

    gaussx works with positive-semidefinite operators, so this sketches the
    PSD operator ``B = -A`` as ``B ≈ U Λ̂ Uᵀ`` (rank ``l``) and applies

    ``M⁻¹ r = -[ U (Λ̂ + μ)⁻¹ Uᵀ r + (r - U Uᵀ r) / (λ̂_l + μ) ]``,

    i.e. the exact inverse on the captured top of the spectrum and a scalar
    ``1 / (λ̂_l + μ)`` on its complement. ``μ = √ε · λ̂_1`` is a relative
    regulariser (``ε`` the machine epsilon of the dtype), so the
    preconditioner rescales with the operator (contract 6) and stays
    well-defined for a singular ``A`` (e.g. a Neumann Poisson problem).

    The Nyström sketch captures the *largest* eigenvalues of ``-A``. For a
    Laplacian these are the grid-scale modes, so it deflates the top of the
    spectrum but leaves the smooth modes (the small eigenvalues that set the
    condition number) to CG: use ``make_spectral_preconditioner`` or
    ``make_multigrid_preconditioner`` when iteration count matters.

    Parameters
    ----------
    matvec : callable
        Function implementing the symmetric (negative-definite) linear operator
        ``A``.  Signature: ``matvec(x: Array) -> Array``.
    shape : (int, int)
        Spatial shape ``(Ny, Nx)`` of the 2-D fields.
    rank : int
        Number of probing vectors (approximation rank).  Default: 50.
    key : jax.Array or None
        PRNG key for the random probing matrix.  If ``None``, uses
        ``jax.random.PRNGKey(0)``.

    Returns
    -------
    Callable
        A function ``M_inv(r: Array) -> Array`` that applies the
        approximate inverse.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    Ny, Nx = shape
    n = Ny * Nx

    # gaussx Nyström expects a PSD operator; probe -A (which is PSD) on flat
    # vectors, then negate the approximate inverse to recover A^{-1}.
    def _neg_flat_matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
        return -matvec(v.reshape(Ny, Nx)).ravel()

    operator = gaussx.as_linear_operator(
        _neg_flat_matvec, shape=(n, n), positive_semidefinite=True
    )
    # B = -A ≈ U Λ̂ Uᵀ, U: [n, l], Λ̂: [l] descending
    approx = gaussx.randomized_nystrom(operator, min(rank, n), key=key)
    eigenvalues = approx.d
    # μ = √ε · λ̂_1: relative, so it rescales with the operator (no absolute ε)
    shift = jnp.sqrt(jnp.finfo(eigenvalues.dtype).eps) * eigenvalues[0]
    preconditioner = gaussx.NystromPreconditioner(
        basis=approx.U, eigenvalues=eigenvalues, shift=shift
    )
    # P⁻¹ = (λ̂_l + μ) [U (Λ̂ + μ)⁻¹ Uᵀ + (I - U Uᵀ) / (λ̂_l + μ)]
    minv = preconditioner.as_operator()
    scale = eigenvalues[-1] + shift  # undo the (λ̂_l + μ) factor → ≈ (-A)⁻¹

    def _preconditioner(r: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return -minv.mv(r.reshape(n)).reshape(Ny, Nx) / scale

    return _preconditioner


# ---------------------------------------------------------------------------
# Multigrid preconditioner
# ---------------------------------------------------------------------------


def make_multigrid_preconditioner(
    mg_solver: MultigridSolver,
) -> Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]]:
    """Return a preconditioner closure that applies a single multigrid V-cycle.

    The returned callable approximates ``A^{-1} r`` by running one V-cycle
    from a zero initial guess, which is sufficient as a preconditioner
    (it doesn't need to converge — it just needs to be a good approximation).

    This is compatible with :func:`~finitevolx._src.solvers.iterative.solve_cg`:
    pass the returned closure as the ``preconditioner`` argument.  CG then
    converges in very few iterations (typically 5-10 instead of hundreds)
    because multigrid captures both high- and low-frequency components of
    the inverse.

    Parameters
    ----------
    mg_solver : MultigridSolver
        A pre-built multigrid solver
        (from :func:`~finitevolx._src.solvers.multigrid.build_multigrid_solver`).

    Returns
    -------
    callable
        ``preconditioner(r) -> approx_solution``, where ``r`` has shape
        ``(Ny, Nx)`` and the output has the same shape.

    Examples
    --------
    >>> mg = build_multigrid_solver(mask, dx, dy, lambda_=10.0)
    >>> precond = make_multigrid_preconditioner(mg)
    >>> u, info = solve_cg(A, rhs, preconditioner=precond)
    """

    def _preconditioner(r: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return mg_solver.v_cycle(jnp.zeros_like(r), r)

    return _preconditioner


# ---------------------------------------------------------------------------
# Preconditioner factory
# ---------------------------------------------------------------------------


def make_preconditioner(
    kind: str,
    *,
    dx: float | None = None,
    dy: float | None = None,
    lambda_: float = 0.0,
    bc: str = "fft",
    matvec: Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]] | None = None,
    shape: tuple[int, int] | None = None,
    rank: int = 50,
    key: jax.Array | None = None,
    mg_solver: MultigridSolver | None = None,
) -> Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]]:
    """Convenience factory that builds a preconditioner by name.

    Dispatches to :func:`make_spectral_preconditioner`,
    :func:`make_nystrom_preconditioner`, or
    :func:`make_multigrid_preconditioner` based on *kind*.

    Parameters
    ----------
    kind : {"spectral", "nystrom", "multigrid"}
        Which preconditioner to build.

    dx, dy : float or None
        Grid spacings.  Required for ``kind="spectral"``.
    lambda_ : float
        Helmholtz parameter.  Used by ``kind="spectral"``.  Default: 0.0.
    bc : {"fft", "dst", "dct"}
        Spectral solver type.  Used by ``kind="spectral"``.  Default: ``"fft"``.

    matvec : callable or None
        Operator ``A``.  Required for ``kind="nystrom"``.
    shape : (int, int) or None
        Spatial shape ``(Ny, Nx)``.  Required for ``kind="nystrom"``.
    rank : int
        Approximation rank.  Used by ``kind="nystrom"``.  Default: 50.
    key : jax.Array or None
        PRNG key.  Used by ``kind="nystrom"``.

    mg_solver : MultigridSolver or None
        Pre-built multigrid solver.  Required for ``kind="multigrid"``.

    Returns
    -------
    callable
        A function ``M_inv(r: Array) -> Array`` that applies the
        approximate inverse, compatible with :func:`solve_cg`.

    Raises
    ------
    ValueError
        If *kind* is unknown or required arguments are missing.

    Examples
    --------
    >>> pc = make_preconditioner("spectral", dx=dx, dy=dy, lambda_=10.0)
    >>> pc = make_preconditioner("nystrom", matvec=A, shape=(64, 64), rank=30)
    >>> pc = make_preconditioner("multigrid", mg_solver=mg)
    """
    if kind == "spectral":
        if dx is None or dy is None:
            raise ValueError("kind='spectral' requires dx and dy")
        return make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc=bc)

    if kind == "nystrom":
        if matvec is None or shape is None:
            raise ValueError("kind='nystrom' requires matvec and shape")
        return make_nystrom_preconditioner(matvec, shape, rank=rank, key=key)

    if kind == "multigrid":
        if mg_solver is None:
            raise ValueError("kind='multigrid' requires mg_solver")
        return make_multigrid_preconditioner(mg_solver)

    raise ValueError(
        f"kind must be 'spectral', 'nystrom', or 'multigrid'; got {kind!r}"
    )
