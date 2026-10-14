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

``make_gaussx_multigrid_preconditioner``
    The same V-cycle, symmetrised (V, then its adjoint), as a
    ``gaussx.OperatorPreconditioner`` (positive sign convention, flat
    vectors) for gaussx's CG strategies.  Unlike the other ``make_*``
    factories it returns a gaussx object, not a 2-D closure.
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


def make_gaussx_multigrid_preconditioner(
    mg_solver: MultigridSolver,
) -> gaussx.OperatorPreconditioner:
    r"""The multigrid V-cycle as a symmetric gaussx preconditioner.

    Wraps a pre-built :class:`~finitevolx.MultigridSolver` in
    ``gaussx.OperatorPreconditioner``, so the geometric V-cycle plugs into
    gaussx's strategies (``gaussx.CGSolver(preconditioner=...)``,
    ``gaussx.PreconditionedCGSolver``, ``gaussx.linear_solve``).

    gaussx works with positive semidefinite systems, so it solves
    ``(−A) x = −b`` for finitevolX's negative (semi)definite Helmholtz
    operator ``A = ∇·(c∇) − λ``. Let ``V(v) = −Vcycle(0, v) ≈ (−A)⁻¹ v``.
    One V-cycle is **not** symmetric: restriction (mask-normalised 2×2
    averaging) is not the transpose of prolongation (bilinear), so ``V``
    differs from ``Vᵀ`` by a few per cent, and CG needs a symmetric positive
    definite preconditioner. With ``K = −A``, this applies the symmetrised
    two-step iteration (V, then its adjoint)

    ``M⁻¹ v = V v + Vᵀ v − Vᵀ K V v``,

    with ``Vᵀ`` the exact transpose of the (linear) V-cycle from
    ``jax.linear_transpose``: two V-cycles and one operator application per
    use. ``M⁻¹`` is symmetric by construction, and since
    ``I − M⁻¹K = (I − VᵀK)(I − VK)`` it is positive definite exactly when
    one V-cycle reduces the error in the energy norm, ``‖I − VK‖_K < 1`` —
    i.e. when the multigrid itself converges. The default
    ``build_multigrid_solver`` settings do; a divergent configuration (one
    smoothing sweep with ``omega`` near 1, almost no coarse solve, high
    coefficient contrast) makes no symmetric form of the V-cycle positive
    definite, so check the solver converges on its own first. On the default
    settings this takes fewer CG steps than either the raw V-cycle or its
    symmetric part ``½(V + Vᵀ)``.

    The V-cycle runs in the hierarchy's dtype (float64); each application
    returns the dtype of the vector it was given. The structure comes from
    the system operator, so pass it when calling the preconditioner directly:
    ``pc(v, operator)``.

    For finitevolX's own :func:`~finitevolx.solve_cg`, which takes a callable
    approximating ``A⁻¹`` on 2-D fields, use
    :func:`make_multigrid_preconditioner` on the same solver.

    Parameters
    ----------
    mg_solver : MultigridSolver
        A pre-built solver from :func:`~finitevolx.build_multigrid_solver`.

    Returns
    -------
    gaussx.OperatorPreconditioner
        Applies ``M⁻¹`` on flat vectors of length ``Ny·Nx`` (row-major).

    Examples
    --------
    >>> mg = build_multigrid_solver(mask, dx, dy, lambda_=10.0)
    >>> neg_A = gaussx.as_linear_operator(
    ...     lambda v: -A(v.reshape(Ny, Nx)).ravel(),
    ...     shape=(Ny * Nx, Ny * Nx),
    ...     positive_semidefinite=True,
    ... )
    >>> pc = make_gaussx_multigrid_preconditioner(mg)
    >>> x = gaussx.CGSolver(preconditioner=pc).solve(neg_A, -rhs.ravel())
    """
    from finitevolx._src.solvers.multigrid import _apply_operator

    level0 = mg_solver.levels[0]
    mask = level0.mask  # [Ny, Nx], the hierarchy's dtype
    shape = mask.shape

    def _v_cycle(v: Float[Array, " n"]) -> Float[Array, " n"]:
        # V v = -Vcycle(0, v) ~ K^-1 v with K = -A, [Ny*Nx]; linear in v
        r = v.reshape(shape)  # [Ny, Nx]
        return -mg_solver.v_cycle(jnp.zeros_like(r), r).ravel()

    def _k(v: Float[Array, " n"]) -> Float[Array, " n"]:
        # K v = -A v = -(div(c grad) - lambda) v on the finest level, [Ny*Nx]
        return -_apply_operator(v.reshape(shape), level0).ravel()

    _v_cycle_t = jax.linear_transpose(
        _v_cycle, jax.ShapeDtypeStruct((mask.size,), mask.dtype)
    )

    def _approx_inverse(v: Float[Array, " n"]) -> Float[Array, " n"]:
        w = v.astype(mask.dtype)  # [Ny*Nx], hierarchy dtype
        vw = _v_cycle(w)  # V w
        # M^-1 w = V w + V^T w - V^T K V w = V w + V^T (w - K V w)
        (vt,) = _v_cycle_t(w - _k(vw))
        return (vw + vt).astype(v.dtype)

    return gaussx.OperatorPreconditioner(approx_inverse=_approx_inverse)


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
        ``(Ny, Nx)`` and the output has the same shape.  For gaussx's
        solvers, use :func:`make_gaussx_multigrid_preconditioner`.

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
