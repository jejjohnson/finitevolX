"""Preconditioners for iterative elliptic solvers.

Spectral preconditioner
-----------------------
:func:`make_spectral_preconditioner` returns a callable that applies a
rectangular spectral solve (DST/DCT/FFT) as an approximate inverse of the
Helmholtz operator ``(∇² − λ)``.  Effective for masked-domain problems
where the physical domain is a subset of a rectangle.

Randomized Nyström preconditioner
---------------------------------
:func:`make_nystrom_preconditioner` builds a low-rank approximate inverse
from randomized probing of the operator.  Useful when the operator is
expensive or when only matrix-vector products are available.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

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

    Uses random probing vectors to approximate the action of ``A^{-1}`` via
    a low-rank Nyström factorisation.  The resulting preconditioner is a
    callable that can be passed to :func:`~finitevolx._src.solvers.iterative.solve_cg`.

    Algorithm
    ---------
    1. Draw a Gaussian random matrix ``Ω ∈ ℝ^{n × k}`` (``k = rank``).
    2. Compute ``Y = A Ω`` (``k`` matvec applications).
    3. Form the small matrix ``B = Ω^T Y ∈ ℝ^{k × k}``.
    4. Compute ``B = U S U^T`` (eigendecomposition of the small matrix).
    5. The preconditioner applies ``M^{-1} x ≈ (Ω U S^{-1} U^T Ω^T) x``.

    The operator ``A`` is assumed to be symmetric **negative definite**.
    The eigenvalues of ``B`` will be negative; their absolute values are
    used internally for inversion.

    Parameters
    ----------
    matvec : callable
        Function implementing the symmetric linear operator ``A``.
        Signature: ``matvec(x: Array) -> Array``.
    shape : (int, int)
        Spatial shape ``(Ny, Nx)`` of the 2-D fields.
    rank : int
        Number of probing vectors (approximation rank).  Higher values
        give a better preconditioner but cost more setup time.
        Default: 50.
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

    # Clamp rank to problem size
    k = min(rank, n)

    # Step 1: Random probing matrix Omega in R^{n x k}
    omega = jax.random.normal(key, (n, k))

    # Step 2: Y = A Ω  (k matvec applications)
    def _apply_col(col: Float[Array, " n"]) -> Float[Array, " n"]:
        return matvec(col.reshape(Ny, Nx)).ravel()

    Y = jax.vmap(_apply_col, in_axes=1, out_axes=1)(omega)  # [n, k]

    # Step 3: Small matrix B = Omega^T Y in R^{k x k}
    B = omega.T @ Y  # [k, k]

    # Step 4: Eigendecomposition of B (symmetric)
    eigvals, U = jnp.linalg.eigh(B)  # eigvals sorted ascending

    # For a negative-definite A, eigvals should be negative.
    # Invert using absolute values, with a floor for numerical safety.
    abs_eigvals = jnp.abs(eigvals)
    eps = jnp.finfo(abs_eigvals.dtype).eps * n
    s_inv = jnp.where(abs_eigvals > eps, 1.0 / abs_eigvals, 0.0)
    # Preserve the sign: A^{-1} is also negative definite
    s_inv = -s_inv  # negate so preconditioner ≈ A^{-1} (negative)

    # Precompute the projection matrix: P = Ω U diag(s_inv) Uᵀ Ωᵀ
    # For memory efficiency, store the factor F = Ω U diag(sqrt|s_inv|)
    # and apply as F Fᵀ x (with sign).
    # But simpler: store W = Omega @ U in R^{n x k} and s_inv in R^{k}.
    W = omega @ U  # [n, k]

    def _preconditioner(r: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        r_flat = r.ravel()  # [n]
        # M^{-1} r ≈ W diag(s_inv) Wᵀ r
        coeffs = W.T @ r_flat  # [k]
        result = W @ (s_inv * coeffs)  # [n]
        return result.reshape(Ny, Nx)

    return _preconditioner
