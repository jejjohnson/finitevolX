"""Spectral preconditioner for iterative elliptic solvers.

``make_spectral_preconditioner`` returns a callable ``M_inv(r) -> Array`` that
approximates the action of ``A^{-1}`` on a residual: the rectangular spectral
solve (DST/DCT/FFT) as an approximate inverse.  Nearly free (one FFT pair),
and best when the domain is close to rectangular.  It is meant for
``gaussx.CGSolver(preconditioner=...)``.

For other preconditioners use ``MultigridSolver.as_preconditioner()`` or
``gaussx.NystromPreconditioner``.
"""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import Array, Float

# ---------------------------------------------------------------------------
# Spectral preconditioner
# ---------------------------------------------------------------------------


def make_spectral_preconditioner(
    dx: float,
    dy: float,
    lambda_: float | Array = 0.0,
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

    from gridax._src.solvers.spectral import _spectral_solve

    def _preconditioner(r: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return _spectral_solve(r, dx, dy, lambda_, bc)

    return _preconditioner
