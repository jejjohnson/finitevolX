"""Spectral elliptic solvers — re-exported from spectraldiffx.

Solves the equation (∇² − λ) ψ = f on rectangular domains using
DST-I (Dirichlet), DCT-II (Neumann), or FFT (periodic) spectral methods.

See spectraldiffx documentation for full solver details.
"""

from __future__ import annotations

from collections.abc import Callable

from spectraldiffx import (
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
# Internal dispatch helper
# ---------------------------------------------------------------------------

# Lookup for Helmholtz solvers by BC type — safe to call with JAX tracers.
_HELMHOLTZ_DISPATCH: dict[str, Callable] = {
    "fft": solve_helmholtz_fft,
    "dst": solve_helmholtz_dst,
    "dct": solve_helmholtz_dct,
}


def _spectral_solve(
    rhs,
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
):
    """Dispatch to the appropriate rectangular spectral solver.

    Uses the Helmholtz solvers unconditionally (they handle lambda=0
    internally via null-mode guards) so that this function is safe to
    call when *lambda_* is a JAX tracer (e.g. inside ``vmap``).
    """
    solver = _HELMHOLTZ_DISPATCH.get(bc)
    if solver is None:
        raise ValueError(f"base_bc must be 'fft', 'dst', or 'dct'; got {bc!r}")
    return solver(rhs, dx, dy, lambda_)


__all__ = [
    "dst1_eigenvalues",
    "dct2_eigenvalues",
    "fft_eigenvalues",
    "solve_helmholtz_dst",
    "solve_helmholtz_dct",
    "solve_helmholtz_fft",
    "solve_poisson_dst",
    "solve_poisson_dct",
    "solve_poisson_fft",
]
