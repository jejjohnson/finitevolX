"""The C-grid Helmholtz operator as a structured gaussx linear operator.

finitevolX solves the Helmholtz equation ``(∇² − λ) ψ = f`` with the
rectangular spectral solvers re-exported from spectraldiffx and, on masked
basins, with the capacitance method or CG on :func:`masked_laplacian`. The
same operator, written as a gaussx linear operator, also gives its
log-determinant, square root, inverse and exact masked solves through
gaussx's structured primitives (``gaussx.solve``, ``gaussx.logdet``,
``gaussx.sqrt``, ``gaussx.inv``), without materialising it.  (Not
``gaussx.diag_inv``: for a ``DiagonalizedOperator`` gaussx 0.6.5 builds the
dense ``(Ny·Nx)²`` basis; use the separable 1-D bases instead, as
``docs/notebooks/spde_prior.py`` does.) That is
what a Gaussian-process or data-assimilation prior on the model grid needs:
an SPDE (Matérn-type) precision ``Q = τ² (κ² − ∇²)^α`` built from the model's
own discrete Laplacian and land mask.

``helmholtz_operator``
    The rectangular operator as a ``gaussx.DiagonalizedOperator``: the same
    transform (FFT, orthonormal DST-I or DCT-II) and the same FD2 eigenvalues
    as ``solve_helmholtz_fft`` / ``_dst`` / ``_dct``, so ``gaussx.solve`` on it
    reproduces those solvers.

``masked_helmholtz_operator``
    The masked operator applied by :func:`masked_laplacian`, as a
    ``gaussx.MaskedOperator`` over the periodic rectangular operator, with
    the coupling ring precomputed so ``gaussx.solve`` uses the capacitance
    method (two FFT solves per right-hand side).

Both act on **flat** vectors (row-major), as every lineax operator does:
``op.mv(psi.ravel())``. The masked operator acts on the wet values only,
``psi.ravel()[mask.ravel()]``.
"""

from __future__ import annotations

import numbers

import gaussx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
import lineax as lx
import numpy as np

from finitevolx._src.mask import Mask2D
from finitevolx._src.solvers.spectral import (
    dct2_eigenvalues,
    dst1_eigenvalues,
    fft_eigenvalues,
)
from finitevolx._src.solvers.spectral_transforms import dctn, dstn, idctn, idstn

# ---------------------------------------------------------------------------
# Orthonormal transform pairs (module-level so gaussx's static fields hash
# stably and jit does not recompile per operator instance)
# ---------------------------------------------------------------------------


def _dst1_forward(x: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
    # orthonormal DST-I along both axes, [Ny, Nx] -> [Ny, Nx]
    return dstn(x, type=1, axes=[0, 1], norm="ortho")


def _dst1_inverse(c: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
    return idstn(c, type=1, axes=[0, 1], norm="ortho")


def _dct2_forward(x: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
    # orthonormal DCT-II along both axes, [Ny, Nx] -> [Ny, Nx]
    return dctn(x, type=2, axes=[0, 1], norm="ortho")


def _dct2_inverse(c: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
    return idctn(c, type=2, axes=[0, 1], norm="ortho")


# 1-D FD2 eigenvalue function per rectangle BC (as the spectral solvers use)
_EIGENVALUES = {
    "fft": fft_eigenvalues,
    "dst": dst1_eigenvalues,
    "dct": dct2_eigenvalues,
}
# orthonormal (forward V, inverse V⁻¹) pairs; the FFT pair comes from
# gaussx.circulant_from_symbol
_ORTHO_PAIRS = {
    "dst": (_dst1_forward, _dst1_inverse),
    "dct": (_dct2_forward, _dct2_inverse),
}


def _sign_tags(lambda_: float | Array) -> frozenset[object]:
    """``negative_semidefinite_tag`` when ``λ`` is a concrete number ``≥ 0``."""
    if isinstance(lambda_, numbers.Real) and lambda_ >= 0:
        return frozenset({lx.negative_semidefinite_tag})
    return frozenset()


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def helmholtz_operator(
    shape: tuple[int, int],
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    bc: str = "fft",
) -> gaussx.DiagonalizedOperator:
    r"""The rectangular Helmholtz operator ``(∇² − λ)`` as a gaussx operator.

    The 5-point FD2 Laplacian on a rectangle with periodic (``"fft"``),
    homogeneous Dirichlet (``"dst"``, DST-I) or homogeneous Neumann
    (``"dct"``, DCT-II) boundaries is diagonalised by that transform:

    ``A = V⁻¹ Λ V``,  ``Λ[j, i] = Λʸ_j + Λˣ_i − λ``

    with the same eigenvalues (``fft_eigenvalues``, ``dst1_eigenvalues``,
    ``dct2_eigenvalues``) as the spectral solvers, so ``gaussx.solve(A, f)``
    equals ``solve_helmholtz_{bc}(f, dx, dy, λ)``. ``gaussx.logdet``,
    ``gaussx.sqrt``, ``gaussx.inv`` and ``gaussx.trace`` are elementwise in
    ``Λ`` (``gaussx.diag_inv`` is not: it densifies), and the algebra stays
    closed (``-A``, ``A - c I``, ``c A``), so a
    PSD ``κ² − ∇²`` is ``-helmholtz_operator(..., lambda_=κ²)``.

    Parameters
    ----------
    shape : (int, int)
        Field shape ``(Ny, Nx)``; the operator acts on flat vectors of length
        ``Ny·Nx`` (row-major), the layout the spectral solvers use.
    dx, dy : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter ``λ ≥ 0`` (contract 5). Default ``0`` (Poisson).
        A concrete ``λ ≥ 0`` tags the operator negative semidefinite.
    bc : {"fft", "dst", "dct"}
        Rectangle boundary condition, as for the spectral solvers.

    Notes
    -----
    **The Poisson null mode.** With ``λ = 0`` and ``bc="fft"`` or ``"dct"``
    the constant is in the null space: ``Λ[0, 0] = 0``. gaussx treats an
    exactly-zero eigenvalue as the pseudo-inverse (a zero coefficient), so
    ``gaussx.solve`` and ``gaussx.inv`` return the zero-mean solution — the
    same gauge as ``solve_helmholtz_fft`` / ``_dct`` — and stay finite, while
    ``gaussx.logdet`` is ``-inf`` (the ``slogdet`` convention for a singular
    operator). Use ``λ > 0`` (or ``bc="dst"``) for a log-determinant. A
    right-hand side with a non-zero mean has no solution; the pseudo-inverse
    silently drops that mean, exactly as the spectral solvers do.

    Returns
    -------
    gaussx.DiagonalizedOperator
        Symmetric, ``(Ny·Nx, Ny·Nx)``, never materialised.

    Examples
    --------
    >>> A = helmholtz_operator((Ny, Nx), dx, dy, lambda_=1.0, bc="dst")
    >>> psi = gaussx.solve(A, f.ravel()).reshape(Ny, Nx)  # = solve_helmholtz_dst
    >>> logdet = gaussx.logdet(-A)  # log det(1 - ∇²)
    """
    if bc not in _EIGENVALUES:
        raise ValueError(f"bc must be 'fft', 'dst', or 'dct'; got {bc!r}")
    ny, nx = (int(n) for n in shape)
    eig_1d = _EIGENVALUES[bc]
    # Λ[j, i] = Λʸ_j + Λˣ_i - λ, [Ny, Nx]
    eigenvalues = eig_1d(ny, dy)[:, None] + eig_1d(nx, dx)[None, :] - lambda_
    if bc == "fft":
        # A x = ifft2(Λ ⊙ fft2(x)); gaussx checks Λ is conjugate-even (real here)
        return gaussx.circulant_from_symbol(eigenvalues, tags=_sign_tags(lambda_))
    forward, inverse = _ORTHO_PAIRS[bc]
    return gaussx.DiagonalizedOperator(
        eigenvalues,
        forward,
        inverse,
        (ny, nx),
        normal=True,
        tags=_sign_tags(lambda_),
    )


def masked_helmholtz_operator(
    mask: Float[Array, "Ny Nx"] | Bool[Array, "Ny Nx"] | Mask2D,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
) -> gaussx.MaskedOperator:
    r"""The masked Helmholtz operator of :func:`masked_laplacian` as a gaussx operator.

    ``masked_laplacian`` applies the periodic 5-point ``(∇² − λ)`` to ``ψ``
    zeroed outside the mask (homogeneous Dirichlet on the coast) and keeps
    the wet rows. As a matrix that is the wet sub-block ``A[m][:, m]`` of the
    periodic rectangular operator ``A`` (:func:`helmholtz_operator` with
    ``bc="fft"``). This returns that sub-block as a ``gaussx.MaskedOperator``
    acting on the wet values only, with the coupling ring (the dry cells the
    stencil reaches) precomputed, so ``gaussx.solve`` is the capacitance
    method: exact, two FFT solves per right-hand side.

    The mask is concrete and the construction runs on the host (it builds
    the ``|C| × |C|`` capacitance factorisation); build it once per geometry
    and reuse it, like ``build_capacitance_solver``.

    Parameters
    ----------
    mask : array of shape (Ny, Nx), or Mask2D
        Wet cells (``True`` / ``1``). A :class:`Mask2D` contributes its
        ``xy_corner_strict`` (ψ-point) mask, as in ``masked_laplacian``.
    dx, dy : float
        Grid spacings.
    lambda_ : float
        Helmholtz parameter ``λ ≥ 0``. Default ``0`` (Poisson). With
        ``λ = 0`` the periodic base operator is singular (the constant mode),
        but the masked block is not as long as the mask has dry cells: gaussx
        derives the base's null vector and borders the capacitance system
        with it, so the masked solve is still exact.

    Returns
    -------
    gaussx.MaskedOperator
        ``(n_wet, n_wet)`` with ``n_wet = mask.sum()``; vectors are
        ``psi.ravel()[mask.ravel()]``.

    Examples
    --------
    >>> A_m = masked_helmholtz_operator(mask, dx, dy, lambda_=1.0)
    >>> wet = np.asarray(mask, bool).ravel()
    >>> psi_wet = gaussx.solve(A_m, f.ravel()[wet])
    >>> psi = jnp.zeros(mask.size).at[wet].set(psi_wet).reshape(mask.shape)
    """
    if isinstance(mask, Mask2D):
        mask = mask.xy_corner_strict
    wet = np.asarray(mask).astype(bool)  # [Ny, Nx], concrete
    base = helmholtz_operator(wet.shape, dx, dy, lambda_=lambda_, bc="fft")
    flat = jnp.asarray(wet.ravel())  # [Ny*Nx]
    coupling = gaussx.grid_coupling_indices(jnp.asarray(wet), periodic=True)
    return gaussx.MaskedOperator(
        base,
        flat,
        flat,
        coupling_indices=coupling,
        tags=_sign_tags(lambda_) | {lx.symmetric_tag},
    )
