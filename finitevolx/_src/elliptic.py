"""Spectral, capacitance, and iterative solvers for 2-D elliptic PDEs.

Solves the equation

    (∇² − λ) ψ = f

where ``∇² = ∂²/∂x² + ∂²/∂y²`` is the discrete 5-point Laplacian.

Four solver families are provided:

Spectral solvers (rectangular domains)
---------------------------------------
* **Dirichlet** (``solve_poisson_dst`` / ``solve_helmholtz_dst``):
  ψ = 0 on all four edges.  Uses DST-I in both directions.

* **Neumann** (``solve_poisson_dct`` / ``solve_helmholtz_dct``):
  ∂ψ/∂n = 0 on all four edges.  Uses DCT-II in both directions.
  Poisson null space (λ=0) handled by zero-mean gauge.

* **Periodic** (``solve_poisson_fft`` / ``solve_helmholtz_fft``):
  Periodic in both directions.  Uses the 2-D FFT.
  Poisson null space (λ=0) handled by zero-mean gauge.

Capacitance matrix method (irregular/masked domains)
-----------------------------------------------------
Extends the fast spectral solver to domains that are subsets of a
rectangle (e.g. ocean basins with land masks) using the classic
Sherman-Morrison correction via boundary Green's functions.

``build_capacitance_solver`` performs a one-time offline precomputation
(N_b rectangular solves, where N_b = number of irregular-boundary points).
The returned ``CapacitanceSolver`` callable is then cheap to evaluate for
any right-hand side.

Preconditioned Conjugate Gradient (CG)
---------------------------------------
``solve_cg`` solves A·ψ = f for any symmetric operator A via the
Preconditioned Conjugate Gradient algorithm.  It runs inside JAX's
``lax.while_loop`` so the entire iteration is JIT-compilable.

``make_spectral_preconditioner`` returns a preconditioner that applies
the rectangular spectral solve; this is an effective preconditioner for
masked-domain problems.

Eigenvalue helpers
------------------
* DST-I  (N interior pts, spacing dx):
    λ_k = −4/dx² · sin²(π(k+1) / (2(N+1)))   k = 0, …, N−1

* DCT-II (N pts, spacing dx):
    λ_k = −4/dx² · sin²(πk / (2N))             k = 0, …, N−1

* FFT    (N pts, spacing dx):
    λ_k = −4/dx² · sin²(πk / N)               k = 0, …, N−1

Usage example
-------------
>>> import jax.numpy as jnp
>>> from finitevolx._src.elliptic import solve_poisson_dst
>>> Ny, Nx = 10, 12
>>> dx, dy = 1.0 / (Nx + 1), 1.0 / (Ny + 1)
>>> # DST-I eigenfunction (exact solution)
>>> i = jnp.arange(Ny)[:, None]
... j = jnp.arange(Nx)[None, :]
>>> psi_exact = jnp.sin(jnp.pi * (i + 1) / (Ny + 1)) * jnp.sin(
...     jnp.pi * (j + 1) / (Nx + 1)
... )
>>> lam_x = -4 / dx**2 * jnp.sin(jnp.pi / (2 * (Nx + 1))) ** 2
>>> lam_y = -4 / dy**2 * jnp.sin(jnp.pi / (2 * (Ny + 1))) ** 2
>>> rhs = (lam_x + lam_y) * psi_exact
>>> psi = solve_poisson_dst(rhs, dx, dy)
>>> bool(jnp.allclose(psi, psi_exact, atol=1e-5))
True
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np

from finitevolx._src.spectral_transforms import dctn, dstn, idctn, idstn

# ---------------------------------------------------------------------------
# Eigenvalue helpers
# ---------------------------------------------------------------------------


def dst1_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for DST-I (Dirichlet BCs).

    For a second-order finite-difference Laplacian on *N* interior points
    with homogeneous Dirichlet boundary conditions and grid spacing *dx*:

        λ_k = −4/dx² · sin²(π(k+1) / (2(N+1)))   k = 0, …, N−1

    Parameters
    ----------
    N : int
        Number of interior grid points.
    dx : float
        Grid spacing.

    Returns
    -------
    Float[Array, " N"]
        Array of N eigenvalues, all ≤ 0.
    """
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * (k + 1) / (2 * (N + 1))) ** 2


def dct2_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for DCT-II (Neumann BCs).

    For a second-order finite-difference Laplacian on *N* points with
    homogeneous Neumann boundary conditions and grid spacing *dx*:

        λ_k = −4/dx² · sin²(πk / (2N))   k = 0, …, N−1

    Note: λ_0 = 0 (null mode / constant solution) for the Poisson problem.

    Parameters
    ----------
    N : int
        Number of grid points.
    dx : float
        Grid spacing.

    Returns
    -------
    Float[Array, " N"]
        Array of N eigenvalues, all ≤ 0.
    """
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * k / (2 * N)) ** 2


def fft_eigenvalues(N: int, dx: float) -> Float[Array, " N"]:
    """1-D Laplacian eigenvalues for the FFT (periodic BCs).

    For a second-order finite-difference Laplacian on *N* points with
    periodic boundary conditions and grid spacing *dx*:

        λ_k = −4/dx² · sin²(πk / N)   k = 0, …, N−1

    Note: λ_0 = 0 (null mode / constant solution) for the Poisson problem.

    Parameters
    ----------
    N : int
        Number of grid points.
    dx : float
        Grid spacing.

    Returns
    -------
    Float[Array, " N"]
        Array of N eigenvalues, all ≤ 0.
    """
    k = jnp.arange(N)
    return -4.0 / dx**2 * jnp.sin(jnp.pi * k / N) ** 2


# ---------------------------------------------------------------------------
# DST-I: Dirichlet spectral solvers
# ---------------------------------------------------------------------------


def solve_helmholtz_dst(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
) -> Float[Array, "Ny Nx"]:
    """Solve (∇² − λ)ψ = f with homogeneous Dirichlet BCs using DST-I.

    The input *rhs* contains values at the **interior** grid points only
    (boundary values are implicitly zero).  Grid spacings *dx* and *dy*
    correspond to the same interior grid.

    The spectral solve:

    1. Forward DST-I in x, then y:  F̂ = DST-I_y(DST-I_x(f))
    2. Divide by spectral operator:  Ψ̂[j,i] = F̂[j,i] / (λ_j^y + λ_i^x − λ)
    3. Inverse DST-I in y, then x:  ψ = IDST-I_x(IDST-I_y(Ψ̂))

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side of the PDE at interior grid points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  For the Poisson equation use ``lambda_=0.0``.
        Must satisfy λ < λ_min (smallest eigenvalue) to ensure invertibility.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ at interior grid points, same shape as *rhs*.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx._src.elliptic import solve_helmholtz_dst, dst1_eigenvalues
    >>> Ny, Nx = 8, 10
    >>> dx, dy = 0.1, 0.1
    >>> lam_x = dst1_eigenvalues(Nx, dx)[0]
    >>> lam_y = dst1_eigenvalues(Ny, dy)[0]
    >>> i = jnp.arange(Ny)[:, None]
    ... j = jnp.arange(Nx)[None, :]
    >>> psi = jnp.sin(jnp.pi * (i + 1) / (Ny + 1)) * jnp.sin(
    ...     jnp.pi * (j + 1) / (Nx + 1)
    ... )
    >>> lam_helm = -0.5
    >>> rhs = (lam_x + lam_y - lam_helm) * psi
    >>> out = solve_helmholtz_dst(rhs, dx, dy, lambda_=lam_helm)
    >>> bool(jnp.allclose(out, psi, atol=1e-5))
    True
    """
    Ny, Nx = rhs.shape
    # Forward 2-D DST-I
    rhs_hat = dstn(rhs, type=1, axes=[0, 1])
    # 2-D eigenvalue matrix
    eigx = dst1_eigenvalues(Nx, dx)  # [Nx]
    eigy = dst1_eigenvalues(Ny, dy)  # [Ny]
    eig2d = eigy[:, None] + eigx[None, :] - lambda_  # [Ny, Nx]
    # Spectral division
    psi_hat = rhs_hat / eig2d
    # Inverse 2-D DST-I  (IDST-I = DST-I / (2*(N+1)) for each axis)
    return idstn(psi_hat, type=1, axes=[0, 1])


def solve_poisson_dst(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²ψ = f with homogeneous Dirichlet BCs using DST-I.

    Convenience wrapper around :func:`solve_helmholtz_dst` with ``lambda_=0``.

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side at interior grid points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    return solve_helmholtz_dst(rhs, dx, dy, lambda_=0.0)


# ---------------------------------------------------------------------------
# DCT-II: Neumann spectral solvers
# ---------------------------------------------------------------------------


def solve_helmholtz_dct(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
) -> Float[Array, "Ny Nx"]:
    """Solve (∇² − λ)ψ = f with homogeneous Neumann BCs using DCT-II.

    Uses DCT-II (type 2) for the spectral decomposition associated with
    ∂ψ/∂n = 0 boundary conditions.

    For the Poisson problem (λ = 0) the system is singular (the (0,0) mode
    has eigenvalue 0). In that case callers should use
    :func:`solve_poisson_dct`, which enforces a zero-mean gauge.

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Must satisfy λ ≠ any Neumann eigenvalue.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ, same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    # Forward 2-D DCT-II
    rhs_hat = dctn(rhs, type=2, axes=[0, 1])
    # 2-D eigenvalue matrix
    eigx = dct2_eigenvalues(Nx, dx)
    eigy = dct2_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :] - lambda_
    # Spectral division
    psi_hat = rhs_hat / eig2d
    # Inverse 2-D DCT-II
    return idctn(psi_hat, type=2, axes=[0, 1])


def solve_poisson_dct(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²ψ = f with homogeneous Neumann BCs using DCT-II.

    The Poisson problem has a one-dimensional null space (constant solutions).
    This function fixes the gauge by forcing the domain-mean of ψ to zero
    (i.e., the (0,0) spectral coefficient is set to zero).

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side.  For exact solvability the mean of *rhs* should
        be zero; a non-zero mean is handled by projecting it out.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Zero-mean solution ψ, same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    rhs_hat = dctn(rhs, type=2, axes=[0, 1])
    eigx = dct2_eigenvalues(Nx, dx)
    eigy = dct2_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :]  # [Ny, Nx]
    # Avoid division by zero at the (0,0) null mode; set that coefficient to 0
    # (enforces zero-mean solution)
    eig2d_safe = jnp.where(eig2d == 0.0, 1.0, eig2d)
    psi_hat = rhs_hat / eig2d_safe
    psi_hat = psi_hat.at[0, 0].set(0.0)
    return idctn(psi_hat, type=2, axes=[0, 1])


# ---------------------------------------------------------------------------
# FFT: periodic spectral solvers
# ---------------------------------------------------------------------------


def solve_helmholtz_fft(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
) -> Float[Array, "Ny Nx"]:
    """Solve (∇² − λ)ψ = f with periodic BCs using the 2-D FFT.

    For the Poisson problem (λ = 0) the system is singular (the (0,0) mode
    has eigenvalue 0). In that case callers should use
    :func:`solve_poisson_fft`, which enforces a zero-mean gauge.

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side on the periodic domain.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Must satisfy λ ≠ any FFT eigenvalue.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Solution ψ (real), same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    rhs_hat = jnp.fft.fft2(rhs)
    eigx = fft_eigenvalues(Nx, dx)
    eigy = fft_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :] - lambda_
    psi_hat = rhs_hat / eig2d
    return jnp.real(jnp.fft.ifft2(psi_hat))


def solve_poisson_fft(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Solve ∇²ψ = f with periodic BCs using the 2-D FFT.

    The Poisson problem has a one-dimensional null space.  This function
    fixes the gauge by forcing the domain-mean of ψ to zero (the (0,0)
    Fourier coefficient is set to zero).

    Parameters
    ----------
    rhs : Float[Array, "Ny Nx"]
        Right-hand side on the periodic domain.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Zero-mean solution ψ (real), same shape as *rhs*.
    """
    Ny, Nx = rhs.shape
    rhs_hat = jnp.fft.fft2(rhs)
    eigx = fft_eigenvalues(Nx, dx)
    eigy = fft_eigenvalues(Ny, dy)
    eig2d = eigy[:, None] + eigx[None, :]
    # Avoid division by zero at the (0,0) null mode
    eig2d_safe = jnp.where(eig2d == 0.0, 1.0, eig2d)
    psi_hat = rhs_hat / eig2d_safe
    psi_hat = psi_hat.at[0, 0].set(0.0)
    return jnp.real(jnp.fft.ifft2(psi_hat))


# ---------------------------------------------------------------------------
# Preconditioned Conjugate Gradient (PCG) solver
# ---------------------------------------------------------------------------


class CGInfo(NamedTuple):
    """Convergence diagnostics returned by :func:`solve_cg`."""

    iterations: int
    """Number of PCG iterations performed."""
    residual_norm: float
    """L2 norm of the final residual."""
    converged: bool
    """True if the solver converged within *tol*."""


def solve_cg(
    matvec: Callable[[Float[Array, ...]], Float[Array, ...]],
    rhs: Float[Array, ...],
    x0: Float[Array, ...] | None = None,
    preconditioner: Callable[[Float[Array, ...]], Float[Array, ...]] | None = None,
    tol: float = 1e-6,
    maxiter: int = 500,
) -> tuple[Float[Array, ...], CGInfo]:
    """Preconditioned Conjugate Gradient solver for symmetric linear operators.

    Solves ``A(x) = rhs`` where ``matvec`` implements the symmetric linear
    operator ``A``.  The operator need not be explicitly formed; only
    matrix-vector products are required.

    The iteration is implemented via ``jax.lax.while_loop``, making the
    entire solve JIT-compilable and differentiable through unrolling.

    Algorithm (standard PCG):

    .. code-block:: text

        r = rhs − A(x₀)
        z = M⁻¹(r),  p = z,  ρ = ⟨r, z⟩
        for k = 0, 1, …:
            Ap = A(p)
            α  = ρ / ⟨p, Ap⟩
            x  = x + α p
            r  = r − α Ap
            z  = M⁻¹(r)
            ρ′ = ⟨r, z⟩
            β  = ρ′ / ρ
            p  = z + β p
            ρ  = ρ′
            if ‖r‖₂ < tol: break

    Parameters
    ----------
    matvec : callable
        Function implementing the symmetric linear operator ``A``.
        Signature: ``matvec(x: Array) -> Array``.  The input and output
        arrays must have the same shape as *rhs*.
    rhs : Float[Array, "..."]
        Right-hand side of the linear system.
    x0 : Float[Array, "..."] or None
        Initial guess.  Defaults to zeros with the same shape as *rhs*.
    preconditioner : callable or None
        Approximate inverse of ``A`` used to accelerate convergence.
        Signature: ``preconditioner(r: Array) -> Array``.
        Defaults to the identity (i.e., no preconditioning).
    tol : float
        Convergence tolerance.  Iteration stops when ``‖r‖₂ < tol``.
        Default: 1e-6.
    maxiter : int
        Maximum number of iterations.  Default: 500.

    Returns
    -------
    x : Float[Array, "..."]
        Approximate solution, same shape as *rhs*.
    info : CGInfo
        Named tuple with fields ``iterations``, ``residual_norm``, and
        ``converged``.

    Examples
    --------
    Solve the 2-D Poisson equation on a periodic domain as a sanity-check:

    >>> import jax.numpy as jnp
    >>> from finitevolx._src.elliptic import solve_cg, solve_poisson_fft
    >>> Ny, Nx = 8, 10
    >>> dx, dy = 2 * jnp.pi / Nx, 2 * jnp.pi / Ny
    >>> lam = -1.0
    >>> i = jnp.arange(Nx)[None, :]
    ... j = jnp.arange(Ny)[:, None]
    >>> psi_ref = jnp.cos(2 * jnp.pi * i / Nx) + jnp.cos(2 * jnp.pi * j / Ny)
    >>> from finitevolx._src.elliptic import fft_eigenvalues
    >>> rhs = (
    ...     fft_eigenvalues(Nx, float(dx))[1]
    ...     * jnp.cos(2 * jnp.pi * i / Nx)
    ...     * jnp.ones((Ny, 1))
    ...     + fft_eigenvalues(Ny, float(dy))[1]
    ...     * jnp.cos(2 * jnp.pi * j / Ny)
    ...     * jnp.ones((1, Nx))
    ...     - lam * psi_ref
    ... )
    >>> def A(x):
    ...     return jnp.real(
    ...         jnp.fft.ifft2(
    ...             (
    ...                 fft_eigenvalues(Nx, float(dx))[None, :]
    ...                 + fft_eigenvalues(Ny, float(dy))[:, None]
    ...                 - lam
    ...             )
    ...             * jnp.fft.fft2(x)
    ...         )
    ...     )
    >>> psi, info = solve_cg(A, rhs, tol=1e-8)
    >>> info.converged
    True
    """
    if x0 is None:
        x0 = jnp.zeros_like(rhs)
    if preconditioner is None:
        preconditioner = lambda z: z

    def _inner(a: Array, b: Array) -> Array:
        return jnp.sum(a * b)

    r0 = rhs - matvec(x0)
    z0 = preconditioner(r0)
    p0 = z0
    rz0 = _inner(r0, z0)

    # State: (x, r, p, z, rz, k)
    init_state = (x0, r0, p0, z0, rz0, jnp.int32(0))

    def cond_fn(state: tuple) -> Array:
        _x, r, _p, _z, _rz, k = state
        return (k < maxiter) & (jnp.sqrt(_inner(r, r)) > tol)

    def body_fn(state: tuple) -> tuple:
        x, r, p, _z, rz, k = state
        Ap = matvec(p)
        pAp = _inner(p, Ap)
        alpha = rz / pAp
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = preconditioner(r_new)
        rz_new = _inner(r_new, z_new)
        beta = rz_new / rz
        p_new = z_new + beta * p
        return (x_new, r_new, p_new, z_new, rz_new, k + 1)

    x_out, r_out, _p, _z, _rz, k_out = jax.lax.while_loop(cond_fn, body_fn, init_state)
    res_norm = jnp.sqrt(_inner(r_out, r_out))
    info = CGInfo(
        iterations=int(k_out),
        residual_norm=float(res_norm),
        converged=bool(res_norm <= tol),
    )
    return x_out, info


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

    def _preconditioner(r: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return _spectral_solve(r, dx, dy, lambda_, bc)

    return _preconditioner


def masked_laplacian(
    psi: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"],
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
    mask : Float[Array, "Ny Nx"]
        Binary mask: 1 inside the physical domain, 0 outside (land/exterior).
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
    psi_m = psi * mask  # enforce zero outside domain
    # 5-point finite-difference stencil with periodic roll at edges
    lap = (
        jnp.roll(psi_m, 1, axis=1) + jnp.roll(psi_m, -1, axis=1) - 2.0 * psi_m
    ) / dx**2 + (
        jnp.roll(psi_m, 1, axis=0) + jnp.roll(psi_m, -1, axis=0) - 2.0 * psi_m
    ) / dy**2
    return (lap - lambda_ * psi_m) * mask


# ---------------------------------------------------------------------------
# Capacitance matrix solver (irregular / masked domains)
# ---------------------------------------------------------------------------


class CapacitanceSolver(eqx.Module):
    """Spectral Poisson/Helmholtz solver for masked irregular domains.

    Uses the **capacitance matrix method** (Buzbee, Golub & Nielson 1970) to
    extend a fast rectangular spectral solver to a domain defined by a binary
    mask.

    The algorithm relies on two observations:

    1. The irregular-domain solution ``ψ`` equals the rectangular spectral
       solution ``u`` minus a correction ``Σ_k α_k g_k``, where ``g_k`` are
       Green's functions (response to unit sources at each inner-boundary
       point ``b_k``).

    2. Requiring ``ψ(b_k) = 0`` at every inner-boundary point yields the
       linear system ``C α = u[B]``, where ``C[k,l] = g_l(b_k)`` is the
       **capacitance matrix**.

    Construct with :func:`build_capacitance_solver` (offline, runs *N_b*
    spectral solves where *N_b* = number of inner-boundary points).

    Attributes
    ----------
    _C_inv : Float[Array, "Nb Nb"]
        Pre-inverted capacitance matrix.
    _green_flat : Float[Array, "Nb NyNx"]
        Green's functions (one row per boundary point), stored flat.
    _j_b : Int[Array, "Nb"]
        Row indices of inner-boundary points.
    _i_b : Int[Array, "Nb"]
        Column indices of inner-boundary points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter.
    base_bc : str
        Spectral solver used as the rectangular base (``"fft"``, ``"dst"``,
        or ``"dct"``).

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> from finitevolx._src.elliptic import build_capacitance_solver
    >>> # small 8×10 ocean basin with a 2-cell land border
    >>> mask = np.ones((8, 10), dtype=bool)
    >>> mask[:, 0] = mask[:, -1] = mask[0, :] = mask[-1, :] = False
    >>> solver = build_capacitance_solver(mask, dx=0.1, dy=0.1)
    >>> rhs = jnp.zeros((8, 10))
    >>> psi = solver(rhs)  # masked Poisson solve
    >>> psi.shape
    (8, 10)
    """

    _C_inv: Float[Array, "Nb Nb"]
    _green_flat: Float[Array, "Nb NyNx"]
    _j_b: Array
    _i_b: Array
    dx: float
    dy: float
    lambda_: float
    base_bc: str

    def __call__(
        self,
        rhs: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Solve (∇² − λ)ψ = rhs on the masked domain.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Right-hand side, defined on the full rectangular grid.
            Values outside the mask are ignored.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Solution ψ, satisfying ψ = 0 at all inner-boundary points
            and (approximately) (∇² − λ)ψ = rhs at interior points.
        """
        Ny, Nx = rhs.shape
        # Step 1: rectangular spectral solve
        u = _spectral_solve(rhs, self.dx, self.dy, self.lambda_, self.base_bc)
        # Step 2: values of u at inner-boundary points
        u_b = u[self._j_b, self._i_b]  # [Nb]
        # Step 3: correction coefficients  alpha = C^{-1} u_b
        alpha = self._C_inv @ u_b  # [Nb]
        # Step 4: correction field  sum_k alpha_k g_k
        correction = (self._green_flat.T @ alpha).reshape(Ny, Nx)  # [Ny, Nx]
        return u - correction


def _spectral_solve(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
    bc: str,
) -> Float[Array, "Ny Nx"]:
    """Dispatch to the appropriate rectangular spectral solver."""
    if bc == "fft":
        return (
            solve_poisson_fft(rhs, dx, dy)
            if lambda_ == 0.0
            else solve_helmholtz_fft(rhs, dx, dy, lambda_)
        )
    if bc == "dst":
        return (
            solve_poisson_dst(rhs, dx, dy)
            if lambda_ == 0.0
            else solve_helmholtz_dst(rhs, dx, dy, lambda_)
        )
    if bc == "dct":
        return (
            solve_poisson_dct(rhs, dx, dy)
            if lambda_ == 0.0
            else solve_helmholtz_dct(rhs, dx, dy, lambda_)
        )
    raise ValueError(f"base_bc must be 'fft', 'dst', or 'dct'; got {bc!r}")


def build_capacitance_solver(
    mask: np.ndarray,
    dx: float,
    dy: float,
    lambda_: float = 0.0,
    base_bc: str = "fft",
) -> CapacitanceSolver:
    """Pre-compute the capacitance matrix and return a ready-to-use solver.

    This is an **offline** function that performs *N_b* rectangular spectral
    solves (``N_b`` = number of inner-boundary points).  The result is a
    :class:`CapacitanceSolver` whose ``__call__`` method is JIT-compilable.

    Algorithm (Buzbee, Golub & Nielson 1970):

    1. Find inner-boundary points ``B`` = mask points adjacent to exterior.
    2. For each ``b_k ∈ B``, solve ``L_rect g_k = e_{b_k}`` (Green's function).
    3. Build ``C[k, l] = g_l(b_k)``  and invert to ``C⁻¹``.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (Ny, Nx)
        Physical domain mask.  ``True`` = interior (ocean/fluid),
        ``False`` = exterior (land/walls).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter λ.  Use ``0.0`` for pure Poisson.
    base_bc : {"fft", "dst", "dct"}
        Rectangular spectral solver used as the base.  ``"fft"`` (periodic)
        is a good default; ``"dst"`` (Dirichlet) handles rectangle-boundary
        conditions directly.

    Returns
    -------
    CapacitanceSolver
        A callable equinox Module with all precomputed arrays baked in.

    Notes
    -----
    Memory cost: ``O(N_b × Ny × Nx)`` for the Green's function matrix.
    Time cost (offline): ``O(N_b × Ny × Nx × log(Ny × Nx))``.
    Time cost (online): ``O(N_b² + Ny × Nx × log(Ny × Nx))``.

    Raises
    ------
    ValueError
        If the mask has no inner-boundary points (e.g. all-ones mask).
    """
    from scipy.ndimage import binary_dilation  # local import (offline only)

    mask = np.asarray(mask, dtype=bool)
    Ny, Nx = mask.shape

    # Inner-boundary: mask-interior cells adjacent to at least one exterior cell
    exterior = ~mask
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    dilated = binary_dilation(exterior, structure=struct)
    inner_boundary = mask & dilated  # [Ny, Nx] bool

    j_b, i_b = np.where(inner_boundary)  # row/col indices of boundary points
    N_b = len(j_b)
    if N_b == 0:
        raise ValueError(
            "No inner-boundary points found.  Check that the mask has a "
            "non-trivial interior/exterior structure."
        )

    # Helper: one rectangular spectral solve (numpy interface)
    def _base_solve_np(f_2d: np.ndarray) -> np.ndarray:
        f_jax = jnp.array(f_2d, dtype=float)
        result = _spectral_solve(f_jax, dx, dy, lambda_, base_bc)
        return np.array(result)

    # Green's functions: G[k] = solution to L_rect g_k = e_{b_k}
    # Shape: [N_b, Ny, Nx]
    green = np.zeros((N_b, Ny, Nx), dtype=float)
    for k in range(N_b):
        e_k = np.zeros((Ny, Nx), dtype=float)
        e_k[j_b[k], i_b[k]] = 1.0
        green[k] = _base_solve_np(e_k)

    # Capacitance matrix C[k, l] = green[l] evaluated at boundary point b_k
    # green[:, j_b, i_b] has shape [N_b, N_b] with element [l, k] = green[l, b_k]
    # We need C[k, l], so transpose.
    C = green[:, j_b, i_b].T  # [N_b, N_b]
    C_inv = np.linalg.inv(C)

    return CapacitanceSolver(
        _C_inv=jnp.array(C_inv),
        _green_flat=jnp.array(green.reshape(N_b, Ny * Nx)),
        _j_b=jnp.array(j_b),
        _i_b=jnp.array(i_b),
        dx=float(dx),
        dy=float(dy),
        lambda_=float(lambda_),
        base_bc=base_bc,
    )
