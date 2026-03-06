"""Spectral Poisson and Helmholtz solvers for rectangular 2-D domains.

Solves the equation

    (∇² − λ) ψ = f

where ``∇² = ∂²/∂x² + ∂²/∂y²`` is the discrete 5-point Laplacian, using
diagonalization via spectral transforms.  Three boundary-condition families
are supported:

* **Dirichlet** (``solve_poisson_dst`` / ``solve_helmholtz_dst``):
  ψ = 0 on all four edges.  Uses DST-I in both directions.
  The input ``rhs`` should contain only the *interior* grid values
  (ghost-cells excluded).

* **Neumann** (``solve_poisson_dct`` / ``solve_helmholtz_dct``):
  ∂ψ/∂n = 0 on all four edges.  Uses DCT-II in both directions.
  The Poisson problem (λ = 0) has a one-dimensional null space (constant
  solutions); the gauge is fixed by setting the domain-mean of ψ to zero.

* **Periodic** (``solve_poisson_fft`` / ``solve_helmholtz_fft``):
  Periodic in both directions.  Uses the 2-D FFT.
  The Poisson problem (λ = 0) has a one-dimensional null space; the gauge
  is fixed by setting the domain-mean of ψ to zero.

Eigenvalue helpers
------------------
The spectral decomposition exploits that the discrete Laplacian is
diagonalized by the transform basis.  The 1-D eigenvalues are:

* DST-I  (N interior pts, spacing dx):
    λ_k = −4/dx² · sin²(π(k+1) / (2(N+1)))   k = 0, …, N−1

* DCT-II (N pts, spacing dx):
    λ_k = −4/dx² · sin²(πk / (2N))             k = 0, …, N−1
    (λ_0 = 0: null mode for Neumann Poisson)

* FFT    (N pts, spacing dx):
    λ_k = −4/dx² · sin²(πk / N)               k = 0, …, N−1
    (λ_0 = 0: null mode for periodic Poisson)

For 2-D, the full eigenvalue matrix is the outer sum:
    Λ[j, i] = λ_j^y + λ_i^x

Usage example
-------------
>>> import jax.numpy as jnp
>>> from finitevolx._src.elliptic import solve_poisson_dst
>>> Ny, Nx = 10, 12
>>> dx, dy = 1.0 / (Nx + 1), 1.0 / (Ny + 1)
>>> # DST-I eigenfunction (exact solution)
>>> i = jnp.arange(Ny)[:, None]; j = jnp.arange(Nx)[None, :]
>>> psi_exact = jnp.sin(jnp.pi * (i + 1) / (Ny + 1)) * jnp.sin(jnp.pi * (j + 1) / (Nx + 1))
>>> lam_x = -4 / dx**2 * jnp.sin(jnp.pi / (2 * (Nx + 1)))**2
>>> lam_y = -4 / dy**2 * jnp.sin(jnp.pi / (2 * (Ny + 1)))**2
>>> rhs = (lam_x + lam_y) * psi_exact
>>> psi = solve_poisson_dst(rhs, dx, dy)
>>> bool(jnp.allclose(psi, psi_exact, atol=1e-5))
True
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

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
    >>> i = jnp.arange(Ny)[:, None]; j = jnp.arange(Nx)[None, :]
    >>> psi = jnp.sin(jnp.pi*(i+1)/(Ny+1)) * jnp.sin(jnp.pi*(j+1)/(Nx+1))
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
