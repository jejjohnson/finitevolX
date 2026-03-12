"""
Diagnostic quantities on Arakawa C-grids.

Includes kinetic / potential energy, strain, enstrophy, Okubo-Weiss,
and domain-integrated conservation diagnostics.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
)

from finitevolx._src.grid.constants import GRAVITY
from finitevolx._src.operators.difference import _curl_2d

# ======================================================================
# Kinetic energy & Bernoulli potential  (existing)
# ======================================================================


def kinetic_energy(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Kinetic energy at T-points (cell centers) on an Arakawa C-grid.

    Eq:
        ke[j, i] = 0.5 * (u²_on_T[j, i] + v²_on_T[j, i])

    where u² and v² are averaged from face-points to T-points:
        u²_on_T[j, i] = 0.5 * (u[j, i+1/2]² + u[j, i-1/2]²)
                       = 0.5 * (u[j, i]² + u[j, i-1]²)
        v²_on_T[j, i] = 0.5 * (v[j+1/2, i]² + v[j-1/2, i]²)
                       = 0.5 * (v[j, i]² + v[j-1, i]²)

    Args:
        u (Array): x-velocity at U-points (east faces), shape [Ny, Nx].
        v (Array): y-velocity at V-points (north faces), shape [Ny, Nx].
        mask (Array | None): optional binary mask at T-points.  If provided,
            KE is zeroed where mask is 0.

    Returns:
        ke (Array): kinetic energy at T-points, shape [Ny, Nx].
            Ghost ring is zero; interior is [1:-1, 1:-1].
    """
    dtype = jnp.result_type(u, v, 0.0)
    u_float = jnp.asarray(u, dtype=dtype)
    v_float = jnp.asarray(v, dtype=dtype)
    u2 = u_float**2
    v2 = v_float**2
    out = jnp.zeros_like(u_float)
    # u²_on_T[j, i] = 0.5 * (u²[j, i] + u²[j, i-1])  (east + west U-faces)
    # v²_on_T[j, i] = 0.5 * (v²[j, i] + v²[j-1, i])  (north + south V-faces)
    u2_on_T = 0.5 * (u2[1:-1, 1:-1] + u2[1:-1, :-2])
    v2_on_T = 0.5 * (v2[1:-1, 1:-1] + v2[:-2, 1:-1])
    ke_int = 0.5 * (u2_on_T + v2_on_T)
    if mask is not None:
        ke_int = ke_int * mask[1:-1, 1:-1]
    out = out.at[1:-1, 1:-1].set(ke_int)
    return out


def bernoulli_potential(
    h: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    gravity: float = GRAVITY,
) -> Float[Array, "Ny Nx"]:
    """Bernoulli potential at T-points on an Arakawa C-grid.

    Eq:
        p[j, i] = ke[j, i] + g * h[j, i]

    where ke is the kinetic energy at T-points.

    Args:
        h (Array): layer thickness at T-points, shape [Ny, Nx].
        u (Array): x-velocity at U-points (east faces), shape [Ny, Nx].
        v (Array): y-velocity at V-points (north faces), shape [Ny, Nx].
        gravity (float): gravitational acceleration. Default = 9.81.

    Returns:
        p (Array): Bernoulli potential at T-points, shape [Ny, Nx].
            Ghost ring is zero; interior is [1:-1, 1:-1].

    Example:
        >>> u, v, h = ...
        >>> p = bernoulli_potential(h=h, u=u, v=v)
    """
    dtype = jnp.result_type(h, u, v, 0.0)
    h_float = jnp.asarray(h, dtype=dtype)
    ke = kinetic_energy(u=u, v=v)
    out = jnp.zeros_like(h_float)
    out = out.at[1:-1, 1:-1].set(ke[1:-1, 1:-1] + gravity * h_float[1:-1, 1:-1])
    return out


# ======================================================================
# Standalone vorticity functions  (Issue #73)
# ======================================================================


def relative_vorticity_cgrid(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Relative vorticity at X-points (corners) on an Arakawa C-grid.

    zeta[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                        - (u[j+1, i+1/2] - u[j, i+1/2]) / dy

    This is a standalone functional form of the class-based
    :meth:`Vorticity2D.relative_vorticity`.  Both share the same
    underlying implementation
    (:func:`finitevolx._src.operators.difference._curl_2d`).

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "Ny Nx"]
        y-velocity at V-points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Relative vorticity at X-points.  Ghost ring is zero.
    """
    return _curl_2d(u, v, dx, dy)


def potential_vorticity(
    omega: Float[Array, "Ny Nx"],
    f: Float[Array, "Ny Nx"],
    h: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """Potential vorticity (pointwise).

    q = (omega + f) / h

    All inputs must live on the same grid points (typically X-points
    after interpolating ``f`` and ``h`` from T-points).

    Parameters
    ----------
    omega : Float[Array, "Ny Nx"]
        Relative vorticity.
    f : Float[Array, "Ny Nx"]
        Coriolis parameter.
    h : Float[Array, "Ny Nx"]
        Layer thickness (or depth).

    Returns
    -------
    Float[Array, "Ny Nx"]
        Potential vorticity.  Where ``h == 0`` the result is NaN.
    """
    return jnp.where(h == 0, jnp.nan, (omega + f) / h)


# ======================================================================
# Strain operators  (Issue #2)
# ======================================================================


def shear_strain(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Shear strain at X-points (corners) on an Arakawa C-grid.

    Ss[j+1/2, i+1/2] = dv/dx + du/dy
        = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
        + (u[j+1, i+1/2] - u[j, i+1/2]) / dy

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "Ny Nx"]
        y-velocity at V-points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Shear strain at X-points.  Ghost ring is zero.
    """
    out = jnp.zeros_like(u)
    # dv/dx[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
    dv_dx = (v[1:-1, 2:] - v[1:-1, 1:-1]) / dx
    # du/dy[j+1/2, i+1/2] = (u[j+1, i+1/2] - u[j, i+1/2]) / dy
    du_dy = (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy
    out = out.at[1:-1, 1:-1].set(dv_dx + du_dy)
    return out


def tensor_strain(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Tensor (normal) strain at T-points on an Arakawa C-grid.

    Sn[j, i] = du/dx - dv/dy
        = (u[j, i+1/2] - u[j, i-1/2]) / dx
        - (v[j+1/2, i] - v[j-1/2, i]) / dy

    Parameters
    ----------
    u : Float[Array, "Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "Ny Nx"]
        y-velocity at V-points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Tensor strain at T-points.  Ghost ring is zero.
    """
    out = jnp.zeros_like(u)
    # du/dx[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx
    du_dx = (u[1:-1, 1:-1] - u[1:-1, :-2]) / dx
    # dv/dy[j, i] = (v[j+1/2, i] - v[j-1/2, i]) / dy
    dv_dy = (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dy
    out = out.at[1:-1, 1:-1].set(du_dx - dv_dy)
    return out


def strain_magnitude_squared(
    sn: Float[Array, "Ny Nx"],
    ss: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """Squared strain magnitude (pointwise).

    sigma^2 = Sn^2 + Ss^2

    Inputs must live on the same grid points.  If ``sn`` is at T-points
    and ``ss`` at X-points, interpolate one to the other before calling.

    Parameters
    ----------
    sn : Float[Array, "Ny Nx"]
        Tensor (normal) strain.
    ss : Float[Array, "Ny Nx"]
        Shear strain.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Squared strain magnitude.
    """
    return sn**2 + ss**2


def okubo_weiss(
    sn: Float[Array, "Ny Nx"],
    ss: Float[Array, "Ny Nx"],
    omega: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """Okubo-Weiss parameter (pointwise).

    OW = Sn^2 + Ss^2 - omega^2

    All inputs must live on the same grid points.

    Parameters
    ----------
    sn : Float[Array, "Ny Nx"]
        Tensor (normal) strain.
    ss : Float[Array, "Ny Nx"]
        Shear strain.
    omega : Float[Array, "Ny Nx"]
        Relative vorticity.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Okubo-Weiss parameter.  Positive = strain-dominated,
        negative = vorticity-dominated.
    """
    return sn**2 + ss**2 - omega**2


# ======================================================================
# Enstrophy  (Issues #2 and #73)
# ======================================================================


def enstrophy(
    omega: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Enstrophy (pointwise).

    Z = 0.5 * omega^2

    Parameters
    ----------
    omega : Float[Array, "Ny Nx"]
        Relative vorticity (typically at X-points).
    mask : Float[Array, "Ny Nx"] | None, optional
        Binary mask.  If provided, result is zeroed where mask is 0.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Enstrophy at the same grid points as *omega*.
    """
    out = 0.5 * omega**2
    if mask is not None:
        out = out * mask
    return out


def potential_enstrophy(
    q: Float[Array, "Ny Nx"],
    h: Float[Array, "Ny Nx"],
    mask: Float[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Potential enstrophy (pointwise).

    PE = 0.5 * q^2 * h

    Inputs must live on the same grid points.

    Parameters
    ----------
    q : Float[Array, "Ny Nx"]
        Potential vorticity.
    h : Float[Array, "Ny Nx"]
        Layer thickness (or depth).
    mask : Float[Array, "Ny Nx"] | None, optional
        Binary mask.  If provided, result is zeroed where mask is 0.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Potential enstrophy.
    """
    out = 0.5 * q**2 * h
    if mask is not None:
        out = out * mask
    return out


# ======================================================================
# Potential energy  (Issue #73)
# ======================================================================


def available_potential_energy(
    h: Float[Array, "Ny Nx"],
    H: Float[Array, "Ny Nx"],
    g_prime: float,
) -> Float[Array, "Ny Nx"]:
    """Available potential energy at T-points.

    APE = 0.5 * g' * (h - H)^2

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Layer thickness at T-points.
    H : Float[Array, "Ny Nx"]
        Mean (reference) layer thickness at T-points.
    g_prime : float
        Reduced gravity.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Available potential energy at T-points.
    """
    eta = h - H
    return 0.5 * g_prime * eta**2


# ======================================================================
# QG potential vorticity  (Issue #73)
# ======================================================================


def qg_potential_vorticity(
    psi: Float[Array, "Ny Nx"],
    f0: float,
    beta: float,
    dx: float,
    dy: float,
    y: Float[Array, "Ny Nx"],
    y0: float,
) -> Float[Array, "Ny Nx"]:
    """QG potential vorticity at T-points (single layer).

    q = nabla^2(psi) / f0 + beta * (y - y0) / f0

    This computes the per-layer (barotropic) QG PV for a single ``[Ny, Nx]``
    streamfunction.  For multi-layer QG, lift this function with
    :func:`~finitevolx.multilayer` and subtract the cross-layer stretching
    term separately::

        from finitevolx import multilayer, qg_potential_vorticity

        # Per-layer PV (vmapped over the layer axis)
        q = multilayer(lambda p: qg_potential_vorticity(p, f0, beta, dx, dy, y, y0))(
            psi
        )  # psi: [nl, Ny, Nx] -> q: [nl, Ny, Nx]

        # Cross-layer stretching (A couples layers, cannot be vmapped)
        q = q - stretching_term(A, psi)

    Parameters
    ----------
    psi : Float[Array, "Ny Nx"]
        Streamfunction at T-points.
    f0 : float
        Reference Coriolis parameter.
    beta : float
        Meridional gradient of the Coriolis parameter.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    y : Float[Array, "Ny Nx"]
        Meridional coordinate at T-points.
    y0 : float
        Reference latitude.

    Returns
    -------
    Float[Array, "Ny Nx"]
        QG potential vorticity at T-points.
        Ghost ring is zero; interior is ``[1:-1, 1:-1]``.
    """
    out = jnp.zeros_like(psi)
    # Laplacian at T-points via centred second differences:
    # d2psi/dx2[j,i] = (psi[j,i+1] - 2*psi[j,i] + psi[j,i-1]) / dx^2
    d2x = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dx**2
    # d2psi/dy2[j,i] = (psi[j+1,i] - 2*psi[j,i] + psi[j-1,i]) / dy^2
    d2y = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dy**2
    # q[j,i] = (d2psi/dx2 + d2psi/dy2) / f0 + beta * (y - y0) / f0
    q_int = (d2x + d2y) / f0 + beta * (y[1:-1, 1:-1] - y0) / f0
    out = out.at[1:-1, 1:-1].set(q_int)
    return out


def stretching_term(
    A: Float[Array, "nl nl"],
    psi: Float[Array, "nl Ny Nx"],
) -> Float[Array, "nl Ny Nx"]:
    """Cross-layer stretching term for multi-layer QG models.

    Computes ``A . psi`` — the vertical coupling that mixes information
    across layers.  This is a cross-layer operation and cannot be vmapped;
    it should be used alongside :func:`qg_potential_vorticity` (which
    handles the per-layer Laplacian + beta terms via
    :func:`~finitevolx.multilayer`).

    Parameters
    ----------
    A : Float[Array, "nl nl"]
        Coupling (stretching) matrix.  Shape ``(nl, nl)`` where ``nl``
        is the number of layers.
    psi : Float[Array, "nl Ny Nx"]
        Streamfunction at T-points for all layers.

    Returns
    -------
    Float[Array, "nl Ny Nx"]
        Stretching contribution, same shape as ``psi``.
        Ghost ring is zero; interior is ``[:, 1:-1, 1:-1]``.
    """
    # (A . psi)_k = sum_j A[k,j] * psi_j  for each spatial point
    # Flatten spatial dims, matmul along layer axis, reshape back.
    nl = A.shape[0]
    Ny, Nx = psi.shape[-2], psi.shape[-1]
    psi_flat = psi.reshape(nl, -1)  # [nl, Ny*Nx]
    Ap_flat = A @ psi_flat  # [nl, Ny*Nx]
    Ap = Ap_flat.reshape(nl, Ny, Nx)  # [nl, Ny, Nx]
    out = jnp.zeros_like(psi)
    out = out.at[:, 1:-1, 1:-1].set(Ap[:, 1:-1, 1:-1])
    return out


# ======================================================================
# Domain-integrated diagnostics  (Issue #73)
# ======================================================================


def total_energy(
    ke: Float[Array, "Ny Nx"],
    ape: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, ""]:
    """Domain-integrated total energy.

    E = sum_{interior}(KE + APE) * dx * dy

    Parameters
    ----------
    ke : Float[Array, "Ny Nx"]
        Kinetic energy at T-points.
    ape : Float[Array, "Ny Nx"]
        Available potential energy at T-points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, ""]
        Scalar total energy.
    """
    return jnp.sum(ke[1:-1, 1:-1] + ape[1:-1, 1:-1]) * dx * dy


def total_enstrophy(
    ens: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, ""]:
    """Domain-integrated enstrophy.

    Z = sum_{interior}(0.5 * omega^2) * dx * dy

    Parameters
    ----------
    ens : Float[Array, "Ny Nx"]
        Enstrophy field (e.g. from :func:`enstrophy`).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, ""]
        Scalar total enstrophy.
    """
    return jnp.sum(ens[1:-1, 1:-1]) * dx * dy


# ======================================================================
# Vertical velocity diagnostic  (Issue #82)
# ======================================================================


def vertical_velocity(
    u: Float[Array, "Nz Ny Nx"],
    v: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
    mask: Float[Array, "Nz Ny Nx"] | None = None,
) -> Float[Array, "Nzp1 Ny Nx"]:
    """Vertical velocity from the continuity equation.

    Integrates the horizontal divergence from bottom (w=0) to top:

        w[k+1/2] = w[k-1/2] - (du/dx + dv/dy)[k] * dz

    where the horizontal divergence is computed at T-points for each
    z-level.

    Parameters
    ----------
    u : Float[Array, "Nz Ny Nx"]
        x-velocity at U-points, with ghost cells in all three dimensions.
    v : Float[Array, "Nz Ny Nx"]
        y-velocity at V-points, with ghost cells in all three dimensions.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dz : float
        Grid spacing in z.
    mask : Float[Array, "Nz Ny Nx"] | None, optional
        Binary mask at T-points.  If provided, the horizontal divergence
        is zeroed where mask is 0 before vertical integration.

    Returns
    -------
    Float[Array, "Nzp1 Ny Nx"]
        Vertical velocity at W-points (cell interfaces in z).
        Shape is ``(Nz + 1, Ny, Nx)``.  ``w[0]`` is the bottom boundary
        (zero) and ``w[Nz]`` is the top.  Only the horizontal interior
        ``[1:-1, 1:-1]`` contains meaningful values.
    """
    Nz, Ny, Nx = u.shape
    # Horizontal divergence at T-points for each z-level
    # du/dx[k, j, i] = (u[k, j, i] - u[k, j, i-1]) / dx
    # dv/dy[k, j, i] = (v[k, j, i] - v[k, j-1, i]) / dy
    du_dx = jnp.zeros_like(u)
    du_dx = du_dx.at[:, 1:-1, 1:-1].set((u[:, 1:-1, 1:-1] - u[:, 1:-1, :-2]) / dx)
    dv_dy = jnp.zeros_like(v)
    dv_dy = dv_dy.at[:, 1:-1, 1:-1].set((v[:, 1:-1, 1:-1] - v[:, :-2, 1:-1]) / dy)
    div_h = du_dx + dv_dy  # [Nz, Ny, Nx]

    if mask is not None:
        div_h = div_h * mask

    # Integrate bottom-to-top: w[0] = 0 (bottom)
    # w[k+1] = w[k] - div_h[k] * dz  for k = 0, ..., Nz-1
    w = jnp.zeros((Nz + 1, Ny, Nx), dtype=u.dtype)
    cumsum = jnp.cumsum(div_h, axis=0)  # [Nz, Ny, Nx]
    w = w.at[1:, :, :].set(-cumsum * dz)
    return w
