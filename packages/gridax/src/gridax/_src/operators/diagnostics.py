"""
Diagnostic quantities on Arakawa C-grids.

Includes kinetic / potential energy, strain, enstrophy, Okubo-Weiss,
and domain-integrated conservation diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from geophysix.constants import STANDARD
from jaxtyping import Float
import numpy as np

from gridax._src.backend import set_at, vmap
from gridax._src.mask import Mask3D
from gridax._src.operators._ghost import interior
from gridax._src.operators.difference import _curl_2d, _laplacian_2d_interior
from gridax._src.operators.stencils import (
    avg_x_bwd_2d,
    avg_xfwd_yfwd_2d,
    avg_y_bwd_2d,
    diff_x_bwd_2d,
    diff_x_bwd_3d,
    diff_x_fwd_2d,
    diff_y_bwd_2d,
    diff_y_bwd_3d,
    diff_y_fwd_2d,
)
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

# ======================================================================
# Kinetic energy & Bernoulli potential  (existing)
# ======================================================================


def kinetic_energy(
    u: Float[Array, "*batch Ny Nx"],
    v: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Kinetic energy at T-points (cell centers) on an Arakawa C-grid.

    Eq:
        ke[j, i] = 0.5 * (u²_on_T[j, i] + v²_on_T[j, i])

    where u² and v² are averaged from face-points to T-points:
        u²_on_T[j, i] = 0.5 * (u[j, i+1/2]² + u[j, i-1/2]²)
                       = 0.5 * (u[j, i]² + u[j, i-1]²)
        v²_on_T[j, i] = 0.5 * (v[j+1/2, i]² + v[j-1/2, i]²)
                       = 0.5 * (v[j, i]² + v[j-1, i]²)

    This is a mask-free Layer-2 functional helper per #209.  For masked
    kinetic energy, multiply the result by your T-point mask at the call
    site: ``ke = kinetic_energy(u, v) * mask.h``.

    Args:
        u (Array): x-velocity at U-points (east faces), shape [Ny, Nx].
        v (Array): y-velocity at V-points (north faces), shape [Ny, Nx].

    Returns:
        ke (Array): kinetic energy at T-points, shape [Ny, Nx].
            Ghost ring is zero; interior is [1:-1, 1:-1].
    """
    xp = array_namespace(u, v)
    dtype = xp.result_type(u, v, 0.0)
    u_float = xp.asarray(u, dtype=dtype)
    v_float = xp.asarray(v, dtype=dtype)
    u2 = u_float**2
    v2 = v_float**2
    # u²_on_T[j, i] = 0.5 * (u²[j, i] + u²[j, i-1])  (east + west U-faces)
    # v²_on_T[j, i] = 0.5 * (v²[j, i] + v²[j-1, i])  (north + south V-faces)
    u2_on_T = avg_x_bwd_2d(u2)
    v2_on_T = avg_y_bwd_2d(v2)
    ke_int = 0.5 * (u2_on_T + v2_on_T)
    out = interior(ke_int, u_float)
    return out


def bernoulli_potential(
    h: Float[Array, "*batch Ny Nx"],
    u: Float[Array, "*batch Ny Nx"],
    v: Float[Array, "*batch Ny Nx"],
    gravity: float = STANDARD.standard_gravity,
) -> Float[Array, "*batch Ny Nx"]:
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
    xp = array_namespace(h, u, v)
    dtype = xp.result_type(h, u, v, 0.0)
    h_float = xp.asarray(h, dtype=dtype)
    ke = kinetic_energy(u=u, v=v)
    out = interior(ke[..., 1:-1, 1:-1] + gravity * h_float[..., 1:-1, 1:-1], h_float)
    return out


# ======================================================================
# Standalone vorticity functions  (Issue #73)
# ======================================================================


def relative_vorticity_cgrid(
    u: Float[Array, "*batch Ny Nx"],
    v: Float[Array, "*batch Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "*batch Ny Nx"]:
    """Relative vorticity at X-points (corners) on an Arakawa C-grid.

    zeta[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                        - (u[j+1, i+1/2] - u[j, i+1/2]) / dy

    This is a standalone functional form of the class-based
    :meth:`Vorticity2D.relative_vorticity`.  Both share the same
    underlying implementation
    (:func:`gridax._src.operators.difference._curl_2d`).

    Parameters
    ----------
    u : Float[Array, "*batch Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "*batch Ny Nx"]
        y-velocity at V-points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Relative vorticity at X-points.  Ghost ring is zero.
    """
    return _curl_2d(u, v, dx, dy)


def potential_vorticity(
    omega: Float[Array, "*batch Ny Nx"],
    f: Float[Array, "*batch Ny Nx"],
    h: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Potential vorticity (pointwise).

    q = (omega + f) / h

    All inputs must live on the same grid points (typically X-points
    after interpolating ``f`` and ``h`` from T-points).

    Parameters
    ----------
    omega : Float[Array, "*batch Ny Nx"]
        Relative vorticity.
    f : Float[Array, "*batch Ny Nx"]
        Coriolis parameter.
    h : Float[Array, "*batch Ny Nx"]
        Layer thickness (or depth).

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Potential vorticity.  Where ``h == 0`` the result is NaN.
    """
    xp = array_namespace(omega, f, h)
    return xp.where(h == 0, xp.nan, (omega + f) / h)


# ======================================================================
# Strain operators  (Issue #2)
# ======================================================================


def shear_strain(
    u: Float[Array, "*batch Ny Nx"],
    v: Float[Array, "*batch Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "*batch Ny Nx"]:
    """Shear strain at X-points (corners) on an Arakawa C-grid.

    Ss[j+1/2, i+1/2] = dv/dx + du/dy
        = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
        + (u[j+1, i+1/2] - u[j, i+1/2]) / dy

    Parameters
    ----------
    u : Float[Array, "*batch Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "*batch Ny Nx"]
        y-velocity at V-points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Shear strain at X-points.  Ghost ring is zero.
    """
    out = interior(diff_x_fwd_2d(v) / dx + diff_y_fwd_2d(u) / dy, u)
    return out


def tensor_strain(
    u: Float[Array, "*batch Ny Nx"],
    v: Float[Array, "*batch Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "*batch Ny Nx"]:
    """Tensor (normal) strain at T-points on an Arakawa C-grid.

    Sn[j, i] = du/dx - dv/dy
        = (u[j, i+1/2] - u[j, i-1/2]) / dx
        - (v[j+1/2, i] - v[j-1/2, i]) / dy

    Parameters
    ----------
    u : Float[Array, "*batch Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "*batch Ny Nx"]
        y-velocity at V-points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Tensor strain at T-points.  Ghost ring is zero.
    """
    out = interior(diff_x_bwd_2d(u) / dx - diff_y_bwd_2d(v) / dy, u)
    return out


def strain_magnitude_squared(
    sn: Float[Array, "*batch Ny Nx"],
    ss: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Squared strain magnitude (pointwise).

    sigma^2 = Sn^2 + Ss^2

    Inputs must live on the same grid points.  If ``sn`` is at T-points
    and ``ss`` at X-points, interpolate one to the other before calling.

    Parameters
    ----------
    sn : Float[Array, "*batch Ny Nx"]
        Tensor (normal) strain.
    ss : Float[Array, "*batch Ny Nx"]
        Shear strain.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Squared strain magnitude.
    """
    return sn**2 + ss**2


def okubo_weiss(
    sn: Float[Array, "*batch Ny Nx"],
    ss: Float[Array, "*batch Ny Nx"],
    omega: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Okubo-Weiss parameter (pointwise).

    OW = Sn^2 + Ss^2 - omega^2

    All inputs must live on the same grid points.

    Parameters
    ----------
    sn : Float[Array, "*batch Ny Nx"]
        Tensor (normal) strain.
    ss : Float[Array, "*batch Ny Nx"]
        Shear strain.
    omega : Float[Array, "*batch Ny Nx"]
        Relative vorticity.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Okubo-Weiss parameter.  Positive = strain-dominated,
        negative = vorticity-dominated.
    """
    return sn**2 + ss**2 - omega**2


# ======================================================================
# Enstrophy  (Issues #2 and #73)
# ======================================================================


def enstrophy(
    omega: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Enstrophy (pointwise).

    Z = 0.5 * omega^2

    Mask-free Layer-2 functional helper per #209.  For masked enstrophy,
    multiply the result by your mask at the call site:
    ``z = enstrophy(omega) * mask.xy_corner_strict``.

    Parameters
    ----------
    omega : Float[Array, "*batch Ny Nx"]
        Relative vorticity (typically at X-points).

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Enstrophy at the same grid points as *omega*.
    """
    return 0.5 * omega**2


def potential_enstrophy(
    q: Float[Array, "*batch Ny Nx"],
    h: Float[Array, "*batch Ny Nx"],
) -> Float[Array, "*batch Ny Nx"]:
    """Potential enstrophy (pointwise).

    PE = 0.5 * q^2 * h

    Inputs must live on the same grid points.  Mask-free Layer-2
    functional helper per #209; apply masks at the call site.

    Parameters
    ----------
    q : Float[Array, "*batch Ny Nx"]
        Potential vorticity.
    h : Float[Array, "*batch Ny Nx"]
        Layer thickness (or depth).

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Potential enstrophy.
    """
    return 0.5 * q**2 * h


# ======================================================================
# Potential energy  (Issue #73)
# ======================================================================


def available_potential_energy(
    h: Float[Array, "*batch Ny Nx"],
    H: Float[Array, "*batch Ny Nx"],
    g_prime: float | Float[Array, ""],
) -> Float[Array, "*batch Ny Nx"]:
    """Available potential energy at T-points.

    APE = 0.5 * g' * (h - H)^2

    Parameters
    ----------
    h : Float[Array, "*batch Ny Nx"]
        Layer thickness at T-points.
    H : Float[Array, "*batch Ny Nx"]
        Mean (reference) layer thickness at T-points.
    g_prime : float
        Reduced gravity.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Available potential energy at T-points.
    """
    eta = h - H
    return 0.5 * g_prime * eta**2


# ======================================================================
# QG potential vorticity  (Issue #73)
# ======================================================================


def qg_potential_vorticity(
    psi: Float[Array, "*batch Ny Nx"],
    f0: float | Float[Array, ""],
    beta: float | Float[Array, ""],
    dx: float,
    dy: float,
    y: Float[Array, "*batch Ny Nx"],
    y0: float,
) -> Float[Array, "*batch Ny Nx"]:
    """QG potential vorticity at T-points (single layer).

    q = nabla^2(psi) / f0 + beta * (y - y0) / f0

    This computes the per-layer (barotropic) QG PV for a single ``[Ny, Nx]``
    streamfunction.  For multi-layer QG, lift this function with
    ``jax.vmap`` and subtract the cross-layer stretching
    term separately::

        import jax
        from gridax_jax import qg_potential_vorticity

        # Per-layer PV (vmapped over the layer axis)
        q = jax.vmap(lambda p: qg_potential_vorticity(p, f0, beta, dx, dy, y, y0))(
            psi
        )  # psi: [nl, Ny, Nx] -> q: [nl, Ny, Nx]

        # Cross-layer stretching (A couples layers, cannot be vmapped);
        # A is the raw build_coupling_matrix output, so scale by f0
        q = q - f0 * stretching_term(A, psi)

    Parameters
    ----------
    psi : Float[Array, "*batch Ny Nx"]
        Streamfunction at T-points.
    f0 : float
        Reference Coriolis parameter.
    beta : float
        Meridional gradient of the Coriolis parameter.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    y : Float[Array, "*batch Ny Nx"]
        Meridional coordinate at T-points.
    y0 : float
        Reference latitude.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        QG potential vorticity at T-points.
        Ghost ring is zero; interior is ``[1:-1, 1:-1]``.
    """
    # q[j,i] = (d2psi/dx2 + d2psi/dy2) / f0 + beta * (y - y0) / f0, with the
    # Laplacian of Difference2D.laplacian
    q_int = (
        _laplacian_2d_interior(psi, dx, dy) / f0 + beta * (y[..., 1:-1, 1:-1] - y0) / f0
    )
    out = interior(q_int, psi)
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
    ``jax.vmap``).

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
    xp = array_namespace(A, psi)
    # (A . psi)_k = sum_j A[k,j] * psi_j  for each spatial point
    # Flatten spatial dims, matmul along layer axis, reshape back.
    nl = A.shape[0]
    Ny, Nx = psi.shape[-2], psi.shape[-1]
    psi_flat = psi.reshape(nl, -1)  # [nl, Ny*Nx]
    Ap_flat = A @ psi_flat  # [nl, Ny*Nx]
    Ap = Ap_flat.reshape(nl, Ny, Nx)  # [nl, Ny, Nx]
    out = xp.zeros_like(psi)
    out = set_at(
        out, (..., slice(None), slice(1, -1), slice(1, -1)), Ap[..., :, 1:-1, 1:-1]
    )
    return out


# ======================================================================
# Domain-integrated diagnostics  (Issue #73)
# ======================================================================


def total_energy(
    ke: Float[Array, "*batch Ny Nx"],
    ape: Float[Array, "*batch Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, ""]:
    """Domain-integrated total energy.

    E = sum_{interior}(KE + APE) * dx * dy

    Parameters
    ----------
    ke : Float[Array, "*batch Ny Nx"]
        Kinetic energy at T-points.
    ape : Float[Array, "*batch Ny Nx"]
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
    xp = array_namespace(ke, ape)
    return xp.sum(ke[..., 1:-1, 1:-1] + ape[..., 1:-1, 1:-1], axis=(-2, -1)) * dx * dy


def total_enstrophy(
    ens: Float[Array, "*batch Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, ""]:
    """Domain-integrated enstrophy.

    Z = sum_{interior}(0.5 * omega^2) * dx * dy

    Parameters
    ----------
    ens : Float[Array, "*batch Ny Nx"]
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
    xp = array_namespace(ens)
    return xp.sum(ens[..., 1:-1, 1:-1], axis=(-2, -1)) * dx * dy


# ======================================================================
# Vertical velocity diagnostic  (Issue #82)
# ======================================================================


def vertical_velocity(
    u: Float[Array, "Nz Ny Nx"],
    v: Float[Array, "Nz Ny Nx"],
    dx: float,
    dy: float,
    dz: float,
    mask: Mask3D | Float[Array, "Nz Ny Nx"] | None = None,
) -> Float[Array, "Nz Ny Nx"]:
    """Vertical velocity from the continuity equation.

    Integrates the horizontal divergence of the physical levels upward from
    a rigid floor (``w = 0``):

        w[k+1/2] = w[k-1/2] - (du/dx + dv/dy)[k] * dz,   k = 1 .. Nz-2

    The result uses the same ``[Nz, Ny, Nx]`` convention as ``Mask3D.w``:
    ``w[k]`` is the **top face** of ``T[k]`` (``k + 1/2``).  So ``w[0]`` is
    the floor (the bottom face of the first physical level ``T[1]``, zero),
    ``w[1:-1]`` are the interfaces above the physical levels, ``w[Nz-2]`` is
    the surface (zero for a rigid lid when the column conserves mass) and
    ``w[Nz-1]`` is outside the domain (zero).  The z-ghost slabs ``k = 0``
    and ``k = Nz-1`` of ``u`` and ``v`` are never read, so BC-filled
    z-ghosts do not change the answer.

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
    mask : Mask3D or Float[Array, "Nz Ny Nx"] or None, optional
        A :class:`Mask3D`: ``u`` / ``v`` are zeroed on dry faces with
        ``jnp.where`` before the stencil (so NaN-filled land cannot leak),
        the divergence on dry T-points (``mask.h``) before the integration,
        and the output on dry W-faces (``mask.w``).  ``mask.w[Nz-2]`` (the
        surface) is dry wherever the top ghost slab is dry, so a non-zero
        rigid-lid residual there is masked out too.  A raw float T-point
        mask (deprecated) only zeroes the divergence.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Vertical velocity at W-points, ``w[k]`` on the top face of ``T[k]``,
        written on ``[1:-1, 1:-1, 1:-1]`` with a zero ghost ring (``w[0]``
        is the zero floor).

    Examples
    --------
    >>> grid = CartesianGrid3D.from_interior(6, 4, 3, Lx=1.0, Ly=1.0, Lz=1.0)
    >>> u = jnp.broadcast_to(jnp.arange(grid.Nx) * grid.dx, (grid.Nz, grid.Ny, grid.Nx))
    >>> w = vertical_velocity(u, jnp.zeros_like(u), grid.dx, grid.dy, grid.dz)
    >>> w.shape, float(w[0, 2, 2]), float(w[1, 2, 2])  # du/dx = 1
    ((5, 6, 8), 0.0, -0.333...)
    """
    xp = array_namespace(u, v, mask)
    if isinstance(mask, Mask3D):
        # zero dry faces before the stencil reads them (NaN-filled land)
        u = xp.where(mask.u, u, 0.0)
        v = xp.where(mask.v, v, 0.0)
    # Horizontal divergence at the interior T-points of the physical levels
    # div[k, j, i] = (u[k, j, i+1/2] - u[k, j, i-1/2]) / dx
    #              + (v[k, j+1/2, i] - v[k, j-1/2, i]) / dy,   k = 1 .. Nz-2
    div_h = diff_x_bwd_3d(u) / dx + diff_y_bwd_3d(v) / dy  # [Nz-2, Ny-2, Nx-2]

    if isinstance(mask, Mask3D):
        div_h = xp.where(mask.h[..., 1:-1, 1:-1, 1:-1], div_h, 0.0)
    elif mask is not None:
        warnings.warn(
            "vertical_velocity(mask=<float array>) is deprecated; pass a "
            "Mask3D (its h zeroes the divergence and its w the output).",
            DeprecationWarning,
            stacklevel=2,
        )
        div_h = div_h * mask[..., 1:-1, 1:-1, 1:-1]

    # w[k+1/2] = -sum_{k'=1..k} div[k'] * dz  ->  w[k] for k = 1 .. Nz-2
    w_int = -xp.cumsum(div_h, axis=0) * dz  # [Nz-2, Ny-2, Nx-2]
    w = interior(w_int, u)  # [Nz, Ny, Nx]; w[0] = 0 is the floor
    if isinstance(mask, Mask3D):
        w = xp.where(mask.w, w, 0.0)
    return w


# ======================================================================
# Streamfunction ↔ SSH conversion  (Issue #153)
# ======================================================================


def streamfn_to_ssh(
    psi: Float[Array, "..."],
    f0: float,
    g: float = STANDARD.standard_gravity,
) -> Float[Array, "..."]:
    """Convert streamfunction to sea surface height.

    eta = (f0 / g) * psi

    Parameters
    ----------
    psi : Float[Array, "..."]
        Streamfunction.
    f0 : float
        Reference Coriolis parameter (1/s).
    g : float
        Gravitational acceleration (m/s²).  Default: standard gravity, 9.80665.

    Returns
    -------
    Float[Array, "..."]
        Sea surface height, same shape as *psi*.
    """
    return (f0 / g) * psi


def ssh_to_streamfn(
    ssh: Float[Array, "..."],
    f0: float,
    g: float = STANDARD.standard_gravity,
) -> Float[Array, "..."]:
    """Convert sea surface height to streamfunction.

    psi = (g / f0) * eta

    Parameters
    ----------
    ssh : Float[Array, "..."]
        Sea surface height.
    f0 : float
        Reference Coriolis parameter (1/s).
    g : float
        Gravitational acceleration (m/s²).  Default: standard gravity, 9.80665.

    Returns
    -------
    Float[Array, "..."]
        Streamfunction, same shape as *ssh*.
    """
    return (g / f0) * ssh


# ======================================================================
# Multilayer QG potential vorticity  (Issue #153)
# ======================================================================


def potential_vorticity_multilayer(
    psi: Float[Array, "nl Ny Nx"],
    A: Float[Array, "nl nl"],
    f0: float,
    beta: float,
    dx: float,
    dy: float,
    y: Float[Array, "Ny Nx"],
    y0: float,
) -> Float[Array, "nl Ny Nx"]:
    """Multi-layer QG potential vorticity.

    q_k = nabla^2(psi_k) / f0 + beta*(y - y0)/f0 - f0 * (A . psi)_k

    This is the MQGeometry form ``nabla^2(psi) - f0^2 A psi + beta (y - y0)``
    divided by ``f0``.  ``A`` is the raw output of
    :func:`~finitevolx.build_coupling_matrix` (``1 / (H g')``, units
    s^2/m^2): ``f0 * A . psi`` is dimensionless like the other two terms,
    so callers must **not** pre-multiply ``A`` by ``f0``.

    Vmaps :func:`qg_potential_vorticity` over the layer axis and
    subtracts ``f0 *`` :func:`stretching_term` in a single call.

    Parameters
    ----------
    psi : Float[Array, "nl Ny Nx"]
        Streamfunction at T-points for all layers.
    A : Float[Array, "nl nl"]
        Layer coupling (stretching) matrix, the raw output of
        :func:`~finitevolx.build_coupling_matrix` (s^2/m^2); the factor
        ``f0`` is applied here.
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
        Reference meridional coordinate (m).

    Returns
    -------
    Float[Array, "nl Ny Nx"]
        QG potential vorticity for all layers.
        Ghost ring is zero; interior is ``[:, 1:-1, 1:-1]``.
    """
    qg_pv_layer = vmap(lambda p: qg_potential_vorticity(p, f0, beta, dx, dy, y, y0))
    # q_k = lap(psi_k)/f0 + beta (y - y0)/f0 - f0 (A psi)_k   [nl, Ny, Nx]
    return qg_pv_layer(psi) - f0 * stretching_term(A, psi)


# ======================================================================
# Shallow-water potential vorticity from prognostic variables  (Issue #153)
# ======================================================================


def _interp_T_to_X(field: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny Nx"]:
    """Interpolate a T-point field to X-points (corners) via 4-point average.

    X[j+1/2, i+1/2] = 0.25 * (T[j,i] + T[j,i+1] + T[j+1,i] + T[j+1,i+1])

    Uses the same NE-corner stencil as :meth:`Interpolation2D.T_to_X`.
    """
    return interior(avg_xfwd_yfwd_2d(field), field)


def sw_potential_vorticity(
    u: Float[Array, "*batch Ny Nx"],
    v: Float[Array, "*batch Ny Nx"],
    h: Float[Array, "*batch Ny Nx"],
    f: Float[Array, "*batch Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "*batch Ny Nx"]:
    """Shallow-water potential vorticity at X-points on an Arakawa C-grid.

    q[j+1/2, i+1/2] = (omega[j+1/2, i+1/2] + f_X[j+1/2, i+1/2])
                       / h_X[j+1/2, i+1/2]

    Computes relative vorticity from ``(u, v)`` at X-points, interpolates
    ``f`` and ``h`` from T-points to X-points, then forms the PV ratio.

    Parameters
    ----------
    u : Float[Array, "*batch Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "*batch Ny Nx"]
        y-velocity at V-points.
    h : Float[Array, "*batch Ny Nx"]
        Layer thickness at T-points.
    f : Float[Array, "*batch Ny Nx"]
        Coriolis parameter at T-points.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "*batch Ny Nx"]
        Potential vorticity at X-points.  Ghost ring is zero.
        Where ``h_X == 0`` the result is NaN.
    """
    xp = array_namespace(u, v, h, f)
    omega = _curl_2d(u, v, dx, dy)
    f_X = _interp_T_to_X(f)
    h_X = _interp_T_to_X(h)
    # Use 0 in ghost ring (where h_X == 0) rather than NaN, matching
    # the ghost-ring-is-zero convention used by all other operators.
    return xp.where(h_X == 0, 0.0, (omega + f_X) / h_X)


def sw_potential_vorticity_multilayer(
    u: Float[Array, "nl Ny Nx"],
    v: Float[Array, "nl Ny Nx"],
    h: Float[Array, "nl Ny Nx"],
    f: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "nl Ny Nx"]:
    """Multi-layer shallow-water potential vorticity at X-points.

    q_k = (omega_k + f) / h_k

    Vmaps :func:`sw_potential_vorticity` over the leading layer axis.
    Layer coupling in SWM occurs through pressure (interface heights),
    not through PV, so no coupling matrix is needed.

    Parameters
    ----------
    u : Float[Array, "nl Ny Nx"]
        x-velocity at U-points for all layers.
    v : Float[Array, "nl Ny Nx"]
        y-velocity at V-points for all layers.
    h : Float[Array, "nl Ny Nx"]
        Layer thickness at T-points for all layers.
    f : Float[Array, "Ny Nx"]
        Coriolis parameter at T-points (shared across layers).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    Float[Array, "nl Ny Nx"]
        Potential vorticity at X-points for all layers.
        Ghost ring is zero.
    """
    return vmap(lambda u_k, v_k, h_k: sw_potential_vorticity(u_k, v_k, h_k, f, dx, dy))(
        u, v, h
    )
