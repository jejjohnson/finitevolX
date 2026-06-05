"""
Vorticity and potential-vorticity flux operators on Arakawa C-grids.

Composes Difference2D and Interpolation2D primitives.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators.difference import Difference2D, _curl_2d
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.operators.stencils import (
    avg_x_bwd,
    avg_x_fwd,
    avg_xbwd_yfwd,
    avg_xfwd_ybwd,
    avg_y_bwd,
    avg_y_fwd,
)


class Vorticity2D(eqx.Module):
    """2-D vorticity and PV-flux operators.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, both the internal
        ``Difference2D`` and ``Interpolation2D`` are constructed with
        the same mask, so every method's output inherits the correct
        post-compute zero via the stagger-matched mask field.

        :meth:`potential_vorticity` is the only method that needs
        explicit NaN-sanitisation: under a mask, the denominator
        ``h_on_q`` is zero at every dry X-corner, which would trigger
        the "degenerate layer thickness" NaN sentinel.  We preserve
        that sentinel for *wet* corners (genuine numerical bugs) and
        force dry corners back to exact zero — see the method body.
    """

    grid: CartesianGrid2D
    mask: Mask2D | None
    diff: Difference2D
    interp: Interpolation2D

    def __init__(
        self,
        grid: CartesianGrid2D,
        mask: Mask2D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.diff = Difference2D(grid=grid, mask=mask)
        self.interp = Interpolation2D(grid=grid, mask=mask)

    def relative_vorticity(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Relative vorticity at X-points (corners).

        zeta[j+1/2, i+1/2] = dv_dx[j+1/2, i+1/2] - du_dy[j+1/2, i+1/2]
                            = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                            - (u[j+1, i+1/2] - u[j, i+1/2]) / dy

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Relative vorticity at X-points.
        """
        return self.diff.curl(u, v)

    def potential_vorticity(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        h: Float[Array, "Ny Nx"],
        f: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Potential vorticity at X-points (corners).

        q[j+1/2, i+1/2] = (zeta[j+1/2, i+1/2] + f_on_q[j+1/2, i+1/2])
                         / h_on_q[j+1/2, i+1/2]

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        h : Float[Array, "Ny Nx"]
            Layer thickness at T-points.
        f : Float[Array, "Ny Nx"]
            Coriolis parameter at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Potential vorticity at X-points.  Zero-thickness X-corners
            produce a ``NaN`` sentinel at *wet* corners (a numerical
            bug signal); under ``self.mask``, dry corners are forced
            to exact ``0`` instead so the NaN only fires where the
            user would care about it.
        """
        zeta = self.relative_vorticity(u, v)  # zeta at X-points
        f_on_q = self.interp.T_to_X(f)  # f interpolated to X-points
        h_on_q = self.interp.T_to_X(h)  # h interpolated to X-points
        # q[j+1/2, i+1/2] = (zeta + f) / h  at X-points
        num = zeta[1:-1, 1:-1] + f_on_q[1:-1, 1:-1]
        den = h_on_q[1:-1, 1:-1]
        pv = jnp.where(den == 0, jnp.nan, num / den)
        out = interior(pv, h)
        if self.mask is not None:
            # Under pass-down masking, h_on_q is zero at dry X-corners, so
            # every dry corner hit the NaN branch above — but that's a
            # mask artefact, not a degenerate-thickness bug.  Restore the
            # post-compute-zero semantic: dry corners exactly 0, wet
            # corners keep any NaN they got.
            out = jnp.where(self.mask.xy_corner_strict, out, 0.0)
        return out

    def pv_flux_energy_conserving(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple:
        """Energy-conserving PV flux.

        Interpolate q and velocity independently to faces, then multiply.

        qu[j, i+1/2] = q_on_u[j, i+1/2] * u[j, i+1/2]
        qv[j+1/2, i] = q_on_v[j+1/2, i] * v[j+1/2, i]

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Potential vorticity at X-points.
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        tuple
            (qu at U-points, qv at V-points)
        """
        q_on_u = self.interp.X_to_U(q)  # q_on_u[j, i+1/2] = avg in y
        q_on_v = self.interp.X_to_V(q)  # q_on_v[j+1/2, i] = avg in x
        # qu[j, i+1/2] = q_on_u[j, i+1/2] * u[j, i+1/2]
        qu = interior(q_on_u[1:-1, 1:-1] * u[1:-1, 1:-1], u)
        # qv[j+1/2, i] = q_on_v[j+1/2, i] * v[j+1/2, i]
        qv = interior(q_on_v[1:-1, 1:-1] * v[1:-1, 1:-1], v)
        return qu, qv

    def pv_flux_enstrophy_conserving(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> tuple:
        """Enstrophy-conserving PV flux.

        Multiply q*u at corners/faces, then interpolate to faces.

        qu[j, i+1/2] = X_to_U(q * U_to_X(u))
        qv[j+1/2, i] = X_to_V(q * V_to_X(v))

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Potential vorticity at X-points.
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        tuple
            (qu at U-points, qv at V-points)
        """
        u_on_q = self.interp.U_to_X(u)  # u_on_q[j+1/2, i+1/2]
        v_on_q = self.interp.V_to_X(v)  # v_on_q[j+1/2, i+1/2]
        # Multiply at corners
        # qu_at_q[j+1/2, i+1/2] = q[j+1/2, i+1/2] * u_on_q[j+1/2, i+1/2]
        qu_at_q = interior(q[1:-1, 1:-1] * u_on_q[1:-1, 1:-1], q)
        # qv_at_q[j+1/2, i+1/2] = q[j+1/2, i+1/2] * v_on_q[j+1/2, i+1/2]
        qv_at_q = interior(q[1:-1, 1:-1] * v_on_q[1:-1, 1:-1], q)
        # Interpolate back to faces
        qu = self.interp.X_to_U(qu_at_q)  # qu[j, i+1/2]
        qv = self.interp.X_to_V(qv_at_q)  # qv[j+1/2, i]
        return qu, qv

    def pv_flux_arakawa_lamb(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        alpha: float = 1.0 / 3.0,
    ) -> tuple:
        """Arakawa-Lamb PV flux: weighted blend of energy and enstrophy.

        flux = alpha * energy_conserving + (1 - alpha) * enstrophy_conserving

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Potential vorticity at X-points.
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        alpha : float
            Blending weight.  Default 1/3 gives Arakawa-Lamb scheme.

        Returns
        -------
        tuple
            (qu at U-points, qv at V-points)
        """
        qu_e, qv_e = self.pv_flux_energy_conserving(q, u, v)
        qu_s, qv_s = self.pv_flux_enstrophy_conserving(q, u, v)
        # Weighted blend
        qu = alpha * qu_e + (1.0 - alpha) * qu_s
        qv = alpha * qv_e + (1.0 - alpha) * qv_s
        return qu, qv


class Vorticity3D(eqx.Module):
    """3-D vorticity operators (horizontal plane per z-level).

    Parameters
    ----------
    grid : CartesianGrid3D
    mask : Mask3D or None, optional
        Optional land/ocean mask.  When provided, the result of
        :meth:`relative_vorticity` is post-multiplied by
        ``mask.xy_corner_strict``.  Applied externally rather than
        pass-down because the underlying ``_curl_2d`` is a free
        function (no sub-operator to inject a mask into).
    """

    grid: CartesianGrid3D
    mask: Mask3D | None = None

    def relative_vorticity(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Relative vorticity at X-points over all z-levels.

        zeta[k, j+1/2, i+1/2] = (v[k, j+1/2, i+1] - v[k, j+1/2, i]) / dx
                               - (u[k, j+1, i+1/2] - u[k, j, i+1/2]) / dy

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Relative vorticity at X-points.  When ``self.mask`` is
            set, dry X-corners are zero via ``* mask.xy_corner_strict``.
        """
        out = eqx.filter_vmap(
            lambda u_k, v_k: _curl_2d(u_k, v_k, self.grid.dx, self.grid.dy)
        )(u, v)
        # Zero z-ghost slices to match 3D ghost-ring convention.
        out = zero_z_ghosts(out)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out


def pv_flux_arakawa_lamb(
    q: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    *,
    alpha: float = 1.0 / 3.0,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Arakawa--Lamb (1981) PV flux as a grid-free free function.

    Public functional form of :meth:`Vorticity2D.pv_flux_arakawa_lamb` — the
    energy- and enstrophy-conserving shallow-water vortex-force flux, a
    weighted blend of the energy-conserving and enstrophy-conserving schemes:

        flux = alpha * energy_conserving + (1 - alpha) * enstrophy_conserving

    The blend at ``alpha = 1/3`` is the Arakawa--Lamb scheme that conserves
    both total energy and total potential enstrophy.  This free function
    reproduces the unmasked :class:`Vorticity2D` method bit for bit using only
    the C-grid averaging stencils, so it is reachable without constructing a
    grid or instantiating the class (matching the ``divergence_2d`` style).
    For masked domains, use :class:`Vorticity2D` with a ``mask=``.

    Parameters
    ----------
    q : Float[Array, "Ny Nx"]
        Potential vorticity at X-points (corners), including ghost ring.
    u : Float[Array, "Ny Nx"]
        x-velocity (or x mass flux) at U-points.
    v : Float[Array, "Ny Nx"]
        y-velocity (or y mass flux) at V-points.
    alpha : float, optional
        Blending weight.  Default ``1/3`` (Arakawa--Lamb).

    Returns
    -------
    tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
        ``(qu, qv)`` — the PV flux at U-points and V-points.

    References
    ----------
    Arakawa, A. & Lamb, V. R. (1981). A potential enstrophy and energy
    conserving scheme for the shallow water equations. *Mon. Wea. Rev.* 109,
    18--36.
    """
    # Energy-conserving: interpolate q and velocity to faces independently.
    q_on_u = interior(avg_y_bwd(q), q)  # X_to_U
    q_on_v = interior(avg_x_bwd(q), q)  # X_to_V
    qu_e = interior(q_on_u[1:-1, 1:-1] * u[1:-1, 1:-1], u)
    qv_e = interior(q_on_v[1:-1, 1:-1] * v[1:-1, 1:-1], v)

    # Enstrophy-conserving: multiply at corners, then interpolate to faces.
    u_on_q = interior(avg_y_fwd(u), u)  # U_to_X
    v_on_q = interior(avg_x_fwd(v), v)  # V_to_X
    qu_at_q = interior(q[1:-1, 1:-1] * u_on_q[1:-1, 1:-1], q)
    qv_at_q = interior(q[1:-1, 1:-1] * v_on_q[1:-1, 1:-1], q)
    qu_s = interior(avg_y_bwd(qu_at_q), qu_at_q)  # X_to_U
    qv_s = interior(avg_x_bwd(qv_at_q), qv_at_q)  # X_to_V

    qu = alpha * qu_e + (1.0 - alpha) * qu_s
    qv = alpha * qv_e + (1.0 - alpha) * qv_s
    return qu, qv


# Linear (optimal-polynomial) one-sided reconstruction weights, divided by their
# common denominator and indexed from the most-upwind cell.  The ``*_POS``
# stencil is left-biased (used where the advecting velocity is >= 0); ``*_NEG``
# is right-biased (used where it is < 0).
_UP3_DENOM = 6.0
_UP3_POS = (-1.0, 5.0, 2.0)
_UP3_NEG = (2.0, 5.0, -1.0)
_UP5_DENOM = 60.0
_UP5_POS = (2.0, -13.0, 47.0, 27.0, -3.0)
_UP5_NEG = (-3.0, 27.0, 47.0, -13.0, 2.0)


def _upwind_face_bwd_last(
    omega: Float[Array, "..."],
    vel: Float[Array, "..."],
    order: int,
) -> Float[Array, "..."]:
    """Upwind-reconstruct ``omega`` to the interior faces along the last axis.

    Faces follow the backward-average stagger: face ``i`` lies between cells
    ``i-1`` (minus side) and ``i`` (plus side), so the centred limit matches
    :func:`avg_x_bwd` / :func:`avg_y_bwd`.  The upwind bias is chosen by the
    sign of ``vel`` at each face.  Returns the ``N - 2`` interior-face values
    along the last axis (all other axes untouched).  Higher orders fall back
    to a lower-order stencil on the outer face(s) where the wide stencil does
    not fit.
    """
    v = vel[..., 1:-1]  # advecting velocity at interior faces i = 1 .. N-2

    # Order 1: pure upwind neighbour.
    f1 = jnp.where(v >= 0.0, omega[..., 0:-2], omega[..., 1:-1])
    if order == 1:
        return f1

    n = omega.shape[-1]
    # 3rd-order stencil valid for faces i = 2 .. N-2 (needs cell i-2 .. i+1).
    o0, o1, o2, o3 = (
        omega[..., 0:-3],
        omega[..., 1:-2],
        omega[..., 2:-1],
        omega[..., 3:],
    )
    v3 = vel[..., 2:-1]
    f3_pos = (_UP3_POS[0] * o0 + _UP3_POS[1] * o1 + _UP3_POS[2] * o2) / _UP3_DENOM
    f3_neg = (_UP3_NEG[0] * o1 + _UP3_NEG[1] * o2 + _UP3_NEG[2] * o3) / _UP3_DENOM
    f3 = jnp.where(v3 >= 0.0, f3_pos, f3_neg)
    # Outermost interior face (i = 1) falls back to 1st-order upwind.
    face3 = jnp.concatenate([f1[..., 0:1], f3], axis=-1)
    if order == 3 or n < 7:
        return face3

    # 5th-order stencil valid for faces i = 3 .. N-3 (needs cell i-3 .. i+2).
    p0, p1, p2, p3, p4, p5 = (
        omega[..., 0:-5],
        omega[..., 1:-4],
        omega[..., 2:-3],
        omega[..., 3:-2],
        omega[..., 4:-1],
        omega[..., 5:],
    )
    v5 = vel[..., 3:-2]
    f5_pos = (
        _UP5_POS[0] * p0
        + _UP5_POS[1] * p1
        + _UP5_POS[2] * p2
        + _UP5_POS[3] * p3
        + _UP5_POS[4] * p4
    ) / _UP5_DENOM
    f5_neg = (
        _UP5_NEG[0] * p1
        + _UP5_NEG[1] * p2
        + _UP5_NEG[2] * p3
        + _UP5_NEG[3] * p4
        + _UP5_NEG[4] * p5
    ) / _UP5_DENOM
    f5 = jnp.where(v5 >= 0.0, f5_pos, f5_neg)
    # The two leading and one trailing interior faces fall back to 3rd order.
    return jnp.concatenate([face3[..., 0:2], f5, face3[..., -1:]], axis=-1)


def vorticity_flux_upwind(
    omega: Float[Array, "Ny Nx"],
    U: Float[Array, "Ny Nx"],
    V: Float[Array, "Ny Nx"],
    order: int = 3,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    r"""Dissipative upwind vortex-force fluxes on a C-grid.

    The MASSH-parity alternative to the energy/enstrophy-conserving schemes
    (:func:`pv_flux_arakawa_lamb`): instead of averaging, the (absolute)
    vorticity at corners is reconstructed onto the velocity faces with an
    order-``{1, 3, 5}`` *upwind* stencil and multiplied by the cross mass
    flux.  Upwinding makes it monotone and dissipative — a robust, cheap
    baseline whose adjoint stays well-defined (the bias only kinks where the
    advecting flux changes sign).

    For the u-momentum equation the flux is the corner vorticity reconstructed
    in y (advected by the v mass flux ``V`` averaged to U-points) times that
    flux; for the v-momentum equation it is reconstructed in x (advected by
    ``U`` averaged to V-points).  The momentum tendencies are then
    ``dt_u += omega_V`` and ``dt_v -= omega_U`` (caller's responsibility).
    Mirrors ``MASSH sw.py::_omega_adv_upwind{3,5}``.

    Parameters
    ----------
    omega : Float[Array, "Ny Nx"]
        Absolute vorticity ``zeta + f`` (or PV) at X-points (corners),
        including ghost ring.
    U : Float[Array, "Ny Nx"]
        x mass flux ``h * u`` at U-points.
    V : Float[Array, "Ny Nx"]
        y mass flux ``h * v`` at V-points.
    order : int, optional
        Upwind stencil width, one of ``{1, 3, 5}``.  Default ``3``.  Near the
        boundary, where the wide stencil does not fit, it falls back to a
        lower order.

    Returns
    -------
    tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
        ``(omega_V, omega_U)`` — the vortex-force flux at U-points (for the
        u-equation) and V-points (for the v-equation).  Ghost ring is zero.

    Raises
    ------
    ValueError
        If ``order`` is not ``1``, ``3``, or ``5``.
    """
    if order not in (1, 3, 5):
        raise ValueError(f"order must be 1, 3, or 5, got {order!r}")

    # Cross mass flux advecting each velocity face.
    v_on_u = interior(avg_xfwd_ybwd(V), V)  # V_to_U: north flux at U-points
    u_on_v = interior(avg_xbwd_yfwd(U), U)  # U_to_V: east flux at V-points

    # omega reconstructed to U-points along y (axis 0): swap y to the last axis.
    omega_y = jnp.swapaxes(omega, 0, -1)
    vel_y = jnp.swapaxes(v_on_u, 0, -1)
    face_y = jnp.swapaxes(_upwind_face_bwd_last(omega_y, vel_y, order), 0, -1)
    omega_V = interior(face_y[:, 1:-1] * v_on_u[1:-1, 1:-1], omega)

    # omega reconstructed to V-points along x (axis -1, already last).
    face_x = _upwind_face_bwd_last(omega, u_on_v, order)
    omega_U = interior(face_x[1:-1, :] * u_on_v[1:-1, 1:-1], omega)

    return omega_V, omega_U
