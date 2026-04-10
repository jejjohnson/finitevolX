"""
Vorticity and potential-vorticity flux operators on Arakawa C-grids.

Composes Difference2D and Interpolation2D primitives.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.grid.grid import ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators.difference import Difference2D, _curl_2d
from finitevolx._src.operators.interpolation import Interpolation2D


class Vorticity2D(eqx.Module):
    """2-D vorticity and PV-flux operators.

    Parameters
    ----------
    grid : ArakawaCGrid2D
    """

    grid: ArakawaCGrid2D
    diff: Difference2D
    interp: Interpolation2D

    def __init__(self, grid: ArakawaCGrid2D) -> None:
        self.grid = grid
        self.diff = Difference2D(grid=grid)
        self.interp = Interpolation2D(grid=grid)

    def relative_vorticity(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
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
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the corner-point output
            is multiplied by ``mask.psi`` (strict 4-of-4 corner mask).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Relative vorticity at X-points.
        """
        return self.diff.curl(u, v, mask=mask)

    def potential_vorticity(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        h: Float[Array, "Ny Nx"],
        f: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
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
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the corner-point output
            is multiplied by ``mask.psi``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Potential vorticity at X-points.
        """
        zeta = self.relative_vorticity(u, v)  # zeta at X-points
        f_on_q = self.interp.T_to_X(f)  # f interpolated to X-points
        h_on_q = self.interp.T_to_X(h)  # h interpolated to X-points
        # q[j+1/2, i+1/2] = (zeta + f) / h  at X-points
        num = zeta[1:-1, 1:-1] + f_on_q[1:-1, 1:-1]
        den = h_on_q[1:-1, 1:-1]
        out = interior(jnp.where(den == 0, jnp.nan, num / den), h)
        if mask is not None:
            out = out * mask.psi
        return out

    def pv_flux_energy_conserving(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
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
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, ``qu`` is multiplied by
            ``mask.u`` and ``qv`` by ``mask.v``.

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
        if mask is not None:
            qu = qu * mask.u
            qv = qv * mask.v
        return qu, qv

    def pv_flux_enstrophy_conserving(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
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
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, ``qu`` is multiplied by
            ``mask.u`` and ``qv`` by ``mask.v``.

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
        if mask is not None:
            qu = qu * mask.u
            qv = qv * mask.v
        return qu, qv

    def pv_flux_arakawa_lamb(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        alpha: float = 1.0 / 3.0,
        mask: ArakawaCGridMask | None = None,
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
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, ``qu`` is multiplied by
            ``mask.u`` and ``qv`` by ``mask.v``.

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
        if mask is not None:
            qu = qu * mask.u
            qv = qv * mask.v
        return qu, qv


class Vorticity3D(eqx.Module):
    """3-D vorticity operators (horizontal plane per z-level).

    Parameters
    ----------
    grid : ArakawaCGrid3D
    """

    grid: ArakawaCGrid3D

    def relative_vorticity(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
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
        mask : ArakawaCGridMask or None
            Optional 2-D land/ocean mask, broadcast over all z-levels. If
            provided, the output is multiplied by ``mask.psi``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Relative vorticity at X-points.
        """
        out = eqx.filter_vmap(
            lambda u_k, v_k: _curl_2d(u_k, v_k, self.grid.dx, self.grid.dy)
        )(u, v)
        # Zero z-ghost slices to match 3D ghost-ring convention.
        out = zero_z_ghosts(out)
        if mask is not None:
            out = out * mask.psi
        return out
