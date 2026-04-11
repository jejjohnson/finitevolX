"""
Vorticity and potential-vorticity flux operators on Arakawa C-grids.

Composes Difference2D and Interpolation2D primitives.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.cartesian import CartesianGrid2D, CartesianGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior, zero_z_ghosts
from finitevolx._src.operators.difference import Difference2D, _curl_2d
from finitevolx._src.operators.interpolation import Interpolation2D


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
