"""
Interpolation (averaging) operators on Arakawa C-grids.

All operators follow the interior-point idiom:
  * Output array has the same shape as the input.
  * Only interior cells [1:-1, 1:-1] are written; the ghost ring is zero.

Half-index notation
-------------------
  T[j, i]  cell centre    (j,     i    )
  U[j, i]  east face      (j,     i+1/2)
  V[j, i]  north face     (j+1/2, i    )
  X[j, i]  NE corner      (j+1/2, i+1/2)
"""

import equinox as eqx
from jaxtyping import Array, Float

from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.grid.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators.stencils import (
    avg_x_bwd,
    avg_x_bwd_1d,
    avg_x_bwd_3d,
    avg_x_fwd,
    avg_x_fwd_1d,
    avg_x_fwd_3d,
    avg_xbwd_yfwd,
    avg_xfwd_ybwd,
    avg_xy_bwd,
    avg_xy_fwd,
    avg_y_bwd,
    avg_y_bwd_3d,
    avg_y_fwd,
    avg_y_fwd_3d,
)


class Interpolation1D(eqx.Module):
    """1-D averaging operators on an Arakawa C-grid.

    Parameters
    ----------
    grid : ArakawaCGrid1D
        The underlying 1-D grid.
    """

    grid: ArakawaCGrid1D

    def T_to_U(self, h: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
        """Interpolate T-point -> U-point (east face).

        h_on_u[i+1/2] = 1/2 * (h[i] + h[i+1])

        Parameters
        ----------
        h : Float[Array, "Nx"]
            Scalar at T-points.

        Returns
        -------
        Float[Array, "Nx"]
            Scalar interpolated to U-points.
        """
        out = interior(avg_x_fwd_1d(h), h)
        return out

    def U_to_T(self, u: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
        """Interpolate U-point -> T-point (cell centre).

        u_on_h[i] = 1/2 * (u[i+1/2] + u[i-1/2])

        Parameters
        ----------
        u : Float[Array, "Nx"]
            Velocity at U-points.

        Returns
        -------
        Float[Array, "Nx"]
            Velocity interpolated to T-points.
        """
        out = interior(avg_x_bwd_1d(u), u)
        return out


class Interpolation2D(eqx.Module):
    """2-D averaging operators on an Arakawa C-grid.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        The underlying 2-D grid.
    """

    grid: ArakawaCGrid2D

    # ------------------------------------------------------------------
    # T-point -> faces / corners
    # ------------------------------------------------------------------

    def T_to_U(
        self,
        h: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """T-point -> U-point (east face), x-average.

        h_on_u[j, i+1/2] = 1/2 * (h[j, i] + h[j, i+1])

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the U-point output is
            multiplied by ``mask.u``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to U-points.
        """
        out = interior(avg_x_fwd(h), h)
        if mask is not None:
            out = out * mask.u
        return out

    def T_to_V(
        self,
        h: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """T-point -> V-point (north face), y-average.

        h_on_v[j+1/2, i] = 1/2 * (h[j, i] + h[j+1, i])

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the V-point output is
            multiplied by ``mask.v``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to V-points.
        """
        out = interior(avg_y_fwd(h), h)
        if mask is not None:
            out = out * mask.v
        return out

    def T_to_X(
        self,
        h: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """T-point -> X-point (NE corner), bilinear average.

        h_on_q[j+1/2, i+1/2] = 1/4 * (h[j,i] + h[j,i+1] + h[j+1,i] + h[j+1,i+1])

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the X-point output is
            multiplied by ``mask.psi`` (strict 4-of-4 corner mask).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to X-points (corners).
        """
        out = interior(avg_xy_fwd(h), h)
        if mask is not None:
            out = out * mask.psi
        return out

    # ------------------------------------------------------------------
    # X-point -> face points
    # ------------------------------------------------------------------

    def X_to_U(
        self,
        q: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> U-point (east face), y-average.

        q_on_u[j, i+1/2] = 1/2 * (q[j+1/2, i+1/2] + q[j-1/2, i+1/2])

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Scalar at X-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the U-point output is
            multiplied by ``mask.u``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to U-points.
        """
        out = interior(avg_y_bwd(q), q)
        if mask is not None:
            out = out * mask.u
        return out

    def X_to_V(
        self,
        q: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> V-point (north face), x-average.

        q_on_v[j+1/2, i] = 1/2 * (q[j+1/2, i+1/2] + q[j+1/2, i-1/2])

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Scalar at X-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the V-point output is
            multiplied by ``mask.v``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to V-points.
        """
        out = interior(avg_x_bwd(q), q)
        if mask is not None:
            out = out * mask.v
        return out

    # ------------------------------------------------------------------
    # Face points -> T-point
    # ------------------------------------------------------------------

    def U_to_T(
        self,
        u: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """U-point -> T-point, x-average.

        u_on_h[j, i] = 1/2 * (u[j, i+1/2] + u[j, i-1/2])

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity at U-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the T-point output is
            multiplied by ``mask.h``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to T-points.
        """
        out = interior(avg_x_bwd(u), u)
        if mask is not None:
            out = out * mask.h
        return out

    def V_to_T(
        self,
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """V-point -> T-point, y-average.

        v_on_h[j, i] = 1/2 * (v[j+1/2, i] + v[j-1/2, i])

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity at V-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the T-point output is
            multiplied by ``mask.h``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to T-points.
        """
        out = interior(avg_y_bwd(v), v)
        if mask is not None:
            out = out * mask.h
        return out

    def X_to_T(
        self,
        q: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> T-point, bilinear average.

        q_on_h[j, i] = 1/4 * (q[j+1/2,i+1/2] + q[j-1/2,i+1/2]
                             + q[j+1/2,i-1/2] + q[j-1/2,i-1/2])

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Scalar at X-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the T-point output is
            multiplied by ``mask.h``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to T-points.
        """
        out = interior(avg_xy_bwd(q), q)
        if mask is not None:
            out = out * mask.h
        return out

    # ------------------------------------------------------------------
    # Face/center -> X-point (corner)
    # ------------------------------------------------------------------

    def U_to_X(
        self,
        u: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """U-point -> X-point (corner), y-average.

        u_on_q[j+1/2, i+1/2] = 1/2 * (u[j, i+1/2] + u[j+1, i+1/2])

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity at U-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the X-point output is
            multiplied by ``mask.psi`` (strict 4-of-4 corner mask).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to X-points.
        """
        out = interior(avg_y_fwd(u), u)
        if mask is not None:
            out = out * mask.psi
        return out

    def V_to_X(
        self,
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """V-point -> X-point (corner), x-average.

        v_on_q[j+1/2, i+1/2] = 1/2 * (v[j+1/2, i] + v[j+1/2, i+1])

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity at V-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the X-point output is
            multiplied by ``mask.psi`` (strict 4-of-4 corner mask).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to X-points.
        """
        out = interior(avg_x_fwd(v), v)
        if mask is not None:
            out = out * mask.psi
        return out

    # ------------------------------------------------------------------
    # Cross-face (bilinear 4-point)
    # ------------------------------------------------------------------

    def U_to_V(
        self,
        u: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """U-point -> V-point (cross-face bilinear, 4-point).

        u_on_v[j+1/2, i] = 1/4 * (u[j,   i+1/2] + u[j+1, i+1/2]
                                 + u[j,   i-1/2] + u[j+1, i-1/2])

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity at U-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the V-point output is
            multiplied by ``mask.v``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to V-points.
        """
        out = interior(avg_xbwd_yfwd(u), u)
        if mask is not None:
            out = out * mask.v
        return out

    def V_to_U(
        self,
        v: Float[Array, "Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """V-point -> U-point (cross-face bilinear, 4-point).

        v_on_u[j, i+1/2] = 1/4 * (v[j+1/2, i] + v[j-1/2, i]
                                 + v[j+1/2, i+1] + v[j-1/2, i+1])

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity at V-points.
        mask : ArakawaCGridMask or None
            Optional land/ocean mask. If provided, the U-point output is
            multiplied by ``mask.u``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to U-points.
        """
        out = interior(avg_xfwd_ybwd(v), v)
        if mask is not None:
            out = out * mask.u
        return out


class Interpolation3D(eqx.Module):
    """3-D averaging operators on an Arakawa C-grid.

    Operates on the horizontal (y, x) plane for each z-level.
    Array shape is [Nz, Ny, Nx].

    Parameters
    ----------
    grid : ArakawaCGrid3D
        The underlying 3-D grid.
    """

    grid: ArakawaCGrid3D

    def T_to_U(
        self,
        h: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """T -> U (x-average) over all z-levels.

        h_on_u[k, j, i+1/2] = 1/2 * (h[k, j, i] + h[k, j, i+1])

        ``mask`` is an optional 2-D :class:`ArakawaCGridMask` broadcast
        over all z-levels; if provided, the output is multiplied by
        ``mask.u``.
        """
        out = interior(avg_x_fwd_3d(h), h)
        if mask is not None:
            out = out * mask.u
        return out

    def T_to_V(
        self,
        h: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """T -> V (y-average) over all z-levels.

        h_on_v[k, j+1/2, i] = 1/2 * (h[k, j, i] + h[k, j+1, i])

        ``mask`` is an optional 2-D :class:`ArakawaCGridMask` broadcast
        over all z-levels; if provided, the output is multiplied by
        ``mask.v``.
        """
        out = interior(avg_y_fwd_3d(h), h)
        if mask is not None:
            out = out * mask.v
        return out

    def U_to_T(
        self,
        u: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """U -> T (x-average) over all z-levels.

        u_on_h[k, j, i] = 1/2 * (u[k, j, i+1/2] + u[k, j, i-1/2])

        ``mask`` is an optional 2-D :class:`ArakawaCGridMask` broadcast
        over all z-levels; if provided, the output is multiplied by
        ``mask.h``.
        """
        out = interior(avg_x_bwd_3d(u), u)
        if mask is not None:
            out = out * mask.h
        return out

    def V_to_T(
        self,
        v: Float[Array, "Nz Ny Nx"],
        mask: ArakawaCGridMask | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """V -> T (y-average) over all z-levels.

        v_on_h[k, j, i] = 1/2 * (v[k, j+1/2, i] + v[k, j-1/2, i])

        ``mask`` is an optional 2-D :class:`ArakawaCGridMask` broadcast
        over all z-levels; if provided, the output is multiplied by
        ``mask.h``.
        """
        out = interior(avg_y_bwd_3d(v), v)
        if mask is not None:
            out = out * mask.h
        return out
