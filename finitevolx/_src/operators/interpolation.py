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

from finitevolx._src.grid.base import (
    CurvilinearGrid1D,
    CurvilinearGrid2D,
    CurvilinearGrid3D,
)
from finitevolx._src.mask import Mask1D, Mask2D, Mask3D
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
    grid : CurvilinearGrid1D
        The underlying 1-D grid.
    mask : Mask1D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching the
        output stagger (``mask.h`` for T-output, ``mask.u`` for
        U-output).  ``None`` (default) leaves outputs untouched.
    """

    grid: CurvilinearGrid1D
    mask: Mask1D | None = None

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
            Scalar interpolated to U-points.  When ``self.mask`` is
            set, the output is zeroed at dry U-faces via ``* mask.u``.
        """
        out = interior(avg_x_fwd_1d(h), h)
        if self.mask is not None:
            out = out * self.mask.u
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
            Velocity interpolated to T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.h``.
        """
        out = interior(avg_x_bwd_1d(u), u)
        if self.mask is not None:
            out = out * self.mask.h
        return out


class Interpolation2D(eqx.Module):
    """2-D averaging operators on an Arakawa C-grid.

    Parameters
    ----------
    grid : CurvilinearGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching its
        output stagger:

        * T-output → ``mask.h`` (U_to_T, V_to_T, X_to_T)
        * U-output → ``mask.u`` (T_to_U, X_to_U, V_to_U)
        * V-output → ``mask.v`` (T_to_V, X_to_V, U_to_V)
        * X-output → ``mask.xy_corner_strict`` (T_to_X, U_to_X, V_to_X)

        **Cross-face methods (U_to_V, V_to_U, U_to_X, V_to_X, X_to_U,
        X_to_V):** the post-compute multiply zeros the *dry* output
        cells, but wet output cells that read across a coast still
        contain contributions from the input field at the dry input
        cells (``u[dry]`` / ``v[dry]`` / ``q[dry]`` — whatever the
        caller stored there).  This is the accepted convention — see
        ``tests/test_interpolation_masks.py::TestCrossFaceAudit`` for
        the pinned semantic.
    """

    grid: CurvilinearGrid2D
    mask: Mask2D | None = None

    # ------------------------------------------------------------------
    # T-point -> faces / corners
    # ------------------------------------------------------------------

    def T_to_U(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """T-point -> U-point (east face), x-average.

        h_on_u[j, i+1/2] = 1/2 * (h[j, i] + h[j, i+1])
        """
        out = interior(avg_x_fwd(h), h)
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def T_to_V(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """T-point -> V-point (north face), y-average.

        h_on_v[j+1/2, i] = 1/2 * (h[j, i] + h[j+1, i])
        """
        out = interior(avg_y_fwd(h), h)
        if self.mask is not None:
            out = out * self.mask.v
        return out

    def T_to_X(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """T-point -> X-point (NE corner), bilinear average.

        h_on_q[j+1/2, i+1/2] = 1/4 * (h[j,i] + h[j,i+1] + h[j+1,i] + h[j+1,i+1])
        """
        out = interior(avg_xy_fwd(h), h)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    # ------------------------------------------------------------------
    # X-point -> face points
    # ------------------------------------------------------------------

    def X_to_U(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> U-point (east face), y-average.

        q_on_u[j, i+1/2] = 1/2 * (q[j+1/2, i+1/2] + q[j-1/2, i+1/2])
        """
        out = interior(avg_y_bwd(q), q)
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def X_to_V(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> V-point (north face), x-average.

        q_on_v[j+1/2, i] = 1/2 * (q[j+1/2, i+1/2] + q[j+1/2, i-1/2])
        """
        out = interior(avg_x_bwd(q), q)
        if self.mask is not None:
            out = out * self.mask.v
        return out

    # ------------------------------------------------------------------
    # Face points -> T-point
    # ------------------------------------------------------------------

    def U_to_T(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """U-point -> T-point, x-average.

        u_on_h[j, i] = 1/2 * (u[j, i+1/2] + u[j, i-1/2])
        """
        out = interior(avg_x_bwd(u), u)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def V_to_T(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """V-point -> T-point, y-average.

        v_on_h[j, i] = 1/2 * (v[j+1/2, i] + v[j-1/2, i])
        """
        out = interior(avg_y_bwd(v), v)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def X_to_T(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> T-point, bilinear average.

        q_on_h[j, i] = 1/4 * (q[j+1/2,i+1/2] + q[j-1/2,i+1/2]
                             + q[j+1/2,i-1/2] + q[j-1/2,i-1/2])
        """
        out = interior(avg_xy_bwd(q), q)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    # ------------------------------------------------------------------
    # Face/center -> X-point (corner)
    # ------------------------------------------------------------------

    def U_to_X(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """U-point -> X-point (corner), y-average.

        u_on_q[j+1/2, i+1/2] = 1/2 * (u[j, i+1/2] + u[j+1, i+1/2])
        """
        out = interior(avg_y_fwd(u), u)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def V_to_X(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """V-point -> X-point (corner), x-average.

        v_on_q[j+1/2, i+1/2] = 1/2 * (v[j+1/2, i] + v[j+1/2, i+1])
        """
        out = interior(avg_x_fwd(v), v)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    # ------------------------------------------------------------------
    # Cross-face (bilinear 4-point)
    # ------------------------------------------------------------------

    def U_to_V(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """U-point -> V-point (cross-face bilinear, 4-point).

        u_on_v[j+1/2, i] = 1/4 * (u[j,   i+1/2] + u[j+1, i+1/2]
                                 + u[j,   i-1/2] + u[j+1, i-1/2])
        """
        out = interior(avg_xbwd_yfwd(u), u)
        if self.mask is not None:
            out = out * self.mask.v
        return out

    def V_to_U(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """V-point -> U-point (cross-face bilinear, 4-point).

        v_on_u[j, i+1/2] = 1/4 * (v[j+1/2, i] + v[j-1/2, i]
                                 + v[j+1/2, i+1] + v[j-1/2, i+1])
        """
        out = interior(avg_xfwd_ybwd(v), v)
        if self.mask is not None:
            out = out * self.mask.u
        return out


class Interpolation3D(eqx.Module):
    """3-D averaging operators on an Arakawa C-grid.

    Operates on the horizontal (y, x) plane for each z-level.
    Array shape is [Nz, Ny, Nx].

    Parameters
    ----------
    grid : CurvilinearGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching the
        output stagger (``mask.h`` / ``mask.u`` / ``mask.v``).
    """

    grid: CurvilinearGrid3D
    mask: Mask3D | None = None

    def T_to_U(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """T -> U (x-average) over all z-levels.

        h_on_u[k, j, i+1/2] = 1/2 * (h[k, j, i] + h[k, j, i+1])
        """
        out = interior(avg_x_fwd_3d(h), h)
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def T_to_V(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """T -> V (y-average) over all z-levels.

        h_on_v[k, j+1/2, i] = 1/2 * (h[k, j, i] + h[k, j+1, i])
        """
        out = interior(avg_y_fwd_3d(h), h)
        if self.mask is not None:
            out = out * self.mask.v
        return out

    def U_to_T(self, u: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """U -> T (x-average) over all z-levels.

        u_on_h[k, j, i] = 1/2 * (u[k, j, i+1/2] + u[k, j, i-1/2])
        """
        out = interior(avg_x_bwd_3d(u), u)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def V_to_T(self, v: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """V -> T (y-average) over all z-levels.

        v_on_h[k, j, i] = 1/2 * (v[k, j+1/2, i] + v[k, j-1/2, i])
        """
        out = interior(avg_y_bwd_3d(v), v)
        if self.mask is not None:
            out = out * self.mask.h
        return out
