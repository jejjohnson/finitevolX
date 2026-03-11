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
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D


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
        out = jnp.zeros_like(h)
        # h_on_u[i+1/2] = 1/2 * (h[i] + h[i+1])
        out = out.at[1:-1].set(0.5 * (h[1:-1] + h[2:]))
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
        out = jnp.zeros_like(u)
        # u_on_h[i] = 1/2 * (u[i+1/2] + u[i-1/2])
        out = out.at[1:-1].set(0.5 * (u[1:-1] + u[:-2]))
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

    def T_to_U(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """T-point -> U-point (east face), x-average.

        h_on_u[j, i+1/2] = 1/2 * (h[j, i] + h[j, i+1])

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to U-points.
        """
        out = jnp.zeros_like(h)
        # h_on_u[j, i+1/2] = 1/2 * (h[j, i] + h[j, i+1])
        out = out.at[1:-1, 1:-1].set(0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:]))
        return out

    def T_to_V(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """T-point -> V-point (north face), y-average.

        h_on_v[j+1/2, i] = 1/2 * (h[j, i] + h[j+1, i])

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to V-points.
        """
        out = jnp.zeros_like(h)
        # h_on_v[j+1/2, i] = 1/2 * (h[j, i] + h[j+1, i])
        out = out.at[1:-1, 1:-1].set(0.5 * (h[1:-1, 1:-1] + h[2:, 1:-1]))
        return out

    def T_to_X(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """T-point -> X-point (NE corner), bilinear average.

        h_on_q[j+1/2, i+1/2] = 1/4 * (h[j,i] + h[j,i+1] + h[j+1,i] + h[j+1,i+1])

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to X-points (corners).
        """
        out = jnp.zeros_like(h)
        # h_on_q[j+1/2, i+1/2] = 1/4 * (h[j,i] + h[j,i+1] + h[j+1,i] + h[j+1,i+1])
        out = out.at[
            1:-1, 1:-1
        ].set(
            0.25
            * (
                h[1:-1, 1:-1]  # h[j,   i  ]
                + h[1:-1, 2:]  # h[j,   i+1]
                + h[2:, 1:-1]  # h[j+1, i  ]
                + h[2:, 2:]  # h[j+1, i+1]
            )
        )
        return out

    # ------------------------------------------------------------------
    # X-point -> face points
    # ------------------------------------------------------------------

    def X_to_U(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> U-point (east face), y-average.

        q_on_u[j, i+1/2] = 1/2 * (q[j+1/2, i+1/2] + q[j-1/2, i+1/2])

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Scalar at X-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to U-points.
        """
        out = jnp.zeros_like(q)
        # q_on_u[j, i+1/2] = 1/2 * (q[j+1/2, i+1/2] + q[j-1/2, i+1/2])
        out = out.at[1:-1, 1:-1].set(0.5 * (q[1:-1, 1:-1] + q[:-2, 1:-1]))
        return out

    def X_to_V(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> V-point (north face), x-average.

        q_on_v[j+1/2, i] = 1/2 * (q[j+1/2, i+1/2] + q[j+1/2, i-1/2])

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Scalar at X-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to V-points.
        """
        out = jnp.zeros_like(q)
        # q_on_v[j+1/2, i] = 1/2 * (q[j+1/2, i+1/2] + q[j+1/2, i-1/2])
        out = out.at[1:-1, 1:-1].set(0.5 * (q[1:-1, 1:-1] + q[1:-1, :-2]))
        return out

    # ------------------------------------------------------------------
    # Face points -> T-point
    # ------------------------------------------------------------------

    def U_to_T(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """U-point -> T-point, x-average.

        u_on_h[j, i] = 1/2 * (u[j, i+1/2] + u[j, i-1/2])

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity at U-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to T-points.
        """
        out = jnp.zeros_like(u)
        # u_on_h[j, i] = 1/2 * (u[j, i+1/2] + u[j, i-1/2])
        out = out.at[1:-1, 1:-1].set(0.5 * (u[1:-1, 1:-1] + u[1:-1, :-2]))
        return out

    def V_to_T(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """V-point -> T-point, y-average.

        v_on_h[j, i] = 1/2 * (v[j+1/2, i] + v[j-1/2, i])

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to T-points.
        """
        out = jnp.zeros_like(v)
        # v_on_h[j, i] = 1/2 * (v[j+1/2, i] + v[j-1/2, i])
        out = out.at[1:-1, 1:-1].set(0.5 * (v[1:-1, 1:-1] + v[:-2, 1:-1]))
        return out

    def X_to_T(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """X-point (corner) -> T-point, bilinear average.

        q_on_h[j, i] = 1/4 * (q[j+1/2,i+1/2] + q[j-1/2,i+1/2]
                             + q[j+1/2,i-1/2] + q[j-1/2,i-1/2])

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Scalar at X-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Scalar interpolated to T-points.
        """
        out = jnp.zeros_like(q)
        # q_on_h[j, i] = 1/4 * (q[j+1/2,i+1/2] + q[j-1/2,i+1/2]
        #                      + q[j+1/2,i-1/2] + q[j-1/2,i-1/2])
        out = out.at[
            1:-1, 1:-1
        ].set(
            0.25
            * (
                q[1:-1, 1:-1]  # q[j+1/2, i+1/2]
                + q[:-2, 1:-1]  # q[j-1/2, i+1/2]
                + q[1:-1, :-2]  # q[j+1/2, i-1/2]
                + q[:-2, :-2]  # q[j-1/2, i-1/2]
            )
        )
        return out

    # ------------------------------------------------------------------
    # Face/center -> X-point (corner)
    # ------------------------------------------------------------------

    def U_to_X(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """U-point -> X-point (corner), y-average.

        u_on_q[j+1/2, i+1/2] = 1/2 * (u[j, i+1/2] + u[j+1, i+1/2])

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity at U-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to X-points.
        """
        out = jnp.zeros_like(u)
        # u_on_q[j+1/2, i+1/2] = 1/2 * (u[j, i+1/2] + u[j+1, i+1/2])
        out = out.at[1:-1, 1:-1].set(0.5 * (u[1:-1, 1:-1] + u[2:, 1:-1]))
        return out

    def V_to_X(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """V-point -> X-point (corner), x-average.

        v_on_q[j+1/2, i+1/2] = 1/2 * (v[j+1/2, i] + v[j+1/2, i+1])

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to X-points.
        """
        out = jnp.zeros_like(v)
        # v_on_q[j+1/2, i+1/2] = 1/2 * (v[j+1/2, i] + v[j+1/2, i+1])
        out = out.at[1:-1, 1:-1].set(0.5 * (v[1:-1, 1:-1] + v[1:-1, 2:]))
        return out

    # ------------------------------------------------------------------
    # Cross-face (bilinear 4-point)
    # ------------------------------------------------------------------

    def U_to_V(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """U-point -> V-point (cross-face bilinear, 4-point).

        u_on_v[j+1/2, i] = 1/4 * (u[j,   i+1/2] + u[j+1, i+1/2]
                                 + u[j,   i-1/2] + u[j+1, i-1/2])

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity at U-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to V-points.
        """
        out = jnp.zeros_like(u)
        # u_on_v[j+1/2, i] = 1/4 * (u[j,i+1/2] + u[j+1,i+1/2] + u[j,i-1/2] + u[j+1,i-1/2])
        out = out.at[
            1:-1, 1:-1
        ].set(
            0.25
            * (
                u[1:-1, 1:-1]  # u[j,   i+1/2]
                + u[2:, 1:-1]  # u[j+1, i+1/2]
                + u[1:-1, :-2]  # u[j,   i-1/2]
                + u[2:, :-2]  # u[j+1, i-1/2]
            )
        )
        return out

    def V_to_U(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """V-point -> U-point (cross-face bilinear, 4-point).

        v_on_u[j, i+1/2] = 1/4 * (v[j+1/2, i] + v[j-1/2, i]
                                 + v[j+1/2, i+1] + v[j-1/2, i+1])

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Velocity interpolated to U-points.
        """
        out = jnp.zeros_like(v)
        # v_on_u[j, i+1/2] = 1/4 * (v[j+1/2,i] + v[j-1/2,i] + v[j+1/2,i+1] + v[j-1/2,i+1])
        out = out.at[
            1:-1, 1:-1
        ].set(
            0.25
            * (
                v[1:-1, 1:-1]  # v[j+1/2, i  ]
                + v[:-2, 1:-1]  # v[j-1/2, i  ]
                + v[1:-1, 2:]  # v[j+1/2, i+1]
                + v[:-2, 2:]  # v[j-1/2, i+1]
            )
        )
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

    def T_to_U(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """T -> U (x-average) over all z-levels.

        h_on_u[k, j, i+1/2] = 1/2 * (h[k, j, i] + h[k, j, i+1])
        """
        out = jnp.zeros_like(h)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 1:-1, 2:])
        )
        return out

    def T_to_V(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """T -> V (y-average) over all z-levels.

        h_on_v[k, j+1/2, i] = 1/2 * (h[k, j, i] + h[k, j+1, i])
        """
        out = jnp.zeros_like(h)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            0.5 * (h[1:-1, 1:-1, 1:-1] + h[1:-1, 2:, 1:-1])
        )
        return out

    def U_to_T(self, u: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """U -> T (x-average) over all z-levels.

        u_on_h[k, j, i] = 1/2 * (u[k, j, i+1/2] + u[k, j, i-1/2])
        """
        out = jnp.zeros_like(u)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            0.5 * (u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2])
        )
        return out

    def V_to_T(self, v: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """V -> T (y-average) over all z-levels.

        v_on_h[k, j, i] = 1/2 * (v[k, j+1/2, i] + v[k, j-1/2, i])
        """
        out = jnp.zeros_like(v)
        out = out.at[1:-1, 1:-1, 1:-1].set(
            0.5 * (v[1:-1, 1:-1, 1:-1] + v[1:-1, :-2, 1:-1])
        )
        return out
