"""
Finite-difference operators on Arakawa C-grids.

All operators follow the interior-point idiom:
  * Output array has the same shape as the input.
  * Only interior cells [1:-1, 1:-1] are written; the ghost ring is zero.
  * The caller is responsible for boundary conditions.

Half-index notation
-------------------
Storage index [j, i] encodes:
  T[j, i]  at cell centre    (j,     i    )
  U[j, i]  at east face      (j,     i+1/2)
  V[j, i]  at north face     (j+1/2, i    )
  X[j, i]  at NE corner      (j+1/2, i+1/2)
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)
from finitevolx._src.mask import Mask1D, Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators.stencils import (
    diff_x_bwd,
    diff_x_bwd_1d,
    diff_x_bwd_3d,
    diff_x_fwd,
    diff_x_fwd_1d,
    diff_x_fwd_3d,
    diff_y_bwd,
    diff_y_bwd_3d,
    diff_y_fwd,
    diff_y_fwd_3d,
)

# ======================================================================
# Shared primitive implementations (used by both class and functional APIs)
# ======================================================================


def _curl_2d(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Curl (relative vorticity) at X-points.  Shared implementation.

    zeta[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx
                        - (u[j+1, i+1/2] - u[j, i+1/2]) / dy
    """
    out = interior(diff_x_fwd(v) / dx - diff_y_fwd(u) / dy, u)
    return out


def _divergence_2d(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
) -> Float[Array, "Ny Nx"]:
    """Divergence at T-points.  Shared implementation.

    delta[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx
                + (v[j+1/2, i] - v[j-1/2, i]) / dy
    """
    like = jnp.zeros(u.shape, dtype=jnp.result_type(u, v))
    return interior(diff_x_bwd(u) / dx + diff_y_bwd(v) / dy, like)


class Difference1D(eqx.Module):
    """Finite-difference operators on a 1-D Arakawa C-grid.

    Parameters
    ----------
    grid : CartesianGrid1D
        The underlying 1-D grid.
    mask : Mask1D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching the
        output stagger:

        * T-output → ``mask.h``
        * U-output → ``mask.u``

        ``None`` (default) leaves every output untouched.
    """

    grid: CartesianGrid1D
    mask: Mask1D | None = None

    def diff_x_T_to_U(self, h: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
        """Forward difference in x: T-point -> U-point.

        dh_dx[i+1/2] = (h[i+1] - h[i]) / dx

        Parameters
        ----------
        h : Float[Array, "Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nx"]
            Forward x-difference at U-points, same shape as input.
            When ``self.mask`` is set, the output is zeroed at dry
            U-faces via ``* self.mask.u``.
        """
        out = interior(diff_x_fwd_1d(h) / self.grid.dx, h)
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def diff_x_U_to_T(self, u: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
        """Backward difference in x: U-point -> T-point.

        du_dx[i] = (u[i+1/2] - u[i-1/2]) / dx

        Parameters
        ----------
        u : Float[Array, "Nx"]
            Velocity field at U-points.

        Returns
        -------
        Float[Array, "Nx"]
            Backward x-difference at T-points, same shape as input.
            When ``self.mask`` is set, the output is zeroed at dry
            T-cells via ``* self.mask.h``.
        """
        out = interior(diff_x_bwd_1d(u) / self.grid.dx, u)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def laplacian(self, h: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
        """Laplacian at T-points.

        nabla2_h[i] = (h[i+1] - 2*h[i] + h[i-1]) / dx^2

        Parameters
        ----------
        h : Float[Array, "Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nx"]
            Laplacian at T-points, same shape as input.  When
            ``self.mask`` is set, the output is zeroed at dry T-cells
            via ``* self.mask.h``.
        """
        out = interior((diff_x_fwd_1d(h) - diff_x_bwd_1d(h)) / self.grid.dx**2, h)
        if self.mask is not None:
            out = out * self.mask.h
        return out


class Difference2D(eqx.Module):
    """Finite-difference operators on a 2-D Arakawa C-grid.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching the
        output stagger:

        * T-output → ``mask.h`` (divergence, laplacian, U→T, V→T)
        * U-output → ``mask.u`` (T→U, X→U, grad_perp's u component)
        * V-output → ``mask.v`` (T→V, X→V, grad_perp's v component)
        * X-output → ``mask.xy_corner_strict`` (curl, U→X, V→X)

        The strict corner mask is used because an X-point output is
        trusted only when all four surrounding T-cells are wet.
        ``None`` (default) leaves every output untouched.
    """

    grid: CartesianGrid2D
    mask: Mask2D | None = None

    # ------------------------------------------------------------------
    # Forward differences
    # ------------------------------------------------------------------

    def diff_x_T_to_U(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Forward x-difference: T-point -> U-point.

        dh_dx[j, i+1/2] = (h[j, i+1] - h[j, i]) / dx

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Forward x-difference at U-points.  When ``self.mask`` is
            set, the output is zeroed at dry U-faces via ``* mask.u``.
        """
        out = interior(diff_x_fwd(h) / self.grid.dx, h)
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def diff_y_T_to_V(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Forward y-difference: T-point -> V-point.

        dh_dy[j+1/2, i] = (h[j+1, i] - h[j, i]) / dy

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Forward y-difference at V-points.  When ``self.mask`` is
            set, the output is zeroed at dry V-faces via ``* mask.v``.
        """
        out = interior(diff_y_fwd(h) / self.grid.dy, h)
        if self.mask is not None:
            out = out * self.mask.v
        return out

    def diff_y_U_to_X(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Forward y-difference: U-point -> X-point (corner).

        du_dy[j+1/2, i+1/2] = (u[j+1, i+1/2] - u[j, i+1/2]) / dy

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity field at U-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Forward y-difference at X-points.  When ``self.mask`` is
            set, the output is zeroed at dry X-corners via
            ``* mask.xy_corner_strict``.
        """
        out = interior(diff_y_fwd(u) / self.grid.dy, u)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def diff_x_V_to_X(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Forward x-difference: V-point -> X-point (corner).

        dv_dx[j+1/2, i+1/2] = (v[j+1/2, i+1] - v[j+1/2, i]) / dx

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity field at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Forward x-difference at X-points.  When ``self.mask`` is
            set, the output is zeroed at dry X-corners via
            ``* mask.xy_corner_strict``.
        """
        out = interior(diff_x_fwd(v) / self.grid.dx, v)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def diff_y_X_to_U(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Backward y-difference: X-point -> U-point.

        dq_dy[j, i] = (q[j+1/2, i+1/2] - q[j-1/2, i+1/2]) / dy

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Scalar field at X-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Backward y-difference at U-points.  When ``self.mask`` is
            set, the output is zeroed at dry U-faces via ``* mask.u``.
        """
        out = interior(diff_y_bwd(q) / self.grid.dy, q)
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def diff_x_X_to_V(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Backward x-difference: X-point -> V-point.

        dq_dx[j, i] = (q[j+1/2, i+1/2] - q[j+1/2, i-1/2]) / dx

        Parameters
        ----------
        q : Float[Array, "Ny Nx"]
            Scalar field at X-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Backward x-difference at V-points.  When ``self.mask`` is
            set, the output is zeroed at dry V-faces via ``* mask.v``.
        """
        out = interior(diff_x_bwd(q) / self.grid.dx, q)
        if self.mask is not None:
            out = out * self.mask.v
        return out

    # ------------------------------------------------------------------
    # Backward differences
    # ------------------------------------------------------------------

    def diff_x_U_to_T(self, u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Backward x-difference: U-point -> T-point.

        du_dx[j, i] = (u[j, i+1/2] - u[j, i-1/2]) / dx

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            Velocity field at U-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Backward x-difference at T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.h``.
        """
        out = interior(diff_x_bwd(u) / self.grid.dx, u)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def diff_y_V_to_T(self, v: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Backward y-difference: V-point -> T-point.

        dv_dy[j, i] = (v[j+1/2, i] - v[j-1/2, i]) / dy

        Parameters
        ----------
        v : Float[Array, "Ny Nx"]
            Velocity field at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Backward y-difference at T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.h``.
        """
        out = interior(diff_y_bwd(v) / self.grid.dy, v)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    # ------------------------------------------------------------------
    # Compound operators
    # ------------------------------------------------------------------

    def divergence(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Divergence of (u, v) at T-points.

        delta[j, i] = du_dx[j, i] + dv_dy[j, i]
                    = (u[j, i+1/2] - u[j, i-1/2]) / dx
                    + (v[j+1/2, i] - v[j-1/2, i]) / dy

        Parameters
        ----------
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Divergence at T-points.  When ``self.mask`` is set, the
            output is zeroed at dry T-cells via ``* mask.h``.
        """
        out = _divergence_2d(u, v, self.grid.dx, self.grid.dy)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def curl(
        self,
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Curl (relative vorticity) of (u, v) at X-points (corners).

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
            Relative vorticity at X-points (corners).  When
            ``self.mask`` is set, the output is zeroed at dry X-corners
            via ``* mask.xy_corner_strict``.
        """
        out = _curl_2d(u, v, self.grid.dx, self.grid.dy)
        if self.mask is not None:
            out = out * self.mask.xy_corner_strict
        return out

    def laplacian(self, h: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Laplacian at T-points.

        nabla2_h[j, i] = (h[j, i+1] - 2*h[j, i] + h[j, i-1]) / dx^2
                       + (h[j+1, i] - 2*h[j, i] + h[j-1, i]) / dy^2

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Laplacian at T-points.  When ``self.mask`` is set, the
            output is zeroed at dry T-cells via ``* mask.h``.
        """
        # nabla2_h[j, i] = d^2h/dx^2 + d^2h/dy^2
        d2x = (diff_x_fwd(h) - diff_x_bwd(h)) / self.grid.dx**2
        d2y = (diff_y_fwd(h) - diff_y_bwd(h)) / self.grid.dy**2
        out = interior(d2x + d2y, h)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def grad_perp(
        self,
        psi: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Perpendicular gradient: T-point streamfunction to geostrophic velocity.

        Maps a streamfunction ψ at T-points to face-centred geostrophic
        velocities (u, v) = (-∂ψ/∂y, ∂ψ/∂x) on an Arakawa C-grid.

        Expanding the T→X bilinear interpolation into the backward X→U / X→V
        differences gives a compact stencil that reads the T-point ghost cells
        of ψ directly:

        u[j, i+1/2] = -(ψ[j+1,i] + ψ[j+1,i+1] - ψ[j-1,i] - ψ[j-1,i+1]) / (4·dy)
        v[j+1/2, i] =  (ψ[j,i+1] + ψ[j+1,i+1] - ψ[j,i-1] - ψ[j+1,i-1]) / (4·dx)

        The resulting (unmasked) velocity field is discretely non-divergent:
        div(u, v) = 0 at T-points. When ``self.mask`` is set, ``u`` is
        multiplied by ``self.mask.u`` and ``v`` by ``self.mask.v``; the
        divergence-free property no longer holds in general in that case.

        Parameters
        ----------
        psi : Float[Array, "Ny Nx"]
            Streamfunction at T-points.

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            (u, v) — zonal velocity at U-points and meridional velocity at
            V-points.

        References
        ----------
        .. [1] louity/MQGeometry ``fd.py`` — ``grad_perp`` function.
           https://github.com/louity/MQGeometry/blob/main/fd.py
        .. [2] louity/qgsw-pytorch ``finite_diff.py`` — ``grad_perp`` function.
           https://github.com/louity/qgsw-pytorch/blob/main/src/finite_diff.py

        Examples
        --------
        >>> from finitevolx import CartesianGrid2D, Difference2D
        >>> grid = CartesianGrid2D.from_interior(8, 8, 1e3, 1e3)
        >>> diff = Difference2D(grid=grid)
        >>> psi = jnp.ones((grid.Ny, grid.Nx))
        >>> u, v = diff.grad_perp(psi)
        """
        # u[j, i+1/2] = -(ψ[j+1,i] + ψ[j+1,i+1] - ψ[j-1,i] - ψ[j-1,i+1]) / (4·dy)
        u = interior(
            -(psi[2:, 1:-1] + psi[2:, 2:] - psi[:-2, 1:-1] - psi[:-2, 2:])
            / (4.0 * self.grid.dy),
            psi,
        )

        # v[j+1/2, i] = (ψ[j,i+1] + ψ[j+1,i+1] - ψ[j,i-1] - ψ[j+1,i-1]) / (4·dx)
        v = interior(
            (psi[1:-1, 2:] + psi[2:, 2:] - psi[1:-1, :-2] - psi[2:, :-2])
            / (4.0 * self.grid.dx),
            psi,
        )

        if self.mask is not None:
            u = u * self.mask.u
            v = v * self.mask.v

        return u, v


class Difference3D(eqx.Module):
    """Finite-difference operators on a 3-D Arakawa C-grid.

    Operates on the horizontal (y, x) plane for each z-level.
    The 3-D array shape is [Nz, Ny, Nx].

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional land/ocean mask.  When provided, every method
        post-multiplies its output by the mask field matching the
        output stagger (``mask.h``, ``mask.u``, or ``mask.v``).
        ``None`` (default) leaves every output untouched.
    """

    grid: CartesianGrid3D
    mask: Mask3D | None = None

    def diff_x_T_to_U(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Forward x-difference over all z-levels: T -> U.

        dh_dx[k, j, i+1/2] = (h[k, j, i+1] - h[k, j, i]) / dx

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Forward x-difference at U-points.  When ``self.mask`` is
            set, the output is zeroed at dry U-faces via ``* mask.u``.
        """
        out = interior(diff_x_fwd_3d(h) / self.grid.dx, h)
        if self.mask is not None:
            out = out * self.mask.u
        return out

    def diff_y_T_to_V(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Forward y-difference over all z-levels: T -> V.

        dh_dy[k, j+1/2, i] = (h[k, j+1, i] - h[k, j, i]) / dy

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Forward y-difference at V-points.  When ``self.mask`` is
            set, the output is zeroed at dry V-faces via ``* mask.v``.
        """
        out = interior(diff_y_fwd_3d(h) / self.grid.dy, h)
        if self.mask is not None:
            out = out * self.mask.v
        return out

    def diff_x_U_to_T(self, u: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Backward x-difference over all z-levels: U -> T.

        du_dx[k, j, i] = (u[k, j, i+1/2] - u[k, j, i-1/2]) / dx

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Backward x-difference at T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.h``.
        """
        out = interior(diff_x_bwd_3d(u) / self.grid.dx, u)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def diff_y_V_to_T(self, v: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Backward y-difference over all z-levels: V -> T.

        dv_dy[k, j, i] = (v[k, j+1/2, i] - v[k, j-1/2, i]) / dy

        Parameters
        ----------
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Backward y-difference at T-points.  When ``self.mask`` is
            set, the output is zeroed at dry T-cells via ``* mask.h``.
        """
        out = interior(diff_y_bwd_3d(v) / self.grid.dy, v)
        if self.mask is not None:
            out = out * self.mask.h
        return out

    def divergence(
        self,
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Horizontal divergence at T-points over all z-levels.

        delta[k, j, i] = du_dx[k, j, i] + dv_dy[k, j, i]

        Parameters
        ----------
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Divergence at T-points.  When ``self.mask`` is set, the
            output is zeroed at dry T-cells via ``* mask.h`` (applied
            inside the ``diff_*`` sub-calls — both sub-methods share
            the same T-point output stagger, so the mask is already
            baked into each addend).
        """
        return self.diff_x_U_to_T(u) + self.diff_y_V_to_T(v)

    def laplacian(self, h: Float[Array, "Nz Ny Nx"]) -> Float[Array, "Nz Ny Nx"]:
        """Horizontal Laplacian at T-points over all z-levels.

        nabla2_h[k, j, i] = (h[k, j, i+1] - 2*h[k, j, i] + h[k, j, i-1]) / dx^2
                           + (h[k, j+1, i] - 2*h[k, j, i] + h[k, j-1, i]) / dy^2

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar field at T-points.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Laplacian at T-points.  When ``self.mask`` is set, the
            output is zeroed at dry T-cells via ``* mask.h``.
        """
        d2x = (diff_x_fwd_3d(h) - diff_x_bwd_3d(h)) / self.grid.dx**2
        d2y = (diff_y_fwd_3d(h) - diff_y_bwd_3d(h)) / self.grid.dy**2
        out = interior(d2x + d2y, h)
        if self.mask is not None:
            out = out * self.mask.h
        return out
