"""
Advection operators for Arakawa C-grids.

Computes -div(h * u_vec) at T-points using face-value reconstruction.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)

# TVD limiter names supported by the advection operators.
_TVD_LIMITERS = frozenset({"minmod", "van_leer", "superbee", "mc"})


class Advection1D(eqx.Module):
    """1-D advection operator.

    Parameters
    ----------
    grid : ArakawaCGrid1D
    """

    grid: ArakawaCGrid1D
    recon: Reconstruction1D

    def __init__(self, grid: ArakawaCGrid1D) -> None:
        self.grid = grid
        self.recon = Reconstruction1D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
        method: str = "upwind1",
    ) -> Float[Array, "Nx"]:
        """Advective tendency -d(h*u)/dx at T-points.

        dh[i] = -(fe[i+1/2] - fe[i-1/2]) / dx

        Parameters
        ----------
        h : Float[Array, "Nx"]
            Scalar at T-points.
        u : Float[Array, "Nx"]
            Velocity at U-points.
        method : str
            Reconstruction method: ``'naive'``, ``'upwind1'``, ``'upwind2'``,
            ``'upwind3'``, or a flux-limiter TVD scheme:
            ``'minmod'``, ``'van_leer'``, ``'superbee'``, ``'mc'``.

        Returns
        -------
        Float[Array, "Nx"]
            Advective tendency at T-points.
        """
        if method == "naive":
            fe = self.recon.naive_x(h, u)
        elif method == "upwind1":
            fe = self.recon.upwind1_x(h, u)
        elif method == "upwind2":
            fe = self.recon.upwind2_x(h, u)
        elif method == "upwind3":
            fe = self.recon.upwind3_x(h, u)
        elif method in _TVD_LIMITERS:
            fe = self.recon.tvd_x(h, u, limiter=method)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        out = jnp.zeros_like(h)
        # dh[i] = -(fe[i+1/2] - fe[i-1/2]) / dx
        # fe[i] represents the flux at the east face of cell i (at i+1/2)
        # For cell i, we need fe[i] (east) and fe[i-1] (west)
        # Only use face fluxes that are defined by the reconstruction scheme,
        # avoiding the ghost-ring entries fe[0] and fe[-1].
        out = out.at[2:-2].set(-(fe[2:-2] - fe[1:-3]) / self.grid.dx)
        return out


class Advection2D(eqx.Module):
    """2-D advection operator.

    Parameters
    ----------
    grid : ArakawaCGrid2D
    """

    grid: ArakawaCGrid2D
    recon: Reconstruction2D

    def __init__(self, grid: ArakawaCGrid2D) -> None:
        self.grid = grid
        self.recon = Reconstruction2D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        method: str = "upwind1",
    ) -> Float[Array, "Ny Nx"]:
        """Advective tendency -div(h * u_vec) at T-points.

        dh[j, i] = -( (fe[j, i+1/2] - fe[j, i-1/2]) / dx
                    + (fn[j+1/2, i] - fn[j-1/2, i]) / dy )

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        method : str
            Reconstruction method: ``'naive'``, ``'upwind1'``, ``'upwind2'``,
            ``'upwind3'``, or a flux-limiter TVD scheme:
            ``'minmod'``, ``'van_leer'``, ``'superbee'``, ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Advective tendency at T-points.
        """
        if method == "naive":
            fe = self.recon.naive_x(h, u)
            fn = self.recon.naive_y(h, v)
        elif method == "upwind1":
            fe = self.recon.upwind1_x(h, u)
            fn = self.recon.upwind1_y(h, v)
        elif method == "upwind2":
            fe = self.recon.upwind2_x(h, u)
            fn = self.recon.upwind2_y(h, v)
        elif method == "upwind3":
            fe = self.recon.upwind3_x(h, u)
            fn = self.recon.upwind3_y(h, v)
        elif method in _TVD_LIMITERS:
            fe = self.recon.tvd_x(h, u, limiter=method)
            fn = self.recon.tvd_y(h, v, limiter=method)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        out = jnp.zeros_like(h)
        # dh[j, i] = -( (fe[j, i+1/2] - fe[j, i-1/2])/dx
        #             + (fn[j+1/2, i] - fn[j-1/2, i])/dy )
        # fe[j,i] is flux at east face of cell [j,i], fn[j,i] is flux at north face
        # For cell [j,i], we need fe[j,i] (east) and fe[j,i-1] (west),
        #                      and fn[j,i] (north) and fn[j-1,i] (south)
        # Only use face fluxes that are defined by the reconstruction scheme,
        # avoiding ghost-ring flux entries.
        out = out.at[2:-2, 2:-2].set(
            -(
                (fe[2:-2, 2:-2] - fe[2:-2, 1:-3]) / self.grid.dx
                + (fn[2:-2, 2:-2] - fn[1:-3, 2:-2]) / self.grid.dy
            )
        )
        return out


class Advection3D(eqx.Module):
    """3-D advection operator (horizontal plane per z-level).

    Parameters
    ----------
    grid : ArakawaCGrid3D
    """

    grid: ArakawaCGrid3D
    recon: Reconstruction3D

    def __init__(self, grid: ArakawaCGrid3D) -> None:
        self.grid = grid
        self.recon = Reconstruction3D(grid=grid)

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        method: str = "upwind1",
    ) -> Float[Array, "Nz Ny Nx"]:
        """Advective tendency -div(h * u_vec) at T-points over all z-levels.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar at T-points.
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.
        method : str
            Reconstruction method: ``'naive'``, ``'upwind1'``, or a
            flux-limiter TVD scheme:
            ``'minmod'``, ``'van_leer'``, ``'superbee'``, ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Advective tendency at T-points.
        """
        if method == "naive":
            fe = self.recon.naive_x(h, u)
            fn = self.recon.naive_y(h, v)
        elif method == "upwind1":
            fe = self.recon.upwind1_x(h, u)
            fn = self.recon.upwind1_y(h, v)
        elif method in _TVD_LIMITERS:
            fe = self.recon.tvd_x(h, u, limiter=method)
            fn = self.recon.tvd_y(h, v, limiter=method)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        out = jnp.zeros_like(h)
        # dh[k, j, i] = -( (fe[k,j,i+1/2] - fe[k,j,i-1/2])/dx
        #                 + (fn[k,j+1/2,i] - fn[k,j-1/2,i])/dy )
        # Reconstruction writes to [1:-1, 1:-1, 1:-1]; the west flux at i=0
        # and south flux at j=0 are ghost cells (value 0, not filled).
        # Consistent with 1D/2D operators, skip the ghost-adjacent interior
        # ring in the horizontal plane so we never read ghost flux cells.
        # All z-levels are independent, so z uses the full interior [1:-1].
        out = out.at[1:-1, 2:-2, 2:-2].set(
            -(
                (fe[1:-1, 2:-2, 2:-2] - fe[1:-1, 2:-2, 1:-3]) / self.grid.dx
                + (fn[1:-1, 2:-2, 2:-2] - fn[1:-1, 1:-3, 2:-2]) / self.grid.dy
            )
        )
        return out
