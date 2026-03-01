"""
Advection operators for Arakawa C-grids.

Computes -div(h * u_vec) at T-points using face-value reconstruction.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)


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
            Reconstruction method: 'naive', 'upwind1', 'upwind2', 'upwind3'.

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
        else:
            raise ValueError(f"Unknown method: {method!r}")

        out = jnp.zeros_like(h)
        # dh[i] = -(fe[i+1/2] - fe[i-1/2]) / dx
        out = out.at[1:-1].set(-(fe[1:-1] - fe[:-2]) / self.grid.dx)
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
            Reconstruction method: 'naive', 'upwind1', 'upwind2', 'upwind3'.

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
        else:
            raise ValueError(f"Unknown method: {method!r}")

        out = jnp.zeros_like(h)
        # dh[j, i] = -( (fe[j, i+1/2] - fe[j, i-1/2])/dx
        #             + (fn[j+1/2, i] - fn[j-1/2, i])/dy )
        out = out.at[1:-1, 1:-1].set(
            -(
                (fe[1:-1, 1:-1] - fe[1:-1, :-2]) / self.grid.dx
                + (fn[1:-1, 1:-1] - fn[:-2, 1:-1]) / self.grid.dy
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
            Reconstruction method: 'naive' or 'upwind1'.

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
        else:
            raise ValueError(f"Unknown method: {method!r}")

        out = jnp.zeros_like(h)
        # dh[k, j, i] = -( (fe[k,j,i+1/2] - fe[k,j,i-1/2])/dx
        #                 + (fn[k,j+1/2,i] - fn[k,j-1/2,i])/dy )
        out = out.at[1:-1, 1:-1, 1:-1].set(
            -(
                (fe[1:-1, 1:-1, 1:-1] - fe[1:-1, 1:-1, :-2]) / self.grid.dx
                + (fn[1:-1, 1:-1, 1:-1] - fn[1:-1, :-2, 1:-1]) / self.grid.dy
            )
        )
        return out
