"""
Horizontal diffusion operator (flux form) on Arakawa C-grids.

Computes the tracer diffusion tendency ∂h/∂t = ∇·(κ ∇h) at T-points from
staggered face fluxes via forward-then-backward finite differences.

Algorithm (2-D uniform grid with spacing dx, dy)
-------------------------------------------------
Step 1 – East-face flux at U-points (forward diff T → U):

    flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx

Step 2 – North-face flux at V-points (forward diff T → V):

    flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy

Step 3 – Tendency at T-points (backward diff of fluxes, U → T and V → T):

    dh[j, i] = (flux_x[j, i+1/2] - flux_x[j, i-1/2]) / dx
             + (flux_y[j+1/2, i] - flux_y[j-1/2, i]) / dy

Zero-initialized ghost cells on the flux arrays give zero-flux (no-flux) BCs
at domain walls by default.  The caller may overwrite ghost cells before
calling the operator to impose other flux BCs.

Masking
-------
If mask arrays are supplied (1 = ocean, 0 = land):

* ``flux_x *= mask_u`` — zero face flux through land boundaries (U-points).
* ``flux_y *= mask_v`` — zero face flux through land boundaries (V-points).
* ``tendency *= mask_h`` — zero tendency in land cells (T-points).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.grid import ArakawaCGrid2D, ArakawaCGrid3D


def diffusion_2d(
    h: Float[Array, "Ny Nx"],
    kappa: float | Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    mask_h: Float[Array, "Ny Nx"] | None = None,
    mask_u: Float[Array, "Ny Nx"] | None = None,
    mask_v: Float[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    """Horizontal tracer diffusion tendency at T-points (flux form).

    Computes ∂h/∂t = ∇·(κ ∇h) = ∂/∂x(κ ∂h/∂x) + ∂/∂y(κ ∂h/∂y)
    at interior T-points using forward-then-backward finite differences.

    Only interior cells ``[1:-1, 1:-1]`` are written; the ghost ring is left
    as zero.  Zero ghost cells on the intermediate flux arrays correspond to
    zero-flux (no-flux) BCs at domain walls.

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Tracer field at T-points.
    kappa : float or Float[Array, "Ny Nx"]
        Diffusion coefficient.  May be a scalar or an array of the same shape
        as ``h`` (spatially varying diffusivity at T-points).  When ``kappa``
        is an array, the value at the source T-point is used for each face
        flux (i.e., the western/southern cell value for east/north faces).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    mask_h : Float[Array, "Ny Nx"] | None, optional
        Ocean mask at T-points (1 = ocean, 0 = land).  If provided, land-cell
        tendencies are zeroed.
    mask_u : Float[Array, "Ny Nx"] | None, optional
        Ocean mask at U-points (1 = ocean, 0 = land).  If provided, east-face
        fluxes through land boundaries are zeroed.
    mask_v : Float[Array, "Ny Nx"] | None, optional
        Ocean mask at V-points (1 = ocean, 0 = land).  If provided, north-face
        fluxes through land boundaries are zeroed.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Diffusion tendency ∂h/∂t at T-points, same shape as ``h``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> h = jnp.zeros((10, 10))
    >>> tendency = diffusion_2d(h, kappa=1.0, dx=0.1, dy=0.1)
    >>> tendency.shape
    (10, 10)
    """
    # Support both scalar κ and T-point array κ.  When κ is a full [Ny, Nx]
    # array, slice to interior T-cells [1:-1, 1:-1] — the source cell for
    # each face flux (western cell for east face, southern cell for north face).
    kappa_arr = jnp.asarray(kappa)
    kappa_int = kappa_arr[1:-1, 1:-1] if kappa_arr.ndim >= 2 else kappa_arr

    # Step 1: East-face flux at U-points
    # flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx
    flux_x = jnp.zeros_like(h)
    flux_x = flux_x.at[1:-1, 1:-1].set(kappa_int * (h[1:-1, 2:] - h[1:-1, 1:-1]) / dx)
    if mask_u is not None:
        flux_x = flux_x * mask_u

    # Step 2: North-face flux at V-points
    # flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy
    flux_y = jnp.zeros_like(h)
    flux_y = flux_y.at[1:-1, 1:-1].set(kappa_int * (h[2:, 1:-1] - h[1:-1, 1:-1]) / dy)
    if mask_v is not None:
        flux_y = flux_y * mask_v

    # Step 3: Tendency at T-points (divergence of flux)
    # dh[j, i] = (flux_x[j, i+1/2] - flux_x[j, i-1/2]) / dx
    #           + (flux_y[j+1/2, i] - flux_y[j-1/2, i]) / dy
    out = jnp.zeros_like(h)
    du = (flux_x[1:-1, 1:-1] - flux_x[1:-1, :-2]) / dx
    dv = (flux_y[1:-1, 1:-1] - flux_y[:-2, 1:-1]) / dy
    out = out.at[1:-1, 1:-1].set(du + dv)

    if mask_h is not None:
        out = out * mask_h

    return out


class Diffusion2D(eqx.Module):
    """Horizontal diffusion operator (flux form) on a 2-D Arakawa C-grid.

    Computes ∂h/∂t = ∇·(κ ∇h) at T-points from staggered face fluxes via
    forward-then-backward finite differences.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        The underlying 2-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid2D, Diffusion2D
    >>> grid = ArakawaCGrid2D.from_interior(8, 8, 1.0, 1.0)
    >>> diff_op = Diffusion2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> tendency = diff_op(h, kappa=1e-3)  # zero for constant tracer
    >>> flux_x, flux_y = diff_op.fluxes(h, kappa=1e-3)
    """

    grid: ArakawaCGrid2D

    def __call__(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float | Float[Array, "Ny Nx"],
        mask_h: Float[Array, "Ny Nx"] | None = None,
        mask_u: Float[Array, "Ny Nx"] | None = None,
        mask_v: Float[Array, "Ny Nx"] | None = None,
    ) -> Float[Array, "Ny Nx"]:
        """Diffusion tendency ∂h/∂t = ∇·(κ ∇h) at T-points.

        dh[j, i] = (flux_x[j, i+1/2] - flux_x[j, i-1/2]) / dx
                 + (flux_y[j+1/2, i] - flux_y[j-1/2, i]) / dy

        where:
            flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx
            flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Ny Nx"]
            Diffusion coefficient (scalar or T-point field).
        mask_h : Float[Array, "Ny Nx"] | None, optional
            Ocean mask at T-points (1 = ocean, 0 = land).
        mask_u : Float[Array, "Ny Nx"] | None, optional
            Ocean mask at U-points (1 = ocean, 0 = land).
        mask_v : Float[Array, "Ny Nx"] | None, optional
            Ocean mask at V-points (1 = ocean, 0 = land).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Diffusion tendency at T-points.
        """
        return diffusion_2d(
            h,
            kappa,
            self.grid.dx,
            self.grid.dy,
            mask_h=mask_h,
            mask_u=mask_u,
            mask_v=mask_v,
        )

    def fluxes(
        self,
        h: Float[Array, "Ny Nx"],
        kappa: float | Float[Array, "Ny Nx"],
        mask_u: Float[Array, "Ny Nx"] | None = None,
        mask_v: Float[Array, "Ny Nx"] | None = None,
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Diagnostic diffusive face fluxes at U- and V-points.

        Returns the east-face and north-face diffusive fluxes before the
        divergence step, useful for flux-conservative time-stepping and
        diagnostics.

            flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx  (U-points)
            flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy  (V-points)

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Ny Nx"]
            Diffusion coefficient (scalar or T-point field).
        mask_u : Float[Array, "Ny Nx"] | None, optional
            Ocean mask at U-points (1 = ocean, 0 = land).
        mask_v : Float[Array, "Ny Nx"] | None, optional
            Ocean mask at V-points (1 = ocean, 0 = land).

        Returns
        -------
        tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]
            ``(flux_x, flux_y)`` — east-face fluxes at U-points and
            north-face fluxes at V-points.
        """
        # Support both scalar κ and T-point array κ.  When κ is a full [Ny, Nx]
        # array, slice to interior T-cells — the source cell for each face flux
        # (western cell for east face, southern cell for north face).
        kappa_arr = jnp.asarray(kappa)
        kappa_int = kappa_arr[1:-1, 1:-1] if kappa_arr.ndim >= 2 else kappa_arr

        # flux_x[j, i+1/2] = κ * (h[j, i+1] - h[j, i]) / dx
        flux_x = jnp.zeros_like(h)
        flux_x = flux_x.at[1:-1, 1:-1].set(
            kappa_int * (h[1:-1, 2:] - h[1:-1, 1:-1]) / self.grid.dx
        )
        if mask_u is not None:
            flux_x = flux_x * mask_u

        # flux_y[j+1/2, i] = κ * (h[j+1, i] - h[j, i]) / dy
        flux_y = jnp.zeros_like(h)
        flux_y = flux_y.at[1:-1, 1:-1].set(
            kappa_int * (h[2:, 1:-1] - h[1:-1, 1:-1]) / self.grid.dy
        )
        if mask_v is not None:
            flux_y = flux_y * mask_v

        return flux_x, flux_y


class Diffusion3D(eqx.Module):
    """Horizontal diffusion operator (flux form) on a 3-D Arakawa C-grid.

    Applies ∂h/∂t = ∇·(κ ∇h) independently at each z-level.
    The 3-D array shape is [Nz, Ny, Nx].

    Parameters
    ----------
    grid : ArakawaCGrid3D
        The underlying 3-D grid.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import ArakawaCGrid3D, Diffusion3D
    >>> grid = ArakawaCGrid3D.from_interior(4, 8, 8, 1.0, 1.0, 1.0)
    >>> diff_op = Diffusion3D(grid=grid)
    >>> h = jnp.ones((grid.Nz, grid.Ny, grid.Nx))
    >>> tendency = diff_op(h, kappa=1e-3)  # zero for constant tracer
    """

    grid: ArakawaCGrid3D

    def __call__(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float | Float[Array, "Nz Ny Nx"],
        mask_h: Float[Array, "Nz Ny Nx"] | None = None,
        mask_u: Float[Array, "Nz Ny Nx"] | None = None,
        mask_v: Float[Array, "Nz Ny Nx"] | None = None,
    ) -> Float[Array, "Nz Ny Nx"]:
        """Diffusion tendency ∂h/∂t = ∇·(κ ∇h) at T-points over all z-levels.

        Applies the horizontal diffusion stencil independently at each
        z-level.  Only interior cells ``[1:-1, 1:-1, 1:-1]`` are written;
        the ghost ring is left as zero.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Nz Ny Nx"]
            Diffusion coefficient (scalar or T-point field).
        mask_h : Float[Array, "Nz Ny Nx"] | None, optional
            Ocean mask at T-points (1 = ocean, 0 = land).
        mask_u : Float[Array, "Nz Ny Nx"] | None, optional
            Ocean mask at U-points (1 = ocean, 0 = land).
        mask_v : Float[Array, "Nz Ny Nx"] | None, optional
            Ocean mask at V-points (1 = ocean, 0 = land).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Diffusion tendency at T-points.
        """
        # Support both scalar κ and T-point array κ (slice to interior).
        kappa_arr = jnp.asarray(kappa)
        kappa_int = kappa_arr[1:-1, 1:-1, 1:-1] if kappa_arr.ndim >= 3 else kappa_arr

        # Step 1: East-face flux at U-points over all z-levels
        # flux_x[k, j, i+1/2] = κ * (h[k, j, i+1] - h[k, j, i]) / dx
        flux_x = jnp.zeros_like(h)
        flux_x = flux_x.at[1:-1, 1:-1, 1:-1].set(
            kappa_int * (h[1:-1, 1:-1, 2:] - h[1:-1, 1:-1, 1:-1]) / self.grid.dx
        )
        if mask_u is not None:
            flux_x = flux_x * mask_u

        # Step 2: North-face flux at V-points over all z-levels
        # flux_y[k, j+1/2, i] = κ * (h[k, j+1, i] - h[k, j, i]) / dy
        flux_y = jnp.zeros_like(h)
        flux_y = flux_y.at[1:-1, 1:-1, 1:-1].set(
            kappa_int * (h[1:-1, 2:, 1:-1] - h[1:-1, 1:-1, 1:-1]) / self.grid.dy
        )
        if mask_v is not None:
            flux_y = flux_y * mask_v

        # Step 3: Tendency at T-points over all z-levels
        # dh[k, j, i] = (flux_x[k, j, i+1/2] - flux_x[k, j, i-1/2]) / dx
        #             + (flux_y[k, j+1/2, i] - flux_y[k, j-1/2, i]) / dy
        out = jnp.zeros_like(h)
        du = (flux_x[1:-1, 1:-1, 1:-1] - flux_x[1:-1, 1:-1, :-2]) / self.grid.dx
        dv = (flux_y[1:-1, 1:-1, 1:-1] - flux_y[1:-1, :-2, 1:-1]) / self.grid.dy
        out = out.at[1:-1, 1:-1, 1:-1].set(du + dv)

        if mask_h is not None:
            out = out * mask_h

        return out

    def fluxes(
        self,
        h: Float[Array, "Nz Ny Nx"],
        kappa: float | Float[Array, "Nz Ny Nx"],
        mask_u: Float[Array, "Nz Ny Nx"] | None = None,
        mask_v: Float[Array, "Nz Ny Nx"] | None = None,
    ) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
        """Diagnostic diffusive face fluxes at U- and V-points, all z-levels.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Tracer field at T-points.
        kappa : float or Float[Array, "Nz Ny Nx"]
            Diffusion coefficient (scalar or T-point field).
        mask_u : Float[Array, "Nz Ny Nx"] | None, optional
            Ocean mask at U-points (1 = ocean, 0 = land).
        mask_v : Float[Array, "Nz Ny Nx"] | None, optional
            Ocean mask at V-points (1 = ocean, 0 = land).

        Returns
        -------
        tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]
            ``(flux_x, flux_y)`` — east-face fluxes at U-points and
            north-face fluxes at V-points.
        """
        # Support both scalar κ and T-point array κ.  When κ is a full
        # [Nz, Ny, Nx] array, slice to interior T-cells — the source cell for
        # each face flux (western cell for east face, southern cell for north face).
        kappa_arr = jnp.asarray(kappa)
        kappa_int = kappa_arr[1:-1, 1:-1, 1:-1] if kappa_arr.ndim >= 3 else kappa_arr

        # flux_x[k, j, i+1/2] = κ * (h[k, j, i+1] - h[k, j, i]) / dx
        flux_x = jnp.zeros_like(h)
        flux_x = flux_x.at[1:-1, 1:-1, 1:-1].set(
            kappa_int * (h[1:-1, 1:-1, 2:] - h[1:-1, 1:-1, 1:-1]) / self.grid.dx
        )
        if mask_u is not None:
            flux_x = flux_x * mask_u

        # flux_y[k, j+1/2, i] = κ * (h[k, j+1, i] - h[k, j, i]) / dy
        flux_y = jnp.zeros_like(h)
        flux_y = flux_y.at[1:-1, 1:-1, 1:-1].set(
            kappa_int * (h[1:-1, 2:, 1:-1] - h[1:-1, 1:-1, 1:-1]) / self.grid.dy
        )
        if mask_v is not None:
            flux_y = flux_y * mask_v

        return flux_x, flux_y
