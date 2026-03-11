"""
Advection operators for Arakawa C-grids.

Computes -div(h * u_vec) at T-points using face-value reconstruction.

When an :class:`~finitevolx.ArakawaCGridMask` is supplied to the 2-D or 3-D
operators, the flux computation automatically falls back to a lower-order
stencil near irregular boundaries, using
:func:`~finitevolx.upwind_flux` as the unified dispatch mechanism.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.flux import upwind_flux
from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.masks.cgrid_mask import ArakawaCGridMask
from finitevolx._src.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)

# TVD limiter names supported by the advection operators.
_TVD_LIMITERS = frozenset({"minmod", "van_leer", "superbee", "mc"})


def _rec_funcs_for_method_2d(
    recon: Reconstruction2D,
    method: str,
) -> tuple[
    dict[int, Callable] | None,
    dict[int, Callable] | None,
]:
    """Return ``(rec_funcs_x, rec_funcs_y)`` for :func:`upwind_flux` dispatch.

    Maps a method string to the stencil-hierarchy dicts expected by
    :func:`~finitevolx.upwind_flux`.  Returns ``(None, None)`` for methods
    that do not have a meaningful upwind stencil hierarchy (e.g. ``'naive'``).

    Each dict maps stencil size → reconstruction callable ``(h, vel) -> flux``.
    The lowest tier is always ``{2: upwind1}`` so that there is always a valid
    fallback for cells next to irregular boundaries.
    """
    u1x, u1y = recon.upwind1_x, recon.upwind1_y

    if method == "upwind1":
        return {2: u1x}, {2: u1y}
    if method == "upwind2":
        return {2: u1x, 4: recon.upwind2_x}, {2: u1y, 4: recon.upwind2_y}
    if method == "upwind3":
        return {2: u1x, 4: recon.upwind3_x}, {2: u1y, 4: recon.upwind3_y}
    if method == "weno3":
        return {2: u1x, 4: recon.weno3_x}, {2: u1y, 4: recon.weno3_y}
    if method == "weno5":
        return (
            {2: u1x, 4: recon.weno3_x, 6: recon.weno5_x},
            {2: u1y, 4: recon.weno3_y, 6: recon.weno5_y},
        )
    if method == "weno7":
        return (
            {2: u1x, 4: recon.weno3_x, 6: recon.weno5_x, 8: recon.weno7_x},
            {2: u1y, 4: recon.weno3_y, 6: recon.weno5_y, 8: recon.weno7_y},
        )
    if method == "weno9":
        return (
            {
                2: u1x,
                4: recon.weno3_x,
                6: recon.weno5_x,
                8: recon.weno7_x,
                10: recon.weno9_x,
            },
            {
                2: u1y,
                4: recon.weno3_y,
                6: recon.weno5_y,
                8: recon.weno7_y,
                10: recon.weno9_y,
            },
        )
    if method in _TVD_LIMITERS:
        # Capture the limiter string by value in the closure default argument.
        def tvd_x(h: Array, u: Array, limiter: str = method) -> Array:
            return recon.tvd_x(h, u, limiter=limiter)

        def tvd_y(h: Array, v: Array, limiter: str = method) -> Array:
            return recon.tvd_y(h, v, limiter=limiter)

        return {2: u1x, 4: tvd_x}, {2: u1y, 4: tvd_y}
    # "naive" and anything else: no mask-aware dispatch
    return None, None


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
            ``'upwind3'``, ``'weno3'``, ``'weno5'``, ``'weno7'``, ``'weno9'``,
            or a flux-limiter TVD scheme: ``'minmod'``, ``'van_leer'``,
            ``'superbee'``, ``'mc'``.

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
        elif method == "weno3":
            fe = self.recon.weno3_x(h, u)
        elif method == "weno5":
            fe = self.recon.weno5_x(h, u)
        elif method == "weno7":
            fe = self.recon.weno7_x(h, u)
        elif method == "weno9":
            fe = self.recon.weno9_x(h, u)
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
        mask: ArakawaCGridMask | None = None,
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
            ``'upwind3'``, ``'weno3'``, ``'weno5'``, ``'weno7'``, ``'weno9'``,
            or a flux-limiter TVD scheme: ``'minmod'``, ``'van_leer'``,
            ``'superbee'``, ``'mc'``.
        mask : ArakawaCGridMask or None
            Optional domain mask.  When provided the flux computation
            automatically falls back to a lower-order stencil near irregular
            boundaries using :func:`~finitevolx.upwind_flux`.  Passing a mask
            for a rectangular all-ocean domain is valid and selects the best
            available stencil at every face (including reduced-order near the
            ghost ring).  Has no effect for ``method='naive'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Advective tendency at T-points.
        """
        if mask is not None:
            rec_funcs_x, rec_funcs_y = _rec_funcs_for_method_2d(self.recon, method)
            if rec_funcs_x is not None:
                assert rec_funcs_y is not None, (
                    "rec_funcs_y must be paired with rec_funcs_x in _rec_funcs_for_method_2d"
                )
                stencil_sizes = tuple(sorted(rec_funcs_x.keys()))
                fe = upwind_flux(
                    h,
                    u,
                    dim=1,
                    rec_funcs=rec_funcs_x,
                    mask_hierarchy=mask.get_adaptive_masks(
                        direction="x", stencil_sizes=stencil_sizes
                    ),
                )
                fn = upwind_flux(
                    h,
                    v,
                    dim=0,
                    rec_funcs=rec_funcs_y,
                    mask_hierarchy=mask.get_adaptive_masks(
                        direction="y", stencil_sizes=stencil_sizes
                    ),
                )
                out = jnp.zeros_like(h)
                out = out.at[2:-2, 2:-2].set(
                    -(
                        (fe[2:-2, 2:-2] - fe[2:-2, 1:-3]) / self.grid.dx
                        + (fn[2:-2, 2:-2] - fn[1:-3, 2:-2]) / self.grid.dy
                    )
                )
                return out
            # method == "naive" (or unknown): fall through to non-masked path

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
        elif method == "weno3":
            fe = self.recon.weno3_x(h, u)
            fn = self.recon.weno3_y(h, v)
        elif method == "weno5":
            fe = self.recon.weno5_x(h, u)
            fn = self.recon.weno5_y(h, v)
        elif method == "weno7":
            fe = self.recon.weno7_x(h, u)
            fn = self.recon.weno7_y(h, v)
        elif method == "weno9":
            fe = self.recon.weno9_x(h, u)
            fn = self.recon.weno9_y(h, v)
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
        mask: ArakawaCGridMask | None = None,
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
            Reconstruction method: ``'naive'``, ``'upwind1'``, ``'weno3'``,
            ``'weno5'``, ``'weno7'``, ``'weno9'``, or a flux-limiter TVD
            scheme: ``'minmod'``, ``'van_leer'``, ``'superbee'``, ``'mc'``.
        mask : ArakawaCGridMask or None
            Optional 2-D domain mask, broadcast over all z-levels.  When
            provided the flux computation automatically falls back to a
            lower-order stencil near irregular boundaries.
            Supported methods: ``'upwind1'``, ``'weno3'``, ``'weno5'``,
            TVD limiters.  Has no effect for ``method='naive'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Advective tendency at T-points.
        """
        if mask is not None:
            if method == "upwind1":
                # 1st-order stencil is always available; masked and unmasked are
                # identical, but routing through the masked path is harmless.
                fe = self.recon.upwind1_x(h, u)
                fn = self.recon.upwind1_y(h, v)
            elif method == "weno3":
                fe = self.recon.weno3_x_masked(h, u, mask)
                fn = self.recon.weno3_y_masked(h, v, mask)
            elif method == "weno5":
                fe = self.recon.weno5_x_masked(h, u, mask)
                fn = self.recon.weno5_y_masked(h, v, mask)
            elif method in _TVD_LIMITERS:
                fe = self.recon.tvd_x_masked(h, u, mask, limiter=method)
                fn = self.recon.tvd_y_masked(h, v, mask, limiter=method)
            elif method == "naive":
                # weno7, weno9, naive: no 3-D masked variant; use unmasked
                fe = self.recon.naive_x(h, u)
                fn = self.recon.naive_y(h, v)
            elif method == "weno7":
                fe = self.recon.weno7_x(h, u)
                fn = self.recon.weno7_y(h, v)
            elif method == "weno9":
                fe = self.recon.weno9_x(h, u)
                fn = self.recon.weno9_y(h, v)
            else:
                raise ValueError(f"Unknown method: {method!r}")
        elif method == "naive":
            fe = self.recon.naive_x(h, u)
            fn = self.recon.naive_y(h, v)
        elif method == "upwind1":
            fe = self.recon.upwind1_x(h, u)
            fn = self.recon.upwind1_y(h, v)
        elif method == "weno3":
            fe = self.recon.weno3_x(h, u)
            fn = self.recon.weno3_y(h, v)
        elif method == "weno5":
            fe = self.recon.weno5_x(h, u)
            fn = self.recon.weno5_y(h, v)
        elif method == "weno7":
            fe = self.recon.weno7_x(h, u)
            fn = self.recon.weno7_y(h, v)
        elif method == "weno9":
            fe = self.recon.weno9_x(h, u)
            fn = self.recon.weno9_y(h, v)
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
