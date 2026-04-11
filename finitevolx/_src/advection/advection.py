"""
Advection operators for Arakawa C-grids.

Computes -div(h * u_vec) at T-points using face-value reconstruction.

When a :class:`~finitevolx.Mask1D` / :class:`~finitevolx.Mask2D` /
:class:`~finitevolx.Mask3D` is supplied at construction, the flux
computation automatically falls back to a lower-order stencil near
irregular boundaries, using :func:`~finitevolx.upwind_flux` as the
unified dispatch mechanism.

The adaptive stencil hierarchy is pre-built in ``__init__`` for every
dimension and direction, so ``__call__`` pays only the cost of
tier-selection (a few ``jnp.where``) plus the reconstruction.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from finitevolx._src.advection.flux import narrow_mask_hierarchy, upwind_flux
from finitevolx._src.advection.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)
from finitevolx._src.grid.cartesian import (
    CartesianGrid1D,
    CartesianGrid2D,
    CartesianGrid3D,
)
from finitevolx._src.mask import Mask1D, Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior

# TVD limiter names supported by the advection operators.
_TVD_LIMITERS = frozenset({"minmod", "van_leer", "superbee", "mc"})

# Methods that support mask-based stencil dispatch via upwind_flux.
_MASK_DISPATCHABLE = frozenset({"weno3", "weno5", "wenoz5"}) | _TVD_LIMITERS

# Kept for backwards-compat within this module (Advection2D/3D used to
# have separate 2-D / 3-D dispatchable sets).  They now reference the
# same unified set.
_MASK_DISPATCHABLE_2D = _MASK_DISPATCHABLE
_MASK_DISPATCHABLE_3D = _MASK_DISPATCHABLE

# Pre-built adaptive-hierarchy sizes — the widest set any dispatchable
# method needs in one direction.  Specific methods may use a subset;
# ``narrow_mask_hierarchy`` folds the unused tiers into the largest
# requested one at dispatch time.
_HIERARCHY_SIZES: tuple[int, ...] = (2, 4, 6)


def _rec_funcs_for_method_1d(
    recon: Reconstruction1D, method: str
) -> tuple[dict[int, Callable], tuple[int, ...]]:
    """Build a stencil-hierarchy dict for a given method name in 1-D.

    Returns
    -------
    rec_funcs : dict[int, Callable]
        {stencil_size: reconstruction_fn} for the x-direction.
    stencil_sizes : tuple[int, ...]
        Stencil sizes used (for matching against the pre-built hierarchy).
    """
    if method == "weno5":
        return (
            {2: recon.upwind1_x, 4: recon.weno3_x, 6: recon.weno5_x},
            (2, 4, 6),
        )
    if method == "weno3":
        return (
            {2: recon.upwind1_x, 4: recon.weno3_x},
            (2, 4),
        )
    if method in _TVD_LIMITERS:
        return (
            {
                2: recon.upwind1_x,
                4: lambda q, u, _lim=method: recon.tvd_x(q, u, limiter=_lim),
            },
            (2, 4),
        )
    raise ValueError(f"Method {method!r} does not support mask-based stencil dispatch")


def _rec_funcs_for_method_2d(
    recon: Reconstruction2D, method: str
) -> tuple[dict[int, Callable], dict[int, Callable], tuple[int, ...]]:
    """Build stencil-hierarchy dicts for a given method name.

    Returns
    -------
    rec_funcs_x : dict[int, Callable]
        {stencil_size: reconstruction_fn} for x-direction.
    rec_funcs_y : dict[int, Callable]
        {stencil_size: reconstruction_fn} for y-direction.
    stencil_sizes : tuple[int, ...]
        Stencil sizes used (for ``get_adaptive_masks``).
    """
    if method == "weno5":
        return (
            {2: recon.upwind1_x, 4: recon.weno3_x, 6: recon.weno5_x},
            {2: recon.upwind1_y, 4: recon.weno3_y, 6: recon.weno5_y},
            (2, 4, 6),
        )
    if method == "wenoz5":
        return (
            {2: recon.upwind1_x, 4: recon.wenoz3_x, 6: recon.wenoz5_x},
            {2: recon.upwind1_y, 4: recon.wenoz3_y, 6: recon.wenoz5_y},
            (2, 4, 6),
        )
    if method == "weno3":
        return (
            {2: recon.upwind1_x, 4: recon.weno3_x},
            {2: recon.upwind1_y, 4: recon.weno3_y},
            (2, 4),
        )
    # TVD limiters
    if method in _TVD_LIMITERS:
        return (
            {
                2: recon.upwind1_x,
                4: lambda q, u, _lim=method: recon.tvd_x(q, u, limiter=_lim),
            },
            {
                2: recon.upwind1_y,
                4: lambda q, v, _lim=method: recon.tvd_y(q, v, limiter=_lim),
            },
            (2, 4),
        )
    raise ValueError(f"Method {method!r} does not support mask-based stencil dispatch")


def _rec_funcs_for_method_3d(
    recon: Reconstruction3D, method: str
) -> tuple[dict[int, Callable], dict[int, Callable], tuple[int, ...]]:
    """Build stencil-hierarchy dicts for a given 3-D method name.

    Shares the same tier structure as :func:`_rec_funcs_for_method_2d`
    but uses the native 3-D reconstruction primitives from
    :class:`Reconstruction3D`.

    Note: ``Reconstruction3D`` does not expose ``wenoz5_*`` as of this
    branch, so ``wenoz5`` falls back to ``weno5`` in 3-D.
    """
    if method == "weno5":
        return (
            {2: recon.upwind1_x, 4: recon.weno3_x, 6: recon.weno5_x},
            {2: recon.upwind1_y, 4: recon.weno3_y, 6: recon.weno5_y},
            (2, 4, 6),
        )
    if method == "weno3":
        return (
            {2: recon.upwind1_x, 4: recon.weno3_x},
            {2: recon.upwind1_y, 4: recon.weno3_y},
            (2, 4),
        )
    if method in _TVD_LIMITERS:
        return (
            {
                2: recon.upwind1_x,
                4: lambda q, u, _lim=method: recon.tvd_x(q, u, limiter=_lim),
            },
            {
                2: recon.upwind1_y,
                4: lambda q, v, _lim=method: recon.tvd_y(q, v, limiter=_lim),
            },
            (2, 4),
        )
    raise ValueError(f"Method {method!r} does not support mask-based stencil dispatch")


class Advection1D(eqx.Module):
    """1-D advection operator.

    Parameters
    ----------
    grid : CartesianGrid1D
        The underlying 1-D grid.
    mask : Mask1D or None, optional
        Optional land/ocean mask.  When provided and ``__call__`` is
        invoked with a mask-dispatchable method (WENO3/5, any TVD
        limiter), ``upwind_flux`` is used with the pre-built adaptive
        stencil hierarchy, falling back to lower-order stencils near
        coastlines.  For non-dispatchable methods (``naive``,
        ``upwind1/2/3``, ``weno7``, ``weno9``) the unmasked code path
        runs and the final tendency is post-multiplied by ``mask.h``
        so dry T-cells stay exactly zero.  ``None`` (default) matches
        the pre-existing unmasked behaviour bit for bit.
    """

    grid: CartesianGrid1D
    mask: Mask1D | None
    recon: Reconstruction1D
    _mask_hierarchy_x: dict[int, Bool[Array, "Nx"]] | None

    def __init__(
        self,
        grid: CartesianGrid1D,
        mask: Mask1D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.recon = Reconstruction1D(grid=grid)
        if mask is not None:
            self._mask_hierarchy_x = mask.get_adaptive_masks(
                stencil_sizes=_HIERARCHY_SIZES
            )
        else:
            self._mask_hierarchy_x = None

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
        # ── masked path: route through upwind_flux with adaptive stencils ─
        if self._mask_hierarchy_x is not None and method in _MASK_DISPATCHABLE:
            rfx, sizes = _rec_funcs_for_method_1d(self.recon, method)
            mh = narrow_mask_hierarchy(self._mask_hierarchy_x, sizes)
            fe = upwind_flux(h, u, dim=0, rec_funcs=rfx, mask_hierarchy=mh)
        # ── unmasked path: existing per-method dispatch ─────────────────
        elif method == "naive":
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
        if self.mask is not None:
            out = out * self.mask.h
        return out


class Advection2D(eqx.Module):
    """2-D advection operator.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided, the ``(2, 4, 6)``
        adaptive stencil hierarchies for both directions are
        pre-built once in ``__init__`` and reused on every call —
        this is the main reason the API moved from a per-call
        keyword to a class field.  For mask-dispatchable methods
        (WENO3/5, WENOz5, any TVD limiter), ``upwind_flux`` is used
        with the stored hierarchy narrowed to the method's stencil
        sizes.  For non-dispatchable methods the unmasked code path
        runs and the final tendency is post-multiplied by ``mask.h``.
        ``None`` (default) matches the pre-existing unmasked
        behaviour bit for bit.
    """

    grid: CartesianGrid2D
    mask: Mask2D | None
    recon: Reconstruction2D
    _mask_hierarchy_x: dict[int, Bool[Array, "Ny Nx"]] | None
    _mask_hierarchy_y: dict[int, Bool[Array, "Ny Nx"]] | None

    def __init__(
        self,
        grid: CartesianGrid2D,
        mask: Mask2D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.recon = Reconstruction2D(grid=grid)
        if mask is not None:
            self._mask_hierarchy_x = mask.get_adaptive_masks(
                direction="x", stencil_sizes=_HIERARCHY_SIZES
            )
            self._mask_hierarchy_y = mask.get_adaptive_masks(
                direction="y", stencil_sizes=_HIERARCHY_SIZES
            )
        else:
            self._mask_hierarchy_x = None
            self._mask_hierarchy_y = None

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
            ``'upwind3'``, ``'weno3'``, ``'weno5'``, ``'weno7'``, ``'weno9'``,
            ``'wenoz5'``, or a flux-limiter TVD scheme: ``'minmod'``,
            ``'van_leer'``, ``'superbee'``, ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Advective tendency at T-points.
        """
        # ── masked path: route through upwind_flux with pre-built hier. ──
        mh_x = self._mask_hierarchy_x
        mh_y = self._mask_hierarchy_y
        if mh_x is not None and mh_y is not None and method in _MASK_DISPATCHABLE:
            rfx, rfy, sizes = _rec_funcs_for_method_2d(self.recon, method)
            mask_x = narrow_mask_hierarchy(mh_x, sizes)
            mask_y = narrow_mask_hierarchy(mh_y, sizes)
            fe = upwind_flux(h, u, dim=1, rec_funcs=rfx, mask_hierarchy=mask_x)
            fn = upwind_flux(h, v, dim=0, rec_funcs=rfy, mask_hierarchy=mask_y)
        # ── unmasked path: existing dispatch ────────────────────────────
        elif method == "naive":
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
        elif method == "wenoz5":
            fe = self.recon.wenoz5_x(h, u)
            fn = self.recon.wenoz5_y(h, v)
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

        # dh[j, i] = -( (fe[j, i+1/2] - fe[j, i-1/2])/dx
        #             + (fn[j+1/2, i] - fn[j-1/2, i])/dy )
        # fe[j,i] is flux at east face of cell [j,i], fn[j,i] is flux at north face
        # For cell [j,i], we need fe[j,i] (east) and fe[j,i-1] (west),
        #                      and fn[j,i] (north) and fn[j-1,i] (south)
        # Only use face fluxes that are defined by the reconstruction scheme,
        # avoiding ghost-ring flux entries.
        out = interior(
            -(
                (fe[2:-2, 2:-2] - fe[2:-2, 1:-3]) / self.grid.dx
                + (fn[2:-2, 2:-2] - fn[1:-3, 2:-2]) / self.grid.dy
            ),
            h,
            ghost=2,
        )
        if self.mask is not None:
            out = out * self.mask.h
        return out


class Advection3D(eqx.Module):
    """3-D advection operator (horizontal plane per z-level).

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional 3-D land/ocean mask.  Pre-builds native 3-D adaptive
        stencil hierarchies ``(2, 4, 6)`` in both horizontal directions
        in ``__init__``.  For mask-dispatchable methods (WENO3/5, any
        TVD limiter), ``upwind_flux`` is used with the stored 3-D
        hierarchies narrowed to the method's stencil sizes — the z
        axis is treated as a batch dimension throughout.  For
        non-dispatchable methods the unmasked code path runs and the
        final tendency is post-multiplied by ``mask.h``.

        Pivoted from ``Mask2D`` to ``Mask3D`` per issue #209 Q4 for
        type-uniformity with the rest of the 3-D operator suite.  A
        vertically-homogeneous 3-D mask (the 2-D mask broadcast over
        z) reproduces the old 2-D-mask-broadcast behaviour.  A
        genuinely vertically-varying mask is now natively supported.
    """

    grid: CartesianGrid3D
    mask: Mask3D | None
    recon: Reconstruction3D
    _mask_hierarchy_x: dict[int, Bool[Array, "Nz Ny Nx"]] | None
    _mask_hierarchy_y: dict[int, Bool[Array, "Nz Ny Nx"]] | None

    def __init__(
        self,
        grid: CartesianGrid3D,
        mask: Mask3D | None = None,
    ) -> None:
        self.grid = grid
        self.mask = mask
        self.recon = Reconstruction3D(grid=grid)
        if mask is not None:
            self._mask_hierarchy_x = mask.get_adaptive_masks(
                direction="x", stencil_sizes=_HIERARCHY_SIZES
            )
            self._mask_hierarchy_y = mask.get_adaptive_masks(
                direction="y", stencil_sizes=_HIERARCHY_SIZES
            )
        else:
            self._mask_hierarchy_x = None
            self._mask_hierarchy_y = None

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
            Reconstruction method: ``'naive'``, ``'upwind1'``, ``'weno3'``,
            ``'weno5'``, ``'weno7'``, ``'weno9'``, or a flux-limiter TVD
            scheme: ``'minmod'``, ``'van_leer'``, ``'superbee'``, ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Advective tendency at T-points.
        """
        # ── masked path: route through upwind_flux with pre-built 3-D hier. ──
        mh_x = self._mask_hierarchy_x
        mh_y = self._mask_hierarchy_y
        if mh_x is not None and mh_y is not None and method in _MASK_DISPATCHABLE:
            rfx, rfy, sizes = _rec_funcs_for_method_3d(self.recon, method)
            mask_x = narrow_mask_hierarchy(mh_x, sizes)
            mask_y = narrow_mask_hierarchy(mh_y, sizes)
            # 3-D arrays: x-flux axis is 2, y-flux axis is 1.
            fe = upwind_flux(h, u, dim=2, rec_funcs=rfx, mask_hierarchy=mask_x)
            fn = upwind_flux(h, v, dim=1, rec_funcs=rfy, mask_hierarchy=mask_y)
        # ── unmasked path: existing dispatch ────────────────────────────
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
        if self.mask is not None:
            out = out * self.mask.h
        return out
