"""
Spherical-coordinate advection operators for Arakawa C-grids.

Computes ``∂h/∂t = −∇·(h·u_vec)`` on the sphere using face-value
reconstruction (which is coordinate-independent) followed by a
spherical flux-divergence::

    ∂h/∂t = − 1/(R·cosφ) · [ ∂(h·u)/∂λ + ∂(h·v·cosφ)/∂φ ]

Only the final flux-divergence step differs from the Cartesian
:class:`~finitevolx.Advection2D`.  The reconstruction stencils are
exactly the Cartesian ones — they take a ``Float[Array, "Ny Nx"]``
and produce face fluxes on an Arakawa C-grid without reading any
grid-spacing fields.  The supported ``method`` strings differ by
dimensionality, matching the corresponding Cartesian operators:

* :class:`SphericalAdvection2D` (matches :class:`Advection2D`):
  ``'naive'``, ``'upwind1'``, ``'upwind2'``, ``'upwind3'``,
  ``'weno3'``, ``'weno5'``, ``'wenoz5'``, ``'weno7'``, ``'weno9'``,
  and the TVD limiters ``'minmod'``, ``'van_leer'``, ``'superbee'``,
  ``'mc'``.
* :class:`SphericalAdvection3D` (matches :class:`Advection3D`) —
  the 3-D subset: ``'naive'``, ``'upwind1'``, ``'weno3'``,
  ``'weno5'``, ``'weno7'``, ``'weno9'``, and the TVD limiters.
  (``upwind2``, ``upwind3``, and ``wenoz5`` are 2-D-only because
  :class:`~finitevolx.Reconstruction3D` does not expose native 3-D
  versions of those stencils.)

The spherical 2-D class mirrors the public API of :class:`Advection2D`
(same ``method`` alphabet, same mask-adaptive stencil dispatch) and
differs only by:

* ``grid`` is a :class:`SphericalGrid2D` instead of ``CartesianGrid2D``;
* the final divergence step applies the cos(lat) metric weights.

At the equator (``cos(lat) = 1``) the spherical operator reduces to
the Cartesian advection with ``dx = R·dlon`` and ``dy = R·dlat``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from finitevolx._src.advection.advection import (
    _HIERARCHY_SIZES,
    _MASK_DISPATCHABLE,
    _TVD_LIMITERS,
    _rec_funcs_for_method_2d,
    _rec_funcs_for_method_3d,
)
from finitevolx._src.advection.flux import narrow_mask_hierarchy, upwind_flux
from finitevolx._src.advection.reconstruction import (
    Reconstruction2D,
    Reconstruction3D,
)
from finitevolx._src.grid.spherical import SphericalGrid2D, SphericalGrid3D
from finitevolx._src.mask import Mask2D, Mask3D
from finitevolx._src.operators._ghost import interior
from finitevolx._src.operators._utils import _safe_div_cos


def _reconstruct_faces_2d(
    recon: Reconstruction2D,
    h: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    method: str,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Compute unmasked east/north face fluxes for the given method."""
    if method == "naive":
        return recon.naive_x(h, u), recon.naive_y(h, v)
    if method == "upwind1":
        return recon.upwind1_x(h, u), recon.upwind1_y(h, v)
    if method == "upwind2":
        return recon.upwind2_x(h, u), recon.upwind2_y(h, v)
    if method == "upwind3":
        return recon.upwind3_x(h, u), recon.upwind3_y(h, v)
    if method == "weno3":
        return recon.weno3_x(h, u), recon.weno3_y(h, v)
    if method == "weno5":
        return recon.weno5_x(h, u), recon.weno5_y(h, v)
    if method == "wenoz5":
        return recon.wenoz5_x(h, u), recon.wenoz5_y(h, v)
    if method == "weno7":
        return recon.weno7_x(h, u), recon.weno7_y(h, v)
    if method == "weno9":
        return recon.weno9_x(h, u), recon.weno9_y(h, v)
    if method in _TVD_LIMITERS:
        return (
            recon.tvd_x(h, u, limiter=method),
            recon.tvd_y(h, v, limiter=method),
        )
    raise ValueError(f"Unknown method: {method!r}")


def _reconstruct_faces_3d(
    recon: Reconstruction3D,
    h: Float[Array, "Nz Ny Nx"],
    u: Float[Array, "Nz Ny Nx"],
    v: Float[Array, "Nz Ny Nx"],
    method: str,
) -> tuple[Float[Array, "Nz Ny Nx"], Float[Array, "Nz Ny Nx"]]:
    """Compute unmasked east/north face fluxes for the given 3-D method."""
    if method == "naive":
        return recon.naive_x(h, u), recon.naive_y(h, v)
    if method == "upwind1":
        return recon.upwind1_x(h, u), recon.upwind1_y(h, v)
    if method == "weno3":
        return recon.weno3_x(h, u), recon.weno3_y(h, v)
    if method == "weno5":
        return recon.weno5_x(h, u), recon.weno5_y(h, v)
    if method == "weno7":
        return recon.weno7_x(h, u), recon.weno7_y(h, v)
    if method == "weno9":
        return recon.weno9_x(h, u), recon.weno9_y(h, v)
    if method in _TVD_LIMITERS:
        return (
            recon.tvd_x(h, u, limiter=method),
            recon.tvd_y(h, v, limiter=method),
        )
    raise ValueError(f"Unknown method: {method!r}")


class SphericalAdvection2D(eqx.Module):
    """2-D tracer advection on a spherical Arakawa C-grid.

    Computes ``∂h/∂t = −∇·(h·u_vec)`` on the sphere, using the same
    Cartesian face-value reconstruction primitives (WENO, TVD, upwind,
    etc.) and applying the cos(lat) metric weights only in the final
    flux-divergence step.

    Parameters
    ----------
    grid : SphericalGrid2D
        The underlying 2-D spherical grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask.  When provided and a
        mask-dispatchable method (WENO3/5, WENOz5, any TVD limiter) is
        used, the ``(2, 4, 6)`` adaptive stencil hierarchies for both
        directions are pre-built once in ``__init__`` and reused on
        every call via :func:`upwind_flux` — identical approach to
        :class:`Advection2D`.  For non-dispatchable methods the
        unmasked code path runs and the final tendency is post-
        multiplied by ``mask.h``.  ``None`` (default) gives the
        unmasked behaviour.  Per #209 Q2 a Cartesian ``Mask2D`` is
        used pending a dedicated ``SphericalMask2D`` follow-up.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import SphericalGrid2D, SphericalAdvection2D
    >>> grid = SphericalGrid2D.from_interior(
    ...     nx_interior=16,
    ...     ny_interior=8,
    ...     lon_range=(0.0, 360.0),
    ...     lat_range=(-40.0, 40.0),
    ... )
    >>> op = SphericalAdvection2D(grid=grid)
    >>> h = jnp.ones((grid.Ny, grid.Nx))
    >>> u = jnp.zeros((grid.Ny, grid.Nx))
    >>> v = jnp.zeros((grid.Ny, grid.Nx))
    >>> tend = op(h, u, v, method="upwind1")
    >>> tend.shape
    (10, 18)
    """

    grid: SphericalGrid2D
    mask: Mask2D | None
    recon: Reconstruction2D
    _mask_hierarchy_x: dict[int, Bool[Array, "Ny Nx"]] | None
    _mask_hierarchy_y: dict[int, Bool[Array, "Ny Nx"]] | None

    def __init__(
        self,
        grid: SphericalGrid2D,
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
        """Advective tendency ``−∇·(h·u_vec)`` at T-points on a sphere.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Tracer at T-points.
        u : Float[Array, "Ny Nx"]
            Zonal velocity at U-points.
        v : Float[Array, "Ny Nx"]
            Meridional velocity at V-points.
        method : str
            Reconstruction scheme.  Same alphabet as
            :meth:`Advection2D.__call__`: ``'naive'``, ``'upwind1'``,
            ``'upwind2'``, ``'upwind3'``, ``'weno3'``, ``'weno5'``,
            ``'wenoz5'``, ``'weno7'``, ``'weno9'``, or a TVD limiter
            (``'minmod'``, ``'van_leer'``, ``'superbee'``, ``'mc'``).

        Returns
        -------
        Float[Array, "Ny Nx"]
            Advective tendency at T-points.  Ghost ring and the
            outermost interior ring are zero (matching the Cartesian
            :class:`Advection2D` convention of only writing
            ``[2:-2, 2:-2]``).
        """
        mh_x = self._mask_hierarchy_x
        mh_y = self._mask_hierarchy_y
        if mh_x is not None and mh_y is not None and method in _MASK_DISPATCHABLE:
            rfx, rfy, sizes = _rec_funcs_for_method_2d(self.recon, method)
            mask_x = narrow_mask_hierarchy(mh_x, sizes)
            mask_y = narrow_mask_hierarchy(mh_y, sizes)
            fe = upwind_flux(h, u, dim=1, rec_funcs=rfx, mask_hierarchy=mask_x)
            fn = upwind_flux(h, v, dim=0, rec_funcs=rfy, mask_hierarchy=mask_y)
        else:
            fe, fn = _reconstruct_faces_2d(self.recon, h, u, v, method)

        dlon = self.grid.dlon
        dlat = self.grid.dlat
        R = self.grid.R

        # Flux divergence on the sphere at interior cells [2:-2, 2:-2].
        # Matches the Cartesian Advection2D writing convention
        # (avoids reading ghost-adjacent flux entries).
        cos_T_c = self.grid.cos_lat_T[2:-2, 2:-2]
        cos_V_N = self.grid.cos_lat_V[2:-2, 2:-2]
        cos_V_S = self.grid.cos_lat_V[1:-3, 2:-2]

        du = (fe[2:-2, 2:-2] - fe[2:-2, 1:-3]) / dlon
        dv_term = cos_V_N * fn[2:-2, 2:-2] - cos_V_S * fn[1:-3, 2:-2]
        dv = dv_term / dlat
        out = interior(-_safe_div_cos(du + dv, cos_T_c, R), h, ghost=2)
        if self.mask is not None:
            out = out * self.mask.h
        return out


class SphericalAdvection3D(eqx.Module):
    """3-D tracer advection on a spherical Arakawa C-grid.

    Applies the 2-D spherical advection stencil independently at each
    z-level.  Like :class:`Advection3D`, the mask-dispatched path uses
    native 3-D reconstruction primitives with the z-axis treated as a
    batch dimension; the unmasked path also uses 3-D reconstruction
    primitives.

    Parameters
    ----------
    grid : SphericalGrid3D
    mask : Mask3D or None, optional
        Optional 3-D land/ocean mask.  Pre-builds native 3-D adaptive
        stencil hierarchies ``(2, 4, 6)`` in both horizontal directions
        in ``__init__``.  Non-dispatchable methods use the unmasked path
        and then post-multiply by ``mask.h``.  Per #209 Q2/Q3 a
        Cartesian ``Mask3D`` is used pending a ``SphericalMask3D``
        follow-up.
    """

    grid: SphericalGrid3D
    mask: Mask3D | None
    recon: Reconstruction3D
    _mask_hierarchy_x: dict[int, Bool[Array, "Nz Ny Nx"]] | None
    _mask_hierarchy_y: dict[int, Bool[Array, "Nz Ny Nx"]] | None

    def __init__(
        self,
        grid: SphericalGrid3D,
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
        """3-D advective tendency on a sphere (vmap over z-levels).

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Tracer at T-points.
        u : Float[Array, "Nz Ny Nx"]
            Zonal velocity at U-points.
        v : Float[Array, "Nz Ny Nx"]
            Meridional velocity at V-points.
        method : str
            Reconstruction scheme, same alphabet as
            :meth:`Advection3D.__call__`.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Advective tendency at T-points.
        """
        mh_x = self._mask_hierarchy_x
        mh_y = self._mask_hierarchy_y
        if mh_x is not None and mh_y is not None and method in _MASK_DISPATCHABLE:
            rfx, rfy, sizes = _rec_funcs_for_method_3d(self.recon, method)
            mask_x = narrow_mask_hierarchy(mh_x, sizes)
            mask_y = narrow_mask_hierarchy(mh_y, sizes)
            fe = upwind_flux(h, u, dim=2, rec_funcs=rfx, mask_hierarchy=mask_x)
            fn = upwind_flux(h, v, dim=1, rec_funcs=rfy, mask_hierarchy=mask_y)
        else:
            fe, fn = _reconstruct_faces_3d(self.recon, h, u, v, method)

        dlon = self.grid.dlon
        dlat = self.grid.dlat
        R = self.grid.R

        # cos_* arrays are 2-D; broadcast over the leading Nz axis.
        cos_T_c = self.grid.cos_lat_T[2:-2, 2:-2]
        cos_V_N = self.grid.cos_lat_V[2:-2, 2:-2]
        cos_V_S = self.grid.cos_lat_V[1:-3, 2:-2]

        du = (fe[1:-1, 2:-2, 2:-2] - fe[1:-1, 2:-2, 1:-3]) / dlon
        dv_term = cos_V_N * fn[1:-1, 2:-2, 2:-2] - cos_V_S * fn[1:-1, 1:-3, 2:-2]
        dv = dv_term / dlat
        horiz = -_safe_div_cos(du + dv, cos_T_c, R)

        out = jnp.zeros_like(h).at[1:-1, 2:-2, 2:-2].set(horiz)
        if self.mask is not None:
            out = out * self.mask.h
        return out
