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
  ``'weno5'``, ``'wenoz5'``, ``'weno7'``, ``'weno9'``, and the TVD
  limiters.  (``upwind2`` and ``upwind3`` are 2-D-only because
  :class:`~finitevolx.Reconstruction3D` does not expose native 3-D
  versions of those stencils.)

The spherical 2-D class mirrors the public API of :class:`Advection2D`
(same ``method`` alphabet, same mask-adaptive stencil dispatch) and
differs only by:

* ``grid`` is a :class:`SphericalGrid2D` instead of ``CartesianGrid2D``;
* the final divergence step applies the cos(lat) metric weights.

At the equator (``cos(lat) = 1``) the spherical operator reduces to
the Cartesian advection with ``dx = R·dlon`` and ``dy = R·dlat``.

Both classes take the Cartesian ``wall="closed" | "open"`` argument with the
same semantics.  The flux divergence telescopes against
:func:`~finitevolx.spherical_area_weights` (``R² cos(lat_T) dlon dlat``, the
cell area the divergence divides by), so ``Σ A · tendency`` is the net flux
through the faces bounding the written cells, and is zero to rounding when
those cancel:

* ``wall="closed"`` writes ``[2:-2, 2:-2]``, so the sum over the domain
  closes only if the velocity vanishes on the faces bounding that region
  (``i = 1, Nx-3``; ``j = 1, Ny-3``) — not on a zonally periodic grid, where
  the seam-adjacent physical columns ``i = 1, Nx-2`` get no tendency at all
  (#247).
* ``wall="open"`` writes ``[1:-1, 1:-1]``: with zero velocity on the
  boundary faces, or a zonally periodic ghost ring (``h``, ``u`` and ``v``
  wrapped by the caller), mass is conserved exactly — on a one-cell ghost
  ring the seam face is then first-order upwind, and equal on both sides.

Periodic halo.  For the tendency of the physical cells to be the
*periodic* scheme's (full-order stencils across the seam, not the boundary
fallbacks), build the grid with ``H - 1`` extra columns on each side, fill
the outer ``H`` columns of every field by wrapping, and keep the physical
columns ``[H:-H]``.  Measured minimum ``H``:

=================================================================  ===
method                                                             H
=================================================================  ===
``upwind1``                                                        1
``naive``, ``upwind2``, ``upwind3``, ``weno3``, the TVD limiters   2
``weno5``, ``wenoz5``                                              3
``weno7``                                                          4
``weno9``                                                          5
=================================================================  ===

with ``wall="open"``; ``wall="closed"`` needs ``max(H, 2)``.  ``Σ A ·
tendency`` over ``[H:-H]`` is then zero to rounding, the two seam faces
carrying the same flux.  A halo *between* 1 and that ``H`` is not exactly
conservative: the two seam faces then use different boundary fallbacks
(measured ``sum(A t) / sum(A |t|) ~ 5e-7`` for weno5 with ``H = 2``).
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
    _validate_wall,
    _wall_upwind_flux,
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


def _open_wall_fluxes(
    h: Float[Array, "*Nz Ny Nx"],
    u: Float[Array, "*Nz Ny Nx"],
    v: Float[Array, "*Nz Ny Nx"],
    fe: Float[Array, "*Nz Ny Nx"],
    fn: Float[Array, "*Nz Ny Nx"],
) -> tuple[Float[Array, "*Nz Ny Nx"], Float[Array, "*Nz Ny Nx"]]:
    """Overwrite the four domain-wall faces with the ``wall="open"`` flux.

    West / east faces ``fe[..., 0]`` (``i = 1/2``) and ``fe[..., -2]``
    (``i = Nx-3/2``), south / north faces ``fn[..., 0, :]`` and
    ``fn[..., -2, :]``: first-order upwind on the caller-filled ghost ring,
    exactly as :class:`~finitevolx.Advection2D` (``_wall_upwind_flux``).
    """
    # fe[j, 1/2] = upwind(h[j, 0], h[j, 1]) · u[j, 1/2]   (west wall)
    fe = fe.at[..., 0].set(_wall_upwind_flux(u[..., 0], h[..., 0], h[..., 1]))
    # fe[j, Nx-3/2] = upwind(h[j, Nx-2], h[j, Nx-1]) · u   (east wall)
    fe = fe.at[..., -2].set(_wall_upwind_flux(u[..., -2], h[..., -2], h[..., -1]))
    # fn[1/2, i] = upwind(h[0, i], h[1, i]) · v[1/2, i]   (south wall)
    fn = fn.at[..., 0, :].set(
        _wall_upwind_flux(v[..., 0, :], h[..., 0, :], h[..., 1, :])
    )
    # fn[Ny-3/2, i] = upwind(h[Ny-2, i], h[Ny-1, i]) · v   (north wall)
    fn = fn.at[..., -2, :].set(
        _wall_upwind_flux(v[..., -2, :], h[..., -2, :], h[..., -1, :])
    )
    return fe, fn


def _spherical_flux_divergence(
    fe: Float[Array, "*Nz Ny Nx"],
    fn: Float[Array, "*Nz Ny Nx"],
    grid: SphericalGrid2D | SphericalGrid3D,
    ghost: int,
) -> Float[Array, "*Nz Ny-2g Nx-2g"]:
    """``-1/(R cosφ_T) [Δλ fe / dλ + Δφ (cosφ_V fn) / dφ]`` on ``[g:-g, g:-g]``.

    dh[j, i] = -( (fe[j, i+1/2] - fe[j, i-1/2]) / dlon
                + (cos_V[j+1/2] fn[j+1/2, i] - cos_V[j-1/2] fn[j-1/2, i]) / dlat )
               / (R cos_T[j])

    ``ghost=2`` (``wall="closed"``) never reads the wall faces; ``ghost=1``
    (``wall="open"``) reads them.  Leading (z) axes are batch axes.
    """
    c = slice(ghost, -ghost)  # the written cells
    s = slice(ghost - 1, -ghost - 1)  # one step west / south
    cos_T_c = grid.cos_lat_T[c, c]  # [Ny-2g, Nx-2g]
    cos_V_N = grid.cos_lat_V[c, c]  # north face j+1/2
    cos_V_S = grid.cos_lat_V[s, c]  # south face j-1/2
    du = (fe[..., c, c] - fe[..., c, s]) / grid.dlon
    dv = (cos_V_N * fn[..., c, c] - cos_V_S * fn[..., s, c]) / grid.dlat
    return -_safe_div_cos(du + dv, cos_T_c, grid.R)


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
        Optional land/ocean mask; every method is then land-safe,
        exactly as in :class:`Advection2D`: dry inputs are zeroed with
        ``jnp.where``, the upwind-biased methods go through
        :func:`upwind_flux` with the ``(2, 4, 6, 8, 10)`` adaptive
        stencil hierarchies pre-built once in ``__init__``, the face
        fluxes on dry faces (``mask.u`` / ``mask.v`` false) are zeroed
        before the divergence, and dry T-cells are exactly zero.
        ``None`` (default) gives the unmasked behaviour.  Per #209 Q2 a
        Cartesian ``Mask2D`` is used pending a dedicated
        ``SphericalMask2D`` follow-up.

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
        wall: str = "closed",
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
        wall : {"closed", "open"}, default "closed"
            Lateral-boundary treatment, as in :meth:`Advection2D.__call__`.
            ``"closed"`` (default, unchanged) writes ``[2:-2, 2:-2]`` and
            treats the four domain walls as no-flux.  ``"open"`` writes
            ``[1:-1, 1:-1]`` and computes the four wall-face fluxes with a
            first-order upwind reconstruction that reads the caller-filled
            ghost ring (periodic, Dirichlet, outflow); the interior faces
            keep ``method`` and the flux stays single-valued, so the
            area-weighted tendency telescopes (see the module docstring for
            the halo a periodic seam needs).  ``"open"`` is not supported
            together with a ``mask``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Advective tendency at T-points.  The ghost ring is zero, and
            with ``wall="closed"`` the outermost interior ring too.
        """
        _validate_wall(wall, self.mask)
        if self.mask is not None:
            # Zero the dry inputs before any stencil reads them (NaN-safe).
            h = jnp.where(self.mask.h, h, 0.0)  # [Ny, Nx]
            u = jnp.where(self.mask.u, u, 0.0)  # [Ny, Nx]
            v = jnp.where(self.mask.v, v, 0.0)  # [Ny, Nx]
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
        if self.mask is not None:
            # No flux through a dry face (mask.u / mask.v false).
            fe = jnp.where(self.mask.u, fe, 0.0)  # [Ny, Nx]
            fn = jnp.where(self.mask.v, fn, 0.0)  # [Ny, Nx]

        if wall == "open":
            fe, fn = _open_wall_fluxes(h, u, v, fe, fn)
            ghost = 1  # write [1:-1, 1:-1], reading the wall faces
        else:
            ghost = 2  # write [2:-2, 2:-2], never reading a wall face
        out = interior(
            _spherical_flux_divergence(fe, fn, self.grid, ghost), h, ghost=ghost
        )
        if self.mask is not None:
            out = jnp.where(self.mask.h, out, 0.0)
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
        Optional 3-D land/ocean mask; every method is then land-safe,
        as in :class:`Advection3D` (dry inputs zeroed, native 3-D
        adaptive stencil hierarchies ``(2, 4, 6, 8, 10)`` pre-built in
        ``__init__``, dry-face fluxes zeroed).  Per #209 Q2/Q3 a
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
        wall: str = "closed",
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
        wall : {"closed", "open"}, default "closed"
            Lateral-boundary treatment, as in :meth:`Advection3D.__call__`
            and :meth:`SphericalAdvection2D.__call__`: ``"closed"`` writes
            ``[1:-1, 2:-2, 2:-2]``, ``"open"`` writes ``[1:-1, 1:-1, 1:-1]``
            with first-order upwind wall-face fluxes read from the ghost
            ring.  ``"open"`` is not supported together with a ``mask``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Advective tendency at T-points.
        """
        _validate_wall(wall, self.mask)
        if self.mask is not None:
            # Zero the dry inputs before any stencil reads them (NaN-safe).
            h = jnp.where(self.mask.h, h, 0.0)  # [Nz, Ny, Nx]
            u = jnp.where(self.mask.u, u, 0.0)  # [Nz, Ny, Nx]
            v = jnp.where(self.mask.v, v, 0.0)  # [Nz, Ny, Nx]
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
        if self.mask is not None:
            # No flux through a dry face (mask.u / mask.v false).
            fe = jnp.where(self.mask.u, fe, 0.0)  # [Nz, Ny, Nx]
            fn = jnp.where(self.mask.v, fn, 0.0)  # [Nz, Ny, Nx]

        if wall == "open":
            fe, fn = _open_wall_fluxes(h, u, v, fe, fn)
            g = 1  # write [1:-1, 1:-1, 1:-1], reading the wall faces
        else:
            g = 2  # write [1:-1, 2:-2, 2:-2]
        # cos_* arrays are 2-D; they broadcast over the leading Nz axis,
        # whose ghost slabs k = 0, Nz-1 are skipped (z is a batch axis).
        horiz = _spherical_flux_divergence(fe[1:-1], fn[1:-1], self.grid, g)
        out = jnp.zeros_like(h).at[1:-1, g:-g, g:-g].set(horiz)
        if self.mask is not None:
            out = jnp.where(self.mask.h, out, 0.0)
        return out
