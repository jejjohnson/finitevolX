"""
Advection operators for Arakawa C-grids.

Computes -div(h * u_vec) at T-points using face-value reconstruction.

When a :class:`~finitevolx.Mask1D` / :class:`~finitevolx.Mask2D` /
:class:`~finitevolx.Mask3D` is supplied at construction, every method is
land-safe: dry inputs are zeroed with ``jnp.where`` before any stencil
reads them, the upwind reconstructions fall back to a lower-order stencil
near irregular boundaries (using :func:`~finitevolx.upwind_flux` as the
unified dispatch mechanism), and the face fluxes on every dry face
(wet→dry and dry→dry, ``mask.u`` / ``mask.v`` false) are zeroed before the
divergence, as in :class:`~finitevolx.Diffusion2D`.  The wet tendencies
therefore never depend on the values stored on land, and no mass crosses
the coastline.

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

# Methods whose masked face flux is blended from stencil-width tiers by
# ``upwind_flux``: every upwind-biased reconstruction.  ``naive`` (centred,
# reads only the two cells adjacent to the face) needs no tiers: zeroing the
# dry-face fluxes is enough to make it land-safe.
_MASK_DISPATCHABLE = (
    frozenset(
        {"upwind1", "upwind2", "upwind3", "weno3", "weno5", "wenoz5", "weno7", "weno9"}
    )
    | _TVD_LIMITERS
)

# The set ``uv_center_flux`` / ``uv_node_flux`` dispatch on (unchanged by
# #296, which made the advection *operators* land-safe).
_MASK_DISPATCHABLE_2D = frozenset({"weno3", "weno5", "wenoz5"}) | _TVD_LIMITERS

# Pre-built adaptive-hierarchy sizes — the widest set any dispatchable
# method needs in one direction (size 10 = WENO9's 9-cell stencil).
# Specific methods may use a subset; ``narrow_mask_hierarchy`` folds the
# unused tiers into the largest requested one at dispatch time.
_HIERARCHY_SIZES: tuple[int, ...] = (2, 4, 6, 8, 10)

# Supported lateral-wall treatments (see the ``wall`` argument on the
# advection operators).
_WALL_MODES = frozenset({"closed", "open"})


def _validate_wall(wall: str, mask: object) -> None:
    """Validate the ``wall`` argument and reject unsupported combinations."""
    if wall not in _WALL_MODES:
        raise ValueError(f"wall must be one of {sorted(_WALL_MODES)!r}; got {wall!r}")
    if wall == "open" and mask is not None:
        raise NotImplementedError(
            "wall='open' is not supported together with a mask; the open-wall "
            "ghost-flux path assumes a rectangular domain with no interior "
            "land. Use wall='closed' with the mask, or drop the mask."
        )


def _wall_upwind_flux(
    vel_wall: Float[Array, "..."],
    q_lo: Float[Array, "..."],
    q_hi: Float[Array, "..."],
) -> Float[Array, "..."]:
    """First-order upwind flux at a single domain-wall face.

    ``q_lo`` is the cell on the low-index side of the face (the upwind cell
    when ``vel_wall >= 0``) and ``q_hi`` the high-index side.  Matches the
    ``u >= 0`` convention used by the reconstruction primitives.

    Ghost-cell contract for ``wall="open"``
    ---------------------------------------
    On an inflow wall the flux carries the ghost cell **as-is**, i.e. the
    ghost is interpreted as the *exterior cell-centred* tracer value.  This is
    exact for:

    * **periodic** — the ghost is the opposite interior cell, so the seam
      flux uses the wrapped upwind cell (and the paired walls net to zero,
      conserving mass); and
    * **outflow** — zero-gradient, the interior cell is used regardless.

    For a **Dirichlet inflow** wall the ghost must therefore hold the boundary
    concentration *as a cell value* (``ghost = value``).  Note that
    :class:`~finitevolx.Dirichlet1D` instead writes the *face* reflection
    ``2*value - interior`` (chosen so a centred stencil sees ``value`` at the
    face — the right convention for :class:`~finitevolx.Diffusion2D`).  Pairing
    that atom with open-mode advection makes the inflow carry
    ``2*value - interior`` rather than ``value``; fill the ghost with the
    plain boundary value when an exact advective Dirichlet inflow is required.
    A single BC-agnostic wall flux cannot satisfy both the periodic and the
    face-Dirichlet conventions at once, so this is resolved by the ghost fill,
    not the operator.
    """
    return jnp.where(vel_wall >= 0.0, q_lo, q_hi) * vel_wall


def _tier_funcs(recon: object, method: str, axis: str) -> dict[int, Callable]:
    """Stencil-width tiers ``{size: reconstruction}`` for one direction.

    Size ``s`` is the tier :meth:`~finitevolx.Mask2D.get_adaptive_masks`
    assigns to an upwind cell with ``s // 2`` contiguous wet cells on each
    side (itself included), i.e. a centred footprint of ``s - 1`` cells:
    upwind1 needs 1 (size 2), upwind2 / upwind3 / WENO3 / the TVD limiters
    3 (size 4), WENO5 5 (size 6), WENO7 7 (size 8), WENO9 9 (size 10).  Each
    method falls back through the narrower tiers of its own family, so the
    chosen stencil never reads a dry cell.

    Parameters
    ----------
    recon : Reconstruction1D, Reconstruction2D or Reconstruction3D
        The reconstruction engine.
    method : str
        Mask-dispatchable method name (see ``_MASK_DISPATCHABLE``).
    axis : {"x", "y"}
        Direction (selects ``recon.<scheme>_x`` or ``recon.<scheme>_y``).

    Raises
    ------
    ValueError
        If *method* has no tiers (or the engine lacks the stencil, e.g.
        ``upwind2`` on :class:`Reconstruction3D`).
    """

    def rec(name: str) -> Callable:
        fn = getattr(recon, f"{name}_{axis}", None)
        if fn is None:
            raise ValueError(f"Unknown method: {method!r}")
        return fn

    if method in _TVD_LIMITERS:
        tvd = rec("tvd")
        return {
            2: rec("upwind1"),
            4: lambda q, vel, _lim=method: tvd(q, vel, limiter=_lim),
        }
    if method == "wenoz5":
        return {2: rec("upwind1"), 4: rec("wenoz3"), 6: rec("wenoz5")}
    if method in ("upwind2", "upwind3"):
        return {2: rec("upwind1"), 4: rec(method)}
    weno_ladder = ("upwind1", "weno3", "weno5", "weno7", "weno9")
    if method in weno_ladder:
        top = weno_ladder.index(method)
        return {2 * (k + 1): rec(weno_ladder[k]) for k in range(top + 1)}
    raise ValueError(f"Method {method!r} does not support mask-based stencil dispatch")


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
    rfx = _tier_funcs(recon, method, "x")
    return rfx, tuple(sorted(rfx))


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
    rfx = _tier_funcs(recon, method, "x")
    rfy = _tier_funcs(recon, method, "y")
    return rfx, rfy, tuple(sorted(rfx))


def _rec_funcs_for_method_3d(
    recon: Reconstruction3D, method: str
) -> tuple[dict[int, Callable], dict[int, Callable], tuple[int, ...]]:
    """Build stencil-hierarchy dicts for a given 3-D method name.

    Shares the same tier structure as :func:`_rec_funcs_for_method_2d`
    but uses the native 3-D reconstruction primitives from
    :class:`Reconstruction3D` (which has no ``upwind2`` / ``upwind3``).
    """
    rfx = _tier_funcs(recon, method, "x")
    rfy = _tier_funcs(recon, method, "y")
    return rfx, rfy, tuple(sorted(rfx))


class Advection1D(eqx.Module):
    """1-D advection operator.

    Parameters
    ----------
    grid : CartesianGrid1D
        The underlying 1-D grid.
    mask : Mask1D or None, optional
        Optional land/ocean mask; every method is then land-safe.  Dry
        inputs are zeroed with ``jnp.where`` (so ``NaN`` on land cannot
        leak), the upwind-biased methods go through ``upwind_flux`` with
        the pre-built adaptive stencil hierarchy (falling back to
        narrower stencils near coastlines, so no chosen stencil reads a
        dry cell), the face fluxes on dry faces (``mask.u`` false) are
        zeroed before the divergence (no flux through the coast), and
        dry T-cells are exactly zero.  ``None`` (default) matches the
        unmasked behaviour bit for bit.
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
        wall: str = "closed",
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
            ``'upwind3'``, ``'weno3'``, ``'weno5'``, ``'wenoz5'``,
            ``'weno7'``, ``'weno9'``, or a flux-limiter TVD scheme:
            ``'minmod'``, ``'van_leer'``, ``'superbee'``, ``'mc'``.
        wall : {"closed", "open"}, default "closed"
            Lateral-boundary treatment.  ``"closed"`` (default, unchanged)
            writes the tendency only on the strict interior ``[2:-2]`` and
            treats the domain walls as no-flux — the wall-adjacent interior
            cells hold stale values.  ``"open"`` writes the full interior
            ``[1:-1]`` and computes the two domain-wall face fluxes with a
            first-order upwind reconstruction that reads the ghost ring, so
            Dirichlet / outflow / periodic boundaries (applied by the caller
            to the ghost cells beforehand) drive the advective transport.
            The interior faces keep the requested high-order ``method``; the
            flux field stays single-valued, so mass is conserved exactly.
            ``"open"`` is not supported together with a ``mask``.

        Returns
        -------
        Float[Array, "Nx"]
            Advective tendency at T-points.
        """
        _validate_wall(wall, self.mask)
        if self.mask is not None:
            # Zero the dry inputs before any stencil reads them, so land
            # values (even NaN) reach neither the output nor its gradient.
            h = jnp.where(self.mask.h, h, 0.0)  # [Nx]
            u = jnp.where(self.mask.u, u, 0.0)  # [Nx]
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
        elif method == "wenoz5":
            fe = self.recon.wenoz5_x(h, u)
        elif method == "weno7":
            fe = self.recon.weno7_x(h, u)
        elif method == "weno9":
            fe = self.recon.weno9_x(h, u)
        elif method in _TVD_LIMITERS:
            fe = self.recon.tvd_x(h, u, limiter=method)
        else:
            raise ValueError(f"Unknown method: {method!r}")
        if self.mask is not None:
            # No flux through a dry face: fe[i+1/2] = 0 unless both h[i]
            # and h[i+1] are wet (mask.u), as in Diffusion2D.
            fe = jnp.where(self.mask.u, fe, 0.0)  # [Nx]

        out = jnp.zeros_like(h)
        # dh[i] = -(fe[i+1/2] - fe[i-1/2]) / dx
        # fe[i] represents the flux at the east face of cell i (at i+1/2)
        # For cell i, we need fe[i] (east) and fe[i-1] (west)
        if wall == "open":
            # Overwrite the two domain-wall faces (west face 1/2 == fe[0],
            # east face Nx-3/2 == fe[-2]) with a first-order upwind flux that
            # reads the ghost ring, then write the full interior [1:-1].
            fe = fe.at[0].set(_wall_upwind_flux(u[0], h[0], h[1]))
            fe = fe.at[-2].set(_wall_upwind_flux(u[-2], h[-2], h[-1]))
            out = out.at[1:-1].set(-(fe[1:-1] - fe[0:-2]) / self.grid.dx)
        else:
            # Only use face fluxes that are defined by the reconstruction
            # scheme, avoiding the ghost-ring entries fe[0] and fe[-1].
            out = out.at[2:-2].set(-(fe[2:-2] - fe[1:-3]) / self.grid.dx)
        if self.mask is not None:
            out = jnp.where(self.mask.h, out, 0.0)
        return out


class Advection2D(eqx.Module):
    """2-D advection operator.

    Parameters
    ----------
    grid : CartesianGrid2D
        The underlying 2-D grid.
    mask : Mask2D or None, optional
        Optional land/ocean mask; every method is then land-safe.  The
        ``(2, 4, 6, 8, 10)`` adaptive stencil hierarchies for both
        directions are pre-built once in ``__init__`` and reused on
        every call — this is the main reason the API moved from a
        per-call keyword to a class field.  Dry inputs are zeroed with
        ``jnp.where`` (so ``NaN`` on land cannot leak); the
        upwind-biased methods (everything but ``naive``) go through
        ``upwind_flux`` with the stored hierarchy narrowed to the
        method's stencil sizes, so no chosen stencil reads a dry cell;
        the face fluxes on dry faces (``mask.u`` / ``mask.v`` false) are
        zeroed before the divergence, as in :class:`Diffusion2D`, so no
        mass crosses the coast; and dry T-cells are exactly zero.
        ``None`` (default) matches the unmasked behaviour bit for bit.
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
        wall: str = "closed",
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
        wall : {"closed", "open"}, default "closed"
            Lateral-boundary treatment.  ``"closed"`` (default, unchanged)
            writes the tendency only on the strict interior ``[2:-2, 2:-2]``
            and treats all four domain walls as no-flux.  ``"open"`` writes
            the full interior ``[1:-1, 1:-1]`` and computes the four
            domain-wall face fluxes with a first-order upwind reconstruction
            that reads the ghost ring, so Dirichlet / outflow / periodic
            lateral boundaries (applied by the caller to the ghost cells
            beforehand) drive horizontal transport.  Interior faces keep the
            requested high-order ``method`` and the flux field stays
            single-valued, so mass is conserved exactly.  ``"open"`` is not
            supported together with a ``mask``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            Advective tendency at T-points.
        """
        _validate_wall(wall, self.mask)
        if self.mask is not None:
            # Zero the dry inputs before any stencil reads them, so land
            # values (even NaN) reach neither the output nor its gradient.
            h = jnp.where(self.mask.h, h, 0.0)  # [Ny, Nx]
            u = jnp.where(self.mask.u, u, 0.0)  # [Ny, Nx]
            v = jnp.where(self.mask.v, v, 0.0)  # [Ny, Nx]
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
        if self.mask is not None:
            # No flux through a dry face: fe[j, i+1/2] = 0 unless h[j, i] and
            # h[j, i+1] are both wet (mask.u), fn[j+1/2, i] likewise with
            # mask.v — the Diffusion2D face-flux masking.
            fe = jnp.where(self.mask.u, fe, 0.0)  # [Ny, Nx]
            fn = jnp.where(self.mask.v, fn, 0.0)  # [Ny, Nx]

        # dh[j, i] = -( (fe[j, i+1/2] - fe[j, i-1/2])/dx
        #             + (fn[j+1/2, i] - fn[j-1/2, i])/dy )
        # fe[j,i] is flux at east face of cell [j,i], fn[j,i] is flux at north face
        # For cell [j,i], we need fe[j,i] (east) and fe[j,i-1] (west),
        #                      and fn[j,i] (north) and fn[j-1,i] (south)
        if wall == "open":
            # Overwrite the west/east (fe cols 0, -2) and south/north
            # (fn rows 0, -2) domain-wall faces with a first-order upwind
            # flux that reads the ghost ring, then write the full interior.
            fe = fe.at[:, 0].set(_wall_upwind_flux(u[:, 0], h[:, 0], h[:, 1]))
            fe = fe.at[:, -2].set(_wall_upwind_flux(u[:, -2], h[:, -2], h[:, -1]))
            fn = fn.at[0, :].set(_wall_upwind_flux(v[0, :], h[0, :], h[1, :]))
            fn = fn.at[-2, :].set(_wall_upwind_flux(v[-2, :], h[-2, :], h[-1, :]))
            out = jnp.zeros_like(h)
            out = out.at[1:-1, 1:-1].set(
                -(
                    (fe[1:-1, 1:-1] - fe[1:-1, 0:-2]) / self.grid.dx
                    + (fn[1:-1, 1:-1] - fn[0:-2, 1:-1]) / self.grid.dy
                )
            )
        else:
            # Only use face fluxes defined by the reconstruction scheme,
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
            out = jnp.where(self.mask.h, out, 0.0)
        return out


class Advection3D(eqx.Module):
    """3-D advection operator (horizontal plane per z-level).

    Parameters
    ----------
    grid : CartesianGrid3D
        The underlying 3-D grid.
    mask : Mask3D or None, optional
        Optional 3-D land/ocean mask; every method is then land-safe.
        Pre-builds native 3-D adaptive stencil hierarchies
        ``(2, 4, 6, 8, 10)`` in both horizontal directions in
        ``__init__``.  As in :class:`Advection2D`, dry inputs are zeroed
        with ``jnp.where``, the upwind-biased methods go through
        ``upwind_flux`` with the stored 3-D hierarchies narrowed to the
        method's stencil sizes — the z axis is treated as a batch
        dimension throughout — and the face fluxes on dry faces
        (``mask.u`` / ``mask.v`` false) are zeroed before the divergence.

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
        wall: str = "closed",
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
            ``'weno5'``, ``'wenoz5'``, ``'weno7'``, ``'weno9'``, or a
            flux-limiter TVD scheme: ``'minmod'``, ``'van_leer'``,
            ``'superbee'``, ``'mc'``.  (``upwind2`` / ``upwind3`` are 1-D /
            2-D only: :class:`Reconstruction3D` has no such stencils.)
        wall : {"closed", "open"}, default "closed"
            Lateral-boundary treatment for the horizontal plane.
            ``"closed"`` (default, unchanged) writes the horizontal tendency
            only on the strict interior ``[1:-1, 2:-2, 2:-2]`` and treats the
            four lateral walls as no-flux — the wall-adjacent interior ring
            holds stale values.  ``"open"`` writes the full interior
            ``[1:-1, 1:-1, 1:-1]`` and computes the lateral domain-wall face
            fluxes with a first-order upwind reconstruction that reads the
            ghost ring, so Dirichlet / outflow / periodic lateral boundaries
            (applied by the caller to the ghost cells beforehand) drive
            horizontal transport.  Interior faces keep the requested
            high-order ``method`` and the flux field stays single-valued, so
            mass is conserved exactly.  The z axis is a batch dimension and is
            unaffected.  ``"open"`` is not supported together with a ``mask``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            Advective tendency at T-points.
        """
        _validate_wall(wall, self.mask)
        if self.mask is not None:
            # Zero the dry inputs before any stencil reads them, so land
            # values (even NaN) reach neither the output nor its gradient.
            h = jnp.where(self.mask.h, h, 0.0)  # [Nz, Ny, Nx]
            u = jnp.where(self.mask.u, u, 0.0)  # [Nz, Ny, Nx]
            v = jnp.where(self.mask.v, v, 0.0)  # [Nz, Ny, Nx]
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
        if self.mask is not None:
            # No flux through a dry face (mask.u / mask.v false).
            fe = jnp.where(self.mask.u, fe, 0.0)  # [Nz, Ny, Nx]
            fn = jnp.where(self.mask.v, fn, 0.0)  # [Nz, Ny, Nx]

        out = jnp.zeros_like(h)
        # dh[k, j, i] = -( (fe[k,j,i+1/2] - fe[k,j,i-1/2])/dx
        #                 + (fn[k,j+1/2,i] - fn[k,j-1/2,i])/dy )
        if wall == "open":
            # Overwrite the west/east (fe cols 0, -2) and south/north
            # (fn rows 0, -2) lateral wall faces with a first-order upwind
            # flux that reads the ghost ring, then write the full horizontal
            # interior.  z is a batch dimension and keeps the full [1:-1].
            fe = fe.at[:, :, 0].set(
                _wall_upwind_flux(u[:, :, 0], h[:, :, 0], h[:, :, 1])
            )
            fe = fe.at[:, :, -2].set(
                _wall_upwind_flux(u[:, :, -2], h[:, :, -2], h[:, :, -1])
            )
            fn = fn.at[:, 0, :].set(
                _wall_upwind_flux(v[:, 0, :], h[:, 0, :], h[:, 1, :])
            )
            fn = fn.at[:, -2, :].set(
                _wall_upwind_flux(v[:, -2, :], h[:, -2, :], h[:, -1, :])
            )
            out = out.at[1:-1, 1:-1, 1:-1].set(
                -(
                    (fe[1:-1, 1:-1, 1:-1] - fe[1:-1, 1:-1, 0:-2]) / self.grid.dx
                    + (fn[1:-1, 1:-1, 1:-1] - fn[1:-1, 0:-2, 1:-1]) / self.grid.dy
                )
            )
        else:
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
            out = jnp.where(self.mask.h, out, 0.0)
        return out
