"""Cartesian Arakawa C-grid masks (1-D / 2-D / 3-D).

Concrete mask classes for uniform-spacing Cartesian grids.  All
construction logic uses numpy/scipy (masks are built once at setup
time and stored as NumPy arrays; JAX converts them when a kernel reads
them, and each class is a JAX pytree when JAX is importable).

Same-index colocation convention (matches the grid module's
``CartesianGrid2D`` / ``CartesianGrid3D`` positive-half-step rule)::

    h[j, i]                cell centre  at ( j,        i       )
    u[j, i]                east face    at ( j,        i + 1/2 )
    v[j, i]                north face   at ( j + 1/2,  i       )
    xy_corner[j, i]        NE corner    at ( j + 1/2,  i + 1/2 )  (lenient)
    xy_corner_strict[j, i] NE corner    at ( j + 1/2,  i + 1/2 )  (strict)

Staggered masks are derived from the h-mask using n-D average pooling
with **trailing-side** zero padding so the output shape equals the
input shape and ``mask.u[j, i]`` / ``mask.v[j, i]`` align with the
grid module's ``U[j, i]`` / ``V[j, i]`` positions::

    u[j, i]                = (h[j, i  ] + h[j,   i+1]) / 2 > 3/4
    v[j, i]                = (h[j, i  ] + h[j+1, i  ]) / 2 > 3/4
    xy_corner[j, i]        = sum of 4 NE-corner h-cells / 4 > 1/8  (lenient)
    xy_corner_strict[j, i] = sum of 4 NE-corner h-cells / 4 > 7/8  (strict)
"""

from __future__ import annotations

import typing as tp
from typing import TYPE_CHECKING, Any

from jaxtyping import Bool, Float, Int
import numpy as np

from finitevolx._src.mask.base import (
    StencilCapability1D,
    StencilCapability2D,
    StencilCapability3D,
)
from finitevolx._src.mask.utils import (
    dilate_mask,
    h_from_pooled,
    pool_bool,
)
from finitevolx._src.utils.pytree import array_namespace, pytree_dataclass

# The mask arrays are NumPy when a mask is built and JAX arrays (or tracers)
# once a transform has flattened it. ``jaxtyping.Array`` imports JAX, so it
# names the type for the type checker only.
if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# Mask1D
# ──────────────────────────────────────────────────────────────────────────────


@pytree_dataclass
class Mask1D:
    """Unified Arakawa C-grid mask for a 1-D Cartesian domain.

    Stores binary masks on the two staggerings (cell centre ``h``,
    x-face ``u``), a 4-level land/coast classification, directional
    stencil capability.  The 1-D grid has no v/xy-corner concepts.

    Construct via one of the factory class-methods:

    * :meth:`from_mask`       — from a binary h-grid mask array.
    * :meth:`from_center`     — from a float field at cell centres
      (NaN = dry).
    * :meth:`from_u_face`     — from a float field at u-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_dimensions` — all-ocean domain of given size.

    Parameters
    ----------
    h : Bool[Array, "Nx"]
        Wet mask at cell centres (T-points).  ``True`` = wet / ocean,
        ``False`` = dry / land.  This is the canonical mask that the
        other staggered masks are derived from.
    u : Bool[Array, "Nx"]
        Wet mask at x-faces (the boundaries between adjacent cell
        centres).  An x-face is wet iff *both* of its adjacent cell
        centres are wet — i.e. flow through the face is physically
        meaningful.
    classification : Int[Array, "Nx"]
        4-level integer classification: 0 = land, 1 = coast, 2 = near-coast,
        3 = open ocean.
    stencil_capability : StencilCapability1D
        Directional contiguous-wet-cell counts on the h-grid.
    """

    h: Bool[Array, "Nx"]
    u: Bool[Array, "Nx"]
    classification: Int[Array, "Nx"]
    stencil_capability: StencilCapability1D

    # ── inverted masks (derived, not stored) ─────────────────────────────────

    @property
    def not_h(self) -> Bool[Array, "Nx"]:
        """Logical inverse of ``h`` (dry where ``h`` is wet)."""
        return ~self.h

    @property
    def not_u(self) -> Bool[Array, "Nx"]:
        """Logical inverse of ``u`` (dry where ``u`` is wet)."""
        return ~self.u

    # ── Boolean accessors for land/coast classification ───────────────────────

    @property
    def ind_land(self) -> Bool[Array, "Nx"]:
        """Boolean mask: land cells (classification == 0)."""
        return self.classification == 0

    @property
    def ind_coast(self) -> Bool[Array, "Nx"]:
        """Boolean mask: coast cells (classification == 1)."""
        return self.classification == 1

    @property
    def ind_near_coast(self) -> Bool[Array, "Nx"]:
        """Boolean mask: near-coast cells (classification == 2)."""
        return self.classification == 2

    @property
    def ind_ocean(self) -> Bool[Array, "Nx"]:
        """Boolean mask: open-ocean cells (classification == 3)."""
        return self.classification == 3

    @property
    def ind_boundary(self) -> Bool[Array, "Nx"]:
        """Boolean mask: outermost domain-boundary endpoints."""
        (Nx,) = self.h.shape
        bnd = np.zeros((Nx,), dtype=bool)
        bnd[0] = True
        bnd[-1] = True
        return bnd

    # ── adaptive stencil masks ────────────────────────────────────────────────

    def get_adaptive_masks(
        self,
        stencil_sizes: tp.Sequence[int] = (2, 4, 6, 8, 10),
    ) -> dict[int, Bool[Array, "Nx"]]:
        """Per-point adaptive stencil-size masks for 1-D operators.

        For each stencil size *s*, returns a boolean mask that is ``True``
        at cells where a symmetric stencil of half-width ``s//2`` is fully
        supported by contiguous wet neighbours.  The masks are mutually
        exclusive hierarchical tiers (the mask for size *s* is True only
        where *s* is the *largest* usable stencil).

        These masks are useful for any operator that wants to fall back
        to a narrower stencil near coastlines or irregular boundaries —
        not just WENO reconstruction, but also higher-order upwind
        advection, higher-order finite-difference operators, and
        higher-order interpolation kernels.

        Parameters
        ----------
        stencil_sizes : sequence of int
            Ordered candidate stencil sizes (even integers).

        Returns
        -------
        dict[int, Bool[Array, "Nx"]]
            Mapping from stencil size to its mutually-exclusive boolean mask.
        """
        sc = self.stencil_capability
        cnt_pos, cnt_neg = sc.x_pos, sc.x_neg

        xp = array_namespace(cnt_pos, cnt_neg)
        max_s = xp.zeros(cnt_pos.shape, dtype=xp.int32)
        for s in sorted(stencil_sizes):
            hw = s // 2
            can_use = (cnt_pos >= hw) & (cnt_neg >= hw)
            max_s = xp.where(can_use, s, max_s)

        return {s: (max_s == s) for s in stencil_sizes}

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    def from_mask(
        cls,
        mask_hgrid: np.ndarray | Bool[Array, "Nx"],
    ) -> Mask1D:
        """Construct from a binary h-grid (cell-centre) mask.

        Parameters
        ----------
        mask_hgrid : array-like [Nx]
            Binary wet (1 / True) / dry (0 / False) mask at cell centres.

        Returns
        -------
        Mask1D

        Notes
        -----
        The staggered u-face mask is derived from ``mask_hgrid`` via
        :func:`pool_bool` with the following kernel/threshold pair
        (trailing-pad direction → ``u[i] = h[i] AND h[i+1]``):

        ====== ========= =========  ====================================
        Mask   Kernel    Threshold  Semantics
        ====== ========= =========  ====================================
        ``u``  ``(2,)``  ``3/4``    both adjacent h-cells wet (strict)
        ====== ========= =========  ====================================

        The land/coast classification uses two successive
        :func:`dilate_mask` passes on ``~mask_hgrid``; the stencil
        capability is built via :meth:`StencilCapability1D.from_mask`.
        """
        h_np = np.asarray(mask_hgrid, dtype=bool)
        (Nx,) = h_np.shape
        hf = h_np.astype(np.float32)

        # Staggered u-face (east face of T[i]): kernel (2,) trailing
        # → u[i] = (h[i] + h[i+1]) / 2 > 3/4 (both adjacent h-cells wet)
        u_np = pool_bool(hf, kernel=(2,), threshold=3.0 / 4.0, direction="trailing")

        # ── land / coast classification ───────────────────────────────────
        # 0 = land, 1 = coast (ocean adj. to land), 2 = near-coast, 3 = ocean
        land = ~h_np
        land_d1 = dilate_mask(land)
        coast = h_np & land_d1
        land_d2 = dilate_mask(land_d1)
        near_coast = h_np & land_d2 & ~coast
        open_ocean = h_np & ~land_d2

        classification = np.zeros(Nx, dtype=np.int8)
        classification[coast] = 1
        classification[near_coast] = 2
        classification[open_ocean] = 3

        # ── stencil capability ────────────────────────────────────────────
        sc = StencilCapability1D.from_mask(h_np)

        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "h": h_np,
            "u": u_np,
            "classification": classification,
            "stencil_capability": sc,
        }
        return cls(**fields)

    @classmethod
    def from_center(
        cls,
        field: Float[Array, "Nx"],
    ) -> Mask1D:
        """Construct from a field at cell centres (T-points).

        ``NaN`` values are treated as dry; finite values as wet.  This
        is the natural constructor for any tracer-like quantity (SSH,
        temperature, salinity, pressure, …) that lives at cell centres.

        Parameters
        ----------
        field : array-like [Nx]
            Float field at cell centres; ``NaN`` marks dry cells.

        Returns
        -------
        Mask1D
        """
        h_mask = np.isfinite(np.asarray(field))
        return cls.from_mask(h_mask)

    @classmethod
    def from_u_face(
        cls,
        field: Float[Array, "Nx"],
        mode: str = "permissive",
    ) -> Mask1D:
        """Construct from a field at u-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry u-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel ``(2,)``.

        Parameters
        ----------
        field : array-like [Nx]
            Float field at u-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  ``'permissive'`` (default) marks an
            h-cell wet if *any* of its two adjacent u-faces is wet
            (logical OR).  ``'conservative'`` requires *all* adjacent
            u-faces to be wet (logical AND).

        Returns
        -------
        Mask1D
        """
        u_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(u_mask, kernel=(2,), mode=mode, direction="trailing")
        return cls.from_mask(h_mask)

    @classmethod
    def from_dimensions(
        cls,
        nx: int,
    ) -> Mask1D:
        """Construct an all-ocean 1-D domain of given length.

        Parameters
        ----------
        nx : int
            Total grid length (including ghost cells).

        Returns
        -------
        Mask1D
        """
        return cls.from_mask(np.ones(nx, dtype=bool))


# ──────────────────────────────────────────────────────────────────────────────
# Mask2D
# ──────────────────────────────────────────────────────────────────────────────


@pytree_dataclass
class Mask2D:
    """Unified Arakawa C-grid mask for a 2-D Cartesian domain.

    Stores binary masks on all five Arakawa C-grid staggerings (cell
    centre ``h``, x-face ``u``, y-face ``v``, xy-corner lenient
    ``xy_corner``, xy-corner strict ``xy_corner_strict``), boundary-type
    flags for the corner cells, a 4-level land/coast classification,
    directional stencil capability, and an optional sea-floor index array.

    Construct via one of the factory class-methods:

    * :meth:`from_mask`       — from a binary h-grid mask array.
    * :meth:`from_center`     — from a float field at cell centres
      (NaN = dry).
    * :meth:`from_u_face`     — from a float field at u-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_v_face`     — from a float field at v-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_corner`     — from a float field at xy-corners
      (vertices), with a ``mode=`` choice of inversion strategy.
    * :meth:`from_dimensions` — all-ocean domain of given size.

    Parameters
    ----------
    h : Bool[Array, "Ny Nx"]
        Wet mask at cell centres (T-points), where tracer-like quantities
        (temperature, salinity, SSH, pressure) live.  ``True`` = wet /
        ocean, ``False`` = dry / land.  This is the canonical mask that
        the other staggered masks are derived from.
    u : Bool[Array, "Ny Nx"]
        Wet mask at x-faces (the west/east boundaries of each cell), where
        the zonal velocity ``u`` lives.  ``u[j, i]`` is the *east* face of
        ``h[j, i]`` (positive half-step / NE convention).  Wet iff *both*
        of the zonally adjacent cell centres are wet — i.e. flow through
        the face is physically meaningful.
    v : Bool[Array, "Ny Nx"]
        Wet mask at y-faces (the south/north boundaries of each cell), where
        the meridional velocity ``v`` lives.  ``v[j, i]`` is the *north*
        face of ``h[j, i]``.  Wet iff *both* of the meridionally adjacent
        cell centres are wet.
    xy_corner : Bool[Array, "Ny Nx"]
        Lenient wet mask at xy-corners (vertices).  ``True`` wherever
        *at least one* of the four surrounding cell centres is wet —
        useful for detecting any ocean presence at a corner.
    xy_corner_strict : Bool[Array, "Ny Nx"]
        Strict wet mask at xy-corners (vertices).  ``True`` only where
        *all four* surrounding cell centres are wet — useful for
        quantities (e.g. relative vorticity) that require the full
        4-point horizontal stencil to be inside the fluid.
    xy_corner_y_wall : Bool[Array, "Ny Nx"]
        xy-corner cells on a y-direction wall: wet xy_corner cell where
        at least one of the two u-faces ``u[j, i]``, ``u[j+1, i]`` meeting
        at the corner is dry.
    xy_corner_x_wall : Bool[Array, "Ny Nx"]
        xy-corner cells on an x-direction wall: wet xy_corner cell where
        at least one of the two v-faces ``v[j, i]``, ``v[j, i+1]`` meeting
        at the corner is dry.
    xy_corner_convex : Bool[Array, "Ny Nx"]
        xy-corner cells at convex corners (on both x- and y-walls).
    xy_corner_valid : Bool[Array, "Ny Nx"]
        Interior xy-corner cells: wet ``xy_corner`` with all 4 adjacent
        faces wet.
    classification : Int[Array, "Ny Nx"]
        4-level integer classification: 0 = land, 1 = coast, 2 = near-coast,
        3 = open ocean.
    stencil_capability : StencilCapability2D
        Directional contiguous-wet-cell counts on the h-grid.
    k_bottom : Array or None
        Optional 2-D array of vertical sea-floor indices (3-D domains).
    """

    # ── staggered masks ───────────────────────────────────────────────────────
    h: Bool[Array, "Ny Nx"]
    u: Bool[Array, "Ny Nx"]
    v: Bool[Array, "Ny Nx"]
    xy_corner: Bool[Array, "Ny Nx"]
    xy_corner_strict: Bool[Array, "Ny Nx"]

    # ── corner boundary classification ────────────────────────────────────────
    xy_corner_y_wall: Bool[Array, "Ny Nx"]
    xy_corner_x_wall: Bool[Array, "Ny Nx"]
    xy_corner_convex: Bool[Array, "Ny Nx"]
    xy_corner_valid: Bool[Array, "Ny Nx"]

    # ── land/coast classification ─────────────────────────────────────────────
    classification: Int[Array, "Ny Nx"]

    # ── stencil capability ────────────────────────────────────────────────────
    stencil_capability: StencilCapability2D

    # ── optional arrays ───────────────────────────────────────────────────────
    k_bottom: Array | None = None

    # ── inverted masks (derived, not stored) ─────────────────────────────────

    @property
    def not_h(self) -> Bool[Array, "Ny Nx"]:
        """Logical inverse of ``h`` (dry where ``h`` is wet)."""
        return ~self.h

    @property
    def not_u(self) -> Bool[Array, "Ny Nx"]:
        """Logical inverse of ``u`` (dry where ``u`` is wet)."""
        return ~self.u

    @property
    def not_v(self) -> Bool[Array, "Ny Nx"]:
        """Logical inverse of ``v`` (dry where ``v`` is wet)."""
        return ~self.v

    @property
    def not_xy_corner(self) -> Bool[Array, "Ny Nx"]:
        """Logical inverse of ``xy_corner`` (dry where ``xy_corner`` is wet)."""
        return ~self.xy_corner

    @property
    def not_xy_corner_strict(self) -> Bool[Array, "Ny Nx"]:
        """Logical inverse of ``xy_corner_strict`` (dry where ``xy_corner_strict`` is wet)."""
        return ~self.xy_corner_strict

    # ── Boolean accessors for land/coast classification ───────────────────────

    @property
    def ind_land(self) -> Bool[Array, "Ny Nx"]:
        """Boolean mask: land cells (classification == 0)."""
        return self.classification == 0

    @property
    def ind_coast(self) -> Bool[Array, "Ny Nx"]:
        """Boolean mask: coast cells (classification == 1)."""
        return self.classification == 1

    @property
    def ind_near_coast(self) -> Bool[Array, "Ny Nx"]:
        """Boolean mask: near-coast cells (classification == 2)."""
        return self.classification == 2

    @property
    def ind_ocean(self) -> Bool[Array, "Ny Nx"]:
        """Boolean mask: open-ocean cells (classification == 3)."""
        return self.classification == 3

    @property
    def ind_boundary(self) -> Bool[Array, "Ny Nx"]:
        """Boolean mask: outermost domain-boundary ring."""
        Ny, Nx = self.h.shape
        bnd = np.zeros((Ny, Nx), dtype=bool)
        bnd[0, :] = True
        bnd[-1, :] = True
        bnd[:, 0] = True
        bnd[:, -1] = True
        return bnd

    # ── adaptive stencil masks ────────────────────────────────────────────────

    def get_adaptive_masks(
        self,
        direction: str = "x",
        source: str = "h",
        stencil_sizes: tp.Sequence[int] = (2, 4, 6, 8, 10),
    ) -> dict[int, Bool[Array, "Ny Nx"]]:
        """Per-point adaptive stencil-size masks for 2-D operators.

        For each stencil size *s* in ``stencil_sizes``, returns a boolean mask
        that is ``True`` at cells where a symmetric stencil of half-width
        *s//2* is fully supported by contiguous wet neighbours.

        These masks are useful for any operator that wants to fall back
        to a narrower stencil near coastlines or irregular boundaries.
        Common consumers include:

        * **WENO reconstruction**: size 2 → upwind1, 4 → WENO3, 6 → WENO5,
          8 → WENO7, 10 → WENO9 (half-widths 1 / 2 / 3 / 4 / 5).
        * **Higher-order upwind advection** (e.g. 3rd / 5th-order upwind).
        * **Higher-order centred finite-difference** operators (4th / 6th /
          8th-order Laplacians, gradients, etc.).
        * **Higher-order interpolation** kernels (cubic / quintic
          T↔face / T↔corner interpolations).

        The returned masks are **mutually exclusive** hierarchical tiers: the
        mask for size *s* is ``True`` only where *s* is the *largest* usable
        stencil.

        Parameters
        ----------
        direction : {'x', 'y'}
            Stencil direction.
        source : {'h', 'u', 'v', 'xy_corner', 'xy_corner_strict'}
            Source grid whose stencil capability to use.
        stencil_sizes : sequence of int
            Ordered candidate stencil sizes (even integers).

        Returns
        -------
        dict[int, Bool[Array, "Ny Nx"]]
            Mapping from stencil size to its mutually-exclusive boolean mask.
        """
        sc = self._stencil_capability_for(source)
        if direction == "x":
            cnt_pos, cnt_neg = sc.x_pos, sc.x_neg
        elif direction == "y":
            cnt_pos, cnt_neg = sc.y_pos, sc.y_neg
        else:
            raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")

        # Maximum usable stencil size at each point
        xp = array_namespace(cnt_pos, cnt_neg)
        max_s = xp.zeros(cnt_pos.shape, dtype=xp.int32)
        for s in sorted(stencil_sizes):
            hw = s // 2
            can_use = (cnt_pos >= hw) & (cnt_neg >= hw)
            max_s = xp.where(can_use, s, max_s)

        # Mutually-exclusive masks
        return {s: (max_s == s) for s in stencil_sizes}

    def _stencil_capability_for(self, source: str) -> StencilCapability2D:
        """Return a :class:`StencilCapability2D` for the given source grid.

        For non-``'h'`` sources the capability is re-computed from the stored
        staggered mask.  This should **not** be called inside a JIT-compiled
        function for non-h sources (numpy conversion required).

        Parameters
        ----------
        source : {'h', 'u', 'v', 'xy_corner', 'xy_corner_strict'}
        """
        grid_map = {
            "h": self.h,
            "u": self.u,
            "v": self.v,
            "xy_corner": self.xy_corner,
            "xy_corner_strict": self.xy_corner_strict,
        }
        if source not in grid_map:
            raise ValueError(
                f"source must be one of {list(grid_map)!r}, got {source!r}"
            )
        if source == "h":
            return self.stencil_capability
        return StencilCapability2D.from_mask(grid_map[source])

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    def from_mask(
        cls,
        mask_hgrid: np.ndarray | Bool[Array, "Ny Nx"],
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a binary h-grid (cell-centre) mask.

        All intermediate computations use numpy/scipy for efficiency.
        Stored arrays are converted to JAX at the end.

        Parameters
        ----------
        mask_hgrid : array-like [Ny, Nx]
            Binary wet (1 / True) / dry (0 / False) mask at cell centres.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices for 3-D domains.

        Returns
        -------
        Mask2D

        Notes
        -----
        The four staggered masks are derived from ``mask_hgrid`` via
        :func:`pool_bool` with the following kernel/threshold pairs
        (trailing-pad direction → positive half-step convention):

        ==================== ========== =========  ====================================
        Mask                 Kernel     Threshold  Semantics
        ==================== ========== =========  ====================================
        ``u`` (east face)    ``(1, 2)`` ``3/4``    ``h[j, i] AND h[j, i+1]``
        ``v`` (north face)   ``(2, 1)`` ``3/4``    ``h[j, i] AND h[j+1, i]``
        ``xy_corner``        ``(2, 2)`` ``1/8``    ≥ 1 of 4 NE-corner h-cells wet
        ``xy_corner_strict`` ``(2, 2)`` ``7/8``    all 4 NE-corner h-cells wet
        ==================== ========== =========  ====================================

        The land/coast classification uses two successive
        :func:`dilate_mask` passes on ``~mask_hgrid``; the stencil
        capability is built via :meth:`StencilCapability2D.from_mask`.
        """
        h_np = np.asarray(mask_hgrid, dtype=bool)
        Ny, Nx = h_np.shape
        hf = h_np.astype(np.float32)

        # ── staggered masks ───────────────────────────────────────────────
        # All staggered masks use the trailing-pad (positive half-step)
        # convention, matching the grid module's same-index rule:
        # U[j, i] at east face (i+1/2), V[j, i] at north face (j+1/2),
        # X[j, i] at NE corner (i+1/2, j+1/2).

        # u[j, i] = (h[j, i] + h[j, i+1]) / 2 > 3/4  (east face = U-point)
        u_np = pool_bool(hf, kernel=(1, 2), threshold=3.0 / 4.0, direction="trailing")
        # v[j, i] = (h[j, i] + h[j+1, i]) / 2 > 3/4  (north face = V-point)
        v_np = pool_bool(hf, kernel=(2, 1), threshold=3.0 / 4.0, direction="trailing")
        # xy_corner[j, i]: at least 1 of 4 NE-corner h-cells wet  (lenient)
        xy_corner_np = pool_bool(
            hf, kernel=(2, 2), threshold=1.0 / 8.0, direction="trailing"
        )
        # xy_corner_strict[j, i]: all 4 NE-corner h-cells wet     (strict)
        xy_corner_strict_np = pool_bool(
            hf, kernel=(2, 2), threshold=7.0 / 8.0, direction="trailing"
        )

        # ── corner boundary classification ────────────────────────────────
        # For xy_corner[j, i] at NE corner of h[j, i], the 4 incident
        # velocity faces are:
        #   u[j,   i] = east face of h[j,i]    → south of corner (vertical)
        #   u[j+1, i] = east face of h[j+1,i]  → north of corner (vertical)
        #   v[j, i  ] = north face of h[j,i]   → west of corner  (horizontal)
        #   v[j, i+1] = north face of h[j,i+1] → east of corner  (horizontal)
        # u_north and v_east are the +1-shifted neighbours; trailing-side
        # zero-pad supplies the implicit boundary face beyond the array.
        u_north = np.pad(u_np[1:, :], ((0, 1), (0, 0)))  # u[j+1, i]
        v_east = np.pad(v_np[:, 1:], ((0, 0), (0, 1)))  # v[j, i+1]

        # y-wall (vertical wall): one of the two u-faces (vertical lines) dry
        xy_corner_y_wall = xy_corner_np & (~u_np | ~u_north)
        # x-wall (horizontal wall): one of the two v-faces (horizontal lines) dry
        xy_corner_x_wall = xy_corner_np & (~v_np | ~v_east)
        # convex corner: both walls present
        xy_corner_convex = xy_corner_y_wall & xy_corner_x_wall
        # valid interior corner: all 4 adjacent faces wet
        xy_corner_valid = xy_corner_np & u_np & u_north & v_np & v_east

        # ── land / coast classification ───────────────────────────────────
        # 0 = land, 1 = coast (ocean adj. to land), 2 = near-coast, 3 = ocean
        land = ~h_np
        land_d1 = dilate_mask(land)
        coast = h_np & land_d1  # first ring of ocean
        land_d2 = dilate_mask(land_d1)
        near_coast = h_np & land_d2 & ~coast  # second ring
        open_ocean = h_np & ~land_d2  # interior ocean

        classification = np.zeros((Ny, Nx), dtype=np.int8)
        classification[coast] = 1
        classification[near_coast] = 2
        classification[open_ocean] = 3

        # ── stencil capability ────────────────────────────────────────────
        sc = StencilCapability2D.from_mask(h_np)

        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "h": h_np,
            "u": u_np,
            "v": v_np,
            "xy_corner": xy_corner_np,
            "xy_corner_strict": xy_corner_strict_np,
            "xy_corner_y_wall": xy_corner_y_wall,
            "xy_corner_x_wall": xy_corner_x_wall,
            "xy_corner_convex": xy_corner_convex,
            "xy_corner_valid": xy_corner_valid,
            "classification": classification,
            "stencil_capability": sc,
            "k_bottom": np.asarray(k_bottom) if k_bottom is not None else None,
        }
        return cls(**fields)

    @classmethod
    def from_center(
        cls,
        field: Float[Array, "Ny Nx"],
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a field at cell centres (T-points).

        ``NaN`` values are treated as dry; finite values as wet.  This
        is the natural constructor for any tracer-like quantity (SSH,
        temperature, salinity, pressure, …) that lives at cell centres.

        Parameters
        ----------
        field : array-like [Ny, Nx]
            Float field at cell centres; ``NaN`` marks dry cells.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        h_mask = np.isfinite(np.asarray(field))
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_u_face(
        cls,
        field: Float[Array, "Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a field at u-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry u-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel ``(1, 2)``
        (trailing-pad direction, matching the grid's positive-half-step
        convention) and the requested ``mode``; the remaining staggered
        masks are produced by :meth:`from_mask`.

        Parameters
        ----------
        field : array-like [Ny, Nx]
            Float field at u-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        u_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(u_mask, kernel=(1, 2), mode=mode, direction="trailing")
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_v_face(
        cls,
        field: Float[Array, "Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a field at v-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry v-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel ``(2, 1)``
        (trailing-pad direction) and the requested ``mode``.

        Parameters
        ----------
        field : array-like [Ny, Nx]
            Float field at v-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        v_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(v_mask, kernel=(2, 1), mode=mode, direction="trailing")
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_corner(
        cls,
        field: Float[Array, "Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a field at xy-corners (vertices).

        ``NaN`` values are treated as dry corners.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel ``(2, 2)``
        (trailing-pad direction) and the requested ``mode``.  This is
        the natural constructor for vertex-stored quantities such as
        relative vorticity.

        Parameters
        ----------
        field : array-like [Ny, Nx]
            Float field at xy-corners; ``NaN`` marks dry corners.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        c_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(c_mask, kernel=(2, 2), mode=mode, direction="trailing")
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_dimensions(
        cls,
        ny: int,
        nx: int,
    ) -> Mask2D:
        """Construct an all-ocean domain of given shape.

        Parameters
        ----------
        ny, nx : int
            Total grid dimensions (including ghost cells).

        Returns
        -------
        Mask2D
        """
        return cls.from_mask(np.ones((ny, nx), dtype=bool))

    @classmethod
    def from_bathymetry(
        cls,
        depth: np.ndarray | Float[Array, "Ny Nx"],
        min_depth: float = 0.0,
        min_area: int = 1,
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a bathymetry array on the T-points.

        ``h[j, i] = depth[j, i] > min_depth``, then wet regions with fewer
        than ``min_area`` cells (4-connected) are made dry, and the rest is
        :meth:`from_mask`. ``NaN`` depths are dry.

        This is the array-level constructor only. Reading, regridding and
        cleaning a bathymetry product (lakes, spurs, coarsening) belongs in
        ``xrtoolz.geo.bathymetry_to_mask``.

        Parameters
        ----------
        depth : array-like [Ny, Nx]
            Water depth at cell centres, positive down; zero or negative on
            land.
        min_depth : float, optional
            Cells at or shallower than this depth are dry. Default ``0.0``.
        min_area : int, optional
            Wet regions smaller than this many cells are made dry. The
            default ``1`` keeps every wet cell.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices. See :meth:`from_mask`.

        Returns
        -------
        Mask2D

        Examples
        --------
        >>> depth = np.full((6, 6), 100.0)
        >>> depth[2:4, 2:4] = -5.0  # an island
        >>> mask = Mask2D.from_bathymetry(depth, min_depth=10.0)
        >>> int(mask.h.sum())
        32
        """
        depth_np = np.asarray(depth, dtype=float)
        with np.errstate(invalid="ignore"):
            wet = depth_np > min_depth
        if min_area > 1:
            from scipy.ndimage import label

            labels, _ = label(wet)
            sizes = np.bincount(labels.ravel())
            small = sizes < min_area
            small[0] = False  # label 0 is the dry background
            wet = wet & ~small[labels]
        return cls.from_mask(wet, k_bottom=k_bottom)


# ──────────────────────────────────────────────────────────────────────────────
# Mask3D
# ──────────────────────────────────────────────────────────────────────────────


@pytree_dataclass
class Mask3D:
    """Unified Arakawa C-grid mask for a 3-D Cartesian domain.

    Stores binary masks on the six 3-D Arakawa C-grid staggerings,
    boundary-type flags for the xy-corner cells (replicated per z-level),
    a 4-level land/coast classification, directional stencil capability,
    and an optional sea-floor index array.

    Construct via one of the factory class-methods:

    * :meth:`from_mask`       — from a binary 3-D h-grid mask array.
    * :meth:`from_center`     — from a float field at cell centres
      (NaN = dry).
    * :meth:`from_u_face`     — from a float field at u-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_v_face`     — from a float field at v-faces, with a
      ``mode=`` choice of inversion strategy.
    * :meth:`from_w_face`     — from a float field at w-faces (vertical
      velocity), with a ``mode=`` choice of inversion strategy.
    * :meth:`from_corner`     — from a float field at xy-corners
      (vertices), with a ``mode=`` choice of inversion strategy.
    * :meth:`from_dimensions` — all-ocean domain of given size.

    Parameters
    ----------
    h : Bool[Array, "Nz Ny Nx"]
        Wet mask at cell centres (T-points), where tracer-like quantities
        (temperature, salinity, pressure) live.  ``True`` = wet / ocean,
        ``False`` = dry / land.  This is the canonical mask that the
        other staggered masks are derived from.
    u : Bool[Array, "Nz Ny Nx"]
        Wet mask at x-faces (the west/east boundaries of each cell), where
        the zonal velocity ``u`` lives.  ``u[k, j, i]`` is the *east* face
        of ``h[k, j, i]`` (positive half-step / NE convention).  Wet iff
        *both* of the zonally adjacent cell centres at the same z-level
        are wet.
    v : Bool[Array, "Nz Ny Nx"]
        Wet mask at y-faces (the south/north boundaries of each cell), where
        the meridional velocity ``v`` lives.  ``v[k, j, i]`` is the *north*
        face of ``h[k, j, i]``.  Wet iff *both* of the meridionally adjacent
        cell centres at the same z-level are wet.
    w : Bool[Array, "Nz Ny Nx"]
        Wet mask at z-faces (the top/bottom boundaries of each cell), where
        the vertical velocity ``w`` lives.  ``w[k, j, i]`` is the *top*
        face of cell ``(k, j, i)`` (positive half-step convention, matching
        ``u`` and ``v``).  Wet iff *both* of the vertically adjacent cell
        centres at the same horizontal position are wet.  Stored at the
        same shape as ``h`` (no extra ``Nz+1`` layer — the bottom surface
        of the lowest interior cell is implicit in the bottom ghost shell).
    xy_corner : Bool[Array, "Nz Ny Nx"]
        Lenient wet mask at xy-corners (vertices), per z-level.
        ``True`` wherever *at least one* of the four surrounding cell
        centres at the same z-level is wet — useful for detecting any
        ocean presence at a corner.
    xy_corner_strict : Bool[Array, "Nz Ny Nx"]
        Strict wet mask at xy-corners (vertices), per z-level.
        ``True`` only where *all four* surrounding cell centres at the
        same z-level are wet — useful for quantities (e.g. relative
        vorticity) that require the full 4-point horizontal stencil to
        be inside the fluid.
    xy_corner_y_wall : Bool[Array, "Nz Ny Nx"]
        xy-corner cells on a y-direction wall: wet xy_corner cell where
        at least one of the two u-faces meeting at the corner is dry
        (per z-level).
    xy_corner_x_wall : Bool[Array, "Nz Ny Nx"]
        xy-corner cells on an x-direction wall: wet xy_corner cell where
        at least one of the two v-faces meeting at the corner is dry
        (per z-level).
    xy_corner_convex : Bool[Array, "Nz Ny Nx"]
        xy-corner cells at convex corners (on both walls).
    xy_corner_valid : Bool[Array, "Nz Ny Nx"]
        Interior xy-corner cells with all 4 adjacent faces wet.
    classification : Int[Array, "Nz Ny Nx"]
        4-level integer classification: 0 = land, 1 = coast, 2 = near-coast,
        3 = open ocean.  Built from a 3-D 6-connected dilation.
    stencil_capability : StencilCapability3D
        Directional contiguous-wet-cell counts on the h-grid (x, y, z).
    k_bottom : Array or None
        Optional 2-D ``[Ny, Nx]`` array of vertical sea-floor indices.
    """

    # ── staggered masks ───────────────────────────────────────────────────────
    h: Bool[Array, "Nz Ny Nx"]
    u: Bool[Array, "Nz Ny Nx"]
    v: Bool[Array, "Nz Ny Nx"]
    w: Bool[Array, "Nz Ny Nx"]
    xy_corner: Bool[Array, "Nz Ny Nx"]
    xy_corner_strict: Bool[Array, "Nz Ny Nx"]

    # ── corner boundary classification (per z-level) ──────────────────────────
    xy_corner_y_wall: Bool[Array, "Nz Ny Nx"]
    xy_corner_x_wall: Bool[Array, "Nz Ny Nx"]
    xy_corner_convex: Bool[Array, "Nz Ny Nx"]
    xy_corner_valid: Bool[Array, "Nz Ny Nx"]

    # ── land/coast classification ─────────────────────────────────────────────
    classification: Int[Array, "Nz Ny Nx"]

    # ── stencil capability ────────────────────────────────────────────────────
    stencil_capability: StencilCapability3D

    # ── optional arrays ───────────────────────────────────────────────────────
    k_bottom: Array | None = None

    # ── inverted masks (derived, not stored) ─────────────────────────────────

    @property
    def not_h(self) -> Bool[Array, "Nz Ny Nx"]:
        """Logical inverse of ``h`` (dry where ``h`` is wet)."""
        return ~self.h

    @property
    def not_u(self) -> Bool[Array, "Nz Ny Nx"]:
        """Logical inverse of ``u`` (dry where ``u`` is wet)."""
        return ~self.u

    @property
    def not_v(self) -> Bool[Array, "Nz Ny Nx"]:
        """Logical inverse of ``v`` (dry where ``v`` is wet)."""
        return ~self.v

    @property
    def not_w(self) -> Bool[Array, "Nz Ny Nx"]:
        """Logical inverse of ``w`` (dry where ``w`` is wet)."""
        return ~self.w

    @property
    def not_xy_corner(self) -> Bool[Array, "Nz Ny Nx"]:
        """Logical inverse of ``xy_corner`` (dry where ``xy_corner`` is wet)."""
        return ~self.xy_corner

    @property
    def not_xy_corner_strict(self) -> Bool[Array, "Nz Ny Nx"]:
        """Logical inverse of ``xy_corner_strict`` (dry where ``xy_corner_strict`` is wet)."""
        return ~self.xy_corner_strict

    # ── Boolean accessors for land/coast classification ───────────────────────

    @property
    def ind_land(self) -> Bool[Array, "Nz Ny Nx"]:
        """Boolean mask: land cells (classification == 0)."""
        return self.classification == 0

    @property
    def ind_coast(self) -> Bool[Array, "Nz Ny Nx"]:
        """Boolean mask: coast cells (classification == 1)."""
        return self.classification == 1

    @property
    def ind_near_coast(self) -> Bool[Array, "Nz Ny Nx"]:
        """Boolean mask: near-coast cells (classification == 2)."""
        return self.classification == 2

    @property
    def ind_ocean(self) -> Bool[Array, "Nz Ny Nx"]:
        """Boolean mask: open-ocean cells (classification == 3)."""
        return self.classification == 3

    @property
    def ind_boundary(self) -> Bool[Array, "Nz Ny Nx"]:
        """Boolean mask: outermost domain-boundary ring (all six faces)."""
        Nz, Ny, Nx = self.h.shape
        bnd = np.zeros((Nz, Ny, Nx), dtype=bool)
        bnd[0, :, :] = True
        bnd[-1, :, :] = True
        bnd[:, 0, :] = True
        bnd[:, -1, :] = True
        bnd[:, :, 0] = True
        bnd[:, :, -1] = True
        return bnd

    # ── adaptive stencil masks ────────────────────────────────────────────────

    def get_adaptive_masks(
        self,
        direction: str = "x",
        stencil_sizes: tp.Sequence[int] = (2, 4, 6, 8, 10),
    ) -> dict[int, Bool[Array, "Nz Ny Nx"]]:
        """Per-point adaptive stencil-size masks for 3-D operators.

        For each stencil size *s*, returns a boolean mask that is ``True``
        at cells where a symmetric stencil of half-width ``s//2`` is
        fully supported by contiguous wet neighbours along ``direction``.

        These masks are useful for any operator that wants to fall back
        to a narrower stencil near coastlines or irregular boundaries —
        not just WENO reconstruction, but also higher-order upwind
        advection, higher-order finite-difference operators, and
        higher-order interpolation kernels.

        Parameters
        ----------
        direction : {'x', 'y', 'z'}
            Stencil direction.
        stencil_sizes : sequence of int
            Ordered candidate stencil sizes (even integers).

        Returns
        -------
        dict[int, Bool[Array, "Nz Ny Nx"]]
            Mapping from stencil size to its mutually-exclusive boolean mask.
        """
        sc = self.stencil_capability
        if direction == "x":
            cnt_pos, cnt_neg = sc.x_pos, sc.x_neg
        elif direction == "y":
            cnt_pos, cnt_neg = sc.y_pos, sc.y_neg
        elif direction == "z":
            cnt_pos, cnt_neg = sc.z_pos, sc.z_neg
        else:
            raise ValueError(f"direction must be 'x', 'y' or 'z', got {direction!r}")

        xp = array_namespace(cnt_pos, cnt_neg)
        max_s = xp.zeros(cnt_pos.shape, dtype=xp.int32)
        for s in sorted(stencil_sizes):
            hw = s // 2
            can_use = (cnt_pos >= hw) & (cnt_neg >= hw)
            max_s = xp.where(can_use, s, max_s)

        return {s: (max_s == s) for s in stencil_sizes}

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    def from_mask(
        cls,
        mask_hgrid: np.ndarray | Bool[Array, "Nz Ny Nx"],
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a binary 3-D h-grid (cell-centre) mask.

        Parameters
        ----------
        mask_hgrid : array-like [Nz, Ny, Nx]
            Binary wet (1 / True) / dry (0 / False) mask at cell centres.
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices (one per horizontal column).

        Returns
        -------
        Mask3D

        Notes
        -----
        The five staggered masks are derived from ``mask_hgrid`` via
        :func:`pool_bool` with the following kernel/threshold pairs
        (trailing-pad direction → positive half-step convention):

        ==================== ============= =========  =========================================
        Mask                 Kernel        Threshold  Semantics
        ==================== ============= =========  =========================================
        ``u`` (east face)    ``(1, 1, 2)`` ``3/4``    ``h[k, j, i] AND h[k, j, i+1]``
        ``v`` (north face)   ``(1, 2, 1)`` ``3/4``    ``h[k, j, i] AND h[k, j+1, i]``
        ``w`` (top z-face)   ``(2, 1, 1)`` ``3/4``    ``h[k, j, i] AND h[k+1, j, i]``
        ``xy_corner``        ``(1, 2, 2)`` ``1/8``    ≥ 1 of 4 NE-corner h-cells wet (per k)
        ``xy_corner_strict`` ``(1, 2, 2)`` ``7/8``    all 4 NE-corner h-cells wet (per k)
        ==================== ============= =========  =========================================

        The land/coast classification uses two successive
        :func:`dilate_mask` passes on ``~mask_hgrid`` with a 6-connected
        cross structuring element; the stencil capability is built via
        :meth:`StencilCapability3D.from_mask`.
        """
        h_np = np.asarray(mask_hgrid, dtype=bool)
        Nz, Ny, Nx = h_np.shape
        hf = h_np.astype(np.float32)

        # ── staggered masks ───────────────────────────────────────────────
        # All staggered masks use the trailing-pad (positive half-step)
        # convention, matching the grid module's same-index rule.

        # u[k, j, i] = (h[k, j, i] + h[k, j, i+1]) / 2 > 3/4  (east face = U-point)
        u_np = pool_bool(
            hf, kernel=(1, 1, 2), threshold=3.0 / 4.0, direction="trailing"
        )
        # v[k, j, i] = (h[k, j, i] + h[k, j+1, i]) / 2 > 3/4  (north face = V-point)
        v_np = pool_bool(
            hf, kernel=(1, 2, 1), threshold=3.0 / 4.0, direction="trailing"
        )
        # w[k, j, i] = (h[k, j, i] + h[k+1, j, i]) / 2 > 3/4  (top z-face)
        w_np = pool_bool(
            hf, kernel=(2, 1, 1), threshold=3.0 / 4.0, direction="trailing"
        )
        # xy_corner[k, j, i]: at least 1 of 4 NE-corner h-cells wet (lenient)
        xy_corner_np = pool_bool(
            hf, kernel=(1, 2, 2), threshold=1.0 / 8.0, direction="trailing"
        )
        # xy_corner_strict: all 4 NE-corner h-cells wet (strict)
        xy_corner_strict_np = pool_bool(
            hf, kernel=(1, 2, 2), threshold=7.0 / 8.0, direction="trailing"
        )

        # ── corner boundary classification (per z-level) ──────────────────
        # For xy_corner[k, j, i] at NE corner of h[k, j, i] (per z-level),
        # the 4 incident horizontal velocity faces are u[k, j, i] /
        # u[k, j+1, i] (vertical lines) and v[k, j, i] / v[k, j, i+1]
        # (horizontal lines).  Trailing-side zero-pad supplies the implicit
        # boundary face beyond the array.
        u_north = np.pad(u_np[:, 1:, :], ((0, 0), (0, 1), (0, 0)))  # u[k, j+1, i]
        v_east = np.pad(v_np[:, :, 1:], ((0, 0), (0, 0), (0, 1)))  # v[k, j, i+1]

        xy_corner_y_wall = xy_corner_np & (~u_np | ~u_north)
        xy_corner_x_wall = xy_corner_np & (~v_np | ~v_east)
        xy_corner_convex = xy_corner_y_wall & xy_corner_x_wall
        xy_corner_valid = xy_corner_np & u_np & u_north & v_np & v_east

        # ── land / coast classification (3-D 6-connected dilation) ────────
        # 0 = land, 1 = coast (ocean adj. to land), 2 = near-coast, 3 = ocean
        land = ~h_np
        land_d1 = dilate_mask(land)
        coast = h_np & land_d1
        land_d2 = dilate_mask(land_d1)
        near_coast = h_np & land_d2 & ~coast
        open_ocean = h_np & ~land_d2

        classification = np.zeros((Nz, Ny, Nx), dtype=np.int8)
        classification[coast] = 1
        classification[near_coast] = 2
        classification[open_ocean] = 3

        # ── stencil capability (3-D, includes z) ──────────────────────────
        sc = StencilCapability3D.from_mask(h_np)

        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "h": h_np,
            "u": u_np,
            "v": v_np,
            "w": w_np,
            "xy_corner": xy_corner_np,
            "xy_corner_strict": xy_corner_strict_np,
            "xy_corner_y_wall": xy_corner_y_wall,
            "xy_corner_x_wall": xy_corner_x_wall,
            "xy_corner_convex": xy_corner_convex,
            "xy_corner_valid": xy_corner_valid,
            "classification": classification,
            "stencil_capability": sc,
            "k_bottom": np.asarray(k_bottom) if k_bottom is not None else None,
        }
        return cls(**fields)

    @classmethod
    def from_center(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at cell centres (T-points).

        ``NaN`` values are treated as dry; finite values as wet.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at cell centres; ``NaN`` marks dry cells.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        h_mask = np.isfinite(np.asarray(field))
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_u_face(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at u-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry u-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel
        ``(1, 1, 2)`` (trailing-pad direction) and the requested
        ``mode``.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at u-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        u_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(
            u_mask, kernel=(1, 1, 2), mode=mode, direction="trailing"
        )
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_v_face(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at v-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry v-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel
        ``(1, 2, 1)`` (trailing-pad direction) and the requested
        ``mode``.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at v-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        v_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(
            v_mask, kernel=(1, 2, 1), mode=mode, direction="trailing"
        )
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_w_face(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at w-faces, deriving the h-grid mask.

        ``NaN`` values are treated as dry w-faces.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel
        ``(2, 1, 1)`` (trailing-pad direction) and the requested
        ``mode``.  This is the natural constructor for vertical-velocity
        fields stored at z-faces.

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at w-faces; ``NaN`` marks dry faces.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        w_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(
            w_mask, kernel=(2, 1, 1), mode=mode, direction="trailing"
        )
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_corner(
        cls,
        field: Float[Array, "Nz Ny Nx"],
        mode: str = "permissive",
        k_bottom: Array | None = None,
    ) -> Mask3D:
        """Construct from a 3-D field at xy-corners (vertices).

        ``NaN`` values are treated as dry corners.  The h-grid mask is
        then inferred via :func:`h_from_pooled` with kernel
        ``(1, 2, 2)`` (trailing-pad direction) and the requested
        ``mode``.  Corners in 3-D are per-z-level (no z-staggering on
        corners).

        Parameters
        ----------
        field : array-like [Nz, Ny, Nx]
            Float field at xy-corners; ``NaN`` marks dry corners.
        mode : {'permissive', 'conservative'}
            Inversion strategy.  See :meth:`Mask1D.from_u_face` for
            details.
        k_bottom : array-like [Ny, Nx], optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask3D
        """
        c_mask = np.isfinite(np.asarray(field))
        h_mask = h_from_pooled(
            c_mask, kernel=(1, 2, 2), mode=mode, direction="trailing"
        )
        return cls.from_mask(h_mask, k_bottom=k_bottom)

    @classmethod
    def from_dimensions(
        cls,
        nz: int,
        ny: int,
        nx: int,
    ) -> Mask3D:
        """Construct an all-ocean 3-D domain of given shape.

        Parameters
        ----------
        nz, ny, nx : int
            Total grid dimensions (including ghost cells).

        Returns
        -------
        Mask3D
        """
        return cls.from_mask(np.ones((nz, ny, nx), dtype=bool))
