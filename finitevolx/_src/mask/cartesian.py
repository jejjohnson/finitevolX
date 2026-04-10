"""Cartesian Arakawa C-grid masks (1-D / 2-D / 3-D).

Concrete mask classes for uniform-spacing Cartesian grids.  All
construction logic uses numpy/scipy (masks are built once at setup
time and stored as JAX arrays for use in JIT-traced kernels).

Same-index colocation convention (matches the grid module)::

    h[j, i]    cell centre   at (j,     i    )
    u[j, i]    y-face        at (j-1/2, i    )   [kernel (2,1) from h]
    v[j, i]    x-face        at (j,     i-1/2)   [kernel (1,2) from h]
    xy_corner[j, i]          SW corner            [kernel (2,2), lenient]
    xy_corner_strict[j, i]   SW corner            [kernel (2,2), strict]

Staggered masks are derived from the h-mask using n-D average pooling
with leading-side zero-padding so the output shape equals the input
shape::

    u[j, i]                = (h[j, i] + h[j-1, i]) / 2 > 3/4
    v[j, i]                = (h[j, i] + h[j, i-1]) / 2 > 3/4
    xy_corner[j, i]        = sum of 4 SW-corner h-cells / 4 > 1/8  (lenient)
    xy_corner_strict[j, i] = sum of 4 SW-corner h-cells / 4 > 7/8  (strict)
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import numpy as np

from finitevolx._src.mask.base import StencilCapability2D
from finitevolx._src.mask.utils import (
    _dilate,
    _make_sponge,
    _pool_bool,
)

# jaxtyping dimension variable for irregular-boundary arrays (dynamic size)
type Nirr = int


# ──────────────────────────────────────────────────────────────────────────────
# Mask2D
# ──────────────────────────────────────────────────────────────────────────────


class Mask2D(eqx.Module):
    """Unified Arakawa C-grid mask for a 2-D Cartesian domain.

    Stores binary masks on all five Arakawa C-grid staggerings (cell
    centre ``h``, x-face ``u``, y-face ``v``, xy-corner lenient
    ``xy_corner``, xy-corner strict ``xy_corner_strict``), boundary-type
    flags for the corner cells, irregular-boundary indices, a 4-level
    land/coast classification, directional stencil capability, and
    optional sponge/bathymetry arrays.

    Construct via one of the factory class-methods:

    * :meth:`from_mask`       — from a binary h-grid mask array.
    * :meth:`from_ssh`        — from an SSH field (NaN = land).
    * :meth:`from_dimensions` — all-ocean domain of given size.

    Parameters
    ----------
    h : Bool[Array, "Ny Nx"]
        Cell-centre (tracer) wet mask.
    u : Bool[Array, "Ny Nx"]
        y-face wet mask  [kernel (2,1) from h, threshold 3/4].
    v : Bool[Array, "Ny Nx"]
        x-face wet mask  [kernel (1,2) from h, threshold 3/4].
    xy_corner : Bool[Array, "Ny Nx"]
        xy-corner wet mask, lenient (≥1 of 4 surrounding h-cells wet)
        [kernel (2,2) from h, threshold 1/8].
    xy_corner_strict : Bool[Array, "Ny Nx"]
        xy-corner wet mask, strict (all 4 surrounding h-cells wet)
        [kernel (2,2) from h, threshold 7/8].
    not_h, not_u, not_v, not_xy_corner, not_xy_corner_strict : Bool[Array, "Ny Nx"]
        Logical inverses of the corresponding masks.
    xy_corner_y_wall : Bool[Array, "Ny Nx"]
        xy-corner cells on a y-direction wall: wet xy_corner cell where
        at least one y-adjacent v-face is dry.
    xy_corner_x_wall : Bool[Array, "Ny Nx"]
        xy-corner cells on an x-direction wall: wet xy_corner cell where
        at least one x-adjacent u-face is dry.
    xy_corner_convex : Bool[Array, "Ny Nx"]
        xy-corner cells at convex corners (on both x- and y-walls).
    xy_corner_valid : Bool[Array, "Ny Nx"]
        Interior xy-corner cells: wet ``xy_corner`` with all 4 adjacent
        faces wet.
    xy_corner_strict_irrbound_rows : Int[Array, "Nirr"]
        Row (j) indices of irregular-boundary ``xy_corner_strict`` cells
        in the interior ``[1:-1, 1:-1]``: dry corner cells that
        neighbour at least one wet corner cell in their 3×3 neighbourhood.
    xy_corner_strict_irrbound_cols : Int[Array, "Nirr"]
        Column (i) indices paired with ``xy_corner_strict_irrbound_rows``.
    classification : Int[Array, "Ny Nx"]
        4-level integer classification: 0 = land, 1 = coast, 2 = near-coast,
        3 = open ocean.
    stencil_capability : StencilCapability2D
        Directional contiguous-wet-cell counts on the h-grid.
    sponge : Float[Array, "Ny Nx"]
        Sponge-layer weight: 0 at domain walls, 1 in the interior.
        All-ones when no sponge width is requested.
    k_bottom : Array or None
        Optional 2-D array of vertical sea-floor indices (3-D domains).
    """

    # ── staggered masks ───────────────────────────────────────────────────────
    h: Bool[Array, "Ny Nx"]
    u: Bool[Array, "Ny Nx"]
    v: Bool[Array, "Ny Nx"]
    xy_corner: Bool[Array, "Ny Nx"]
    xy_corner_strict: Bool[Array, "Ny Nx"]

    # ── inverted masks ────────────────────────────────────────────────────────
    not_h: Bool[Array, "Ny Nx"]
    not_u: Bool[Array, "Ny Nx"]
    not_v: Bool[Array, "Ny Nx"]
    not_xy_corner: Bool[Array, "Ny Nx"]
    not_xy_corner_strict: Bool[Array, "Ny Nx"]

    # ── corner boundary classification ────────────────────────────────────────
    xy_corner_y_wall: Bool[Array, "Ny Nx"]
    xy_corner_x_wall: Bool[Array, "Ny Nx"]
    xy_corner_convex: Bool[Array, "Ny Nx"]
    xy_corner_valid: Bool[Array, "Ny Nx"]

    # ── irregular boundary indices (dynamic shape — do not use inside jit) ───
    xy_corner_strict_irrbound_rows: Int[Array, Nirr]
    xy_corner_strict_irrbound_cols: Int[Array, Nirr]

    # ── land/coast classification ─────────────────────────────────────────────
    classification: Int[Array, "Ny Nx"]

    # ── stencil capability ────────────────────────────────────────────────────
    stencil_capability: StencilCapability2D

    # ── optional arrays ───────────────────────────────────────────────────────
    sponge: Float[Array, "Ny Nx"]
    k_bottom: Array | None

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
        bnd = jnp.zeros((Ny, Nx), dtype=bool)
        bnd = bnd.at[0, :].set(True)
        bnd = bnd.at[-1, :].set(True)
        bnd = bnd.at[:, 0].set(True)
        bnd = bnd.at[:, -1].set(True)
        return bnd

    # ── adaptive WENO stencil masks ───────────────────────────────────────────

    def get_adaptive_masks(
        self,
        direction: str = "x",
        source: str = "h",
        stencil_sizes: tp.Sequence[int] = (2, 4, 6, 8, 10),
    ) -> dict[int, Bool[Array, "Ny Nx"]]:
        """Per-point adaptive stencil-size masks for WENO reconstruction.

        For each stencil size *s* in ``stencil_sizes``, returns a boolean mask
        that is ``True`` at cells where a symmetric stencil of half-width
        *s//2* is fully supported by contiguous wet neighbours.

        Stencil sizes correspond to WENO orders:

        * size 2  → 1st-order upwind  (half-width 1)
        * size 4  → WENO3             (half-width 2)
        * size 6  → WENO5             (half-width 3)
        * size 8  → WENO7             (half-width 4)
        * size 10 → WENO9             (half-width 5)

        The returned masks are **mutually exclusive** hierarchical tiers: the
        mask for size *s* is ``True`` only where *s* is the *largest* usable
        stencil.

        Parameters
        ----------
        direction : {'x', 'y'}
            Reconstruction direction.
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
        max_s = jnp.zeros(self.h.shape, dtype=jnp.int32)
        for s in sorted(stencil_sizes):
            hw = s // 2
            can_use = (cnt_pos >= hw) & (cnt_neg >= hw)
            max_s = jnp.where(can_use, s, max_s)

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
        sponge_width: int | None = None,
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a binary h-grid (cell-centre) mask.

        All intermediate computations use numpy/scipy for efficiency.
        Stored arrays are converted to JAX at the end.

        Parameters
        ----------
        mask_hgrid : array-like [Ny, Nx]
            Binary wet (1 / True) / dry (0 / False) mask at cell centres.
        sponge_width : int, optional
            Width (in grid cells) of the linear sponge ramp.  ``None``
            produces an all-ones sponge (no damping).
        k_bottom : array-like [Ny, Nx], optional
            Vertical sea-floor indices for 3-D domains.

        Returns
        -------
        Mask2D
        """
        h_np = np.asarray(mask_hgrid, dtype=bool)
        Ny, Nx = h_np.shape
        hf = h_np.astype(np.float32)

        # ── staggered masks ───────────────────────────────────────────────
        # u[j, i] = (h[j, i] + h[j-1, i]) / 2 > 3/4  (y-direction)
        u_np = _pool_bool(hf, kernel=(2, 1), threshold=3.0 / 4.0)
        # v[j, i] = (h[j, i] + h[j, i-1]) / 2 > 3/4  (x-direction)
        v_np = _pool_bool(hf, kernel=(1, 2), threshold=3.0 / 4.0)
        # xy_corner[j, i]: at least 1 of 4 SW-corner h-cells wet  (lenient)
        xy_corner_np = _pool_bool(hf, kernel=(2, 2), threshold=1.0 / 8.0)
        # xy_corner_strict[j, i]: all 4 SW-corner h-cells wet     (strict)
        xy_corner_strict_np = _pool_bool(hf, kernel=(2, 2), threshold=7.0 / 8.0)

        # ── corner boundary classification ────────────────────────────────
        # For xy_corner[j, i] at SW corner of h[j, i], the 4 adjacent
        # velocity faces are u[j,i] (east), u[j,i-1] (west), v[j,i] (north),
        # v[j-1,i] (south).
        u_west = np.pad(u_np[:, :-1], ((0, 0), (1, 0)))  # u[j, i-1]
        v_south = np.pad(v_np[:-1, :], ((1, 0), (0, 0)))  # v[j-1, i]

        # y-wall: v-face (north or south) dry
        xy_corner_y_wall = xy_corner_np & (~v_np | ~v_south)
        # x-wall: u-face (east or west) dry
        xy_corner_x_wall = xy_corner_np & (~u_np | ~u_west)
        # convex corner: both walls present
        xy_corner_convex = xy_corner_y_wall & xy_corner_x_wall
        # valid interior corner: all 4 adjacent faces wet
        xy_corner_valid = xy_corner_np & u_np & u_west & v_np & v_south

        # ── irregular xy_corner_strict boundary indices ───────────────────
        # Dry xy_corner_strict cells in [1:-1, 1:-1] with >=1 wet
        # xy_corner_strict cell in their 3x3 neighbourhood.
        psif = xy_corner_strict_np.astype(np.float32)
        if Ny >= 3 and Nx >= 3:
            pool3 = np.zeros((Ny - 2, Nx - 2), dtype=np.float32)
            for di in range(3):
                for dj in range(3):
                    pool3 += psif[di : Ny - 2 + di, dj : Nx - 2 + dj]
            pool3 /= 9.0
            irrbound = (~xy_corner_strict_np[1:-1, 1:-1]) & (pool3 > 1.0 / 18.0)
            rows, cols = np.where(irrbound)
            # Map back from interior slice to full-array coordinates.
            rows = rows + 1
            cols = cols + 1
        else:
            rows = np.empty(0, dtype=np.int32)
            cols = np.empty(0, dtype=np.int32)

        # ── land / coast classification ───────────────────────────────────
        # 0 = land, 1 = coast (ocean adj. to land), 2 = near-coast, 3 = ocean
        land = ~h_np
        land_d1 = _dilate(land)
        coast = h_np & land_d1  # first ring of ocean
        land_d2 = _dilate(land_d1)
        near_coast = h_np & land_d2 & ~coast  # second ring
        open_ocean = h_np & ~land_d2  # interior ocean

        classification = np.zeros((Ny, Nx), dtype=np.int32)
        classification[coast] = 1
        classification[near_coast] = 2
        classification[open_ocean] = 3

        # ── stencil capability ────────────────────────────────────────────
        sc = StencilCapability2D.from_mask(h_np)

        # ── sponge layer ──────────────────────────────────────────────────
        if sponge_width is None or sponge_width == 0:
            sponge_np = np.ones((Ny, Nx), dtype=np.float32)
        else:
            if sponge_width < 0:
                raise ValueError(
                    f"sponge_width must be non-negative; got {sponge_width!r}"
                )
            sponge_np = _make_sponge((Ny, Nx), sponge_width)

        return cls(
            h=jnp.asarray(h_np),
            u=jnp.asarray(u_np),
            v=jnp.asarray(v_np),
            xy_corner=jnp.asarray(xy_corner_np),
            xy_corner_strict=jnp.asarray(xy_corner_strict_np),
            not_h=jnp.asarray(~h_np),
            not_u=jnp.asarray(~u_np),
            not_v=jnp.asarray(~v_np),
            not_xy_corner=jnp.asarray(~xy_corner_np),
            not_xy_corner_strict=jnp.asarray(~xy_corner_strict_np),
            xy_corner_y_wall=jnp.asarray(xy_corner_y_wall),
            xy_corner_x_wall=jnp.asarray(xy_corner_x_wall),
            xy_corner_convex=jnp.asarray(xy_corner_convex),
            xy_corner_valid=jnp.asarray(xy_corner_valid),
            xy_corner_strict_irrbound_rows=jnp.asarray(rows.astype(np.int32)),
            xy_corner_strict_irrbound_cols=jnp.asarray(cols.astype(np.int32)),
            classification=jnp.asarray(classification),
            stencil_capability=sc,
            sponge=jnp.asarray(sponge_np),
            k_bottom=jnp.asarray(k_bottom) if k_bottom is not None else None,
        )

    @classmethod
    def from_ssh(
        cls,
        ssh: Float[Array, "Ny Nx"],
        sponge_width: int | None = None,
        k_bottom: Array | None = None,
    ) -> Mask2D:
        """Construct from a sea-surface-height field (NaN marks land).

        Parameters
        ----------
        ssh : array-like [Ny, Nx]
            SSH field; cells with ``NaN`` are treated as land.
        sponge_width : int, optional
            Sponge layer width.  See :meth:`from_mask`.
        k_bottom : array-like, optional
            Sea-floor indices.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        ssh_np = np.asarray(ssh)
        h_mask = np.isfinite(ssh_np)
        return cls.from_mask(h_mask, sponge_width=sponge_width, k_bottom=k_bottom)

    @classmethod
    def from_dimensions(
        cls,
        ny: int,
        nx: int,
        sponge_width: int | None = None,
    ) -> Mask2D:
        """Construct an all-ocean domain of given shape.

        Parameters
        ----------
        ny, nx : int
            Total grid dimensions (including ghost cells).
        sponge_width : int, optional
            Sponge layer width.  See :meth:`from_mask`.

        Returns
        -------
        Mask2D
        """
        return cls.from_mask(np.ones((ny, nx), dtype=bool), sponge_width=sponge_width)
