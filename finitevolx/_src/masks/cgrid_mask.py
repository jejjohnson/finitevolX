"""
Arakawa C-grid mask module for finitevolX.

Provides a unified mask class for 2-D Arakawa C-grid domains with land/sea
topology, staggered face/corner masks, boundary classification, directional
stencil capability, and adaptive WENO stencil selection.

Grid layout (Arakawa & Lamb 1977)
----------------------------------
::

    y
    ^
    :           :
    w-----v-----w..
    |           |
    |           |
    u     h     u
    |           |
    |           |
    w-----v-----w..   > x

Index convention (same as AGENTS.md)
--------------------------------------
::

    h[j, i]    cell centre   at (j,     i    )
    u[j, i]    y-face        at (j-1/2, i    )   [kernel (2,1) from h]
    v[j, i]    x-face        at (j,     i-1/2)   [kernel (1,2) from h]
    w[j, i]    SW corner     at (j-1/2, i-1/2)   [kernel (2,2), lenient]
    psi[j, i]  SW corner     at (j-1/2, i-1/2)   [kernel (2,2), strict]

Staggered masks are derived from the h-mask using 2-D average pooling with
top/left zero-padding so the output shape equals the input shape:

    u[j, i]   = (h[j, i] + h[j-1, i]) / 2  > 3/4  → both y-neighbours wet
    v[j, i]   = (h[j, i] + h[j, i-1]) / 2  > 3/4  → both x-neighbours wet
    w[j, i]   = (h[j,i]+h[j-1,i]+h[j,i-1]+h[j-1,i-1]) / 4 > 1/8  → ≥1 wet
    psi[j, i] = same sum                             / 4 > 7/8  → all 4 wet

All heavy computation in the factory methods uses **numpy / scipy** (since
masks are built once, not traced through JAX JIT).  The resulting arrays are
stored as JAX arrays for use in downstream JAX computations.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import numpy as np
from scipy.ndimage import binary_dilation

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers — pure numpy/scipy, used only during construction
# ──────────────────────────────────────────────────────────────────────────────


def _pool2d_bool(h: np.ndarray, ky: int, kx: int, threshold: float) -> np.ndarray:
    """2-D average-pool of a mask with top/left zero-padding.

    Pads ``(ky-1)`` rows at the top and ``(kx-1)`` cols at the left so the
    output shape equals the input shape.

        pool[j, i] = mean(h[j-(ky-1) : j+1, i-(kx-1) : i+1]) > threshold

    Parameters
    ----------
    h : np.ndarray [Ny, Nx]
        Input mask (float values in {0, 1}).
    ky, kx : int
        Kernel height and width.
    threshold : float
        Wet/dry threshold applied to the local mean.

    Returns
    -------
    np.ndarray [Ny, Nx] bool
    """
    h_padded = np.pad(h.astype(float), ((ky - 1, 0), (kx - 1, 0)))
    total = np.zeros_like(h, dtype=float)
    for di in range(ky):
        for dj in range(kx):
            total += h_padded[di : di + h.shape[0], dj : dj + h.shape[1]]
    return total / (ky * kx) > threshold


def _dilate2d(mask: np.ndarray) -> np.ndarray:
    """Binary dilation by 1 cell (4-connectivity, zero boundary condition).

    Uses ``scipy.ndimage.binary_dilation`` with a cross structuring element.

    Parameters
    ----------
    mask : np.ndarray [Ny, Nx] bool

    Returns
    -------
    np.ndarray [Ny, Nx] bool
    """
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    return binary_dilation(mask.astype(bool), structure=struct, border_value=0)


def _count_contiguous(h: np.ndarray, axis: int, forward: bool) -> np.ndarray:
    """Count contiguous wet cells from each point along one axis.

    For each cell (j, i) the result is the number of consecutive wet cells
    starting **at** (j, i) and moving in the chosen direction.  A wet cell
    at the start counts as 1; a dry cell returns 0.

    Parameters
    ----------
    h : np.ndarray [Ny, Nx] bool
        Wet/dry mask.
    axis : int
        0 → y-direction, 1 → x-direction.
    forward : bool
        ``True``  → positive-axis direction (+x or +y).
        ``False`` → negative-axis direction (−x or −y).

    Returns
    -------
    np.ndarray [Ny, Nx] int32
    """
    h_int = np.asarray(h, dtype=np.int32)
    Ny, Nx = h_int.shape
    count = np.zeros_like(h_int)

    if axis == 1:
        if forward:
            # scan right-to-left: count[j, i] = h[j, i] * (1 + count[j, i+1])
            count[:, -1] = h_int[:, -1]
            for i in range(Nx - 2, -1, -1):
                count[:, i] = h_int[:, i] * (1 + count[:, i + 1])
        else:
            # scan left-to-right: count[j, i] = h[j, i] * (1 + count[j, i-1])
            count[:, 0] = h_int[:, 0]
            for i in range(1, Nx):
                count[:, i] = h_int[:, i] * (1 + count[:, i - 1])
    elif forward:
        # scan top-to-bottom (reversed): count[j, i] = h * (1 + count[j+1, i])
        count[-1, :] = h_int[-1, :]
        for j in range(Ny - 2, -1, -1):
            count[j, :] = h_int[j, :] * (1 + count[j + 1, :])
    else:
        # scan bottom-to-top: count[j, i] = h * (1 + count[j-1, i])
        count[0, :] = h_int[0, :]
        for j in range(1, Ny):
            count[j, :] = h_int[j, :] * (1 + count[j - 1, :])
    return count


def _make_sponge(Ny: int, Nx: int, width: int) -> np.ndarray:
    """Linear sponge ramp: 0 at domain walls, 1 at distance ≥ width inside.

    Parameters
    ----------
    Ny, Nx : int
        Grid dimensions.
    width : int
        Number of cells over which the ramp rises from 0 to 1.

    Returns
    -------
    np.ndarray [Ny, Nx] float32
    """
    ix = np.arange(Nx, dtype=np.float32)
    iy = np.arange(Ny, dtype=np.float32)
    ramp_x = np.clip(np.minimum(ix, (Nx - 1) - ix) / float(width), 0.0, 1.0)
    ramp_y = np.clip(np.minimum(iy, (Ny - 1) - iy) / float(width), 0.0, 1.0)
    return (ramp_y[:, None] * ramp_x[None, :]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# StencilCapability
# ──────────────────────────────────────────────────────────────────────────────


class StencilCapability(eqx.Module):
    """Directional count of contiguous wet neighbours for each grid cell.

    At each cell (j, i), stores the number of consecutive wet cells (including
    the cell itself) reachable before hitting a dry cell or the domain edge.

    Parameters
    ----------
    x_pos : Int[Array, "Ny Nx"]
        Count in the +x direction.
    x_neg : Int[Array, "Ny Nx"]
        Count in the −x direction.
    y_pos : Int[Array, "Ny Nx"]
        Count in the +y direction.
    y_neg : Int[Array, "Ny Nx"]
        Count in the −y direction.
    """

    x_pos: Int[Array, "Ny Nx"]
    x_neg: Int[Array, "Ny Nx"]
    y_pos: Int[Array, "Ny Nx"]
    y_neg: Int[Array, "Ny Nx"]

    @classmethod
    def from_mask(cls, h: Bool[Array, "Ny Nx"]) -> StencilCapability:
        """Build stencil capability from a wet/dry mask.

        Construction uses numpy; stored arrays are JAX int32.

        Parameters
        ----------
        h : array-like [Ny, Nx] bool
            Wet (True) / dry (False) mask.

        Returns
        -------
        StencilCapability
        """
        h_np = np.asarray(h, dtype=bool)
        return cls(
            x_pos=jnp.asarray(_count_contiguous(h_np, axis=1, forward=True)),
            x_neg=jnp.asarray(_count_contiguous(h_np, axis=1, forward=False)),
            y_pos=jnp.asarray(_count_contiguous(h_np, axis=0, forward=True)),
            y_neg=jnp.asarray(_count_contiguous(h_np, axis=0, forward=False)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# ArakawaCGridMask
# ──────────────────────────────────────────────────────────────────────────────


class ArakawaCGridMask(eqx.Module):
    """Unified Arakawa C-grid mask for a 2-D domain.

    Stores binary masks on all five Arakawa C-grid staggerings (h, u, v, w,
    psi), boundary-type flags for vorticity cells, irregular-boundary indices,
    a 4-level land/coast classification, directional stencil capability, and
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
    w : Bool[Array, "Ny Nx"]
        Corner wet mask (lenient)  [kernel (2,2) from h, threshold 1/8].
    psi : Bool[Array, "Ny Nx"]
        Corner wet mask (strict)   [kernel (2,2) from h, threshold 7/8].
    not_h, not_u, not_v, not_w, not_psi : Bool[Array, "Ny Nx"]
        Logical inverses of the corresponding masks.
    w_vertical_bound : Bool[Array, "Ny Nx"]
        Vorticity cells on a vertical (y-direction) boundary: wet w cell
        where at least one y-adjacent u-face is dry.
    w_horizontal_bound : Bool[Array, "Ny Nx"]
        Vorticity cells on a horizontal (x-direction) boundary: wet w cell
        where at least one x-adjacent v-face is dry.
    w_cornerout_bound : Bool[Array, "Ny Nx"]
        Vorticity cells at convex corners (both types of boundary).
    w_valid : Bool[Array, "Ny Nx"]
        Interior vorticity cells (wet w, all 4 adjacent faces wet).
    psi_irrbound_xids : Int[Array, "Nirr"]
        Row (j) indices of irregular-boundary psi cells in the interior
        ``[1:-1, 1:-1]``: dry psi cells that neighbour at least one wet psi
        cell in their 3×3 neighbourhood.
    psi_irrbound_yids : Int[Array, "Nirr"]
        Column (i) indices paired with ``psi_irrbound_xids``.
    classification : Int[Array, "Ny Nx"]
        4-level integer classification: 0 = land, 1 = coast, 2 = near-coast,
        3 = open ocean.
    stencil_capability : StencilCapability
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
    w: Bool[Array, "Ny Nx"]
    psi: Bool[Array, "Ny Nx"]

    # ── inverted masks ────────────────────────────────────────────────────────
    not_h: Bool[Array, "Ny Nx"]
    not_u: Bool[Array, "Ny Nx"]
    not_v: Bool[Array, "Ny Nx"]
    not_w: Bool[Array, "Ny Nx"]
    not_psi: Bool[Array, "Ny Nx"]

    # ── vorticity boundary classification ─────────────────────────────────────
    w_vertical_bound: Bool[Array, "Ny Nx"]
    w_horizontal_bound: Bool[Array, "Ny Nx"]
    w_cornerout_bound: Bool[Array, "Ny Nx"]
    w_valid: Bool[Array, "Ny Nx"]

    # ── irregular boundary indices (dynamic shape — do not use inside jit) ───
    psi_irrbound_xids: Int[Array, Nirr]
    psi_irrbound_yids: Int[Array, Nirr]

    # ── land/coast classification ─────────────────────────────────────────────
    classification: Int[Array, "Ny Nx"]

    # ── stencil capability ────────────────────────────────────────────────────
    stencil_capability: StencilCapability

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
        source : {'h', 'u', 'v', 'w', 'psi'}
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

    def _stencil_capability_for(self, source: str) -> StencilCapability:
        """Return a :class:`StencilCapability` for the given source grid.

        For non-``'h'`` sources the capability is re-computed from the stored
        staggered mask.  This should **not** be called inside a JIT-compiled
        function for non-h sources (numpy conversion required).

        Parameters
        ----------
        source : {'h', 'u', 'v', 'w', 'psi'}
        """
        grid_map = {
            "h": self.h,
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "psi": self.psi,
        }
        if source not in grid_map:
            raise ValueError(
                f"source must be one of {list(grid_map)!r}, got {source!r}"
            )
        if source == "h":
            return self.stencil_capability
        return StencilCapability.from_mask(grid_map[source])

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    def from_mask(
        cls,
        mask_hgrid: Bool[Array, "Ny Nx"],
        sponge_width: int | None = None,
        k_bottom: Array | None = None,
    ) -> ArakawaCGridMask:
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
        ArakawaCGridMask
        """
        h_np = np.asarray(mask_hgrid, dtype=bool)
        Ny, Nx = h_np.shape
        hf = h_np.astype(np.float32)

        # ── staggered masks ───────────────────────────────────────────────
        # u[j, i] = (h[j, i] + h[j-1, i]) / 2 > 3/4  (y-direction)
        u_np = _pool2d_bool(hf, ky=2, kx=1, threshold=3.0 / 4.0)
        # v[j, i] = (h[j, i] + h[j, i-1]) / 2 > 3/4  (x-direction)
        v_np = _pool2d_bool(hf, ky=1, kx=2, threshold=3.0 / 4.0)
        # w[j, i]: at least 1 of 4 SW-corner h-cells is wet  (lenient)
        w_np = _pool2d_bool(hf, ky=2, kx=2, threshold=1.0 / 8.0)
        # psi[j, i]: all 4 SW-corner h-cells are wet          (strict)
        psi_np = _pool2d_bool(hf, ky=2, kx=2, threshold=7.0 / 8.0)

        # ── vorticity boundary classification ─────────────────────────────
        # For w[j, i] (SW corner of h[j, i]):
        #   y-adjacent u-faces: u[j, i] and u[j+1, i]  (shift u up by one)
        #   x-adjacent v-faces: v[j, i] and v[j, i+1]  (shift v left by one)
        u_up = np.pad(u_np[1:, :], ((0, 1), (0, 0)))  # u[j+1, i], pad bottom
        v_left = np.pad(v_np[:, 1:], ((0, 0), (0, 1)))  # v[j, i+1], pad right

        # vertical boundary: w wet AND at least one y-adjacent u-face dry
        w_vb = w_np & (~u_np | ~u_up)
        # horizontal boundary: w wet AND at least one x-adjacent v-face dry
        w_hb = w_np & (~v_np | ~v_left)
        w_co = w_vb & w_hb  # corner-out: both
        w_va = w_np & u_np & u_up & v_np & v_left  # valid interior

        # ── irregular psi boundary indices ────────────────────────────────
        # Dry psi cells in [1:-1, 1:-1] with >=1 wet psi cell in 3x3 hood.
        psif = psi_np.astype(np.float32)
        if Ny >= 3 and Nx >= 3:
            pool3 = np.zeros((Ny - 2, Nx - 2), dtype=np.float32)
            for di in range(3):
                for dj in range(3):
                    pool3 += psif[di : Ny - 2 + di, dj : Nx - 2 + dj]
            pool3 /= 9.0
            irrbound = (~psi_np[1:-1, 1:-1]) & (pool3 > 1.0 / 18.0)
            rows, cols = np.where(irrbound)
        else:
            rows = np.empty(0, dtype=np.int32)
            cols = np.empty(0, dtype=np.int32)

        # ── land / coast classification ───────────────────────────────────
        # 0 = land, 1 = coast (ocean adj. to land), 2 = near-coast, 3 = ocean
        land = ~h_np
        land_d1 = _dilate2d(land)
        coast = h_np & land_d1  # first ring of ocean
        land_d2 = _dilate2d(land_d1)
        near_coast = h_np & land_d2 & ~coast  # second ring
        open_ocean = h_np & ~land_d2  # interior ocean

        classification = np.zeros((Ny, Nx), dtype=np.int32)
        classification[coast] = 1
        classification[near_coast] = 2
        classification[open_ocean] = 3

        # ── stencil capability ────────────────────────────────────────────
        sc = StencilCapability.from_mask(h_np)

        # ── sponge layer ──────────────────────────────────────────────────
        if sponge_width is None or sponge_width == 0:
            sponge_np = np.ones((Ny, Nx), dtype=np.float32)
        else:
            if sponge_width < 0:
                raise ValueError(
                    f"sponge_width must be non-negative; got {sponge_width!r}"
                )
            sponge_np = _make_sponge(Ny, Nx, sponge_width)

        return cls(
            h=jnp.asarray(h_np),
            u=jnp.asarray(u_np),
            v=jnp.asarray(v_np),
            w=jnp.asarray(w_np),
            psi=jnp.asarray(psi_np),
            not_h=jnp.asarray(~h_np),
            not_u=jnp.asarray(~u_np),
            not_v=jnp.asarray(~v_np),
            not_w=jnp.asarray(~w_np),
            not_psi=jnp.asarray(~psi_np),
            w_vertical_bound=jnp.asarray(w_vb),
            w_horizontal_bound=jnp.asarray(w_hb),
            w_cornerout_bound=jnp.asarray(w_co),
            w_valid=jnp.asarray(w_va),
            psi_irrbound_xids=jnp.asarray(rows.astype(np.int32)),
            psi_irrbound_yids=jnp.asarray(cols.astype(np.int32)),
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
    ) -> ArakawaCGridMask:
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
        ArakawaCGridMask
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
    ) -> ArakawaCGridMask:
        """Construct an all-ocean domain of given shape.

        Parameters
        ----------
        ny, nx : int
            Total grid dimensions (including ghost cells).
        sponge_width : int, optional
            Sponge layer width.  See :meth:`from_mask`.

        Returns
        -------
        ArakawaCGridMask
        """
        return cls.from_mask(np.ones((ny, nx), dtype=bool), sponge_width=sponge_width)


# ──────────────────────────────────────────────────────────────────────────────
# Optional visualisation
# ──────────────────────────────────────────────────────────────────────────────


def visualize_masks(
    masks: ArakawaCGridMask,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Plot the five staggered masks plus the land/coast classification.

    Requires ``matplotlib``.

    Parameters
    ----------
    masks : ArakawaCGridMask
        Mask object to visualise.
    figsize : tuple[int, int]
        Figure size passed to ``plt.subplots``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualize_masks(). "
            "Install it with: pip install matplotlib"
        ) from exc

    fields = {
        "h (centre)": masks.h,
        "u (y-face)": masks.u,
        "v (x-face)": masks.v,
        "w (corner, lenient)": masks.w,
        "psi (corner, strict)": masks.psi,
        "classification": masks.classification,
    }
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for ax, (title, data) in zip(axes.ravel(), fields.items(), strict=True):
        ax.imshow(
            np.asarray(data), origin="lower", interpolation="nearest", cmap="viridis"
        )
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("ArakawaCGridMask", fontsize=14)
    plt.tight_layout()
    plt.show()
