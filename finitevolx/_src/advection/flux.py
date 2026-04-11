"""
Upwind flux with mask-aware stencil-width dispatch for non-rectangular domains.

The :func:`upwind_flux` function blends face-flux fields from multiple
reconstruction functions using mutually-exclusive masks that indicate the
highest-order stencil available at each upwind cell.  Near irregular
boundaries where only a narrow stencil fits, it falls back to lower-order
reconstructions.

Supports 1-D, 2-D, and 3-D arrays.  The ``dim`` parameter is the axis along
which the flux is computed (the axis *perpendicular* to the face):

* 1-D: ``dim=0``.
* 2-D: ``dim=1`` (x, east face) or ``dim=0`` (y, north face).
* 3-D: ``dim=2`` (x, east face) or ``dim=1`` (y, north face).  The z axis
  is treated as a batch dimension.

Typical 2-D usage::

    from finitevolx import Mask2D, upwind_flux
    from finitevolx import Reconstruction2D, CartesianGrid2D

    grid = CartesianGrid2D.from_interior(nx_interior, ny_interior, Lx, Ly)
    recon = Reconstruction2D(grid=grid)
    mask = Mask2D.from_mask(h_mask)

    mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
    rec_funcs = {2: recon.upwind1_x, 4: recon.weno3_x, 6: recon.weno5_x}

    fe = upwind_flux(q, u, dim=1, rec_funcs=rec_funcs, mask_hierarchy=mask_hier)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def narrow_mask_hierarchy(
    hierarchy: dict[int, Bool[Array, "..."]],
    target_sizes: Sequence[int],
) -> dict[int, Bool[Array, "..."]]:
    """Fold a wide stencil-hierarchy into a narrower one by merging oversized tiers.

    The hierarchy returned by :meth:`~finitevolx.Mask2D.get_adaptive_masks`
    is a dict ``{size -> boolean_mask}`` where each cell is in the bucket
    corresponding to the **largest** stencil it can support.  If a caller
    wants to dispatch onto a narrower set of stencil sizes (e.g. a WENO3
    method with sizes ``(2, 4)`` consuming a ``(2, 4, 6)`` hierarchy
    pre-built in ``__init__``), they need to fold the 6-tier cells down
    into the 4-tier — since any cell that can support stencil 6 can also
    support stencil 4.

    Parameters
    ----------
    hierarchy : dict[int, Bool[Array, "..."]]
        Source hierarchy with all stencil sizes present.
    target_sizes : sequence of int
        Sorted subset of stencil sizes to narrow to.  The largest element
        receives all tiers at or above itself (OR-fold); intermediate and
        smaller tiers are unchanged.

    Returns
    -------
    dict[int, Bool[Array, "..."]]
        Narrowed hierarchy containing exactly ``target_sizes`` keys.
        Still mutually exclusive across the new tier set.
    """
    sizes_sorted = sorted(target_sizes)
    if not sizes_sorted:
        raise ValueError("target_sizes must not be empty")

    biggest_target = sizes_sorted[-1]
    narrowed: dict[int, Bool[Array, "..."]] = {}
    for s in sizes_sorted[:-1]:
        if s not in hierarchy:
            raise ValueError(
                f"hierarchy is missing stencil size {s}; "
                f"available: {sorted(hierarchy.keys())}"
            )
        narrowed[s] = hierarchy[s]

    # OR-fold every source tier >= biggest_target into the biggest target slot.
    fold = None
    for src_size, src_mask in hierarchy.items():
        if src_size < biggest_target:
            continue
        fold = src_mask if fold is None else (fold | src_mask)
    if fold is None:
        raise ValueError(
            f"hierarchy has no tier >= biggest target size {biggest_target}; "
            f"available: {sorted(hierarchy.keys())}"
        )
    narrowed[biggest_target] = fold
    return narrowed


def upwind_flux(
    q: Float[Array, "..."],
    u: Float[Array, "..."],
    dim: int,
    rec_funcs: dict[int, Callable[..., Float[Array, "..."]]],
    mask_hierarchy: dict[int, Bool[Array, "..."]],
) -> Float[Array, "..."]:
    """Compute upwind flux with automatic stencil fallback near boundaries.

    Blends face-flux fields produced by multiple reconstruction functions
    using **mutually-exclusive** boolean masks that indicate the largest
    supported stencil width at each upwind cell.  In the open ocean interior
    the highest-order stencil is used; near irregular coastlines or masked
    boundaries the function falls back to lower-order reconstructions
    automatically.

    Works for 1-D, 2-D, and 3-D arrays — the ``dim`` parameter is the axis
    along which the flux is computed.  For 3-D arrays the z axis is
    untouched (batch dimension).

    The stencil decision is based on the **upwind** cell (the cell that the
    tracer is coming *from*):

    * positive flow: upwind cell is at the standard interior index
      ``[1:-1, ..., 1:-1]``;
    * negative flow: upwind cell is one step in the positive direction
      along ``dim``.

    Parameters
    ----------
    q : Float[Array, "..."]
        Cell-centred tracer field at T-points (includes ghost ring).  Must
        be 1-D, 2-D, or 3-D.
    u : Float[Array, "..."]
        Face-centred transport velocity with the same shape as ``q``.
    dim : int
        Axis index of the direction perpendicular to the face.  1-D:
        ``dim=0``.  2-D: ``dim=1`` (x / east face) or ``dim=0`` (y /
        north face).  3-D: ``dim=2`` (x) or ``dim=1`` (y).
    rec_funcs : dict[int, Callable]
        Mapping from stencil size (even integer, e.g. ``2``, ``4``, ``6``)
        to a reconstruction callable with signature
        ``fn(q, u) -> flux_field``.  Each callable must return a full array
        of the same shape as *q* with the ghost ring set to zero — exactly
        the convention used by all ``Reconstruction1D/2D/3D`` methods
        (e.g. ``recon.upwind1_x``, ``recon.weno3_x``, ``recon.weno5_x``).
    mask_hierarchy : dict[int, Bool[Array, "..."]]
        Mutually-exclusive cell-centred boolean masks produced by
        :meth:`~finitevolx.Mask1D.get_adaptive_masks` /
        :meth:`~finitevolx.Mask2D.get_adaptive_masks` /
        :meth:`~finitevolx.Mask3D.get_adaptive_masks`, keyed by the
        same stencil sizes as *rec_funcs*.  At each interior cell
        exactly one mask is ``True``; land cells and cells where even the
        smallest stencil cannot be supported are ``False`` in every mask.

    Returns
    -------
    Float[Array, "..."]
        Blended face flux with zero ghost ring, same shape as *q*.

    Raises
    ------
    ValueError
        If *rec_funcs* is empty, *dim* is out of range for the input array,
        or *mask_hierarchy* is missing keys present in *rec_funcs*.

    Notes
    -----
    The blending formula for mutually-exclusive masks is::

        F[i + 1 / 2] = sum_s(M_s(i_up) * F_s(q, u)[i + 1 / 2])

    where ``M_s`` is the mask for stencil size *s*, evaluated at the upwind
    cell index ``i_up``, and ``F_s`` is the face flux computed by
    ``rec_funcs[s]``.  Because the masks are mutually exclusive and cover all
    wet interior cells, exactly one term is non-zero per face.

    This function is fully **JAX-compatible**: it uses only ``jnp.where``
    and elementwise multiplication — no Python branching on array data.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import (
    ...     CartesianGrid2D,
    ...     Mask2D,
    ...     Reconstruction2D,
    ...     upwind_flux,
    ... )
    >>> Ny, Nx = 10, 10
    >>> grid = CartesianGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
    >>> recon = Reconstruction2D(grid=grid)
    >>> mask = Mask2D.from_dimensions(Ny, Nx)
    >>> mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
    >>> rec_funcs = {2: recon.upwind1_x, 4: recon.weno3_x, 6: recon.weno5_x}
    >>> q = jnp.ones((Ny, Nx))
    >>> u = jnp.ones((Ny, Nx))
    >>> fe = upwind_flux(q, u, dim=1, rec_funcs=rec_funcs, mask_hierarchy=mask_hier)
    >>> fe.shape
    (10, 10)
    """
    if not rec_funcs:
        raise ValueError("rec_funcs must not be empty")
    ndim = q.ndim
    if not 0 <= dim < ndim:
        raise ValueError(f"dim={dim!r} is out of range for {ndim}-D input")
    if not mask_hierarchy:
        raise ValueError("mask_hierarchy must not be empty")

    stencil_sizes = sorted(rec_funcs.keys())
    missing_masks = set(stencil_sizes) - set(mask_hierarchy.keys())
    if missing_masks:
        raise ValueError(
            "mask_hierarchy is missing masks for stencil sizes "
            f"{sorted(missing_masks)}; got keys {sorted(mask_hierarchy.keys())}"
        )

    # Slice tuples generalised to the input's dimensionality.  The interior
    # slice strips one ghost ring on every axis; the shifted slice rolls
    # forward by one step along ``dim`` only (for the negative-flow branch).
    interior_slice = tuple(slice(1, -1) for _ in range(ndim))
    shifted_slice = tuple(
        slice(2, None) if ax == dim else slice(1, -1) for ax in range(ndim)
    )

    # Compute each face-flux field: fn(q, u) → same shape as q, ghost ring = 0
    fluxes = {s: rec_funcs[s](q, u) for s in stencil_sizes}

    # Velocity sign at interior faces — used to identify the upwind cell.
    pos_flow = u[interior_slice] >= 0.0

    def _upwind_mask(m: Bool[Array, "..."]) -> Bool[Array, "..."]:
        """Select mask at the upwind cell for each interior face.

        For positive flow the upwind cell is at the standard interior index
        ``[1:-1, ..., 1:-1]``.  For negative flow it is one step in the
        positive direction along ``dim``.
        """
        return jnp.where(pos_flow, m[interior_slice], m[shifted_slice])

    # Blend using mutually-exclusive masks.
    # Because exactly one mask[s] is True at each wet upwind cell, the sum
    # selects a single reconstruction tier per face:
    #   result = Σ_s  face_mask[s] * flux[s]
    interior = jnp.zeros_like(q[interior_slice])
    for s in stencil_sizes:
        face_mask = _upwind_mask(mask_hierarchy[s])
        # Use jnp.where instead of face_mask * flux to avoid propagating
        # NaNs/Infs from unused stencil tiers into the result.
        safe_flux = jnp.where(face_mask, fluxes[s][interior_slice], 0.0)
        interior = interior + safe_flux

    out = jnp.zeros_like(q)
    return out.at[interior_slice].set(interior)
