"""
Upwind flux with mask-aware stencil-width dispatch for non-rectangular domains.

The :func:`upwind_flux` function blends face-flux fields from multiple
reconstruction functions using mutually-exclusive masks that indicate the
highest-order stencil available at each upwind cell.  Near irregular
boundaries where only a narrow stencil fits, it falls back to lower-order
reconstructions.

Typical usage::

    from finitevolx import ArakawaCGridMask, upwind_flux
    from finitevolx import Reconstruction2D, ArakawaCGrid2D

    grid = ArakawaCGrid2D.from_interior(Ny, Nx, dy, dx)
    recon = Reconstruction2D(grid=grid)
    mask = ArakawaCGridMask.from_mask(h_mask)

    mask_hier = mask.get_adaptive_masks(direction="x", stencil_sizes=(2, 4, 6))
    rec_funcs = {2: recon.upwind1_x, 4: recon.weno3_x, 6: recon.weno5_x}

    fe = upwind_flux(q, u, dim=1, rec_funcs=rec_funcs, mask_hierarchy=mask_hier)
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def upwind_flux(
    q: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    dim: int,
    rec_funcs: dict[
        int,
        Callable[[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]],
    ],
    mask_hierarchy: dict[int, Bool[Array, "Ny Nx"]],
) -> Float[Array, "Ny Nx"]:
    """Compute upwind flux with automatic stencil fallback near boundaries.

    Blends face-flux fields produced by multiple reconstruction functions
    using **mutually-exclusive** boolean masks that indicate the largest
    supported stencil width at each upwind cell.  In the open ocean interior
    the highest-order stencil is used; near irregular coastlines or masked
    boundaries the function falls back to lower-order reconstructions
    automatically.

    The stencil decision is based on the **upwind** cell (the cell that the
    tracer is coming *from*):

    * ``dim=1`` (x / east-face flux): positive flow → upwind at ``[j, i]``;
      negative flow → upwind at ``[j, i+1]``.
    * ``dim=0`` (y / north-face flux): positive flow → upwind at ``[j, i]``;
      negative flow → upwind at ``[j+1, i]``.

    Parameters
    ----------
    q : Float[Array, "Ny Nx"]
        Cell-centred tracer field at T-points (includes ghost ring).
    u : Float[Array, "Ny Nx"]
        Face-centred transport velocity — U-points for ``dim=1``,
        V-points for ``dim=0`` (includes ghost ring).
    dim : int
        Spatial dimension: ``1`` for x (east-face flux), ``0`` for y
        (north-face flux).
    rec_funcs : dict[int, Callable]
        Mapping from stencil size (even integer, e.g. ``2``, ``4``, ``6``)
        to a reconstruction callable with signature
        ``fn(q, u) -> flux_field``.  Each callable must return a full array
        of the same shape as *q* with the ghost ring set to zero — exactly
        the convention used by all ``Reconstruction2D`` methods
        (e.g. ``recon.upwind1_x``, ``recon.weno3_x``, ``recon.weno5_x``).
    mask_hierarchy : dict[int, Bool[Array, "Ny Nx"]]
        Mutually-exclusive cell-centred boolean masks produced by
        :meth:`~finitevolx.ArakawaCGridMask.get_adaptive_masks`, keyed by
        the same stencil sizes as *rec_funcs*.  At each interior cell
        exactly one mask is ``True``; land cells and cells where even the
        smallest stencil cannot be supported are ``False`` in every mask.

    Returns
    -------
    Float[Array, "Ny Nx"]
        Blended face flux with zero ghost ring, same shape as *q*.

    Raises
    ------
    ValueError
        If *rec_funcs* is empty, *dim* is not 0 or 1, or
        *mask_hierarchy* is missing keys present in *rec_funcs*.

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
    >>> import numpy as np
    >>> from finitevolx import (
    ...     ArakawaCGrid2D,
    ...     ArakawaCGridMask,
    ...     Reconstruction2D,
    ...     upwind_flux,
    ... )
    >>> Ny, Nx = 10, 10
    >>> grid = ArakawaCGrid2D.from_interior(Ny - 2, Nx - 2, 1.0, 1.0)
    >>> recon = Reconstruction2D(grid=grid)
    >>> mask = ArakawaCGridMask.from_dimensions(Ny, Nx)
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
    if dim not in (0, 1):
        raise ValueError(f"dim must be 0 (y) or 1 (x), got {dim!r}")

    if not mask_hierarchy:
        raise ValueError("mask_hierarchy must not be empty")

    stencil_sizes = sorted(rec_funcs.keys())
    missing_masks = set(stencil_sizes) - set(mask_hierarchy.keys())
    if missing_masks:
        raise ValueError(
            "mask_hierarchy is missing masks for stencil sizes "
            f"{sorted(missing_masks)}; got keys {sorted(mask_hierarchy.keys())}"
        )

    # Compute each face-flux field: fn(q, u) → same shape as q, ghost ring = 0
    fluxes = {s: rec_funcs[s](q, u) for s in stencil_sizes}

    # Velocity sign at interior faces — used to identify the upwind cell
    # u[1:-1, 1:-1] is the face velocity at the interior faces [j, i+1/2]
    pos_flow = u[1:-1, 1:-1] >= 0.0

    def _upwind_mask(m: Bool[Array, "Ny Nx"]) -> Bool[Array, "interior"]:
        """Select mask at the upwind cell for each interior face.

        For positive flow the upwind cell is at the standard interior index
        [1:-1, 1:-1].  For negative flow it is one step in the positive
        direction (east for dim=1, north for dim=0).
        """
        if dim == 1:
            # x-direction: negative flow → upwind at [j, i+1]  →  m[1:-1, 2:]
            return jnp.where(pos_flow, m[1:-1, 1:-1], m[1:-1, 2:])
        else:
            # y-direction: negative flow → upwind at [j+1, i]  →  m[2:, 1:-1]
            return jnp.where(pos_flow, m[1:-1, 1:-1], m[2:, 1:-1])

    # Blend using mutually-exclusive masks.
    # Because exactly one mask[s] is True at each wet upwind cell, the sum
    # selects a single reconstruction tier per face:
    #   result = Σ_s  face_mask[s] * flux[s]
    interior = jnp.zeros_like(q[1:-1, 1:-1])
    for s in stencil_sizes:
        face_mask = _upwind_mask(mask_hierarchy[s])
        # Use jnp.where instead of face_mask * flux to avoid propagating
        # NaNs/Infs from unused stencil tiers into the result.
        safe_flux = jnp.where(face_mask, fluxes[s][1:-1, 1:-1], 0.0)
        interior = interior + safe_flux

    out = jnp.zeros_like(q)
    return out.at[1:-1, 1:-1].set(interior)
