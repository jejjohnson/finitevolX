"""
Public functional API for raw face fluxes on Arakawa C-grids.

These functions expose the intermediate face-flux arrays ``(fe, fn)`` that
:class:`~finitevolx.Advection2D` computes internally but does not return.
The face fluxes are the building blocks for custom divergence operators,
momentum advection, and RHS assembly.

Typical usage::

    from finitevolx import uv_center_flux, ArakawaCGrid2D

    grid = ArakawaCGrid2D.from_interior(nx, ny, Lx, Ly)
    fe, fn = uv_center_flux(h, u, v, grid)

See Also
--------
:func:`uv_node_flux` : Analogous function for node-centred (q-point) fluxes.
"""

from __future__ import annotations

from jaxtyping import Array, Float

from finitevolx._src.advection.advection import (
    _MASK_DISPATCHABLE_2D,
    _TVD_LIMITERS,
    _rec_funcs_for_method_2d,
)
from finitevolx._src.advection.flux import upwind_flux
from finitevolx._src.advection.reconstruction import Reconstruction2D
from finitevolx._src.grid.cgrid_mask import ArakawaCGridMask
from finitevolx._src.grid.grid import ArakawaCGrid2D


def uv_center_flux(
    h: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    grid: ArakawaCGrid2D,
    method: str = "upwind1",
    mask: ArakawaCGridMask | None = None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Compute raw face fluxes for a cell-centred scalar on a C-grid.

    Returns the east-face and north-face flux arrays ``(fe, fn)`` for the
    transport of scalar *h* by velocity ``(u, v)``.  These are the same
    intermediate quantities that :class:`~finitevolx.Advection2D` computes
    internally before taking the divergence.

    The advective tendency at T-points can be recovered as::

        dh[j, i] = -((fe[j, i] - fe[j, i - 1]) / dx + (fn[j, i] - fn[j - 1, i]) / dy)

    (using the interior indexing convention where ``fe[j, i]`` is the flux
    at the east face of cell ``[j, i]``).

    Parameters
    ----------
    h : Float[Array, "Ny Nx"]
        Scalar at T-points (cell centres).  Includes ghost ring.
    u : Float[Array, "Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "Ny Nx"]
        y-velocity at V-points.
    grid : ArakawaCGrid2D
        Grid object (used only for creating the reconstruction engine).
    method : str
        Reconstruction method: ``'naive'``, ``'upwind1'``, ``'upwind2'``,
        ``'upwind3'``, ``'weno3'``, ``'weno5'``, ``'wenoz5'``, ``'weno7'``,
        ``'weno9'``, or a TVD limiter: ``'minmod'``, ``'van_leer'``,
        ``'superbee'``, ``'mc'``.
    mask : ArakawaCGridMask | None
        When provided and *method* supports mask dispatch, stencil-width
        fallback is applied via :func:`~finitevolx.upwind_flux`.

    Returns
    -------
    fe : Float[Array, "Ny Nx"]
        East-face flux (h * u reconstructed at east faces).
    fn : Float[Array, "Ny Nx"]
        North-face flux (h * v reconstructed at north faces).
    """
    recon = Reconstruction2D(grid=grid)
    return _compute_face_fluxes(recon, h, u, v, method, mask)


def uv_node_flux(
    q: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    grid: ArakawaCGrid2D,
    method: str = "upwind1",
    mask: ArakawaCGridMask | None = None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Compute raw face fluxes for a node-centred tracer on a C-grid.

    Analogous to :func:`uv_center_flux` but for a tracer *q* that lives at
    grid nodes (vorticity / psi points) rather than cell centres.  The
    reconstruction uses the same stencil methods as ``Advection2D`` — the
    x-stencil for east faces and the y-stencil for north faces — applied
    to the q-point field.

    Parameters
    ----------
    q : Float[Array, "Ny Nx"]
        Tracer at node points (q/psi grid).  Includes ghost ring.
    u : Float[Array, "Ny Nx"]
        x-velocity at U-points.
    v : Float[Array, "Ny Nx"]
        y-velocity at V-points.
    grid : ArakawaCGrid2D
        Grid object (used only for creating the reconstruction engine).
    method : str
        Reconstruction method (same options as :func:`uv_center_flux`).
    mask : ArakawaCGridMask | None
        Optional mask for stencil-width fallback.

    Returns
    -------
    uq_flux : Float[Array, "Ny Nx"]
        East-face flux (q * u reconstructed at east faces).
    vq_flux : Float[Array, "Ny Nx"]
        North-face flux (q * v reconstructed at north faces).
    """
    recon = Reconstruction2D(grid=grid)
    return _compute_face_fluxes(recon, q, u, v, method, mask)


def _compute_face_fluxes(
    recon: Reconstruction2D,
    h: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    method: str,
    mask: ArakawaCGridMask | None,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Shared implementation for face-flux computation.

    Mirrors the dispatch logic of ``Advection2D.__call__`` but returns
    the raw ``(fe, fn)`` face fluxes instead of the divergence tendency.
    """
    # ── masked path ───────────────────────────────────────────────────
    if mask is not None and method in _MASK_DISPATCHABLE_2D:
        rfx, rfy, sizes = _rec_funcs_for_method_2d(recon, method)
        mask_x = mask.get_adaptive_masks(direction="x", stencil_sizes=sizes)
        mask_y = mask.get_adaptive_masks(direction="y", stencil_sizes=sizes)
        fe = upwind_flux(h, u, dim=1, rec_funcs=rfx, mask_hierarchy=mask_x)
        fn = upwind_flux(h, v, dim=0, rec_funcs=rfy, mask_hierarchy=mask_y)
        return fe, fn

    # ── unmasked path ─────────────────────────────────────────────────
    if method == "naive":
        fe = recon.naive_x(h, u)
        fn = recon.naive_y(h, v)
    elif method == "upwind1":
        fe = recon.upwind1_x(h, u)
        fn = recon.upwind1_y(h, v)
    elif method == "upwind2":
        fe = recon.upwind2_x(h, u)
        fn = recon.upwind2_y(h, v)
    elif method == "upwind3":
        fe = recon.upwind3_x(h, u)
        fn = recon.upwind3_y(h, v)
    elif method == "weno3":
        fe = recon.weno3_x(h, u)
        fn = recon.weno3_y(h, v)
    elif method == "weno5":
        fe = recon.weno5_x(h, u)
        fn = recon.weno5_y(h, v)
    elif method == "wenoz5":
        fe = recon.wenoz5_x(h, u)
        fn = recon.wenoz5_y(h, v)
    elif method == "weno7":
        fe = recon.weno7_x(h, u)
        fn = recon.weno7_y(h, v)
    elif method == "weno9":
        fe = recon.weno9_x(h, u)
        fn = recon.weno9_y(h, v)
    elif method in _TVD_LIMITERS:
        fe = recon.tvd_x(h, u, limiter=method)
        fn = recon.tvd_y(h, v, limiter=method)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return fe, fn
