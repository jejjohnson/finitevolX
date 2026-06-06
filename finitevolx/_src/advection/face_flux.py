"""
Public functional API for raw face fluxes on Arakawa C-grids.

These functions expose the intermediate face-flux arrays ``(fe, fn)`` that
:class:`~finitevolx.Advection2D` computes internally but does not return.
The face fluxes are the building blocks for custom divergence operators,
momentum advection, and RHS assembly.

Typical usage::

    from finitevolx import uv_center_flux, CartesianGrid2D

    grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
    fe, fn = uv_center_flux(h, u, v, grid)

See Also
--------
:func:`uv_node_flux` : Analogous function for node-centred (q-point) fluxes.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.advection.advection import (
    _MASK_DISPATCHABLE_2D,
    _TVD_LIMITERS,
    _rec_funcs_for_method_2d,
)
from finitevolx._src.advection.flux import upwind_flux
from finitevolx._src.advection.reconstruction import Reconstruction2D
from finitevolx._src.grid.cartesian import CartesianGrid2D
from finitevolx._src.mask import Mask2D
from finitevolx._src.operators.differentiable import smooth_abs


def uv_center_flux(
    h: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    grid: CartesianGrid2D,
    method: str = "upwind1",
    mask: Mask2D | None = None,
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
    grid : CartesianGrid2D
        Grid object (used only for creating the reconstruction engine).
    method : str
        Reconstruction method: ``'naive'``, ``'upwind1'``, ``'upwind2'``,
        ``'upwind3'``, ``'weno3'``, ``'weno5'``, ``'wenoz5'``, ``'weno7'``,
        ``'weno9'``, or a TVD limiter: ``'minmod'``, ``'van_leer'``,
        ``'superbee'``, ``'mc'``.
    mask : Mask2D | None
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
    grid: CartesianGrid2D,
    method: str = "upwind1",
    mask: Mask2D | None = None,
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
    grid : CartesianGrid2D
        Grid object (used only for creating the reconstruction engine).
    method : str
        Reconstruction method (same options as :func:`uv_center_flux`).
    mask : Mask2D | None
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


def rusanov_flux(
    q: Float[Array, "..."],
    a: Float[Array, "..."],
    axis: int = -1,
    eps: float = 1e-8,
) -> Float[Array, "..."]:
    r"""Local Lax--Friedrichs (Rusanov) numerical flux at faces along ``axis``.

    For a conserved scalar ``q`` advected at face-normal velocity ``a``, the
    Rusanov flux at the face between two adjacent cells is

    .. math::

        F = \tfrac12\, a\, (q_L + q_R) - \tfrac12\, |a|_\varepsilon\, (q_R - q_L),
        \qquad |a|_\varepsilon = \sqrt{a^2 + \varepsilon^2},

    where ``q_L``/``q_R`` are the cell values on either side of the face.  The
    dissipation term :math:`\tfrac12 |a| (q_R - q_L)` makes the flux monotone
    (first-order, no reconstruction).  This is the standard robust, AD-friendly
    fallback for the continuity/height flux: unlike WENO it has no nonlinear
    smoothness weights, so its adjoint stays well-conditioned.

    The absolute value uses the smooth surrogate :func:`smooth_abs`
    (:math:`\sqrt{a^2 + \varepsilon^2}`) so the dissipation term is
    differentiable at ``a = 0``.  Pass ``eps=0`` to recover the exact textbook
    flux with a hard :func:`jax.numpy.abs` (non-smooth at ``a = 0``).

    Parameters
    ----------
    q : Float[Array, "..."]
        Cell-centred scalar (e.g. layer thickness on the h-grid), including
        any ghost ring.  Reconstructed left/right values are the adjacent
        cells along ``axis``.
    a : Float[Array, "..."]
        Face-normal advecting velocity sampled at the ``N - 1`` faces along
        ``axis`` (i.e. broadcastable to ``q`` with one fewer element along
        ``axis``).  Following the MASSH convention, ``a`` already lives on the
        faces; no interpolation is performed here.
    axis : int, optional
        Axis along which to take the flux.  Default ``-1`` (x / east faces);
        use ``-2`` for y / north faces.
    eps : float, optional
        Smoothing floor for ``|a|``.  ``eps > 0`` (default ``1e-8``) gives the
        AD-safe variant; ``eps == 0`` gives the exact non-smooth flux.

    Returns
    -------
    Float[Array, "..."]
        Numerical flux on the ``N - 1`` interior faces along ``axis``.  This is
        a reduced-shape array (one fewer element along ``axis``), **not** a full
        ghost-padded C-grid field, so :func:`divergence_2d` (which expects full
        ``[Ny, Nx]`` inputs) does not apply to it directly — take the divergence
        by differencing the faces along the same axis instead.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from finitevolx import rusanov_flux
    >>> h = jnp.ones((6, 6))
    >>> u = jnp.ones((6, 5))  # face-normal velocity on the 5 x-faces
    >>> fx = rusanov_flux(h, u, axis=-1)
    >>> fx.shape
    (6, 5)
    """
    ndim = q.ndim
    ax = axis % ndim
    left = [slice(None)] * ndim
    right = [slice(None)] * ndim
    left[ax] = slice(None, -1)
    right[ax] = slice(1, None)
    q_l = q[tuple(left)]
    q_r = q[tuple(right)]
    speed = smooth_abs(a, eps) if eps > 0.0 else jnp.abs(a)
    return 0.5 * a * (q_l + q_r) - 0.5 * speed * (q_r - q_l)


def _compute_face_fluxes(
    recon: Reconstruction2D,
    h: Float[Array, "Ny Nx"],
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    method: str,
    mask: Mask2D | None,
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
