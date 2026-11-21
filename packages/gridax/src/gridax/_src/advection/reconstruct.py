"""One axis-generic upwind reconstruction (Layer 2), for every scheme and rank.

``reconstruct(q, u, axis=..., scheme=...)`` returns the advective flux
``F[i+1/2] = q[i+1/2] u[i+1/2]`` on the faces along ``axis`` of a C-grid
array with one ghost ring. The face value comes from the widest tier of
``scheme`` that fits: inside the array, and with ``mask`` on wet cells too.
Upwind reconstructions narrow at a wall rather than shift, so they never
read a cell downwind of the face. Every ``Reconstruction1D/2D/3D`` method is
a call of this function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Float
import numpy as np

from gridax._src.advection.flux import upwind_flux
from gridax._src.advection.schemes import Scheme, Tier, _Ladder
from gridax._src.operators._ghost import interior
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["reconstruct"]

_DIRECTIONS = {-1: "x", -2: "y", -3: "z"}


def reconstruct(
    q: Float[Array, "*spatial"],
    u: Float[Array, "*spatial"],
    dim: int | None = None,
    method: str | None = None,
    num_pts: int | None = None,
    *,
    axis: int | None = None,
    scheme: Scheme | None = None,
    mask: Any = None,
) -> Float[Array, "*spatial"]:
    """The upwind advective flux ``q_face * u`` on the faces along ``axis``.

    For flow towards ``+axis`` the face ``i + 1/2`` takes the cell averages
    ``q[i + lo] … q[i + hi]`` of the widest tier of ``scheme`` that fits; for
    flow towards ``-axis`` it takes their mirror ``q[i + 1 - hi] … q[i + 1 -
    lo]``. Each face takes the widest tier whose cells are inside the array
    and, when ``mask`` is given, wet (``mask.get_adaptive_masks``, as
    ``Advection*D`` uses it); the last tier, first-order upwind, always fits.

        F[i+1/2] = u[i+1/2] * R+(q[i+lo .. i+hi])        u[i+1/2] >= 0
                 = u[i+1/2] * R-(q[i+1-hi .. i+1-lo])    u[i+1/2] <  0

    Parameters
    ----------
    q : Float[Array, "*spatial"]
        Cell averages at T points, with one ghost ring on every axis.
    u : Float[Array, "*spatial"]
        The velocity normal to the faces along ``axis``, on ``q``'s shape
        (U points for x, V points for y).
    axis : int
        The axis of the faces: ``-1`` (x), ``-2`` (y) or ``-3`` (z), or the
        same as a non-negative index.
    scheme : Scheme
        ``Upwind(order)``, ``Centred()``, ``WENO(order, variant)`` or
        ``TVD(limiter)`` from ``gridax.advection``.
    mask : Mask1D, Mask2D or Mask3D, optional
        A wet mask of ``q``'s rank; the tiers then also stop at the coast.
    dim, method, num_pts
        Deprecated: the old 2-D call ``reconstruct(q, u, dim, method="weno",
        num_pts=5)`` still works for one release and warns. ``dim=1`` is
        ``axis=-1`` and ``dim=0`` is ``axis=-2``; ``method="weno"`` /
        ``"wenoz"`` / ``"linear"`` with ``num_pts`` 1, 3 or 5 is
        ``WENO(num_pts)``, ``WENO(num_pts, "z")`` or ``Upwind(num_pts)``
        (``Upwind(1)`` for ``num_pts=1``). The results are bit for bit the
        same.

    Returns
    -------
    Float[Array, "*spatial"]
        The flux on the faces along ``axis``, ``q``'s shape, with a zero
        ghost ring.

    Examples
    --------
    >>> import numpy as np
    >>> import gridax.advection as gadv
    >>> q, u = np.ones((6, 8)), np.ones((6, 8))
    >>> flux = gadv.reconstruct(q, u, axis=-1, scheme=gadv.WENO(5))
    >>> flux.shape, float(flux[2, 3])
    ((6, 8), 1.0)
    """
    if scheme is None:
        axis, scheme = _legacy(dim, method, num_pts, axis)
    elif dim is not None or method is not None or num_pts is not None:
        raise TypeError(
            "pass either axis= and scheme=, or the deprecated dim, method, num_pts"
        )
    if axis is None:
        raise TypeError("reconstruct needs axis= with scheme=")
    position = axis % q.ndim
    if mask is not None:
        return _masked(q, u, position, scheme, mask)
    xp = array_namespace(q, u)
    last = xp.moveaxis(q, position, -1)
    velocity = xp.moveaxis(u, position, -1)
    # the other axes keep their interior only; the faces along `axis` are 1 .. N-2
    crop = (*([slice(1, -1)] * (q.ndim - 1)), slice(None))
    last, velocity = last[crop], velocity[crop][..., 1:-1]
    tiers = scheme.tiers
    positive = _faces(xp, last, tiers, upwind="left")
    if scheme.upwinded:
        negative = _faces(xp, last, tiers, upwind="right")
        face = xp.where(velocity >= 0.0, positive, negative)
    else:
        face = positive
    return interior(xp.moveaxis(face * velocity, -1, position), q)


def _legacy(
    dim: int | None, method: str | None, num_pts: int | None, axis: int | None
) -> tuple[int, Scheme]:
    """Map the old ``reconstruct(q, u, dim, method, num_pts)`` call, with a warning."""
    import warnings

    from gridax._src.advection.schemes import WENO, Upwind

    method = "weno" if method is None else method
    num_pts = 5 if num_pts is None else num_pts
    if dim not in (0, 1) or axis is not None:
        raise ValueError(f"dim must be 0 (y) or 1 (x), got {dim!r}")
    if num_pts not in (1, 3, 5):
        raise ValueError(f"num_pts must be 1, 3, or 5, got {num_pts!r}")
    if method not in ("weno", "wenoz", "linear"):
        raise ValueError(
            f"method must be one of ('weno', 'wenoz', 'linear'), got {method!r}"
        )
    if num_pts == 1:
        scheme: Scheme = Upwind(1)
    elif method == "linear":
        scheme = Upwind(num_pts)  # type: ignore[arg-type]
    else:
        scheme = WENO(num_pts, "z" if method == "wenoz" else "js")  # type: ignore[arg-type]
    warnings.warn(
        "reconstruct(q, u, dim, method, num_pts) is deprecated and will be removed in"
        f" the next release: use reconstruct(q, u, axis={-1 if dim == 1 else -2},"
        f" scheme={scheme!r}) from gridax.advection.",
        DeprecationWarning,
        stacklevel=3,
    )
    return (-1 if dim == 1 else -2), scheme


def _faces(xp: Any, q: Array, tiers: tuple[Tier, ...], *, upwind: str) -> Array:
    """Face values ``i + 1/2``, ``i = 1 .. N-2``, along the last axis.

    Each face takes the first (widest) tier whose cells are in ``[0, N-1]``;
    faces with the same tier form a run, computed with one call on slices.
    """
    size = q.shape[-1]
    runs: list[tuple[int, int, int]] = []  # (tier, first face, last face + 1)
    for i in range(1, size - 1):
        fits = [
            i + lo >= 0 and i + hi <= size - 1
            for lo, hi in (_reads(tier, upwind) for tier in tiers)
        ]
        if not any(fits):
            raise ValueError(f"no tier of the scheme fits an axis of length {size}")
        k = fits.index(True)
        if runs and runs[-1][0] == k:
            runs[-1] = (k, runs[-1][1], i + 1)
        else:
            runs.append((k, i, i + 1))
    pieces = []
    for k, start, stop in runs:
        tier = tiers[k]
        lo, hi = _reads(tier, upwind)
        fn = tier.left if upwind == "left" else tier.right
        cells = [q[..., start + o : stop + o] for o in range(lo, hi + 1)]
        pieces.append(fn(*cells))
    return pieces[0] if len(pieces) == 1 else xp.concat(pieces, axis=-1)


def _reads(tier: Tier, upwind: str) -> tuple[int, int]:
    """The cells the face ``i + 1/2`` reads, relative to ``i``."""
    if upwind == "left":
        return tier.lo, tier.hi
    return 1 - tier.hi, 1 - tier.lo


def _masked(q: Array, u: Array, position: int, scheme: Scheme, mask: Any) -> Array:
    """Blend the tiers by the mask's adaptive hierarchy, as ``Advection*D`` does."""
    tiers = scheme.tiers
    sizes = tuple(sorted({tier.size for tier in tiers}))
    if q.ndim == 1:
        hierarchy = mask.get_adaptive_masks(stencil_sizes=sizes)
    else:
        hierarchy = mask.get_adaptive_masks(
            direction=_DIRECTIONS[position - q.ndim], stencil_sizes=sizes
        )
    # each size reconstructs with that rung and the narrower ones below it
    rec_funcs = {}
    for k, tier in enumerate(tiers):
        rung = _Ladder(tiers[k:])

        def rec(field: Array, vel: Array, _rung: Scheme = rung) -> Array:
            return reconstruct(field, vel, axis=position, scheme=_rung)

        rec_funcs.setdefault(tier.size, rec)
    return upwind_flux(
        q, u, dim=position, rec_funcs=rec_funcs, mask_hierarchy=hierarchy
    )
