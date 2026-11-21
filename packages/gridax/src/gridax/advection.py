"""Advection: upwind reconstruction schemes and one axis-generic reconstruct.

``reconstruct(q, u, axis=..., scheme=...)`` is the advective flux on the faces
along any axis of a C-grid array, for any scheme: :class:`Upwind` (orders 1, 2,
3, 5), :class:`Centred`, :class:`WENO` (orders 3, 5, 7, 9; Jiang–Shu or WENO-Z)
and :class:`TVD` (four limiters). A scheme is a ladder of :class:`Tier` s,
widest first, and each face takes the widest one that fits in the array and,
with a mask, on wet cells. ``Reconstruction1D/2D/3D`` and ``Advection1D/2D/3D``
are built on it.

Example:
    >>> import numpy as np
    >>> import gridax.advection as gadv
    >>> q = np.random.default_rng(0).standard_normal((10, 12))
    >>> u = np.ones((10, 12))
    >>> gadv.reconstruct(q, u, axis=-1, scheme=gadv.WENO(5)).shape
    (10, 12)
"""

from gridax._src.advection.reconstruct import reconstruct
from gridax._src.advection.schemes import TVD, WENO, Centred, Scheme, Tier, Upwind

__all__ = ["Centred", "Scheme", "TVD", "Tier", "Upwind", "WENO", "reconstruct"]
