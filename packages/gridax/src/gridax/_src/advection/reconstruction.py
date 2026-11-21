"""Face-value reconstruction operators for advective fluxes on Arakawa C-grids.

Every method of ``Reconstruction1D/2D/3D`` is one call of the axis-generic
:func:`~gridax._src.advection.reconstruct.reconstruct` with a scheme
(``Upwind``, ``Centred``, ``WENO``, ``TVD``; #464): each face takes the widest
tier of the scheme that fits in the array (and, for the ``_masked`` methods,
on wet cells), down to first-order upwind. All operators follow the
interior-point idiom (write only ``[1:-1, 1:-1]``).

Notation
--------
  fe[j, i+1/2]  east-face flux
  fn[j+1/2, i]  north-face flux
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Float
import numpy as np

from gridax._src.advection.reconstruct import reconstruct
from gridax._src.advection.schemes import (
    _LIMITERS,
    _TVD_EPS_REL,
    TVD,
    WENO,
    Centred,
    Scheme,
    Upwind,
    _Ladder,
    _slope_ratio,
)
from gridax._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from gridax._src.mask import Mask2D
from gridax._src.operators._ghost import interior
from gridax._src.pytree import array_namespace, pytree_dataclass

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

# kept importable here for the tests that read them (#294)
__all__ = ["_LIMITERS", "_TVD_EPS_REL", "_slope_ratio"]


def _masked_columns(
    h: Array, u: Array, mask: Mask2D, scheme: Scheme, *, axis: int
) -> Array:
    """3-D masked flux: a 2-D mask broadcast over z, the tier chosen at the upwind cell.

    The widest tier whose mask holds at the upwind cell (``(j, i)`` for flow
    towards ``+axis``, the next cell along ``axis`` otherwise) is selected
    with nested ``where``; a cell no tier supports gets first-order upwind.
    """
    xp = array_namespace(h, u)
    tiers = scheme.tiers
    sizes = tuple(sorted({tier.size for tier in tiers}))
    amasks = mask.get_adaptive_masks(
        direction="x" if axis == -1 else "y", stencil_sizes=sizes
    )
    inner = (slice(1, -1), slice(1, -1), slice(1, -1))
    pos_flow = u[inner] >= 0.0
    # the upwind cell for negative flow is one step along `axis`
    shifted = (
        (None, slice(1, -1), slice(2, None))
        if axis == -1
        else (None, slice(2, None), slice(1, -1))
    )
    selected = reconstruct(h, u, axis=axis, scheme=_Ladder(tiers[-1:]))[inner]
    for k in range(len(tiers) - 2, -1, -1):
        m = amasks[tiers[k].size]
        use = xp.where(pos_flow, m[None, 1:-1, 1:-1], m[shifted])
        flux = reconstruct(h, u, axis=axis, scheme=_Ladder(tiers[k:]))
        selected = xp.where(use, flux[inner], selected)
    return interior(selected, h)


@pytree_dataclass
class Reconstruction1D:
    """1-D face-value reconstruction.

    Parameters
    ----------
    grid : ArakawaCGrid1D
        Any 1-D Arakawa C-grid.  Reconstruction is coordinate-agnostic —
        the grid is stored for downstream consumers but its spacing
        fields are never read by the reconstruction stencils themselves.
    """

    grid: ArakawaCGrid1D

    def naive_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """Naive (centred) reconstruction at east face.

        fe[i+1/2] = 1/2 * (h[i] + h[i+1]) * u[i+1/2]
        """
        return reconstruct(h, u, axis=-1, scheme=Centred())

    def upwind1_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """1st-order upwind reconstruction at east face.

        fe[i+1/2] = h[i]   * u[i+1/2]   if u[i+1/2] >= 0
                  = h[i+1] * u[i+1/2]   otherwise
        """
        return reconstruct(h, u, axis=-1, scheme=Upwind(1))

    def upwind2_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """2nd-order upwind reconstruction at east face with boundary fallback.

        Positive flow:  h_face[i+1/2] = 3/2*h[i]   - 1/2*h[i-1]
        Negative flow:  h_face[i+1/2] = 3/2*h[i+1] - 1/2*h[i+2]
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=Upwind(2))

    def upwind3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3rd-order upwind reconstruction at east face with boundary fallback.

        Positive flow:  h_face = -1/6*h[i-1] + 5/6*h[i]   + 1/3*h[i+1]
        Negative flow:  h_face =  1/3*h[i]   + 5/6*h[i+1] - 1/6*h[i+2]
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=Upwind(3))

    def weno3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3-point WENO east-face flux with boundary fallback.

        Positive flow:  h_face[i+1/2] = WENO3(h[i-1], h[i],   h[i+1])
        Negative flow:  h_face[i+1/2] = WENO3_right(h[i], h[i+1], h[i+2])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(3))

    def wenoz3_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """3-point WENO-Z east-face flux with boundary fallback.

        Positive flow:  h_face[i+1/2] = WENOZ3(h[i-1], h[i],   h[i+1])
        Negative flow:  h_face[i+1/2] = WENOZ3_right(h[i], h[i+1], h[i+2])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(3, "z"))

    def weno5_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """5-point WENO east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENO5(h[i-2], h[i-1], h[i], h[i+1], h[i+2])
            for i = 2..Nx-3; WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENO5_right(h[i-1], h[i], h[i+1], h[i+2], h[i+3])
            for i = 1..Nx-4; WENO3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(5))

    def wenoz5_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """5-point WENO-Z east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENOZ5(h[i-2], h[i-1], h[i], h[i+1], h[i+2])
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENOZ5_right(h[i-1], h[i], h[i+1], h[i+2], h[i+3])
            for i = 1..Nx-4; WENO-Z-3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(5, "z"))

    def weno7_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """7-point WENO east-face flux with hierarchical boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENO7(h[i-3..i+3])
            for i = 3..Nx-4; WENO5 fallback at i = 2 and i = Nx-3;
            WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENO7(h[i+3..i-3])
            for i = 3..Nx-4; WENO3 fallback at i = 1; WENO5 fallback
            at i = 2 and i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(7))

    def weno9_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
    ) -> Float[Array, "Nx"]:
        """9-point WENO east-face flux with hierarchical boundary fallbacks.

        Positive flow:  h_face[i+1/2] = WENO9(h[i-4..i+4])
            for i = 4..Nx-5; WENO7 fallback at i = 3 and i = Nx-4;
            WENO5 fallback at i = 2 and i = Nx-3; WENO3 fallback at
            i = 1 and i = Nx-2.
        Negative flow:  h_face[i+1/2] = WENO9(h[i+4..i-4])
            for i = 4..Nx-5; WENO7 fallback at i = 3 and i = Nx-4;
            WENO5 fallback at i = 2 and i = Nx-3; WENO3 fallback at
            i = 1; 1st-order upwind at i = Nx-2.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(9))

    def tvd_x(
        self,
        h: Float[Array, "Nx"],
        u: Float[Array, "Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Nx"]:
        """TVD east-face flux using a flux limiter.

        Blends 1st-order upwind with an anti-diffusive correction limited by
        φ(r) to preserve monotonicity:

        Positive flow:
            r     = (h[i] − h[i−1]) / (h[i+1] − h[i])
            h_L   = h[i]   + ½ φ(r) (h[i+1] − h[i])

        Negative flow:
            r     = (h[i+1] − h[i+2]) / (h[i] − h[i+1])
            h_R   = h[i+1] + ½ φ(r) (h[i]   − h[i+1])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Nx"]
            Scalar at T-points.
        u : Float[Array, "Nx"]
            Velocity at U-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nx"]
            East-face flux with zero ghost entries.
        """
        return reconstruct(h, u, axis=-1, scheme=TVD(limiter))


@pytree_dataclass
class Reconstruction2D:
    """2-D face-value reconstruction for advective fluxes.

    Parameters
    ----------
    grid : ArakawaCGrid2D
        Any 2-D Arakawa C-grid (Cartesian, spherical, etc.).
        Reconstruction stencils are coordinate-agnostic — the grid is
        stored for downstream consumers but the spacing fields are
        never read by the reconstruction itself.
    """

    grid: ArakawaCGrid2D

    def naive_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Naive (centred) east-face flux.

        fe[j, i+1/2] = 1/2 * (h[j, i] + h[j, i+1]) * u[j, i+1/2]
        """
        return reconstruct(h, u, axis=-1, scheme=Centred())

    def naive_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Naive (centred) north-face flux.

        fn[j+1/2, i] = 1/2 * (h[j, i] + h[j+1, i]) * v[j+1/2, i]
        """
        return reconstruct(h, v, axis=-2, scheme=Centred())

    def upwind1_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """1st-order upwind east-face flux.

        fe[j, i+1/2] = h[j, i]   * u  if u >= 0
                     = h[j, i+1] * u  otherwise
        """
        return reconstruct(h, u, axis=-1, scheme=Upwind(1))

    def upwind1_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """1st-order upwind north-face flux.

        fn[j+1/2, i] = h[j, i]   * v  if v >= 0
                     = h[j+1, i] * v  otherwise
        """
        return reconstruct(h, v, axis=-2, scheme=Upwind(1))

    def upwind2_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """2nd-order upwind east-face flux with boundary fallback.

        Positive: h_face = 3/2*h[j,i] - 1/2*h[j,i-1]
        Negative: h_face = 3/2*h[j,i+1] - 1/2*h[j,i+2]
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=Upwind(2))

    def upwind2_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """2nd-order upwind north-face flux with boundary fallback.

        Positive: h_face = 3/2*h[j,i] - 1/2*h[j-1,i]
        Negative: h_face = 3/2*h[j+1,i] - 1/2*h[j+2,i]
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        return reconstruct(h, v, axis=-2, scheme=Upwind(2))

    def upwind3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3rd-order upwind east-face flux with boundary fallback.

        Positive: h_face = -1/6*h[j,i-1] + 5/6*h[j,i]   + 1/3*h[j,i+1]
        Negative: h_face =  1/3*h[j,i]   + 5/6*h[j,i+1] - 1/6*h[j,i+2]
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=Upwind(3))

    def upwind3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3rd-order upwind north-face flux with boundary fallback.

        Positive: h_face = -1/6*h[j-1,i] + 5/6*h[j,i]   + 1/3*h[j+1,i]
        Negative: h_face =  1/3*h[j,i]   + 5/6*h[j+1,i] - 1/6*h[j+2,i]
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        return reconstruct(h, v, axis=-2, scheme=Upwind(3))

    def weno3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO east-face flux with boundary fallback.

        Positive flow:  fe[j,i+1/2] = WENO3(h[j,i-1], h[j,i],   h[j,i+1]) * u
        Negative flow:  fe[j,i+1/2] = WENO3_right(h[j,i], h[j,i+1], h[j,i+2]) * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(3))

    def weno3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO north-face flux with boundary fallback.

        Positive flow:  fn[j+1/2,i] = WENO3(h[j-1,i], h[j,i],   h[j+1,i]) * v
        Negative flow:  fn[j+1/2,i] = WENO3_right(h[j,i], h[j+1,i], h[j+2,i]) * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(3))

    def wenoz3_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO-Z east-face flux with boundary fallback.

        Positive flow:  fe[j,i+1/2] = WENOZ3(h[j,i-1], h[j,i],   h[j,i+1]) * u
        Negative flow:  fe[j,i+1/2] = WENOZ3_right(h[j,i], h[j,i+1], h[j,i+2]) * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(3, "z"))

    def wenoz3_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """3-point WENO-Z north-face flux with boundary fallback.

        Positive flow:  fn[j+1/2,i] = WENOZ3(h[j-1,i], h[j,i],   h[j+1,i]) * v
        Negative flow:  fn[j+1/2,i] = WENOZ3_right(h[j,i], h[j+1,i], h[j+2,i]) * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(3, "z"))

    def weno5_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fe[j,i+1/2] = WENO5(h[j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[j,i+1/2] = WENO5_right(h[j,i-1], h[j,i], h[j,i+1], h[j,i+2], h[j,i+3]) * u
            for i = 1..Nx-4; WENO3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(5))

    def weno5_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO north-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fn[j+1/2,i] = WENO5(h[j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[j+1/2,i] = WENO5_right(h[j-1,i], h[j,i], h[j+1,i], h[j+2,i], h[j+3,i]) * v
            for j = 1..Ny-4; WENO3_right fallback at j = Ny-3; 1st-order upwind at j = Ny-2.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(5))

    def wenoz5_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z east-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fe[j,i+1/2] = WENOZ5(h[j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[j,i+1/2] = WENOZ5_right(h[j,i-1], h[j,i], h[j,i+1], h[j,i+2], h[j,i+3]) * u
            for i = 1..Nx-4; WENO-Z-3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(5, "z"))

    def wenoz5_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z north-face flux with sign-dependent boundary fallbacks.

        Positive flow:  fn[j+1/2,i] = WENOZ5(h[j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO-Z-3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[j+1/2,i] = WENOZ5_right(h[j-1,i], h[j,i], h[j+1,i], h[j+2,i], h[j+3,i]) * v
            for j = 1..Ny-4; WENO-Z-3_right fallback at j = Ny-3; 1st-order upwind at j = Ny-2.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(5, "z"))

    def weno7_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """7-point WENO east-face flux with hierarchical boundary fallbacks."""
        return reconstruct(h, u, axis=-1, scheme=WENO(7))

    def weno7_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """7-point WENO north-face flux with hierarchical boundary fallbacks."""
        return reconstruct(h, v, axis=-2, scheme=WENO(7))

    def weno9_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """9-point WENO east-face flux with hierarchical boundary fallbacks."""
        return reconstruct(h, u, axis=-1, scheme=WENO(9))

    def weno9_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """9-point WENO north-face flux with hierarchical boundary fallbacks."""
        return reconstruct(h, v, axis=-2, scheme=WENO(9))

    def tvd_x(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Ny Nx"]:
        """TVD east-face flux using a flux limiter.

        Positive flow:
            r         = (h[j,i] − h[j,i−1]) / (h[j,i+1] − h[j,i])
            h_L[j,i+½] = h[j,i]   + ½ φ(r) (h[j,i+1] − h[j,i])

        Negative flow:
            r         = (h[j,i+1] − h[j,i+2]) / (h[j,i] − h[j,i+1])
            h_R[j,i+½] = h[j,i+1] + ½ φ(r) (h[j,i]   − h[j,i+1])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.
        u : Float[Array, "Ny Nx"]
            x-velocity at U-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return reconstruct(h, u, axis=-1, scheme=TVD(limiter))

    def tvd_y(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Ny Nx"]:
        """TVD north-face flux using a flux limiter.

        Positive flow:
            r         = (h[j,i] − h[j−1,i]) / (h[j+1,i] − h[j,i])
            h_L[j+½,i] = h[j,i]   + ½ φ(r) (h[j+1,i] − h[j,i])

        Negative flow:
            r         = (h[j+1,i] − h[j+2,i]) / (h[j,i] − h[j+1,i])
            h_R[j+½,i] = h[j+1,i] + ½ φ(r) (h[j,i]   − h[j+1,i])
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Scalar at T-points.
        v : Float[Array, "Ny Nx"]
            y-velocity at V-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return reconstruct(h, v, axis=-2, scheme=TVD(limiter))

    def tvd_x_masked(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        mask: Mask2D,
        limiter: str = "minmod",
    ) -> Float[Array, "Ny Nx"]:
        """TVD east-face flux with mask-aware stencil selection.

        Delegates to :func:`~gridax._src.advection.flux.upwind_flux` with a
        two-tier hierarchy (upwind1 / TVD).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Ny Nx"]
            East-face velocity.
        mask : Mask2D
            Arakawa C-grid mask providing stencil-capability information.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return reconstruct(h, u, axis=-1, scheme=TVD(limiter), mask=mask)

    def tvd_y_masked(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: Mask2D,
        limiter: str = "minmod",
    ) -> Float[Array, "Ny Nx"]:
        """TVD north-face flux with mask-aware stencil selection.

        Delegates to :func:`~gridax._src.advection.flux.upwind_flux` with a
        two-tier hierarchy (upwind1 / TVD).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Ny Nx"]
            North-face velocity.
        mask : Mask2D
            Arakawa C-grid mask providing stencil-capability information.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return reconstruct(h, v, axis=-2, scheme=TVD(limiter), mask=mask)

    def weno5_x_masked(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO east-face flux with mask-aware adaptive stencil selection.

        Delegates to :func:`~gridax._src.advection.flux.upwind_flux` with a
        three-tier hierarchy (upwind1 / WENO3 / WENO5).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Ny Nx"]
            East-face velocity.
        mask : Mask2D
            Arakawa C-grid mask providing stencil-capability information.

        Returns
        -------
        Float[Array, "Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(5), mask=mask)

    def weno5_y_masked(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO north-face flux with mask-aware adaptive stencil selection.

        Delegates to :func:`~gridax._src.advection.flux.upwind_flux` with a
        three-tier hierarchy (upwind1 / WENO3 / WENO5).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Ny Nx"]
            North-face velocity.
        mask : Mask2D
            Arakawa C-grid mask providing stencil-capability information.

        Returns
        -------
        Float[Array, "Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(5), mask=mask)

    def wenoz5_x_masked(
        self,
        h: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z east-face flux with mask-aware adaptive stencil selection.

        Delegates to :func:`~gridax._src.advection.flux.upwind_flux` with a
        three-tier hierarchy (upwind1 / WENO-Z-3 / WENO-Z-5).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Ny Nx"]
            East-face velocity.
        mask : Mask2D
            Arakawa C-grid mask providing stencil-capability information.

        Returns
        -------
        Float[Array, "Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(5, "z"), mask=mask)

    def wenoz5_y_masked(
        self,
        h: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Ny Nx"]:
        """5-point WENO-Z north-face flux with mask-aware adaptive stencil selection.

        Delegates to :func:`~gridax._src.advection.flux.upwind_flux` with a
        three-tier hierarchy (upwind1 / WENO-Z-3 / WENO-Z-5).

        Parameters
        ----------
        h : Float[Array, "Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Ny Nx"]
            North-face velocity.
        mask : Mask2D
            Arakawa C-grid mask providing stencil-capability information.

        Returns
        -------
        Float[Array, "Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(5, "z"), mask=mask)


@pytree_dataclass
class Reconstruction3D:
    """3-D face-value reconstruction.

    Parameters
    ----------
    grid : ArakawaCGrid3D
        Any 3-D Arakawa C-grid.  Reconstruction stencils are
        coordinate-agnostic — the grid is stored but its spacing
        fields are never read by the reconstruction itself.
    """

    grid: ArakawaCGrid3D

    def naive_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Naive east-face flux over all z-levels.

        fe[k, j, i+1/2] = 1/2 * (h[k,j,i] + h[k,j,i+1]) * u[k,j,i+1/2]
        """
        return reconstruct(h, u, axis=-1, scheme=Centred())

    def naive_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """Naive north-face flux over all z-levels.

        fn[k, j+1/2, i] = 1/2 * (h[k,j,i] + h[k,j+1,i]) * v[k,j+1/2,i]
        """
        return reconstruct(h, v, axis=-2, scheme=Centred())

    def upwind1_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """1st-order upwind east-face flux over all z-levels."""
        return reconstruct(h, u, axis=-1, scheme=Upwind(1))

    def upwind1_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """1st-order upwind north-face flux over all z-levels."""
        return reconstruct(h, v, axis=-2, scheme=Upwind(1))

    def weno3_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO east-face flux over all z-levels with boundary fallback.

        Positive flow:  fe[k,j,i+1/2] = WENO3(h[k,j,i-1], h[k,j,i],   h[k,j,i+1]) * u
        Negative flow:  fe[k,j,i+1/2] = WENO3_right(h[k,j,i], h[k,j,i+1], h[k,j,i+2]) * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(3))

    def weno3_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO north-face flux over all z-levels with boundary fallback.

        Positive flow:  fn[k,j+1/2,i] = WENO3(h[k,j-1,i], h[k,j,i],   h[k,j+1,i]) * v
        Negative flow:  fn[k,j+1/2,i] = WENO3_right(h[k,j,i], h[k,j+1,i], h[k,j+2,i]) * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(3))

    def wenoz3_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO-Z east-face flux over all z-levels with boundary fallback.

        Positive flow:  fe[k,j,i+1/2] = WENOZ3(h[k,j,i-1], h[k,j,i],   h[k,j,i+1]) * u
        Negative flow:  fe[k,j,i+1/2] = WENOZ3_right(h[k,j,i], h[k,j,i+1], h[k,j,i+2]) * u
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(3, "z"))

    def wenoz3_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO-Z north-face flux over all z-levels with boundary fallback.

        Positive flow:  fn[k,j+1/2,i] = WENOZ3(h[k,j-1,i], h[k,j,i],   h[k,j+1,i]) * v
        Negative flow:  fn[k,j+1/2,i] = WENOZ3_right(h[k,j,i], h[k,j+1,i], h[k,j+2,i]) * v
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(3, "z"))

    def weno5_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO east-face flux over all z-levels with sign-dependent fallbacks.

        Positive flow:  fe[k,j,i+1/2] = WENO5(h[k,j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[k,j,i+1/2] = WENO5_right(h[k,j,i-1], h[k,j,i], h[k,j,i+1], h[k,j,i+2], h[k,j,i+3]) * u
            for i = 1..Nx-4; WENO3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(5))

    def weno5_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO north-face flux over all z-levels with sign-dependent fallbacks.

        Positive flow:  fn[k,j+1/2,i] = WENO5(h[k,j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[k,j+1/2,i] = WENO5_right(h[k,j-1,i], h[k,j,i], h[k,j+1,i], h[k,j+2,i], h[k,j+3,i]) * v
            for j = 1..Ny-4; WENO3_right fallback at j = Ny-3; 1st-order upwind at j = Ny-2.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(5))

    def wenoz5_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO-Z east-face flux over all z-levels with sign-dependent fallbacks.

        Positive flow:  fe[k,j,i+1/2] = WENOZ5(h[k,j,i-2..i+2]) * u
            for i = 2..Nx-3; WENO-Z-3 fallback at i = 1 and i = Nx-2.
        Negative flow:  fe[k,j,i+1/2] = WENOZ5_right(h[k,j,i-1], h[k,j,i], h[k,j,i+1], h[k,j,i+2], h[k,j,i+3]) * u
            for i = 1..Nx-4; WENO-Z-3_right fallback at i = Nx-3; 1st-order upwind at i = Nx-2.
        """
        return reconstruct(h, u, axis=-1, scheme=WENO(5, "z"))

    def wenoz5_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO-Z north-face flux over all z-levels with sign-dependent fallbacks.

        Positive flow:  fn[k,j+1/2,i] = WENOZ5(h[k,j-2..j+2,i]) * v
            for j = 2..Ny-3; WENO-Z-3 fallback at j = 1 and j = Ny-2.
        Negative flow:  fn[k,j+1/2,i] = WENOZ5_right(h[k,j-1,i], h[k,j,i], h[k,j+1,i], h[k,j+2,i], h[k,j+3,i]) * v
            for j = 1..Ny-4; WENO-Z-3_right fallback at j = Ny-3; 1st-order upwind at j = Ny-2.
        """
        return reconstruct(h, v, axis=-2, scheme=WENO(5, "z"))

    def weno7_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """7-point WENO east-face flux over all z-levels with hierarchical fallbacks."""
        return reconstruct(h, u, axis=-1, scheme=WENO(7))

    def weno7_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """7-point WENO north-face flux over all z-levels with hierarchical fallbacks."""
        return reconstruct(h, v, axis=-2, scheme=WENO(7))

    def weno9_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """9-point WENO east-face flux over all z-levels with hierarchical fallbacks."""
        return reconstruct(h, u, axis=-1, scheme=WENO(9))

    def weno9_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """9-point WENO north-face flux over all z-levels with hierarchical fallbacks."""
        return reconstruct(h, v, axis=-2, scheme=WENO(9))

    def tvd_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Nz Ny Nx"]:
        """TVD east-face flux over all z-levels using a flux limiter.

        Positive flow:
            r           = (h[k,j,i] − h[k,j,i−1]) / (h[k,j,i+1] − h[k,j,i])
            h_L[k,j,i+½] = h[k,j,i]   + ½ φ(r) (h[k,j,i+1] − h[k,j,i])

        Negative flow:
            r           = (h[k,j,i+1] − h[k,j,i+2]) / (h[k,j,i] − h[k,j,i+1])
            h_R[k,j,i+½] = h[k,j,i+1] + ½ φ(r) (h[k,j,i]   − h[k,j,i+1])
        Falls back to 1st-order upwind on east boundary where i+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar at T-points.
        u : Float[Array, "Nz Ny Nx"]
            x-velocity at U-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return reconstruct(h, u, axis=-1, scheme=TVD(limiter))

    def tvd_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        limiter: str = "minmod",
    ) -> Float[Array, "Nz Ny Nx"]:
        """TVD north-face flux over all z-levels using a flux limiter.

        Positive flow:
            r           = (h[k,j,i] − h[k,j−1,i]) / (h[k,j+1,i] − h[k,j,i])
            h_L[k,j+½,i] = h[k,j,i]   + ½ φ(r) (h[k,j+1,i] − h[k,j,i])

        Negative flow:
            r           = (h[k,j+1,i] − h[k,j+2,i]) / (h[k,j,i] − h[k,j+1,i])
            h_R[k,j+½,i] = h[k,j+1,i] + ½ φ(r) (h[k,j,i]   − h[k,j+1,i])
        Falls back to 1st-order upwind on north boundary where j+2 unavailable.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Scalar at T-points.
        v : Float[Array, "Nz Ny Nx"]
            y-velocity at V-points.
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return reconstruct(h, v, axis=-2, scheme=TVD(limiter))

    def tvd_x_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        mask: Mask2D,
        limiter: str = "minmod",
    ) -> Float[Array, "Nz Ny Nx"]:
        """TVD east-face flux over all z-levels with mask-aware stencil selection.

        The 2-D ``mask`` is broadcast over the z-dimension.  Falls back to
        1st-order upwind near coastlines where the TVD stencil crosses land.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Nz Ny Nx"]
            East-face velocity.
        mask : Mask2D
            2-D Arakawa C-grid mask (broadcast over z).
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return _masked_columns(h, u, mask, TVD(limiter), axis=-1)

    def tvd_y_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: Mask2D,
        limiter: str = "minmod",
    ) -> Float[Array, "Nz Ny Nx"]:
        """TVD north-face flux over all z-levels with mask-aware stencil selection.

        The 2-D ``mask`` is broadcast over the z-dimension.  Falls back to
        1st-order upwind near coastlines where the TVD stencil crosses land.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Nz Ny Nx"]
            North-face velocity.
        mask : Mask2D
            2-D Arakawa C-grid mask (broadcast over z).
        limiter : str
            Flux limiter name: ``'minmod'``, ``'van_leer'``, ``'superbee'``,
            or ``'mc'``.

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return _masked_columns(h, v, mask, TVD(limiter), axis=-2)

    def weno3_x_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO east-face flux over all z-levels with mask-aware stencil selection.

        The 2-D ``mask`` is broadcast over the z-dimension.  Falls back to
        1st-order upwind near coastlines where the WENO3 stencil crosses land.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Nz Ny Nx"]
            East-face velocity.
        mask : Mask2D
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return _masked_columns(h, u, mask, WENO(3), axis=-1)

    def weno3_y_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Nz Ny Nx"]:
        """3-point WENO north-face flux over all z-levels with mask-aware stencil selection.

        The 2-D ``mask`` is broadcast over the z-dimension.  Falls back to
        1st-order upwind near coastlines where the WENO3 stencil crosses land.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Nz Ny Nx"]
            North-face velocity.
        mask : Mask2D
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return _masked_columns(h, v, mask, WENO(3), axis=-2)

    def weno5_x_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO east-face flux over all z-levels with mask-aware adaptive stencil.

        The 2-D ``mask`` is broadcast over the z-dimension so that the same
        horizontal stencil-capability map is applied at every depth level.
        Adaptively falls back to WENO3 or 1st-order upwind near coastlines.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Nz Ny Nx"]
            East-face velocity.
        mask : Mask2D
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return _masked_columns(h, u, mask, WENO(5), axis=-1)

    def weno5_y_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO north-face flux over all z-levels with mask-aware adaptive stencil.

        The 2-D ``mask`` is broadcast over the z-dimension so that the same
        horizontal stencil-capability map is applied at every depth level.
        Adaptively falls back to WENO3 or 1st-order upwind near coastlines.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Nz Ny Nx"]
            North-face velocity.
        mask : Mask2D
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return _masked_columns(h, v, mask, WENO(5), axis=-2)

    def wenoz5_x_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO-Z east-face flux over all z-levels with mask-aware adaptive stencil.

        The 2-D ``mask`` is broadcast over the z-dimension so that the same
        horizontal stencil-capability map is applied at every depth level.
        Adaptively falls back to WENO-Z-3 or 1st-order upwind near coastlines.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        u : Float[Array, "Nz Ny Nx"]
            East-face velocity.
        mask : Mask2D
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            East-face flux with zero ghost ring.
        """
        return _masked_columns(h, u, mask, WENO(5, "z"), axis=-1)

    def wenoz5_y_masked(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
        mask: Mask2D,
    ) -> Float[Array, "Nz Ny Nx"]:
        """5-point WENO-Z north-face flux over all z-levels with mask-aware adaptive stencil.

        The 2-D ``mask`` is broadcast over the z-dimension so that the same
        horizontal stencil-capability map is applied at every depth level.
        Adaptively falls back to WENO-Z-3 or 1st-order upwind near coastlines.

        Parameters
        ----------
        h : Float[Array, "Nz Ny Nx"]
            Cell-centre tracer field.
        v : Float[Array, "Nz Ny Nx"]
            North-face velocity.
        mask : Mask2D
            2-D Arakawa C-grid mask (broadcast over z).

        Returns
        -------
        Float[Array, "Nz Ny Nx"]
            North-face flux with zero ghost ring.
        """
        return _masked_columns(h, v, mask, WENO(5, "z"), axis=-2)

    def upwind2_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """2nd-order upwind east-face flux over all z-levels (as ``Reconstruction2D.upwind2_x``)."""
        return reconstruct(h, u, axis=-1, scheme=Upwind(2))

    def upwind2_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """2nd-order upwind north-face flux over all z-levels (as ``Reconstruction2D.upwind2_y``)."""
        return reconstruct(h, v, axis=-2, scheme=Upwind(2))

    def upwind3_x(
        self,
        h: Float[Array, "Nz Ny Nx"],
        u: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3rd-order upwind east-face flux over all z-levels (as ``Reconstruction2D.upwind3_x``)."""
        return reconstruct(h, u, axis=-1, scheme=Upwind(3))

    def upwind3_y(
        self,
        h: Float[Array, "Nz Ny Nx"],
        v: Float[Array, "Nz Ny Nx"],
    ) -> Float[Array, "Nz Ny Nx"]:
        """3rd-order upwind north-face flux over all z-levels (as ``Reconstruction2D.upwind3_y``)."""
        return reconstruct(h, v, axis=-2, scheme=Upwind(3))
