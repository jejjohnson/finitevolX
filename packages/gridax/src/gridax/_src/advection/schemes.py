"""Upwind reconstruction schemes as ladders of stencil widths.

A scheme is the list of its tiers, widest first, ending with first-order
upwind. Each tier is one Layer 1 reconstruction and its mirror: for flow
towards ``+axis`` the face ``i + 1/2`` reads the cell averages
``q[i + lo] … q[i + hi]`` (the upwind cell is ``i``), and for flow towards
``-axis`` it reads ``q[i + 1 - hi] … q[i + 1 - lo]`` (the upwind cell is
``i + 1``). :func:`~gridax.advection.reconstruct` gives each face the widest
tier that fits in the array and, with a mask, on wet cells; that is how every
``Reconstruction1D/2D/3D`` method has treated walls.

The WENO nonlinear weights and the TVD limiters are not stencils: they are what
these schemes do with the Layer 1 reconstructions, whose linear parts come from
Layer 0 (``gridax.stencils``).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gridax._src.advection import linear, weno
from gridax._src.advection.limiters import mc, minmod, superbee, van_leer
from gridax._src.pytree import array_namespace

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

__all__ = ["TVD", "WENO", "Centred", "Scheme", "Tier", "Upwind"]

# Relative TVD floor (#294): ``r = num / (den + copysign(eps, den))`` with
# ``eps = eps_rel · |num| + tiny``, a floor relative to the stencil's own
# slopes with the sign of ``den`` kept.  ``r`` therefore scales exactly with
# the field (an absolute floor made every limiter first-order upwind at small
# amplitude), is within a relative ``eps_rel`` of the exact ratio whenever
# ``|r| ≪ 1 / eps_rel``, and a tiny downwind slope cannot flip its sign.  A
# flat stencil gives ``r = 0`` (the floor is the square root of the dtype's
# smallest normal, so the gradient stays finite there).
_TVD_EPS_REL: float = 1e-6


def _slope_ratio(num: Array, den: Array) -> Array:
    """TVD slope ratio ``num / (den + copysign(eps_rel |num| + tiny, den))``."""
    xp = array_namespace(num, den)
    # sqrt(tiny): its square (in the gradient's 1 / (den + eps)²) stays normal,
    # so a flat stencil has a finite (zero) gradient, float32 included
    floor = float(np.sqrt(xp.finfo(xp.result_type(den)).tiny))
    return num / (den + xp.copysign(_TVD_EPS_REL * xp.abs(num) + floor, den))


@dataclass(frozen=True)
class Tier:
    """One stencil width of a scheme.

    Attributes
    ----------
    lo, hi : int
        For flow towards ``+axis`` the face ``i + 1/2`` reads cells
        ``i + lo … i + hi``; for flow towards ``-axis`` it reads
        ``i + 1 - hi … i + 1 - lo``.
    left : Callable
        ``hi - lo + 1`` cell averages, in order → the face value, for flow
        towards ``+axis``.
    right : Callable
        The mirror, for flow towards ``-axis``.
    """

    lo: int
    hi: int
    left: Callable[..., Array]
    right: Callable[..., Array]

    @property
    def size(self) -> int:
        """The adaptive-mask tier: ``2 w + 2`` for a footprint of ``w`` cells each side."""
        return 2 * max(-self.lo, self.hi) + 2


def _first(q0: Array) -> Array:
    return q0


def _upwind2_left(qm: Array, q0: Array) -> Array:
    # q[i+1/2] = 3/2 q[i] - 1/2 q[i-1]
    return 1.5 * q0 - 0.5 * qm


def _upwind2_right(qp: Array, qpp: Array) -> Array:
    # q[i+1/2] = 3/2 q[i+1] - 1/2 q[i+2]
    return 1.5 * qp - 0.5 * qpp


def _centred(q0: Array, qp: Array) -> Array:
    # q[i+1/2] = 1/2 (q[i] + q[i+1]), for either sign of the velocity
    return 0.5 * (q0 + qp)


UPWIND1 = Tier(0, 0, _first, _first)
_LINEAR = {
    3: Tier(-1, 1, linear.linear_3pts_left, linear.linear_3pts_right),
    5: Tier(-2, 2, linear.linear_5pts_left, linear.linear_5pts_right),
}
_WENO_JS = {
    3: Tier(-1, 1, weno.weno_3pts, weno.weno_3pts_right),
    5: Tier(-2, 2, weno.weno_5pts, weno.weno_5pts_right),
    7: Tier(-3, 3, weno.weno_7pts, weno.weno_7pts_right),
    9: Tier(-4, 4, weno.weno_9pts, weno.weno_9pts_right),
}
_WENO_Z = {
    3: Tier(-1, 1, weno.weno_3pts_improved, weno.weno_3pts_improved_right),
    5: Tier(-2, 2, weno.weno_5pts_improved, weno.weno_5pts_improved_right),
}
_LIMITERS: dict[str, Callable[[Any], Any]] = {
    "minmod": minmod,
    "van_leer": van_leer,
    "superbee": superbee,
    "mc": mc,
}


class Scheme(abc.ABC):
    """An upwind reconstruction: its tiers, widest first, ending with upwind."""

    @property
    @abc.abstractmethod
    def tiers(self) -> tuple[Tier, ...]:
        """Widest first; the last one always fits."""

    @property
    def upwinded(self) -> bool:
        """Whether the face value depends on the sign of the velocity."""
        return True


@dataclass(frozen=True)
class Upwind(Scheme):
    """Linear upwind-biased reconstruction of ``order`` 1, 2, 3 or 5.

    ``order=1`` reads the upwind cell; ``2`` the upwind cell and the one
    behind it, ``3/2 q[i] - 1/2 q[i-1]``; ``3`` and ``5`` the optimal linear
    weights of ``linear_3pts_left`` and ``linear_5pts_left`` (the limits of
    WENO-3 and WENO-5 on smooth data).
    """

    order: Literal[1, 2, 3, 5] = 1

    def __post_init__(self) -> None:
        if self.order not in (1, 2, 3, 5):
            raise ValueError(f"Upwind order must be 1, 2, 3 or 5; got {self.order!r}")

    @property
    def tiers(self) -> tuple[Tier, ...]:
        if self.order == 1:
            return (UPWIND1,)
        if self.order == 2:
            return (Tier(-1, 0, _upwind2_left, _upwind2_right), UPWIND1)
        if self.order == 3:
            return (_LINEAR[3], UPWIND1)
        return (_LINEAR[5], _LINEAR[3], UPWIND1)


@dataclass(frozen=True)
class Centred(Scheme):
    """Today's ``naive_*``: ``1/2 (q[i] + q[i+1])`` for either sign; one tier, always fits."""

    @property
    def tiers(self) -> tuple[Tier, ...]:
        return (Tier(0, 1, _centred, _centred),)

    @property
    def upwinded(self) -> bool:
        return False


@dataclass(frozen=True)
class WENO(Scheme):
    """WENO of ``order`` 3, 5, 7 or 9, Jiang–Shu (``"js"``) or WENO-Z (``"z"``, 3 and 5).

    Near a wall the face falls back through the narrower WENO stencils to
    first-order upwind, so it never reads a downwind cell.
    """

    order: Literal[3, 5, 7, 9] = 5
    variant: Literal["js", "z"] = "js"

    def __post_init__(self) -> None:
        family = _WENO_JS if self.variant == "js" else _WENO_Z
        if self.variant not in ("js", "z") or self.order not in family:
            raise ValueError(
                f"WENO supports orders {sorted(_WENO_JS)} (js) and {sorted(_WENO_Z)}"
                f" (z); got order={self.order!r}, variant={self.variant!r}"
            )

    @property
    def tiers(self) -> tuple[Tier, ...]:
        family = _WENO_JS if self.variant == "js" else _WENO_Z
        return (
            *(family[k] for k in sorted(family, reverse=True) if k <= self.order),
            UPWIND1,
        )


@dataclass(frozen=True)
class TVD(Scheme):
    """Second-order TVD: first-order upwind plus a limited anti-diffusive correction.

    ``q[i+1/2] = q[i] + 1/2 phi(r) (q[i+1] - q[i])`` with
    ``r = (q[i] - q[i-1]) / (q[i+1] - q[i])`` for flow towards ``+axis``,
    and the mirror for flow towards ``-axis``.
    """

    #: ``"minmod"``, ``"van_leer"``, ``"superbee"`` or ``"mc"``
    limiter: str = "minmod"

    def __post_init__(self) -> None:
        if self.limiter not in _LIMITERS:
            raise ValueError(
                f"Unknown flux limiter {self.limiter!r}. Choose one of {list(_LIMITERS)!r}."
            )

    @property
    def tiers(self) -> tuple[Tier, ...]:
        phi = _LIMITERS[self.limiter]

        def left(qm: Array, q0: Array, qp: Array) -> Array:
            # r = (q[i] - q[i-1]) / (q[i+1] - q[i])
            diff_fwd = qp - q0
            diff_bwd = q0 - qm
            return q0 + 0.5 * phi(_slope_ratio(diff_bwd, diff_fwd)) * diff_fwd

        def right(q0: Array, qp: Array, qpp: Array) -> Array:
            # r = (q[i+1] - q[i+2]) / (q[i] - q[i+1])
            diff_neg = q0 - qp
            diff_neg2 = qp - qpp
            return qp + 0.5 * phi(_slope_ratio(diff_neg2, diff_neg)) * diff_neg

        return (Tier(-1, 1, left, right), UPWIND1)


@dataclass(frozen=True)
class _Ladder(Scheme):
    """A scheme cut down to some of its tiers (the narrower rungs of a mask hierarchy)."""

    rungs: tuple[Tier, ...]

    @property
    def tiers(self) -> tuple[Tier, ...]:
        return self.rungs
