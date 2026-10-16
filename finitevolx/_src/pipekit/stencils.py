"""Stencil, interpolation, boundary and mask operators (2-D Cartesian).

Each operator is one Tier-A call. Fields are ``[*batch, Ny, Nx]`` with the
ghost ring included; outputs follow Tier A exactly (interior written, ghost
ring zero, unless the operator fills it).

Staggering is carried by configuration, not checked on the carrier::

    Difference(grid=G, axis="x", src="T", dst="U")      δₓ : T → U
    Interp(grid=G, src="U", dst="T")                     U → T (two-point mean)

The valid ``(axis, src, dst)`` and ``(src, dst)`` sets are read off the
Tier-A method names (``Difference2D.diff_x_T_to_U``,
``Interpolation2D.U_to_T``) at import, so a Tier-A method added or renamed
changes Tier B with it.
"""

from __future__ import annotations

from collections.abc import Callable
import inspect
import re
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.boundary.boundary import (
    enforce_periodic,
    free_slip_boundaries,
    no_flux_boundaries,
    no_slip_boundaries,
    wall_boundaries,
    zero_boundaries,
    zero_gradient_boundaries,
)
from finitevolx._src.mask.cartesian import Mask2D
from finitevolx._src.operators.coriolis import Coriolis2D
from finitevolx._src.operators.diagnostics import coriolis_fn
from finitevolx._src.operators.difference import Difference2D
from finitevolx._src.operators.interpolation import Interpolation2D
from finitevolx._src.pipekit.base import FVXOperator, GridOperator

Field = Float[Array, "*batch Ny Nx"]

STAGGERS = ("T", "U", "V", "X")


def _methods(cls: type, pattern: str) -> dict[tuple[str, ...], str]:
    """``{match groups: method name}`` for the public methods of ``cls``."""
    rx = re.compile(pattern)
    out = {}
    for name, _ in inspect.getmembers(cls, inspect.isfunction):
        m = rx.fullmatch(name)
        if m:
            out[m.groups()] = name
    return out


#: ``(axis, src, dst) -> Difference2D method``, e.g. ``("x", "T", "U")``.
DIFFERENCES = _methods(Difference2D, r"diff_([xy])_([TUVX])_to_([TUVX])")
#: ``(src, dst) -> Interpolation2D method``, e.g. ``("U", "T")``.
INTERPOLATIONS = _methods(Interpolation2D, r"([TUVX])_to_([TUVX])")


def _check_choice(what: str, got: tuple[str, ...], valid: dict) -> None:
    if got not in valid:
        options = ", ".join("/".join(k) for k in sorted(valid))
        raise ValueError(f"{what} {'/'.join(got)} is not available; valid: {options}")


class Difference(GridOperator):
    """One-sided difference between staggered points, ``δ / Δ``.

    ``axis="x", src="T", dst="U"``: δₓh[j, i+½] = (h[j, i+1] − h[j, i]) / Δx.

    Attributes
    ----------
    grid : GridSpec
        ``(nx_interior, ny_interior, Lx, Ly)``.
    axis : {"x", "y"}
        Difference direction.
    src, dst : {"T", "U", "V", "X"}
        Source and destination stagger; see `DIFFERENCES` for the valid
        triples.
    """

    axis: str = eqx.field(static=True, kw_only=True)
    src: str = eqx.field(static=True, kw_only=True)
    dst: str = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        _check_choice("difference", (self.axis, self.src, self.dst), DIFFERENCES)

    def _apply(self, x: Field) -> Field:
        method = DIFFERENCES[(self.axis, self.src, self.dst)]
        return getattr(Difference2D(grid=self.cartesian_grid), method)(x)


class Divergence(GridOperator):
    """∇·(u, v) at T-points: δₓu / Δx + δᵧv / Δy.  ``(U, V) → T``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return Difference2D(grid=self.cartesian_grid).divergence(u, v)


class Curl(GridOperator):
    """Relative vorticity ζ = δₓv / Δx − δᵧu / Δy at X-points.  ``(U, V) → X``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return Difference2D(grid=self.cartesian_grid).curl(u, v)


class Laplacian(GridOperator):
    """Five-point Laplacian ∇²h at T-points.  ``T → T``."""

    def _apply(self, h: Field) -> Field:
        return Difference2D(grid=self.cartesian_grid).laplacian(h)


class GradPerp(GridOperator):
    """Rotated gradient (u, v) = (−∂ᵧψ, ∂ₓψ) of a streamfunction.  ``X → (U, V)``."""

    def _apply(self, psi: Field) -> tuple[Field, Field]:
        return Difference2D(grid=self.cartesian_grid).grad_perp(psi)


class Interp(GridOperator):
    """Average between staggered points (two- or four-point mean).

    Attributes
    ----------
    grid : GridSpec
        ``(nx_interior, ny_interior, Lx, Ly)``.
    src, dst : {"T", "U", "V", "X"}
        Source and destination stagger; see `INTERPOLATIONS`.
    """

    src: str = eqx.field(static=True, kw_only=True)
    dst: str = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        _check_choice("interpolation", (self.src, self.dst), INTERPOLATIONS)

    def _apply(self, x: Field) -> Field:
        method = INTERPOLATIONS[(self.src, self.dst)]
        return getattr(Interpolation2D(grid=self.cartesian_grid), method)(x)


class Coriolis(GridOperator):
    """Coriolis tendencies on a β-plane, ``(U, V) → (U, V)``.

    f = f₀ + β (y − y₀) at T-points, ghost ring included (y = j Δy), then
    ``Coriolis2D(u, v, f)``: (+f v̄ at U, −f ū at V), energy conserving.
    ``f0`` and ``beta`` are learnable leaves.

    Attributes
    ----------
    grid : GridSpec
        ``(nx_interior, ny_interior, Lx, Ly)``.
    f0 : Array
        Reference Coriolis parameter.
    beta : Array
        Meridional gradient ∂f/∂y.
    y0 : float or None
        Reference latitude coordinate; ``None`` uses the mean of the T-point
        y coordinates (as `coriolis_fn`).
    """

    f0: jax.Array = eqx.field(converter=jnp.asarray, kw_only=True)
    beta: jax.Array = eqx.field(converter=jnp.asarray, default=0.0, kw_only=True)
    y0: float | None = eqx.field(static=True, default=None, kw_only=True)

    def coriolis_parameter(self, dtype: Any = None) -> Float[Array, "Ny Nx"]:
        """f at the T-points of the grid, ghost ring included."""
        g = self.cartesian_grid
        y = jnp.arange(g.Ny, dtype=dtype) * g.dy  # T[j] sits at y = j Δy
        Y = jnp.broadcast_to(y[:, None], (g.Ny, g.Nx))
        return coriolis_fn(Y, f0=self.f0, beta=self.beta, y0=self.y0)

    def _apply(self, u: Field, v: Field) -> tuple[Field, Field]:
        f = self.coriolis_parameter(jnp.result_type(u, v))
        return Coriolis2D(grid=self.cartesian_grid)(u, v, f)


#: ``kind -> ghost-ring filler`` for `FillGhosts`.
GHOST_FILLERS: dict[str, Callable[..., Field]] = {
    "periodic": enforce_periodic,
    "zero": zero_boundaries,
    "zero_gradient": zero_gradient_boundaries,
    "no_flux": no_flux_boundaries,
    "no_slip": no_slip_boundaries,
    "free_slip": free_slip_boundaries,
    "wall": wall_boundaries,
}


class FillGhosts(FVXOperator):
    """Fill the ghost ring with a boundary condition.  ``x → x``.

    Attributes
    ----------
    kind : str
        One of `GHOST_FILLERS`: ``periodic``, ``zero``, ``zero_gradient``,
        ``no_flux``, ``no_slip``, ``free_slip`` or ``wall``.
    staggering : {"h", "u", "v", "q"}
        Only for ``kind="wall"``: the C-grid variable (see
        `wall_boundaries`).
    """

    kind: str = eqx.field(static=True)
    staggering: str = eqx.field(static=True, default="h")

    def __check_init__(self) -> None:
        if self.kind not in GHOST_FILLERS:
            raise ValueError(
                f"kind {self.kind!r} is not available; valid: {sorted(GHOST_FILLERS)}"
            )

    def _apply(self, x: Field) -> Field:
        if self.kind == "wall":
            return wall_boundaries(x, staggering=self.staggering)
        return GHOST_FILLERS[self.kind](x)


#: ``stagger -> Mask2D attribute`` used by `ApplyMask` (contract 3).
MASK_ATTRIBUTES = {"T": "h", "U": "u", "V": "v", "X": "xy_corner_strict"}


class ApplyMask(FVXOperator):
    """Zero a field on the dry points of its stagger (``jnp.where``, NaN-safe).

    The mask is an array-holding module, so the config has no JSON form:
    the operator is ``forbid_in_yaml`` and persists through
    ``serialize_weights`` / ``with_weights``.

    Attributes
    ----------
    mask : Mask2D
        Land / ocean mask.
    stagger : {"T", "U", "V", "X"}
        Which wet mask to apply: T → ``mask.h``, U → ``mask.u``, V →
        ``mask.v``, X → ``mask.xy_corner_strict``.
    """

    forbid_in_yaml: ClassVar[bool] = True

    mask: Mask2D
    stagger: str = eqx.field(static=True, default="T")

    def __check_init__(self) -> None:
        if self.stagger not in MASK_ATTRIBUTES:
            raise ValueError(
                f"stagger {self.stagger!r} is not available; valid: {list(STAGGERS)}"
            )

    def _apply(self, x: Field) -> Field:
        wet = getattr(self.mask, MASK_ATTRIBUTES[self.stagger])
        return jnp.where(wet, x, jnp.zeros((), x.dtype))


class Pick(FVXOperator):
    """Select one output of a tuple-valued node: ``(a, b, …) → [index]``.

    Attributes
    ----------
    index : int
        Position to select.
    """

    index: int = eqx.field(static=True)

    def _apply(self, xs: tuple[Any, ...]) -> Any:
        return xs[self.index]


class LinComb(FVXOperator):
    """Linear combination y = ∑ᵢ cᵢ xᵢ (coefficients are learnable leaves).

    Attributes
    ----------
    coeffs : Array
        ``[n]`` coefficients, one per input.
    """

    coeffs: jax.Array = eqx.field(converter=jnp.asarray)

    def _apply(self, *xs: Field) -> Field:
        if len(xs) != self.coeffs.shape[0]:
            raise ValueError(
                f"LinComb has {self.coeffs.shape[0]} coefficients, got {len(xs)} inputs"
            )
        return sum((c * x for c, x in zip(self.coeffs, xs, strict=True)), start=0.0)
