"""Diagnostic and reduction operators (2-D Cartesian).

Thin wrappers over the diagnostic surface (`Vorticity2D`, `Energetics2D`,
`Strain2D`, `QGPotentialVorticity2D`, `total_energy`, `area_sum`,
`area_mean`): each body is a Tier-A call. Fields are ``[*batch, Ny, Nx]``;
the reductions keep the leading axes::

    x : [T, Ny, Nx]  ──DomainIntegral──►  [T]

so a conservation check over a trajectory is a three-node graph::

    E = TotalEnergy(grid=G)(
        KineticEnergy(grid=G)(u, v), AvailablePE(grid=G, g_prime=g)(h)
    )
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from finitevolx._src.operators.diagnostic_operators import (
    Energetics2D,
    QGPotentialVorticity2D,
    Strain2D,
)
from finitevolx._src.operators.diagnostics import total_energy
from finitevolx._src.operators.reductions import area_mean, area_sum
from finitevolx._src.operators.vorticity import Vorticity2D
from finitevolx._src.pipekit.base import GridOperator, over_leading

Field = Float[Array, "*batch Ny Nx"]
Scalar = Float[Array, "*batch"]


class RelativeVorticity(GridOperator):
    """ζ = δₓv / Δx − δᵧu / Δy at X-points.  ``(U, V) → X``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return Vorticity2D(grid=self.cartesian_grid).relative_vorticity(u, v)


class PotentialVorticity(GridOperator):
    """Shallow-water PV q = (ζ + f) / h at X-points.  ``(U, V, T, T) → X``."""

    def _apply(self, u: Field, v: Field, h: Field, f: Field) -> Field:
        return Vorticity2D(grid=self.cartesian_grid).potential_vorticity(u, v, h, f)


class Enstrophy(GridOperator):
    """½ ζ² at X-points.  ``(U, V) → X``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return Vorticity2D(grid=self.cartesian_grid).enstrophy(u, v)


class QGPotentialVorticity(GridOperator):
    """Single-layer QG PV q = (∇²ψ + β (y − y₀)) / f₀ at T-points.  ``T → T``.

    y = j Δy at the T-points (ghost ring included).

    Attributes
    ----------
    grid : GridSpec
        ``(nx_interior, ny_interior, Lx, Ly)``.
    f0 : Array
        Reference Coriolis parameter (learnable).
    beta : Array
        Meridional gradient ∂f/∂y (learnable).
    y0 : float or None
        Reference y; ``None`` is the domain centre, ``(Ny − 1) Δy / 2``.
    """

    f0: jax.Array = eqx.field(converter=jnp.asarray, kw_only=True)
    beta: jax.Array = eqx.field(converter=jnp.asarray, default=0.0, kw_only=True)
    y0: float | None = eqx.field(static=True, default=None, kw_only=True)

    def _apply(self, psi: Field) -> Field:
        g = self.cartesian_grid
        y = jnp.arange(g.Ny, dtype=psi.dtype) * g.dy  # T[j] sits at y = j Δy
        Y = jnp.broadcast_to(y[:, None], (g.Ny, g.Nx))
        y0 = (g.Ny - 1) * g.dy / 2 if self.y0 is None else self.y0
        qg = QGPotentialVorticity2D(grid=g)
        return qg(psi, self.f0, self.beta, Y, y0)


class KineticEnergy(GridOperator):
    """½ (ū² + v̄²) at T-points (velocities averaged to T).  ``(U, V) → T``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return Energetics2D(grid=self.cartesian_grid).kinetic_energy(u, v)


class AvailablePE(GridOperator):
    """½ g′ (h − H)² at T-points.  ``T → T``.

    Attributes
    ----------
    grid : GridSpec
        ``(nx_interior, ny_interior, Lx, Ly)``.
    g_prime : Array
        Reduced gravity (learnable).
    H : Array
        Reference thickness (learnable), broadcast over the grid.
    """

    g_prime: jax.Array = eqx.field(converter=jnp.asarray, kw_only=True)
    H: jax.Array = eqx.field(converter=jnp.asarray, default=0.0, kw_only=True)

    def _apply(self, h: Field) -> Field:
        H = jnp.broadcast_to(self.H, h.shape)
        energetics = Energetics2D(grid=self.cartesian_grid)
        return energetics.available_potential_energy(h, H, self.g_prime)


#: ``component -> Strain2D method`` for `Strain`.
STRAIN_COMPONENTS = {
    "shear": "shear",
    "tensor": "tensor",
    "magnitude_squared": "magnitude_squared",
}


class Strain(GridOperator):
    """A strain component: shear σₛ (X), tensor σₙ (T) or σₛ² + σₙ² (T).

    Attributes
    ----------
    grid : GridSpec
        ``(nx_interior, ny_interior, Lx, Ly)``.
    component : {"shear", "tensor", "magnitude_squared"}
        Which component.
    """

    component: str = eqx.field(static=True, kw_only=True)

    def __check_init__(self) -> None:
        if self.component not in STRAIN_COMPONENTS:
            raise ValueError(
                f"component {self.component!r} is not available; "
                f"valid: {sorted(STRAIN_COMPONENTS)}"
            )

    def _apply(self, u: Field, v: Field) -> Field:
        strain = Strain2D(grid=self.cartesian_grid)
        return getattr(strain, STRAIN_COMPONENTS[self.component])(u, v)


class OkuboWeiss(GridOperator):
    """W = σₙ² + σₛ² − ζ² at T-points (W < 0: vortex cores).  ``(U, V) → T``."""

    def _apply(self, u: Field, v: Field) -> Field:
        return Strain2D(grid=self.cartesian_grid).okubo_weiss(u, v)


class TotalEnergy(GridOperator):
    """E = ∑_interior (KE + APE) Δx Δy, one value per leading index.

    ``(T, T) → [*batch]``.
    """

    def _apply(self, ke: Field, ape: Field) -> Scalar:
        g = self.cartesian_grid
        return total_energy(ke, ape, g.dx, g.dy)


class DomainIntegral(GridOperator):
    """∑_interior x ΔA over the T-cells (`area_sum`).  ``T → [*batch]``."""

    def _apply(self, x: Field) -> Scalar:
        return over_leading(lambda f: area_sum(f, self.cartesian_grid))(x)


class DomainMean(GridOperator):
    """Area-weighted interior mean ⟨x⟩ (`area_mean`).  ``T → [*batch]``."""

    def _apply(self, x: Field) -> Scalar:
        return over_leading(lambda f: area_mean(f, self.cartesian_grid))(x)
