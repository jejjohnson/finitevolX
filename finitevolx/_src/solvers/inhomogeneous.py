"""Known-value lifting for inhomogeneous Dirichlet boundary conditions.

Provides the building blocks for solving elliptic PDEs with prescribed
(non-zero) values at boundary cells and/or sparse interior locations:

Primitives
----------
:func:`boundary_ring` — identify the inner boundary ring (wet cells
adjacent to at least one dry cell) from a binary mask.

Operator API
------------
:class:`SolveDomain` — bundles all derived masks (boundary ring, all-known
cells, effective solve domain) from a user mask + optional known_mask.

:class:`KnownValueLifting` — pre/post-processing wrapper for the lifting
trick.  Sandwiches any solver with RHS correction and reconstruction.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from finitevolx._src.mask import Mask2D
from finitevolx._src.solvers.iterative import masked_laplacian


# ---------------------------------------------------------------------------
# Primitive: boundary_ring
# ---------------------------------------------------------------------------


def boundary_ring(mask: Float[Array, "Ny Nx"]) -> Bool[Array, "Ny Nx"]:
    """Wet cells adjacent to at least one dry cell (the inner ring).

    Uses pad-then-slice to avoid ``jnp.roll`` wrap-around artifacts.
    JIT-compatible but intended for offline (one-shot) use — the mask
    is typically fixed at setup time.

    Parameters
    ----------
    mask : Float[Array, "Ny Nx"]
        Binary wet/dry mask (1 = wet, 0 = dry).

    Returns
    -------
    Bool[Array, "Ny Nx"]
        True at wet cells that neighbor at least one dry cell.
    """
    wet = mask > 0.5
    dry = ~wet
    padded_dry = jnp.pad(dry, pad_width=1, mode="constant", constant_values=False)
    adjacent_to_dry = (
        padded_dry[2:, 1:-1]      # south neighbor is dry
        | padded_dry[:-2, 1:-1]   # north neighbor is dry
        | padded_dry[1:-1, 2:]    # east neighbor is dry
        | padded_dry[1:-1, :-2]   # west neighbor is dry
    )
    return wet & adjacent_to_dry


# ---------------------------------------------------------------------------
# SolveDomain
# ---------------------------------------------------------------------------


class SolveDomain(eqx.Module):
    """Derived masks for the lifting trick.

    Computes the inner boundary ring, merges with user-supplied
    ``known_mask``, and partitions the wet domain into known cells and
    the effective solve domain.

    Parameters
    ----------
    mask : Float[Array, "Ny Nx"] or Mask2D
        Binary wet/dry mask (1 = wet, 0 = dry).
    known_mask : Bool[Array, "Ny Nx"] or None
        Locations of additional known values (e.g., sparse interior
        observations).  Merged with the auto-derived boundary ring.
    """

    wet_mask: Bool[Array, "Ny Nx"]
    boundary_ring: Bool[Array, "Ny Nx"]
    all_known: Bool[Array, "Ny Nx"]
    effective_mask: Bool[Array, "Ny Nx"]

    def __init__(
        self,
        mask: Float[Array, "Ny Nx"] | Mask2D,
        known_mask: Bool[Array, "Ny Nx"] | None = None,
    ):
        if isinstance(mask, Mask2D):
            mask_arr = jnp.asarray(mask.xy_corner_strict, dtype=jnp.float32)
        else:
            mask_arr = mask

        self.wet_mask = mask_arr > 0.5
        self.boundary_ring = boundary_ring(mask_arr)

        if known_mask is not None:
            self.all_known = self.boundary_ring | known_mask
        else:
            self.all_known = self.boundary_ring

        self.effective_mask = self.wet_mask & ~self.all_known


# ---------------------------------------------------------------------------
# KnownValueLifting
# ---------------------------------------------------------------------------


class KnownValueLifting(eqx.Module):
    """Pre/post-processing wrapper for the lifting trick.

    Does **not** own or call a solver.  The user builds their solver
    separately (with ``domain.effective_mask``) and calls it themselves.
    This operator only sandwiches the solve with pre- and post-processing.

    Parameters
    ----------
    domain : SolveDomain
        Precomputed domain masks.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    lambda_ : float
        Helmholtz parameter.
    """

    domain: SolveDomain
    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)
    lambda_: float = eqx.field(static=True)

    def preprocess(
        self,
        rhs: Float[Array, "Ny Nx"],
        known_values: Float[Array, "Ny Nx"],
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Correct RHS for known values.

        Returns ``(rhs_corrected, value_lift)``.  Feed ``rhs_corrected``
        to your solver, then pass both to :meth:`postprocess`.

        Parameters
        ----------
        rhs : Float[Array, "Ny Nx"]
            Original right-hand side.
        known_values : Float[Array, "Ny Nx"]
            Prescribed values at known cells.  Values at non-known cells
            are ignored.

        Returns
        -------
        rhs_corrected : Float[Array, "Ny Nx"]
            Corrected RHS restricted to the effective solve domain.
        value_lift : Float[Array, "Ny Nx"]
            Lifting function — add to homogeneous solution to get the
            full solution.
        """
        value_lift = jnp.where(self.domain.all_known, known_values, 0.0)

        wet_mask_f = self.domain.wet_mask.astype(rhs.dtype)
        A_lift = masked_laplacian(
            value_lift, wet_mask_f, self.dx, self.dy, self.lambda_
        )

        eff_mask_f = self.domain.effective_mask.astype(rhs.dtype)
        rhs_corrected = (rhs - A_lift) * eff_mask_f

        return rhs_corrected, value_lift

    def postprocess(
        self,
        psi_hom: Float[Array, "Ny Nx"],
        value_lift: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Reconstruct full solution from homogeneous solve + lift."""
        return value_lift + psi_hom
