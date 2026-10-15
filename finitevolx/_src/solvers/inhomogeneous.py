"""Known-value lifting for inhomogeneous Dirichlet boundary conditions.

Provides the building blocks for solving elliptic PDEs with prescribed
(non-zero) values at boundary cells and/or sparse interior locations:

Where the known values sit
--------------------------
The prescribed values live on the **coast ring**: the dry cells 4-adjacent
to a wet cell.  That is where :func:`~finitevolx.masked_laplacian` (and so
``method="cg"``) puts its homogeneous Dirichlet wall -- for ψ on the X points
of a ``Mask2D.xy_corner_strict`` mask, the C-grid wall nodes.  Every wet
cell stays an unknown unless ``known_mask`` pins it.

Primitives
----------
:func:`coast_ring` — the dry cells adjacent to at least one wet cell (where
the known values sit).

:func:`boundary_ring` — the inner boundary ring (wet cells adjacent to at
least one dry cell).

Operator API
------------
:class:`SolveDomain` — bundles all derived masks (coast ring, all-known
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
    mask : Float[Array, "Ny Nx"] or Bool[Array, "Ny Nx"]
        Binary wet/dry mask (1 / True = wet, 0 / False = dry).

    Returns
    -------
    Bool[Array, "Ny Nx"]
        True at wet cells that neighbor at least one dry cell.
    """
    wet = mask > 0.5
    dry = ~wet
    # padded_dry[j+1, i+1] = dry[j, i]; cells outside the array count as wet.
    padded_dry = jnp.pad(dry, pad_width=1, mode="constant", constant_values=False)
    # adjacent_to_dry[j, i] = dry[j+1, i] | dry[j-1, i] | dry[j, i+1] | dry[j, i-1]
    adjacent_to_dry = (
        padded_dry[2:, 1:-1]  # dry[j+1, i]  (north neighbour)
        | padded_dry[:-2, 1:-1]  # dry[j-1, i]  (south neighbour)
        | padded_dry[1:-1, 2:]  # dry[j, i+1]  (east neighbour)
        | padded_dry[1:-1, :-2]  # dry[j, i-1]  (west neighbour)
    )
    return wet & adjacent_to_dry


def coast_ring(mask: Float[Array, "Ny Nx"]) -> Bool[Array, "Ny Nx"]:
    """Dry cells adjacent to at least one wet cell (the coast ring).

    This is where :func:`~finitevolx.masked_laplacian` imposes its
    homogeneous Dirichlet wall (a dry neighbour reads as zero), so it is
    where prescribed (known) values sit.  For ψ on a ``from_interior``
    rectangle (``Mask2D.xy_corner_strict`` wet on ``[1:-2, 1:-2]``) it is the
    C-grid wall: X rows / columns ``0`` and ``N-2`` (corners excluded).

    Uses pad-then-slice (cells outside the array count as dry), so it never
    wraps around.

    Parameters
    ----------
    mask : Float[Array, "Ny Nx"] or Bool[Array, "Ny Nx"]
        Binary wet/dry mask (1 / True = wet, 0 / False = dry).

    Returns
    -------
    Bool[Array, "Ny Nx"]
        True at dry cells that neighbour at least one wet cell.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> mask = jnp.zeros((5, 5), dtype=bool).at[1:-2, 1:-2].set(True)
    >>> coast_ring(mask).astype(int)[:, :4]
    Array([[0, 1, 1, 0],
           [1, 0, 0, 1],
           [1, 0, 0, 1],
           [0, 1, 1, 0],
           [0, 0, 0, 0]], dtype=int32)
    """
    wet = jnp.asarray(mask) > 0.5
    # coast = the inner ring of the dry set: dry cells 4-adjacent to a wet one
    # (boundary_ring pads with "not dry", i.e. cells outside the array are dry).
    return boundary_ring(~wet)


# ---------------------------------------------------------------------------
# SolveDomain
# ---------------------------------------------------------------------------


class SolveDomain(eqx.Module):
    """Derived masks for the lifting trick.

    The known values sit on the coast ring -- the dry cells adjacent to the
    wet mask, where ``masked_laplacian`` / ``method="cg"`` put the
    homogeneous wall -- plus any wet ``known_mask`` cells.  The effective
    solve domain is every other wet cell::

        all_known = coast_ring | (known_mask & wet)
        effective_mask = wet & ~known_mask

    Parameters
    ----------
    mask : Float[Array, "Ny Nx"] or Mask2D
        Binary wet/dry mask (1 = wet, 0 = dry).  A :class:`Mask2D` gives its
        ψ (X-point) mask ``xy_corner_strict``.
    known_mask : Bool[Array, "Ny Nx"] or None
        Locations of additional known values (e.g., sparse interior
        observations).  Merged with the coast ring.

    Attributes
    ----------
    wet_mask : Bool[Array, "Ny Nx"]
        The wet cells.
    coast_ring : Bool[Array, "Ny Nx"]
        Dry cells adjacent to a wet cell, where the boundary values sit.
    boundary_ring : Bool[Array, "Ny Nx"]
        Wet cells adjacent to a dry cell (informational; they are unknowns).
    all_known : Bool[Array, "Ny Nx"]
        Cells whose value is prescribed.
    effective_mask : Bool[Array, "Ny Nx"]
        Wet cells that are solved for.
    """

    wet_mask: Bool[Array, "Ny Nx"]
    coast_ring: Bool[Array, "Ny Nx"]
    boundary_ring: Bool[Array, "Ny Nx"]
    all_known: Bool[Array, "Ny Nx"]
    effective_mask: Bool[Array, "Ny Nx"]

    def __init__(
        self,
        mask: Float[Array, "Ny Nx"] | Mask2D,
        known_mask: Bool[Array, "Ny Nx"] | None = None,
    ):
        # Keep the masks boolean: a float32 copy would downcast a weakly
        # typed float64 field it multiplies (the lifting uses jnp.where).
        if isinstance(mask, Mask2D):
            mask_arr = jnp.asarray(mask.xy_corner_strict, dtype=bool)
        else:
            mask_arr = jnp.asarray(mask) > 0.5

        self.wet_mask = mask_arr
        self.coast_ring = coast_ring(mask_arr)
        self.boundary_ring = boundary_ring(mask_arr)

        if known_mask is not None:
            known_wet = jnp.asarray(known_mask, dtype=bool) & self.wet_mask
            self.all_known = self.coast_ring | known_wet
            self.effective_mask = self.wet_mask & ~known_wet
        else:
            self.all_known = self.coast_ring
            self.effective_mask = self.wet_mask


# ---------------------------------------------------------------------------
# KnownValueLifting
# ---------------------------------------------------------------------------


def lift_rhs(
    domain: SolveDomain,
    rhs: Float[Array, "Ny Nx"],
    known_values: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Functional form of :meth:`KnownValueLifting.preprocess`.

    Unlike the operator, ``lambda_`` may be a traced value, so this can be
    vmapped over layers that each carry their own Helmholtz parameter.

    Returns ``(rhs_corrected, value_lift)``; see
    :meth:`KnownValueLifting.preprocess`.
    """
    value_lift = jnp.where(domain.all_known, known_values, 0.0)

    # The lift lives on dry coast cells, so the stencil must read it there:
    # apply the 5-point operator without a mask (all-ones), then restrict the
    # correction to the solve cells:
    #   rhs_corrected[j, i] = f[j, i] - (A - lambda) value_lift [j, i]  on E
    unmasked = jnp.ones(value_lift.shape, dtype=value_lift.dtype)
    A_lift = masked_laplacian(value_lift, unmasked, dx, dy, lambda_)

    rhs_corrected = jnp.where(domain.effective_mask, rhs - A_lift, 0.0)

    return rhs_corrected, value_lift


def reconstruct_from_lift(
    domain: SolveDomain,
    psi_hom: Float[Array, "Ny Nx"],
    value_lift: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """Functional form of :meth:`KnownValueLifting.postprocess`.

    psi = value_lift + psi_hom restricted to the effective solve domain, so
    the coast ring (dry) and the known_mask cells carry the known values.
    """
    return value_lift + jnp.where(domain.effective_mask, psi_hom, 0.0)


class KnownValueLifting(eqx.Module):
    """Pre/post-processing wrapper for the lifting trick.

    Does **not** own or call a solver.  The user builds their solver
    separately and calls it themselves.  This operator only sandwiches the
    solve with pre- and post-processing.

    The solver's unknowns must be exactly ``domain.effective_mask``, with
    zero on every other cell (``masked_laplacian`` on
    ``domain.effective_mask``: a CG or multigrid operator built on it).
    The known values sit on the coast ring (the dry cells adjacent to the
    wet mask) and on the wet ``known_mask`` cells.

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
        return lift_rhs(self.domain, rhs, known_values, self.dx, self.dy, self.lambda_)

    def postprocess(
        self,
        psi_hom: Float[Array, "Ny Nx"],
        value_lift: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """Reconstruct full solution from homogeneous solve + lift."""
        return reconstruct_from_lift(self.domain, psi_hom, value_lift)
