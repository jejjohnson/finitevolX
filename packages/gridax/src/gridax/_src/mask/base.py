"""Shared mask infrastructure: StencilCapability1D / 2D / 3D classes.

These small data classes record, at every grid cell, the number of
contiguous wet neighbours reachable in each axis-aligned direction
without crossing a dry cell or the domain edge.  They are stored on
the dimensional ``Mask*`` classes and consumed by adaptive WENO /
TVD reconstruction code in the advection module.

Construction is numpy-only (the recurrence does not vectorise without
extra machinery, and masks are built once at setup time, not inside
JIT-traced step functions).  Stored arrays are NumPy int8: a count
saturates at 127, far above the widest stencil half-width (5), so every
consumer's comparison ``count >= half_width`` is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Bool, Int
import numpy as np

from gridax._src.mask.utils import count_contiguous
from gridax._src.pytree import pytree_dataclass

if TYPE_CHECKING:
    from jaxtyping import Array
else:
    Array = np.ndarray

#: Counts saturate here; it bounds the int8 storage, not any stencil.
_COUNT_CAP = np.iinfo(np.int8).max


def _counts(h: np.ndarray, axis: int, forward: bool) -> np.ndarray:
    """Contiguous wet counts along ``axis``, saturated at 127, as int8."""
    counts = count_contiguous(h, axis=axis, forward=forward)
    return np.minimum(counts, _COUNT_CAP).astype(np.int8)


@pytree_dataclass
class StencilCapability1D:
    """Directional count of contiguous wet neighbours for each 1-D grid cell.

    At each cell ``i``, stores the number of consecutive wet cells (including
    the cell itself) reachable before hitting a dry cell or the domain edge.

    Parameters
    ----------
    x_pos : Int[Array, "Nx"]
        Count in the +x direction.
    x_neg : Int[Array, "Nx"]
        Count in the −x direction.
    """

    x_pos: Int[Array, "Nx"]
    x_neg: Int[Array, "Nx"]

    @classmethod
    def from_mask(cls, h: np.ndarray | Bool[Array, "Nx"]) -> StencilCapability1D:
        """Build stencil capability from a 1-D wet/dry mask.

        Parameters
        ----------
        h : array-like [Nx] bool

        Returns
        -------
        StencilCapability1D
        """
        h_np = np.asarray(h, dtype=bool)
        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "x_pos": _counts(h_np, axis=0, forward=True),
            "x_neg": _counts(h_np, axis=0, forward=False),
        }
        return cls(**fields)


@pytree_dataclass
class StencilCapability2D:
    """Directional count of contiguous wet neighbours for each 2-D grid cell.

    At each cell ``(j, i)``, stores the number of consecutive wet cells
    (including the cell itself) reachable before hitting a dry cell or
    the domain edge.

    Parameters
    ----------
    x_pos : Int[Array, "Ny Nx"]
        Count in the +x direction.
    x_neg : Int[Array, "Ny Nx"]
        Count in the −x direction.
    y_pos : Int[Array, "Ny Nx"]
        Count in the +y direction.
    y_neg : Int[Array, "Ny Nx"]
        Count in the −y direction.
    """

    x_pos: Int[Array, "Ny Nx"]
    x_neg: Int[Array, "Ny Nx"]
    y_pos: Int[Array, "Ny Nx"]
    y_neg: Int[Array, "Ny Nx"]

    @classmethod
    def from_mask(cls, h: np.ndarray | Bool[Array, "Ny Nx"]) -> StencilCapability2D:
        """Build stencil capability from a 2-D wet/dry mask.

        Construction uses numpy; stored arrays are NumPy int8.

        Parameters
        ----------
        h : array-like [Ny, Nx] bool
            Wet (True) / dry (False) mask.

        Returns
        -------
        StencilCapability2D
        """
        h_np = np.asarray(h, dtype=bool)
        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "x_pos": _counts(h_np, axis=1, forward=True),
            "x_neg": _counts(h_np, axis=1, forward=False),
            "y_pos": _counts(h_np, axis=0, forward=True),
            "y_neg": _counts(h_np, axis=0, forward=False),
        }
        return cls(**fields)


@pytree_dataclass
class StencilCapability3D:
    """Directional count of contiguous wet neighbours for each 3-D grid cell.

    At each cell ``(k, j, i)``, stores the number of consecutive wet cells
    (including the cell itself) reachable before hitting a dry cell or
    the domain edge.

    Parameters
    ----------
    x_pos : Int[Array, "Nz Ny Nx"]
        Count in the +x direction.
    x_neg : Int[Array, "Nz Ny Nx"]
        Count in the −x direction.
    y_pos : Int[Array, "Nz Ny Nx"]
        Count in the +y direction.
    y_neg : Int[Array, "Nz Ny Nx"]
        Count in the −y direction.
    z_pos : Int[Array, "Nz Ny Nx"]
        Count in the +z direction.
    z_neg : Int[Array, "Nz Ny Nx"]
        Count in the −z direction.
    """

    x_pos: Int[Array, "Nz Ny Nx"]
    x_neg: Int[Array, "Nz Ny Nx"]
    y_pos: Int[Array, "Nz Ny Nx"]
    y_neg: Int[Array, "Nz Ny Nx"]
    z_pos: Int[Array, "Nz Ny Nx"]
    z_neg: Int[Array, "Nz Ny Nx"]

    @classmethod
    def from_mask(cls, h: np.ndarray | Bool[Array, "Nz Ny Nx"]) -> StencilCapability3D:
        """Build stencil capability from a 3-D wet/dry mask.

        Parameters
        ----------
        h : array-like [Nz, Ny, Nx] bool

        Returns
        -------
        StencilCapability3D
        """
        h_np = np.asarray(h, dtype=bool)
        # NumPy arrays fill fields the type checker reads as JAX arrays: the
        # type a consumer sees once JAX has flattened the mask.
        fields: dict[str, Any] = {
            "x_pos": _counts(h_np, axis=2, forward=True),
            "x_neg": _counts(h_np, axis=2, forward=False),
            "y_pos": _counts(h_np, axis=1, forward=True),
            "y_neg": _counts(h_np, axis=1, forward=False),
            "z_pos": _counts(h_np, axis=0, forward=True),
            "z_neg": _counts(h_np, axis=0, forward=False),
        }
        return cls(**fields)
