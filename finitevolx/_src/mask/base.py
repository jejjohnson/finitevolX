"""Shared mask infrastructure: StencilCapability1D / 2D / 3D classes.

These small data classes record, at every grid cell, the number of
contiguous wet neighbours reachable in each axis-aligned direction
without crossing a dry cell or the domain edge.  They are stored on
the dimensional ``Mask*`` classes and consumed by adaptive WENO /
TVD reconstruction code in the advection module.

Construction is numpy-only (the recurrence does not vectorise without
extra machinery, and masks are built once at setup time, not inside
JIT-traced step functions).  Stored arrays are JAX int32.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int
import numpy as np

from finitevolx._src.mask.utils import _count_contiguous


class StencilCapability1D(eqx.Module):
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
        return cls(
            x_pos=jnp.asarray(_count_contiguous(h_np, axis=0, forward=True)),
            x_neg=jnp.asarray(_count_contiguous(h_np, axis=0, forward=False)),
        )


class StencilCapability2D(eqx.Module):
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

        Construction uses numpy; stored arrays are JAX int32.

        Parameters
        ----------
        h : array-like [Ny, Nx] bool
            Wet (True) / dry (False) mask.

        Returns
        -------
        StencilCapability2D
        """
        h_np = np.asarray(h, dtype=bool)
        return cls(
            x_pos=jnp.asarray(_count_contiguous(h_np, axis=1, forward=True)),
            x_neg=jnp.asarray(_count_contiguous(h_np, axis=1, forward=False)),
            y_pos=jnp.asarray(_count_contiguous(h_np, axis=0, forward=True)),
            y_neg=jnp.asarray(_count_contiguous(h_np, axis=0, forward=False)),
        )


class StencilCapability3D(eqx.Module):
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
        return cls(
            x_pos=jnp.asarray(_count_contiguous(h_np, axis=2, forward=True)),
            x_neg=jnp.asarray(_count_contiguous(h_np, axis=2, forward=False)),
            y_pos=jnp.asarray(_count_contiguous(h_np, axis=1, forward=True)),
            y_neg=jnp.asarray(_count_contiguous(h_np, axis=1, forward=False)),
            z_pos=jnp.asarray(_count_contiguous(h_np, axis=0, forward=True)),
            z_neg=jnp.asarray(_count_contiguous(h_np, axis=0, forward=False)),
        )
