"""finitevolX masks — base, Cartesian, spherical, cylindrical."""

from finitevolx._src.mask.base import (
    StencilCapability1D,
    StencilCapability2D,
    StencilCapability3D,
)
from finitevolx._src.mask.cartesian import Mask1D, Mask2D, Mask3D

__all__ = [
    # Stencil capability
    "StencilCapability1D",
    "StencilCapability2D",
    "StencilCapability3D",
    # Concrete mask classes
    "Mask1D",
    "Mask2D",
    "Mask3D",
]
