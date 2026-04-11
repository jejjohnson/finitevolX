"""finitevolX masks — base, Cartesian, spherical, cylindrical."""

from finitevolx._src.mask.base import (
    StencilCapability1D,
    StencilCapability2D,
    StencilCapability3D,
)
from finitevolx._src.mask.cartesian import Mask1D, Mask2D, Mask3D
from finitevolx._src.mask.utils import (
    count_contiguous,
    dilate_mask,
    h_from_pooled,
    make_sponge,
    pool_bool,
)

__all__ = [
    # Stencil capability
    "StencilCapability1D",
    "StencilCapability2D",
    "StencilCapability3D",
    # Concrete mask classes
    "Mask1D",
    "Mask2D",
    "Mask3D",
    # Mask construction primitives
    "count_contiguous",
    "dilate_mask",
    "h_from_pooled",
    "make_sponge",
    "pool_bool",
]
