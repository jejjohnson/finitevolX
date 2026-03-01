from finitevolx._src.advection import Advection1D, Advection2D, Advection3D
from finitevolx._src.boundary import enforce_periodic, pad_interior
from finitevolx._src.difference import Difference1D, Difference2D, Difference3D
from finitevolx._src.grid import ArakawaCGrid1D, ArakawaCGrid2D, ArakawaCGrid3D
from finitevolx._src.interpolation import (
    Interpolation1D,
    Interpolation2D,
    Interpolation3D,
)
from finitevolx._src.masks.cgrid_mask import ArakawaCGridMask, StencilCapability
from finitevolx._src.reconstruction import (
    Reconstruction1D,
    Reconstruction2D,
    Reconstruction3D,
)
from finitevolx._src.vorticity import Vorticity2D, Vorticity3D

try:
    from finitevolx._src.interp.interp import (
        avg_arithmetic,
        avg_geometric,
        avg_harmonic,
        avg_pool,
        avg_quadratic,
        center_avg_2D,
        x_avg_1D,
        x_avg_2D,
        y_avg_2D,
    )
    from finitevolx._src.operators.operators import (
        absolute_vorticity,
        bernoulli_potential,
        difference,
        divergence,
        geostrophic_gradient,
        kinetic_energy,
        laplacian,
        relative_vorticity,
    )
    from finitevolx._src.reconstructions.reconstruct import (
        reconstruct,
        reconstruct_1pt,
        reconstruct_3pt,
        reconstruct_5pt,
    )
    from finitevolx._src.reconstructions.upwind import (
        upwind_1pt,
        upwind_2pt_bnds,
        upwind_3pt,
        upwind_3pt_bnds,
        upwind_5pt,
    )

    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False

__all__ = [
    # New refactored API
    "Advection1D",
    "Advection2D",
    "Advection3D",
    "ArakawaCGrid1D",
    "ArakawaCGrid2D",
    "ArakawaCGrid3D",
    "ArakawaCGridMask",
    "Difference1D",
    "Difference2D",
    "Difference3D",
    "enforce_periodic",
    "Interpolation1D",
    "Interpolation2D",
    "Interpolation3D",
    "pad_interior",
    "Reconstruction1D",
    "Reconstruction2D",
    "Reconstruction3D",
    "StencilCapability",
    "Vorticity2D",
    "Vorticity3D",
    # Legacy API
    "absolute_vorticity",
    "avg_arithmetic",
    "avg_geometric",
    "avg_harmonic",
    "avg_pool",
    "avg_quadratic",
    "bernoulli_potential",
    "center_avg_2D",
    "difference",
    "divergence",
    "geostrophic_gradient",
    "kinetic_energy",
    "laplacian",
    "reconstruct",
    "reconstruct_1pt",
    "reconstruct_3pt",
    "reconstruct_5pt",
    "relative_vorticity",
    "upwind_1pt",
    "upwind_2pt_bnds",
    "upwind_3pt",
    "upwind_3pt_bnds",
    "upwind_5pt",
    "x_avg_1D",
    "x_avg_2D",
    "y_avg_2D",
]
