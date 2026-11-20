"""Analysis kernels on collocated grids: finite differences and remapping.

Every function here computes in the namespace of its input, so it runs on
NumPy, JAX (under ``jit`` too), CuPy, or any backend ``array-api-compat``
knows, and returns the input's dtype. They are the kernels behind geoops'
and xrops' ``Gradient``, ``Divergence``, ``Curl``, ``Laplacian`` and
vertical-remap operators (#423).

- Stencils (L0): :func:`difference_offsets`, :func:`difference_coeffs`,
  :func:`nonuniform_coeffs`, from integers alone.
- Differences (L1): :func:`difference` (a NaN propagates) and
  :func:`adaptive_difference` (the widest stencil that fits on valid data),
  and the 1-D / 2-D / 3-D gradient, divergence, curl and Laplacian built on
  them, each with an ``adaptive_`` twin.
- Remapping: :func:`remap_axis` (one 1-D source coordinate) and
  :func:`remap_axis_columns` (one set of levels per column).
"""

from gridax._src.analysis.differences import (
    adaptive_curl_2d,
    adaptive_curl_3d,
    adaptive_difference,
    adaptive_divergence_1d,
    adaptive_divergence_2d,
    adaptive_divergence_3d,
    adaptive_gradient_1d,
    adaptive_gradient_2d,
    adaptive_gradient_3d,
    adaptive_laplacian_1d,
    adaptive_laplacian_2d,
    adaptive_laplacian_3d,
    curl_2d,
    curl_3d,
    difference,
    divergence_1d,
    divergence_2d,
    divergence_3d,
    gradient_1d,
    gradient_2d,
    gradient_3d,
    laplacian_1d,
    laplacian_2d,
    laplacian_3d,
)
from gridax._src.analysis.remap import Extrapolate, remap_axis, remap_axis_columns
from gridax._src.analysis.stencils import (
    METHODS,
    Method,
    difference_coeffs,
    difference_offsets,
    nonuniform_coeffs,
)

__all__ = [
    # Stencils (L0)
    "METHODS",
    "Method",
    "difference_coeffs",
    "difference_offsets",
    "nonuniform_coeffs",
    # Differences along one axis (L1)
    "difference",
    "adaptive_difference",
    # 1-D calculus
    "gradient_1d",
    "divergence_1d",
    "laplacian_1d",
    "adaptive_gradient_1d",
    "adaptive_divergence_1d",
    "adaptive_laplacian_1d",
    # 2-D calculus
    "gradient_2d",
    "divergence_2d",
    "curl_2d",
    "laplacian_2d",
    "adaptive_gradient_2d",
    "adaptive_divergence_2d",
    "adaptive_curl_2d",
    "adaptive_laplacian_2d",
    # 3-D calculus
    "gradient_3d",
    "divergence_3d",
    "curl_3d",
    "laplacian_3d",
    "adaptive_gradient_3d",
    "adaptive_divergence_3d",
    "adaptive_curl_3d",
    "adaptive_laplacian_3d",
    # Remapping along one axis
    "Extrapolate",
    "remap_axis",
    "remap_axis_columns",
]
