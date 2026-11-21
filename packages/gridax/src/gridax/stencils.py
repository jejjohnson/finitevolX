"""Stencils: exact coefficients (Layer 0) and the raw C-grid stencils (Layer 1).

One home for every finite-difference stencil in gridax, collocated or
staggered.

- **Layer 0, coefficients.** :func:`difference_offsets` and
  :func:`difference_coeffs` solve the Taylor conditions
  :math:`\\sum_k c_k s_k^m = n!\\,\\delta_{mn}` exactly, for integer offsets
  (collocated points, as :mod:`gridax.analysis` uses them) and half-integer
  offsets (``method="staggered"``, the faces of a C-grid);
  :func:`nonuniform_coeffs` solves them for arbitrary distances, on any
  backend.
- **Layer 1, raw stencils.** The 32 named C-grid stencils
  ``<op>_<axis>_<side>_<d>d`` (``diff_x_fwd_2d``, ``avg_xfwd_yfwd_3d``, …):
  index arithmetic with no metric and no padding, and :func:`interior`, which
  pads a result back to the full shape. They are the two-point staggered
  stencils of Layer 0, :math:`(-1, 1)` for ``diff_*`` and
  :math:`(\\frac12, \\frac12)` for ``avg_*``. The same names stay importable
  from the top level of ``gridax``.

Everything here imports without JAX.

Example:
    >>> import gridax.stencils as gs
    >>> offsets = gs.difference_offsets(derivative=1, method="staggered")
    >>> offsets, gs.difference_coeffs(offsets, 1)
    ((-0.5, 0.5), (-1.0, 1.0))
"""

from gridax._src.operators._ghost import interior
from gridax._src.operators.stencils import (
    avg_x_bwd_1d,
    avg_x_bwd_2d,
    avg_x_bwd_3d,
    avg_x_fwd_1d,
    avg_x_fwd_2d,
    avg_x_fwd_3d,
    avg_xbwd_ybwd_2d,
    avg_xbwd_ybwd_3d,
    avg_xbwd_yfwd_2d,
    avg_xbwd_yfwd_3d,
    avg_xfwd_ybwd_2d,
    avg_xfwd_ybwd_3d,
    avg_xfwd_yfwd_2d,
    avg_xfwd_yfwd_3d,
    avg_y_bwd_2d,
    avg_y_bwd_3d,
    avg_y_fwd_2d,
    avg_y_fwd_3d,
    avg_z_bwd_3d,
    avg_z_fwd_3d,
    diff_x_bwd_1d,
    diff_x_bwd_2d,
    diff_x_bwd_3d,
    diff_x_fwd_1d,
    diff_x_fwd_2d,
    diff_x_fwd_3d,
    diff_y_bwd_2d,
    diff_y_bwd_3d,
    diff_y_fwd_2d,
    diff_y_fwd_3d,
    diff_z_bwd_3d,
    diff_z_fwd_3d,
)
from gridax._src.stencils.coefficients import (
    METHODS,
    Method,
    difference_coeffs,
    difference_offsets,
    nonuniform_coeffs,
)

__all__ = [
    # Layer 0: coefficients
    "METHODS",
    "Method",
    "difference_offsets",
    "difference_coeffs",
    "nonuniform_coeffs",
    # Layer 1: raw C-grid stencils, 1-D
    "diff_x_fwd_1d",
    "diff_x_bwd_1d",
    "avg_x_fwd_1d",
    "avg_x_bwd_1d",
    # 2-D
    "diff_x_fwd_2d",
    "diff_x_bwd_2d",
    "diff_y_fwd_2d",
    "diff_y_bwd_2d",
    "avg_x_fwd_2d",
    "avg_x_bwd_2d",
    "avg_y_fwd_2d",
    "avg_y_bwd_2d",
    "avg_xfwd_yfwd_2d",
    "avg_xbwd_ybwd_2d",
    "avg_xbwd_yfwd_2d",
    "avg_xfwd_ybwd_2d",
    # 3-D
    "diff_x_fwd_3d",
    "diff_x_bwd_3d",
    "diff_y_fwd_3d",
    "diff_y_bwd_3d",
    "diff_z_fwd_3d",
    "diff_z_bwd_3d",
    "avg_x_fwd_3d",
    "avg_x_bwd_3d",
    "avg_y_fwd_3d",
    "avg_y_bwd_3d",
    "avg_z_fwd_3d",
    "avg_z_bwd_3d",
    "avg_xfwd_yfwd_3d",
    "avg_xbwd_ybwd_3d",
    "avg_xbwd_yfwd_3d",
    "avg_xfwd_ybwd_3d",
    # padding back to the full array
    "interior",
]
