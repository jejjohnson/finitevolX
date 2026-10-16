"""finitevolX operators as pipekit Operators (Tier B).

Requires the ``pipekit`` extra (``pip install "finitevolx[pipekit]"``);
``import finitevolx`` never imports pipekit.

Every operator here is a `pipekit_jax.JaxOperator`: a pipekit ``Operator``
(JSON config, ``|`` composition, ``Sequential`` / ``Graph`` nodes) that is
also an ``eqx.Module`` (runs under ``jit`` / ``grad`` / ``vmap``). Fields
are C-grid arrays ``[*batch, Ny, Nx]``; the grid is a static
``(nx_interior, ny_interior, Lx, Ly)`` tuple.
"""

from __future__ import annotations

try:
    import pipekit_jax as _pipekit_jax  # noqa: F401
except ImportError as err:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "finitevolx.pipekit needs the `pipekit` extra: "
        'pip install "finitevolx[pipekit]"'
    ) from err

from finitevolx._src.pipekit.base import (
    FVXOperator,
    GridOperator,
    GridSpec,
    as_grid_spec,
    build_grid,
    over_leading,
)
from finitevolx._src.pipekit.stencils import (
    DIFFERENCES,
    GHOST_FILLERS,
    INTERPOLATIONS,
    MASK_ATTRIBUTES,
    ApplyMask,
    Coriolis,
    Curl,
    Difference,
    Divergence,
    FillGhosts,
    GradPerp,
    Interp,
    Laplacian,
    LinComb,
    Pick,
)

__all__ = [
    # base
    "FVXOperator",
    "GridOperator",
    "GridSpec",
    "as_grid_spec",
    "build_grid",
    "over_leading",
    # stencils, interpolation, boundaries, masks
    "DIFFERENCES",
    "GHOST_FILLERS",
    "INTERPOLATIONS",
    "MASK_ATTRIBUTES",
    "ApplyMask",
    "Coriolis",
    "Curl",
    "Difference",
    "Divergence",
    "FillGhosts",
    "GradPerp",
    "Interp",
    "Laplacian",
    # graph plumbing
    "LinComb",
    "Pick",
]
